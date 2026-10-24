"""Fraud scoring API: load artifacts once at startup, score per request."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from api import database as db
from api import explain
from api.schemas import (
    HealthResponse,
    ScoringResponse,
    ThresholdResponse,
    ThresholdUpdate,
    TransactionRequest,
)
from api.scoring import Scorer
from pipeline.config import (
    ARTIFACTS,
    DB_PATH,
    FEATURE_ORDER,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
)

log = logging.getLogger("api")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def store_and_explain(
    state: Any, record: db.TransactionRecord, raw: np.ndarray
) -> None:
    """Background task: insert the row, then SHAP it if flagged.

    Failures are logged, never raised into the already-sent response. SHAP only runs if the
    insert succeeded: on a duplicate transaction_id the insert is rejected, and explaining
    would overwrite the original row's explanation with one for a different feature vector.
    """
    try:
        db.insert_transaction(state.db_path, record)
    except sqlite3.IntegrityError:
        log.warning("duplicate transaction_id %s ignored", record.transaction_id)
        return
    except Exception:
        log.exception("failed to store transaction %s", record.transaction_id)
        return
    if not (record.flagged and state.explainer is not None):
        return
    try:
        scorer = state.scorer
        top = explain.top_contributions(
            state.explainer, scorer.scale_row(raw), raw, scorer.feature_order
        )
        db.set_shap_top(state.db_path, record.transaction_id, top)
    except Exception:
        log.exception("failed to explain transaction %s", record.transaction_id)


def load_state(artifacts_dir: Path) -> dict:
    """Load model, scaler, and threshold.json; fail fast if artifacts are missing or inconsistent."""
    needed = [artifacts_dir / p.name for p in (MODEL_PATH, SCALER_PATH, THRESHOLD_PATH)]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Missing artifacts {missing}; run `make split train evaluate` first."
        )
    meta = json.loads((artifacts_dir / THRESHOLD_PATH.name).read_text())
    if (
        list(meta["feature_order"]) != FEATURE_ORDER
    ):  # order matters, not just the names
        raise RuntimeError(
            "threshold.json feature_order does not match the fixed feature order."
        )
    metrics_file = artifacts_dir / METRICS_PATH.name
    if metrics_file.exists():
        trained = json.loads(metrics_file.read_text()).get("model_version")
        if trained != meta["model_version"]:
            raise RuntimeError(
                f"threshold.json is for model {meta['model_version']} but the trained model is "
                f"{trained}; run `make evaluate` to pick a threshold for the current model."
            )
    model = joblib.load(artifacts_dir / MODEL_PATH.name)
    scaler = joblib.load(artifacts_dir / SCALER_PATH.name)
    return {
        "scorer": Scorer.from_artifacts(model, scaler, meta["feature_order"]),
        "explainer": explain.build_explainer(model),
        "threshold": float(meta["threshold"]),
        "threshold_source": "artifact",
        "artifact_threshold": float(meta["threshold"]),
        "model_version": meta["model_version"],
    }


def create_app(artifacts_dir: Path = ARTIFACTS, db_path: Path = DB_PATH) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for key, value in load_state(artifacts_dir).items():
            setattr(app.state, key, value)
        db.init_db(db_path)
        app.state.db_path = db_path
        app.state.threshold_lock = threading.Lock()
        yield

    app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

    @app.post("/score", response_model=ScoringResponse)
    async def score(
        req: TransactionRequest, request: Request, background: BackgroundTasks
    ) -> ScoringResponse:  # type: ignore[valid-type]
        state = request.app.state
        started = time.perf_counter()
        raw = state.scorer.vectorize(req)
        probability = state.scorer.predict_proba(raw)
        threshold = state.threshold
        latency_ms = (time.perf_counter() - started) * 1000
        transaction_id = req.transaction_id or str(uuid.uuid4())
        flagged = probability >= threshold
        record = db.TransactionRecord(
            transaction_id=transaction_id,
            scored_at=utc_now_iso(),
            amount=req.Amount,
            fraud_probability=probability,
            flagged=flagged,
            threshold=threshold,
            latency_ms=latency_ms,
            true_label=req.true_label,  # stored for simulation stats only
        )
        background.add_task(store_and_explain, state, record, raw)
        return ScoringResponse(
            transaction_id=transaction_id,
            fraud_probability=probability,
            flagged=flagged,
            decision_threshold=threshold,
            model_version=state.model_version,
            latency_ms=latency_ms,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        state = request.app.state
        return HealthResponse(
            status="ok", model_version=state.model_version, threshold=state.threshold
        )

    @app.get("/threshold", response_model=ThresholdResponse)
    async def get_threshold(request: Request) -> ThresholdResponse:
        state = request.app.state
        return ThresholdResponse(
            threshold=state.threshold, source=state.threshold_source
        )

    @app.put("/threshold", response_model=ThresholdResponse)
    def put_threshold(body: ThresholdUpdate, request: Request) -> ThresholdResponse:
        """Update the in-memory threshold and log the change. Not on the /score path, so the write is inline."""
        state = request.app.state
        with (
            state.threshold_lock
        ):  # keeps the history chain and the threshold/source pair consistent
            old = state.threshold
            db.insert_threshold_change(
                state.db_path, utc_now_iso(), old, body.threshold
            )
            state.threshold = body.threshold
            state.threshold_source = "override"
        log.info("threshold changed %.4f -> %.4f", old, body.threshold)
        return ThresholdResponse(threshold=body.threshold, source="override")

    @app.get("/explain/{transaction_id}")
    def explain_endpoint(
        transaction_id: str, request: Request
    ) -> dict:  # sync: runs in the threadpool
        top = db.get_shap_top(request.app.state.db_path, transaction_id)
        if top is None:
            raise HTTPException(
                status_code=404, detail="No explanation stored for this transaction"
            )
        return {"transaction_id": transaction_id, "top_features": top}

    @app.get("/metrics/summary")
    def metrics_summary(
        request: Request, minutes: int = 5
    ) -> dict:  # sync: runs in the threadpool
        minutes = max(1, min(minutes, 24 * 60))
        since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(
            timespec="milliseconds"
        )
        count, flagged, p95 = db.window_summary(request.app.state.db_path, since)
        return {
            "window_minutes": minutes,
            "count": count,
            "flagged": flagged,
            "flag_rate": flagged / count if count else 0.0,
            "p95_latency_ms": p95,
        }

    return app


app = create_app()
