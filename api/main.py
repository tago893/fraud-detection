"""Fraud scoring API: load artifacts once at startup, score per request."""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import BackgroundTasks, FastAPI, Request

from api import database as db
from api.schemas import HealthResponse, ScoringResponse, TransactionRequest
from api.scoring import Scorer
from pipeline.config import ARTIFACTS, DB_PATH, FEATURE_ORDER, MODEL_PATH, SCALER_PATH, THRESHOLD_PATH

log = logging.getLogger("api")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def persist_transaction(db_path: Path, record: db.TransactionRecord) -> None:
    """Background task. A failed write is logged, never raised into the already-sent response."""
    try:
        db.insert_transaction(db_path, record)
    except Exception:
        log.exception("failed to store transaction %s", record.transaction_id)


def load_state(artifacts_dir: Path) -> dict:
    """Load model, scaler, and threshold.json; fail fast if artifacts are missing or inconsistent."""
    needed = [artifacts_dir / p.name for p in (MODEL_PATH, SCALER_PATH, THRESHOLD_PATH)]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing artifacts {missing}; run `make split train evaluate` first.")
    meta = json.loads((artifacts_dir / THRESHOLD_PATH.name).read_text())
    if set(meta["feature_order"]) != set(FEATURE_ORDER):
        raise RuntimeError("threshold.json feature_order does not match the API schema.")
    model = joblib.load(artifacts_dir / MODEL_PATH.name)
    scaler = joblib.load(artifacts_dir / SCALER_PATH.name)
    return {
        "scorer": Scorer.from_artifacts(model, scaler, meta["feature_order"]),
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
        yield

    app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

    @app.post("/score", response_model=ScoringResponse)
    async def score(req: TransactionRequest, request: Request, background: BackgroundTasks) -> ScoringResponse:  # type: ignore[valid-type]
        state = request.app.state
        started = time.perf_counter()
        probability = state.scorer.predict_proba(state.scorer.vectorize(req))
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
        background.add_task(persist_transaction, state.db_path, record)
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
        return HealthResponse(status="ok", model_version=state.model_version, threshold=state.threshold)

    return app


app = create_app()
