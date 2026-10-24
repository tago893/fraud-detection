"""Pick the cost-optimal threshold on val, then report on test once."""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np

from pipeline.config import (
    ARTIFACTS,
    C_FP,
    FEATURE_ORDER,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TARGET,
    THRESHOLD_PATH,
)
from pipeline.cost_matrix import sweep_thresholds
from pipeline.split import load_splits

log = logging.getLogger(__name__)


def score_split(name: str, art_dir=ARTIFACTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, fraud probability, raw Amount) for a split using saved artifacts."""
    part = load_splits()[name]
    scaler = joblib.load(art_dir / SCALER_PATH.name)
    model = joblib.load(art_dir / MODEL_PATH.name)
    proba = model.predict_proba(scaler.transform(part[FEATURE_ORDER]))[:, 1]
    return part[TARGET].to_numpy(), proba, part["Amount"].to_numpy()  # raw Amount for cost


def write_threshold(art_dir=ARTIFACTS) -> dict:
    metrics = json.loads((art_dir / METRICS_PATH.name).read_text())
    y, proba, amount = score_split("val", art_dir)
    threshold, val_cost = sweep_thresholds(y, proba, amount, C_FP)
    selected = metrics["selected"]
    payload = {
        "threshold": threshold,
        "c_fp": C_FP,
        "model_name": selected,
        "model_version": metrics["model_version"],
        "feature_order": FEATURE_ORDER,
        "val_cost": val_cost,
        "val_pr_auc": metrics["candidates"][selected]["val_pr_auc"],
    }
    (art_dir / THRESHOLD_PATH.name).write_text(json.dumps(payload, indent=2))
    log.info("threshold t*=%.2f val_cost=%.2f", threshold, val_cost)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    write_threshold()


if __name__ == "__main__":
    main()
