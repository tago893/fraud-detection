"""Pick the cost-optimal threshold on val, then report on test once."""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
from pipeline.cost_matrix import cost_at, flag_nothing_cost, sweep_thresholds
from pipeline.split import load_splits

log = logging.getLogger(__name__)


def score_split(
    name: str, art_dir=ARTIFACTS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, fraud probability, raw Amount) for a split using saved artifacts."""
    part = load_splits()[name]
    scaler = joblib.load(art_dir / SCALER_PATH.name)
    model = joblib.load(art_dir / MODEL_PATH.name)
    proba = model.predict_proba(scaler.transform(part[FEATURE_ORDER]))[:, 1]
    return (
        part[TARGET].to_numpy(),
        proba,
        part["Amount"].to_numpy(),
    )  # raw Amount for cost


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


def split_report(
    y: np.ndarray, proba: np.ndarray, amount: np.ndarray, threshold: float, c_fp: float
) -> dict:
    """Threshold-free (PR/ROC-AUC) and at-threshold metrics, plus cost against two baselines."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "cost": cost_at(y, proba, amount, threshold, c_fp),
        "cost_flag_nothing": flag_nothing_cost(y, amount),
        "cost_at_0.5": cost_at(y, proba, amount, 0.5, c_fp),
    }


def downsample_pr_curve(
    y: np.ndarray, proba: np.ndarray, max_points: int = 200
) -> list[dict]:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    idx = np.unique(
        np.linspace(0, len(thresholds) - 1, min(max_points, len(thresholds))).astype(
            int
        )
    )
    return [
        {
            "threshold": float(thresholds[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
        }
        for i in idx
    ]


def format_table(val: dict, test: dict) -> str:
    rows = [
        ("threshold", "threshold", "{:.2f}"),
        ("PR-AUC", "pr_auc", "{:.4f}"),
        ("ROC-AUC", "roc_auc", "{:.4f}"),
        ("precision", "precision", "{:.4f}"),
        ("recall", "recall", "{:.4f}"),
        ("F1", "f1", "{:.4f}"),
        ("cost @ t*", "cost", "${:,.2f}"),
        ("cost @ 0.5", "cost_at_0.5", "${:,.2f}"),
        ("cost, flag nothing", "cost_flag_nothing", "${:,.2f}"),
    ]
    lines = [f"{'metric':<20}{'val':>14}{'test':>14}", "-" * 48]
    lines += [
        f"{label:<20}{fmt.format(val[k]):>14}{fmt.format(test[k]):>14}"
        for label, k, fmt in rows
    ]
    for part in ("tn", "fp", "fn", "tp"):
        lines.append(
            f"{'confusion ' + part:<20}{val['confusion_matrix'][part]:>14,}{test['confusion_matrix'][part]:>14,}"
        )
    return "\n".join(lines)


def write_test_report(threshold: float, art_dir=ARTIFACTS) -> str:
    """Score test exactly once at the val-chosen threshold and store it in metrics.json."""
    y_val, p_val, a_val = score_split("val", art_dir)
    y_test, p_test, a_test = score_split("test", art_dir)
    val = split_report(y_val, p_val, a_val, threshold, C_FP)
    test = split_report(y_test, p_test, a_test, threshold, C_FP)
    path = art_dir / METRICS_PATH.name
    metrics = json.loads(path.read_text())
    metrics.update(
        {"test": test, "val": val, "pr_curve": downsample_pr_curve(y_val, p_val)}
    )
    path.write_text(json.dumps(metrics, indent=2))
    return format_table(val, test)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = write_threshold()
    print(write_test_report(payload["threshold"]))


if __name__ == "__main__":
    main()
