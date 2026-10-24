"""SHAP explanations for flagged transactions (tree models only)."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import shap
from xgboost import XGBClassifier

TOP_K = 5


def unwrap_estimator(model: Any) -> Any:
    """The final estimator, if the saved model is an (imblearn) Pipeline."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def build_explainer(model: Any) -> shap.TreeExplainer | None:
    """A TreeExplainer if the model is tree based, else None (logistic regression has no SHAP here)."""
    estimator = unwrap_estimator(model)
    return shap.TreeExplainer(estimator) if isinstance(estimator, XGBClassifier) else None


def top_contributions(
    explainer: shap.TreeExplainer,
    scaled_row: np.ndarray,
    raw_row: np.ndarray,
    feature_order: Sequence[str],
    k: int = TOP_K,
) -> list[dict]:
    """Top-k features by |SHAP value| for one row. SHAP is computed on what the model saw (scaled)."""
    values = np.asarray(explainer.shap_values(scaled_row.reshape(1, -1)))
    values = values[-1] if values.ndim == 3 else values  # some versions return one array per class
    contributions = values.reshape(-1)
    order = np.argsort(-np.abs(contributions))[:k]
    return [
        {
            "feature": feature_order[i],
            "shap_value": float(contributions[i]),
            "feature_value": float(raw_row[i]),
        }
        for i in order
    ]
