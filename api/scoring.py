"""Turn a validated request into a fraud probability. Only the 30 features reach the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer


@dataclass(frozen=True)
class Scorer:
    """Scales a raw feature row with the train-fitted RobustScaler parameters, then scores it.

    The ColumnTransformer needs a DataFrame; applying its fitted center/scale to a NumPy row
    gives identical numbers without building a frame on the hot path (parity is tested).
    """

    model: Any
    center: np.ndarray
    scale: np.ndarray
    feature_order: tuple[str, ...]

    @classmethod
    def from_artifacts(
        cls, model: Any, scaler: ColumnTransformer, feature_order: Sequence[str]
    ) -> Scorer:
        order = tuple(feature_order)
        center, scale = np.zeros(len(order)), np.ones(len(order))
        robust = scaler.named_transformers_["robust"]
        scaled_columns = scaler.transformers_[0][2]
        for i, name in enumerate(scaled_columns):
            idx = order.index(name)
            center[idx], scale[idx] = robust.center_[i], robust.scale_[i]
        return cls(model=model, center=center, scale=scale, feature_order=order)

    def vectorize(self, request: Any) -> np.ndarray:
        """Raw feature row in the fixed feature order. true_label is never read."""
        values = (getattr(request, name) for name in self.feature_order)
        return np.fromiter(values, dtype=float, count=len(self.feature_order))

    def scale_row(self, raw: np.ndarray) -> np.ndarray:
        return (raw - self.center) / self.scale

    def predict_proba(self, raw: np.ndarray) -> float:
        return float(self.model.predict_proba(self.scale_row(raw).reshape(1, -1))[0, 1])
