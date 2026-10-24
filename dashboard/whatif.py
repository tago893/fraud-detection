"""What-if threshold analysis from stored scores. Pure NumPy/pandas: no API, no database access."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipeline.cost_matrix import cost_at


@dataclass(frozen=True)
class WhatIf:
    threshold: float
    labeled: int
    frauds: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None  # None when nothing is flagged
    recall: float | None  # None when there is no fraud in the sample
    cost: float


def what_if(scores: pd.DataFrame, threshold: float, c_fp: float) -> WhatIf:
    """Precision, recall, and business cost if `threshold` had been used on the stored labeled rows.

    `scores` needs fraud_probability, true_label, and the raw amount. Flagged means
    probability >= threshold, the same rule the API applies. Cost uses raw Amount for misses.
    """
    proba = scores["fraud_probability"].to_numpy(dtype=float)
    label = scores["true_label"].to_numpy(dtype=int)
    amount = scores["amount"].to_numpy(dtype=float)
    flagged = proba >= threshold
    tp = int((flagged & (label == 1)).sum())
    fp = int((flagged & (label == 0)).sum())
    fn = int((~flagged & (label == 1)).sum())
    tn = int((~flagged & (label == 0)).sum())
    return WhatIf(
        threshold=threshold,
        labeled=len(scores),
        frauds=tp + fn,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        cost=cost_at(label, proba, amount, threshold, c_fp),
    )
