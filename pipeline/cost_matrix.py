"""Business cost of a threshold: C_FP per false alarm plus the raw Amount of each missed fraud."""

from __future__ import annotations

import numpy as np

THRESHOLD_GRID = np.round(np.arange(1, 100) / 100, 2)  # 0.01 .. 0.99


def cost_at(
    y_true: np.ndarray,
    proba: np.ndarray,
    amount: np.ndarray,
    threshold: float,
    c_fp: float,
) -> float:
    """Cost(t) = c_fp * FP(t) + sum of raw Amount over FN(t). Flagged means proba >= threshold."""
    flagged = proba >= threshold
    false_pos = int(np.sum(flagged & (y_true == 0)))
    missed = float(np.sum(amount[~flagged & (y_true == 1)]))
    return c_fp * false_pos + missed


def flag_nothing_cost(y_true: np.ndarray, amount: np.ndarray) -> float:
    return float(np.sum(amount[y_true == 1]))


def sweep_thresholds(
    y_true: np.ndarray, proba: np.ndarray, amount: np.ndarray, c_fp: float
) -> tuple[float, float]:
    """Return (threshold, cost) with the lowest cost on a 0.01..0.99 grid; ties take the lowest threshold."""
    costs = np.array(
        [cost_at(y_true, proba, amount, float(t), c_fp) for t in THRESHOLD_GRID]
    )
    best = int(np.argmin(costs))
    return float(THRESHOLD_GRID[best]), float(costs[best])
