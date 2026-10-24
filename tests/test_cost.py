import numpy as np
import pytest

from pipeline.cost_matrix import cost_at, flag_nothing_cost, sweep_thresholds

# Hand-computed 5-row example, C_FP = 10.
#   row:    0     1     2     3     4
#   y:      1     0     1     0     0
#   amount: 100   50    200   30    20
#   proba:  0.9   0.6   0.3   0.2   0.1
Y = np.array([1, 0, 1, 0, 0])
AMOUNT = np.array([100.0, 50.0, 200.0, 30.0, 20.0])
PROBA = np.array([0.9, 0.6, 0.3, 0.2, 0.1])


def test_cost_at_threshold_half():
    # flags rows 0,1 -> FP=1 (row 1), FN=row 2 (raw Amount 200): 10 + 200
    assert cost_at(Y, PROBA, AMOUNT, 0.5, c_fp=10) == 210.0


def test_cost_catching_everything_costs_only_false_alarms():
    # t=0.25 flags rows 0,1,2 -> FP=1, no FN
    assert cost_at(Y, PROBA, AMOUNT, 0.25, c_fp=10) == 10.0
    # t=0.15 also flags row 3 -> FP=2
    assert cost_at(Y, PROBA, AMOUNT, 0.15, c_fp=10) == 20.0


def test_flag_nothing_cost_is_total_fraud_amount():
    assert flag_nothing_cost(Y, AMOUNT) == 300.0
    # t=0.99 is above every score, so nothing is flagged
    assert cost_at(Y, PROBA, AMOUNT, 0.99, c_fp=10) == 300.0


def test_sweep_picks_lowest_cost_and_beats_half():
    best_t, best_cost = sweep_thresholds(Y, PROBA, AMOUNT, c_fp=10)
    assert best_cost == 10.0
    assert 0.2 < best_t <= 0.3
    assert best_cost <= cost_at(Y, PROBA, AMOUNT, 0.5, c_fp=10)


def test_uses_raw_amount_not_scaled():
    scaled = (AMOUNT - AMOUNT.mean()) / AMOUNT.std()
    assert cost_at(Y, PROBA, AMOUNT, 0.5, 10) != cost_at(Y, PROBA, scaled, 0.5, 10)


def test_val_chosen_threshold_not_worse_than_half():
    from pipeline.config import ARTIFACTS, C_FP, FEATURE_ORDER
    from pipeline.evaluate import score_split

    if not (ARTIFACTS / "model.joblib").exists():
        pytest.skip("run make train first")
    y, proba, amount = score_split("val")
    t, c = sweep_thresholds(y, proba, amount, C_FP)
    assert c <= cost_at(y, proba, amount, 0.5, C_FP)
    assert 0.01 <= t <= 0.99 and len(FEATURE_ORDER) == 30
