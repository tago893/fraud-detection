import numpy as np

from pipeline.evaluate import pr_curve_points, split_report

Y = np.array([1, 0, 1, 0, 0])
AMOUNT = np.array([100.0, 50.0, 200.0, 30.0, 20.0])
PROBA = np.array([0.9, 0.6, 0.3, 0.2, 0.1])


def test_split_report_at_threshold():
    r = split_report(Y, PROBA, AMOUNT, threshold=0.5, c_fp=10)
    assert r["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5
    assert (
        r["cost"] == 210.0
        and r["cost_flag_nothing"] == 300.0
        and r["cost_at_0.5"] == 210.0
    )


def test_pr_curve_has_about_200_points_spaced_by_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    curve = pr_curve_points(y, rng.random(5000))
    assert 150 <= len(curve) <= 200
    assert set(curve[0]) == {"threshold", "precision", "recall"}
    gaps = np.diff([p["threshold"] for p in curve])
    assert np.allclose(gaps, 0.005)


def test_pr_curve_covers_the_useful_range_when_scores_are_skewed_toward_zero():
    # 99 % of scores below 0.001, like a fraud model: rank-based sampling would miss 0.01-0.99
    rng = np.random.default_rng(1)
    proba = np.where(
        rng.random(20000) < 0.99, rng.random(20000) * 0.001, rng.random(20000)
    )
    y = (proba > 0.7).astype(int)
    curve = pr_curve_points(y, proba)
    mid = [p for p in curve if 0.01 <= p["threshold"] <= 0.99]
    assert len(mid) >= 150


def test_pr_curve_point_matches_direct_computation():
    point = {p["threshold"]: p for p in pr_curve_points(Y, PROBA)}[0.5]
    # threshold 0.5 flags rows 0 and 1: tp=1 of 2 flagged, 1 of 2 positives
    assert (point["precision"], point["recall"]) == (0.5, 0.5)


def test_pr_curve_skips_thresholds_where_nothing_is_flagged():
    assert max(p["threshold"] for p in pr_curve_points(Y, PROBA)) <= 0.9
