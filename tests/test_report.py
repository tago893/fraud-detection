import numpy as np

from pipeline.evaluate import downsample_pr_curve, split_report

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


def test_pr_curve_downsampled_to_about_200():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    curve = downsample_pr_curve(y, rng.random(5000), max_points=200)
    assert 100 <= len(curve) <= 201
    assert set(curve[0]) == {"threshold", "precision", "recall"}
