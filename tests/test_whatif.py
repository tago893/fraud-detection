import ast
from pathlib import Path

import pandas as pd
import pytest

from dashboard.whatif import what_if

# Same hand-computed 5-row example as tests/test_cost.py, C_FP = 10
SCORES = pd.DataFrame(
    {
        "true_label": [1, 0, 1, 0, 0],
        "amount": [100.0, 50.0, 200.0, 30.0, 20.0],
        "fraud_probability": [0.9, 0.6, 0.3, 0.2, 0.1],
    }
)


def test_threshold_half():
    r = what_if(SCORES, 0.5, c_fp=10)
    assert (r.tp, r.fp, r.fn, r.tn) == (1, 1, 1, 2)
    assert (r.precision, r.recall, r.cost) == (0.5, 0.5, 210.0)  # 10 + missed 200


def test_lower_threshold_catches_everything():
    r = what_if(SCORES, 0.25, c_fp=10)
    assert (r.tp, r.fp, r.fn) == (2, 1, 0)
    assert r.precision == pytest.approx(2 / 3) and r.recall == 1.0 and r.cost == 10.0


def test_nothing_flagged_has_no_precision():
    r = what_if(SCORES, 0.99, c_fp=10)
    assert r.precision is None and r.recall == 0.0 and r.cost == 300.0


def test_no_fraud_in_sample_has_no_recall():
    r = what_if(SCORES.assign(true_label=0), 0.5, c_fp=10)
    assert (
        r.recall is None and r.frauds == 0 and r.fp == 2 and r.cost == 20.0
    )  # rows 0 and 1 flagged


def test_flag_rule_matches_the_api_inclusive_comparison():
    assert what_if(SCORES, 0.6, c_fp=10).fp == 1  # probability == threshold is flagged


def test_whatif_module_cannot_reach_the_api_or_the_database():
    tree = ast.parse(
        (Path(__file__).parent.parent / "dashboard" / "whatif.py").read_text()
    )
    imported = {
        a.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for a in n.names
    }
    imported |= {
        n.module.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not imported & {"httpx", "requests", "sqlite3", "api", "dashboard"}
