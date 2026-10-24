import json

import pytest

from pipeline.readme_table import END, START, render_results, splice, update_readme

CM = {"tn": 10, "fp": 2, "fn": 3, "tp": 9}
SPLIT = {
    "threshold": 0.85,
    "pr_auc": 0.8,
    "roc_auc": 0.98,
    "precision": 0.75,
    "recall": 0.5,
    "f1": 0.6,
    "cost": 276.3,
    "cost_at_0.5": 1051.55,
    "cost_flag_nothing": 8397.7,
    "confusion_matrix": CM,
}
METRICS = {
    "selected": "xgboost",
    "model_version": "2026-01-01T00:00:00Z",
    "candidates": {
        "logreg": {"val_pr_auc": 0.8395, "val_roc_auc": 0.9829},
        "xgboost": {"val_pr_auc": 0.8522, "val_roc_auc": 0.9862, "cv_pr_auc": [0.1]},
    },
    "val": SPLIT,
    "test": {**SPLIT, "pr_auc": 0.7, "cost": 1743.31},
}


def test_render_contains_every_candidate_and_marks_the_winner():
    text = render_results(METRICS)
    assert "| `logreg` | 0.8395 | 0.9829 |" in text
    assert "| `xgboost` (selected) | 0.8522 | 0.9862 |" in text


def test_render_shows_val_and_test_side_by_side():
    text = render_results(METRICS)
    assert "| PR-AUC | 0.8000 | 0.7000 |" in text
    assert "| Cost at t* | $276.30 | $1,743.31 |" in text
    assert "| False negatives | 3 | 3 |" in text
    assert "t* = 0.85" in text and "2026-01-01T00:00:00Z" in text


def test_splice_replaces_only_the_marked_block():
    readme = f"intro\n{START}\nold numbers\n{END}\noutro\n"
    out = splice(readme, "NEW")
    assert out == f"intro\n{START}\nNEW\n{END}\noutro\n"
    assert splice(out, "NEW") == out  # idempotent


def test_splice_without_markers_fails_loudly():
    with pytest.raises(ValueError, match="markers"):
        splice("no markers here", "x")


def test_update_readme_end_to_end(tmp_path):
    metrics_path, readme_path = tmp_path / "metrics.json", tmp_path / "README.md"
    metrics_path.write_text(json.dumps(METRICS))
    readme_path.write_text(f"# T\n{START}\n{END}\n")
    update_readme(readme_path, metrics_path)
    assert "| `xgboost` (selected)" in readme_path.read_text()
