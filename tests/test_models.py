import pytest

from pipeline.train import train_logreg, val_scores

from .conftest import needs_splits

pytestmark = needs_splits


@pytest.fixture(scope="module")
def logreg(xy):
    return train_logreg(*xy["train"])


def test_logreg_val_pr_auc_above_half(logreg, xy):
    scores = val_scores(logreg, *xy["val"])
    print(f"LR val PR-AUC={scores['val_pr_auc']:.4f} ROC-AUC={scores['val_roc_auc']:.4f}")
    assert scores["val_pr_auc"] > 0.5
