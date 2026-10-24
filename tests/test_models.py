import pytest

from pipeline.train import train_logreg, val_scores

from .conftest import needs_splits

pytestmark = needs_splits


@pytest.fixture(scope="module")
def logreg(xy):
    return train_logreg(*xy["train"])


def test_logreg_val_pr_auc_above_half(logreg, xy):
    scores = val_scores(logreg, *xy["val"])
    print(
        f"LR val PR-AUC={scores['val_pr_auc']:.4f} ROC-AUC={scores['val_roc_auc']:.4f}"
    )
    assert scores["val_pr_auc"] > 0.5


def test_xgboost_matches_or_beats_logreg(logreg, xy):
    from pipeline.train import train_xgboost

    model = train_xgboost(*xy["train"], *xy["val"])
    xgb = val_scores(model, *xy["val"])
    lr = val_scores(logreg, *xy["val"])
    print(
        f"XGB val PR-AUC={xgb['val_pr_auc']:.4f} ROC-AUC={xgb['val_roc_auc']:.4f} "
        f"best_iter={model.best_iteration} | LR PR-AUC={lr['val_pr_auc']:.4f}"
    )
    assert xgb["val_pr_auc"] >= lr["val_pr_auc"]


def test_smote_branch_leaves_val_and_test_untouched(splits, xy):
    from imblearn.pipeline import Pipeline as ImbPipeline

    from pipeline.split import load_splits
    from pipeline.train import train_smote_xgboost

    sizes_before = {n: len(splits[n]) for n in ("val", "test")}
    x_val_before, x_test_before = xy["val"][0].copy(), xy["test"][0].copy()
    train_rows = len(xy["train"][1])

    model, cv = train_smote_xgboost(*xy["train"])

    assert isinstance(model, ImbPipeline) and len(cv) == 5
    assert len(xy["train"][1]) == train_rows  # caller's train arrays are not resampled
    reloaded = load_splits()
    assert {n: len(reloaded[n]) for n in ("val", "test")} == sizes_before
    assert (
        xy["val"][0].shape[0] == sizes_before["val"]
        and xy["test"][0].shape[0] == sizes_before["test"]
    )
    assert (xy["val"][0] == x_val_before).all() and (
        xy["test"][0] == x_test_before
    ).all()
    scores = val_scores(model, *xy["val"])
    print(
        f"SMOTE+XGB cv_pr_auc={[round(v, 3) for v in cv]} val PR-AUC={scores['val_pr_auc']:.4f}"
    )
