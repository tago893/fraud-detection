from pipeline.train import select_winner


def test_select_winner_by_val_pr_auc():
    cands = {
        "logreg": {"val_pr_auc": 0.80, "val_roc_auc": 0.99},
        "xgboost": {"val_pr_auc": 0.85, "val_roc_auc": 0.97},
        "smote_xgboost": {"val_pr_auc": 0.83, "val_roc_auc": 0.99},
    }
    assert select_winner(cands) == "xgboost"
