"""Fit the scaler and train candidate models. Fit on train only; val is for selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from pipeline.config import FEATURE_ORDER, SCALED_COLUMNS, SEED, TARGET


def fit_scaler(train: pd.DataFrame) -> ColumnTransformer:
    """RobustScaler on Time and Amount only, fit on the train slice. V1-V28 pass through.

    Output column order equals FEATURE_ORDER because Time and Amount lead it.
    """
    scaler = ColumnTransformer(
        [("robust", RobustScaler(), SCALED_COLUMNS)], remainder="passthrough", verbose_feature_names_out=False
    )
    return scaler.fit(train[FEATURE_ORDER])


def to_xy(scaler: ColumnTransformer, part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Scale a split with the already-fitted scaler (never refit)."""
    return scaler.transform(part[FEATURE_ORDER]), part[TARGET].to_numpy()


def val_scores(model, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """PR-AUC and ROC-AUC. Accuracy is meaningless at 0.17% fraud."""
    proba = model.predict_proba(x_val)[:, 1]
    return {
        "val_pr_auc": float(average_precision_score(y_val, proba)),
        "val_roc_auc": float(roc_auc_score(y_val, proba)),
    }


def train_logreg(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)
    return model.fit(x_train, y_train)


def train_xgboost(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray
) -> XGBClassifier:
    """Class-weighted XGBoost; early stopping on the val slice (val is never trained on)."""
    pos = int(y_train.sum())
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y_train) - pos) / pos,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )
    return model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)


def train_smote_xgboost(
    x_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5
) -> tuple[ImbPipeline, list[float]]:
    """SMOTE + XGBoost inside an imblearn pipeline, scored with TimeSeriesSplit on train.

    SMOTE lives in the pipeline so it only resamples the training part of each fold,
    never the validation fold, val, or test.
    """
    pipe = ImbPipeline(
        [
            ("smote", SMOTE(sampling_strategy=0.1, random_state=SEED)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="aucpr",
                    tree_method="hist",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    cv_scores = cross_val_score(
        pipe, x_train, y_train, cv=TimeSeriesSplit(n_splits=n_splits), scoring="average_precision"
    )
    return pipe.fit(x_train, y_train), [float(v) for v in cv_scores]
