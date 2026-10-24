"""Fit the scaler and train candidate models. Fit on train only; val is for selection."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from pipeline.config import FEATURE_ORDER, SCALED_COLUMNS


def fit_scaler(train: pd.DataFrame) -> ColumnTransformer:
    """RobustScaler on Time and Amount only, fit on the train slice. V1-V28 pass through.

    Output column order equals FEATURE_ORDER because Time and Amount lead it.
    """
    scaler = ColumnTransformer(
        [("robust", RobustScaler(), SCALED_COLUMNS)], remainder="passthrough", verbose_feature_names_out=False
    )
    return scaler.fit(train[FEATURE_ORDER])
