import pandas as pd
import pytest

from pipeline.config import DATA_PATH, FEATURE_ORDER, TARGET

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="creditcard.csv missing")


def test_raw_data_shape_and_fraud_count():
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (284_807, 31)
    assert int((df[TARGET] == 1).sum()) == 492
    assert set(FEATURE_ORDER) | {TARGET} == set(df.columns)
