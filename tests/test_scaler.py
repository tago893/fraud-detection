import numpy as np
import pytest

from pipeline.config import FEATURE_ORDER, SPLITS_DIR
from pipeline.split import load_splits
from pipeline.train import fit_scaler

pytestmark = pytest.mark.skipif(
    not (SPLITS_DIR / "train.parquet").exists(), reason="run make split first"
)


@pytest.fixture(scope="module")
def splits():
    return load_splits()


@pytest.fixture(scope="module")
def scaler(splits):
    return fit_scaler(splits["train"])


def test_scaler_fit_on_train_only(splits, scaler):
    scaled = scaler.transform(splits["train"][FEATURE_ORDER])
    amount_col = FEATURE_ORDER.index("Amount")
    assert abs(np.median(scaled[:, amount_col])) < 1e-9
    # fitted center is the train median, not the val median
    assert (
        scaler.named_transformers_["robust"].center_[1]
        == splits["train"]["Amount"].median()
    )
    assert splits["train"]["Amount"].median() != splits["val"]["Amount"].median()


def test_v_columns_pass_through_and_order_kept(splits, scaler):
    val = splits["val"]
    out = scaler.transform(val[FEATURE_ORDER])
    np.testing.assert_allclose(out[:, 2:], val[FEATURE_ORDER[2:]].to_numpy())
    assert out.shape[1] == len(FEATURE_ORDER)
