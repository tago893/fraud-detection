import pytest

from pipeline.config import SPLITS_DIR
from pipeline.split import load_splits
from pipeline.train import fit_scaler, to_xy

needs_splits = pytest.mark.skipif(not (SPLITS_DIR / "train.parquet").exists(), reason="run make split first")


@pytest.fixture(scope="session")
def splits():
    return load_splits()


@pytest.fixture(scope="session")
def scaler(splits):
    return fit_scaler(splits["train"])


@pytest.fixture(scope="session")
def xy(splits, scaler):
    return {name: to_xy(scaler, part) for name, part in splits.items()}
