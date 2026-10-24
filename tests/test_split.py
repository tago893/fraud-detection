import pandas as pd
import pytest

from pipeline.config import DATA_PATH, SPLIT_FRACTIONS
from pipeline.split import SPLIT_ORDER, load_splits, split_frame, write_splits

pytestmark = pytest.mark.skipif(not DATA_PATH.exists(), reason="creditcard.csv missing")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def splits(raw):
    return split_frame(raw)


def test_time_ordered_without_overlap(splits):
    for a, b in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        assert splits[a]["Time"].max() <= splits[b]["Time"].min()


def test_sizes_match_fractions(raw, splits):
    for name in SPLIT_ORDER:
        assert abs(len(splits[name]) - SPLIT_FRACTIONS[name] * len(raw)) <= 1
    assert sum(len(s) for s in splits.values()) == len(raw)


def test_no_row_in_two_splits(raw, splits):
    ids = [i for name in SPLIT_ORDER for i in splits[name].index]
    assert len(ids) == len(set(ids)) == len(raw)


def test_parquet_roundtrip_keeps_raw_values(raw, splits, tmp_path):
    write_splits(splits, tmp_path)
    loaded = load_splits(tmp_path)
    for name in SPLIT_ORDER:
        pd.testing.assert_frame_equal(loaded[name], splits[name])
    # raw Amount, not scaled
    assert loaded["train"]["Amount"].max() == raw.loc[loaded["train"].index, "Amount"].max()
