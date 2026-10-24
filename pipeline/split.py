"""Time-ordered split: sort by Time, slice by row position. Never shuffle."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import DATA_PATH, SPLIT_FRACTIONS, SPLITS_DIR, TARGET

log = logging.getLogger(__name__)

SPLIT_ORDER = ("train", "val", "test", "replay")


def split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stable-sort by Time and slice by cumulative fractions. Index = original row id."""
    ordered = df.sort_values("Time", kind="stable")
    edges = np.round(np.cumsum([SPLIT_FRACTIONS[n] for n in SPLIT_ORDER]) * len(ordered)).astype(int)
    edges[-1] = len(ordered)
    starts = [0, *edges[:-1]]
    return {name: ordered.iloc[s:e] for name, s, e in zip(SPLIT_ORDER, starts, edges)}


def write_splits(splits: dict[str, pd.DataFrame], out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=True)


def load_splits(src_dir: Path = SPLITS_DIR) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(src_dir / f"{name}.parquet") for name in SPLIT_ORDER}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}; download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud).")
    splits = split_frame(pd.read_csv(DATA_PATH))
    write_splits(splits)
    for name, part in splits.items():
        log.info("%-6s rows=%7d fraud=%4d (%.3f%%) Time=[%.0f, %.0f]", name, len(part),
                 int(part[TARGET].sum()), 100 * part[TARGET].mean(), part["Time"].min(), part["Time"].max())


if __name__ == "__main__":
    main()
