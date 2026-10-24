"""Single source of truth for paths, seeds, costs, and feature order."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
ARTIFACTS = ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS / "splits"
MODEL_PATH = ARTIFACTS / "model.joblib"
SCALER_PATH = ARTIFACTS / "scaler.joblib"
THRESHOLD_PATH = ARTIFACTS / "threshold.json"
METRICS_PATH = ARTIFACTS / "metrics.json"

SEED = 42
C_FP = 10.0  # dollars of friction per false alarm

# Time-ordered slice boundaries (fractions of rows): train, val, test, replay.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.10, "replay": 0.05}

FEATURE_ORDER = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
SCALED_COLUMNS = ["Time", "Amount"]
TARGET = "Class"
