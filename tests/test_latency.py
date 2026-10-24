import pytest

from api.latency_check import P95_BUDGET_MS, run
from pipeline.config import ARTIFACTS, SPLITS_DIR

pytestmark = pytest.mark.skipif(
    not (
        (ARTIFACTS / "threshold.json").exists()
        and (SPLITS_DIR / "replay.parquet").exists()
    ),
    reason="run make split train evaluate first",
)


def test_p95_latency_under_budget():
    result = run(n=1000)
    print({k: round(v, 3) for k, v in result.items()})
    assert result["requests"] == 1000
    assert result["p95_ms"] < P95_BUDGET_MS
