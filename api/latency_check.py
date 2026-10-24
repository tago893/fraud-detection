"""Send N requests to the app in process and report latency percentiles.

python -m api.latency_check
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app
from pipeline.config import FEATURE_ORDER, SEED
from pipeline.split import load_splits

log = logging.getLogger(__name__)

N_REQUESTS = 1000
WARMUP = 20
P95_BUDGET_MS = 10.0


def run(n: int = N_REQUESTS, warmup: int = WARMUP) -> dict[str, float]:
    """Score n random replay-slice rows. latency_ms is the server-side figure from each response;
    round-trip is wall time per call through the in-process client (validation, JSON, and background
    work included), so it is an upper bound on what a real client would see minus the network."""
    replay = load_splits()["replay"]
    sample = replay.sample(n=n + warmup, random_state=SEED)
    payloads = sample[FEATURE_ORDER].to_dict(orient="records")

    latencies: list[float] = []
    round_trips: list[float] = []
    flagged = 0
    with (
        tempfile.TemporaryDirectory() as tmp,
        TestClient(create_app(db_path=Path(tmp) / "latency.db")) as client,
    ):
        for i, payload in enumerate(payloads):
            started = time.perf_counter()
            response = client.post("/score", json=payload)
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            if i >= warmup:
                body = response.json()
                latencies.append(body["latency_ms"])
                round_trips.append(elapsed_ms)
                flagged += body["flagged"]

    return {
        "requests": float(len(latencies)),
        "flagged": float(flagged),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "max_ms": float(np.max(latencies)),
        "round_trip_p95_ms": float(np.percentile(round_trips, 95)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(
        logging.WARNING
    )  # one INFO line per request otherwise
    result = run()
    for key, value in result.items():
        log.info("%-18s %10.3f", key, value)
    verdict = "PASS" if result["p95_ms"] < P95_BUDGET_MS else "FAIL"
    log.info(
        "p95 %.3f ms vs budget %.1f ms: %s", result["p95_ms"], P95_BUDGET_MS, verdict
    )


if __name__ == "__main__":
    main()
