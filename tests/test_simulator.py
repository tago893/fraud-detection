import asyncio
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from api import database as db
from api.main import create_app
from pipeline.config import ARTIFACTS, FEATURE_ORDER, REPLAY_PATH, TARGET
from simulator import stream_txns
from simulator.stream_txns import ANOMALY_PREFIX, SimConfig, payload_stream, run

pytestmark = pytest.mark.skipif(
    not ((ARTIFACTS / "threshold.json").exists() and REPLAY_PATH.exists()),
    reason="run make split train evaluate first",
)


@pytest.fixture(scope="module")
def replay() -> pd.DataFrame:
    return stream_txns.load_replay()


def simulate_in_process(cfg: SimConfig, db_path: Path):
    """Run the simulator against the real app in this process (no sockets)."""
    app = create_app(db_path=db_path)

    async def go():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                return await run(cfg, client=client)

    return asyncio.run(go())


def stored_rows(db_path: Path) -> list[tuple]:
    with db.connect(db_path) as conn:
        return conn.execute(
            "SELECT transaction_id, scored_at, amount, true_label FROM transactions"
        ).fetchall()


# ---- 3.1 replay stream -------------------------------------------------------------------


def test_reads_only_the_replay_parquet(monkeypatch):
    read: list[Path] = []
    real = pd.read_parquet
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path, *a, **k: (read.append(Path(path)), real(path, *a, **k))[1],
    )
    stream_txns.load_replay()
    assert read == [REPLAY_PATH]


def test_simulator_source_never_references_other_splits():
    source = Path(stream_txns.__file__).read_text()
    for forbidden in (
        "load_splits",
        "SPLITS_DIR",
        "train.parquet",
        "val.parquet",
        "test.parquet",
    ):
        assert forbidden not in source


def test_stream_replays_rows_in_order_with_real_labels(replay):
    rows = [
        next(s)
        for s in [payload_stream(replay, 0.0, np.random.default_rng(1))]
        for _ in range(50)
    ]
    expected = replay.head(50)
    assert [r["Time"] for r in rows] == expected["Time"].tolist()
    assert [r["true_label"] for r in rows] == expected[TARGET].astype(int).tolist()
    assert all(
        set(r) == set(FEATURE_ORDER) | {"true_label", "transaction_id"} for r in rows
    )
    assert not any(r["transaction_id"].startswith(ANOMALY_PREFIX) for r in rows)
    assert len({r["transaction_id"] for r in rows}) == 50


def test_stream_loops_when_replay_is_exhausted(replay):
    stream = payload_stream(replay.head(3), 0.0, np.random.default_rng(1))
    assert [next(stream)["Time"] for _ in range(6)] == replay.head(3)[
        "Time"
    ].tolist() * 2


def test_steady_rate_sends_expected_requests_with_zero_errors(tmp_path):
    stats = simulate_in_process(
        SimConfig(rate=100, duration=1, burst_every=0), tmp_path / "tx.db"
    )
    assert (stats.sent, stats.errors) == (100, 0)
    assert len(stored_rows(tmp_path / "tx.db")) == 100


def test_count_mode(tmp_path):
    stats = simulate_in_process(
        SimConfig(rate=200, count=37, burst_every=0), tmp_path / "tx.db"
    )
    assert stats.sent == 37 and len(stored_rows(tmp_path / "tx.db")) == 37


def test_server_errors_are_counted_not_raised(replay):
    transport = httpx.MockTransport(lambda request: httpx.Response(500))

    async def go():
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await run(
                SimConfig(rate=200, count=10, burst_every=0),
                client=client,
                replay=replay,
            )

    stats = asyncio.run(go())
    assert (stats.sent, stats.errors, stats.ok) == (10, 10, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate": 0},
        {"duration": 1, "count": 1},
        {"anomaly_rate": 2},
        {"burst_size": -1},
        {"count": 0},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SimConfig(**kwargs).validate()


def test_cli_rejects_both_duration_and_count():
    with pytest.raises(SystemExit):
        stream_txns.parse_args(["--duration", "1", "--count", "1"])


# ---- 3.2 bursts and anomalies ------------------------------------------------------------


def test_anomalies_are_amount_multiplied_copies_with_prefix(replay):
    stream = payload_stream(replay, 1.0, np.random.default_rng(7))
    for original, body in zip(
        replay.head(100).to_dict(orient="records"), (next(stream) for _ in range(100))
    ):
        assert body["transaction_id"].startswith(ANOMALY_PREFIX)
        ratio = body["Amount"] / original["Amount"] if original["Amount"] else 10.0
        assert 10.0 <= ratio <= 50.0 or original["Amount"] == 0
        assert {k: body[k] for k in FEATURE_ORDER if k != "Amount"} == {
            k: original[k] for k in FEATURE_ORDER if k != "Amount"
        }
        assert body["true_label"] == int(original[TARGET])


def test_anomaly_rows_reach_the_database(tmp_path):
    simulate_in_process(
        SimConfig(rate=200, count=100, anomaly_rate=0.3, burst_every=0),
        tmp_path / "tx.db",
    )
    ids = [r[0] for r in stored_rows(tmp_path / "tx.db")]
    anomalies = [i for i in ids if i.startswith(ANOMALY_PREFIX)]
    assert 10 <= len(anomalies) <= 50 and len(ids) == 100


def test_burst_shows_up_as_a_per_second_spike(tmp_path):
    stats = simulate_in_process(
        SimConfig(rate=10, duration=2, burst_every=1, burst_size=40, anomaly_rate=0),
        tmp_path / "tx.db",
    )
    assert stats.errors == 0 and stats.sent >= 20 + 40
    per_second = Counter(
        scored_at[:19] for _, scored_at, _, _ in stored_rows(tmp_path / "tx.db")
    )
    assert max(per_second.values()) >= 40 > 10  # baseline is ~10/s
