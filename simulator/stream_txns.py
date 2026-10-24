"""Replay the held-out `replay` slice against the scoring API.

    python -m simulator.stream_txns --rate 10 --duration 10

Reads only artifacts/splits/replay.parquet. Never touches the train, val, or test slices.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Iterator

import httpx
import numpy as np
import pandas as pd

from pipeline.config import FEATURE_ORDER, REPLAY_PATH, SEED, TARGET

log = logging.getLogger("simulator")

ANOMALY_PREFIX = "anom-"
ANOMALY_MULTIPLIER = (10.0, 50.0)
STATS_EVERY_S = 5.0
MAX_CONNECTIONS = 200  # bursts fire many requests at once


@dataclass(frozen=True)
class SimConfig:
    url: str = "http://localhost:8000"
    rate: float = 10.0
    duration: float | None = None
    count: int | None = None
    burst_every: float = 60.0
    burst_size: int = 100
    anomaly_rate: float = 0.01
    seed: int = SEED

    def validate(self) -> None:
        if self.rate <= 0:
            raise ValueError("--rate must be > 0")
        if self.duration is not None and self.count is not None:
            raise ValueError("use either --duration or --count, not both")
        if self.duration is not None and self.duration <= 0:
            raise ValueError("--duration must be > 0")
        if self.count is not None and self.count <= 0:
            raise ValueError("--count must be > 0")
        if not 0.0 <= self.anomaly_rate <= 1.0:
            raise ValueError("--anomaly-rate must be in [0, 1]")
        if self.burst_size < 0 or self.burst_every < 0:
            raise ValueError("--burst-every and --burst-size must be >= 0")

    @property
    def planned_requests(self) -> int | None:
        """Base-rate requests to send (bursts are extra), or None to run until interrupted."""
        if self.count is not None:
            return self.count
        if self.duration is not None:
            return round(self.duration * self.rate)
        return None


@dataclass
class Stats:
    sent: int = 0
    flagged: int = 0
    errors: int = 0
    rtt_ms_total: float = 0.0
    server_ms_total: float = 0.0
    ok: int = 0

    def line(self) -> str:
        mean_rtt = self.rtt_ms_total / self.ok if self.ok else 0.0
        mean_server = self.server_ms_total / self.ok if self.ok else 0.0
        return (
            f"sent={self.sent} flagged={self.flagged} errors={self.errors} "
            f"mean_rtt_ms={mean_rtt:.1f} mean_server_ms={mean_server:.2f}"
        )


def load_replay() -> pd.DataFrame:
    """The replay slice, in time order. This is the only data file the simulator reads."""
    if not REPLAY_PATH.exists():
        raise FileNotFoundError(f"{REPLAY_PATH} not found; run `make split` first.")
    return pd.read_parquet(REPLAY_PATH)


def payload_stream(
    replay: pd.DataFrame, anomaly_rate: float, rng: np.random.Generator
) -> Iterator[dict]:
    """Endless stream of request bodies: replay rows in time order, looping at the end.

    true_label is the row's real Class (simulation stats only). An anomaly is a copy of the
    current row with Amount multiplied by 10-50 and an `anom-` transaction_id; it keeps the
    copied row's label.
    """
    features = replay[FEATURE_ORDER].to_dict(orient="records")
    labels = replay[TARGET].astype(int).tolist()
    while True:
        for row, label in zip(features, labels):
            body = {**row, "true_label": label, "transaction_id": uuid.uuid4().hex}
            if rng.random() < anomaly_rate:
                body["Amount"] = float(row["Amount"]) * float(
                    rng.uniform(*ANOMALY_MULTIPLIER)
                )
                body["transaction_id"] = ANOMALY_PREFIX + uuid.uuid4().hex
            yield body


async def send_one(client: httpx.AsyncClient, body: dict, stats: Stats) -> None:
    stats.sent += 1
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        response = await client.post("/score", json=body)
        response.raise_for_status()
        data = response.json()
        flagged, server_ms = bool(data["flagged"]), float(data["latency_ms"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        stats.errors += 1
        if stats.errors <= 3:
            log.warning("request failed: %s", exc)
        return
    stats.ok += 1
    stats.flagged += flagged
    stats.rtt_ms_total += (loop.time() - started) * 1000
    stats.server_ms_total += server_ms


async def report_periodically(stats: Stats) -> None:
    while True:
        await asyncio.sleep(STATS_EVERY_S)
        log.info(stats.line())


async def run(
    cfg: SimConfig,
    client: httpx.AsyncClient | None = None,
    replay: pd.DataFrame | None = None,
) -> Stats:
    """Send at a steady rate (fixed schedule, so slow responses do not lower it), plus bursts."""
    cfg.validate()
    replay = load_replay() if replay is None else replay
    stream = payload_stream(replay, cfg.anomaly_rate, np.random.default_rng(cfg.seed))
    stats = Stats()
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=cfg.url,
            timeout=10.0,
            limits=httpx.Limits(max_connections=MAX_CONNECTIONS),
        )

    loop = asyncio.get_running_loop()
    in_flight: set[asyncio.Task] = set()

    def spawn(n: int) -> None:
        for _ in range(n):
            task = asyncio.create_task(send_one(client, next(stream), stats))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

    reporter = asyncio.create_task(report_periodically(stats))
    planned = cfg.planned_requests
    interval = 1.0 / cfg.rate
    start = loop.time()
    next_burst = cfg.burst_every if cfg.burst_every and cfg.burst_size else None
    i = 0
    try:
        while planned is None or i < planned:
            await asyncio.sleep(max(0.0, start + i * interval - loop.time()))
            spawn(1)
            i += 1
            if next_burst is not None and loop.time() - start >= next_burst:
                log.info("burst: %d requests", cfg.burst_size)
                spawn(cfg.burst_size)
                next_burst += cfg.burst_every
        await asyncio.gather(*in_flight)
    finally:
        reporter.cancel()
        for task in list(in_flight):
            task.cancel()
        if owns_client:
            await client.aclose()
    log.info("done: %s", stats.line())
    return stats


def parse_args(argv: list[str] | None = None) -> SimConfig:
    p = argparse.ArgumentParser(
        description="Stream replay transactions to the scoring API."
    )
    p.add_argument("--url", default=SimConfig.url)
    p.add_argument(
        "--rate", type=float, default=SimConfig.rate, help="requests per second"
    )
    p.add_argument("--duration", type=float, help="seconds to run")
    p.add_argument("--count", type=int, help="number of base-rate requests to send")
    p.add_argument(
        "--burst-every",
        type=float,
        default=SimConfig.burst_every,
        help="seconds; 0 disables",
    )
    p.add_argument("--burst-size", type=int, default=SimConfig.burst_size)
    p.add_argument("--anomaly-rate", type=float, default=SimConfig.anomaly_rate)
    p.add_argument("--seed", type=int, default=SimConfig.seed)
    a = p.parse_args(argv)
    cfg = SimConfig(
        url=a.url,
        rate=a.rate,
        duration=a.duration,
        count=a.count,
        burst_every=a.burst_every,
        burst_size=a.burst_size,
        anomaly_rate=a.anomaly_rate,
        seed=a.seed,
    )
    try:
        cfg.validate()
    except ValueError as exc:
        p.error(str(exc))
    return cfg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg = parse_args(argv)
    try:
        stats = asyncio.run(run(cfg))
    except KeyboardInterrupt:
        return 130
    return 1 if stats.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
