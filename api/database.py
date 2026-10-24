"""SQLite storage for scored transactions. WAL mode; the API shares one persistent connection."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
  transaction_id TEXT PRIMARY KEY,
  scored_at TEXT NOT NULL,
  amount REAL NOT NULL,
  fraud_probability REAL NOT NULL,
  flagged INTEGER NOT NULL,
  threshold REAL NOT NULL,
  latency_ms REAL NOT NULL,
  true_label INTEGER,
  shap_top TEXT
);
CREATE INDEX IF NOT EXISTS idx_tx_scored_at ON transactions(scored_at);
CREATE TABLE IF NOT EXISTS threshold_history (changed_at TEXT, old REAL, new REAL);
"""


@dataclass(frozen=True)
class TransactionRecord:
    transaction_id: str
    scored_at: str
    amount: float
    fraud_probability: float
    flagged: bool
    threshold: float
    latency_ms: float
    true_label: int | None


# Persistent connections for a running API, keyed by database path. A fresh connection per call
# is cheap on a local disk but costly on a Docker bind mount (about 35 ms per insert there):
# in WAL mode every open/close creates and removes the -wal/-shm files. One shared connection,
# serialized by a lock, removes that churn.
_shared: dict[Path, tuple[sqlite3.Connection, threading.RLock]] = {}


def _open(path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5.0, check_same_thread=check_same_thread)
    conn.execute("PRAGMA synchronous=NORMAL")  # per connection, unlike journal_mode
    return conn


def open_shared(path: Path) -> None:
    """Keep one connection open for `path` (idempotent). Call close_shared on shutdown."""
    if path not in _shared:
        _shared[path] = (_open(path, check_same_thread=False), threading.RLock())


def close_shared(path: Path) -> None:
    shared = _shared.pop(path, None)
    if shared is not None:
        shared[0].close()


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on error. Uses the shared connection if one is open for
    `path` (held under its lock for the whole block); otherwise opens and closes its own."""
    shared = _shared.get(path)
    if shared is not None:
        conn, lock = shared
        with lock, conn:
            yield conn
        return
    with closing(_open(path)) as conn, conn:
        yield conn


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")  # persistent in the db file
        conn.executescript(SCHEMA)


def journal_mode(path: Path) -> str:
    with connect(path) as conn:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]


def insert_transaction(path: Path, rec: TransactionRecord) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO transactions (transaction_id, scored_at, amount, fraud_probability, flagged,"
            " threshold, latency_ms, true_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.transaction_id,
                rec.scored_at,
                rec.amount,
                rec.fraud_probability,
                int(rec.flagged),
                rec.threshold,
                rec.latency_ms,
                rec.true_label,
            ),
        )


def count_transactions(path: Path) -> int:
    with connect(path) as conn:
        return conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]


def insert_threshold_change(
    path: Path, changed_at: str, old: float, new: float
) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT INTO threshold_history (changed_at, old, new) VALUES (?, ?, ?)",
            (changed_at, old, new),
        )


def threshold_history(path: Path) -> list[tuple[str, float, float]]:
    with connect(path) as conn:
        return conn.execute(
            "SELECT changed_at, old, new FROM threshold_history ORDER BY rowid"
        ).fetchall()


def set_shap_top(path: Path, transaction_id: str, contributions: list[dict]) -> None:
    with connect(path) as conn:
        conn.execute(
            "UPDATE transactions SET shap_top = ? WHERE transaction_id = ?",
            (json.dumps(contributions), transaction_id),
        )


def get_shap_top(path: Path, transaction_id: str) -> list[dict] | None:
    """Top SHAP contributions, or None if the transaction is unknown or has none stored."""
    with connect(path) as conn:
        row = conn.execute(
            "SELECT shap_top FROM transactions WHERE transaction_id = ?",
            (transaction_id,),
        ).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def window_summary(path: Path, since_iso: str) -> tuple[int, int, float | None]:
    """(count, flagged count, nearest-rank p95 latency) since since_iso, aggregated in SQL.

    Only one latency value is read back, so a long window does not pull every row into Python.
    """
    with connect(path) as conn:
        count, flagged = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(flagged), 0) FROM transactions WHERE scored_at >= ?",
            (since_iso,),
        ).fetchone()
        if count == 0:
            return 0, 0, None
        rank = -(-95 * count // 100) - 1  # ceil(0.95 * count) - 1
        (p95,) = conn.execute(
            "SELECT latency_ms FROM transactions WHERE scored_at >= ? "
            "ORDER BY latency_ms LIMIT 1 OFFSET ?",
            (since_iso, rank),
        ).fetchone()
    return count, flagged, float(p95)
