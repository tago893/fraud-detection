"""SQLite storage for scored transactions. WAL mode, one short-lived connection per call."""
from __future__ import annotations

import json
import sqlite3
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


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, always close. synchronous is per connection."""
    with closing(sqlite3.connect(path, timeout=5.0)) as conn:
        conn.execute("PRAGMA synchronous=NORMAL")
        with conn:
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


def insert_threshold_change(path: Path, changed_at: str, old: float, new: float) -> None:
    with connect(path) as conn:
        conn.execute("INSERT INTO threshold_history (changed_at, old, new) VALUES (?, ?, ?)", (changed_at, old, new))


def threshold_history(path: Path) -> list[tuple[str, float, float]]:
    with connect(path) as conn:
        return conn.execute("SELECT changed_at, old, new FROM threshold_history ORDER BY rowid").fetchall()


def set_shap_top(path: Path, transaction_id: str, contributions: list[dict]) -> None:
    with connect(path) as conn:
        conn.execute(
            "UPDATE transactions SET shap_top = ? WHERE transaction_id = ?",
            (json.dumps(contributions), transaction_id),
        )


def get_shap_top(path: Path, transaction_id: str) -> list[dict] | None:
    """Top SHAP contributions, or None if the transaction is unknown or has none stored."""
    with connect(path) as conn:
        row = conn.execute("SELECT shap_top FROM transactions WHERE transaction_id = ?", (transaction_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def window_stats(path: Path, since_iso: str) -> tuple[int, int, list[float]]:
    """(count, flagged count, latencies) for transactions scored at or after since_iso."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT flagged, latency_ms FROM transactions WHERE scored_at >= ?", (since_iso,)
        ).fetchall()
    return len(rows), sum(r[0] for r in rows), [r[1] for r in rows]
