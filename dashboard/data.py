"""Read-only access to the transactions database. The dashboard never writes to SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

KPI_WINDOW_S = 60
FEED_SIZE = 50
CHART_WINDOW_MIN = 10
MAX_LABELED_ROWS = 100_000


def iso_utc(moment: datetime) -> str:
    """Same fixed-width format the API stores, so string comparison orders by time."""
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def open_readonly(path: Path) -> sqlite3.Connection | None:
    """A read-only connection, or None while the API has not created the database yet."""
    if not path.exists():
        return None
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


@dataclass(frozen=True)
class Kpis:
    total: int
    flagged: int
    flag_rate: float
    requests_per_s: float
    p95_latency_ms: float | None
    caught_fraud_dollars: float
    missed_fraud_dollars: float


def load_kpis(conn: sqlite3.Connection, now: datetime) -> Kpis:
    """Totals are over everything stored; throughput and p95 cover the last 60 seconds."""
    total, flagged, caught, missed, first_at = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(flagged), 0),"
        " COALESCE(SUM(CASE WHEN flagged = 1 AND true_label = 1 THEN amount END), 0),"
        " COALESCE(SUM(CASE WHEN flagged = 0 AND true_label = 1 THEN amount END), 0),"
        " MIN(scored_at) FROM transactions"
    ).fetchone()
    since = iso_utc(now - timedelta(seconds=KPI_WINDOW_S))
    recent = conn.execute("SELECT COUNT(*) FROM transactions WHERE scored_at >= ?", (since,)).fetchone()[0]
    p95 = None
    if recent:
        rank = -(-95 * recent // 100) - 1  # nearest-rank: ceil(0.95 * n) - 1
        p95 = float(
            conn.execute(
                "SELECT latency_ms FROM transactions WHERE scored_at >= ? ORDER BY latency_ms LIMIT 1 OFFSET ?",
                (since, rank),
            ).fetchone()[0]
        )
    # Do not divide by a full 60 s until the stream has actually run that long.
    span_s = KPI_WINDOW_S
    if first_at is not None:
        age = (now - datetime.fromisoformat(first_at)).total_seconds()
        span_s = min(KPI_WINDOW_S, max(age, 1.0))
    return Kpis(
        total=total,
        flagged=flagged,
        flag_rate=flagged / total if total else 0.0,
        requests_per_s=recent / span_s,
        p95_latency_ms=p95,
        caught_fraud_dollars=float(caught),
        missed_fraud_dollars=float(missed),
    )


def format_shap_top(raw: str | None) -> str:
    """'V14 -2.10, V4 +1.30, ...' from the stored JSON, or an empty string while SHAP is pending."""
    if not raw:
        return ""
    return ", ".join(f"{c['feature']} {c['shap_value']:+.2f}" for c in json.loads(raw))


def load_flagged_feed(conn: sqlite3.Connection, limit: int = FEED_SIZE) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT scored_at, transaction_id, amount, fraud_probability, true_label, shap_top"
        " FROM transactions WHERE flagged = 1 ORDER BY scored_at DESC LIMIT ?",
        conn,
        params=(limit,),
    )
    frame["top_features"] = frame.pop("shap_top").map(format_shap_top)
    return frame


def load_recent_scores(conn: sqlite3.Connection, now: datetime, minutes: int = CHART_WINDOW_MIN) -> pd.DataFrame:
    since = iso_utc(now - timedelta(minutes=minutes))
    frame = pd.read_sql_query(
        "SELECT scored_at, fraud_probability, flagged FROM transactions WHERE scored_at >= ? ORDER BY scored_at",
        conn,
        params=(since,),
    )
    frame["scored_at"] = pd.to_datetime(frame["scored_at"], utc=True, format="ISO8601")
    return frame


def load_labeled_scores(conn: sqlite3.Connection) -> pd.DataFrame:
    """Stored (probability, true label, raw amount) for every scored transaction that has a label."""
    return pd.read_sql_query(
        "SELECT fraud_probability, true_label, amount FROM transactions"
        " WHERE true_label IS NOT NULL ORDER BY scored_at DESC LIMIT ?",
        conn,
        params=(MAX_LABELED_ROWS,),
    )


def latest_threshold(conn: sqlite3.Connection) -> float | None:
    """Threshold used for the most recent row; a fallback when the API cannot be reached."""
    row = conn.execute("SELECT threshold FROM transactions ORDER BY scored_at DESC LIMIT 1").fetchone()
    return float(row[0]) if row else None
