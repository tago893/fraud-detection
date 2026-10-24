"""Live fraud dashboard. Reads SQLite read-only; changes the threshold only via PUT /threshold.

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python`/streamlit find the repo packages

import streamlit as st  # noqa: E402

from dashboard import data  # noqa: E402
from pipeline.config import ARTIFACTS, DB_PATH  # noqa: E402

REFRESH_S = 2


@dataclass(frozen=True)
class Settings:
    db_path: Path
    api_url: str
    artifacts_dir: Path


def load_settings() -> Settings:
    return Settings(
        db_path=Path(os.environ.get("DASHBOARD_DB_PATH", DB_PATH)),
        api_url=os.environ.get("API_URL", "http://localhost:8000").rstrip("/"),
        artifacts_dir=Path(os.environ.get("DASHBOARD_ARTIFACTS_DIR", ARTIFACTS)),
    )


def render_kpis(kpis: data.Kpis) -> None:
    cols = st.columns(6)
    cols[0].metric("Requests / s (last 60 s)", f"{kpis.requests_per_s:.1f}")
    p95 = "n/a" if kpis.p95_latency_ms is None else f"{kpis.p95_latency_ms:.1f} ms"
    cols[1].metric("p95 latency (last 60 s)", p95)
    cols[2].metric("Flagged", f"{kpis.flagged:,}", help=f"of {kpis.total:,} scored")
    cols[3].metric("Flag rate", f"{kpis.flag_rate:.2%}")
    cols[4].metric("Caught fraud $", f"${kpis.caught_fraud_dollars:,.0f}", help="flagged and true_label = 1")
    cols[5].metric("Missed fraud $", f"${kpis.missed_fraud_dollars:,.0f}", help="not flagged and true_label = 1")


def render_feed(conn) -> None:
    st.subheader("Latest flagged transactions")
    feed = data.load_flagged_feed(conn)
    if feed.empty:
        st.caption("No flagged transactions yet.")
        return
    st.dataframe(
        feed,
        hide_index=True,
        use_container_width=True,
        column_config={
            "scored_at": st.column_config.TextColumn("Scored at (UTC)"),
            "transaction_id": st.column_config.TextColumn("Transaction"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "fraud_probability": st.column_config.ProgressColumn("Fraud probability", min_value=0.0, max_value=1.0),
            "true_label": st.column_config.NumberColumn("True label"),
            "top_features": st.column_config.TextColumn("Top SHAP features", width="large"),
        },
    )


@st.fragment(run_every=REFRESH_S)
def live_view() -> None:
    settings = load_settings()
    conn = data.open_readonly(settings.db_path)
    if conn is None:
        st.info(f"Waiting for the API to create {settings.db_path} ...")
        return
    try:
        now = datetime.now(timezone.utc)
        render_kpis(data.load_kpis(conn, now))
        render_feed(conn)
    finally:
        conn.close()


st.set_page_config(page_title="Fraud monitor", layout="wide")
st.title("Fraud monitor")
live_view()
