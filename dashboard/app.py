"""Live fraud dashboard. Reads SQLite read-only; changes the threshold only via PUT /threshold.

streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)  # so `python`/streamlit find the repo packages

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client, charts, data, whatif  # noqa: E402
from pipeline.config import ARTIFACTS, C_FP, DB_PATH  # noqa: E402

REFRESH_S = 2
SLIDER_KEY = "whatif_threshold"
APPLY_RESULT_KEY = "apply_result"


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
    cols[4].metric(
        "Caught fraud $",
        f"${kpis.caught_fraud_dollars:,.0f}",
        help="flagged and true_label = 1",
    )
    cols[5].metric(
        "Missed fraud $",
        f"${kpis.missed_fraud_dollars:,.0f}",
        help="not flagged and true_label = 1",
    )


def resolve_threshold(settings: Settings, conn) -> float | None:
    """Current threshold from the API; falls back to the last stored row if the API is unreachable."""
    state = api_client.get_threshold(settings.api_url)
    return state.threshold if state is not None else data.latest_threshold(conn)


def render_charts(conn, now: datetime, threshold: float | None) -> None:
    recent = data.load_recent_scores(conn, now)
    left, right = st.columns(2)
    with left:
        st.subheader("Flag rate over time")
        if recent.empty:
            st.caption("No transactions in the last 10 minutes.")
        else:
            st.plotly_chart(
                charts.flag_rate_figure(charts.flag_rate_series(recent)),
                use_container_width=True,
            )
    with right:
        st.subheader("Score distribution")
        if recent.empty:
            st.caption("No transactions in the last 10 minutes.")
        else:
            st.plotly_chart(
                charts.score_histogram_figure(recent["fraud_probability"], threshold),
                use_container_width=True,
            )


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
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud probability", min_value=0.0, max_value=1.0
            ),
            "true_label": st.column_config.NumberColumn("True label"),
            "top_features": st.column_config.TextColumn(
                "Top SHAP features", width="large"
            ),
        },
    )


def apply_threshold(api_url: str) -> None:
    """Button callback: the dashboard's only write. Goes through PUT /threshold, never a file or the DB."""
    value = round(float(st.session_state[SLIDER_KEY]), 2)
    try:
        state = api_client.put_threshold(api_url, value)
    except (httpx.HTTPError, ValueError) as exc:
        st.session_state[APPLY_RESULT_KEY] = (
            "error",
            f"Could not apply threshold {value:.2f}: {exc}",
        )
        return
    st.session_state[APPLY_RESULT_KEY] = (
        "success",
        f"Threshold is now {state.threshold:.2f} ({state.source}); new transactions use it immediately.",
    )


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_what_if(labeled, slider_value: float, current: float | None) -> None:
    """Recalculated from stored scores and labels only: no API call, no re-scoring."""
    if labeled.empty:
        st.caption(
            "No labeled transactions stored yet; run the simulator to see what-if results."
        )
        return
    chosen = whatif.what_if(labeled, slider_value, C_FP)
    base = whatif.what_if(labeled, current, C_FP) if current is not None else None
    cols = st.columns(3)
    cols[0].metric("Precision", pct(chosen.precision))
    cols[1].metric("Recall", pct(chosen.recall))
    change = None if base is None else chosen.cost - base.cost
    delta = None if change is None else f"{change:+,.0f} vs current"
    # lower cost is better, so colour inversely; no colour when nothing changes
    cols[2].metric(
        "Cost",
        f"${chosen.cost:,.0f}",
        delta=delta,
        delta_color="off" if change == 0 else "inverse",
    )
    st.caption(
        f"{chosen.labeled:,} labeled transactions, {chosen.frauds} fraud. "
        f"TP {chosen.tp}, FP {chosen.fp}, FN {chosen.fn}. Cost = ${C_FP:g} per false alarm + amount of missed fraud."
    )
    if chosen.frauds == 0:
        st.caption(
            "No fraud in the stored sample yet, so recall is undefined and cost counts only false alarms."
        )


def render_threshold_panel(settings: Settings, conn, current: float | None) -> None:
    st.subheader("Threshold")
    # Start at the live threshold. Passing it as the widget default (rather than presetting
    # session_state) is what a real browser session honours; the slider resets only when the
    # live threshold itself changes, e.g. right after Apply.
    default = min(max(round(current if current is not None else 0.5, 2), 0.01), 0.99)
    left, right = st.columns(2)
    with left:
        value = st.slider(
            "What-if threshold", 0.01, 0.99, value=default, step=0.01, key=SLIDER_KEY
        )
        render_what_if(data.load_labeled_scores(conn), value, current)
        st.button(
            "Apply threshold",
            key="apply_threshold",
            on_click=apply_threshold,
            args=(settings.api_url,),
        )
        outcome = st.session_state.get(APPLY_RESULT_KEY)
        if outcome:
            (st.success if outcome[0] == "success" else st.error)(outcome[1])
    with right:
        curve = data.load_pr_curve(settings.artifacts_dir)
        if curve:
            st.plotly_chart(
                charts.pr_curve_figure(curve, current, value), use_container_width=True
            )
        else:
            st.caption("No PR curve found; run `make evaluate`.")


@st.fragment(run_every=REFRESH_S)
def live_view() -> None:
    settings = load_settings()
    conn = data.open_readonly(settings.db_path)
    if conn is None:
        st.info(f"Waiting for the API to create {settings.db_path} ...")
        return
    try:
        now = datetime.now(timezone.utc)
        threshold = resolve_threshold(settings, conn)
        render_kpis(data.load_kpis(conn, now))
        render_charts(conn, now, threshold)
        render_feed(conn)
        render_threshold_panel(settings, conn, threshold)
    finally:
        conn.close()


st.set_page_config(page_title="Fraud monitor", layout="wide")
st.title("Fraud monitor")
live_view()
