"""Plotly figure builders. Pure functions: data in, figure out.

Palette: reference data-viz palette slots 1-3 (blue, orange, aqua). Backgrounds are transparent so
Streamlit's light/dark theme shows through; grid lines are hairline and recessive; lines are 2 px.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BLUE = "#2a78d6"  # series 1
ORANGE = "#eb6834"  # series 2
AQUA = "#1baf7a"  # series 3
GRID = "rgba(137, 135, 129, 0.25)"
HISTOGRAM_BIN = 0.02
HEIGHT = 300


def flag_rate_series(recent: pd.DataFrame, bucket: str = "10s") -> pd.DataFrame:
    """Flag rate and request count per time bucket from (scored_at, flagged) rows."""
    if recent.empty:
        return pd.DataFrame({"time": [], "flag_rate": [], "requests": []})
    grouped = recent.set_index("scored_at")["flagged"].astype(float).resample(bucket)
    series = pd.DataFrame({"flag_rate": grouped.mean(), "requests": grouped.count()})
    series = series[
        series["requests"] > 0
    ]  # drop idle buckets rather than draw them as 0 %
    return series.rename_axis("time").reset_index()


def _base_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        height=HEIGHT,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def flag_rate_figure(series: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series["time"],
            y=series["flag_rate"],
            mode="lines",
            line=dict(color=BLUE, width=2),
            customdata=series["requests"],
            hovertemplate="%{y:.2%} flagged of %{customdata} requests<extra></extra>",
        )
    )
    fig.update_yaxes(tickformat=".1%", rangemode="tozero", title_text="Flag rate")
    fig.update_xaxes(title_text="Time (UTC), 10 s buckets")
    return _base_layout(fig)


def score_histogram_figure(
    probabilities: pd.Series, threshold: float | None
) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=probabilities,
            xbins=dict(start=0.0, end=1.0, size=HISTOGRAM_BIN),
            marker=dict(color=BLUE),
            hovertemplate="score %{x}: %{y} transactions<extra></extra>",
        )
    )
    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line=dict(color=ORANGE, width=2),
            annotation_text=f"threshold {threshold:.2f}",
            annotation_position="top",
        )
    fig.update_yaxes(type="log", title_text="Transactions (log scale)")
    fig.update_xaxes(range=[0, 1], title_text="Fraud probability")
    fig.update_layout(bargap=0.15)
    _base_layout(fig)
    fig.update_layout(
        margin=dict(l=8, r=8, t=32, b=8)
    )  # room for the threshold label above the plot
    return fig


def nearest_point(curve: list[dict], threshold: float) -> dict:
    return min(curve, key=lambda point: abs(point["threshold"] - threshold))


def pr_curve_figure(
    curve: list[dict], current: float | None, what_if: float | None
) -> go.Figure:
    """Validation PR curve with a marker for the live threshold and one for the slider position."""
    fig = go.Figure(
        go.Scatter(
            x=[p["recall"] for p in curve],
            y=[p["precision"] for p in curve],
            mode="lines",
            name="Validation PR curve",
            line=dict(color=BLUE, width=2),
            hovertemplate="recall %{x:.2f}, precision %{y:.2f}<extra></extra>",
        )
    )
    markers = (
        (current, "Current threshold", ORANGE, "circle"),
        (what_if, "What-if threshold", AQUA, "diamond"),
    )
    for value, name, color, symbol in markers:
        if value is None or not curve:
            continue
        point = nearest_point(curve, value)
        fig.add_trace(
            go.Scatter(
                x=[point["recall"]],
                y=[point["precision"]],
                mode="markers+text",
                name=name,
                text=[f"{value:.2f}"],
                textposition="top right" if symbol == "circle" else "bottom left",
                marker=dict(
                    color=color,
                    size=12,
                    symbol=symbol,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                hovertemplate=f"{name} {value:.2f}<extra></extra>",
            )
        )
    fig.update_xaxes(title_text="Recall", range=[0, 1.02])
    fig.update_yaxes(title_text="Precision", range=[0, 1.02])
    _base_layout(fig)
    fig.update_layout(
        showlegend=True, legend=dict(orientation="h", y=-0.25), hovermode="closest"
    )
    return fig
