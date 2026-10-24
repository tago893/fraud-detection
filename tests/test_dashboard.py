import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from api import database as db
from dashboard import data

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
NOW = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)


def add(path, i, *, ago_s, flagged=False, label=None, amount=10.0, prob=0.1, latency=1.0, threshold=0.5):
    db.insert_transaction(
        path,
        db.TransactionRecord(
            f"tx{i}", data.iso_utc(NOW - timedelta(seconds=ago_s)), amount, prob, flagged, threshold, latency, label
        ),
    )


@pytest.fixture
def populated(tmp_path):
    path = tmp_path / "tx.db"
    db.init_db(path)
    # 10 rows in the last minute (latency 1..10 ms), one older row; two flagged frauds, one missed fraud
    for i in range(10):
        add(path, i, ago_s=5 + i, latency=float(i + 1))
    add(path, 100, ago_s=2, flagged=True, label=1, amount=200.0, prob=0.95)
    add(path, 101, ago_s=3, flagged=True, label=0, amount=30.0, prob=0.9)
    add(path, 102, ago_s=4, flagged=False, label=1, amount=75.0, prob=0.3)
    add(path, 103, ago_s=600, flagged=True, label=1, amount=10.0, prob=0.99)
    return path


# ---- read-only access ---------------------------------------------------------------------


def test_connection_is_read_only(populated):
    conn = data.open_readonly(populated)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO transactions VALUES ('x','t',1,0.1,0,0.5,1,NULL,NULL)")
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM transactions")
    conn.close()


def test_missing_database_returns_none(tmp_path):
    assert data.open_readonly(tmp_path / "nope.db") is None


def test_dashboard_source_only_reads_sqlite_and_only_writes_via_the_api():
    root = Path(APP).parent
    sources = {p.name: p.read_text() for p in root.glob("*.py")}
    for name, text in sources.items():
        assert not re.search(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE)\b", text), name
        assert "threshold.json" not in text.replace("DASHBOARD", ""), name
        assert "open(" not in text and "write_text" not in text and "to_parquet" not in text, name
    assert "put_threshold" in sources["api_client.py"] and "httpx.put" in sources["api_client.py"]


# ---- 4.1 KPIs and feed --------------------------------------------------------------------


def test_kpis(populated):
    conn = data.open_readonly(populated)
    k = data.load_kpis(conn, NOW)
    assert (k.total, k.flagged) == (14, 3)
    assert k.flag_rate == pytest.approx(3 / 14)
    assert k.caught_fraud_dollars == 210.0  # tx100 (200) + tx103 (10): flagged and true_label = 1
    assert k.missed_fraud_dollars == 75.0  # tx102: true_label = 1, not flagged
    # 13 rows in the last 60 s; the stream is older than 60 s, so the divisor is the full window
    assert k.requests_per_s == pytest.approx(13 / 60)
    assert k.p95_latency_ms == 10.0  # nearest rank over {1..10, 1, 1, 1}
    conn.close()


def test_kpis_on_empty_database(tmp_path):
    path = tmp_path / "tx.db"
    db.init_db(path)
    conn = data.open_readonly(path)
    k = data.load_kpis(conn, NOW)
    assert (k.total, k.flagged, k.requests_per_s, k.p95_latency_ms) == (0, 0, 0.0, None)


def test_flagged_feed_is_newest_first_and_limited(populated):
    conn = data.open_readonly(populated)
    feed = data.load_flagged_feed(conn, limit=2)
    assert feed["transaction_id"].tolist() == ["tx100", "tx101"]
    assert set(feed.columns) >= {"scored_at", "amount", "fraud_probability", "top_features"}
    conn.close()


def test_feed_shows_top_shap_features(populated):
    db.set_shap_top(
        populated, "tx100", [{"feature": "V14", "shap_value": -2.1, "feature_value": 1.0}, {"feature": "V4", "shap_value": 1.3, "feature_value": 0.2}]
    )
    conn = data.open_readonly(populated)
    row = data.load_flagged_feed(conn).set_index("transaction_id").loc["tx100"]
    assert row["top_features"] == "V14 -2.10, V4 +1.30"
    conn.close()


def at_for(populated, monkeypatch):
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(populated))
    return AppTest.from_file(APP, default_timeout=30)


def test_app_renders_kpis_and_feed(populated, monkeypatch):
    app = at_for(populated, monkeypatch).run()
    assert not app.exception
    labels = {m.label for m in app.metric}
    assert {"Requests / s (last 60 s)", "p95 latency (last 60 s)", "Flagged", "Flag rate", "Caught fraud $", "Missed fraud $"} <= labels
    assert any(d.value.shape[0] == 3 for d in app.dataframe)  # 3 flagged rows in the feed


def test_app_waits_when_database_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(tmp_path / "missing.db"))
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert any("Waiting for the API" in i.value for i in app.info)


# ---- 4.2 charts ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from dashboard import charts  # noqa: E402


def frame(rows):
    df = pd.DataFrame(rows, columns=["scored_at", "fraud_probability", "flagged"])
    df["scored_at"] = pd.to_datetime(df["scored_at"], utc=True)
    return df


def test_flag_rate_series_buckets_and_skips_idle_gaps():
    rows = [("2026-10-24T12:00:01Z", 0.1, 0), ("2026-10-24T12:00:05Z", 0.9, 1),  # bucket :00 -> 50 %
            ("2026-10-24T12:00:12Z", 0.2, 0),                                    # bucket :10 -> 0 %
            ("2026-10-24T12:00:45Z", 0.95, 1)]                                   # bucket :40 -> 100 %
    series = charts.flag_rate_series(frame(rows))
    assert series["flag_rate"].tolist() == [0.5, 0.0, 1.0]
    assert series["requests"].tolist() == [2, 1, 1]  # the empty :20 and :30 buckets are dropped


def test_flag_rate_series_empty():
    assert charts.flag_rate_series(frame([])).empty


def test_histogram_has_threshold_line_at_the_current_value():
    fig = charts.score_histogram_figure(pd.Series([0.01, 0.02, 0.9]), 0.85)
    assert [s.x0 for s in fig.layout.shapes] == [0.85]
    assert fig.layout.yaxis.type == "log" and list(fig.layout.xaxis.range) == [0, 1]
    assert len(fig.data) == 1


def test_histogram_without_threshold_has_no_line():
    assert not charts.score_histogram_figure(pd.Series([0.1]), None).layout.shapes


def live_db(tmp_path):
    """Rows scored in the last few seconds of real time (the app uses the wall clock)."""
    path = tmp_path / "live.db"
    db.init_db(path)
    real_now = datetime.now(timezone.utc)
    for i in range(30):
        db.insert_transaction(
            path,
            db.TransactionRecord(
                f"live{i}", data.iso_utc(real_now - timedelta(seconds=i)), 20.0, 0.9 if i % 10 == 0 else 0.05,
                i % 10 == 0, 0.85, 1.5, 1 if i % 10 == 0 else 0,
            ),
        )
    return path


def test_app_draws_both_charts_with_the_threshold_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_DB_PATH", str(live_db(tmp_path)))
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # unreachable: threshold falls back to the stored one
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2


def test_app_says_so_when_there_is_no_recent_data(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    db.init_db(path)
    db.insert_transaction(
        path, db.TransactionRecord("old1", "2020-01-01T00:00:00.000+00:00", 5.0, 0.1, False, 0.85, 1.0, None)
    )
    app = at_for(path, monkeypatch).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 0
    assert any("last 10 minutes" in c.value for c in app.caption)
