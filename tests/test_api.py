from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import database as db
from api.main import create_app
from api.scoring import Scorer
from pipeline.config import ARTIFACTS, FEATURE_ORDER, TARGET

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "threshold.json").exists(), reason="run make train evaluate first"
)


def row_payload(row: pd.Series, **extra) -> dict:
    return {**{name: float(row[name]) for name in FEATURE_ORDER}, **extra}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(db_path=tmp_path_factory.mktemp("db") / "tx.db")) as c:
        yield c


@pytest.fixture
def fresh_client(tmp_path):
    """App with an empty database, for tests that count rows."""
    with TestClient(create_app(db_path=tmp_path / "tx.db")) as c:
        yield c


@pytest.fixture(scope="module")
def test_rows(splits):
    return splits["test"]


@pytest.fixture(scope="module")
def known_fraud(test_rows, client):
    """The test-slice fraud row the model is most sure about (smoke test only; nothing is tuned on it)."""
    frauds = test_rows[test_rows[TARGET] == 1]
    scored = [
        (client.post("/score", json=row_payload(r)).json()["fraud_probability"], i)
        for i, r in frauds.iterrows()
    ]
    return test_rows.loc[max(scored)[1]]


def test_health(client):
    body = client.get("/health").json()
    assert (
        body["status"] == "ok" and 0 < body["threshold"] < 1 and body["model_version"]
    )


def test_known_fraud_row_scores_above_threshold(client, known_fraud):
    body = client.post("/score", json=row_payload(known_fraud)).json()
    assert body["fraud_probability"] > body["decision_threshold"]
    assert body["flagged"] is True


def test_legit_row_not_flagged(client, test_rows):
    legit = test_rows[test_rows[TARGET] == 0].iloc[0]
    body = client.post("/score", json=row_payload(legit)).json()
    assert body["flagged"] is False and body["latency_ms"] >= 0


def test_transaction_id_echoed_or_generated(client, test_rows):
    row = test_rows.iloc[0]
    assert (
        client.post("/score", json=row_payload(row, transaction_id="tx-1")).json()[
            "transaction_id"
        ]
        == "tx-1"
    )
    assert (
        len(client.post("/score", json=row_payload(row)).json()["transaction_id"]) == 36
    )


@pytest.mark.parametrize(
    "mutate",
    [lambda b: b.pop("V7"), lambda b: b.update(Amount=-1), lambda b: b.update(x=1)],
)
def test_invalid_requests_are_422(client, test_rows, mutate):
    body = row_payload(test_rows.iloc[0])
    mutate(body)
    assert client.post("/score", json=body).status_code == 422


def test_scorer_matches_sklearn_scaler(client, test_rows, scaler):
    scorer: Scorer = client.app.state.scorer
    sample = test_rows.iloc[:50][list(scorer.feature_order)]
    got = np.vstack(
        [scorer.scale_row(sample.iloc[i].to_numpy(float)) for i in range(len(sample))]
    )
    np.testing.assert_allclose(got, scaler.transform(sample))


def test_true_label_never_reaches_model(client, test_rows):
    seen: list[np.ndarray] = []

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def predict_proba(self, x):
            seen.append(np.array(x))
            return self.inner.predict_proba(x)

    real = client.app.state.scorer
    client.app.state.scorer = replace(real, model=Spy(real.model))
    try:
        row = test_rows[test_rows[TARGET] == 1].iloc[0]
        probs = [
            client.post("/score", json=row_payload(row, true_label=label)).json()[
                "fraud_probability"
            ]
            for label in (None, 0, 1)
        ]
    finally:
        client.app.state.scorer = real
    assert len(set(probs)) == 1
    assert len(seen) == 3 and all(x.shape == (1, len(FEATURE_ORDER)) for x in seen)
    np.testing.assert_array_equal(seen[0], seen[1])
    np.testing.assert_array_equal(seen[1], seen[2])


def test_100_calls_store_100_rows_in_wal_mode(fresh_client, test_rows):
    path = fresh_client.app.state.db_path
    for _, row in test_rows.head(100).iterrows():
        assert fresh_client.post("/score", json=row_payload(row)).status_code == 200
    assert db.count_transactions(path) == 100
    assert db.journal_mode(path) == "wal"


def test_stored_row_matches_response_and_keeps_true_label(fresh_client, test_rows):
    row = test_rows[test_rows[TARGET] == 1].iloc[0]
    body = fresh_client.post(
        "/score", json=row_payload(row, transaction_id="tx-a", true_label=1)
    ).json()
    with db.connect(fresh_client.app.state.db_path) as conn:
        stored = conn.execute(
            "SELECT amount, fraud_probability, flagged, threshold, true_label FROM transactions"
        ).fetchone()
    assert stored == (
        float(row["Amount"]),
        body["fraud_probability"],
        int(body["flagged"]),
        body["decision_threshold"],
        1,
    )


def test_insert_runs_after_response_is_sent(tmp_path, test_rows, monkeypatch):
    events: list[str] = []
    real_insert = db.insert_transaction

    def spy_insert(path, rec):
        events.append("insert")
        real_insert(path, rec)

    monkeypatch.setattr(db, "insert_transaction", spy_insert)
    app = create_app(db_path=tmp_path / "tx.db")

    async def traced(scope, receive, send):
        async def traced_send(message):
            if message["type"] == "http.response.body" and not message.get("more_body"):
                events.append("response_sent")
            await send(message)

        await app(scope, receive, traced_send)

    with TestClient(traced) as c:
        c.post("/score", json=row_payload(test_rows.iloc[0]))
    assert events == ["response_sent", "insert"]


def test_duplicate_transaction_id_does_not_break_response(fresh_client, test_rows):
    body = row_payload(test_rows.iloc[0], transaction_id="dup")
    assert fresh_client.post("/score", json=body).status_code == 200
    assert fresh_client.post("/score", json=body).status_code == 200
    assert db.count_transactions(fresh_client.app.state.db_path) == 1


def test_threshold_starts_at_artifact_value(fresh_client):
    body = fresh_client.get("/threshold").json()
    assert body == {
        "threshold": fresh_client.app.state.artifact_threshold,
        "source": "artifact",
    }


def test_put_threshold_changes_flag_and_logs_history(fresh_client, test_rows):
    state = fresh_client.app.state
    # a fraud row the model flags at the artifact threshold but scores below 0.99
    candidates = test_rows[test_rows[TARGET] == 1]
    probs = {
        i: fresh_client.post("/score", json=row_payload(r)).json()["fraud_probability"]
        for i, r in candidates.iterrows()
    }
    idx = next(
        (i for i, p in probs.items() if state.artifact_threshold <= p < 0.99), None
    )
    if idx is None:
        pytest.skip("no test fraud row scores between the artifact threshold and 0.99")
    payload = row_payload(test_rows.loc[idx])

    assert fresh_client.post("/score", json=payload).json()["flagged"] is True
    put = fresh_client.put("/threshold", json={"threshold": 0.99})
    assert put.json() == {"threshold": 0.99, "source": "override"}
    after = fresh_client.post("/score", json=payload).json()
    assert after["flagged"] is False and after["decision_threshold"] == 0.99
    assert fresh_client.get("/threshold").json() == {
        "threshold": 0.99,
        "source": "override",
    }
    history = db.threshold_history(state.db_path)
    assert [(old, new) for _, old, new in history] == [(state.artifact_threshold, 0.99)]


@pytest.mark.parametrize("bad", [1.5, -0.1, "high"])
def test_put_threshold_out_of_range_is_422_and_ignored(fresh_client, bad):
    before = fresh_client.get("/threshold").json()
    assert fresh_client.put("/threshold", json={"threshold": bad}).status_code == 422
    assert fresh_client.get("/threshold").json() == before
    assert db.threshold_history(fresh_client.app.state.db_path) == []


def test_stored_threshold_is_the_one_used_for_that_decision(fresh_client, test_rows):
    row = row_payload(test_rows.iloc[0])
    fresh_client.put("/threshold", json={"threshold": 0.5})
    fresh_client.post("/score", json={**row, "transaction_id": "t1"})
    with db.connect(fresh_client.app.state.db_path) as conn:
        assert (
            conn.execute(
                "SELECT threshold FROM transactions WHERE transaction_id='t1'"
            ).fetchone()[0]
            == 0.5
        )


def test_flagged_row_gets_five_shap_features(fresh_client, known_fraud):
    body = fresh_client.post(
        "/score", json=row_payload(known_fraud, transaction_id="fraud-1")
    ).json()
    assert body["flagged"] is True
    explained = fresh_client.get("/explain/fraud-1")
    assert explained.status_code == 200
    top = explained.json()["top_features"]
    assert len(top) == 5
    assert all(
        set(t) == {"feature", "shap_value", "feature_value"}
        and t["feature"] in FEATURE_ORDER
        for t in top
    )
    magnitudes = [abs(t["shap_value"]) for t in top]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_unknown_or_unflagged_transaction_is_404(fresh_client, test_rows):
    assert fresh_client.get("/explain/nope").status_code == 404
    legit = test_rows[test_rows[TARGET] == 0].iloc[0]
    assert (
        fresh_client.post(
            "/score", json=row_payload(legit, transaction_id="legit-1")
        ).json()["flagged"]
        is False
    )
    assert fresh_client.get("/explain/legit-1").status_code == 404


def test_shap_runs_after_response_is_sent(tmp_path, known_fraud, monkeypatch):
    from api import explain

    events: list[str] = []
    real = explain.top_contributions

    def spy(*args, **kwargs):
        events.append("shap")
        return real(*args, **kwargs)

    monkeypatch.setattr(explain, "top_contributions", spy)
    app = create_app(db_path=tmp_path / "tx.db")

    async def traced(scope, receive, send):
        async def traced_send(message):
            if message["type"] == "http.response.body" and not message.get("more_body"):
                events.append("response_sent")
            await send(message)

        await app(scope, receive, traced_send)

    with TestClient(traced) as c:
        c.post("/score", json=row_payload(known_fraud))
    assert events == ["response_sent", "shap"]


def test_metrics_summary(fresh_client, test_rows):
    assert fresh_client.get("/metrics/summary").json()["count"] == 0
    for _, row in test_rows.head(20).iterrows():
        fresh_client.post("/score", json=row_payload(row))
    body = fresh_client.get("/metrics/summary", params={"minutes": 5}).json()
    assert (
        body["count"] == 20
        and body["p95_latency_ms"] > 0
        and 0 <= body["flag_rate"] <= 1
    )


# ---- fixes from the code review -----------------------------------------------------------


def copy_artifacts(tmp_path):
    import shutil

    dest = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, dest, ignore=shutil.ignore_patterns("splits"))
    return dest


def test_duplicate_id_does_not_overwrite_original_explanation(fresh_client, test_rows):
    frauds = test_rows[test_rows[TARGET] == 1]
    flagged = []
    for _, row in frauds.iterrows():
        body = fresh_client.post("/score", json=row_payload(row)).json()
        if body["flagged"]:
            flagged.append(row)
        if len(flagged) == 2:
            break
    assert len(flagged) == 2
    fresh_client.post("/score", json=row_payload(flagged[0], transaction_id="same"))
    before = fresh_client.get("/explain/same").json()
    retry = fresh_client.post(
        "/score", json=row_payload(flagged[1], transaction_id="same")
    )
    assert retry.status_code == 200
    assert fresh_client.get("/explain/same").json() == before


def test_startup_rejects_reordered_feature_order(tmp_path):
    import json

    art = copy_artifacts(tmp_path)
    meta = json.loads((art / "threshold.json").read_text())
    meta["feature_order"] = list(reversed(meta["feature_order"]))
    (art / "threshold.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="feature_order"):
        with TestClient(create_app(artifacts_dir=art, db_path=tmp_path / "tx.db")):
            pass


def test_startup_rejects_threshold_from_a_different_model(tmp_path):
    import json

    art = copy_artifacts(tmp_path)
    metrics = json.loads((art / "metrics.json").read_text())
    metrics["model_version"] = "2099-01-01T00:00:00Z"
    (art / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="make evaluate"):
        with TestClient(create_app(artifacts_dir=art, db_path=tmp_path / "tx.db")):
            pass


def test_concurrent_threshold_puts_keep_history_chain_intact(fresh_client):
    from concurrent.futures import ThreadPoolExecutor

    values = [round(0.1 + 0.02 * i, 2) for i in range(30)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        codes = list(
            pool.map(
                lambda v: fresh_client.put(
                    "/threshold", json={"threshold": v}
                ).status_code,
                values,
            )
        )
    assert set(codes) == {200}
    history = db.threshold_history(fresh_client.app.state.db_path)
    assert len(history) == 30
    for (_, _, new), (_, old, _) in zip(history, history[1:]):
        assert old == new  # each change starts from where the previous one ended
    assert fresh_client.app.state.threshold == history[-1][2]


def test_window_summary_nearest_rank_p95(tmp_path):
    path = tmp_path / "tx.db"
    db.init_db(path)
    for i in range(1, 21):  # latencies 1..20 ms, every 4th flagged
        db.insert_transaction(
            path,
            db.TransactionRecord(
                f"id{i}",
                "2026-01-01T00:00:00.000+00:00",
                1.0,
                0.1,
                i % 4 == 0,
                0.5,
                float(i),
                None,
            ),
        )
    assert db.window_summary(path, "2026-01-01T00:00:00.000+00:00") == (20, 5, 19.0)
    assert db.window_summary(path, "2027-01-01T00:00:00.000+00:00") == (0, 0, None)
