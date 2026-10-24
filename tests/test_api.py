from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.scoring import Scorer
from pipeline.config import ARTIFACTS, FEATURE_ORDER, TARGET

pytestmark = pytest.mark.skipif(not (ARTIFACTS / "threshold.json").exists(), reason="run make train evaluate first")


def row_payload(row: pd.Series, **extra) -> dict:
    return {**{name: float(row[name]) for name in FEATURE_ORDER}, **extra}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="module")
def test_rows(splits):
    return splits["test"]


@pytest.fixture(scope="module")
def known_fraud(test_rows, client):
    """The test-slice fraud row the model is most sure about (smoke test only; nothing is tuned on it)."""
    frauds = test_rows[test_rows[TARGET] == 1]
    scored = [(client.post("/score", json=row_payload(r)).json()["fraud_probability"], i) for i, r in frauds.iterrows()]
    return test_rows.loc[max(scored)[1]]


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and 0 < body["threshold"] < 1 and body["model_version"]


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
    assert client.post("/score", json=row_payload(row, transaction_id="tx-1")).json()["transaction_id"] == "tx-1"
    assert len(client.post("/score", json=row_payload(row)).json()["transaction_id"]) == 36


@pytest.mark.parametrize("mutate", [lambda b: b.pop("V7"), lambda b: b.update(Amount=-1), lambda b: b.update(x=1)])
def test_invalid_requests_are_422(client, test_rows, mutate):
    body = row_payload(test_rows.iloc[0])
    mutate(body)
    assert client.post("/score", json=body).status_code == 422


def test_scorer_matches_sklearn_scaler(client, test_rows, scaler):
    scorer: Scorer = client.app.state.scorer
    sample = test_rows.iloc[:50][list(scorer.feature_order)]
    got = np.vstack([scorer.scale_row(sample.iloc[i].to_numpy(float)) for i in range(len(sample))])
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
            client.post("/score", json=row_payload(row, true_label=label)).json()["fraud_probability"]
            for label in (None, 0, 1)
        ]
    finally:
        client.app.state.scorer = real
    assert len(set(probs)) == 1
    assert len(seen) == 3 and all(x.shape == (1, len(FEATURE_ORDER)) for x in seen)
    np.testing.assert_array_equal(seen[0], seen[1])
    np.testing.assert_array_equal(seen[1], seen[2])
