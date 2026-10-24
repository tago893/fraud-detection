import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.schemas import TransactionRequest, ThresholdUpdate
from pipeline.config import FEATURE_ORDER

app = FastAPI()


@app.post("/echo")
def echo(req: TransactionRequest) -> dict:  # type: ignore[valid-type]
    return {"ok": True}


@app.put("/t")
def put_t(body: ThresholdUpdate) -> dict:
    return {"ok": True}


client = TestClient(app)


def payload(**overrides) -> dict:
    base = {name: 0.0 for name in FEATURE_ORDER}
    base["Amount"] = 149.62
    return {**base, **overrides}


def test_valid_payload_accepted_with_optional_fields_absent():
    assert client.post("/echo", json=payload()).status_code == 200


def test_valid_payload_with_id_and_label_accepted():
    assert client.post("/echo", json=payload(transaction_id="abc", true_label=1)).status_code == 200


def test_missing_v7_is_422():
    body = payload()
    del body["V7"]
    assert client.post("/echo", json=body).status_code == 422


@pytest.mark.parametrize("field", ["Amount", "Time"])
def test_negative_time_or_amount_is_422(field):
    assert client.post("/echo", json=payload(**{field: -1.0})).status_code == 422


def test_extra_field_is_422():
    assert client.post("/echo", json=payload(bogus=1)).status_code == 422


@pytest.mark.parametrize("bad", [2, -1, "fraud"])
def test_bad_true_label_is_422(bad):
    assert client.post("/echo", json=payload(true_label=bad)).status_code == 422


def test_non_numeric_feature_is_422():
    assert client.post("/echo", json=payload(V3="x")).status_code == 422


@pytest.mark.parametrize("value", [1.5, -0.1])
def test_threshold_out_of_range_is_422(value):
    assert client.put("/t", json={"threshold": value}).status_code == 422


def test_threshold_in_range_ok():
    assert client.put("/t", json={"threshold": 0.99}).status_code == 200
