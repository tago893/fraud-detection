"""The dashboard's only write path: the API's threshold endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

TIMEOUT_S = 3.0


@dataclass(frozen=True)
class ThresholdState:
    threshold: float
    source: str


def get_threshold(api_url: str) -> ThresholdState | None:
    """Current threshold from the API, or None if it cannot be reached."""
    try:
        response = httpx.get(f"{api_url}/threshold", timeout=TIMEOUT_S)
        response.raise_for_status()
        body = response.json()
        return ThresholdState(float(body["threshold"]), str(body["source"]))
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def put_threshold(api_url: str, threshold: float) -> ThresholdState:
    """PUT /threshold. Raises httpx.HTTPError (or ValueError on a bad body) if the change did not apply."""
    response = httpx.put(f"{api_url}/threshold", json={"threshold": threshold}, timeout=TIMEOUT_S)
    response.raise_for_status()
    body = response.json()
    return ThresholdState(float(body["threshold"]), str(body["source"]))
