"""Request and response models for the scoring API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from pipeline.config import FEATURE_ORDER, SCALED_COLUMNS

Feature = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFeature = Annotated[float, Field(allow_inf_nan=False, ge=0)]

_feature_fields = {
    name: (NonNegativeFeature if name in SCALED_COLUMNS else Feature, ...)
    for name in FEATURE_ORDER
}

# All 30 features are required floats; Time and Amount must be >= 0; unknown fields are rejected.
# true_label is stored for simulation stats only and is never read by the scorer.
TransactionRequest = create_model(
    "TransactionRequest",
    __config__=ConfigDict(extra="forbid"),
    transaction_id=(str | None, Field(None, min_length=1, max_length=128)),
    **_feature_fields,
    true_label=(Literal[0, 1] | None, None),
)


class ScoringResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    flagged: bool
    decision_threshold: float
    model_version: str
    latency_ms: float


class ThresholdUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ThresholdResponse(BaseModel):
    threshold: float
    source: Literal["artifact", "override"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    threshold: float
