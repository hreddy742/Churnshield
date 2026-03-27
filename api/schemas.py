"""Shared Pydantic schemas for ChurnGuard API routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]


class CustomerInput(BaseModel):
    """Single customer input payload."""

    customer_id: str | None = None
    tenure_months: float = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    contract_type: str
    num_support_tickets: int = Field(..., ge=0)
    avg_satisfaction_score: float = Field(..., ge=0, le=10)
    customer_notes: str


class PredictionResult(BaseModel):
    """Prediction result returned by the API."""

    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    top_3_features: list[str]


class ShapValueItem(BaseModel):
    """Per-feature explanation value."""

    feature: str
    shap_value: float
    base_value: float


class ExplainResponse(BaseModel):
    """Response payload for the explain endpoint."""

    prediction: float = Field(..., ge=0, le=1)
    shap_values: list[ShapValueItem]
    risk_level: RiskLevel


class PerformancePoint(BaseModel):
    """Single historical MLflow performance point."""

    run_id: str | None
    timestamp: str
    auc: float | None
    model_type: str | None
