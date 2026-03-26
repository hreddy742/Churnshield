from pydantic import BaseModel, Field
from typing import Literal, Optional

class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=1, le=84)
    monthly_charges: float = Field(..., ge=20, le=120)
    contract_type: Literal["Month-to-Month", "One Year", "Two Year"]
    num_support_tickets: int = Field(..., ge=0, le=15)
    avg_satisfaction_score: float = Field(..., ge=1.0, le=5.0)
    payment_method: Literal["Electronic", "Mailed Check", "Bank Transfer", "Credit Card"]
    num_products: int = Field(..., ge=1, le=4)
    customer_notes: Optional[str] = None

class FactorContribution(BaseModel):
    feature: str
    display_name: str
    shap_value: float
    direction: str

class PredictionResult(BaseModel):
    churn_probability: float
    risk_tier: str
    top_factors: list[FactorContribution]
    shap_base_value: float
    inference_ms: int

class DriftFeature(BaseModel):
    feature: str
    type: str
    score: float
    pvalue: Optional[float]
    status: str

class DriftReport(BaseModel):
    computed_at: str
    features: list[DriftFeature]
    overall_status: str
    predictions_analyzed: int
