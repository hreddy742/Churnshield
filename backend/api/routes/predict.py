import time, shap, joblib, numpy as np, pandas as pd
from fastapi import APIRouter, HTTPException
from backend.api.schemas import (
    CustomerInput, PredictionResult, FactorContribution
)
from ml.pipelines.features import extract_sentiment_scores
from pathlib import Path

router = APIRouter(prefix="/predict", tags=["predict"])

_bundle = None
_explainer = None

def get_bundle():
    global _bundle
    if _bundle is None:
        path = Path("ml/models/ensemble.pkl")
        if not path.exists():
            raise RuntimeError(
                "Model not found. Run: python ml/pipelines/train.py"
            )
        _bundle = joblib.load(path)
    return _bundle

def get_explainer():
    global _explainer
    if _explainer is None:
        bundle = get_bundle()
        _explainer = shap.TreeExplainer(bundle["xgb"])
    return _explainer

@router.post("/", response_model=PredictionResult)
def predict(customer: CustomerInput):
    start = time.time()
    bundle = get_bundle()

    row = {
        "tenure_months": customer.tenure_months,
        "monthly_charges": customer.monthly_charges,
        "contract_type": customer.contract_type,
        "num_support_tickets": customer.num_support_tickets,
        "avg_satisfaction_score": customer.avg_satisfaction_score,
        "payment_method": customer.payment_method,
        "num_products": customer.num_products,
        "customer_notes": customer.customer_notes or "no notes provided"
    }

    df = pd.DataFrame([row])
    sentiment_score = extract_sentiment_scores([row["customer_notes"]])[0]
    df["sentiment_score"] = sentiment_score

    feature_cols = [
        "tenure_months", "monthly_charges", "num_support_tickets",
        "avg_satisfaction_score", "num_products", "sentiment_score",
        "contract_type", "payment_method"
    ]
    X = bundle["preprocessor"].transform(df[feature_cols])

    xgb_prob = bundle["xgb"].predict_proba(X)[0, 1]
    lgb_prob = bundle["lgb"].predict_proba(X)[0, 1]
    churn_prob = (xgb_prob + lgb_prob) / 2

    explainer = get_explainer()
    shap_vals = explainer.shap_values(X)[0]

    feature_names = bundle["feature_names"]
    factors = []
    for name, shap_val in zip(feature_names, shap_vals):
        display = name.replace("_", " ").replace("contract type ", "Contract: ")
        display = display.replace("payment method ", "Payment: ").title()
        factors.append(FactorContribution(
            feature=name,
            display_name=display,
            shap_value=round(float(shap_val), 4),
            direction="increases_risk" if shap_val > 0 else "decreases_risk"
        ))

    factors.sort(key=lambda x: abs(x.shap_value), reverse=True)
    top_factors = factors[:5]

    risk_tier = (
        "low" if churn_prob < 0.3
        else "medium" if churn_prob < 0.6
        else "high"
    )

    return PredictionResult(
        churn_probability=round(float(churn_prob), 4),
        risk_tier=risk_tier,
        top_factors=top_factors,
        shap_base_value=round(float(explainer.expected_value), 4),
        inference_ms=int((time.time() - start) * 1000)
    )
