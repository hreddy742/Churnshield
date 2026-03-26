import shap, joblib, numpy as np, pandas as pd
from fastapi import APIRouter, HTTPException
from backend.api.schemas import CustomerInput
from ml.pipelines.features import extract_sentiment_scores
from pathlib import Path

router = APIRouter(prefix="/explain", tags=["explain"])

_bundle = None
_explainer = None

def get_bundle():
    global _bundle
    if _bundle is None:
        path = Path("ml/models/ensemble.pkl")
        if not path.exists():
            raise RuntimeError("Model not found.")
        _bundle = joblib.load(path)
    return _bundle

def get_explainer():
    global _explainer
    if _explainer is None:
        bundle = get_bundle()
        _explainer = shap.TreeExplainer(bundle["xgb"])
    return _explainer

@router.post("/shap")
def explain_shap(customer: CustomerInput):
    bundle = get_bundle()
    explainer = get_explainer()

    notes = customer.customer_notes or "no notes provided"
    sentiment_score = extract_sentiment_scores([notes])[0]

    row = {
        "tenure_months": customer.tenure_months,
        "monthly_charges": customer.monthly_charges,
        "num_support_tickets": customer.num_support_tickets,
        "avg_satisfaction_score": customer.avg_satisfaction_score,
        "num_products": customer.num_products,
        "sentiment_score": sentiment_score,
        "contract_type": customer.contract_type,
        "payment_method": customer.payment_method,
    }

    X = bundle["preprocessor"].transform(pd.DataFrame([row]))
    shap_vals = explainer.shap_values(X)[0]
    feature_names = bundle["feature_names"]

    return {
        "base_value": round(float(explainer.expected_value), 4),
        "shap_values": [
            {
                "feature": name,
                "value": round(float(val), 4),
                "display_name": name.replace("_", " ").title()
            }
            for name, val in zip(feature_names, shap_vals)
        ]
    }
