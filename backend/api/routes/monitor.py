import json
import pandas as pd
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pathlib import Path
from backend.api.schemas import DriftReport, DriftFeature
from ml.monitoring.drift import detect_drift

router = APIRouter(prefix="/monitor", tags=["monitor"])

MODELS_DIR = Path("ml/models")
DATA_DIR = Path("ml/data/raw")

@router.get("/drift", response_model=DriftReport)
def get_drift():
    reference_path = DATA_DIR / "customer_data.csv"
    if not reference_path.exists():
        raise HTTPException(status_code=500, detail="Training data not found")

    reference_df = pd.read_csv(reference_path).sample(
        n=5000, random_state=42
    )

    # For now simulate recent predictions using a slightly perturbed sample
    # In production this would query PostgreSQL for last 500 predictions
    current_df = reference_df.sample(n=min(500, len(reference_df)), random_state=99)

    drift_results = detect_drift(reference_df, current_df)

    features = [
        DriftFeature(
            feature=d.feature,
            type=d.type,
            score=d.score,
            pvalue=d.pvalue,
            status=d.status
        )
        for d in drift_results
    ]

    status_counts = {"stable": 0, "warning": 0, "critical": 0}
    for f in features:
        status_counts[f.status] = status_counts.get(f.status, 0) + 1

    if status_counts.get("critical", 0) > 0:
        overall_status = "critical"
    elif status_counts.get("warning", 0) > 0:
        overall_status = "warning"
    else:
        overall_status = "stable"

    return DriftReport(
        computed_at=datetime.now(timezone.utc).isoformat(),
        features=features,
        overall_status=overall_status,
        predictions_analyzed=len(current_df)
    )

@router.get("/metrics")
def get_metrics():
    try:
        with open(MODELS_DIR / "training_report.json") as f:
            training_report = json.load(f)
        with open(MODELS_DIR / "roc_curve.json") as f:
            roc_curve = json.load(f)
        with open(MODELS_DIR / "feature_importance.json") as f:
            feature_importance = json.load(f)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model artifacts not found: {e}. Run training first."
        )

    return {
        "training_report": training_report,
        "roc_curve": roc_curve,
        "feature_importance": feature_importance
    }
