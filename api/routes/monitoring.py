"""Monitoring routes for drift and performance history."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, Request

from monitoring.drift_detector import DriftResult, detect_drift
from monitoring.performance_tracker import get_auc_history


router = APIRouter(tags=["monitoring"])


@router.get("/drift-report", response_model=list[DriftResult])
async def drift_report(request: Request) -> list[DriftResult]:
    """Compare the latest predictions against the training reference window."""

    reference_df: pd.DataFrame = getattr(request.app.state, "training_reference_df", pd.DataFrame()).copy()
    prediction_store = getattr(request.app.state, "prediction_store", [])[-500:]
    if reference_df.empty or not prediction_store:
        return []

    current_df = pd.DataFrame(prediction_store)
    columns = [column for column in reference_df.columns if column in current_df.columns]
    return detect_drift(reference_df[columns], current_df[columns])


@router.get("/performance")
async def performance() -> dict:
    """Return recent AUC history from MLflow."""

    return {
        "history": get_auc_history(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
