"""Utilities for retrieving model performance history from MLflow."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings


def get_auc_history(n_runs: int = 20) -> list[dict]:
    """Return recent MLflow runs with ensemble AUC metrics sorted by time."""

    try:
        import mlflow
    except ModuleNotFoundError:
        return []

    settings = get_settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    try:
        experiment = mlflow.get_experiment_by_name("churnguard-training")
        if experiment is None:
            return []

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=n_runs,
        )
    except Exception:
        return []

    history: list[dict] = []
    for _, row in runs_df.sort_values("start_time").iterrows():
        timestamp = row.get("start_time")
        if timestamp is None:
            continue
        timestamp_value = datetime.fromtimestamp(float(timestamp) / 1000.0, tz=timezone.utc).isoformat()
        history.append(
            {
                "run_id": row.get("run_id"),
                "timestamp": timestamp_value,
                "auc": row.get("metrics.test_auc"),
                "model_type": row.get("tags.model_family") or row.get("tags.mlflow.runName"),
            }
        )
    return history
