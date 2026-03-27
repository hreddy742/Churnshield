"""Automated retraining workflow triggered by critical drift detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import get_settings
from monitoring.alerts import log_drift_alert
from monitoring.drift_detector import DriftResult, detect_drift

try:
    from .training import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, train_models
except ImportError:
    from training import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, train_models


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _has_critical_drift(drift_results: list[DriftResult]) -> bool:
    return any(result.status == "critical" for result in drift_results)


def _get_previous_best_auc() -> float:
    try:
        import mlflow
    except ModuleNotFoundError:
        return float("-inf")

    settings = get_settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name("churnguard-training")
    if experiment is None:
        return float("-inf")

    try:
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["metrics.test_auc DESC"],
            max_results=1,
        )
    except Exception:
        return float("-inf")

    if runs_df.empty:
        return float("-inf")
    return float(runs_df.iloc[0].get("metrics.test_auc", float("-inf")))


def _register_model_if_better(model_path: Path, new_auc: float, previous_auc: float) -> bool:
    if new_auc <= previous_auc:
        return False

    try:
        import mlflow
    except ModuleNotFoundError:
        return False

    settings = get_settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name="retraining_registration"):
        mlflow.log_metric("candidate_test_auc", new_auc)
        mlflow.log_metric("previous_best_test_auc", previous_auc)
        mlflow.log_artifact(str(model_path), artifact_path="candidate_model")
        mlflow.set_tag("model_registration_status", "accepted")
    return True


def retrain_if_needed(
    reference_path: Path,
    current_path: Path,
    training_data_path: Path,
    model_output_path: Path,
) -> dict[str, Any]:
    """Run drift detection and retrain only when a critical feature is detected."""

    reference_df = _load_dataframe(reference_path)
    current_df = _load_dataframe(current_path)
    drift_results = detect_drift(reference_df, current_df)
    log_drift_alert(drift_results)

    if not _has_critical_drift(drift_results):
        return {
            "retrained": False,
            "registered": False,
            "drift_results": [result.model_dump() for result in drift_results],
        }

    previous_auc = _get_previous_best_auc()
    bundle = train_models(training_data_path, model_output_path, get_settings().MLFLOW_TRACKING_URI)
    new_auc = float(bundle["metrics"]["ensemble"]["test_auc"])
    registered = _register_model_if_better(model_output_path, new_auc, previous_auc)

    return {
        "retrained": True,
        "registered": registered,
        "previous_best_auc": previous_auc,
        "new_auc": new_auc,
        "drift_results": [result.model_dump() for result in drift_results],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for retraining."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--current-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--training-data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-output-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    """Entry point for retraining workflow."""

    args = parse_args()
    result = retrain_if_needed(
        reference_path=args.reference_path,
        current_path=args.current_path,
        training_data_path=args.training_data_path,
        model_output_path=args.model_output_path,
    )
    print(result)


if __name__ == "__main__":
    main()
