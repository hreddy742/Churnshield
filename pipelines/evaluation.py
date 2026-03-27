"""Evaluation and SHAP reporting utilities for the ChurnGuard ensemble."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

try:
    import shap
except ModuleNotFoundError:  # pragma: no cover
    shap = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.preprocessing import transform_features
from pipelines.training import DEFAULT_MODEL_PATH, DEFAULT_TEST_SPLIT_PATH, TARGET_COLUMN

DEFAULT_REPORT_PATH = PROJECT_ROOT / "models" / "evaluation_metrics.json"
DEFAULT_SHAP_PLOT_PATH = PROJECT_ROOT / "models" / "ensemble_feature_importance.png"


def load_bundle(model_path: Path) -> dict[str, Any]:
    """Load the persisted ensemble bundle."""

    with model_path.open("rb") as file_obj:
        return pickle.load(file_obj)


def load_test_split(test_split_path: Path) -> pd.DataFrame:
    """Load the saved evaluation split."""

    if not test_split_path.exists():
        raise FileNotFoundError(f"Saved test split not found: {test_split_path}")
    return pd.read_csv(test_split_path)


def predict_ensemble_probabilities(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Average member-model probabilities for the final ensemble output."""

    probabilities = [
        np.asarray(bundle["models"][model_name].predict_proba(X))[:, 1]
        for model_name in ("logistic_regression", "xgboost", "lightgbm")
    ]
    return np.mean(np.vstack(probabilities), axis=0)


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute standard binary classification metrics for the ensemble."""

    predictions = (probabilities >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, probabilities),
        "f1": f1_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
    }


def compute_tree_ensemble_shap_importance(bundle: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Approximate ensemble SHAP importance by averaging tree-model TreeExplainer outputs."""

    if shap is None:
        return _fallback_feature_importance(bundle, X.columns.tolist())

    shap_matrices: list[np.ndarray] = []
    try:
        for model_name in ("xgboost", "lightgbm"):
            explainer = shap.TreeExplainer(bundle["models"][model_name])
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_array = np.asarray(shap_values[-1])
            else:
                shap_array = np.asarray(shap_values)
            shap_matrices.append(np.abs(shap_array))
    except Exception:
        return _fallback_feature_importance(bundle, X.columns.tolist())

    mean_abs_shap = np.mean(np.stack(shap_matrices, axis=0), axis=0).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": bundle["feature_names"],
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return importance_df.reset_index(drop=True)


def _fallback_feature_importance(bundle: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    importances: list[np.ndarray] = []
    for model_name in ("logistic_regression", "xgboost", "lightgbm"):
        model = bundle["models"].get(model_name)
        if model is None:
            continue
        if hasattr(model, "feature_importances_"):
            importances.append(np.abs(np.asarray(model.feature_importances_, dtype=float)))
        elif hasattr(model, "coef_"):
            importances.append(np.abs(np.ravel(np.asarray(model.coef_, dtype=float))))

    if importances:
        mean_importance = np.mean(np.vstack(importances), axis=0)
    else:
        mean_importance = np.ones(len(feature_names), dtype=float)

    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_importance})
    return importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(importance_df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    """Persist a horizontal SHAP feature-importance plot."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_features = importance_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_features["feature"], top_features["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_ylabel("Feature")
    ax.set_title("Ensemble Feature Importance (Tree SHAP Approximation)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_saved_ensemble(
    model_path: Path,
    test_split_path: Path,
    report_path: Path,
    shap_plot_path: Path,
) -> dict[str, float]:
    """Load the saved ensemble, score the holdout split, and write evaluation artifacts."""

    bundle = load_bundle(model_path)
    test_df = load_test_split(test_split_path)
    transformer = bundle["transformer"]
    preprocessing_config = bundle["preprocessing_config"]

    X_test = transform_features(test_df, transformer, preprocessing_config)
    y_test = test_df[TARGET_COLUMN]
    probabilities = predict_ensemble_probabilities(bundle, X_test)
    metrics = compute_metrics(y_test, probabilities)

    importance_df = compute_tree_ensemble_shap_importance(bundle, X_test)
    save_feature_importance_plot(importance_df, shap_plot_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance_df.to_csv(shap_plot_path.with_suffix(".csv"), index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-split-path", type=Path, default=DEFAULT_TEST_SPLIT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--shap-plot-path", type=Path, default=DEFAULT_SHAP_PLOT_PATH)
    return parser.parse_args()


def main() -> None:
    """Run end-to-end evaluation for the saved ensemble artifact."""

    args = parse_args()
    metrics = evaluate_saved_ensemble(
        model_path=args.model_path,
        test_split_path=args.test_split_path,
        report_path=args.report_path,
        shap_plot_path=args.shap_plot_path,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
