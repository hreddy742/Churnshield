"""Training pipeline for ChurnGuard models and MLflow experiment logging.

# Run data/generate_synthetic.py first to create the required CSV
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import mlflow
except ModuleNotFoundError:  # pragma: no cover
    mlflow = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:  # pragma: no cover
    LGBMClassifier = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.feature_engineering import extract_sentiment_score
from pipelines.preprocessing import fit_transform_features, transform_features

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn_synthetic.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ensemble.pkl"
DEFAULT_TEST_SPLIT_PATH = PROJECT_ROOT / "models" / "test_split.csv"
DEFAULT_EXPERIMENT_NAME = "churnguard-training"
RANDOM_STATE = 42
TARGET_COLUMN = "churn_label"
TEXT_COLUMN = "customer_notes"


MODEL_CONFIGS = {
    "logistic_regression": {
        "factory": LogisticRegression,
        "params": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1500,
            "solver": "lbfgs",
        },
    },
    "xgboost": {
        "factory": XGBClassifier or GradientBoostingClassifier,
        "params": (
            {
                "n_estimators": 320,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_child_weight": 2,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": RANDOM_STATE,
            }
            if XGBClassifier is not None
            else {
                "n_estimators": 200,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "max_depth": 3,
                "random_state": RANDOM_STATE,
            }
        ),
    },
    "lightgbm": {
        "factory": LGBMClassifier or RandomForestClassifier,
        "params": (
            {
                "n_estimators": 320,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": RANDOM_STATE,
                "verbosity": -1,
            }
            if LGBMClassifier is not None
            else {
                "n_estimators": 240,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            }
        ),
    },
}


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load the training dataset from disk."""

    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")
    return pd.read_csv(data_path)


def add_sentiment_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Augment a DataFrame with local DistilBERT sentiment scores."""

    output_df = df.copy()
    output_df["note_sentiment_score"] = extract_sentiment_score(output_df[TEXT_COLUMN].fillna("").astype(str).tolist())
    return output_df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split raw data into train, validation, and test windows."""

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _plot_roc_curve(y_true: pd.Series, probabilities: np.ndarray, title: str, output_path: Path) -> None:
    """Save a ROC curve figure for a model run."""

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_score = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _score_split(model: Any, X: pd.DataFrame, y: pd.Series) -> tuple[float, float, np.ndarray]:
    """Return AUC, PR-AUC, and positive-class probabilities for one split."""

    probabilities = np.asarray(model.predict_proba(X))[:, 1]
    auc_score = roc_auc_score(y, probabilities)
    pr_auc = average_precision_score(y, probabilities)
    return auc_score, pr_auc, probabilities


def _build_model(model_name: str):
    """Instantiate one configured classifier by name."""

    config = MODEL_CONFIGS[model_name]
    return config["factory"](**config["params"])


def _log_run_to_mlflow(
    run_name: str,
    model_name: str,
    params: dict[str, Any],
    y_train: pd.Series,
    train_probabilities: np.ndarray,
    y_val: pd.Series,
    val_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
) -> dict[str, float]:
    """Create a standalone MLflow run for one model and log metrics plus ROC artifact."""

    metrics = {
        "train_auc": roc_auc_score(y_train, train_probabilities),
        "val_auc": roc_auc_score(y_val, val_probabilities),
        "test_auc": roc_auc_score(y_test, test_probabilities),
        "train_pr_auc": average_precision_score(y_train, train_probabilities),
        "val_pr_auc": average_precision_score(y_val, val_probabilities),
        "test_pr_auc": average_precision_score(y_test, test_probabilities),
    }

    if mlflow is None:
        return metrics

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", model_name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / f"{run_name}_roc_curve.png"
            _plot_roc_curve(y_test, test_probabilities, f"{run_name} ROC Curve", artifact_path)
            mlflow.log_artifact(str(artifact_path), artifact_path="plots")
    return metrics


def average_probabilities(probability_arrays: list[np.ndarray]) -> np.ndarray:
    """Compute soft-voting ensemble probabilities by simple averaging."""

    stacked = np.vstack(probability_arrays)
    return np.mean(stacked, axis=0)


def train_models(data_path: Path, model_output_path: Path, mlflow_tracking_uri: str | None = None) -> dict[str, Any]:
    """Train baseline, tree models, and a soft-voting ensemble with MLflow logging."""

    if mlflow is not None and mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    if mlflow is not None:
        mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)

    raw_df = load_dataset(data_path)
    enriched_df = add_sentiment_feature(raw_df)
    train_df, val_df, test_df = split_dataset(enriched_df)

    X_train, y_train, transformer, preprocessing_config = fit_transform_features(train_df)
    X_val = transform_features(val_df, transformer, preprocessing_config)
    X_test = transform_features(test_df, transformer, preprocessing_config)
    feature_names = list(X_train.columns)

    trained_models: dict[str, Any] = {}
    model_probabilities: dict[str, dict[str, np.ndarray]] = {}
    metrics_by_run: dict[str, dict[str, float]] = {}

    for model_name in ("logistic_regression", "xgboost", "lightgbm"):
        model = _build_model(model_name)
        model.fit(X_train, y_train)

        train_auc, train_pr_auc, train_probabilities = _score_split(model, X_train, y_train)
        val_auc, val_pr_auc, val_probabilities = _score_split(model, X_val, y_val := val_df[TARGET_COLUMN])
        test_auc, test_pr_auc, test_probabilities = _score_split(model, X_test, y_test := test_df[TARGET_COLUMN])

        metrics = _log_run_to_mlflow(
            run_name=model_name,
            model_name=model_name,
            params=MODEL_CONFIGS[model_name]["params"],
            y_train=y_train,
            train_probabilities=train_probabilities,
            y_val=y_val,
            val_probabilities=val_probabilities,
            y_test=y_test,
            test_probabilities=test_probabilities,
        )

        trained_models[model_name] = model
        model_probabilities[model_name] = {
            "train": train_probabilities,
            "val": val_probabilities,
            "test": test_probabilities,
        }
        metrics_by_run[model_name] = {
            "train_auc": train_auc,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "train_pr_auc": train_pr_auc,
            "val_pr_auc": val_pr_auc,
            "test_pr_auc": test_pr_auc,
        }

    ensemble_train_probabilities = average_probabilities(
        [model_probabilities[name]["train"] for name in ("logistic_regression", "xgboost", "lightgbm")]
    )
    ensemble_val_probabilities = average_probabilities(
        [model_probabilities[name]["val"] for name in ("logistic_regression", "xgboost", "lightgbm")]
    )
    ensemble_test_probabilities = average_probabilities(
        [model_probabilities[name]["test"] for name in ("logistic_regression", "xgboost", "lightgbm")]
    )

    ensemble_metrics = _log_run_to_mlflow(
        run_name="ensemble",
        model_name="soft_voting_ensemble",
        params={
            "members": "logistic_regression,xgboost,lightgbm",
            "aggregation": "mean_positive_probability",
        },
        y_train=y_train,
        train_probabilities=ensemble_train_probabilities,
        y_val=val_df[TARGET_COLUMN],
        val_probabilities=ensemble_val_probabilities,
        y_test=test_df[TARGET_COLUMN],
        test_probabilities=ensemble_test_probabilities,
    )
    metrics_by_run["ensemble"] = ensemble_metrics

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(DEFAULT_TEST_SPLIT_PATH, index=False)

    bundle = {
        "models": trained_models,
        "transformer": transformer,
        "preprocessing_config": preprocessing_config,
        "feature_names": feature_names,
        "metrics": metrics_by_run,
        "test_split_path": str(DEFAULT_TEST_SPLIT_PATH),
    }
    with model_output_path.open("wb") as file_obj:
        pickle.dump(bundle, file_obj)

    metrics_path = model_output_path.with_name("training_metrics.json")
    metrics_path.write_text(json.dumps(metrics_by_run, indent=2), encoding="utf-8")
    return bundle


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Train all models and persist the ensemble bundle."""

    args = parse_args()
    bundle = train_models(
        data_path=args.data_path,
        model_output_path=args.model_output,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    ensemble_auc = bundle["metrics"]["ensemble"]["test_auc"]
    print(f"Saved ensemble bundle to {args.model_output}")
    print(f"Ensemble test AUC: {ensemble_auc:.4f}")


if __name__ == "__main__":
    main()
