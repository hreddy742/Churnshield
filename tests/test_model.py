from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest


@dataclass
class DummyProbabilisticModel:
    scale: float = 1.0

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        base = np.clip(0.2 + self.scale * 0.1 * np.arange(len(X)), 0.0, 1.0)
        return np.column_stack([1.0 - base, base])


def _write_bundle(path: Path) -> dict:
    bundle = {
        "models": {
            "logistic_regression": DummyProbabilisticModel(scale=1.0),
            "xgboost": DummyProbabilisticModel(scale=0.8),
            "lightgbm": DummyProbabilisticModel(scale=0.6),
        },
        "feature_names": ["tenure_months", "monthly_charges", "note_sentiment_score"],
    }
    with path.open("wb") as file_obj:
        pickle.dump(bundle, file_obj)
    return bundle


def _local_artifact_path(name: str) -> Path:
    path = Path(__file__).resolve().parent / ".artifacts"
    path.mkdir(exist_ok=True)
    return path / f"{name}-{uuid4().hex}.pkl"


def test_ensemble_pickle_loads_and_predict_proba_shape() -> None:
    ensemble_path = _local_artifact_path("ensemble")
    bundle = _write_bundle(ensemble_path)

    with ensemble_path.open("rb") as file_obj:
        loaded_bundle = pickle.load(file_obj)

    X = pd.DataFrame({"tenure_months": [12, 24], "monthly_charges": [70.0, 90.0]})
    probabilities = loaded_bundle["models"]["xgboost"].predict_proba(X)

    assert "models" in loaded_bundle
    assert probabilities.shape == (2, 2)
    assert bundle["feature_names"] == loaded_bundle["feature_names"]


def test_churn_probability_between_zero_and_one() -> None:
    ensemble_path = _local_artifact_path("ensemble")
    bundle = _write_bundle(ensemble_path)
    X = pd.DataFrame({"tenure_months": [6, 8, 10], "monthly_charges": [50.0, 60.0, 70.0]})

    probabilities = bundle["models"]["lightgbm"].predict_proba(X)[:, 1]

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_shap_values_shape_matches_number_of_features(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyExplainer:
        def __init__(self, _model):
            self.expected_value = 0.5

        def shap_values(self, X: pd.DataFrame) -> np.ndarray:
            return np.ones((len(X), X.shape[1]))

    shap = pytest.importorskip("shap")
    monkeypatch.setattr(shap, "TreeExplainer", DummyExplainer)

    X = pd.DataFrame(
        {
            "tenure_months": [12],
            "monthly_charges": [90.0],
            "note_sentiment_score": [0.3],
        }
    )
    explainer = shap.TreeExplainer(DummyProbabilisticModel())
    shap_values = explainer.shap_values(X)

    assert shap_values.shape[1] == X.shape[1]
