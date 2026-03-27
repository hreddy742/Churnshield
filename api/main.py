"""FastAPI application entrypoint for ChurnGuard."""

from __future__ import annotations

import json
import logging
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import PROJECT_ROOT, get_settings


LOGGER = logging.getLogger("churnguard.api")


def _register_pickle_compatibility_aliases() -> None:
    """Support older local-script pickle module paths."""

    try:
        from pipelines import preprocessing as pipelines_preprocessing

        sys.modules.setdefault("preprocessing", pipelines_preprocessing)
    except Exception:
        return


class _FallbackTreeModel:
    """Simple probability model used when no trained artifact is available."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        charges = X.get("monthly_charges", pd.Series(0.0, index=X.index)).astype(float)
        tickets = X.get("num_support_tickets", pd.Series(0.0, index=X.index)).astype(float)
        satisfaction = X.get("avg_satisfaction_score", pd.Series(5.0, index=X.index)).astype(float)
        tenure = X.get("tenure_months", pd.Series(12.0, index=X.index)).astype(float)
        sentiment = X.get("note_sentiment_score", pd.Series(0.5, index=X.index)).astype(float)
        logits = -1.1 + 0.018 * charges + 0.16 * tickets - 0.25 * satisfaction - 0.018 * tenure - 0.9 * sentiment
        positive = 1.0 / (1.0 + np.exp(-logits.to_numpy(dtype=float)))
        negative = 1.0 - positive
        return np.column_stack([negative, positive])


def _configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO), force=True)


def _load_model_bundle(model_path: Path) -> dict[str, Any]:
    if model_path.exists():
        with model_path.open("rb") as file_obj:
            _register_pickle_compatibility_aliases()
            bundle = pickle.load(file_obj)
            bundle.setdefault("feature_names", [])
            return bundle

    feature_names = [
        "tenure_months",
        "monthly_charges",
        "num_support_tickets",
        "avg_satisfaction_score",
        "note_sentiment_score",
    ]
    fallback_model = _FallbackTreeModel()
    return {
        "models": {
            "logistic_regression": fallback_model,
            "xgboost": fallback_model,
            "lightgbm": fallback_model,
        },
        "transformer": None,
        "preprocessing_config": None,
        "feature_names": feature_names,
        "metrics": {},
    }


def _load_reference_dataframe() -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "customer_churn_synthetic.csv",
        PROJECT_ROOT / "data" / "raw" / "sample.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources during startup."""

    _configure_logging()
    settings = get_settings()
    app.state.settings = settings
    app.state.model_bundle = _load_model_bundle(Path(settings.MODEL_PATH))
    app.state.training_reference_df = _load_reference_dataframe()
    app.state.prediction_store = []
    LOGGER.info(json.dumps({"event": "startup_complete", "model_path": settings.MODEL_PATH}))
    yield


app = FastAPI(title="ChurnGuard API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from api.routes import explain, monitoring, predict  # noqa: E402


app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(monitoring.router)
