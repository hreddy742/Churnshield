"""Prediction route for churn scoring."""

from __future__ import annotations

from itertools import count
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request

from api.schemas import CustomerInput, PredictionResult


router = APIRouter(tags=["predict"])
_ID_COUNTER = count(1)


def _extract_sentiment_scores(texts: list[str]) -> list[float]:
    try:
        from pipelines.feature_engineering import extract_sentiment_score

        return extract_sentiment_score(texts)
    except Exception:
        positive_words = {"helpful", "smooth", "reliable", "easy", "improved", "appreciate"}
        negative_words = {"frustrated", "billing", "outage", "complaint", "dissatisfaction", "competitor"}
        scores: list[float] = []
        for text in texts:
            tokens = {token.strip(".,!?").lower() for token in text.split()}
            pos_hits = len(tokens.intersection(positive_words))
            neg_hits = len(tokens.intersection(negative_words))
            raw_score = 0.5 + 0.12 * pos_hits - 0.12 * neg_hits
            scores.append(float(min(1.0, max(0.0, raw_score))))
        return scores


def _risk_level(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability <= 0.6:
        return "medium"
    return "high"


def _build_input_dataframe(customers: list[CustomerInput]) -> pd.DataFrame:
    rows = []
    for customer in customers:
        data = customer.model_dump()
        data["customer_id"] = data["customer_id"] or f"CUST-{next(_ID_COUNTER):06d}"
        rows.append(data)

    df = pd.DataFrame(rows)
    df["note_sentiment_score"] = _extract_sentiment_scores(df["customer_notes"].fillna("").astype(str).tolist())
    return df


def _transform_dataframe(request: Request, df: pd.DataFrame) -> pd.DataFrame:
    bundle = getattr(request.app.state, "model_bundle", {"transformer": None, "preprocessing_config": None})
    transformer = bundle.get("transformer")
    preprocessing_config = bundle.get("preprocessing_config")

    if transformer is None or preprocessing_config is None:
        columns = [
            "tenure_months",
            "monthly_charges",
            "num_support_tickets",
            "avg_satisfaction_score",
            "note_sentiment_score",
        ]
        return df[columns].copy()

    from pipelines.preprocessing import transform_features

    return transform_features(df, transformer, preprocessing_config)


def _predict_probabilities(bundle: dict[str, Any], transformed_df: pd.DataFrame) -> np.ndarray:
    probabilities = []
    for model_name in ("logistic_regression", "xgboost", "lightgbm"):
        model = bundle["models"].get(model_name)
        if model is None:
            continue
        probabilities.append(np.asarray(model.predict_proba(transformed_df))[:, 1])

    if not probabilities:
        return np.zeros(len(transformed_df), dtype=float)
    return np.mean(np.vstack(probabilities), axis=0)


def _top_features(raw_row: pd.Series) -> list[str]:
    contributions = {
        "monthly_charges": float(raw_row.get("monthly_charges", 0.0)) / 100.0,
        "num_support_tickets": float(raw_row.get("num_support_tickets", 0.0)) / 5.0,
        "avg_satisfaction_score": (10.0 - float(raw_row.get("avg_satisfaction_score", 5.0))) / 10.0,
        "tenure_months": max(0.0, 24.0 - float(raw_row.get("tenure_months", 12.0))) / 24.0,
        "note_sentiment_score": 1.0 - float(raw_row.get("note_sentiment_score", 0.5)),
    }
    return [name for name, _ in sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:3]]


@router.post("/predict", response_model=list[PredictionResult])
async def predict(customers: list[CustomerInput], request: Request) -> list[PredictionResult]:
    """Score one or more customers for churn risk."""

    input_df = _build_input_dataframe(customers)
    transformed_df = _transform_dataframe(request, input_df)
    bundle = getattr(request.app.state, "model_bundle", {"models": {}})
    probabilities = _predict_probabilities(bundle, transformed_df)

    outputs: list[PredictionResult] = []
    for idx, probability in enumerate(probabilities):
        raw_row = input_df.iloc[idx]
        outputs.append(
            PredictionResult(
                customer_id=str(raw_row["customer_id"]),
                churn_probability=float(probability),
                risk_level=_risk_level(float(probability)),
                top_3_features=_top_features(raw_row),
            )
        )

    if not hasattr(request.app.state, "prediction_store"):
        request.app.state.prediction_store = []
    request.app.state.prediction_store.extend(input_df.to_dict(orient="records"))
    return outputs
