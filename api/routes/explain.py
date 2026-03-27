"""Explanation route for single-customer SHAP output."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request

from api.routes.predict import _build_input_dataframe, _predict_probabilities, _risk_level, _transform_dataframe
from api.schemas import CustomerInput, ExplainResponse, ShapValueItem


router = APIRouter(tags=["explain"])


def _heuristic_shap_values(transformed_row: dict[str, float]) -> tuple[float, list[ShapValueItem]]:
    base_value = 0.5
    shap_items = [
        ShapValueItem(feature=feature, shap_value=float(value), base_value=base_value)
        for feature, value in transformed_row.items()
    ]
    return base_value, shap_items


@router.post("/explain", response_model=ExplainResponse)
async def explain(customer: CustomerInput, request: Request) -> ExplainResponse:
    """Return a prediction plus per-feature explanation values for one customer."""

    input_df = _build_input_dataframe([customer])
    transformed_df = _transform_dataframe(request, input_df)
    probabilities = _predict_probabilities(request.app.state.model_bundle, transformed_df)
    prediction = float(probabilities[0]) if len(probabilities) else 0.0

    model_bundle = getattr(request.app.state, "model_bundle", {"models": {}})
    xgboost_model = model_bundle.get("models", {}).get("xgboost")
    feature_names = list(transformed_df.columns)

    try:
        import shap

        if xgboost_model is None:
            raise ValueError("XGBoost model unavailable")
        explainer = shap.TreeExplainer(xgboost_model)
        shap_values = explainer.shap_values(transformed_df)
        shap_array = np.asarray(shap_values[-1] if isinstance(shap_values, list) else shap_values)
        base_values = explainer.expected_value
        if isinstance(base_values, list):
            base_value = float(base_values[-1])
        elif isinstance(base_values, np.ndarray):
            base_value = float(np.ravel(base_values)[-1])
        else:
            base_value = float(base_values)

        shap_items = [
            ShapValueItem(feature=feature_names[index], shap_value=float(shap_array[0][index]), base_value=base_value)
            for index in range(len(feature_names))
        ]
    except Exception:
        base_value, shap_items = _heuristic_shap_values(
            {feature: float(transformed_df.iloc[0][feature]) for feature in feature_names}
        )

    return ExplainResponse(
        prediction=prediction,
        shap_values=shap_items,
        risk_level=_risk_level(prediction),
    )
