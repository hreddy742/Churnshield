from __future__ import annotations

import sys
from pathlib import Path

import pytest


pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("pytest_asyncio")

from httpx import ASGITransport, AsyncClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


@pytest.mark.asyncio
async def test_predict_valid_payload_returns_schema() -> None:
    payload = [
        {
            "tenure_months": 12,
            "monthly_charges": 89.5,
            "contract_type": "month-to-month",
            "num_support_tickets": 3,
            "avg_satisfaction_score": 5.8,
            "customer_notes": "Customer is frustrated with billing issues.",
        }
    ]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=payload)

    body = response.json()
    assert response.status_code == 200
    assert isinstance(body, list)
    assert "customer_id" in body[0]
    assert "churn_probability" in body[0]
    assert "risk_level" in body[0]
    assert "top_3_features" in body[0]


@pytest.mark.asyncio
async def test_predict_missing_fields_returns_422() -> None:
    payload = [{"tenure_months": 12, "monthly_charges": 89.5}]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_drift_report_returns_list() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/drift-report")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
