from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


generate_synthetic = load_module("data/generate_synthetic.py", "generate_synthetic")
preprocessing = load_module("pipelines/preprocessing.py", "preprocessing")


def test_generate_synthetic_dataset_schema_and_rate():
    df = generate_synthetic.generate_synthetic_customer_data(n_rows=2_000, seed=7)

    assert list(df.columns) == [
        "customer_id",
        "tenure_months",
        "monthly_charges",
        "contract_type",
        "num_support_tickets",
        "avg_satisfaction_score",
        "customer_notes",
        "churn_label",
    ]
    assert len(df) == 2_000
    assert df["customer_id"].is_unique
    assert 0.18 <= df["churn_label"].mean() <= 0.32
    assert df["customer_notes"].str.len().gt(20).all()


def test_preprocessing_pipeline_encodes_and_scales_features():
    df = pd.DataFrame(
        {
            "customer_id": ["CUST-1", "CUST-2", "CUST-3"],
            "tenure_months": [4, 12, None],
            "monthly_charges": [95.0, 70.0, 80.0],
            "contract_type": ["month-to-month", "one-year", None],
            "num_support_tickets": [3, 1, 2],
            "avg_satisfaction_score": [4.5, None, 7.2],
            "note_sentiment_score": [-0.8, 0.4, 0.1],
            "customer_notes": ["negative note", "positive note", "neutral note"],
            "churn_label": [1, 0, 0],
        }
    )

    transformed_df, y, transformer, config = preprocessing.fit_transform_features(df)

    assert list(y) == [1, 0, 0]
    assert "contract_type_month-to-month" in transformed_df.columns
    assert "contract_type_one-year" in transformed_df.columns
    assert "tenure_months" in transformed_df.columns
    assert transformed_df.isna().sum().sum() == 0

    new_df = df.iloc[[0]].copy()
    new_df["contract_type"] = "two-year"
    transformed_new = preprocessing.transform_features(new_df, transformer, config)

    assert transformed_new.shape[1] == transformed_df.shape[1]
