"""Reusable sklearn preprocessing pipeline for churn modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "churn_label"
TEXT_COLUMN = "customer_notes"
ID_COLUMN = "customer_id"
DEFAULT_NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "num_support_tickets",
    "avg_satisfaction_score",
    "note_sentiment_score",
]
DEFAULT_CATEGORICAL_FEATURES = ["contract_type"]


@dataclass(frozen=True)
class PreprocessingConfig:
    numeric_features: list[str]
    categorical_features: list[str]
    passthrough_features: list[str]
    target_column: str = TARGET_COLUMN
    text_column: str = TEXT_COLUMN
    id_column: str = ID_COLUMN


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def infer_preprocessing_config(df: pd.DataFrame) -> PreprocessingConfig:
    numeric_features = _existing_columns(df, DEFAULT_NUMERIC_FEATURES)
    categorical_features = _existing_columns(df, DEFAULT_CATEGORICAL_FEATURES)
    passthrough_features = [
        column
        for column in df.columns
        if column not in set(numeric_features + categorical_features + [TARGET_COLUMN, TEXT_COLUMN, ID_COLUMN])
    ]
    return PreprocessingConfig(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        passthrough_features=passthrough_features,
    )


def build_preprocessing_pipeline(config: PreprocessingConfig) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if config.numeric_features:
        transformers.append(("numeric", numeric_pipeline, config.numeric_features))
    if config.categorical_features:
        transformers.append(("categorical", categorical_pipeline, config.categorical_features))

    remainder = "drop"
    if config.passthrough_features:
        remainder = "passthrough"

    return ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        verbose_feature_names_out=False,
    )


def prepare_model_inputs(df: pd.DataFrame, config: PreprocessingConfig | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    active_config = config or infer_preprocessing_config(df)
    X = df.drop(columns=[column for column in [active_config.target_column] if column in df.columns]).copy()
    X = X.drop(columns=[column for column in [active_config.text_column, active_config.id_column] if column in X.columns])
    y = df[active_config.target_column].copy() if active_config.target_column in df.columns else None
    return X, y


def fit_transform_features(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series | None, ColumnTransformer, PreprocessingConfig]:
    active_config = config or infer_preprocessing_config(df)
    X, y = prepare_model_inputs(df, active_config)
    transformer = build_preprocessing_pipeline(active_config)
    transformed = transformer.fit_transform(X)
    transformed_df = pd.DataFrame(
        transformed,
        columns=transformer.get_feature_names_out(),
        index=df.index,
    )
    return transformed_df, y, transformer, active_config


def transform_features(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    X, _ = prepare_model_inputs(df, config)
    transformed = transformer.transform(X)
    return pd.DataFrame(
        transformed,
        columns=transformer.get_feature_names_out(),
        index=df.index,
    )
