import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from transformers import pipeline as hf_pipeline
import joblib

NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "num_support_tickets",
    "avg_satisfaction_score", "num_products", "sentiment_score"
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
TARGET = "churn_label"
EXCLUDE = ["customer_id", "customer_notes", "churn_label"]

_sentiment_pipe = None

def get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        print("Loading sentiment model (first time only)...")
        _sentiment_pipe = hf_pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
            max_length=128
        )
    return _sentiment_pipe

def extract_sentiment_scores(texts: list) -> list:
    pipe = get_sentiment_pipe()
    # Replace empty/null with neutral placeholder
    clean = [t if isinstance(t, str) and len(t.strip()) > 0
             else "neutral experience" for t in texts]
    results = pipe(clean, batch_size=64, truncation=True)
    return [
        r["score"] if r["label"] == "POSITIVE" else 1.0 - r["score"]
        for r in results
    ]

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])

def prepare_features(df: pd.DataFrame):
    df = df.copy()
    print("Extracting sentiment features...")
    df["sentiment_score"] = extract_sentiment_scores(
        df["customer_notes"].tolist()
    )
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return X, y
