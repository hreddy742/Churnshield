import io, time, joblib, numpy as np, pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ml.pipelines.features import extract_sentiment_scores
from pathlib import Path

router = APIRouter(prefix="/batch", tags=["batch"])

REQUIRED_COLUMNS = [
    "tenure_months", "monthly_charges", "contract_type",
    "num_support_tickets", "avg_satisfaction_score",
    "payment_method", "num_products", "customer_notes"
]

_bundle = None

def get_bundle():
    global _bundle
    if _bundle is None:
        path = Path("ml/models/ensemble.pkl")
        if not path.exists():
            raise RuntimeError("Model not found. Run: python ml/pipelines/train.py")
        _bundle = joblib.load(path)
    return _bundle

@router.post("/upload")
async def batch_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if len(df) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 rows per upload")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}"
        )

    bundle = get_bundle()

    # Fill missing notes
    df["customer_notes"] = df["customer_notes"].fillna("no notes provided")

    # Sentiment in batch
    df["sentiment_score"] = extract_sentiment_scores(df["customer_notes"].tolist())

    feature_cols = [
        "tenure_months", "monthly_charges", "num_support_tickets",
        "avg_satisfaction_score", "num_products", "sentiment_score",
        "contract_type", "payment_method"
    ]
    X = bundle["preprocessor"].transform(df[feature_cols])

    xgb_proba = bundle["xgb"].predict_proba(X)[:, 1]
    lgb_proba = bundle["lgb"].predict_proba(X)[:, 1]
    churn_proba = (xgb_proba + lgb_proba) / 2

    df["churn_probability"] = churn_proba.round(4)
    df["risk_tier"] = pd.cut(
        churn_proba,
        bins=[-0.001, 0.3, 0.6, 1.001],
        labels=["low", "medium", "high"]
    )

    # Drop internal column before returning
    out_df = df.drop(columns=["sentiment_score"])

    output = io.StringIO()
    out_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"}
    )

@router.get("/template")
def download_template():
    template_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    # Add one example row
    template_df.loc[0] = [
        12, 75.5, "Month-to-Month", 3, 3.2,
        "Electronic", 2, "Mentioned cancellation on support call"
    ]
    output = io.StringIO()
    template_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churnshield_template.csv"}
    )
