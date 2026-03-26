from fastapi import APIRouter
from pathlib import Path

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
def health():
    models_dir = Path("ml/models")
    artifacts = ["ensemble.pkl", "training_report.json", "feature_importance.json"]
    model_status = {a: (models_dir / a).exists() for a in artifacts}

    return {
        "status": "ok",
        "models_loaded": all(model_status.values()),
        "artifacts": model_status
    }
