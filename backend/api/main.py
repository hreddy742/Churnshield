from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import predict, batch, monitor, health, explain

app = FastAPI(
    title="ChurnShield API",
    description="Customer churn prediction with SHAP explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://churnshield.vercel.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(batch.router)
app.include_router(monitor.router)
app.include_router(explain.router)

@app.get("/")
def root():
    return {
        "name": "ChurnShield API",
        "version": "1.0.0",
        "docs": "/docs"
    }
