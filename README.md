# ChurnShield

ChurnShield is an end-to-end customer churn prediction platform for retention teams and ML engineers. It combines a FastAPI inference service, a React dashboard, and a reproducible ML pipeline that trains gradient-boosted churn models, explains predictions with SHAP, and monitors input drift.

The project is designed to demonstrate a production-style churn workflow: generate or ingest customer data, train and evaluate models, serve low-latency predictions, explain the top risk factors, score CSV batches, and expose model health metrics to an operator-facing UI.

## Features

- Real-time churn prediction from a customer profile.
- Risk tiering with low, medium, and high churn bands.
- SHAP-based prediction explanations with top contributing factors.
- Batch CSV scoring for up to 1,000 customers per upload.
- Downloadable CSV template for batch scoring.
- Model performance dashboard with ROC curves, experiment comparison, and feature importance.
- Drift monitoring using PSI for numeric features and chi-squared tests for categorical features.
- Synthetic customer data generation for local experimentation.
- Reproducible training pipeline with Logistic Regression, XGBoost, LightGBM, and an XGBoost plus LightGBM ensemble.
- Docker, Railway, and Vercel-friendly deployment configuration.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts, Axios, lucide-react |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Machine learning | scikit-learn, XGBoost, LightGBM, SHAP, MLflow, joblib |
| NLP feature extraction | Hugging Face Transformers, DistilBERT sentiment model, PyTorch |
| Data and monitoring | pandas, NumPy, SciPy, Faker |
| Database | PostgreSQL service configured for deployment/local compose; no persistence layer is currently implemented |
| Deployment | Docker, Docker Compose, Railway, Vercel-compatible frontend build |

## Architecture

```text
React dashboard
    |
    | HTTP/JSON and CSV upload
    v
FastAPI service
    |
    | loads model artifacts from ml/models
    v
Preprocessor + DistilBERT sentiment feature + XGBoost/LightGBM ensemble
    |
    | prediction, SHAP explanation, metrics, drift report
    v
API response / downloadable CSV
```

The API loads `ml/models/ensemble.pkl`, extracts a sentiment score from `customer_notes`, transforms numeric and categorical features with the saved preprocessor, averages XGBoost and LightGBM probabilities, and returns the churn probability plus a SHAP explanation from the XGBoost model.

Model monitoring is currently file-based. The drift endpoint compares a reference sample from `ml/data/raw/customer_data.csv` with a simulated recent prediction sample. PostgreSQL is configured in Docker and environment files, but the current codebase does not yet persist predictions or run database migrations.

## Project Structure

```text
.
|-- backend/
|   |-- api/
|   |   |-- main.py              # FastAPI app, CORS, router registration
|   |   |-- schemas.py           # Pydantic request/response models
|   |   `-- routes/
|   |       |-- health.py        # Health and artifact checks
|   |       |-- predict.py       # Single-customer prediction endpoint
|   |       |-- batch.py         # CSV template and batch scoring endpoints
|   |       |-- explain.py       # SHAP explanation endpoint
|   |       `-- monitor.py       # Metrics and drift endpoints
|   |-- core/config.py           # Pydantic settings
|   |-- Dockerfile               # Production API image
|   `-- requirements.txt         # Python dependencies
|-- frontend/
|   |-- src/
|   |   |-- api/client.ts        # Axios API client
|   |   |-- components/          # Forms, charts, gauge, upload UI
|   |   |-- pages/               # Predict, Model, Monitor pages
|   |   `-- types/               # Shared TypeScript interfaces
|   |-- package.json             # Frontend scripts and dependencies
|   `-- vite.config.ts           # Vite dev server and proxy config
|-- ml/
|   |-- data/generate.py         # Synthetic customer data generator
|   |-- data/raw/                # Generated reference data
|   |-- pipelines/
|   |   |-- features.py          # Preprocessing and sentiment features
|   |   |-- train.py             # Model training and artifact export
|   |   `-- evaluate.py          # SHAP feature importance export
|   |-- monitoring/drift.py      # Drift detection utilities
|   `-- models/                  # Serialized models and metric artifacts
|-- docker-compose.yml           # Local Postgres, API, and frontend services
|-- railway.json                 # Railway API deployment config
|-- Procfile                     # Process start command for platform deploys
|-- runtime.txt                  # Python runtime hint
`-- .env.example                 # Example environment variables
```

## Model Artifacts

The repository includes trained model artifacts under `ml/models/`:

| Artifact | Purpose |
| --- | --- |
| `ensemble.pkl` | Production inference bundle with XGBoost, LightGBM, preprocessor, feature names, and threshold |
| `xgb_model.pkl` | Trained XGBoost model |
| `lgb_model.pkl` | Trained LightGBM model |
| `preprocessor.pkl` | scikit-learn preprocessing pipeline |
| `training_report.json` | AUC, F1, PR AUC, sample counts, churn rate, and training timestamp |
| `roc_curve.json` | ROC curve points for frontend charts |
| `feature_importance.json` | Mean absolute SHAP feature importances |
| `feature_columns.json` | Numeric, categorical, text, and encoded feature metadata |

Current checked-in training metrics:

| Metric | Value |
| --- | ---: |
| Ensemble test AUC | 0.7757 |
| Ensemble test F1 | 0.3921 |
| Ensemble test PR AUC | 0.5675 |
| Logistic Regression validation AUC | 0.7510 |
| XGBoost validation AUC | 0.7445 |
| LightGBM validation AUC | 0.7465 |
| Training samples | 35,000 |
| Test samples | 7,500 |
| Training churn rate | 27.35% |

## Prerequisites

- Python 3.11
- Node.js 20 or newer
- npm
- Docker and Docker Compose, optional but recommended for production-like local runs

The backend should be run from the repository root so imports and relative artifact paths resolve correctly.

## Environment Variables

Copy the example file for backend/local deployment settings:

```bash
cp .env.example .env
```

| Variable | Used by | Default/example | Description |
| --- | --- | --- | --- |
| `DATABASE_URL` | Backend config, Docker/Railway environments | `postgresql://churnshield:churnshield@localhost:5432/churnshield` | PostgreSQL connection string. Configured but not actively used by the current API routes. |
| `DEBUG` | Backend config | `false` | Enables debug-style settings if wired into future backend code. |
| `VITE_API_URL` | Frontend | `http://localhost:8000` | Base URL for API requests from the React app. In local frontend development, place this in `frontend/.env.local`. |
| `PORT` | Railway/Procfile deployments | platform-provided | Runtime port used by hosted API start commands. |

For local frontend development, create a Vite env file:

```bash
printf "VITE_API_URL=http://localhost:8000\n" > frontend/.env.local
```

## Installation

### 1. Clone and create a Python environment

```bash
git clone <repository-url>
cd <repository-directory>

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Prepare model artifacts, if needed

Trained artifacts are already present in `ml/models/`. To regenerate them:

```bash
python ml/data/generate.py
PYTHONPATH=. python ml/pipelines/train.py
PYTHONPATH=. python ml/pipelines/evaluate.py
```

The first run downloads the DistilBERT sentiment model used to convert `customer_notes` into a numeric feature.

## Running Locally

### Start the API

```bash
source .venv/bin/activate
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

API URLs:

- API root: `http://localhost:8000/`
- Health check: `http://localhost:8000/health/`
- OpenAPI docs: `http://localhost:8000/docs`

### Start the frontend

```bash
cd frontend
printf "VITE_API_URL=http://localhost:8000\n" > .env.local
npm run dev
```

The Vite config uses port `3002`, so the local app runs at:

```text
http://localhost:3002
```

### Run with Docker Compose

```bash
docker compose up --build
```

Compose starts:

- PostgreSQL on `localhost:5432`
- API on `localhost:8000`
- Frontend service from the `frontend/` directory

Note: `frontend/vite.config.ts` currently sets the dev server port to `3002`, while `docker-compose.yml` maps `3000:3000`. If the frontend container is unreachable, either run the frontend locally with `npm run dev`, change the compose mapping to `3002:3002`, or override the Vite command to use port `3000`.

## API Reference

All examples assume the API is running on `http://localhost:8000`.

### Root

```http
GET /
```

Returns API metadata and the docs path.

### Health

```http
GET /health/
```

Checks whether required model artifacts are present.

### Predict churn

```http
POST /predict/
Content-Type: application/json
```

Request:

```json
{
  "tenure_months": 12,
  "monthly_charges": 75.5,
  "contract_type": "Month-to-Month",
  "num_support_tickets": 3,
  "avg_satisfaction_score": 3.2,
  "payment_method": "Electronic",
  "num_products": 2,
  "customer_notes": "Mentioned cancellation on support call"
}
```

Example:

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "contract_type": "Month-to-Month",
    "num_support_tickets": 3,
    "avg_satisfaction_score": 3.2,
    "payment_method": "Electronic",
    "num_products": 2,
    "customer_notes": "Mentioned cancellation on support call"
  }'
```

Response shape:

```json
{
  "churn_probability": 0.6321,
  "risk_tier": "high",
  "top_factors": [
    {
      "feature": "contract_type_Month-to-Month",
      "display_name": "Contract: Month-To-Month",
      "shap_value": 0.4223,
      "direction": "increases_risk"
    }
  ],
  "shap_base_value": -1.0245,
  "inference_ms": 42
}
```

### Explain prediction with SHAP

```http
POST /explain/shap
Content-Type: application/json
```

Accepts the same request body as `/predict/` and returns the SHAP base value plus feature-level SHAP values.

### Download batch template

```http
GET /batch/template
```

```bash
curl -L http://localhost:8000/batch/template -o churnshield_template.csv
```

Required CSV columns:

```text
tenure_months,monthly_charges,contract_type,num_support_tickets,avg_satisfaction_score,payment_method,num_products,customer_notes
```

### Upload batch CSV

```http
POST /batch/upload
Content-Type: multipart/form-data
```

```bash
curl -X POST http://localhost:8000/batch/upload \
  -F "file=@customers.csv" \
  -o churn_predictions.csv
```

The uploaded file must be a CSV with the required columns and at most 1,000 rows. The response is a CSV containing the original fields plus `churn_probability` and `risk_tier`.

### Get model metrics

```http
GET /monitor/metrics
```

Returns `training_report`, `roc_curve`, and `feature_importance` from `ml/models/`.

### Get drift report

```http
GET /monitor/drift
```

Returns per-feature drift scores and an overall status:

- `stable`
- `warning`
- `critical`

## Input Constraints

The prediction schema validates customer inputs before inference.

| Field | Accepted values |
| --- | --- |
| `tenure_months` | Integer from 1 to 84 |
| `monthly_charges` | Float from 20 to 120 |
| `contract_type` | `Month-to-Month`, `One Year`, `Two Year` |
| `num_support_tickets` | Integer from 0 to 15 |
| `avg_satisfaction_score` | Float from 1.0 to 5.0 |
| `payment_method` | `Electronic`, `Mailed Check`, `Bank Transfer`, `Credit Card` |
| `num_products` | Integer from 1 to 4 |
| `customer_notes` | Optional text |

## ML Workflow

Generate synthetic data:

```bash
python ml/data/generate.py
```

Train models and export artifacts:

```bash
PYTHONPATH=. python ml/pipelines/train.py
```

Compute SHAP feature importance:

```bash
PYTHONPATH=. python ml/pipelines/evaluate.py
```

View MLflow runs created during training:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5001
```

Training uses a 70/15/15 train/validation/test split. The production inference bundle averages XGBoost and LightGBM churn probabilities.

## Database and Migrations

`docker-compose.yml` provisions a PostgreSQL 16 service and `.env.example` defines `DATABASE_URL`. The current application does not include ORM models, Alembic migrations, or database-backed prediction logging. Runtime prediction, explanation, metrics, and drift behavior depend on local files under `ml/models/` and `ml/data/raw/`.

If persistence is added later, create migration tooling before relying on `DATABASE_URL` in production.
