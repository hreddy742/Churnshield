# ChurnShield — Customer Churn Prediction Platform

> Predict which customers will churn and why. Real-time predictions with
> SHAP explainability, drift monitoring, and batch scoring. 100% open source.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://churnshield.vercel.app)
[![Open Source](https://img.shields.io/badge/open_source-100%25-green)]()
[![License MIT](https://img.shields.io/badge/license-MIT-blue)]()

**Live demo:** https://churnshield.vercel.app | **API docs:** https://churnshield-api.railway.app/docs

---

## The problem

Telecom and SaaS companies lose significant revenue to customer churn.
Identifying which customers are at risk allows retention teams to
intervene before cancellation. However, two problems prevent adoption:

1. **Black box predictions** — teams don't trust a score they can't explain
2. **Model drift** — predictions degrade as customer behavior changes

ChurnShield addresses both: every prediction includes a SHAP explanation
showing exactly which factors drove the score, and a drift monitor tracks
when the model's input distribution shifts.

---

## What ChurnShield does

**Single prediction:** Enter a customer's profile and get a churn
probability in milliseconds with a SHAP waterfall chart explaining the top 5 factors.

**Batch scoring:** Upload a CSV of customers and download predictions
for your entire book of business.

**Model performance:** View ROC curves, feature importance, and
experiment comparison across all trained models.

**Drift monitoring:** Track Population Stability Index (PSI) for all
features to detect when incoming data deviates from training distribution.

---

## How it works — technical deep dive

### Machine learning pipeline

ChurnShield trains three models and combines XGBoost and LightGBM
into a soft-voting ensemble:

**1. Data generation**
50,000 synthetic customer records with realistic distributions:
- Tenure: gamma distribution (shape=2.2, scale=12 months)
- Churn label: logistic function of tenure, charges, satisfaction,
  contract type, support tickets, and product count
- Target churn rate: ~27% (realistic for telecom)

**2. Feature engineering**
Standard features (tenure, charges, satisfaction) are combined with
a sentiment score extracted from customer notes using
`distilbert-base-uncased-finetuned-sst-2-english`. This model runs
locally on CPU — no API call. It converts free-text notes into a
0.0–1.0 sentiment score that serves as an additional predictive feature.

**3. Model training**
Three models are trained and tracked in MLflow:

| Model | Val AUC | Notes |
|-------|---------|-------|
| Logistic Regression | 0.751 | Baseline |
| XGBoost | 0.745 | Tree boosting, early stopping |
| LightGBM | 0.747 | Gradient boosting, early stopping |
| **Ensemble (XGB+LGB avg)** | **0.776** | **Production model (test set)** |

The ensemble improves AUC by +3.3% over the logistic regression baseline.
The synthetic notes use only 20 fixed templates, so the sentiment feature
adds limited signal here — real customer notes would push AUC higher.

**4. SHAP explainability**
After training, SHAP (SHapley Additive exPlanations) values are
computed using `shap.TreeExplainer`. For every prediction:
- Base value: average model output across training data
- SHAP values: each feature's contribution to pushing the
  prediction above or below the base value
- Positive SHAP: feature increases churn probability
- Negative SHAP: feature decreases churn probability

The waterfall chart shows these values sorted by magnitude,
so users immediately see which factors matter most.

Top predictors (mean |SHAP value|):
1. Tenure months — 0.424
2. Contract type: Month-to-Month — 0.422
3. Support tickets — 0.311
4. Monthly charges — 0.259
5. Satisfaction score — 0.257

**5. Drift detection**
Population Stability Index (PSI) measures how much a feature's
distribution has shifted compared to training:
- PSI < 0.1: stable
- PSI 0.1–0.2: moderate shift, monitor
- PSI > 0.2: significant shift, consider retraining

---

## Performance

| Metric | Value |
|--------|-------|
| Ensemble test AUC | 0.776 |
| Baseline (logistic) AUC | 0.751 |
| AUC improvement over baseline | +3.3% |
| Ensemble test F1 | 0.392 |
| Training churn rate | 27.3% |
| Training data | 50,000 records |
| Features | 6 numeric + 2 categorical + 1 text-derived |
| Single prediction latency | < 50ms |
| Batch (1,000 customers) | < 5 seconds |

---

## Open source components

| Component | Technology | License | Cost |
|-----------|-----------|---------|------|
| Gradient boosting | XGBoost | Apache 2.0 | Free |
| Gradient boosting | LightGBM | MIT | Free |
| Explainability | SHAP | MIT | Free |
| Feature engineering | scikit-learn | BSD | Free |
| Text sentiment | distilbert (local) | Apache 2.0 | Free |
| Experiment tracking | MLflow | Apache 2.0 | Free |
| Backend | FastAPI | MIT | Free |
| Frontend | React + Tailwind | MIT | Free |
| Hosting | Railway + Vercel | — | Free tier |

No LLM API. No cloud AI services. Everything runs locally.

---

## Technical decisions

**Why an ensemble instead of a single model?**
XGBoost and LightGBM use different optimization strategies
(Newton boosting vs gradient boosting with leaf-wise growth).
Averaging their probability outputs reduces variance and consistently
improves AUC over either model alone with no added complexity.

**Why SHAP TreeExplainer instead of feature importance from the model?**
Built-in feature importance (gain, coverage, frequency) measures
how often or how much a feature is used during training, not its
effect on individual predictions. SHAP values are game-theoretically
grounded: they measure each feature's actual contribution to a
specific prediction, making them useful for both global analysis
and individual explanation.

**Why local distilbert instead of a sentiment API?**
The model (67MB) downloads once and runs on CPU in ~20ms per batch.
Using a sentiment API would add network latency, cost, and a privacy
concern (sending customer notes to a third party). Local inference
is faster, cheaper, and keeps data internal.

**Why PSI for drift detection?**
PSI is the industry standard for model monitoring in banking and
insurance. It is interpretable (a number with defined thresholds),
computationally cheap (histogram comparison), and does not require
labels (unlike accuracy-based drift detection that needs ground truth).

---

## Quick start

```bash
git clone https://github.com/harshavardhan/churnshield
cd churnshield

# Generate training data
python ml/data/generate.py

# Train models (takes 5-10 min on first run — downloads distilbert once)
PYTHONPATH=. python ml/pipelines/train.py

# Compute SHAP feature importance
PYTHONPATH=. python ml/pipelines/evaluate.py

# Start API and frontend
cp .env.example .env
docker compose up

open http://localhost:3000
```

---

## Reproducing the training results

```bash
# 1. Generate 50,000 customer records
python ml/data/generate.py
# Output: ml/data/raw/customer_data.csv (churn rate ~27%)

# 2. Train all models
PYTHONPATH=. python ml/pipelines/train.py
# Output: ml/models/ (ensemble.pkl, xgb_model.pkl, training_report.json, ...)
# View experiments: mlflow ui --port 5001

# 3. Compute SHAP feature importance
PYTHONPATH=. python ml/pipelines/evaluate.py
# Output: ml/models/feature_importance.json
```

Expected output:
```
LogReg val AUC: 0.7510
XGBoost val AUC: 0.7445
LightGBM val AUC: 0.7465
Ensemble test AUC: 0.7757
Ensemble test F1: 0.3921
```

---

Built by [Harsha Vardhan Reddy](https://linkedin.com/in/hvreddy)
AI/ML Engineer · Birmingham, AL
