# ChurnGuard

Production-oriented customer churn ML pipeline with synthetic data generation, preprocessing, explainability, monitoring, and retraining workflows.

## Prerequisites

Install dependencies:

```
pip install -r requirements.txt
```

## Quickstart

**Step 1 — Generate synthetic training data (required before training):**

```
python data/generate_synthetic.py
```

**Step 2 — Train models:**

```
python pipelines/training.py
```

**Step 3 — Evaluate:**

```
python pipelines/evaluation.py
```
