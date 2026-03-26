import mlflow, mlflow.sklearn
import pandas as pd, numpy as np, json, joblib, os
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from ml.pipelines.features import prepare_features, build_preprocessor

def train():
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("churnshield")

    print("Loading data...")
    df = pd.read_csv("ml/data/raw/customer_data.csv")
    X_raw, y = prepare_features(df)

    print("Building preprocessor...")
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_raw)

    # Get feature names after preprocessing
    ohe_cats = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(
        ["contract_type", "payment_method"]
    )
    feature_names = [
        "tenure_months", "monthly_charges", "num_support_tickets",
        "avg_satisfaction_score", "num_products", "sentiment_score"
    ] + list(ohe_cats)

    # Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    results = {}

    # 1. Logistic Regression baseline
    with mlflow.start_run(run_name="logistic_regression"):
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr.fit(X_train, y_train)
        val_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:,1])
        mlflow.log_params({"C": 1.0, "max_iter": 1000})
        mlflow.log_metric("val_auc", val_auc)
        results["lr"] = {"model": lr, "val_auc": val_auc}
        print(f"LogReg val AUC: {val_auc:.4f}")

    # 2. XGBoost
    with mlflow.start_run(run_name="xgboost"):
        xgb = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            eval_metric="logloss", random_state=42, verbosity=0,
            early_stopping_rounds=30
        )
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_auc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1])
        mlflow.log_params(xgb.get_params())
        mlflow.log_metric("val_auc", val_auc)
        results["xgb"] = {"model": xgb, "val_auc": val_auc}
        print(f"XGBoost val AUC: {val_auc:.4f}")

    # 3. LightGBM
    with mlflow.start_run(run_name="lightgbm"):
        lgb = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
            random_state=42, verbose=-1
        )
        lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                __import__("lightgbm").early_stopping(30, verbose=False)
            ]
        )
        val_auc = roc_auc_score(y_val, lgb.predict_proba(X_val)[:,1])
        mlflow.log_metric("val_auc", val_auc)
        results["lgb"] = {"model": lgb, "val_auc": val_auc}
        print(f"LightGBM val AUC: {val_auc:.4f}")

    # 4. Ensemble
    with mlflow.start_run(run_name="ensemble"):
        def ensemble_proba(X):
            return (
                results["xgb"]["model"].predict_proba(X)[:,1] +
                results["lgb"]["model"].predict_proba(X)[:,1]
            ) / 2

        test_proba = ensemble_proba(X_test)
        test_pred = (test_proba > 0.5).astype(int)
        test_auc = roc_auc_score(y_test, test_proba)
        test_f1 = f1_score(y_test, test_pred)
        test_pr_auc = average_precision_score(y_test, test_proba)

        mlflow.log_metrics({
            "test_auc": test_auc, "test_f1": test_f1,
            "test_pr_auc": test_pr_auc
        })
        print(f"\nEnsemble test AUC: {test_auc:.4f}")
        print(f"Ensemble test F1: {test_f1:.4f}")

        # ROC curve data for frontend chart
        fpr, tpr, _ = roc_curve(y_test, test_proba)
        lr_proba = results["lr"]["model"].predict_proba(X_test)[:,1]
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)

    # Save all artifacts
    models_dir = Path("ml/models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(results["xgb"]["model"], models_dir / "xgb_model.pkl")
    joblib.dump(results["lgb"]["model"], models_dir / "lgb_model.pkl")
    joblib.dump(preprocessor, models_dir / "preprocessor.pkl")

    # Ensemble bundle
    bundle = {
        "xgb": results["xgb"]["model"],
        "lgb": results["lgb"]["model"],
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "threshold": 0.5,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    joblib.dump(bundle, models_dir / "ensemble.pkl")

    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump({
            "numeric": [
                "tenure_months", "monthly_charges", "num_support_tickets",
                "avg_satisfaction_score", "num_products"
            ],
            "categorical": ["contract_type", "payment_method"],
            "text": ["customer_notes"],
            "all_features": feature_names
        }, f, indent=2)

    with open(models_dir / "training_report.json", "w") as f:
        json.dump({
            "logistic_regression_val_auc": round(results["lr"]["val_auc"], 4),
            "xgboost_val_auc": round(results["xgb"]["val_auc"], 4),
            "lightgbm_val_auc": round(results["lgb"]["val_auc"], 4),
            "ensemble_test_auc": round(test_auc, 4),
            "ensemble_test_f1": round(test_f1, 4),
            "ensemble_test_pr_auc": round(test_pr_auc, 4),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "churn_rate": round(float(y.mean()), 4),
            "trained_at": datetime.now(timezone.utc).isoformat()
        }, f, indent=2)

    with open(models_dir / "roc_curve.json", "w") as f:
        json.dump({
            "ensemble": {
                "fpr": [round(x, 4) for x in fpr.tolist()[::5]],
                "tpr": [round(x, 4) for x in tpr.tolist()[::5]],
                "auc": round(test_auc, 4)
            },
            "logistic_regression": {
                "fpr": [round(x, 4) for x in lr_fpr.tolist()[::5]],
                "tpr": [round(x, 4) for x in lr_tpr.tolist()[::5]],
                "auc": round(roc_auc_score(y_test, lr_proba), 4)
            }
        }, f, indent=2)

    print(f"\n{'='*40}")
    print("TRAINING COMPLETE")
    print(f"Baseline (LogReg) Val AUC: {results['lr']['val_auc']:.4f}")
    print(f"XGBoost Val AUC:           {results['xgb']['val_auc']:.4f}")
    print(f"LightGBM Val AUC:          {results['lgb']['val_auc']:.4f}")
    print(f"Ensemble Test AUC:         {test_auc:.4f}")
    print(f"Ensemble Test F1:          {test_f1:.4f}")
    print(f"{'='*40}")
    print("Artifacts saved to ml/models/")
    print("NEXT: commit ml/models/ to git")

if __name__ == "__main__":
    train()
