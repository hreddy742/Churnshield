import shap, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from ml.pipelines.features import prepare_features

def evaluate():
    models_dir = Path("ml/models")
    bundle = joblib.load(models_dir / "ensemble.pkl")
    xgb = bundle["xgb"]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]

    df = pd.read_csv("ml/data/raw/customer_data.csv")
    X_raw, y = prepare_features(df)
    X = preprocessor.transform(X_raw)

    # SHAP on 2000 test samples
    sample_idx = np.random.choice(len(X), size=2000, replace=False)
    X_sample = X[sample_idx]

    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_sample)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = [
        {"feature": name, "importance": round(float(imp), 4),
         "display_name": name.replace("_", " ").title()}
        for name, imp in sorted(
            zip(feature_names, mean_abs_shap),
            key=lambda x: x[1], reverse=True
        )
    ]

    with open(models_dir / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    print("Feature importance saved.")
    print("Top 5 features:")
    for item in feature_importance[:5]:
        print(f"  {item['display_name']}: {item['importance']:.4f}")

if __name__ == "__main__":
    evaluate()
