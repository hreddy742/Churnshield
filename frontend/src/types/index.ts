export interface CustomerInput {
  tenure_months: number;
  monthly_charges: number;
  contract_type: "Month-to-Month" | "One Year" | "Two Year";
  num_support_tickets: number;
  avg_satisfaction_score: number;
  payment_method: "Electronic" | "Mailed Check" | "Bank Transfer" | "Credit Card";
  num_products: number;
  customer_notes?: string;
}

export interface FactorContribution {
  feature: string;
  display_name: string;
  shap_value: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface PredictionResult {
  churn_probability: number;
  risk_tier: "low" | "medium" | "high";
  top_factors: FactorContribution[];
  shap_base_value: number;
  inference_ms: number;
}

export interface DriftFeature {
  feature: string;
  type: "numeric" | "categorical";
  score: number;
  pvalue: number | null;
  status: "stable" | "warning" | "critical";
}

export interface DriftReport {
  computed_at: string;
  features: DriftFeature[];
  overall_status: "stable" | "warning" | "critical";
  predictions_analyzed: number;
}

export interface TrainingReport {
  logistic_regression_val_auc: number;
  xgboost_val_auc: number;
  lightgbm_val_auc: number;
  ensemble_test_auc: number;
  ensemble_test_f1: number;
  ensemble_test_pr_auc: number;
  training_samples: number;
  test_samples: number;
  churn_rate: number;
  trained_at: string;
}

export interface RocPoint {
  fpr: number[];
  tpr: number[];
  auc: number;
}

export interface RocCurveData {
  ensemble: RocPoint;
  logistic_regression: RocPoint;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  display_name: string;
}

export interface MetricsResponse {
  training_report: TrainingReport;
  roc_curve: RocCurveData;
  feature_importance: FeatureImportance[];
}
