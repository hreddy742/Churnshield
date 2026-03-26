import { useEffect, useState } from "react";
import { getMetrics } from "../api/client";
import type { MetricsResponse } from "../types";
import RocCurveChart from "../components/RocCurveChart";
import FeatureImportanceChart from "../components/FeatureImportanceChart";

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500">{label}</p>
      <p className="text-3xl font-bold text-gray-900 mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  );
}

export default function ModelPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch(() => setError("Could not load metrics. Run training first."));
  }, []);

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700">{error}</div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8 text-gray-500 text-sm">Loading metrics...</div>
    );
  }

  const { training_report: r, roc_curve, feature_importance } = metrics;

  const experiments = [
    { model: "Logistic Regression", auc: r.logistic_regression_val_auc, notes: "Baseline — simple, interpretable" },
    { model: "XGBoost", auc: r.xgboost_val_auc, notes: "Newton boosting, early stopping" },
    { model: "LightGBM", auc: r.lightgbm_val_auc, notes: "Leaf-wise growth, early stopping" },
    { model: "Ensemble (XGB+LGB)", auc: r.ensemble_test_auc, notes: "Soft-voting average — production model" },
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Model Performance</h1>
        <p className="text-gray-500 mt-1">
          Metrics from training on {r.training_samples.toLocaleString()} synthetic customer records
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Ensemble AUC"
          value={r.ensemble_test_auc.toFixed(3)}
          sub="Test set"
        />
        <MetricCard
          label="Baseline AUC"
          value={r.logistic_regression_val_auc.toFixed(3)}
          sub="Logistic Regression"
        />
        <MetricCard
          label="F1 Score"
          value={r.ensemble_test_f1.toFixed(3)}
          sub="Ensemble test"
        />
        <MetricCard
          label="Churn Rate"
          value={`${(r.churn_rate * 100).toFixed(1)}%`}
          sub="Training data"
        />
      </div>

      {/* ROC Curve */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm mb-6">
        <h2 className="text-base font-semibold text-gray-800 mb-4">ROC Curve</h2>
        <RocCurveChart data={roc_curve} />
      </div>

      {/* Feature Importance */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm mb-6">
        <FeatureImportanceChart data={feature_importance} />
      </div>

      {/* Experiment Table */}
      <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
        <h2 className="text-base font-semibold text-gray-800 mb-4">Experiment Comparison</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs uppercase tracking-wider text-gray-500 border-b border-gray-100">
              <th className="pb-3 text-left font-semibold">Model</th>
              <th className="pb-3 text-left font-semibold">Val/Test AUC</th>
              <th className="pb-3 text-left font-semibold">Notes</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((row, i) => (
              <tr key={i} className={`border-b border-gray-50 ${i % 2 === 0 ? "bg-gray-50/50" : ""}`}>
                <td className="py-3 font-medium text-gray-800">{row.model}</td>
                <td className="py-3 font-mono text-blue-700">{row.auc.toFixed(4)}</td>
                <td className="py-3 text-gray-500">{row.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs text-gray-400 mt-4">
          Trained at: {new Date(r.trained_at).toLocaleString()}
        </p>
      </div>
    </div>
  );
}
