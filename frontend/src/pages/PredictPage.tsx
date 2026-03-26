import { useState } from "react";
import CustomerForm from "../components/CustomerForm";
import PredictionGauge from "../components/PredictionGauge";
import ShapChart from "../components/ShapChart";
import BatchUpload from "../components/BatchUpload";
import { predict } from "../api/client";
import type { CustomerInput, PredictionResult } from "../types";

const RECOMMENDATIONS = {
  high: "Recommend immediate outreach from retention team — this customer shows multiple high-risk signals.",
  medium: "Schedule a check-in call within 2 weeks to address concerns before they escalate.",
  low: "Continue standard engagement. No immediate intervention required.",
};

const RECOMMENDATION_COLORS = {
  high: "bg-red-50 border-red-200 text-red-800",
  medium: "bg-amber-50 border-amber-200 text-amber-800",
  low: "bg-green-50 border-green-200 text-green-800",
};

export default function PredictPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (data: CustomerInput) => {
    setLoading(true);
    setError(null);
    try {
      const res = await predict(data);
      setResult(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || "Prediction failed. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Churn Risk Prediction</h1>
        <p className="text-gray-500 mt-1">Enter a customer profile to get a real-time churn probability with SHAP explanation.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* LEFT — Form */}
        <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
          <CustomerForm onSubmit={handleSubmit} loading={loading} />
        </div>

        {/* RIGHT — Results */}
        <div className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">
              {error}
            </div>
          )}

          {result ? (
            <>
              <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                <PredictionGauge
                  probability={result.churn_probability}
                  riskTier={result.risk_tier as any}
                />
                <p className="text-center text-xs text-gray-400 mt-3">
                  Inference time: {result.inference_ms}ms
                </p>
              </div>

              <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
                <ShapChart factors={result.top_factors} />
              </div>

              <div className={`rounded-xl border p-4 text-sm font-medium ${RECOMMENDATION_COLORS[result.risk_tier as keyof typeof RECOMMENDATION_COLORS]}`}>
                <span className="font-semibold">Recommendation: </span>
                {RECOMMENDATIONS[result.risk_tier as keyof typeof RECOMMENDATIONS]}
              </div>
            </>
          ) : (
            <div className="bg-gray-50 rounded-2xl border border-dashed border-gray-200 p-12 text-center text-gray-400">
              <p className="text-sm">Fill in the customer profile and click<br /><strong>Predict Churn Risk</strong> to see results</p>
            </div>
          )}
        </div>
      </div>

      {/* Batch Upload */}
      <div className="mt-12">
        <BatchUpload />
      </div>
    </div>
  );
}
