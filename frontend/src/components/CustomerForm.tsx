import { useState } from "react";
import type { CustomerInput } from "../types";

const HIGH_RISK_EXAMPLE: CustomerInput = {
  tenure_months: 4,
  monthly_charges: 95,
  contract_type: "Month-to-Month",
  num_support_tickets: 8,
  avg_satisfaction_score: 1.5,
  payment_method: "Electronic",
  num_products: 1,
  customer_notes: "Mentioned cancellation on support call",
};

const NOTE_PLACEHOLDERS = [
  "Very satisfied with the service",
  "Mentioned cancellation on support call",
  "Asked about competitor pricing during last call",
];

interface Props {
  onSubmit: (data: CustomerInput) => void;
  loading: boolean;
}

export default function CustomerForm({ onSubmit, loading }: Props) {
  const [form, setForm] = useState<CustomerInput>({
    tenure_months: 24,
    monthly_charges: 65,
    contract_type: "Month-to-Month",
    num_support_tickets: 2,
    avg_satisfaction_score: 3.5,
    payment_method: "Electronic",
    num_products: 2,
    customer_notes: "",
  });

  const [placeholderIdx, setPlaceholderIdx] = useState(0);

  const set = (key: keyof CustomerInput, value: any) =>
    setForm((f) => ({ ...f, [key]: value }));

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(form);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Account Details */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-4">
          Account Details
        </h3>
        <div className="space-y-4">
          {/* Tenure */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Tenure <span className="text-blue-600 font-semibold">{form.tenure_months} months</span>
            </label>
            <input
              type="range" min={1} max={84} value={form.tenure_months}
              onChange={(e) => set("tenure_months", Number(e.target.value))}
              className="w-full accent-blue-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>1 mo</span><span>84 mo</span>
            </div>
          </div>

          {/* Monthly Charges */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Monthly Charges <span className="text-blue-600 font-semibold">${form.monthly_charges}</span>
            </label>
            <input
              type="range" min={20} max={120} step={0.5} value={form.monthly_charges}
              onChange={(e) => set("monthly_charges", Number(e.target.value))}
              className="w-full accent-blue-600"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>$20</span><span>$120</span>
            </div>
          </div>

          {/* Contract Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Contract Type</label>
            <div className="flex rounded-lg border border-gray-200 overflow-hidden">
              {(["Month-to-Month", "One Year", "Two Year"] as const).map((c) => (
                <button
                  key={c} type="button"
                  onClick={() => set("contract_type", c)}
                  className={`flex-1 py-2 text-xs font-medium transition-colors ${
                    form.contract_type === c
                      ? "bg-blue-600 text-white"
                      : "bg-white text-gray-600 hover:bg-gray-50"
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          {/* Number of Products */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Number of Products</label>
            <div className="flex gap-3">
              {[1, 2, 3, 4].map((n) => (
                <label key={n} className="flex items-center gap-1 cursor-pointer">
                  <input
                    type="radio" name="products" value={n}
                    checked={form.num_products === n}
                    onChange={() => set("num_products", n)}
                    className="accent-blue-600"
                  />
                  <span className="text-sm text-gray-700">{n}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Behavior Signals */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-4">
          Behavior Signals
        </h3>
        <div className="space-y-4">
          {/* Support Tickets */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Support Tickets (this period)
            </label>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => set("num_support_tickets", Math.max(0, form.num_support_tickets - 1))}
                className="w-8 h-8 rounded-full border border-gray-300 text-gray-600 hover:bg-gray-100 flex items-center justify-center font-bold"
              >−</button>
              <span className="w-8 text-center font-semibold text-gray-800">{form.num_support_tickets}</span>
              <button
                type="button"
                onClick={() => set("num_support_tickets", Math.min(15, form.num_support_tickets + 1))}
                className="w-8 h-8 rounded-full border border-gray-300 text-gray-600 hover:bg-gray-100 flex items-center justify-center font-bold"
              >+</button>
            </div>
          </div>

          {/* Satisfaction Score */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Satisfaction Score
            </label>
            <div className="flex gap-1">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star} type="button"
                  onClick={() => set("avg_satisfaction_score", star)}
                  className={`text-2xl transition-colors ${
                    form.avg_satisfaction_score >= star ? "text-amber-400" : "text-gray-300"
                  }`}
                >★</button>
              ))}
              <span className="ml-2 text-sm text-gray-500 self-center">
                {form.avg_satisfaction_score}/5
              </span>
            </div>
          </div>

          {/* Payment Method */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Payment Method
            </label>
            <select
              value={form.payment_method}
              onChange={(e) => set("payment_method", e.target.value)}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {["Electronic", "Mailed Check", "Bank Transfer", "Credit Card"].map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Customer Notes */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">
          Customer Notes
        </h3>
        <textarea
          value={form.customer_notes}
          onChange={(e) => set("customer_notes", e.target.value)}
          onFocus={() => setPlaceholderIdx((i) => (i + 1) % NOTE_PLACEHOLDERS.length)}
          placeholder={NOTE_PLACEHOLDERS[placeholderIdx]}
          rows={3}
          className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-2">
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-60 flex items-center justify-center gap-2"
        >
          {loading && (
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
          )}
          {loading ? "Analyzing..." : "Predict Churn Risk"}
        </button>
        <button
          type="button"
          onClick={() => setForm(HIGH_RISK_EXAMPLE)}
          className="w-full py-2.5 border border-red-200 text-red-600 font-medium rounded-xl hover:bg-red-50 transition-colors text-sm"
        >
          Use High-Risk Example
        </button>
      </div>
    </form>
  );
}
