import { useEffect, useState, useCallback } from "react";
import { getDrift } from "../api/client";
import type { DriftReport } from "../types";
import DriftChart from "../components/DriftChart";
import { RefreshCw } from "lucide-react";

const STATUS_COLORS = {
  stable: "bg-green-100 text-green-800",
  warning: "bg-amber-100 text-amber-800",
  critical: "bg-red-100 text-red-800",
};

const INTERPRETATIONS: Record<string, string> = {
  stable: "No significant shift from training distribution.",
  warning: "Moderate distribution shift detected. Monitor closely.",
  critical: "Significant drift — consider retraining the model.",
};

function psiInterpretation(score: number): string {
  if (score < 0.1) return "No significant change";
  if (score < 0.2) return "Moderate shift — monitor";
  return "Major shift — action required";
}

export default function MonitorPage() {
  const [report, setReport] = useState<DriftReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getDrift();
      setReport(data);
      setLastChecked(new Date());
    } catch {
      // silently fail — show stale data
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const counts = report
    ? report.features.reduce(
        (acc, f) => ({ ...acc, [f.status]: (acc[f.status] || 0) + 1 }),
        { stable: 0, warning: 0, critical: 0 } as Record<string, number>
      )
    : null;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Data Drift Monitor</h1>
          <p className="text-gray-500 mt-1">
            Comparing recent predictions to training distribution
          </p>
        </div>
        <button
          onClick={refresh}
          disabled={loading}
          className="flex items-center gap-2 text-sm font-medium text-gray-600 border border-gray-200 px-4 py-2 rounded-lg hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {lastChecked && (
        <p className="text-xs text-gray-400 mb-6">
          Last checked: {lastChecked.toLocaleTimeString()}
        </p>
      )}

      {report && counts && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-3 gap-4 mb-8">
            {(["stable", "warning", "critical"] as const).map((status) => (
              <div key={status} className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
                <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-1">
                  {status}
                </p>
                <p className="text-3xl font-bold text-gray-900">{counts[status]}</p>
                <p className="text-xs text-gray-400 mt-1">features</p>
                <span className={`mt-2 inline-block px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_COLORS[status]}`}>
                  {status}
                </span>
              </div>
            ))}
          </div>

          {/* PSI Chart */}
          <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm mb-6">
            <DriftChart features={report.features} />
          </div>

          {/* Drift Table */}
          <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
            <h2 className="text-base font-semibold text-gray-800 mb-4">Feature Drift Details</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs uppercase tracking-wider text-gray-500 border-b border-gray-100">
                    <th className="pb-3 text-left font-semibold">Feature</th>
                    <th className="pb-3 text-left font-semibold">Type</th>
                    <th className="pb-3 text-left font-semibold">Score</th>
                    <th className="pb-3 text-left font-semibold">Status</th>
                    <th className="pb-3 text-left font-semibold">Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {report.features.map((f, i) => (
                    <tr key={i} className={`border-b border-gray-50 ${i % 2 === 0 ? "bg-gray-50/50" : ""}`}>
                      <td className="py-3 font-medium text-gray-800">{f.feature.replace(/_/g, " ")}</td>
                      <td className="py-3 text-gray-500">{f.type}</td>
                      <td className="py-3 font-mono text-gray-700">
                        {f.score.toFixed(4)}
                        {f.pvalue !== null && (
                          <span className="text-xs text-gray-400 ml-1">(p={f.pvalue?.toFixed(3)})</span>
                        )}
                      </td>
                      <td className="py-3">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${STATUS_COLORS[f.status]}`}>
                          {f.status}
                        </span>
                      </td>
                      <td className="py-3 text-gray-500 text-xs">
                        {f.type === "numeric"
                          ? psiInterpretation(f.score)
                          : INTERPRETATIONS[f.status]}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-400 mt-4">
              {report.predictions_analyzed} predictions analyzed
            </p>
          </div>
        </>
      )}

      {!report && !loading && (
        <div className="text-center text-gray-400 py-16">Could not load drift report.</div>
      )}
    </div>
  );
}
