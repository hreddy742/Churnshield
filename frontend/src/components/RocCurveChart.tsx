import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { RocCurveData } from "../types";

interface Props {
  data: RocCurveData;
}

export default function RocCurveChart({ data }: Props) {
  const ensemblePoints = data.ensemble.fpr.map((fpr, i) => ({
    fpr,
    ensemble_tpr: data.ensemble.tpr[i],
  }));

  const lrPoints = data.logistic_regression.fpr.map((fpr, i) => ({
    fpr,
    lr_tpr: data.logistic_regression.tpr[i],
  }));

  // Merge by fpr index
  const maxLen = Math.max(ensemblePoints.length, lrPoints.length);
  const combined = Array.from({ length: maxLen }, (_, i) => ({
    fpr: ensemblePoints[i]?.fpr ?? lrPoints[i]?.fpr ?? i / maxLen,
    ensemble_tpr: ensemblePoints[i]?.ensemble_tpr,
    lr_tpr: lrPoints[i]?.lr_tpr,
  }));

  // Diagonal reference points
  const diagonal = [
    { fpr: 0, diag: 0 },
    { fpr: 1, diag: 1 },
  ];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="fpr"
          type="number"
          domain={[0, 1]}
          label={{ value: "False Positive Rate", position: "insideBottom", offset: -4, fontSize: 12 }}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          domain={[0, 1]}
          label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fontSize: 12 }}
          tick={{ fontSize: 11 }}
        />
        <Tooltip
          formatter={(val: number) => val?.toFixed(3)}
          labelFormatter={(label) => `FPR: ${Number(label).toFixed(3)}`}
        />
        <Legend />
        {/* Diagonal random classifier */}
        <Line
          data={diagonal}
          dataKey="diag"
          stroke="#d1d5db"
          strokeDasharray="4 4"
          dot={false}
          name="Random (AUC=0.50)"
        />
        {/* Logistic Regression */}
        <Line
          data={combined}
          dataKey="lr_tpr"
          stroke="#9ca3af"
          strokeDasharray="6 3"
          dot={false}
          name={`Logistic Regression (AUC=${data.logistic_regression.auc})`}
        />
        {/* Ensemble */}
        <Line
          data={combined}
          dataKey="ensemble_tpr"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name={`Ensemble (AUC=${data.ensemble.auc})`}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
