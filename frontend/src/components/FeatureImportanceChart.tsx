import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { FeatureImportance } from "../types";

interface Props {
  data: FeatureImportance[];
}

export default function FeatureImportanceChart({ data }: Props) {
  const top10 = data.slice(0, 10);

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        Top predictors (mean |SHAP value|)
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={top10}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={(v) => v.toFixed(3)}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="display_name"
            width={150}
            tick={{ fontSize: 11 }}
          />
          <Tooltip formatter={(v: number) => v.toFixed(4)} />
          <Bar dataKey="importance" fill="#3b82f6" radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
