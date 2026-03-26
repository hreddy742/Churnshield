import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
  ResponsiveContainer,
} from "recharts";
import type { DriftFeature } from "../types";

interface Props {
  features: DriftFeature[];
}

function getColor(status: string) {
  if (status === "stable") return "#22c55e";
  if (status === "warning") return "#f59e0b";
  return "#ef4444";
}

export default function DriftChart({ features }: Props) {
  const numeric = features.filter((f) => f.type === "numeric");

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-1">
        Population Stability Index (PSI) — lower is better
      </h3>
      <p className="text-xs text-gray-500 mb-3">
        &lt;0.1 stable · 0.1–0.2 warning · &gt;0.2 critical
      </p>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart
          data={numeric}
          layout="vertical"
          margin={{ top: 4, right: 40, left: 8, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[0, "auto"]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            width={140}
            tick={{ fontSize: 11 }}
          />
          <Tooltip formatter={(v: number) => `PSI: ${v.toFixed(4)}`} />
          <ReferenceLine x={0.1} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: "0.1", position: "top", fontSize: 10 }} />
          <ReferenceLine x={0.2} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "0.2", position: "top", fontSize: 10 }} />
          <Bar dataKey="score" radius={[0, 3, 3, 0]}>
            {numeric.map((entry, index) => (
              <Cell key={index} fill={getColor(entry.status)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
