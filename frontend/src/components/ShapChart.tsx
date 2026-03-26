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
import type { FactorContribution } from "../types";

interface Props {
  factors: FactorContribution[];
}

export default function ShapChart({ factors }: Props) {
  const data = [...factors]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .map((f) => ({
      name: f.display_name,
      value: f.shap_value,
      direction: f.direction,
    }));

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        What&apos;s driving this prediction
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={["auto", "auto"]}
            tickFormatter={(v) => v.toFixed(2)}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={130}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: number, _name: string, props: any) => [
              `${value > 0 ? "+" : ""}${value.toFixed(4)}`,
              props.payload.direction === "increases_risk"
                ? "Increases risk"
                : "Decreases risk",
            ]}
          />
          <ReferenceLine x={0} stroke="#9ca3af" />
          <Bar dataKey="value" radius={[0, 3, 3, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={index}
                fill={entry.direction === "increases_risk" ? "#ef4444" : "#22c55e"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
