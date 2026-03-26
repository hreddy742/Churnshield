import { useEffect, useState } from "react";

interface Props {
  probability: number; // 0-1
  riskTier: "low" | "medium" | "high";
}

export default function PredictionGauge({ probability, riskTier }: Props) {
  const [animated, setAnimated] = useState(0);

  useEffect(() => {
    setAnimated(0);
    const start = performance.now();
    const duration = 800;
    const frame = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      setAnimated(probability * ease);
      if (progress < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }, [probability]);

  const pct = Math.round(probability * 100);
  const displayPct = Math.round(animated * 100);

  // SVG half-circle gauge
  const R = 80;
  const cx = 100;
  const cy = 100;
  const strokeWidth = 16;
  const circumference = Math.PI * R; // half circle

  // Color zones
  const getColor = (p: number) => {
    if (p < 0.3) return "#22c55e";
    if (p < 0.6) return "#f59e0b";
    return "#ef4444";
  };

  const tierColors = {
    low: "bg-green-100 text-green-800",
    medium: "bg-amber-100 text-amber-800",
    high: "bg-red-100 text-red-800",
  };

  const tierLabels = {
    low: "LOW RISK",
    medium: "MEDIUM RISK",
    high: "HIGH RISK",
  };

  const activeColor = getColor(probability);
  const progress = animated * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width="200" height="120" viewBox="0 0 200 120">
        {/* Background track */}
        <path
          d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Animated progress arc */}
        <path
          d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`}
          fill="none"
          stroke={activeColor}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${progress} ${circumference}`}
          style={{ transition: "stroke 0.3s ease" }}
        />
        {/* Percentage text */}
        <text
          x={cx}
          y={cy - 8}
          textAnchor="middle"
          fontSize="32"
          fontWeight="700"
          fill="#111827"
        >
          {displayPct}%
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fontSize="12" fill="#6b7280">
          churn probability
        </text>
        {/* Zone labels */}
        <text x={cx - R + 2} y={cy + 20} fontSize="10" fill="#6b7280">0%</text>
        <text x={cx + R - 14} y={cy + 20} fontSize="10" fill="#6b7280">100%</text>
      </svg>

      <span
        className={`mt-2 px-3 py-1 rounded-full text-sm font-semibold ${tierColors[riskTier]}`}
      >
        {tierLabels[riskTier]}
      </span>
    </div>
  );
}
