type Point = { date: string; units_sold: number };
type Forecast = {
  date: string;
  point: number;
  lower_80: number | null;
  upper_80: number | null;
  lower_95?: number | null;
  upper_95?: number | null;
};

function shortDate(date?: string) {
  if (!date) return "";
  return new Date(`${date}T00:00:00`).toLocaleDateString("en", {
    month: "short",
    day: "numeric",
    year: "2-digit",
  });
}

function formatAxis(value: number) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 0, notation: "compact" }).format(value);
}

export function DemandChart({ actual, future, loading = false }: { actual: Point[]; future: Forecast[]; loading?: boolean }) {
  const width = 920;
  const height = 286;
  const left = 52;
  const right = 13;
  const top = 14;
  const bottom = 30;
  const values = [
    ...actual.map((point) => point.units_sold),
    ...future.flatMap((point) => [point.point, point.lower_95, point.upper_95, point.lower_80, point.upper_80]
      .filter((value): value is number => value != null)),
  ];

  if (!values.length) {
    return <div className={`chart-empty${loading ? " chart-loading" : ""}`}><span className="empty-mark">⌁</span><b>{loading ? "Loading demand data" : "No observations for this series"}</b><small>{loading ? "Reading the selected series from your local database." : "Choose a different store or product to explore demand."}</small></div>;
  }

  const xCount = Math.max(2, actual.length + future.length);
  const x = (index: number) => left + index / (xCount - 1) * (width - left - right);
  const yMax = Math.max(1, Math.max(...values) * 1.12);
  const y = (value: number) => height - bottom - value / yMax * (height - top - bottom);
  const line = (points: { index: number; value: number }[]) =>
    points.map((point, index) => `${index ? "L" : "M"}${x(point.index)},${y(point.value)}`).join(" ");
  const area = (rows: Forecast[], lower: "lower_80" | "lower_95", upper: "upper_80" | "upper_95") => {
    const available = rows.map((row, i) => ({ row, index: actual.length - 1 + i }))
      .filter(({ row }) => row[lower] != null && row[upper] != null);
    if (!available.length) return "";
    const topEdge = available.map(({ row, index }, i) => `${i ? "L" : "M"}${x(index)},${y(row[upper]! )}`).join(" ");
    const bottomEdge = [...available].reverse().map(({ row, index }) => `L${x(index)},${y(row[lower]! )}`).join(" ");
    return `${topEdge} ${bottomEdge} Z`;
  };

  const actualPath = line(actual.map((point, index) => ({ index, value: point.units_sold })));
  const forecastPath = line(future.map((point, index) => ({ index: actual.length - 1 + index, value: point.point })));
  const forecastStart = x(Math.max(0, actual.length - 1));
  const plotHeight = height - top - bottom;

  return <svg className="demand-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none"
    role="img" aria-label="Daily observed demand with point forecast and 80 and 95 percent prediction intervals">
    <defs>
      <linearGradient id="demandArea" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#d9b65d" stopOpacity=".12"/><stop offset="100%" stopColor="#d9b65d" stopOpacity="0"/></linearGradient>
    </defs>
    {[0, 1, 2, 3, 4].map((tick) => {
      const value = yMax * (1 - tick / 4);
      return <g key={tick}>
        <line x1={left} x2={width - right} y1={top + tick * plotHeight / 4} y2={top + tick * plotHeight / 4} className="grid-line" />
        <text x={left - 10} y={top + tick * plotHeight / 4 + 3} textAnchor="end" className="axis-label">{formatAxis(value)}</text>
      </g>;
    })}
    {actual.length > 1 && <path d={`${actualPath} L${x(actual.length - 1)},${height - bottom} L${x(0)},${height - bottom} Z`} fill="url(#demandArea)" />}
    {area(future, "lower_95", "upper_95") && <path d={area(future, "lower_95", "upper_95")} className="forecast-band-95" />}
    {area(future, "lower_80", "upper_80") && <path d={area(future, "lower_80", "upper_80")} className="forecast-band" />}
    {future.length > 0 && <line x1={forecastStart} x2={forecastStart} y1={top} y2={height - bottom} className="forecast-divider" />}
    {actualPath && <path d={actualPath} className="actual-line" />}
    {forecastPath && <path d={forecastPath} className="forecast-line" />}
    {actual.length > 0 && <circle cx={x(actual.length - 1)} cy={y(actual.at(-1)!.units_sold)} r="3.3" className="actual-point" />}
    {future.length > 0 && <circle cx={x(actual.length - 1 + future.length - 1)} cy={y(future.at(-1)!.point)} r="3.3" className="forecast-point" />}
    <text x={left} y={height - 7} className="axis-label">{shortDate(actual[0]?.date)}</text>
    {future.length > 0 && <text x={forecastStart + 7} y={top + 12} className="forecast-label">FORECAST</text>}
    <text x={width - right} y={height - 7} textAnchor="end" className="axis-label">{shortDate(future.at(-1)?.date ?? actual.at(-1)?.date)}</text>
  </svg>;
}
