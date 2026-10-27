type Point = { date: string; units_sold: number };
type Forecast = { date: string; point: number; lower_80: number | null; upper_80: number | null };

export function DemandChart({ actual, future }: { actual: Point[]; future: Forecast[] }) {
  const w = 900, h = 260, pad = 16, values = [...actual.map((v) => v.units_sold), ...future.map((v) => v.point)];
  if (!values.length) return <div className="chart-empty">No demand history for this series yet.</div>;
  const lo = Math.min(0, ...values), hi = Math.max(1, ...values) * 1.12, n = Math.max(2, actual.length + future.length);
  const x = (i: number) => pad + i / (n - 1) * (w - pad * 2), y = (v: number) => h - pad - (v - lo) / (hi - lo || 1) * (h - pad * 2);
  const line = (pts: { i: number; v: number }[]) => pts.map((p, j) => `${j ? "L" : "M"}${x(p.i)},${y(p.v)}`).join(" ");
  const actualLine = line(actual.map((p, i) => ({ i, v: p.units_sold })));
  const futureLine = line(future.map((p, i) => ({ i: actual.length - 1 + i, v: p.point })));
  const band = future.filter((p) => p.lower_80 != null && p.upper_80 != null);
  const bandPath = band.length ? `${band.map((p, i) => `${i ? "L" : "M"}${x(actual.length - 1 + i)},${y(p.upper_80!)}`).join(" ")} ${band.map((_, j) => band[band.length - 1 - j]).map((p, j) => `L${x(actual.length - 1 + band.length - 1 - j)},${y(p.lower_80!)}`).join(" ")} Z` : "";
  return <svg className="demand-chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label="Observed demand and forecast with 80 percent interval">
    {[0, 1, 2, 3].map((i) => <line key={i} x1="0" x2={w} y1={pad + i * (h - pad * 2) / 3} y2={pad + i * (h - pad * 2) / 3} className="grid-line" />)}
    {actual.length > 1 && <path d={`${actualLine} L${x(actual.length - 1)},${h - pad} L${x(0)},${h - pad} Z`} className="actual-fill" />}
    {bandPath && <path d={bandPath} className="forecast-band" />}<path d={actualLine} className="actual-line" />{futureLine && <path d={futureLine} className="forecast-line" />}
    <line x1={x(Math.max(0, actual.length - 1))} x2={x(Math.max(0, actual.length - 1))} y1={pad} y2={h - pad} className="forecast-divider" />
    <text x={pad} y={h - 1} className="axis-label">{actual[0]?.date ?? ""}</text><text x={w - pad} y={h - 1} textAnchor="end" className="axis-label">{future.at(-1)?.date ?? actual.at(-1)?.date ?? ""}</text>
  </svg>;
}
