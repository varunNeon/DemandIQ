"use client";

import { useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_DEMANDIQ_API_URL ?? "http://localhost:8000";
type Run = { run_id: number; model_name: string; horizon_days: number; store_id: string | null; product_id: string | null; metrics: { mae: number | null; wape: number | null } };
type Overview = { sales_rows: number; store_count: number; product_count: number; forecast_run_count: number; recommendation_count: number; date_start: string | null; date_end: string | null };
type Store = { store_id: string; city: string | null };
type Product = { product_id: string; family: string };
type Point = { date: string; units_sold: number };
type Forecast = { date: string; point: number; lower_80: number | null; upper_80: number | null };
const fmt = (v: number | null | undefined, d = 0) => v == null ? "—" : new Intl.NumberFormat("en-US", { maximumFractionDigits: d }).format(v);

function DemandChart({ actual, future }: { actual: Point[]; future: Forecast[] }) {
  const w = 900, h = 260, pad = 16, values = [...actual.map((v) => v.units_sold), ...future.map((v) => v.point)];
  if (!values.length) return <div className="chart-empty">No demand history for this series yet.</div>;
  const lo = Math.min(0, ...values), hi = Math.max(1, ...values) * 1.12, n = Math.max(2, actual.length + future.length);
  const x = (i: number) => pad + i / (n - 1) * (w - pad * 2), y = (v: number) => h - pad - (v - lo) / (hi - lo || 1) * (h - pad * 2);
  const line = (pts: { i: number; v: number }[]) => pts.map((p, j) => `${j ? "L" : "M"}${x(p.i)},${y(p.v)}`).join(" ");
  const actualLine = line(actual.map((p, i) => ({ i, v: p.units_sold })));
  const futurePts = future.map((p, i) => ({ i: actual.length - 1 + i, v: p.point }));
  const futureLine = line(futurePts);
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

export default function Home() {
  const [summary, setSummary] = useState<Overview | null>(null), [stores, setStores] = useState<Store[]>([]), [products, setProducts] = useState<Product[]>([]), [runs, setRuns] = useState<Run[]>([]);
  const [storeId, setStoreId] = useState(""), [productId, setProductId] = useState(""), [actual, setActual] = useState<Point[]>([]), [future, setFuture] = useState<Forecast[]>([]), [error, setError] = useState(""), [loading, setLoading] = useState(true);
  useEffect(() => {
    Promise.all(["/api/v1/overview", "/api/v1/stores", "/api/v1/products", "/api/v1/forecast-runs?limit=8"].map(async (p) => { const r = await fetch(API + p); if (!r.ok) throw Error(`${p} returned ${r.status}`); return r.json(); }))
      .then(([o, s, p, r]) => { setSummary(o); setStores(s); setProducts(p); setRuns(r); const latest: Run | undefined = r[0]; setStoreId(latest?.store_id ?? s[0]?.store_id ?? ""); setProductId(latest?.product_id ?? p[0]?.product_id ?? ""); })
      .catch((e: Error) => setError(`${e.message}. Start the Python API to connect this workspace.`)).finally(() => setLoading(false));
  }, []);
  useEffect(() => {
    if (!storeId || !productId) return; let active = true;
    Promise.all([fetch(`${API}/api/v1/sales/${encodeURIComponent(storeId)}/${encodeURIComponent(productId)}?limit=120`).then((r) => r.ok ? r.json() : []), fetch(`${API}/api/v1/forecast-runs?store_id=${encodeURIComponent(storeId)}&product_id=${encodeURIComponent(productId)}&limit=1`).then((r) => r.ok ? r.json() : [])])
      .then(async ([history, selectedRuns]) => { if (!active) return; setActual(history); if (!selectedRuns[0]) { setFuture([]); return; } const r = await fetch(`${API}/api/v1/forecast-runs/${selectedRuns[0].run_id}`); const result = r.ok ? await r.json() : null; if (active) setFuture(result?.forecasts ?? []); })
      .catch(() => { if (active) { setActual([]); setFuture([]); } });
    return () => { active = false; };
  }, [storeId, productId]);
  const store = useMemo(() => stores.find((s) => s.store_id === storeId), [stores, storeId]);
  const product = useMemo(() => products.find((p) => p.product_id === productId), [products, productId]);

  return <main className="app-shell">
    <aside className="sidebar"><a className="brand" href="#overview"><span className="brand-mark">D</span><span>DEMAND<span className="gold">IQ</span><small>INTELLIGENCE SYSTEMS</small></span></a><div className="nav-caption">WORKSPACE</div><nav><a className="nav-item selected" href="#overview"><span>◫</span> Overview</a><a className="nav-item" href="#demand"><span>⌁</span> Demand signals</a><a className="nav-item" href="#runs"><span>◷</span> Forecast runs</a><a className="nav-item" href="#inventory"><span>▤</span> Inventory</a></nav><div className="side-foot"><i /> LOCAL ENVIRONMENT<small>DEMANDIQ / 0.1.0</small></div></aside>
    <section className="main-area" id="overview"><header className="topbar"><div>DEMANDIQ <span>/</span> <b>OVERVIEW</b></div><div className="status"><i /> SYSTEM ONLINE <span className="avatar">V</span></div></header>
      <div className="content"><div className="page-head"><div><div className="eyebrow">DEMAND INTELLIGENCE <i /></div><h1>Demand overview</h1><p>A precise view of demand, forecast performance and inventory exposure.</p></div><button onClick={() => window.location.reload()}>↻ <span>Refresh data</span></button></div>
      {error && <div className="error"><b>Connection unavailable</b><span>{error}</span><code>python -m uvicorn demandiq.api.main:app --reload</code></div>}
      <div className="metrics"><Metric title="SALES OBSERVATIONS" value={fmt(summary?.sales_rows)} detail="Daily records in workspace" icon="▦" loading={loading}/><Metric title="ACTIVE LOCATIONS" value={fmt(summary?.store_count)} detail="Stores represented" icon="⌖" loading={loading}/><Metric title="PRODUCTS TRACKED" value={fmt(summary?.product_count)} detail="Distinct product series" icon="◇" loading={loading}/><Metric title="FORECAST RUNS" value={fmt(summary?.forecast_run_count)} detail="Saved model evaluations" icon="⌁" loading={loading} gold/></div>
      <section className="panel demand" id="demand"><div className="panel-heading"><div><div className="kicker">SIGNAL & OUTLOOK</div><h2>Demand trajectory</h2><p>Observed daily demand against the latest saved forecast.</p></div><div className="selectors"><label>LOCATION<select value={storeId} onChange={(e) => setStoreId(e.target.value)}>{stores.map((s) => <option key={s.store_id} value={s.store_id}>{s.store_id}{s.city ? ` · ${s.city}` : ""}</option>)}</select></label><label>PRODUCT<select value={productId} onChange={(e) => setProductId(e.target.value)}>{products.map((p) => <option key={p.product_id} value={p.product_id}>{p.product_id} · {p.family}</option>)}</select></label></div></div><div className="legend"><span><i className="observed"/>Observed demand</span><span><i className="predicted"/>Forecast</span><span><i className="interval"/>80% interval</span><b>UNITS / DAY</b></div><DemandChart actual={actual} future={future}/><div className="chart-caption"><span>{store?.city ? `${store.city} · ` : ""}{storeId || "Select location"} / {(product?.family ?? productId) || "Select product"}</span><span>{future.length ? `${future.length}-DAY OUTLOOK` : "NO SAVED FORECAST FOR THIS SERIES"}</span></div></section>
      <div className="lower"><section className="panel runs" id="runs"><div className="panel-heading compact"><div><div className="kicker">MODEL ACTIVITY</div><h2>Recent forecast runs</h2></div><a href="#runs">VIEW ALL ↗</a></div>{runs.length ? <div className="table"><div className="table-head"><span>MODEL</span><span>SERIES</span><span>MAE</span><span>WAPE</span><span>HORIZON</span></div>{runs.slice(0, 5).map((r) => <div className="run-row" key={r.run_id}><b><i>{r.model_name[0].toUpperCase()}</i>{r.model_name.replaceAll("_", " ")}</b><span>{r.store_id} / {r.product_id}</span><span>{fmt(r.metrics.mae, 2)}</span><span>{r.metrics.wape == null ? "—" : `${fmt(r.metrics.wape * 100, 1)}%`}</span><span>{r.horizon_days} days</span></div>)}</div> : <div className="empty">No forecast runs yet. Generate one in Forecast.</div>}</section>
      <section className="panel health" id="inventory"><div className="kicker">DATA WINDOW</div><h2>Workspace health</h2><div className="date-range"><span>HISTORY AVAILABLE</span><b>{summary?.date_start ?? "—"}<i>→</i>{summary?.date_end ?? "—"}</b></div><div className="recommendations"><span>Inventory recommendations</span><b>{fmt(summary?.recommendation_count)}</b></div><div className="health-note"><i>i</i> Recommendations depend on lead time, costs and target service levels configured for each series.</div></section></div>
      <footer>DEMANDIQ <i>·</i> DECISION SUPPORT FOR INVENTORY PLANNING <span>{summary?.date_start ?? ""}{summary?.date_end ? ` — ${summary.date_end}` : ""}</span></footer></div></section></main>;
}

function Metric({ title, value, detail, icon, loading, gold = false }: { title: string; value: string; detail: string; icon: string; loading: boolean; gold?: boolean }) {
  return <article className={`metric${gold ? " metric-gold" : ""}`}><div className="metric-label">{title}<i>{icon}</i></div><div className={loading ? "metric-value loading" : "metric-value"}>{loading ? "···" : value}</div><div className="metric-detail">{detail}</div></article>;
}
