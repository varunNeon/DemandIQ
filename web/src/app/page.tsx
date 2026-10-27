"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { DashboardShell } from "@/components/dashboard-shell";
import { DemandChart } from "@/components/demand-chart";

const API = process.env.NEXT_PUBLIC_DEMANDIQ_API_URL ?? "http://localhost:8000";
type Run = { run_id: number; model_name: string; horizon_days: number; store_id: string | null; product_id: string | null; metrics: { mae: number | null; wape: number | null } };
type Overview = { sales_rows: number; store_count: number; product_count: number; forecast_run_count: number; recommendation_count: number; date_start: string | null; date_end: string | null };
type Store = { store_id: string; city: string | null };
type Product = { product_id: string; family: string };
type Point = { date: string; units_sold: number };
type Forecast = { date: string; point: number; lower_80: number | null; upper_80: number | null };
const fmt = (v: number | null | undefined, d = 0) => v == null ? "—" : new Intl.NumberFormat("en-US", { maximumFractionDigits: d }).format(v);

export default function Home() {
  const [summary, setSummary] = useState<Overview | null>(null), [stores, setStores] = useState<Store[]>([]), [products, setProducts] = useState<Product[]>([]), [runs, setRuns] = useState<Run[]>([]);
  const [storeId, setStoreId] = useState(""), [productId, setProductId] = useState(""), [actual, setActual] = useState<Point[]>([]), [future, setFuture] = useState<Forecast[]>([]), [error, setError] = useState(""), [loading, setLoading] = useState(true), [seriesLoading, setSeriesLoading] = useState(true);
  useEffect(() => {
    Promise.all(["/api/v1/overview", "/api/v1/stores", "/api/v1/products", "/api/v1/forecast-runs?limit=8"].map(async (p) => { const r = await fetch(API + p); if (!r.ok) throw Error(`${p} returned ${r.status}`); return r.json(); }))
      .then(([o, s, p, r]) => { setSummary(o); setStores(s); setProducts(p); setRuns(r); const latest: Run | undefined = r[0]; setStoreId(latest?.store_id ?? s[0]?.store_id ?? ""); setProductId(latest?.product_id ?? p[0]?.product_id ?? ""); })
      .catch((e: Error) => setError(`${e.message}. Start the Python API to connect this workspace.`)).finally(() => setLoading(false));
  }, []);
  useEffect(() => {
    if (!storeId || !productId) return; let active = true;
    Promise.all([fetch(`${API}/api/v1/sales/${encodeURIComponent(storeId)}/${encodeURIComponent(productId)}?limit=120`).then((r) => r.ok ? r.json() : []), fetch(`${API}/api/v1/forecast-runs?store_id=${encodeURIComponent(storeId)}&product_id=${encodeURIComponent(productId)}&limit=1`).then((r) => r.ok ? r.json() : [])])
      .then(async ([history, selectedRuns]) => { if (!active) return; setActual(history); if (!selectedRuns[0]) { setFuture([]); setSeriesLoading(false); return; } const r = await fetch(`${API}/api/v1/forecast-runs/${selectedRuns[0].run_id}`); const result = r.ok ? await r.json() : null; if (active) { setFuture(result?.forecasts ?? []); setSeriesLoading(false); } })
      .catch(() => { if (active) { setActual([]); setFuture([]); setSeriesLoading(false); } });
    return () => { active = false; };
  }, [storeId, productId]);
  const store = useMemo(() => stores.find((s) => s.store_id === storeId), [stores, storeId]);
  const product = useMemo(() => products.find((p) => p.product_id === productId), [products, productId]);

  return <DashboardShell title="OVERVIEW">
      <div className="content"><div className="page-head"><div><div className="eyebrow">DEMAND INTELLIGENCE <i /></div><h1>Demand overview</h1><p>A precise view of demand, forecast performance and inventory exposure.</p></div><button onClick={() => window.location.reload()}>↻ <span>Refresh data</span></button></div>
      {error && <div className="error"><b>Connection unavailable</b><span>{error}</span><code>python -m uvicorn demandiq.api.main:app --reload</code></div>}
      <div className="metrics"><Metric title="SALES OBSERVATIONS" value={fmt(summary?.sales_rows)} detail="Daily records in workspace" icon="▦" loading={loading}/><Metric title="ACTIVE LOCATIONS" value={fmt(summary?.store_count)} detail="Stores represented" icon="⌖" loading={loading}/><Metric title="PRODUCTS TRACKED" value={fmt(summary?.product_count)} detail="Distinct product series" icon="◇" loading={loading}/><Metric title="FORECAST RUNS" value={fmt(summary?.forecast_run_count)} detail="Saved model evaluations" icon="⌁" loading={loading} gold/></div>
      <section className="panel demand" id="demand"><div className="panel-heading"><div><div className="kicker">SIGNAL & OUTLOOK</div><h2>Demand trajectory</h2><p>Observed daily demand against the latest saved forecast.</p></div><div className="selectors"><label>LOCATION<select value={storeId} onChange={(e) => { setSeriesLoading(true); setStoreId(e.target.value); }}>{stores.map((s) => <option key={s.store_id} value={s.store_id}>{s.store_id}{s.city ? ` · ${s.city}` : ""}</option>)}</select></label><label>PRODUCT<select value={productId} onChange={(e) => { setSeriesLoading(true); setProductId(e.target.value); }}>{products.map((p) => <option key={p.product_id} value={p.product_id}>{p.product_id} · {p.family}</option>)}</select></label></div></div><div className="legend"><span><i className="observed"/>Observed demand</span><span><i className="predicted"/>Forecast</span><span><i className="interval"/>80% range</span><span><i className="interval-95"/>95% range</span><b>UNITS / DAY</b></div><DemandChart actual={actual} future={future} loading={seriesLoading}/><div className="chart-caption"><span>{store?.city ? `${store.city} · ` : ""}{storeId || "Select location"} / {(product?.family ?? productId) || "Select product"}</span><span>{future.length ? `${future.length}-DAY OUTLOOK` : seriesLoading ? "LOADING SELECTED SERIES" : "NO SAVED FORECAST FOR THIS SERIES"}</span></div></section>
      <div className="lower"><section className="panel runs" id="runs"><div className="panel-heading compact"><div><div className="kicker">MODEL ACTIVITY</div><h2>Recent forecast runs</h2></div><Link className="text-link" href="/forecast-runs">VIEW ALL ↗</Link></div>{runs.length ? <div className="table"><div className="table-head"><span>MODEL</span><span>SERIES</span><span>MAE</span><span>WAPE</span><span>HORIZON</span></div>{runs.slice(0, 5).map((r) => <div className="run-row" key={r.run_id}><b><i>{r.model_name[0].toUpperCase()}</i>{r.model_name.replaceAll("_", " ")}</b><span>{r.store_id} / {r.product_id}</span><span>{fmt(r.metrics.mae, 2)}</span><span>{r.metrics.wape == null ? "—" : `${fmt(r.metrics.wape * 100, 1)}%`}</span><span>{r.horizon_days} days</span></div>)}</div> : <div className="empty">No forecast runs yet. Generate one in Forecast.</div>}</section>
      <section className="panel health" id="inventory"><div className="kicker">DATA WINDOW</div><h2>Workspace health</h2><div className="date-range"><span>HISTORY AVAILABLE</span><b>{summary?.date_start ?? "—"}<i>→</i>{summary?.date_end ?? "—"}</b></div><div className="recommendations"><span>Inventory recommendations</span><b>{fmt(summary?.recommendation_count)}</b></div><div className="health-note"><i>i</i> Recommendations depend on lead time, costs and target service levels configured for each series.</div></section></div>
      <footer><span>DEMANDIQ <i>·</i> DECISION SUPPORT FOR INVENTORY PLANNING</span><span>DEMO DATA <i>·</i> {summary?.date_start ?? ""}{summary?.date_end ? ` — ${summary.date_end}` : ""}</span></footer></div></DashboardShell>;
}

function Metric({ title, value, detail, icon, loading, gold = false }: { title: string; value: string; detail: string; icon: string; loading: boolean; gold?: boolean }) {
  return <article className={`metric${gold ? " metric-gold" : ""}`}><div className="metric-label">{title}<i>{icon}</i></div><div className={loading ? "metric-value loading" : "metric-value"}>{loading ? "···" : value}</div><div className="metric-detail">{detail}</div></article>;
}
