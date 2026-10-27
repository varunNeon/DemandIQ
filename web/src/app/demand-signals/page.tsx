"use client";

import { useEffect, useMemo, useState } from "react";
import { DashboardShell } from "@/components/dashboard-shell";
import { DemandChart } from "@/components/demand-chart";

const API = process.env.NEXT_PUBLIC_DEMANDIQ_API_URL ?? "http://localhost:8000";
type Store = { store_id: string; city: string | null };
type Product = { product_id: string; family: string };
type Point = { date: string; units_sold: number; onpromotion: boolean; is_holiday: boolean };

export default function DemandSignalsPage() {
  const [stores, setStores] = useState<Store[]>([]), [products, setProducts] = useState<Product[]>([]);
  const [store, setStore] = useState(""), [product, setProduct] = useState(""), [points, setPoints] = useState<Point[]>([]), [error, setError] = useState("");
  useEffect(() => { Promise.all([fetch(`${API}/api/v1/stores`).then((r) => r.json()), fetch(`${API}/api/v1/products`).then((r) => r.json())]).then(([s, p]) => { setStores(s); setProducts(p); setStore(s[0]?.store_id ?? ""); setProduct(p[0]?.product_id ?? ""); }).catch(() => setError("Could not load store and product lists.")); }, []);
  useEffect(() => { if (!store || !product) return; fetch(`${API}/api/v1/sales/${encodeURIComponent(store)}/${encodeURIComponent(product)}?limit=120`).then((r) => { if (!r.ok) throw Error(); return r.json(); }).then(setPoints).catch(() => setError("Could not load demand history for this series.")); }, [store, product]);
  const latest = points.at(-1), mean = points.length ? points.reduce((sum, p) => sum + p.units_sold, 0) / points.length : null;
  const selectedStore = useMemo(() => stores.find((s) => s.store_id === store), [stores, store]);
  return <DashboardShell title="DEMAND SIGNALS"><div className="content">
    <div className="page-head"><div><div className="eyebrow">SERIES EXPLORER <i /></div><h1>Demand signals</h1><p>Explore observed sales by location and product.</p></div></div>
    {error && <div className="error">{error}</div>}
    <section className="panel route-panel"><div className="panel-heading"><div><div className="kicker">SELECT A SERIES</div><h2>Observed demand</h2><p>Daily sales history, newest 120 records.</p></div><div className="selectors"><label>LOCATION<select value={store} onChange={(e) => setStore(e.target.value)}>{stores.map((s) => <option key={s.store_id} value={s.store_id}>{s.store_id}{s.city ? ` · ${s.city}` : ""}</option>)}</select></label><label>PRODUCT<select value={product} onChange={(e) => setProduct(e.target.value)}>{products.map((p) => <option key={p.product_id} value={p.product_id}>{p.product_id} · {p.family}</option>)}</select></label></div></div>
      <div className="signal-stats"><div><span>OBSERVATIONS SHOWN</span><b>{points.length}</b></div><div><span>LATEST DAY</span><b>{latest?.date ?? "—"}</b></div><div><span>LATEST SALES</span><b>{latest ? `${latest.units_sold.toFixed(1)} units` : "—"}</b></div><div><span>120-DAY MEAN</span><b>{mean == null ? "—" : `${mean.toFixed(1)} units`}</b></div></div>
      <div className="legend"><span><i className="observed"/>Observed demand</span><b>UNITS / DAY</b></div><DemandChart actual={points} future={[]}/>
      <div className="chart-caption"><span>{selectedStore?.city ? `${selectedStore.city} · ` : ""}{store} / {product}</span><span>HISTORICAL OBSERVATIONS</span></div>
    </section>
    <section className="panel route-panel history-panel"><div className="panel-heading compact"><div><div className="kicker">RECENT RECORDS</div><h2>Daily demand detail</h2></div></div><div className="history-table"><div className="table-head"><span>DATE</span><span>UNITS SOLD</span><span>PROMOTION</span><span>HOLIDAY</span></div>{points.slice(-12).reverse().map((p) => <div className="history-row" key={p.date}><span>{p.date}</span><b>{p.units_sold.toFixed(1)}</b><span>{p.onpromotion ? "Yes" : "No"}</span><span>{p.is_holiday ? "Yes" : "No"}</span></div>)}</div></section>
  </div></DashboardShell>;
}
