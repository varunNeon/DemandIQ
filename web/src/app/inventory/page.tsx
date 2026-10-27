"use client";

import { useEffect, useState } from "react";
import { DashboardShell } from "@/components/dashboard-shell";

const API = process.env.NEXT_PUBLIC_DEMANDIQ_API_URL ?? "http://localhost:8000";
type Recommendation = { store_id: string; product_id: string; created_at: string | null; inventory_position: number | null; lead_time_demand: number | null; reorder_point: number | null; suggested_order_qty: number | null; expected_shortage_risk: number | null; explanation: string | null };
const fmt = (n: number | null) => n == null ? "—" : new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(n);

export default function InventoryPage() {
  const [rows, setRows] = useState<Recommendation[]>([]), [error, setError] = useState(""), [isLoading, setIsLoading] = useState(true);
  useEffect(() => { fetch(`${API}/api/v1/inventory/recommendations?limit=100`).then((r) => { if (!r.ok) throw Error(); return r.json(); }).then((data: Recommendation[]) => { setRows(data); setIsLoading(false); }).catch(() => { setError("Could not load saved inventory recommendations."); setIsLoading(false); }); }, []);
  return <DashboardShell title="INVENTORY"><div className="content">
    <div className="page-head"><div><div className="eyebrow">REPLENISHMENT INTELLIGENCE <i /></div><h1>Inventory advisor</h1><p>Review saved reorder suggestions and the assumptions behind them.</p></div></div>
    {error && <div className="error">{error}</div>}
    <div className="run-summary"><div><span>SAVED RECOMMENDATIONS</span><b>{isLoading ? "···" : rows.length}</b></div><div><span>STATUS</span><b>{isLoading ? "CHECKING" : rows.length ? "AVAILABLE" : "NOT GENERATED"}</b></div><div><span>DATA SOURCE</span><b>LOCAL DEMANDIQ DATABASE</b></div></div>
    {rows.length ? <section className="panel route-panel"><div className="panel-heading compact"><div><div className="kicker">REORDER DECISIONS</div><h2>Latest recommendations</h2></div></div><div className="history-table"><div className="table-head inventory-table"><span>SERIES</span><span>INVENTORY POSITION</span><span>LEAD-TIME DEMAND</span><span>REORDER POINT</span><span>SUGGESTED ORDER</span></div>{rows.map((r, i) => <div className="history-row inventory-table" key={`${r.store_id}-${r.product_id}-${i}`}><b>{r.store_id} / {r.product_id}</b><span>{fmt(r.inventory_position)}</span><span>{fmt(r.lead_time_demand)}</span><span>{fmt(r.reorder_point)}</span><b>{fmt(r.suggested_order_qty)}</b></div>)}</div></section> : <section className="panel empty-inventory"><div className="empty-mark">{isLoading ? "⌁" : "—"}</div><div className="kicker">{isLoading ? "CHECKING DATABASE" : "NO SAVED RECOMMENDATIONS"}</div><h2>{isLoading ? "Loading inventory data" : "Nothing to replenish yet"}</h2><p>{isLoading ? "Checking for recommendations saved in your local DemandIQ database." : "The database currently contains no inventory recommendations. Once a recommendation is generated in the existing DemandIQ workflow, it will appear here."}</p>{!isLoading && <div className="assumption-note"><b>What informs a recommendation?</b><span>Demand over supplier lead time, current inventory position, minimum order quantity, unit and holding costs, shortage cost, and the target service level.</span><small>These settings are operational assumptions. They are not inferred from the sales history.</small></div>}</section>}
  </div></DashboardShell>;
}
