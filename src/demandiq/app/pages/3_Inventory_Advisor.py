"""Inventory Advisor page — editable parameters, ROP/EOQ recommendations, simulation."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandiq.app.theme import (
    BLUE, GREEN, MUTED, ORANGE, PLOTLY_TEMPLATE, RED,
    apply_theme, page_header, render_sidebar_nav, stat_card,
)
from demandiq.config import settings
from demandiq.database.connection import get_db, init_db
from demandiq.database.repositories import (
    get_all_products, get_all_stores, get_forecast_runs,
    get_forecasts_dataframe, get_inventory_params, get_sales_dataframe,
    upsert_inventory_params,
)
from demandiq.inventory.policies import InventoryPolicy, ReorderPointPolicy, PeriodicReviewPolicy

st.set_page_config(page_title="Inventory · DemandIQ", layout="wide")
apply_theme()


@st.cache_data(ttl=60, show_spinner=False)
def _get_ids():
    init_db()
    with get_db() as s:
        stores   = [r.store_id   for r in get_all_stores(s)]
        products = [r.product_id for r in get_all_products(s)]
        runs = get_forecast_runs(s)
        default_pair = next(
            ((run.store_id, run.product_id) for run in runs if run.store_id and run.product_id),
            None,
        )
    return stores, products, default_pair


@st.cache_data(ttl=30, show_spinner=False)
def _get_latest_forecast(store_id: str, product_id: str) -> pd.DataFrame:
    with get_db() as s:
        runs = get_forecast_runs(s, store_id=store_id, product_id=product_id)
        run_candidates = [(r.forecast_run_id, r.eval_mae) for r in runs]
    if not run_candidates:
        return pd.DataFrame()
    # Pick best run by lowest eval_mae
    run_id, _ = min(
        (item for item in run_candidates if item[1] is not None),
        key=lambda item: item[1],
        default=run_candidates[0],
    )
    with get_db() as s:
        fc = get_forecasts_dataframe(s, run_id)
    if not fc.empty:
        fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])
    return fc


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.divider()

store_ids, product_ids, default_pair = _get_ids()
if not store_ids:
    page_header("Inventory Advisor", "Replenishment recommendations")
    st.info("No data yet. Run: `python scripts/run_pipeline.py --generate-demo --ingest`")
    st.stop()

with st.sidebar:
    default_store_index = store_ids.index(default_pair[0]) if default_pair else 0
    selected_store = st.selectbox("Store", store_ids, index=default_store_index)
    default_product_index = (
        product_ids.index(default_pair[1])
        if default_pair and default_pair[0] == selected_store
        else 0
    )
    selected_product = st.selectbox(
        "Product", product_ids, index=default_product_index
    )

page_header(
    "Inventory Advisor",
    f"{selected_store} / {selected_product} · Reorder point · EOQ · Simulation",
)

# ── Load existing params from DB (or use defaults) ────────────────────────────
with get_db() as s:
    params_row = get_inventory_params(s, selected_store, selected_product)
    db_params = (
        {
            "lead_time_days": params_row.lead_time_days,
            "unit_cost": params_row.unit_cost,
            "ordering_cost": params_row.ordering_cost,
            "holding_cost_per_unit_day": params_row.holding_cost_per_unit_day,
            "shortage_cost_per_unit": params_row.shortage_cost_per_unit,
            "service_level": params_row.service_level,
            "min_order_quantity": params_row.min_order_quantity,
        }
        if params_row
        else None
    )

def _default(attr: str, fallback):
    return db_params.get(attr, fallback) if db_params else fallback

# ── Inventory parameters form ─────────────────────────────────────────────────
st.subheader("Inventory parameters")
st.info(
    "These are **operational assumptions** — not facts inferred from sales data alone. "
    "Edit them to reflect your actual supply chain. Defaults are clearly labeled.",
)

with st.form("inv_params_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        on_hand      = st.number_input("On-hand inventory (units)",    min_value=0.0, value=0.0, step=1.0, help="Current inventory on the shelf")
        on_order     = st.number_input("On-order inventory (units)",   min_value=0.0, value=0.0, step=1.0, help="Units already ordered, not yet arrived")
        lead_time    = st.number_input("Lead time (days)",             min_value=1,   value=_default("lead_time_days", settings.DEFAULT_LEAD_TIME_DAYS), step=1)
        min_order    = st.number_input("Min order quantity",           min_value=1,   value=_default("min_order_quantity", 1), step=1)
    with c2:
        unit_cost    = st.number_input("Unit cost (\$)",               min_value=0.01, value=_default("unit_cost", settings.DEFAULT_UNIT_COST), step=0.5, format="%.2f")
        ordering_cost= st.number_input("Ordering cost (\$ per order)", min_value=0.0,  value=_default("ordering_cost", settings.DEFAULT_ORDERING_COST), step=1.0, format="%.2f")
        holding_cost = st.number_input("Holding cost (\$/unit/day)",   min_value=0.001,value=_default("holding_cost_per_unit_day", settings.DEFAULT_HOLDING_COST_PER_UNIT_DAY), step=0.001, format="%.3f")
    with c3:
        shortage_cost= st.number_input("Shortage cost (\$/unit lost)", min_value=0.0,  value=_default("shortage_cost_per_unit", settings.DEFAULT_SHORTAGE_COST_PER_UNIT), step=0.5, format="%.2f")
        service_level_pct = st.slider(
            "Target service level",
            50,
            99,
            int(round(_default("service_level", settings.DEFAULT_SERVICE_LEVEL) * 100)),
            step=1,
            format="%d%%",
            help="Target probability of meeting demand during the protection period.",
        )
        policy_choice= st.radio("Policy", ["Reorder Point (ROP)", "Periodic Review"], horizontal=True)

    save_btn = st.form_submit_button("Save & calculate", type="primary", use_container_width=True)

service_level = service_level_pct / 100.0

if save_btn:
    with get_db() as s:
        upsert_inventory_params(s, {
            "store_id":               selected_store,
            "product_id":             selected_product,
            "lead_time_days":         lead_time,
            "unit_cost":              unit_cost,
            "ordering_cost":          ordering_cost,
            "holding_cost_per_unit_day": holding_cost,
            "shortage_cost_per_unit": shortage_cost,
            "service_level":          service_level,
            "min_order_quantity":     min_order,
        })
    st.success("Parameters saved.")

st.divider()

# ── Load forecast for policy computation ─────────────────────────────────────
fc_df = _get_latest_forecast(selected_store, selected_product)

if fc_df.empty:
    st.warning(
        "No forecast available for this series. "
        "Go to the **Forecast** page and run the pipeline first."
    )
    st.stop()

forecast_arr = fc_df["point_forecast"].values
forecast_std = float(fc_df["point_forecast"].std()) if len(fc_df) > 1 else None

# ── Compute recommendation ───────────────────────────────────────────────────
inv_policy = InventoryPolicy(
    lead_time_days=lead_time,
    service_level=service_level,
    on_hand=on_hand,
    on_order=on_order,
    min_order_qty=min_order,
)

if "ROP" in policy_choice:
    policy = ReorderPointPolicy(inv_policy)
    rec = policy.recommend(
        daily_demand_forecast=forecast_arr,
        ordering_cost=ordering_cost,
        holding_cost_per_unit_day=holding_cost,
        unit_cost=unit_cost,
        forecast_std=forecast_std,
    )
    policy_label = "Continuous Review (s, Q)"
else:
    policy = PeriodicReviewPolicy(inv_policy, review_period_days=7)
    rec = policy.recommend(daily_demand_forecast=forecast_arr, forecast_std=forecast_std)
    policy_label = "Periodic Review (R, S)"

# ── Recommendation display ────────────────────────────────────────────────────
st.subheader(f"Recommendation · {policy_label}")

should_order = rec.get("should_order", rec.get("suggested_order_qty", 0) > 0)
order_qty    = rec.get("suggested_order_qty", 0)

# Traffic-light banner
if should_order and order_qty > 0:
    st.markdown(
        f"""<div style="background:rgba(239,68,68,0.15); border:1px solid #ef4444;
                        border-radius:10px; padding:16px 20px; margin-bottom:1rem;">
            <span style="font-size:1.2rem; font-weight:700; color:#E05B55;">ORDER RECOMMENDED</span>
            <span style="color:#f1f5f9; font-size:1.1rem; margin-left:1rem;">
                Place order for <strong>{int(order_qty):,} units</strong>
            </span>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""<div style="background:rgba(216,182,92,0.09); border:1px solid #8B743D;
                        border-radius:5px; padding:16px 20px; margin-bottom:1rem;">
            <span style="font-size:1.2rem; font-weight:700; color:#D8B65C;">STOCK SUFFICIENT</span>
            <span style="color:#f1f5f9; font-size:1rem; margin-left:1rem;">No order required at this time.</span>
        </div>""",
        unsafe_allow_html=True,
    )

# KPI row
kpi_cols = st.columns(4)
kpi_cols[0].markdown(stat_card("Inventory position",   f"{rec.get('inventory_position', on_hand):.0f} units"), unsafe_allow_html=True)
kpi_cols[1].markdown(stat_card("Reorder point",        f"{rec.get('reorder_point', rec.get('order_up_to_level', 0)):.0f} units"), unsafe_allow_html=True)
kpi_cols[2].markdown(stat_card("Safety stock",         f"{rec.get('safety_stock', 0):.0f} units"), unsafe_allow_html=True)
kpi_cols[3].markdown(stat_card("Suggested order qty",  f"{int(order_qty):,} units", color=RED if should_order else GREEN), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Explanation
with st.expander("Recommendation details"):
    st.markdown(rec.get("explanation", "No explanation available."))
    st.markdown("---")
    st.markdown("**Input parameters used:**")
    param_df = pd.DataFrame([{
        "Parameter": k, "Value": f"{v}"
    } for k, v in {
        "Lead time": f"{lead_time} days",
        "Service level": f"{service_level:.0%}",
        "On-hand": f"{on_hand:.0f}",
        "On-order": f"{on_order:.0f}",
        "Mean LT demand (μ)": f"{rec.get('mu_LT', 0):.2f}",
        "Std LT demand (σ)": f"{rec.get('sigma_LT', 0):.2f}",
        "Unit cost": f"\${unit_cost:.2f}",
        "Ordering cost": f"\${ordering_cost:.2f}",
        "Holding cost/unit/day": f"\${holding_cost:.3f}",
        "Shortage cost/unit": f"\${shortage_cost:.2f}",
    }.items()])
    st.dataframe(param_df, hide_index=True, use_container_width=False)

# ── Forecast vs ROP chart ─────────────────────────────────────────────────────
st.subheader("Lead-time demand forecast")

rop = rec.get("reorder_point", rec.get("order_up_to_level", 0))
lt  = lead_time
lt_fc = fc_df.head(lt)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=lt_fc["forecast_date"], y=lt_fc["point_forecast"],
    name="Daily forecast", marker_color=BLUE, opacity=0.8,
))
if "lower_80" in lt_fc.columns and lt_fc["lower_80"].notna().any():
    fig.add_trace(go.Scatter(
        x=lt_fc["forecast_date"], y=lt_fc["upper_80"],
        mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=lt_fc["forecast_date"], y=lt_fc["lower_80"], mode="lines",
        fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
        line={"width": 0}, name="80% PI", hoverinfo="skip",
    ))
# Cumulative forecast line
cum = lt_fc["point_forecast"].cumsum()
fig.add_trace(go.Scatter(
    x=lt_fc["forecast_date"], y=cum,
    mode="lines+markers", name="Cumulative (LT demand)",
    line={"color": ORANGE, "width": 2},
))
fig.add_hline(
    y=rop, line_dash="dash", line_color=RED,
    annotation_text=f"Reorder point: {rop:.0f}",
    annotation_position="top right",
)
fig.update_layout(
    template=PLOTLY_TEMPLATE, height=340,
    margin={"l": 0, "r": 0, "t": 30, "b": 0},
    yaxis_title="Units", hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    legend={"orientation": "h", "y": 1.12},
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"Chart shows the first {lt} days of the forecast (= lead time). "
    f"The orange line shows cumulative expected demand during lead time. "
    f"If it approaches the red line (reorder point), restocking is needed."
)

# ── EOQ cost breakdown (ROP only) ─────────────────────────────────────────────
if "ROP" in policy_choice:
    st.subheader("Cost breakdown (EOQ analysis)")
    annual_demand  = float(np.mean(forecast_arr)) * 365
    annual_holding = (holding_cost + holding_cost * unit_cost) * 365
    eoq_qty        = policy.compute_eoq(float(np.mean(forecast_arr)), ordering_cost, holding_cost, unit_cost)
    n_orders       = annual_demand / max(eoq_qty, 1)
    annual_order_cost   = n_orders * ordering_cost
    annual_holding_cost = (eoq_qty / 2) * annual_holding

    c1, c2, c3 = st.columns(3)
    c1.markdown(stat_card("EOQ",                  f"{eoq_qty:.0f} units"),      unsafe_allow_html=True)
    c2.markdown(stat_card("Est. orders/year",     f"{n_orders:.1f}"),            unsafe_allow_html=True)
    c3.markdown(stat_card("Est. annual order \$", f"\${annual_order_cost:,.0f}"), unsafe_allow_html=True)

    st.caption(
        "EOQ = √(2 × D × K / h). Annual demand D estimated from forecast mean × 365. "
        "Costs are estimates based on the parameters above — not actual accounting data."
    )
