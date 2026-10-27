"""DemandIQ — Overview page (entry point)."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandiq.app.theme import (
    BLUE, BORDER, MODEL_COLORS, ORANGE, PLOTLY_TEMPLATE,
    apply_theme, page_header, render_sidebar_nav, stat_card,
)
from demandiq.database.connection import get_db, init_db
from demandiq.database.repositories import (
    count_sales_rows, get_all_products, get_all_stores,
    get_backtest_summary, get_forecast_runs,
    get_forecasts_dataframe, get_sales_dataframe, get_sales_date_range,
)

st.set_page_config(
    page_title="DemandIQ",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


@st.cache_data(ttl=60, show_spinner=False)
def _load_snapshot() -> dict:
    init_db()
    with get_db() as s:
        store_ids = [r.store_id for r in get_all_stores(s)]
        product_ids = [r.product_id for r in get_all_products(s)]
        dr       = get_sales_date_range(s)
        metrics  = get_backtest_summary(s)
        runs = [
            {"id": r.forecast_run_id, "model": r.model_name,
             "store": r.store_id, "product": r.product_id,
             "cutoff": r.training_cutoff, "created": r.created_at,
             "horizon": r.horizon_days,
             "mae": r.eval_mae, "wape": r.eval_wape}
            for r in get_forecast_runs(s)
        ]
        n_sales  = count_sales_rows(s)
    return {
        "store_ids": store_ids,
        "product_ids": product_ids,
        "n_sales": n_sales,
        "start_date": dr[0], "end_date": dr[1],
        "metrics": metrics,
        "runs": runs,
    }


@st.cache_data(ttl=60, show_spinner=False)
def _load_series(store_id: str, product_id: str, run_id: int | None):
    with get_db() as s:
        history  = get_sales_dataframe(s, store_id=store_id, product_id=product_id)
        forecast = get_forecasts_dataframe(s, run_id) if run_id else pd.DataFrame()
    return history, forecast


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    snap = _load_snapshot()
except Exception as exc:
    page_header("DemandIQ", "Demand Forecasting & Inventory Intelligence")
    st.error("Cannot connect to the database.")
    st.code(str(exc))
    st.info("Run `python scripts/run_pipeline.py --generate-demo --ingest` to create demo data.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.divider()

    # Lead with a saved forecast when one exists, so the first view demonstrates
    # the product's core workflow instead of opening on an unforecasted series.
    first_run = snap["runs"][0] if snap["runs"] else None
    default_store_index = (
        snap["store_ids"].index(first_run["store"])
        if first_run and first_run["store"] in snap["store_ids"]
        else 0
    )
    if snap["store_ids"] and snap["product_ids"]:
        selected_store = st.selectbox(
            "Store", snap["store_ids"], index=default_store_index
        )
        default_product_index = (
            snap["product_ids"].index(first_run["product"])
            if first_run
            and first_run["store"] == selected_store
            and first_run["product"] in snap["product_ids"]
            else 0
        )
        selected_product = st.selectbox(
            "Product", snap["product_ids"], index=default_product_index
        )
    else:
        selected_store = selected_product = None

    # Filter matching runs
    matching_runs = [
        r for r in snap["runs"]
        if r["store"] == selected_store and r["product"] == selected_product
    ] if selected_store else []

    run_labels = {
        f"{r['model'].upper()} · {r['created'].strftime('%b %d %H:%M')} · run #{r['id']}": r["id"]
        for r in matching_runs
    }
    selected_label = st.selectbox(
        "Forecast run",
        ["— no run selected —"] + list(run_labels),
        index=1 if matching_runs else 0,
        help="Forecasts are read from saved runs. Train models on the Forecast page.",
    )
    selected_run_id = run_labels.get(selected_label)

    st.divider()
    if st.button("Refresh data", use_container_width=True):
        _load_snapshot.clear()
        _load_series.clear()
        st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
page_header("Demand overview", "Sales history, forecast signal, and model performance")

# ── KPI row ───────────────────────────────────────────────────────────────────
date_range = ""
if snap["start_date"] and snap["end_date"]:
    date_range = f"{snap['start_date'].strftime('%b %y')} – {snap['end_date'].strftime('%b %y')}"

cols = st.columns(4)
cols[0].markdown(stat_card("Sales records", f"{snap['n_sales']:,}"), unsafe_allow_html=True)
cols[1].markdown(stat_card("Stores", str(len(snap["store_ids"]))), unsafe_allow_html=True)
cols[2].markdown(stat_card("Products", str(len(snap["product_ids"]))), unsafe_allow_html=True)
cols[3].markdown(stat_card("Date range", date_range or "—"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── No data guard ─────────────────────────────────────────────────────────────
if not snap["store_ids"] or snap["n_sales"] == 0:
    st.info("No data ingested yet.")
    st.code("python scripts/run_pipeline.py --generate-demo --ingest")
    st.stop()

# ── Sales + forecast chart ────────────────────────────────────────────────────
history, forecasts = _load_series(selected_store, selected_product, selected_run_id)

fig = go.Figure()
if not history.empty:
    h = history.sort_values("date")
    fig.add_trace(go.Scatter(
        x=h["date"], y=h["units_sold"], mode="lines",
        name="Actual", line={"color": BLUE, "width": 1.5},
    ))

if not forecasts.empty:
    fc = forecasts.sort_values("forecast_date")
    # 95% band
    if fc["lower_95"].notna().any():
        fig.add_trace(go.Scatter(x=fc["forecast_date"], y=fc["upper_95"], mode="lines",
                                 line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["forecast_date"], y=fc["lower_95"], mode="lines",
                                 line={"width": 0}, fill="tonexty",
                                 fillcolor="rgba(249,115,22,0.10)", name="95% PI", hoverinfo="skip"))
    # 80% band
    if fc["lower_80"].notna().any():
        fig.add_trace(go.Scatter(x=fc["forecast_date"], y=fc["upper_80"], mode="lines",
                                 line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc["forecast_date"], y=fc["lower_80"], mode="lines",
                                 line={"width": 0}, fill="tonexty",
                                 fillcolor="rgba(249,115,22,0.20)", name="80% PI", hoverinfo="skip"))
    # Point forecast
    fig.add_trace(go.Scatter(
        x=fc["forecast_date"], y=fc["point_forecast"], mode="lines+markers",
        name="Forecast", line={"color": ORANGE, "width": 2, "dash": "dash"},
        marker={"size": 4},
    ))

fig.update_layout(
    template=PLOTLY_TEMPLATE,
    height=380, margin={"l": 0, "r": 0, "t": 30, "b": 0},
    xaxis_title=None, yaxis_title="Units sold",
    legend={"orientation": "h", "y": 1.12, "x": 0},
    hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

if forecasts.empty and selected_run_id is None:
    st.caption("No forecast run selected. Train a model in Forecast Studio, then select a saved run here.")

# ── Backtest leaderboard ──────────────────────────────────────────────────────
st.subheader("Model leaderboard")
metrics_df = snap["metrics"]
if metrics_df.empty:
    st.info("No backtest scores yet. Run the forecasting pipeline to populate this table.")
else:
    subset = metrics_df[
        (metrics_df["store_id"] == selected_store) &
        (metrics_df["product_id"] == selected_product)
    ].copy()

    if subset.empty:
        st.info(f"No evaluated runs for {selected_store} / {selected_product}.")
    else:
        subset = subset.sort_values("created_at", ascending=False)
        best   = subset.drop_duplicates("model", keep="first").sort_values("MAE")
        best["Model"] = best["model"].str.upper()
        for column in ["WAPE", "Coverage_80", "Coverage_95"]:
            best[column] = best[column].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else "—"
            )

        col_cfg = {
            "Model":       st.column_config.TextColumn("Model"),
            "MAE":         st.column_config.NumberColumn("MAE",          format="%.2f"),
            "RMSE":        st.column_config.NumberColumn("RMSE",         format="%.2f"),
            "WAPE":        st.column_config.TextColumn("WAPE"),
            "MASE":        st.column_config.NumberColumn("MASE",         format="%.2f"),
            "Coverage_80": st.column_config.TextColumn("Coverage 80%"),
            "Coverage_95": st.column_config.TextColumn("Coverage 95%"),
        }
        display_cols = [c for c in ["Model", "MAE", "RMSE", "WAPE", "MASE", "Coverage_80", "Coverage_95"] if c in best.columns]
        st.dataframe(best[display_cols], hide_index=True, use_container_width=True, column_config=col_cfg)
        st.caption("Scores are from rolling-origin backtests. Lower MAE / WAPE = better. Coverage should be close to the nominal level.")

# ── Recent runs table ─────────────────────────────────────────────────────────
with st.expander("Recent forecast runs (all series)"):
    if snap["runs"]:
        df_runs = pd.DataFrame(snap["runs"][:30])
        df_runs["model"] = df_runs["model"].str.upper()
        df_runs["mae"]   = df_runs["mae"].map(lambda x: f"{x:.2f}" if x else "—")
        df_runs["wape"]  = df_runs["wape"].map(lambda x: f"{x:.1%}" if x else "—")
        st.dataframe(
            df_runs[["id", "model", "store", "product", "cutoff", "horizon", "mae", "wape"]].rename(
                columns={"id": "Run", "model": "Model", "store": "Store",
                         "product": "Product", "cutoff": "Cutoff",
                         "horizon": "Horizon (d)", "mae": "MAE", "wape": "WAPE"}
            ),
            hide_index=True, use_container_width=True,
        )
    else:
        st.write("No runs saved yet.")
