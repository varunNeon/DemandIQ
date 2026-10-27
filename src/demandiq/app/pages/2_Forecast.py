"""Forecast page — train 4 models, show backtest comparison and future forecast."""
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
    BLUE, MODEL_COLORS, MUTED, ORANGE, PLOTLY_TEMPLATE,
    apply_theme, page_header, render_sidebar_nav, stat_card,
)
from demandiq.database.connection import get_db, init_db
from demandiq.database.repositories import (
    get_all_products, get_all_stores, get_backtest_summary,
    get_forecast_runs, get_forecasts_dataframe, get_sales_dataframe,
)
from demandiq.services.forecast_service import run_forecast_pipeline

st.set_page_config(page_title="Forecast · DemandIQ", layout="wide")
apply_theme()

HORIZON_OPTIONS = {7: "7 days", 14: "14 days", 28: "28 days"}
MODEL_LABELS = {
    "naive":          "Naive (random walk)",
    "seasonal_naive": "Seasonal Naive (weekly)",
    "ets":            "ETS (Holt-Winters)",
    "lgbm":           "LightGBM (ML)",
}


def _format_metric(value, spec: str) -> str:
    """Format a metric while keeping unavailable values explicit."""
    if value is None or pd.isna(value):
        return "—"
    return format(value, spec)


@st.cache_data(ttl=60, show_spinner=False)
def _get_store_product_ids():
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
def _get_runs(store_id: str, product_id: str) -> list[dict]:
    with get_db() as s:
        return [
            {"id": r.forecast_run_id, "model": r.model_name,
             "cutoff": r.training_cutoff, "created": r.created_at,
             "mae": r.eval_mae, "rmse": r.eval_rmse,
             "wape": r.eval_wape, "mase": r.eval_mase,
             "cov80": r.eval_coverage_80, "cov95": r.eval_coverage_95,
             "horizon": r.horizon_days}
            for r in get_forecast_runs(s, store_id=store_id, product_id=product_id)
        ]


@st.cache_data(ttl=60, show_spinner=False)
def _get_history(store_id: str, product_id: str) -> pd.DataFrame:
    with get_db() as s:
        df = get_sales_dataframe(s, store_id=store_id, product_id=product_id)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.divider()

store_ids, product_ids, default_pair = _get_store_product_ids()
if not store_ids:
    page_header("Forecast", "Model training & evaluation")
    st.info("No data ingested yet. Run: `python scripts/run_pipeline.py --generate-demo --ingest`")
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
    st.divider()
    st.markdown("**Training settings**")
    horizon     = st.select_slider("Horizon", options=list(HORIZON_OPTIONS), format_func=lambda x: HORIZON_OPTIONS[x], value=14)
    n_splits    = st.slider("Backtest folds", 2, 5, 3)
    models_sel  = st.multiselect(
        "Models to train",
        options=list(MODEL_LABELS),
        default=list(MODEL_LABELS),
        format_func=lambda x: MODEL_LABELS[x],
    )
    run_btn = st.button("Run forecast", type="primary", use_container_width=True)

page_header(
    "Forecast",
    f"{selected_store} / {selected_product} · horizon {horizon}d",
)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not models_sel:
        st.warning("Select at least one model.")
    else:
        progress = st.progress(0, text="Starting...")
        status_box = st.empty()

        results = {}
        for i, model in enumerate(models_sel):
            progress.progress((i) / len(models_sel), text=f"Training {MODEL_LABELS[model]}...")
            try:
                res = run_forecast_pipeline(
                    store_id=selected_store, product_id=selected_product,
                    model_names=[model], horizon=horizon, n_backtest_splits=n_splits,
                )
                results.update(res)
                if model in res and "metrics" in res[model]:
                    m = res[model]["metrics"]
                    status_box.success(
                        f"**{MODEL_LABELS[model]} complete** · MAE={m.get('MAE', 0):.2f} · "
                        f"WAPE={m.get('WAPE', 0):.1%}"
                    )
            except Exception as exc:
                status_box.error(f"{MODEL_LABELS[model]} failed: {exc}")

        progress.progress(1.0, text="Done!")
        _get_runs.clear()
        st.rerun()

# ── Load saved runs ───────────────────────────────────────────────────────────
runs = _get_runs(selected_store, selected_product)
history_df = _get_history(selected_store, selected_product)

if not runs:
    st.info("No forecast runs yet. Configure settings in the sidebar and click **Run forecast**.")
    st.stop()

# ── Model leaderboard ─────────────────────────────────────────────────────────
st.subheader("Model comparison")

runs_df = pd.DataFrame(runs)
latest_per_model = (
    runs_df.sort_values("created", ascending=False)
           .drop_duplicates("model", keep="first")
           .sort_values("mae")
)

# Colour-code best model
best_model = latest_per_model.iloc[0]["model"] if not latest_per_model.empty else None

metric_cols = st.columns(len(latest_per_model))
for idx, (_, row) in enumerate(latest_per_model.iterrows()):
    with metric_cols[idx]:
        color = MODEL_COLORS.get(row["model"], BLUE)
        is_best = row["model"] == best_model
        border_style = f"border: 2px solid {color};" if is_best else f"border: 1px solid #2A2B2F;"
        st.markdown(
            f"""
            <div style="background:#101114; {border_style} border-radius:5px; padding:14px 16px;">
                <div style="color:{color}; font-size:0.7rem; font-weight:700;
                            letter-spacing:0.1em; text-transform:uppercase;">
                    {'BEST · ' if is_best else ''}{MODEL_LABELS.get(row['model'], row['model'])}
                </div>
                <div style="font-size:1.4rem; font-weight:700; color:#f1f5f9; margin-top:4px;">
                    {_format_metric(row['mae'], '.2f')}
                </div>
                <div style="color:#94a3b8; font-size:0.75rem;">MAE</div>
                <div style="margin-top:6px; font-size:0.8rem; color:#cbd5e1;">
                    WAPE {_format_metric(row['wape'], '.1%')} &nbsp;·&nbsp; RMSE {_format_metric(row['rmse'], '.2f')}
                </div>
                <div style="font-size:0.75rem; color:#94a3b8;">
                    Cov80 {_format_metric(row['cov80'], '.0%')}  Cov95 {_format_metric(row['cov95'], '.0%')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Future forecast chart (all models overlaid) ───────────────────────────────
st.subheader("Future forecasts")

fig = go.Figure()

# Historical (last 60 days for readability)
if not history_df.empty:
    recent = history_df.sort_values("date").tail(90)
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["units_sold"],
        mode="lines", name="Actual (last 90d)",
        line={"color": BLUE, "width": 1.5}, opacity=0.8,
    ))

# Each model's latest forecast
for _, row in latest_per_model.iterrows():
    model = row["model"]
    color = MODEL_COLORS.get(model, BLUE)
    with get_db() as s:
        fc = get_forecasts_dataframe(s, int(row["id"]))
    if fc.empty:
        continue
    fc["forecast_date"] = pd.to_datetime(fc["forecast_date"])

    # PI band (80%)
    if fc["lower_80"].notna().any():
        fig.add_trace(go.Scatter(
            x=fc["forecast_date"], y=fc["upper_80"], mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc["forecast_date"], y=fc["lower_80"], mode="lines",
            line={"width": 0}, fill="tonexty",
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)",
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=fc["forecast_date"], y=fc["point_forecast"],
        mode="lines+markers",
        name=MODEL_LABELS.get(model, model),
        line={"color": color, "width": 2, "dash": "dash"},
        marker={"size": 5},
    ))

fig.update_layout(
    template=PLOTLY_TEMPLATE, height=420,
    margin={"l": 0, "r": 0, "t": 30, "b": 0},
    yaxis_title="Units sold", hovermode="x unified",
    legend={"orientation": "h", "y": 1.10},
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Shaded regions = 80% prediction intervals. Intervals are conformal for statistical models; residual-quantile for LGBM.")

# ── All runs history table ────────────────────────────────────────────────────
with st.expander("All forecast runs for this series"):
    df_show = runs_df.copy()
    df_show["model"] = df_show["model"].str.upper()
    df_show["mae"]   = df_show["mae"].map(lambda x: f"{x:.2f}" if x else "—")
    df_show["wape"]  = df_show["wape"].map(lambda x: f"{x:.1%}" if x else "—")
    df_show["created"] = df_show["created"].map(lambda x: x.strftime("%Y-%m-%d %H:%M") if x else "—")
    st.dataframe(
        df_show[["id", "model", "cutoff", "horizon", "mae", "wape", "created"]].rename(
            columns={"id": "Run", "model": "Model", "cutoff": "Cutoff",
                     "horizon": "Horizon (d)", "mae": "MAE", "wape": "WAPE", "created": "Created"}
        ),
        hide_index=True, use_container_width=True,
    )
