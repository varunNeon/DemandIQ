"""Forecast lab: run the models live, compare them, and inspect the backtest."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from demandiq.app import charts, data, theme
from demandiq.services.forecast_service import AVAILABLE_MODELS, MODEL_LABELS, best_model


def render() -> None:
    store_id = st.session_state["store_id"]
    product_id = st.session_state["product_id"]
    series = data.load_series(store_id, product_id)
    name = product_id.replace("_", " ").title()

    theme.masthead(
        "Forecast lab",
        f"Run four models on {name} at {store_id}, see how each would have done over the past "
        "twelve forecast windows, and plan a promotion to see how the forecast responds.",
    )

    with st.form("forecast_controls", border=False):
        c1, c2, c3 = st.columns([1.6, 0.9, 1.5], gap="medium")
        models = c1.multiselect(
            "Models", AVAILABLE_MODELS, default=AVAILABLE_MODELS, format_func=MODEL_LABELS.get,
        )
        horizon = c2.select_slider("Days ahead", options=[7, 14, 21, 28], value=14)
        last = series["date"].max()
        day_options = list(range(1, horizon + 1))
        promo_days = c3.multiselect(
            "Planned promotion days (LightGBM only)",
            day_options,
            format_func=lambda d: (last + pd.Timedelta(days=d)).strftime("%a %d %b"),
            help="LightGBM learned how promotions lift demand. Pick future days to see the effect.",
        )
        st.form_submit_button("Run forecast")

    if not models:
        st.info("Choose at least one model to run.")
        return

    with st.spinner("Running the backtest for each model. The first run takes a few seconds…"):
        results = data.run_forecasts(store_id, product_id, tuple(models), horizon,
                                     tuple(sorted(d for d in promo_days if d <= horizon)))

    winner = best_model(results) or models[0]
    shown_model = st.segmented_control(
        "Show forecast from",
        options=list(results),
        default=winner,
        format_func=lambda m: MODEL_LABELS[m] + (" (most accurate)" if m == winner else ""),
        key=f"shown_model_{store_id}_{product_id}",
    ) or winner
    res = results[shown_model]

    theme.chart_legend([
        ("Observed demand", theme.GOLD, False),
        (f"{res.label} forecast", theme.EMBER, False),
        ("80% and 95% intervals", theme.EMBER, True),
    ])
    st.plotly_chart(charts.fan_chart(series, res.future, history_days=70), use_container_width=True,
                    config=theme.PLOTLY_CONFIG)

    # ---- Comparison table ------------------------------------------------
    theme.section(
        "How the models compare on this series",
        f"Averaged over {res.backtest.n_folds} backtest windows of {horizon} days. Bias is the "
        "average over- (positive) or under- (negative) forecast in units per day.",
    )
    rows = []
    for m, r in sorted(results.items(), key=lambda kv: kv[1].metrics.get("WAPE", 9)):
        rows.append({
            "Model": r.label, "WAPE": r.metrics.get("WAPE"), "MAE": r.metrics.get("MAE"),
            "MASE": r.metrics.get("MASE"), "Bias": r.metrics.get("MFE"),
            "80% coverage": r.metrics.get("Coverage_80"), "95% coverage": r.metrics.get("Coverage_95"),
        })
    theme.table(
        pd.DataFrame(rows),
        formats={
            "WAPE": theme.pct, "MAE": lambda v: theme.num(v, 1), "MASE": lambda v: f"{v:.2f}",
            "Bias": lambda v: f"{v:+.1f}", "80% coverage": lambda v: theme.pct(v, 0),
            "95% coverage": lambda v: theme.pct(v, 0),
        },
        highlight_first=True,
    )

    # ---- Backtest replay -------------------------------------------------
    theme.section(
        f"Checking {res.label}'s homework",
        "Every red line is a two-week forecast made with only the data available at that moment, "
        "laid over what actually happened. Shaded bands are the 80% intervals, calibrated on "
        "earlier windows only, so the oldest windows have none while calibration data builds up.",
    )
    st.plotly_chart(charts.backtest_replay(series, res), use_container_width=True,
                    config=theme.PLOTLY_CONFIG)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Do the intervals hold?")
        st.caption("Share of actual days that fell inside each interval. Dotted lines mark the targets.")
        st.plotly_chart(charts.coverage_chart(results), use_container_width=True,
                        config=theme.PLOTLY_CONFIG)
    with right:
        lgbm = results.get("lgbm")
        if lgbm is not None and lgbm.feature_importance is not None:
            st.markdown("##### What LightGBM pays attention to")
            st.caption("Share of total split gain. Demand features are scaled by the recent level.")
            st.plotly_chart(charts.importance_chart(lgbm.feature_importance), use_container_width=True,
                            config=theme.PLOTLY_CONFIG)
        else:
            st.markdown("##### What LightGBM pays attention to")
            st.caption("Add LightGBM to the models above to see its feature importance.")

    # ---- Export ------------------------------------------------------------
    out = res.future.copy()
    out.insert(0, "model", shown_model)
    out.insert(0, "product_id", product_id)
    out.insert(0, "store_id", store_id)
    st.download_button(
        "Download this forecast (CSV)",
        out.to_csv(index=False).encode(),
        file_name=f"demandiq_{store_id}_{product_id}_{shown_model}.csv",
        mime="text/csv",
    )
