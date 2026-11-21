"""Overview: the headline result, a live forecast, and the model leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from demandiq.app import charts, data, theme
from demandiq.services.forecast_service import MODEL_LABELS


def _leaderboard(bench: pd.DataFrame) -> pd.DataFrame:
    wape = bench.pivot_table(index=["store_id", "product_id"], columns="model", values="WAPE")
    wins = wape.idxmin(axis=1).value_counts()
    board = (
        bench.groupby("model")[["WAPE", "MASE", "Coverage_80", "Coverage_95"]].mean()
        .assign(wins=lambda d: d.index.map(wins).fillna(0).astype(int))
        .sort_values("WAPE")
        .reset_index()
    )
    return board


def render() -> None:
    store_id = st.session_state["store_id"]
    product_id = st.session_state["product_id"]

    theme.masthead(
        "Forecasts you can order stock against.",
        "DemandIQ forecasts daily demand for every store and product, says how sure it is, "
        "and turns that into a reorder decision. Every accuracy figure on this site is measured "
        "on weeks the models never saw during training.",
    )

    bench_bundle = data.load_benchmark()
    if bench_bundle is not None:
        bench, by_h, meta = bench_bundle
        board = _leaderboard(bench)
        best = board.iloc[0]
        naive = board[board["model"] == "naive"].iloc[0]
        gain = 1 - best["WAPE"] / naive["WAPE"]
        theme.result_banner(
            f"{gain * 100:.0f}", "%",
            f"{MODEL_LABELS[best['model']]} cuts forecast error by {gain:.0%} against the naive "
            f"baseline across {meta['series']} store–product series.",
            f"Average WAPE {best['WAPE']:.1%} versus {naive['WAPE']:.1%}. Most accurate model on "
            f"{best['wins']} of {meta['series']} series. Measured with a {meta['backtest_folds']}-fold "
            f"rolling-origin backtest, forecasting {meta['horizon_days']} days ahead.",
        )

    # ---- Live forecast for the selected series ---------------------------
    series = data.load_series(store_id, product_id)
    product_name = product_id.replace("_", " ").title()
    theme.section(
        f"The next two weeks for {product_name} at {store_id}",
        "LightGBM forecast with 80% and 95% prediction intervals. Change the series in the sidebar.",
    )
    with st.spinner("Backtesting and forecasting this series…"):
        results = data.run_forecasts(store_id, product_id, ("lgbm",), 14)
    res = results["lgbm"]
    fut = res.future

    total = fut["point_forecast"].sum()
    lo, hi = fut["lower_80"].sum(), fut["upper_80"].sum()
    recent = series["units_sold"].tail(14).sum()
    change = total / recent - 1 if recent else np.nan
    theme.figures([
        (theme.num(total), "units expected over the next 14 days", "ember"),
        (f"{theme.num(lo)}–{theme.num(hi)}", "sum of the daily 80% bounds, a deliberately cautious span", ""),
        (f"{change:+.0%}" if np.isfinite(change) else "—", "against the last 14 days", ""),
        (theme.pct(res.metrics.get("WAPE")), "backtest error (WAPE) on this series", "gold"),
    ])
    theme.chart_legend([
        ("Observed demand", theme.GOLD, False),
        ("Forecast", theme.EMBER, False),
        ("80% interval", theme.EMBER, True),
        ("95% interval", theme.EMBER_SOFT, True),
    ])
    st.plotly_chart(charts.fan_chart(series, fut, history_days=91), use_container_width=True,
                    config=theme.PLOTLY_CONFIG)

    # ---- Leaderboard -------------------------------------------------------
    if bench_bundle is not None:
        theme.section(
            "Which model to trust",
            "Averages across every demo series. WAPE is total error as a share of total demand; "
            "MASE scales error by how well a same-weekday-last-week rule fits the training data. "
            "Lower is better for both. The band columns show how often reality landed inside the "
            "interval, against targets of 80% and 95%.",
        )
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            short = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "ets": "ETS", "lgbm": "LightGBM"}
            shown = board.assign(Model=board["model"].map(short))
            shown = shown[["Model", "WAPE", "MASE", "Coverage_80", "Coverage_95", "wins"]]
            shown.columns = ["Model", "WAPE", "MASE", "In 80% band", "In 95% band", "Best on"]
            theme.table(
                shown,
                formats={
                    "WAPE": theme.pct, "MASE": lambda v: f"{v:.2f}",
                    "In 80% band": lambda v: theme.pct(v, 0), "In 95% band": lambda v: theme.pct(v, 0),
                    "Best on": lambda v: f"{v} / {meta['series']}",
                },
                highlight_first=True,
            )
            theme.note(
                "Intervals run slightly narrow (about 75% and 89% observed). They are calibrated "
                "only on earlier weeks, so seasonal shifts the calibration has not yet seen widen "
                "the gap. The numbers are reported as measured rather than tuned to look nominal."
            )
        with right:
            st.markdown("##### Error by days ahead")
            st.caption(
                "LightGBM stays flat and lowest. The spike at day 8 is Christmas Day landing at "
                "that distance in one backtest window: methods that cannot see the calendar miss it."
            )
            st.plotly_chart(charts.horizon_error_chart(by_h), use_container_width=True,
                            config=theme.PLOTLY_CONFIG)

    # ---- Pipeline ------------------------------------------------------------
    theme.section("From sales history to a purchase order")
    theme.steps([
        ("Validate and load", "Daily sales, promotions and holidays are checked and stored in "
         "SQLite locally or PostgreSQL in production."),
        ("Backtest honestly", "Each model forecasts 12 past fortnights from data available at the "
         "time. No random splits, no peeking."),
        ("Quantify uncertainty", "Prediction intervals come from each model's own past misses at "
         "each distance ahead (conformal calibration)."),
        ("Decide what to order", "Interval widths set safety stock, a reorder point and an order "
         "quantity, tested against two years of demand."),
    ])
