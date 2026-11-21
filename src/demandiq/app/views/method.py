"""How it works: the method, its assumptions and its limits."""

from __future__ import annotations

import streamlit as st

from demandiq.app import data, theme


def render() -> None:
    theme.masthead(
        "How DemandIQ works",
        "The choices behind the numbers, and where they stop being trustworthy.",
    )
    bundle = data.load_benchmark()
    meta = bundle[2] if bundle else {"backtest_folds": 12, "horizon_days": 14, "series": 100}

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f"""
### Evaluation without leakage
Every model is scored with a **rolling-origin backtest**: {meta['backtest_folds']} forecast
origins, one week apart, each forecasting {meta['horizon_days']} days ahead using only the
history before that origin. Random train/test splits are never used, because they let a model
learn from the future.

Accuracy is reported as **WAPE** (total absolute error over total demand), **MASE** (error
relative to a seasonal-naive forecast) and **bias**. MAPE is avoided because it breaks on
zero-sales days.

### Four models, from simple to learned
- **Naive** repeats the last day. **Seasonal naive** repeats the same weekday last week. Any
  model worth its complexity must beat both.
- **ETS (Holt-Winters)** fits level, trend and weekly seasonality, choosing the variant by AIC.
- **LightGBM** is a single direct multi-horizon model. Each training row pairs a forecast
  origin with a day ahead. It sees recent demand, rolling levels and volatility, same-weekday
  history, last year's demand around the target date, and calendar facts known in advance
  (weekday, season, holidays, planned promotions). Demand inputs and the target are divided by
  the recent 28-day level, so the trees learn shapes rather than volumes and can follow trends.
""")
    with right:
        st.markdown("""
### Intervals calibrated on real misses
Prediction intervals use **horizon-aware conformal calibration**. For each distance ahead, the
model's past backtest errors at similar distances set the band. Errors are scaled by the square
root of the forecast level, because busier days miss by more in absolute terms. Signed
quantiles keep the bands asymmetric where demand is skewed.

Coverage is tested honestly: a backtest window only gets intervals from earlier windows that
had fully played out before its forecast origin.

### From forecast to order
The 80% interval is turned into a daily forecast-error standard deviation. Lead-time demand
and its spread set the **reorder point** (expected demand plus safety stock at the chosen
service level); the **order quantity** follows the economic order quantity. A lost-sales
simulator replays six months of real demand to show the cost of each service level.

### Known limits
- The demo data is **synthetic**: realistic patterns (weekly and annual seasonality,
  promotions, holidays, drifting levels, over-dispersed counts, stock-outs) but not real sales.
- Intervals currently come in slightly narrow (about 75% and 89% against 80% and 95% targets).
  Seasonal shifts that earlier calibration windows did not include are the main reason.
- Future promotions are assumed absent unless you plan them in the Forecast lab.
- Inventory costs, lead times and stock levels are operating assumptions, not facts learned
  from the data.
""")

    theme.section("Built with")
    theme.figures([
        ("Python", "pandas, NumPy, statsmodels, LightGBM, SciPy", ""),
        ("SQLAlchemy", "SQLite locally, PostgreSQL in production", ""),
        ("FastAPI", "read API for the Next.js workspace", ""),
        ("Streamlit", "this interactive app, with Plotly charts", "gold"),
    ])
