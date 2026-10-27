"""
Forecast service for DemandIQ.

Orchestrates: data retrieval → model training → backtesting
             → future forecast generation → database storage.

Separates all business logic from the Streamlit UI layer.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from demandiq.config import settings
from demandiq.database.connection import get_db
from demandiq.database.models import Forecast, ForecastRun
from demandiq.database.repositories import (
    get_forecasts_dataframe,
    get_latest_forecast_run,
    get_sales_dataframe,
)
from demandiq.forecasting.backtesting import rolling_origin_backtest
from demandiq.forecasting.baselines import NaiveForecaster, SeasonalNaiveForecaster
from demandiq.forecasting.intervals import ConformalIntervalWrapper
from demandiq.forecasting.metrics import compute_all_metrics
from demandiq.forecasting.ml_models import LGBMForecaster
from demandiq.forecasting.statistical_models import ETSForecaster

logger = logging.getLogger(__name__)

FORECASTER_MAP = {
    "naive": lambda: NaiveForecaster(),
    "seasonal_naive": lambda: SeasonalNaiveForecaster(period=7),
    "ets": lambda: ETSForecaster(seasonal_period=7),
    "lgbm": lambda h: LGBMForecaster(horizon=h),
}


def _get_forecaster(model_name: str, horizon: int):
    """Instantiate the correct forecaster by name."""
    if model_name == "lgbm":
        return LGBMForecaster(horizon=horizon)
    elif model_name == "ets":
        return ETSForecaster(seasonal_period=7)
    elif model_name == "seasonal_naive":
        return SeasonalNaiveForecaster(period=7)
    elif model_name == "naive":
        return NaiveForecaster()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_forecast_pipeline(
    store_id: str,
    product_id: str,
    model_names: list[str] | None = None,
    horizon: int | None = None,
    n_backtest_splits: int | None = None,
    force_retrain: bool = False,
) -> dict[str, dict]:
    """
    Full pipeline for one store-product series:
        1. Load data from DB
        2. Run rolling-origin backtest for each model
        3. Generate future forecasts
        4. Store forecast runs and forecasts in DB
        5. Return summary dict per model

    Parameters
    ----------
    store_id, product_id : Series identifiers.
    model_names : List of model names to train. Default: settings.ENABLED_MODELS.
    horizon : Forecast horizon in days. Default: settings.DEFAULT_HORIZON.
    n_backtest_splits : Number of backtest folds. Default: settings.BACKTEST_SPLITS.
    force_retrain : If True, retrain even if a recent run exists.

    Returns
    -------
    Dict mapping model_name → {run_id, backtest_summary, future_forecast_df, metrics}
    """
    model_names = model_names or settings.ENABLED_MODELS
    horizon = horizon or settings.DEFAULT_HORIZON
    n_splits = n_backtest_splits or settings.BACKTEST_SPLITS

    with get_db() as session:
        df = get_sales_dataframe(session, store_id=store_id, product_id=product_id)

    if df.empty:
        logger.warning("No data found for store=%s product=%s", store_id, product_id)
        return {}

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    series = df.set_index("date")["units_sold"]

    if len(series) < settings.MIN_TRAIN_PERIODS:
        logger.warning(
            "Insufficient data for %s/%s: %d rows (need %d). Skipping.",
            store_id, product_id, len(series), settings.MIN_TRAIN_PERIODS,
        )
        return {}

    results = {}

    for model_name in model_names:
        if model_name not in FORECASTER_MAP and model_name != "lgbm":
            logger.warning("Unknown model '%s'. Skipping.", model_name)
            continue

        try:
            results[model_name] = _train_and_forecast_model(
                model_name=model_name,
                series=series,
                df=df,
                store_id=store_id,
                product_id=product_id,
                horizon=horizon,
                n_splits=n_splits,
            )
        except Exception as exc:
            logger.error("Pipeline failed for model=%s, store=%s, product=%s: %s",
                         model_name, store_id, product_id, exc, exc_info=True)
            results[model_name] = {"error": str(exc)}

    return results


def _train_and_forecast_model(
    model_name: str,
    series: pd.Series,
    df: pd.DataFrame,
    store_id: str,
    product_id: str,
    horizon: int,
    n_splits: int,
) -> dict:
    """Train one model, run backtest, generate future forecast, save to DB."""
    logger.info("Training %s for %s/%s (horizon=%d)", model_name, store_id, product_id, horizon)

    # --- Backtest ---
    def factory():
        return _get_forecaster(model_name, horizon)

    series_df = df if model_name == "lgbm" else None

    backtest = rolling_origin_backtest(
        series=series,
        forecaster_factory=factory,
        horizon=horizon,
        n_splits=n_splits,
        min_train_size=settings.MIN_TRAIN_PERIODS,
        series_df=series_df,
    )

    mean_metrics = backtest.mean_metrics

    # --- Future forecast ---
    future_forecaster = _get_forecaster(model_name, horizon)

    if model_name == "lgbm":
        future_forecaster.fit(df)
        future_result = _prediction_intervals_from_backtest(
            point=future_forecaster.predict(horizon),
            backtest=backtest,
            levels=settings.PI_LEVELS,
        )
    else:
        future_forecaster.fit(series)
        # Wrap with conformal prediction. This expects simple series-based
        # forecasters; ML models that need a full feature frame are handled above.
        conformal = ConformalIntervalWrapper(future_forecaster, calibration_frac=0.2)
        conformal.fit(series)
        future_result = conformal.predict_with_intervals(horizon, levels=settings.PI_LEVELS)

    future_point = future_result["point"]

    # Generate future dates
    last_date = series.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # --- Save to DB ---
    training_cutoff = last_date.date()

    with get_db() as session:
        run = ForecastRun(
            model_name=model_name,
            training_cutoff=training_cutoff,
            horizon_days=horizon,
            store_id=store_id,
            product_id=product_id,
            model_config=json.dumps({"horizon": horizon, "n_backtest_splits": n_splits}),
            eval_mae=mean_metrics.get("MAE"),
            eval_rmse=mean_metrics.get("RMSE"),
            eval_wape=mean_metrics.get("WAPE"),
            eval_mase=mean_metrics.get("MASE"),
            eval_coverage_80=mean_metrics.get("Coverage_80"),
            eval_coverage_95=mean_metrics.get("Coverage_95"),
        )
        session.add(run)
        session.flush()  # Get run_id

        # Save future forecasts
        for i, fdate in enumerate(future_dates):
            fc = Forecast(
                forecast_run_id=run.forecast_run_id,
                forecast_date=fdate.date(),
                store_id=store_id,
                product_id=product_id,
                point_forecast=max(float(future_point[i]), 0.0),
                lower_80=float(future_result.get("lower_80", [None] * horizon)[i]) if "lower_80" in future_result else None,
                upper_80=float(future_result.get("upper_80", [None] * horizon)[i]) if "upper_80" in future_result else None,
                lower_95=float(future_result.get("lower_95", [None] * horizon)[i]) if "lower_95" in future_result else None,
                upper_95=float(future_result.get("upper_95", [None] * horizon)[i]) if "upper_95" in future_result else None,
                is_future=True,
            )
            session.add(fc)

        run_id = run.forecast_run_id

    # Build future forecast DataFrame for return
    future_df = pd.DataFrame({
        "date": future_dates,
        "point_forecast": np.maximum(future_point, 0.0),
        "lower_80": future_result.get("lower_80"),
        "upper_80": future_result.get("upper_80"),
        "lower_95": future_result.get("lower_95"),
        "upper_95": future_result.get("upper_95"),
    })

    logger.info(
        "Model %s: MAE=%.2f, RMSE=%.2f, WAPE=%.3f",
        model_name, mean_metrics.get("MAE", 0), mean_metrics.get("RMSE", 0), mean_metrics.get("WAPE", 0),
    )

    return {
        "run_id": run_id,
        "backtest_summary": backtest,
        "future_forecast_df": future_df,
        "metrics": mean_metrics,
        "model_name": model_name,
    }


def _prediction_intervals_from_backtest(
    point: np.ndarray,
    backtest,
    levels: list[float],
) -> dict[str, np.ndarray]:
    """
    Build simple residual-quantile prediction intervals from temporal backtests.

    The residuals come only from historical validation folds whose training
    windows precede their validation windows. This is not a full conditional
    conformal method, but it avoids future leakage and gives ML forecasts a
    transparent uncertainty estimate until a richer calibration wrapper exists.
    """
    point = np.maximum(np.asarray(point, dtype=float), 0.0)
    result: dict[str, np.ndarray] = {"point": point}

    residuals = []
    for fold in backtest.folds:
        residuals.extend(np.asarray(fold.actual, dtype=float) - np.asarray(fold.predicted, dtype=float))

    if not residuals:
        return result

    abs_residuals = np.abs(np.asarray(residuals, dtype=float))
    for level in levels:
        q = float(np.quantile(abs_residuals, level))
        label = f"{int(level * 100)}"
        result[f"lower_{label}"] = np.maximum(point - q, 0.0)
        result[f"upper_{label}"] = point + q

    return result
