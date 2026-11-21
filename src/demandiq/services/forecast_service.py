"""
Forecast service for DemandIQ.

    data → rolling-origin backtest → conformal intervals → future forecast → database

`forecast_series` is pure (no database), so the Streamlit app, the benchmark
script and tests can all call it. `run_forecast_pipeline` wraps it with
database loading and saving.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from demandiq.config import settings
from demandiq.data.calendar import future_calendar
from demandiq.database.connection import get_db
from demandiq.database.models import Forecast, ForecastRun, ReplenishmentRecommendation
from demandiq.database.repositories import get_sales_dataframe
from demandiq.forecasting.backtesting import BacktestSummary, rolling_origin_backtest
from demandiq.forecasting.baselines import NaiveForecaster, SeasonalNaiveForecaster
from demandiq.forecasting.intervals import horizon_conformal_bands
from demandiq.forecasting.ml_models import LGBMForecaster
from demandiq.forecasting.statistical_models import ETSForecaster

logger = logging.getLogger(__name__)

MODEL_LABELS: dict[str, str] = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal naive",
    "ets": "ETS (Holt-Winters)",
    "lgbm": "LightGBM",
}
AVAILABLE_MODELS = list(MODEL_LABELS)


def get_forecaster(model_name: str, horizon: int):
    """Instantiate a forecaster by name."""
    if model_name == "lgbm":
        return LGBMForecaster(horizon=horizon)
    if model_name == "ets":
        return ETSForecaster(seasonal_period=7)
    if model_name == "seasonal_naive":
        return SeasonalNaiveForecaster(period=7)
    if model_name == "naive":
        return NaiveForecaster()
    raise ValueError(f"Unknown model: {model_name}")


@dataclass
class ModelResult:
    """Everything produced for one model on one series."""

    model_name: str
    backtest: BacktestSummary
    metrics: dict[str, float]
    future: pd.DataFrame
    feature_importance: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model_name, self.model_name)


def prepare_series_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a sales frame to [date, units_sold, onpromotion, is_holiday], sorted."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    for col in ("onpromotion", "is_holiday"):
        out[col] = out[col].astype(int) if col in out.columns else 0
    out["units_sold"] = out["units_sold"].astype(float)
    return out[["date", "units_sold", "onpromotion", "is_holiday"]]


def future_intervals(point: np.ndarray, backtest: BacktestSummary, levels: list[float]) -> dict[str, np.ndarray]:
    """Calibrate future intervals on every backtest fold (all fully observed)."""
    residuals = backtest.residual_matrix()
    if residuals.size == 0:
        return {"point": np.maximum(np.asarray(point, float), 0.0)}
    return horizon_conformal_bands(
        point, residuals, levels, calibration_predictions=backtest.prediction_matrix()
    )


def forecast_series(
    df: pd.DataFrame,
    model_names: list[str] | None = None,
    horizon: int | None = None,
    n_splits: int | None = None,
    promo_plan: pd.DataFrame | None = None,
) -> dict[str, ModelResult]:
    """
    Backtest and forecast one store-product series with each model.

    Parameters
    ----------
    df : Sales frame for one series with [date, units_sold] and optionally
        [onpromotion, is_holiday].
    promo_plan : Optional frame with `onpromotion` for the future days, used
        by models that understand promotions (LightGBM).
    """
    model_names = model_names or settings.ENABLED_MODELS
    horizon = horizon or settings.DEFAULT_HORIZON
    n_splits = n_splits or settings.BACKTEST_SPLITS

    frame = prepare_series_frame(df)
    series = frame.set_index("date")["units_sold"]
    if len(series) < settings.MIN_TRAIN_PERIODS + horizon:
        raise ValueError(
            f"Need at least {settings.MIN_TRAIN_PERIODS + horizon} days of history, got {len(series)}."
        )

    last_date = series.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    exog = future_calendar(last_date, horizon)
    if promo_plan is not None and "onpromotion" in promo_plan.columns:
        exog["onpromotion"] = promo_plan["onpromotion"].astype(int).to_numpy()[:horizon]

    results: dict[str, ModelResult] = {}
    for name in model_names:
        if name not in MODEL_LABELS:
            logger.warning("Unknown model '%s'. Skipping.", name)
            continue

        backtest = rolling_origin_backtest(
            series=series,
            forecaster_factory=lambda name=name: get_forecaster(name, horizon),
            horizon=horizon,
            n_splits=n_splits,
            min_train_size=settings.MIN_TRAIN_PERIODS,
            step_size=7,
            series_df=frame,
            levels=settings.PI_LEVELS,
        )

        model = get_forecaster(name, horizon)
        if getattr(model, "requires_frame", False):
            model.fit(frame)
            point = model.predict(horizon, future_exog=exog)
        else:
            model.fit(series)
            point = model.predict(horizon)

        bands = future_intervals(point, backtest, settings.PI_LEVELS)
        future = pd.DataFrame({"date": future_dates, "point_forecast": bands["point"]})
        for key in ("lower_80", "upper_80", "lower_95", "upper_95"):
            future[key] = bands.get(key)

        importance = model.feature_importance() if hasattr(model, "feature_importance") else None
        results[name] = ModelResult(
            model_name=name,
            backtest=backtest,
            metrics=backtest.mean_metrics,
            future=future,
            feature_importance=importance,
        )
    return results


def best_model(results: dict[str, ModelResult], metric: str = "WAPE") -> str | None:
    """Name of the model with the lowest backtest `metric`."""
    scored = {
        name: r.metrics.get(metric, np.nan)
        for name, r in results.items()
        if np.isfinite(r.metrics.get(metric, np.nan))
    }
    return min(scored, key=scored.get) if scored else None


def _none_if_nan(value) -> float | None:
    return None if value is None or not np.isfinite(value) else float(value)


def save_model_result(store_id: str, product_id: str, result: ModelResult, horizon: int, n_splits: int) -> int:
    """
    Save one model's run and future forecasts, replacing an identical earlier
    run (same series, model, training cutoff and horizon) so re-running the
    pipeline does not create duplicates.
    """
    training_cutoff = (result.future["date"].min() - pd.Timedelta(days=1)).date()
    metrics = result.metrics

    with get_db() as session:
        previous = (
            session.query(ForecastRun)
            .filter(
                ForecastRun.store_id == store_id,
                ForecastRun.product_id == product_id,
                ForecastRun.model_name == result.model_name,
                ForecastRun.training_cutoff == training_cutoff,
                ForecastRun.horizon_days == horizon,
            )
            .all()
        )
        for run in previous:
            session.query(ReplenishmentRecommendation).filter(
                ReplenishmentRecommendation.forecast_run_id == run.forecast_run_id
            ).delete()
            session.query(Forecast).filter(Forecast.forecast_run_id == run.forecast_run_id).delete()
            session.delete(run)
        session.flush()

        run = ForecastRun(
            model_name=result.model_name,
            training_cutoff=training_cutoff,
            horizon_days=horizon,
            store_id=store_id,
            product_id=product_id,
            model_config=json.dumps({
                "horizon": horizon,
                "n_backtest_splits": n_splits,
                "folds_evaluated": result.backtest.n_folds,
                "intervals": "horizon-aware conformal (rolling-origin residuals)",
            }),
            eval_mae=_none_if_nan(metrics.get("MAE")),
            eval_rmse=_none_if_nan(metrics.get("RMSE")),
            eval_wape=_none_if_nan(metrics.get("WAPE")),
            eval_mase=_none_if_nan(metrics.get("MASE")),
            eval_coverage_80=_none_if_nan(metrics.get("Coverage_80")),
            eval_coverage_95=_none_if_nan(metrics.get("Coverage_95")),
        )
        session.add(run)
        session.flush()

        for row in result.future.itertuples(index=False):
            session.add(Forecast(
                forecast_run_id=run.forecast_run_id,
                forecast_date=row.date.date(),
                store_id=store_id,
                product_id=product_id,
                point_forecast=float(row.point_forecast),
                lower_80=_none_if_nan(row.lower_80),
                upper_80=_none_if_nan(row.upper_80),
                lower_95=_none_if_nan(row.lower_95),
                upper_95=_none_if_nan(row.upper_95),
                is_future=True,
            ))
        return run.forecast_run_id


def run_forecast_pipeline(
    store_id: str,
    product_id: str,
    model_names: list[str] | None = None,
    horizon: int | None = None,
    n_backtest_splits: int | None = None,
) -> dict[str, dict]:
    """
    Load one series from the database, forecast it with each model, and save
    the runs. Returns {model_name: {run_id, metrics, future_forecast_df,
    backtest_summary, model_name}} or {model_name: {"error": ...}}.
    """
    horizon = horizon or settings.DEFAULT_HORIZON
    n_splits = n_backtest_splits or settings.BACKTEST_SPLITS

    with get_db() as session:
        df = get_sales_dataframe(session, store_id=store_id, product_id=product_id)
    if df.empty:
        logger.warning("No data found for store=%s product=%s", store_id, product_id)
        return {}

    try:
        results = forecast_series(df, model_names, horizon, n_splits)
    except ValueError as exc:
        logger.warning("Skipping %s/%s: %s", store_id, product_id, exc)
        return {}

    output: dict[str, dict] = {}
    for name, result in results.items():
        try:
            run_id = save_model_result(store_id, product_id, result, horizon, n_splits)
            output[name] = {
                "run_id": run_id,
                "model_name": name,
                "metrics": result.metrics,
                "future_forecast_df": result.future,
                "backtest_summary": result.backtest,
            }
        except Exception as exc:  # keep other models going
            logger.error("Saving %s for %s/%s failed: %s", name, store_id, product_id, exc, exc_info=True)
            output[name] = {"error": str(exc)}
    return output
