"""
Rolling-origin backtesting for DemandIQ.

Time-series cross-validation with expanding training windows.
Training windows ALWAYS precede their validation windows; random splits are
never used because they let a model see the future and inflate accuracy.

Prediction intervals are evaluated honestly as well. For fold k, intervals
are calibrated only on residuals from earlier folds whose validation windows
were fully observed before fold k's forecast origin (see
`horizon_conformal_bands`). Interval coverage is therefore measured on data
the calibration never saw.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from demandiq.forecasting.intervals import horizon_conformal_bands
from demandiq.forecasting.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single backtest fold."""

    fold: int
    train_end_date: object
    val_start_date: object
    val_end_date: object
    n_train: int
    n_val: int
    model_name: str
    actual: np.ndarray
    predicted: np.ndarray
    lower_80: np.ndarray | None = None
    upper_80: np.ndarray | None = None
    lower_95: np.ndarray | None = None
    upper_95: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    train_actual: np.ndarray | None = None
    train_end_index: int = 0
    val_dates: pd.DatetimeIndex | None = None


@dataclass
class BacktestSummary:
    """Aggregated results across all folds."""

    model_name: str
    n_folds: int
    horizon: int
    folds: list[BacktestResult] = field(default_factory=list)

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Average each metric over the folds where it was computed."""
        if not self.folds:
            return {}
        keys: set[str] = set()
        for fold in self.folds:
            keys.update(fold.metrics.keys())
        result = {}
        for key in keys:
            vals = [f.metrics[key] for f in self.folds if key in f.metrics and np.isfinite(f.metrics[key])]
            result[key] = float(np.mean(vals)) if vals else float("nan")
        return result

    def residual_matrix(self) -> np.ndarray:
        """Signed residuals (actual - predicted), shape (n_folds, horizon)."""
        if not self.folds:
            return np.empty((0, self.horizon))
        return np.vstack([np.asarray(f.actual, float) - np.asarray(f.predicted, float) for f in self.folds])

    def prediction_matrix(self) -> np.ndarray:
        """Point forecasts per fold, shape (n_folds, horizon)."""
        if not self.folds:
            return np.empty((0, self.horizon))
        return np.vstack([np.asarray(f.predicted, float) for f in self.folds])

    def to_dataframe(self) -> pd.DataFrame:
        """Fold-level metrics as a DataFrame."""
        rows = []
        for fold in self.folds:
            row = {
                "fold": fold.fold,
                "model": self.model_name,
                "train_end": fold.train_end_date,
                "val_start": fold.val_start_date,
                "val_end": fold.val_end_date,
                "n_train": fold.n_train,
                "n_val": fold.n_val,
            }
            row.update(fold.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def actuals_vs_predicted(self) -> pd.DataFrame:
        """Long table of actual vs predicted values across all folds."""
        rows = []
        for fold in self.folds:
            for i, (a, p) in enumerate(zip(fold.actual, fold.predicted)):
                rows.append({
                    "fold": fold.fold,
                    "date": fold.val_dates[i] if fold.val_dates is not None else None,
                    "horizon_step": i + 1,
                    "actual": a,
                    "predicted": p,
                    "lower_80": fold.lower_80[i] if fold.lower_80 is not None else None,
                    "upper_80": fold.upper_80[i] if fold.upper_80 is not None else None,
                    "lower_95": fold.lower_95[i] if fold.lower_95 is not None else None,
                    "upper_95": fold.upper_95[i] if fold.upper_95 is not None else None,
                })
        return pd.DataFrame(rows)


def _fold_origins(n: int, horizon: int, n_splits: int, step_size: int, min_train_size: int) -> list[int]:
    """Training-end indices (exclusive), chronological."""
    ends = []
    end = n - horizon
    for _ in range(n_splits):
        if end < min_train_size:
            break
        ends.append(end)
        end -= step_size
    return sorted(ends)


def rolling_origin_backtest(
    series: pd.Series,
    forecaster_factory: Callable,
    horizon: int = 14,
    n_splits: int = 10,
    min_train_size: int = 60,
    step_size: int | None = 7,
    dates: pd.Series | None = None,
    series_df: pd.DataFrame | None = None,
    levels: list[float] | None = None,
    min_calibration_folds: int = 3,
) -> BacktestSummary:
    """
    Rolling-origin cross-validation with sequential conformal intervals.

    Parameters
    ----------
    series : Demand series sorted by date (a DatetimeIndex is used for fold dates).
    forecaster_factory : Returns a fresh forecaster per fold.
    horizon : Forecast horizon in days.
    n_splits : Maximum number of folds.
    min_train_size : Minimum training observations in the first fold.
    step_size : Days between fold origins (default 7).
    dates : Optional dates aligned with `series` (else taken from its index).
    series_df : Full frame [date, units_sold, onpromotion, is_holiday] for
        models with `requires_frame = True` (e.g. LightGBM).
    levels : Interval levels to evaluate (default [0.80, 0.95]).
    min_calibration_folds : Earlier, fully observed folds needed before a
        fold gets intervals.
    """
    levels = levels or [0.80, 0.95]
    n = len(series)
    step_size = step_size or horizon
    model_name = forecaster_factory().name

    if dates is None and isinstance(series.index, pd.DatetimeIndex):
        dates = pd.Series(series.index)

    origins = _fold_origins(n, horizon, n_splits, step_size, min_train_size)
    summary = BacktestSummary(model_name=model_name, n_folds=0, horizon=horizon)
    if not origins:
        logger.warning(
            "Backtest: no valid folds (n=%d, horizon=%d, min_train_size=%d).",
            n, horizon, min_train_size,
        )
        return summary

    # --- Pass 1: point forecasts for every fold -------------------------
    for fold_idx, train_end in enumerate(origins):
        train_series = series.iloc[:train_end]
        val_series = series.iloc[train_end : train_end + horizon]
        forecaster = forecaster_factory()
        try:
            if getattr(forecaster, "requires_frame", False) and series_df is not None:
                forecaster.fit(series_df.iloc[:train_end])
                future_exog = series_df.iloc[train_end : train_end + horizon]
                predicted = forecaster.predict(horizon, future_exog=future_exog)
            else:
                forecaster.fit(train_series)
                predicted = forecaster.predict(horizon)
        except Exception as exc:
            logger.warning("Backtest fold %d failed for %s: %s", fold_idx, model_name, exc)
            continue

        actual = val_series.to_numpy(dtype=float)
        predicted = np.maximum(np.asarray(predicted, dtype=float)[: len(actual)], 0.0)
        val_dates = (
            pd.DatetimeIndex(dates.iloc[train_end : train_end + horizon]) if dates is not None else None
        )
        summary.folds.append(BacktestResult(
            fold=fold_idx,
            train_end_date=dates.iloc[train_end - 1] if dates is not None else train_end,
            val_start_date=dates.iloc[train_end] if dates is not None else train_end,
            val_end_date=dates.iloc[train_end + horizon - 1] if dates is not None else train_end + horizon - 1,
            n_train=len(train_series),
            n_val=len(val_series),
            model_name=model_name,
            actual=actual,
            predicted=predicted,
            train_actual=train_series.to_numpy(dtype=float),
            train_end_index=train_end,
            val_dates=val_dates,
        ))

    summary.n_folds = len(summary.folds)

    # --- Pass 2: sequential conformal intervals + metrics ---------------
    for k, fold in enumerate(summary.folds):
        calibration = [
            f for f in summary.folds[:k]
            if f.train_end_index + horizon <= fold.train_end_index
        ]
        bands: dict[str, np.ndarray] = {}
        if len(calibration) >= min_calibration_folds:
            residuals = np.vstack([f.actual - f.predicted for f in calibration])
            cal_preds = np.vstack([f.predicted for f in calibration])
            bands = horizon_conformal_bands(
                fold.predicted, residuals, levels, calibration_predictions=cal_preds
            )
            fold.lower_80, fold.upper_80 = bands.get("lower_80"), bands.get("upper_80")
            fold.lower_95, fold.upper_95 = bands.get("lower_95"), bands.get("upper_95")

        fold.metrics = compute_all_metrics(
            actual=fold.actual,
            predicted=fold.predicted,
            lower_80=fold.lower_80,
            upper_80=fold.upper_80,
            lower_95=fold.lower_95,
            upper_95=fold.upper_95,
            train_actual=fold.train_actual,
            seasonality=7,
        )

    return summary
