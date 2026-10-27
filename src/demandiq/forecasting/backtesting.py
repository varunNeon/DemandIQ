"""
Rolling-origin backtesting for DemandIQ.

Implements time-series cross-validation (rolling-origin or expanding-window).
CRITICAL: Training windows ALWAYS precede their validation windows.
Random train/test splitting is NOT used for time-series — it causes
data leakage and produces misleadingly optimistic results.

Why random splitting is wrong for time series:
- Future observations in the training set would let the model "see" patterns
  it could not know at prediction time.
- This artificially inflates accuracy metrics.
- In practice, the model would fail in deployment because it was evaluated
  on a fundamentally different task than it was tested on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from demandiq.forecasting.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single backtest fold."""

    fold: int
    train_end_date: object  # date
    val_start_date: object  # date
    val_end_date: object  # date
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


@dataclass
class BacktestSummary:
    """Aggregated results across all folds."""

    model_name: str
    n_folds: int
    horizon: int
    folds: list[BacktestResult] = field(default_factory=list)

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Average metrics across all folds."""
        if not self.folds:
            return {}
        all_keys = set()
        for fold in self.folds:
            all_keys.update(fold.metrics.keys())
        result = {}
        for key in all_keys:
            vals = [f.metrics[key] for f in self.folds if key in f.metrics and not np.isnan(f.metrics[key])]
            result[key] = float(np.mean(vals)) if vals else float("nan")
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Return fold-level metrics as a DataFrame."""
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
        """Return concatenated actual vs predicted values across all folds."""
        rows = []
        for fold in self.folds:
            for i, (a, p) in enumerate(zip(fold.actual, fold.predicted)):
                rows.append({
                    "fold": fold.fold,
                    "horizon_step": i + 1,
                    "actual": a,
                    "predicted": p,
                    "lower_80": fold.lower_80[i] if fold.lower_80 is not None else None,
                    "upper_80": fold.upper_80[i] if fold.upper_80 is not None else None,
                    "lower_95": fold.lower_95[i] if fold.lower_95 is not None else None,
                    "upper_95": fold.upper_95[i] if fold.upper_95 is not None else None,
                })
        return pd.DataFrame(rows)


def rolling_origin_backtest(
    series: pd.Series,
    forecaster_factory: Callable,
    horizon: int = 14,
    n_splits: int = 3,
    min_train_size: int = 60,
    step_size: int | None = None,
    dates: pd.Series | None = None,
    series_df: pd.DataFrame | None = None,
) -> BacktestSummary:
    """
    Perform rolling-origin cross-validation.

    Parameters
    ----------
    series : Time-ordered demand series (must be sorted by date).
    forecaster_factory : Callable that returns a fresh forecaster instance.
                         Called once per fold.
    horizon : Forecast horizon in periods.
    n_splits : Number of validation folds.
    min_train_size : Minimum training observations required.
    step_size : How many periods to advance between folds.
               Default: horizon (non-overlapping validation windows).
    dates : Optional date index aligned with series.
    series_df : Optional full DataFrame (used for ML models that need features).

    Returns
    -------
    BacktestSummary with per-fold results and aggregated metrics.
    """
    n = len(series)
    step_size = step_size or horizon

    # Compute fold start points from the end backwards
    # Last fold: train up to n - horizon, validate last horizon steps
    fold_ends = []
    end = n - horizon
    for _ in range(n_splits):
        if end < min_train_size:
            break
        fold_ends.append(end)
        end -= step_size

    if not fold_ends:
        logger.warning(
            "Backtest: no valid folds could be constructed "
            "(n=%d, horizon=%d, min_train_size=%d, n_splits=%d).",
            n, horizon, min_train_size, n_splits,
        )
        return BacktestSummary(
            model_name=forecaster_factory().name,
            n_folds=0,
            horizon=horizon,
        )

    fold_ends = sorted(fold_ends)  # chronological order
    summary = BacktestSummary(
        model_name=forecaster_factory().name,
        n_folds=len(fold_ends),
        horizon=horizon,
    )

    for fold_idx, train_end in enumerate(fold_ends):
        train_series = series.iloc[:train_end]
        val_series = series.iloc[train_end : train_end + horizon]

        if len(val_series) < horizon:
            logger.debug("Fold %d: validation window too short (%d < %d). Skipping.", fold_idx, len(val_series), horizon)
            continue

        # Dates for logging
        train_end_date = dates.iloc[train_end - 1] if dates is not None else train_end
        val_start_date = dates.iloc[train_end] if dates is not None else train_end
        val_end_date = dates.iloc[train_end + horizon - 1] if dates is not None else train_end + horizon - 1

        forecaster = forecaster_factory()

        try:
            # For ML models that accept a DataFrame, pass the full feature set
            if series_df is not None and hasattr(forecaster, "fit") and "series_df" in forecaster.fit.__code__.co_varnames:
                train_df = series_df.iloc[:train_end]
                forecaster.fit(train_df)
            else:
                forecaster.fit(train_series)

            # Get predictions with intervals if supported
            if hasattr(forecaster, "predict_with_intervals"):
                preds_dict = forecaster.predict_with_intervals(horizon)
                predicted = preds_dict["point"]
                lower_80 = preds_dict.get("lower_80")
                upper_80 = preds_dict.get("upper_80")
                lower_95 = preds_dict.get("lower_95")
                upper_95 = preds_dict.get("upper_95")
            else:
                predicted = forecaster.predict(horizon)
                lower_80 = upper_80 = lower_95 = upper_95 = None

        except Exception as exc:
            logger.warning("Backtest fold %d failed for %s: %s", fold_idx, forecaster_factory().name, exc)
            continue

        actual = val_series.values
        predicted = np.maximum(np.array(predicted[: len(actual)]), 0.0)

        metrics = compute_all_metrics(
            actual=actual,
            predicted=predicted,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            train_actual=train_series.values,
            seasonality=7,
        )

        fold_result = BacktestResult(
            fold=fold_idx,
            train_end_date=train_end_date,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
            n_train=len(train_series),
            n_val=len(val_series),
            model_name=forecaster_factory().name,
            actual=actual,
            predicted=predicted,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            metrics=metrics,
            train_actual=train_series.values,
        )
        summary.folds.append(fold_result)
        logger.debug(
            "Fold %d: train=%d, val=%d, MAE=%.2f",
            fold_idx, len(train_series), len(val_series), metrics.get("MAE", float("nan")),
        )

    return summary
