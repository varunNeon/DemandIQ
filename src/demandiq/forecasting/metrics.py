"""
Evaluation metrics for DemandIQ forecasting.

All metrics are implemented from scratch to ensure correctness
and zero-demand safety.

Reference definitions are documented per-function.
MAPE is intentionally excluded as primary metric because it is undefined
or misleading when actual demand is zero or near-zero.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _validate_inputs(actual: np.ndarray, predicted: np.ndarray) -> None:
    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}")
    if len(actual) == 0:
        raise ValueError("Empty arrays provided.")


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    _validate_inputs(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    _validate_inputs(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = sum(|actual - predicted|) / sum(|actual|)

    Unlike MAPE, WAPE is defined even when individual actual values are zero,
    as long as total actual demand is non-zero.
    Returns NaN if total actual demand is zero.
    """
    _validate_inputs(actual, predicted)
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        logger.warning("WAPE: total actual demand is zero — returning NaN.")
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / total_actual)


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    naive_errors: np.ndarray | None = None,
    seasonality: int = 1,
    train_actual: np.ndarray | None = None,
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE(model) / MAE(naive_seasonal)

    The denominator is the in-sample MAE of the seasonal naive forecast.
    If train_actual is provided, the naive error is computed from it.
    If naive_errors is provided directly, those are used.
    Returns NaN if denominator is zero (all-zero or constant training series).

    Seasonality=1 → random walk naive (non-seasonal).
    Seasonality=7 → seasonal naive with period 7 (weekly).
    """
    _validate_inputs(actual, predicted)

    if naive_errors is not None:
        naive_mae = float(np.mean(np.abs(naive_errors)))
    elif train_actual is not None:
        if len(train_actual) <= seasonality:
            logger.warning("MASE: training series too short for seasonality=%d — returning NaN.", seasonality)
            return float("nan")
        naive_mae = float(np.mean(np.abs(train_actual[seasonality:] - train_actual[:-seasonality])))
    else:
        raise ValueError("Either naive_errors or train_actual must be provided for MASE.")

    if naive_mae == 0:
        logger.warning("MASE: naive MAE is zero (constant training series) — returning NaN.")
        return float("nan")

    return mae(actual, predicted) / naive_mae


def mean_forecast_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Forecast Error (Bias).

    MFE = mean(predicted - actual)
    Positive → systematic over-forecasting.
    Negative → systematic under-forecasting.
    """
    _validate_inputs(actual, predicted)
    return float(np.mean(predicted - actual))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Empirical coverage of a prediction interval.

    Returns the fraction of actual observations that fall within [lower, upper].
    A nominal 80% interval should have coverage ≈ 0.80 across a large sample.
    """
    _validate_inputs(actual, lower)
    _validate_inputs(actual, upper)
    within = (actual >= lower) & (actual <= upper)
    return float(np.mean(within))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals (upper - lower)."""
    return float(np.mean(upper - lower))


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_80: np.ndarray | None = None,
    upper_80: np.ndarray | None = None,
    lower_95: np.ndarray | None = None,
    upper_95: np.ndarray | None = None,
    train_actual: np.ndarray | None = None,
    seasonality: int = 7,
) -> dict[str, float]:
    """
    Compute all evaluation metrics at once.

    Returns a dict with keys: MAE, RMSE, WAPE, MASE, MFE,
    and optionally Coverage_80, Coverage_95, Width_80, Width_95.
    """
    metrics: dict[str, float] = {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "WAPE": wape(actual, predicted),
        "MFE": mean_forecast_error(actual, predicted),
    }

    if train_actual is not None:
        metrics["MASE"] = mase(actual, predicted, train_actual=train_actual, seasonality=seasonality)

    if lower_80 is not None and upper_80 is not None:
        metrics["Coverage_80"] = interval_coverage(actual, lower_80, upper_80)
        metrics["Width_80"] = mean_interval_width(lower_80, upper_80)

    if lower_95 is not None and upper_95 is not None:
        metrics["Coverage_95"] = interval_coverage(actual, lower_95, upper_95)
        metrics["Width_95"] = mean_interval_width(lower_95, upper_95)

    return metrics
