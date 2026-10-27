"""
Baseline forecasters for DemandIQ.

Naive and Seasonal-Naive forecasters serve as important benchmarks.
Every complex model MUST beat these baselines to justify its complexity.

Both forecasters follow the same interface:
    fit(series: pd.Series) -> self
    predict(horizon: int) -> np.ndarray
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NaiveForecaster:
    """
    Naive (Random Walk) Forecaster.

    Predicts the last observed value for all future horizons.
    Formally: y_hat(t+h) = y(t) for all h >= 1.

    This is a strong baseline for non-stationary series.
    """

    name: str = "naive"

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, series: pd.Series) -> "NaiveForecaster":
        """Fit by recording the last observed value."""
        if series.empty:
            raise ValueError("Cannot fit NaiveForecaster on empty series.")
        valid = series.dropna()
        if valid.empty:
            raise ValueError("All values in series are NaN.")
        self._last_value = float(valid.iloc[-1])
        logger.debug("NaiveForecaster fitted: last_value=%.2f", self._last_value)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Return the last observed value repeated for `horizon` steps."""
        if self._last_value is None:
            raise RuntimeError("NaiveForecaster must be fitted before calling predict().")
        return np.full(horizon, self._last_value)

    def predict_with_intervals(
        self,
        horizon: int,
        residuals: np.ndarray | None = None,
        levels: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Produce point forecast plus empirical prediction intervals.

        Intervals are constructed from historical one-step-ahead residuals
        (actual - naive_prediction) if provided.
        Intervals widen with horizon (scaled by sqrt(h)) per random-walk theory.
        """
        levels = levels or [0.80, 0.95]
        point = self.predict(horizon)
        result: dict[str, np.ndarray] = {"point": point}

        if residuals is not None and len(residuals) > 0:
            std_residual = float(np.std(residuals, ddof=1))
            for level in levels:
                alpha = 1.0 - level
                from scipy import stats  # optional dependency
                z = stats.norm.ppf(1 - alpha / 2)
                # Scale by sqrt(h) for random-walk interval expansion
                widths = z * std_residual * np.sqrt(np.arange(1, horizon + 1))
                label = f"{int(level * 100)}"
                result[f"lower_{label}"] = np.maximum(point - widths, 0.0)
                result[f"upper_{label}"] = point + widths

        return result


class SeasonalNaiveForecaster:
    """
    Seasonal Naive Forecaster.

    Predicts the value from the same point in the previous season.
    Formally: y_hat(t+h) = y(t + h - m) where m is the seasonality period.

    With period=7 (weekly), this uses demand from the same weekday last week.
    Requires at least `period` historical observations.
    """

    name: str = "seasonal_naive"

    def __init__(self, period: int = 7) -> None:
        if period < 1:
            raise ValueError(f"Seasonality period must be >= 1, got {period}.")
        self.period = period
        self._history: np.ndarray | None = None

    def fit(self, series: pd.Series) -> "SeasonalNaiveForecaster":
        """Store the last `period` observations for prediction."""
        if len(series) < self.period:
            raise ValueError(
                f"SeasonalNaiveForecaster requires >= {self.period} observations, "
                f"got {len(series)}."
            )
        valid = series.dropna()
        self._history = valid.values.copy()
        logger.debug(
            "SeasonalNaiveForecaster fitted: period=%d, history_len=%d",
            self.period, len(self._history),
        )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict using the corresponding seasonal lag."""
        if self._history is None:
            raise RuntimeError("SeasonalNaiveForecaster must be fitted before predict().")
        seasonal_window = self._history[-self.period:]
        preds = np.array([seasonal_window[(h - 1) % self.period] for h in range(1, horizon + 1)])
        return np.maximum(preds, 0.0)

    def predict_with_intervals(
        self,
        horizon: int,
        residuals: np.ndarray | None = None,
        levels: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Produce point forecast with empirical prediction intervals."""
        levels = levels or [0.80, 0.95]
        point = self.predict(horizon)
        result: dict[str, np.ndarray] = {"point": point}

        if residuals is not None and len(residuals) > 0:
            std_residual = float(np.std(residuals, ddof=1))
            for level in levels:
                alpha = 1.0 - level
                z = 1.96 if level == 0.95 else 1.28  # approximations
                label = f"{int(level * 100)}"
                result[f"lower_{label}"] = np.maximum(point - z * std_residual, 0.0)
                result[f"upper_{label}"] = point + z * std_residual

        return result


def get_baseline_forecasters() -> list:
    """Return instantiated baseline forecasters."""
    return [
        NaiveForecaster(),
        SeasonalNaiveForecaster(period=7),
    ]
