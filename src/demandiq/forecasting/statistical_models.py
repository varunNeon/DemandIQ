"""
ETS (Exponential Smoothing) statistical forecaster for DemandIQ.

Uses statsmodels ExponentialSmoothing (Holt-Winters method).
Automatically selects between additive/multiplicative trend and seasonality
based on the data characteristics and AIC.

Minimum history requirement: 2 × seasonality_period observations.
Falls back to NaiveForecaster if insufficient history.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from demandiq.forecasting.baselines import NaiveForecaster

logger = logging.getLogger(__name__)

MIN_HISTORY_MULTIPLIER = 2  # require at least 2 × seasonal_period observations


class ETSForecaster:
    """
    ETS (Error, Trend, Seasonality) Forecaster.

    Wraps statsmodels Holt-Winters ExponentialSmoothing.
    Attempts to fit additive seasonality first; falls back to no-seasonality
    if the series is too short or has non-positive values that prevent
    multiplicative decomposition.

    Model selection: compares AIC between trend variants.
    """

    name: str = "ets"

    def __init__(self, seasonal_period: int = 7) -> None:
        self.seasonal_period = seasonal_period
        self._model = None
        self._fitted = None
        self._fallback: NaiveForecaster | None = None
        self._used_fallback: bool = False
        self._aic: float | None = None

    def fit(self, series: pd.Series) -> "ETSForecaster":
        """
        Fit the ETS model to the provided time series.

        Tries multiple configurations and picks the one with lowest AIC.
        Falls back to NaiveForecaster if fitting fails entirely.
        """
        self._used_fallback = False
        series = series.dropna()

        min_obs = MIN_HISTORY_MULTIPLIER * self.seasonal_period
        if len(series) < min_obs:
            logger.warning(
                "ETS: insufficient history (%d obs, need %d). Using NaiveForecaster fallback.",
                len(series), min_obs,
            )
            self._fallback = NaiveForecaster()
            self._fallback.fit(series)
            self._used_fallback = True
            return self

        # Replace zeros with a small positive value for multiplicative models
        series_positive = series.clip(lower=0.1)
        can_use_multiplicative = (series > 0).all()

        # Candidate configurations: (trend, seasonal, damped)
        configs = [
            ("add", "add", False),
            ("add", "add", True),
            ("add", None, False),
            (None, "add", False),
            (None, None, False),
        ]
        if can_use_multiplicative:
            configs = [
                ("add", "mul", False),
                ("add", "add", False),
                ("add", "mul", True),
                ("add", None, False),
                (None, "mul", False),
                (None, None, False),
            ] + configs

        best_aic = np.inf
        best_fitted = None

        for trend, seasonal, damped in configs:
            use_seasonal = seasonal is not None and len(series) >= 2 * self.seasonal_period
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        series_positive if seasonal == "mul" else series.clip(lower=0),
                        trend=trend,
                        seasonal=seasonal if use_seasonal else None,
                        seasonal_periods=self.seasonal_period if use_seasonal else None,
                        damped_trend=damped if trend is not None else False,
                        initialization_method="estimated",
                    )
                    fitted = model.fit(optimized=True, remove_bias=False)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_fitted = fitted
            except Exception as exc:
                logger.debug("ETS config (%s, %s, damped=%s) failed: %s", trend, seasonal, damped, exc)
                continue

        if best_fitted is None:
            logger.warning("ETS: all configurations failed. Using NaiveForecaster fallback.")
            self._fallback = NaiveForecaster()
            self._fallback.fit(series)
            self._used_fallback = True
            return self

        self._fitted = best_fitted
        self._aic = best_aic
        logger.debug("ETS fitted: AIC=%.2f, used_seasonal=%s", best_aic, best_fitted.model.seasonal)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast `horizon` steps ahead."""
        if self._used_fallback:
            return self._fallback.predict(horizon)
        if self._fitted is None:
            raise RuntimeError("ETSForecaster must be fitted before predict().")
        forecast = self._fitted.forecast(horizon)
        return np.maximum(np.array(forecast), 0.0)

    def predict_with_intervals(
        self,
        horizon: int,
        levels: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Produce point forecast with model-native prediction intervals.

        statsmodels simulates from the fitted model to produce intervals.
        These are model-based intervals, not conformal-calibrated.
        """
        levels = levels or [0.80, 0.95]
        point = self.predict(horizon)
        result: dict[str, np.ndarray] = {"point": point}

        if self._used_fallback or self._fitted is None:
            return result

        try:
            for level in levels:
                alpha = 1.0 - level
                sim = self._fitted.simulate(
                    nsimulations=horizon,
                    repetitions=200,
                    error="add",
                    random_errors="bootstrap",
                )
                lower = np.maximum(np.quantile(sim, alpha / 2, axis=1), 0.0)
                upper = np.quantile(sim, 1 - alpha / 2, axis=1)
                label = f"{int(level * 100)}"
                result[f"lower_{label}"] = lower
                result[f"upper_{label}"] = upper
        except Exception as exc:
            logger.warning("ETS interval simulation failed: %s", exc)

        return result

    @property
    def used_fallback(self) -> bool:
        return self._used_fallback

    @property
    def aic(self) -> float | None:
        return self._aic
