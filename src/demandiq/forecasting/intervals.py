"""
Split Conformal Prediction for DemandIQ.

Produces statistically defensible prediction intervals for any point forecaster.

Method: Split Conformal Prediction (Venn-Abers / residual quantile method)
- Split training data into fit set and calibration set (temporal split, not random)
- Fit model on fit set
- Compute residuals on calibration set
- Use quantile of absolute residuals as the margin

Why not random split? Because future observations cannot be in the calibration set.
The calibration split must always come AFTER the fit split in time.

Assumptions and limitations:
- Assumes exchangeability of residuals (approximately holds under mild distribution shift)
- Coverage guarantee is marginal (averaged across series/time), not conditional
- Does not account for heteroscedasticity (wider intervals for harder-to-predict periods)
- Intervals are clipped at zero for non-negative demand

References:
    Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
    Shafer & Vovk (2008) "A Tutorial on Conformal Prediction"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConformalIntervalWrapper:
    """
    Wraps any point forecaster with split conformal prediction intervals.

    Usage:
        wrapper = ConformalIntervalWrapper(my_forecaster, calibration_frac=0.2)
        wrapper.fit(train_series)
        result = wrapper.predict_with_intervals(horizon=14, levels=[0.80, 0.95])
    """

    def __init__(
        self,
        forecaster,
        calibration_frac: float = 0.2,
        min_calibration_size: int = 20,
    ) -> None:
        """
        Parameters
        ----------
        forecaster : Any forecaster with .fit(series) and .predict(horizon) methods.
        calibration_frac : Fraction of training data to hold out for calibration.
                           Must be at the END of the training period (temporal split).
        min_calibration_size : Minimum required calibration observations.
        """
        self.forecaster = forecaster
        self.calibration_frac = calibration_frac
        self.min_calibration_size = min_calibration_size
        self._calibration_residuals: np.ndarray | None = None
        self._fit_size: int = 0
        self._calibration_size: int = 0

    def fit(self, series: pd.Series) -> "ConformalIntervalWrapper":
        """
        Fit the underlying forecaster and compute calibration residuals.

        Split:
            [0 ... fit_end)     → fit set (model training)
            [fit_end ... n)     → calibration set (residual computation)

        CRITICAL: The calibration set is temporally AFTER the fit set.
        No calibration data is used during model fitting.

        Performance note: We fit the model twice total: once on the fit set
        and once on the full series. The first model produces one fixed-origin
        forecast over the complete calibration horizon. Refitting at every
        calibration point would be slower and would change the calibration
        protocol to rolling-origin residuals.
        """
        n = len(series)
        cal_size = max(
            self.min_calibration_size,
            int(n * self.calibration_frac),
        )

        if cal_size >= n - 10:
            # Not enough data to split — fit on full series, no conformal intervals
            logger.warning(
                "ConformalWrapper: insufficient data for calibration split "
                "(n=%d, cal_size=%d). Fitting on full series without conformal intervals.",
                n, cal_size,
            )
            self.forecaster.fit(series)
            self._calibration_residuals = None
            return self

        fit_end = n - cal_size
        fit_series = series.iloc[:fit_end]
        cal_series = series.iloc[fit_end:]

        self._fit_size = len(fit_series)
        self._calibration_size = len(cal_series)

        # Fit model ONCE on the training (fit) portion
        self.forecaster.fit(fit_series)

        # Generate predictions for each calibration horizon from the same
        # origin. Calling predict(1) repeatedly would reuse the same first-step
        # forecast and miscalibrate seasonal forecasters.
        try:
            calibration_predictions = np.asarray(
                self.forecaster.predict(len(cal_series)), dtype=float
            ).reshape(-1)
            n_predictions = min(len(cal_series), len(calibration_predictions))
            residuals = (
                cal_series.to_numpy(dtype=float)[:n_predictions]
                - calibration_predictions[:n_predictions]
            )
            residuals = residuals[np.isfinite(residuals)]
        except Exception as exc:
            logger.warning("Conformal calibration forecast failed: %s", exc)
            residuals = np.array([], dtype=float)

        if len(residuals) < self.min_calibration_size // 2:
            logger.warning(
                "ConformalWrapper: only %d calibration residuals computed. "
                "Intervals may be unreliable.",
                len(residuals),
            )

        self._calibration_residuals = np.asarray(residuals, dtype=float)

        # Refit ONCE on full training data for the final forecast
        self.forecaster.fit(series)

        logger.debug(
            "ConformalWrapper fitted: fit_size=%d, cal_size=%d, "
            "residual_std=%.3f",
            self._fit_size, self._calibration_size,
            np.std(self._calibration_residuals) if len(self._calibration_residuals) > 0 else 0,
        )
        return self


    def predict_with_intervals(
        self,
        horizon: int,
        levels: list[float] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Produce point forecasts and conformal prediction intervals.

        The conformal margin q_level is the (level * (1 + 1/n))-th quantile
        of absolute calibration residuals, where n = calibration set size.
        This provides finite-sample coverage guarantee (under exchangeability).

        Intervals do NOT widen with horizon in this implementation.
        For longer horizons, interval width is a conservative estimate.
        """
        levels = levels or [0.80, 0.95]
        point = self.forecaster.predict(horizon)
        result: dict[str, np.ndarray] = {"point": np.maximum(point, 0.0)}

        if self._calibration_residuals is None or len(self._calibration_residuals) == 0:
            logger.warning("No calibration residuals available — returning point forecasts only.")
            return result

        abs_residuals = np.abs(self._calibration_residuals)
        n_cal = len(abs_residuals)

        for level in levels:
            # Finite-sample conformal quantile (slightly inflated to guarantee coverage)
            adjusted_level = min(level * (1 + 1 / n_cal), 1.0)
            q = float(np.quantile(abs_residuals, adjusted_level))

            label = f"{int(level * 100)}"
            result[f"lower_{label}"] = np.maximum(point - q, 0.0)
            result[f"upper_{label}"] = point + q

        return result

    def coverage_report(self, actual: np.ndarray, levels: list[float] | None = None) -> dict[str, float]:
        """
        Evaluate empirical coverage on held-out data.

        Returns empirical coverage for each level.
        A well-calibrated 80% interval should have coverage ≈ 0.80.
        """
        from demandiq.forecasting.metrics import interval_coverage

        levels = levels or [0.80, 0.95]
        report = {}

        if self._calibration_residuals is None:
            return report

        abs_residuals = np.abs(self._calibration_residuals)
        n_cal = len(abs_residuals)
        point = self.forecaster.predict(len(actual))

        for level in levels:
            adjusted_level = min(level * (1 + 1 / n_cal), 1.0)
            q = float(np.quantile(abs_residuals, adjusted_level))
            lower = np.maximum(point - q, 0.0)
            upper = point + q
            cov = interval_coverage(actual, lower, upper)
            label = f"{int(level * 100)}"
            report[f"coverage_{label}"] = cov
            report[f"nominal_{label}"] = level

        return report
