from __future__ import annotations

import numpy as np
import pandas as pd

from demandiq.forecasting.baselines import SeasonalNaiveForecaster
from demandiq.forecasting.intervals import ConformalIntervalWrapper, horizon_conformal_bands


def test_conformal_calibration_uses_each_seasonal_horizon_prediction() -> None:
    series = pd.Series(np.arange(1.0, 16.0))
    wrapper = ConformalIntervalWrapper(
        SeasonalNaiveForecaster(period=2),
        calibration_frac=4 / 15,
        min_calibration_size=2,
    )

    wrapper.fit(series)

    # The fit split ends in [10, 11], so its four-step seasonal forecast is
    # [10, 11, 10, 11], compared with calibration actuals [12, 13, 14, 15].
    np.testing.assert_array_equal(wrapper._calibration_residuals, [2.0, 2.0, 4.0, 4.0])


def test_horizon_bands_are_nested_ordered_and_non_negative() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 10, size=(8, 14))
    point = np.linspace(5, 40, 14)

    bands = horizon_conformal_bands(point, residuals, [0.80, 0.95])

    assert (bands["lower_95"] >= 0).all()
    assert (bands["lower_95"] <= bands["lower_80"]).all()
    assert (bands["lower_80"] <= bands["point"]).all()
    assert (bands["point"] <= bands["upper_80"]).all()
    assert (bands["upper_80"] <= bands["upper_95"]).all()


def test_horizon_bands_widen_when_later_steps_miss_by_more() -> None:
    steps = np.arange(1, 15)
    rng = np.random.default_rng(1)
    residuals = rng.normal(0, 1, size=(30, 14)) * steps  # error grows with horizon
    bands = horizon_conformal_bands(np.full(14, 100.0), residuals, [0.80], window=1)

    width = bands["upper_80"] - bands["lower_80"]
    assert width[-1] > 3 * width[0]


def test_horizon_bands_follow_skewed_residuals() -> None:
    # Under-forecasts (positive residuals) are much larger than over-forecasts
    values = np.array([-2.0, -1.0, 0.0, 5.0, 10.0, 20.0])
    residuals = np.tile(values[:, None], (1, 6))  # every step sees the same skewed misses
    bands = horizon_conformal_bands(np.full(6, 50.0), residuals, [0.80], window=0)

    above = bands["upper_80"] - bands["point"]
    below = bands["point"] - bands["lower_80"]
    assert (above > below).all()


def test_level_scaling_widens_bands_for_busier_days() -> None:
    residuals = np.random.default_rng(2).normal(0, 1, size=(20, 2)) * np.sqrt(101)
    cal_preds = np.full((20, 2), 100.0)
    bands = horizon_conformal_bands(
        np.array([25.0, 400.0]), residuals, [0.80], window=0, calibration_predictions=cal_preds
    )
    width = bands["upper_80"] - bands["lower_80"]
    assert width[1] > 2 * width[0]
