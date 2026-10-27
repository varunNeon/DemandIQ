from __future__ import annotations

import numpy as np
import pandas as pd

from demandiq.forecasting.baselines import SeasonalNaiveForecaster
from demandiq.forecasting.intervals import ConformalIntervalWrapper


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


def test_conformal_intervals_are_ordered_and_non_negative() -> None:
    series = pd.Series(np.arange(1.0, 101.0))
    wrapper = ConformalIntervalWrapper(SeasonalNaiveForecaster(period=7)).fit(series)

    result = wrapper.predict_with_intervals(horizon=14)

    assert (result["lower_80"] >= 0).all()
    assert (result["lower_80"] <= result["point"]).all()
    assert (result["point"] <= result["upper_80"]).all()
    assert (result["lower_95"] <= result["lower_80"]).all()
    assert (result["upper_80"] <= result["upper_95"]).all()
