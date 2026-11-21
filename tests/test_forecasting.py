from __future__ import annotations

import numpy as np
import pandas as pd

from demandiq.forecasting.backtesting import rolling_origin_backtest
from demandiq.forecasting.baselines import NaiveForecaster
from demandiq.forecasting.ml_models import LGBMForecaster
from demandiq.services.forecast_service import best_model, forecast_series


def test_backtest_folds_are_chronological_and_never_train_on_validation(weekly_series_df) -> None:
    series = weekly_series_df.set_index("date")["units_sold"]
    summary = rolling_origin_backtest(series, NaiveForecaster, horizon=14, n_splits=6, step_size=7)

    assert summary.n_folds == 6
    origins = [f.train_end_index for f in summary.folds]
    assert origins == sorted(origins)
    for fold in summary.folds:
        assert fold.n_train == fold.train_end_index
        assert fold.val_dates[0] == series.index[fold.train_end_index]


def test_backtest_intervals_only_calibrate_on_fully_observed_earlier_folds(weekly_series_df) -> None:
    series = weekly_series_df.set_index("date")["units_sold"]
    summary = rolling_origin_backtest(
        series, NaiveForecaster, horizon=14, n_splits=8, step_size=7, min_calibration_folds=3
    )
    # With 7-day steps and 14-day windows, fold k can use folds 0..k-2.
    # It needs three of them, so the first fold with intervals is fold 4.
    with_bands = [i for i, f in enumerate(summary.folds) if f.lower_80 is not None]
    assert with_bands[0] == 4
    assert "Coverage_80" in summary.mean_metrics


def test_lgbm_forecast_ignores_demand_after_the_origin(weekly_series_df) -> None:
    train = weekly_series_df.iloc[:300]
    altered = weekly_series_df.copy()
    altered.loc[300:, "units_sold"] = 10_000.0  # future values that must not leak

    a = LGBMForecaster(horizon=7).fit(train).predict(7)
    b = LGBMForecaster(horizon=7).fit(altered.iloc[:300]).predict(7)
    np.testing.assert_allclose(a, b)


def test_lgbm_responds_to_a_planned_promotion(weekly_series_df) -> None:
    model = LGBMForecaster(horizon=7).fit(weekly_series_df)
    base = model.predict(7)
    plan = pd.DataFrame({"onpromotion": [0, 0, 1, 0, 0, 0, 0], "is_holiday": 0})
    promo = model.predict(7, future_exog=plan)

    assert promo[2] > base[2] * 1.15
    np.testing.assert_allclose(np.delete(promo, 2), np.delete(base, 2))


def test_lgbm_beats_naive_on_a_seasonal_series(weekly_series_df) -> None:
    results = forecast_series(weekly_series_df, ["naive", "lgbm"], horizon=14, n_splits=6)

    assert best_model(results) == "lgbm"
    assert results["lgbm"].metrics["WAPE"] < 0.8 * results["naive"].metrics["WAPE"]
    future = results["lgbm"].future
    assert len(future) == 14
    assert (future["lower_80"] <= future["point_forecast"]).all()
    assert (future["point_forecast"] <= future["upper_80"]).all()
    assert results["lgbm"].feature_importance["importance"].sum() > 0.99
