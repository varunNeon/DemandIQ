from __future__ import annotations

import numpy as np

from demandiq.forecasting.backtesting import BacktestResult, BacktestSummary
from demandiq.services.forecast_service import _prediction_intervals_from_backtest


def test_prediction_intervals_from_backtest_are_ordered_and_non_negative() -> None:
    backtest = BacktestSummary(model_name="lgbm", n_folds=1, horizon=3)
    backtest.folds.append(
        BacktestResult(
            fold=0,
            train_end_date=0,
            val_start_date=1,
            val_end_date=3,
            n_train=10,
            n_val=3,
            model_name="lgbm",
            actual=np.array([10.0, 20.0, 30.0]),
            predicted=np.array([12.0, 18.0, 33.0]),
        )
    )

    result = _prediction_intervals_from_backtest(
        point=np.array([1.0, 5.0, 10.0]),
        backtest=backtest,
        levels=[0.80, 0.95],
    )

    assert set(result) == {"point", "lower_80", "upper_80", "lower_95", "upper_95"}
    assert (result["lower_80"] >= 0).all()
    assert (result["lower_80"] <= result["point"]).all()
    assert (result["point"] <= result["upper_80"]).all()
    assert (result["lower_95"] <= result["lower_80"]).all()
    assert (result["upper_80"] <= result["upper_95"]).all()
