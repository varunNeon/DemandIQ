from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandiq.inventory.policies import (
    InventoryPolicy,
    ReorderPointPolicy,
    lead_time_demand_stats,
    std_from_interval,
    stockout_probability,
)
from demandiq.services.inventory_service import (
    InventoryInputs,
    project_stock,
    recommend_from_forecast,
    simulate_policy,
)


def _future(point: float = 100.0, half_width: float = 25.6, days: int = 14) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2016-01-01", periods=days, freq="D"),
        "point_forecast": point,
        "lower_80": point - half_width,
        "upper_80": point + half_width,
    })


def test_std_from_interval_inverts_a_normal_80_percent_band() -> None:
    # A normal 80% band is mean ± 1.2816 sigma
    std = std_from_interval(np.array([100 - 12.816]), np.array([100 + 12.816]), 0.80)
    assert std[0] == pytest.approx(10.0, rel=1e-3)


def test_lead_time_variance_adds_across_days() -> None:
    mu, sigma = lead_time_demand_stats(np.full(14, 50.0), 4, forecast_std=np.full(14, 10.0))
    assert mu == 200.0
    assert sigma == pytest.approx(20.0)


def test_lead_time_longer_than_forecast_is_extended() -> None:
    mu, _ = lead_time_demand_stats(np.full(7, 10.0), 10, forecast_std=1.0)
    assert mu == pytest.approx(100.0)


def test_stockout_probability_is_half_at_the_mean() -> None:
    assert stockout_probability(100.0, 100.0, 15.0) == pytest.approx(0.5)
    assert stockout_probability(200.0, 100.0, 15.0) < 0.001


def test_eoq_uses_holding_cost_per_unit_day() -> None:
    policy = ReorderPointPolicy(InventoryPolicy())
    # D = 100/day * 365, K = 50, h = 0.05 * 365  ->  sqrt(2 * 36500 * 50 / 18.25) = 447.2
    assert policy.compute_eoq(100.0, 50.0, 0.05, 10.0) == pytest.approx(447.2, rel=1e-3)


def test_recommendation_orders_when_position_is_below_reorder_point() -> None:
    low = recommend_from_forecast(_future(), InventoryInputs(on_hand=50, lead_time_days=5))
    high = recommend_from_forecast(_future(), InventoryInputs(on_hand=5_000, lead_time_days=5))

    assert low["should_order"] and low["suggested_order_qty"] > 0
    assert not high["should_order"]
    assert low["uncertainty_source"] == "80% prediction interval"
    # 5 days x sigma 20 -> sigma_LT = 20 * sqrt(5)
    assert low["sigma_LT"] == pytest.approx(20 * np.sqrt(5), rel=1e-2)


def test_projected_stock_never_goes_negative_and_reorders() -> None:
    inputs = InventoryInputs(on_hand=600, lead_time_days=3)
    rec = recommend_from_forecast(_future(), inputs)
    plan = project_stock(_future(), inputs, rec)

    assert (plan["stock"] >= 0).all()
    assert (plan["stock_no_order"] >= 0).all()
    assert plan["order_placed"].sum() > 0
    assert plan["stock"].iloc[-1] > plan["stock_no_order"].iloc[-1]


def test_higher_service_target_holds_more_stock() -> None:
    rng = np.random.default_rng(3)
    demand = pd.Series(
        rng.poisson(60, 400).astype(float), index=pd.date_range("2015-01-01", periods=400)
    )
    low = simulate_policy(demand, InventoryInputs(service_level=0.80, lead_time_days=4), days=150)
    high = simulate_policy(demand, InventoryInputs(service_level=0.99, lead_time_days=4), days=150)

    assert high.avg_inventory > low.avg_inventory
    assert high.service_level_achieved >= low.service_level_achieved
