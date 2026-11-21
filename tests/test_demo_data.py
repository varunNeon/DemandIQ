from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.generate_demo_data import generate_sales


def test_generate_sales_returns_whole_non_negative_units() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    sales = generate_sales(
        rng=rng,
        dates=dates,
        base_demand=20.0,
        store_factor=1.0,
        product_factor=1.0,
        promotion_dates=set(),
        holiday_dates=set(),
    )

    assert len(sales) == len(dates)
    assert sales.index.equals(dates)
    assert (sales >= 0).all()
    assert pd.api.types.is_integer_dtype(sales)


def test_promotions_lift_generated_demand() -> None:
    dates = pd.date_range("2024-01-01", periods=400, freq="D")
    promo = set(dates[::5].strftime("%Y-%m-%d"))
    sales = generate_sales(np.random.default_rng(1), dates, 50.0, 1.0, 1.0, promo, set())
    is_promo = np.array([d.strftime("%Y-%m-%d") in promo for d in dates])

    assert sales[is_promo].mean() > 1.15 * sales[~is_promo].mean()
