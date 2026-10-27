from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.generate_demo_data import generate_sales


def test_generate_sales_returns_mutable_non_negative_series() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")

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
