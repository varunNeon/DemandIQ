from __future__ import annotations

import pandas as pd

from demandiq.data.features import build_time_series_features


def test_grouped_rolling_features_do_not_cross_series_boundaries() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D").tolist() * 2,
            "store_id": ["S1"] * 8 + ["S2"] * 8,
            "product_id": ["P1"] * 16,
            "units_sold": [100.0] * 8 + [1.0] * 8,
        }
    )

    features = build_time_series_features(
        df,
        group_cols=["store_id", "product_id"],
    )

    first_s2 = features[features["store_id"] == "S2"].iloc[0]
    assert pd.isna(first_s2["lag_1d"])
    assert pd.isna(first_s2["roll_mean_7d"])


def test_rolling_features_use_only_prior_observations() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "units_sold": [1.0, 2.0, 3.0, 4.0, 1000.0, 6.0, 7.0, 8.0],
        }
    )

    features = build_time_series_features(df)

    assert features.loc[4, "lag_1d"] == 4.0
    assert features.loc[4, "roll_mean_7d"] == 2.5
