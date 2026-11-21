"""
Feature engineering for DemandIQ.

Constructs lag features, rolling statistics, and calendar features
for use in machine learning forecasting models.

CRITICAL: All features are constructed in a leakage-free manner.
- Lag features use pd.shift() so that at prediction time t, the model
  only sees values from t-1 and earlier.
- Rolling statistics are computed on SHIFTED series so the window
  never includes the current observation.
- Features are fitted on training data only (no look-ahead from test set).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Lag periods to include (in days)
LAG_PERIODS = [1, 7, 14, 21, 28]

# Rolling window sizes (in days) for mean and std
ROLLING_WINDOWS = [7, 14, 28]


def build_time_series_features(
    df: pd.DataFrame,
    target_col: str = "units_sold",
    date_col: str = "date",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix from a time-series DataFrame.

    Parameters
    ----------
    df : DataFrame with at minimum [date, target_col] columns.
         May also contain [store_id, product_id, onpromotion, is_holiday].
    target_col : Name of the sales/demand column.
    date_col : Name of the date column.
    group_cols : Columns defining series identity (e.g. ['store_id', 'product_id']).
                 Lags and rolling stats are computed per group.

    Returns
    -------
    DataFrame with original columns plus feature columns.
    NaN rows from lag construction are retained (caller should dropna).

    Feature naming convention:
        lag_{n}d  : sales at time t - n days
        roll_mean_{n}d : rolling mean over {n} days ending at t-1 (shifted)
        roll_std_{n}d  : rolling std over {n} days ending at t-1 (shifted)
        dow_* : one-hot day of week (Mon=0 ... Sun=6)
        month : month of year (1-12)
        day_of_year : day of year (1-366)
        week_of_year : ISO week number
        is_weekend : 1 if Sat or Sun
        quarter : quarter of year
        onpromotion : 0/1 if available
        is_holiday : 0/1 if available
    """
    if group_cols is None:
        group_cols = []

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([*group_cols, date_col]).reset_index(drop=True)

    # --- Calendar features (no data leakage) ---
    df["dow"] = df[date_col].dt.dayofweek  # Monday=0
    df["month"] = df[date_col].dt.month
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["quarter"] = df[date_col].dt.quarter

    # Day-of-week one-hots
    for d in range(7):
        df[f"dow_{d}"] = (df["dow"] == d).astype(int)

    # --- Lag features ---
    # Computed per group so lags don't bleed across different store-product series
    if group_cols:
        grouped = df.groupby(group_cols)[target_col]
        for lag in LAG_PERIODS:
            df[f"lag_{lag}d"] = grouped.shift(lag)
    else:
        for lag in LAG_PERIODS:
            df[f"lag_{lag}d"] = df[target_col].shift(lag)

    # --- Rolling statistics ---
    # CRITICAL: shift(1) before rolling so the window is [t-window, t-1]
    # This prevents the rolling mean from including the current observation.
    if group_cols:
        df["_shifted_target"] = df.groupby(group_cols)[target_col].shift(1)
        for window in ROLLING_WINDOWS:
            grouped_roll = df.groupby(group_cols)["_shifted_target"].rolling(
                window=window,
                min_periods=max(1, window // 2),
            )
            df[f"roll_mean_{window}d"] = grouped_roll.mean().reset_index(level=group_cols, drop=True)
            df[f"roll_std_{window}d"] = (
                grouped_roll.std().reset_index(level=group_cols, drop=True).fillna(0)
            )
        df = df.drop(columns=["_shifted_target"])
    else:
        shifted = df[target_col].shift(1)
        for window in ROLLING_WINDOWS:
            roll = shifted.rolling(window=window, min_periods=max(1, window // 2))
            df[f"roll_mean_{window}d"] = roll.mean()
            df[f"roll_std_{window}d"] = roll.std().fillna(0)

    # --- Promotion / holiday flags ---
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].astype(int)
    else:
        df["onpromotion"] = 0

    if "is_holiday" in df.columns:
        df["is_holiday"] = df["is_holiday"].astype(int)
    else:
        df["is_holiday"] = 0

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names produced by build_time_series_features."""
    cols = []
    # Lag features
    for lag in LAG_PERIODS:
        cols.append(f"lag_{lag}d")
    # Rolling stats
    for window in ROLLING_WINDOWS:
        cols.append(f"roll_mean_{window}d")
        cols.append(f"roll_std_{window}d")
    # Calendar
    cols += [f"dow_{d}" for d in range(7)]
    cols += ["month", "day_of_year", "week_of_year", "is_weekend", "quarter"]
    # Context
    cols += ["onpromotion", "is_holiday"]
    return cols


def prepare_ml_dataset(
    df: pd.DataFrame,
    target_col: str = "units_sold",
    date_col: str = "date",
    group_cols: list[str] | None = None,
    min_lag: int = max(LAG_PERIODS),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y for ML training.

    Drops rows where lag features are NaN (first `min_lag` rows per series).
    Returns (X, y) ready for model fitting.
    """
    features_df = build_time_series_features(df, target_col, date_col, group_cols)
    feature_cols = get_feature_columns()

    # Only include feature columns that actually exist in the dataframe
    available_feature_cols = [c for c in feature_cols if c in features_df.columns]

    # Drop rows with any NaN lag features (beginning of each series)
    mask = features_df[available_feature_cols].notna().all(axis=1)
    features_df = features_df[mask].copy()

    X = features_df[available_feature_cols]
    y = features_df[target_col]

    logger.debug(
        "ML dataset prepared: %d rows, %d features (dropped %d NaN-lag rows)",
        len(X), len(available_feature_cols), (~mask).sum(),
    )
    return X, y
