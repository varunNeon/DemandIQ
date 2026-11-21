"""
LightGBM forecaster for DemandIQ.

Strategy: direct multi-horizon forecasting with a single model.

Every training row is a pair (origin day t, horizon h) whose target is the
demand on day t + h. Features describe two things:

1. What is known at the end of day t: recent demand, rolling levels and
   volatility, the most recent same-weekday values, and demand around the
   same date last year.
2. What is known in advance about the target day: its weekday, season,
   whether it is a holiday, and whether a promotion is planned.

Demand features and the target are divided by the recent 28-day level, so the
trees learn shapes (weekday ratios, promotion lift) rather than absolute
volumes. This lets the model follow trends and level shifts, which tree
models cannot extrapolate on raw values.

Leakage rules:
- Demand features only use observations at or before the origin day t.
- Target-day inputs are limited to calendar facts and planned promotions.
  For the real future, holidays come from the known calendar and promotions
  are assumed absent unless the caller supplies a promotion plan.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from demandiq.config import settings
from demandiq.data.calendar import future_calendar
from demandiq.forecasting.baselines import SeasonalNaiveForecaster

logger = logging.getLogger(__name__)

LEVEL_WINDOW = 28
RECENT_LAGS = 7
MIN_TRAIN_ROWS = LEVEL_WINDOW + 60

FEATURE_NAMES: list[str] = [
    *[f"lag_{k}" for k in range(RECENT_LAGS)],
    "mean_7", "mean_14", "std_7", "std_28", "ewm_7",
    "same_weekday_last", "same_weekday_mean_4w", "last_year_window",
    "level_raw",
    "horizon", "target_dow", "target_is_weekend", "target_doy_sin", "target_doy_cos",
    "target_month", "target_promo", "target_holiday", "promo_share_28",
]

FEATURE_LABELS: dict[str, str] = {
    "lag_0": "Demand on the origin day",
    "mean_7": "7-day average demand",
    "mean_14": "14-day average demand",
    "std_7": "7-day volatility",
    "std_28": "28-day volatility",
    "ewm_7": "Exponentially weighted demand",
    "same_weekday_last": "Same weekday, last known week",
    "same_weekday_mean_4w": "Same weekday, 4-week average",
    "last_year_window": "Same week last year",
    "level_raw": "Recent volume level",
    "horizon": "Days ahead",
    "target_dow": "Day of week",
    "target_is_weekend": "Weekend",
    "target_doy_sin": "Time of year (sin)",
    "target_doy_cos": "Time of year (cos)",
    "target_month": "Month",
    "target_promo": "Planned promotion",
    "target_holiday": "Holiday",
    "promo_share_28": "Recent promotion share",
}


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean including the current value; NaN until the window is full."""
    s = pd.Series(y)
    return s.rolling(window, min_periods=window).mean().to_numpy()


def _rolling_std(y: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(y)
    return s.rolling(window, min_periods=window).std().to_numpy()


class LGBMForecaster:
    """Direct multi-horizon LightGBM forecaster (one model, horizon as a feature)."""

    name: str = "lgbm"
    requires_frame: bool = True

    def __init__(
        self,
        horizon: int = 14,
        n_estimators: int = 350,
        learning_rate: float = 0.04,
        num_leaves: int = 15,
    ) -> None:
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self._model = None
        self._fallback: SeasonalNaiveForecaster | None = None
        self._used_fallback = False
        self._y: np.ndarray | None = None
        self._promo: np.ndarray | None = None
        self._last_date: pd.Timestamp | None = None
        self._origin: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Feature construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _origin_arrays(y: np.ndarray, promo: np.ndarray) -> dict[str, np.ndarray]:
        """Per-day statistics known at the end of each day."""
        level = _rolling_mean(y, LEVEL_WINDOW)
        return {
            "level": level,
            "mean_7": _rolling_mean(y, 7),
            "mean_14": _rolling_mean(y, 14),
            "std_7": _rolling_std(y, 7),
            "std_28": _rolling_std(y, LEVEL_WINDOW),
            "ewm_7": pd.Series(y).ewm(span=7, adjust=False).mean().to_numpy(),
            "promo_share_28": _rolling_mean(promo.astype(float), LEVEL_WINDOW),
        }

    def _features(
        self,
        y: np.ndarray,
        origin: dict[str, np.ndarray],
        t: np.ndarray,
        h: np.ndarray,
        target_dates: pd.DatetimeIndex,
        target_promo: np.ndarray,
        target_holiday: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, scale) for origin indices `t` and horizons `h`."""
        n = len(y)
        scale = origin["level"][t] + 1.0
        cols: list[np.ndarray] = []

        for k in range(RECENT_LAGS):
            idx = t - k
            cols.append(np.where(idx >= 0, y[np.clip(idx, 0, n - 1)], np.nan) / scale)

        for key in ("mean_7", "mean_14", "std_7", "std_28", "ewm_7"):
            cols.append(origin[key][t] / scale)

        # Most recent same-weekday observation that is known at the origin
        weeks_back = np.ceil(h / 7).astype(int)
        s = t + h - 7 * weeks_back
        cols.append(y[s] / scale)
        stack = []
        for w in range(4):
            idx = s - 7 * w
            stack.append(np.where(idx >= 0, y[np.clip(idx, 0, n - 1)], np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            same_wd = np.nanmean(np.vstack(stack), axis=0)
        cols.append(same_wd / scale)

        # Demand around the same date last year (7-day window, all before origin)
        centre = t + h - 364
        window_vals = []
        for d in range(-3, 4):
            idx = centre + d
            valid = (idx >= 0) & (idx <= t)
            window_vals.append(np.where(valid, y[np.clip(idx, 0, n - 1)], np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            last_year = np.nanmean(np.vstack(window_vals), axis=0)
        cols.append(last_year / scale)

        cols.append(origin["level"][t])

        doy = np.asarray(target_dates.dayofyear, dtype=float)
        dow = np.asarray(target_dates.dayofweek, dtype=float)
        cols += [
            h.astype(float),
            dow,
            (dow >= 5).astype(float),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.asarray(target_dates.month, dtype=float),
            target_promo.astype(float),
            target_holiday.astype(float),
            origin["promo_share_28"][t],
        ]
        return np.column_stack(cols), scale

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, series_df: pd.DataFrame, target_col: str = "units_sold") -> "LGBMForecaster":
        """
        Fit on a frame with columns [date, units_sold] and optionally
        [onpromotion, is_holiday], sorted or unsorted.
        """
        import lightgbm as lgb

        df = series_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        y = df[target_col].to_numpy(dtype=float)
        promo = df.get("onpromotion", pd.Series(0, index=df.index)).astype(int).to_numpy()
        holiday = df.get("is_holiday", pd.Series(0, index=df.index)).astype(int).to_numpy()
        dates = pd.DatetimeIndex(df["date"])

        self._used_fallback = False
        self._model = None
        self._y = y
        self._promo = promo
        self._last_date = dates[-1]

        if len(y) < MIN_TRAIN_ROWS:
            logger.warning(
                "LGBM: %d rows is below the %d-row minimum. Using seasonal naive.",
                len(y), MIN_TRAIN_ROWS,
            )
            self._fallback = SeasonalNaiveForecaster(period=7).fit(pd.Series(y))
            self._used_fallback = True
            return self

        origin = self._origin_arrays(y, promo)
        self._origin = origin

        n = len(y)
        t_list, h_list = [], []
        for h in range(1, self.horizon + 1):
            t = np.arange(LEVEL_WINDOW - 1, n - h)
            t_list.append(t)
            h_list.append(np.full(len(t), h))
        t_all = np.concatenate(t_list)
        h_all = np.concatenate(h_list)
        target_idx = t_all + h_all

        X, scale = self._features(
            y, origin, t_all, h_all,
            dates[target_idx], promo[target_idx], holiday[target_idx],
        )
        target = y[target_idx] / scale

        self._model = lgb.LGBMRegressor(
            objective="l1",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=settings.RANDOM_SEED,
            n_jobs=1,
            verbose=-1,
        )
        self._model.fit(X, target)
        return self

    def predict(self, horizon: int | None = None, future_exog: pd.DataFrame | None = None) -> np.ndarray:
        """
        Forecast `horizon` days after the last training day.

        `future_exog` may supply [onpromotion, is_holiday] for those days (for
        example a promotion plan, or the realised calendar during backtests).
        Without it, holidays come from the known calendar and promotions are
        assumed absent.
        """
        h_max = horizon or self.horizon
        if self._used_fallback:
            return self._fallback.predict(h_max)
        if self._model is None:
            raise RuntimeError("LGBMForecaster must be fitted before predict().")
        if h_max > self.horizon:
            raise ValueError(f"Model was trained for up to {self.horizon} days ahead, not {h_max}.")

        exog = future_calendar(self._last_date, h_max)
        if future_exog is not None:
            supplied = future_exog.reset_index(drop=True).iloc[:h_max]
            for col in ("onpromotion", "is_holiday"):
                if col in supplied.columns:
                    exog.loc[: len(supplied) - 1, col] = supplied[col].astype(int).to_numpy()

        n = len(self._y)
        h = np.arange(1, h_max + 1)
        t = np.full(h_max, n - 1)
        X, scale = self._features(
            self._y, self._origin, t, h,
            pd.DatetimeIndex(exog["date"]),
            exog["onpromotion"].to_numpy(), exog["is_holiday"].to_numpy(),
        )
        return np.maximum(self._model.predict(X) * scale, 0.0)

    @property
    def used_fallback(self) -> bool:
        return self._used_fallback

    def feature_importance(self) -> pd.DataFrame | None:
        """Gain-based feature importance, normalised to sum to 1."""
        if self._used_fallback or self._model is None:
            return None
        gain = self._model.booster_.feature_importance(importance_type="gain")
        total = gain.sum() or 1.0
        df = pd.DataFrame({
            "feature": FEATURE_NAMES,
            "label": [FEATURE_LABELS.get(f, f.replace("_", " ")) for f in FEATURE_NAMES],
            "importance": gain / total,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
