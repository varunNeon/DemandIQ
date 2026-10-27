"""
LightGBM machine learning forecaster for DemandIQ.

Uses a direct multi-step forecasting approach: one model per horizon step,
or a single model with the horizon step as a feature.

We use the recursive strategy (predict h=1, then use that prediction as
a lag to predict h=2, etc.) for simplicity and interpretability.

Feature set: lag features, rolling statistics, calendar features.
All features are constructed to be leakage-free.

Minimum history: max(LAG_PERIODS) + 30 days of clean training data.
Falls back to SeasonalNaiveForecaster for sparse/short series.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from demandiq.config import settings
from demandiq.data.features import LAG_PERIODS, build_time_series_features, get_feature_columns
from demandiq.forecasting.baselines import SeasonalNaiveForecaster

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = max(LAG_PERIODS) + 30  # minimum rows after feature construction


class LGBMForecaster:
    """
    LightGBM direct multi-step forecaster.

    Trains on lag/rolling/calendar features.
    Uses recursive prediction for horizons > 1.
    """

    name: str = "lgbm"

    def __init__(self, horizon: int = 14, n_estimators: int = 200, num_leaves: int = 31) -> None:
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self._models: list = []  # one model per horizon step (direct strategy)
        self._feature_cols: list[str] = []
        self._fallback: SeasonalNaiveForecaster | None = None
        self._used_fallback: bool = False
        self._last_row: pd.Series | None = None  # last row of training data (for prediction)

    def _make_direct_targets(
        self, df: pd.DataFrame, target_col: str = "units_sold"
    ) -> list[tuple[pd.DataFrame, pd.Series]]:
        """
        For direct multi-step forecasting, create (X, y_h) pairs for each step h.

        y_h at row t = target at t + h (shifted back by h).
        Training rows where y_h is NaN (end of series) are dropped.
        """
        feature_df = build_time_series_features(df, target_col=target_col)
        feat_cols = [c for c in get_feature_columns() if c in feature_df.columns]

        # Drop rows with NaN features (beginning of series due to lag construction)
        feature_df = feature_df.dropna(subset=[c for c in feat_cols if "lag" in c or "roll" in c])

        result = []
        for h in range(1, self.horizon + 1):
            y_h = feature_df[target_col].shift(-h)
            mask = y_h.notna()
            X_h = feature_df.loc[mask, feat_cols]
            y_h_clean = y_h[mask]
            result.append((X_h, y_h_clean))

        return result, feat_cols

    def fit(self, series_df: pd.DataFrame, target_col: str = "units_sold") -> "LGBMForecaster":
        """
        Fit LightGBM models on the training series.

        Parameters
        ----------
        series_df : DataFrame with at minimum [date, units_sold].
                    May also contain onpromotion, is_holiday.
        """
        import lightgbm as lgb

        self._used_fallback = False
        self._models = []

        if len(series_df) < MIN_TRAIN_ROWS:
            logger.warning(
                "LGBM: insufficient training data (%d rows, need %d). Falling back to SeasonalNaive.",
                len(series_df), MIN_TRAIN_ROWS,
            )
            self._fallback = SeasonalNaiveForecaster(period=7)
            self._fallback.fit(pd.Series(series_df[target_col].values))
            self._used_fallback = True
            return self

        pairs, feat_cols = self._make_direct_targets(series_df, target_col)
        self._feature_cols = feat_cols

        # Store the last feature row for prediction
        feature_df = build_time_series_features(series_df, target_col=target_col)
        self._last_row = feature_df.iloc[-1:].copy()

        for h, (X_h, y_h) in enumerate(pairs, start=1):
            if len(X_h) < 20:
                logger.warning("LGBM: too few samples for h=%d (%d rows). Skipping.", h, len(X_h))
                self._models.append(None)
                continue

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                learning_rate=0.05,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=settings.RANDOM_SEED,
                verbose=-1,
            )
            model.fit(X_h, y_h)
            self._models.append(model)

        logger.debug("LGBM fitted: %d horizon models", len(self._models))
        return self

    def predict(self, horizon: int | None = None) -> np.ndarray:
        """Predict for each step up to `horizon`."""
        if self._used_fallback:
            h = horizon or self.horizon
            return self._fallback.predict(h)

        if not self._models:
            raise RuntimeError("LGBMForecaster must be fitted before predict().")

        h = min(horizon or self.horizon, len(self._models))
        preds = []
        for i, model in enumerate(self._models[:h]):
            if model is None:
                preds.append(preds[-1] if preds else 0.0)
                continue
            X = self._last_row[self._feature_cols]
            pred = float(model.predict(X)[0])
            preds.append(max(pred, 0.0))

        return np.array(preds)

    def predict_with_intervals(
        self,
        horizon: int | None = None,
        levels: list[float] | None = None,
        calibration_residuals: dict[float, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Produce point forecast with prediction intervals.

        Intervals are obtained via conformal prediction if calibration_residuals
        are provided (preferred), otherwise returns point forecast only.
        """
        h = horizon or self.horizon
        point = self.predict(h)
        result: dict[str, np.ndarray] = {"point": point}

        levels = levels or [0.80, 0.95]
        if calibration_residuals:
            for level in levels:
                if level in calibration_residuals:
                    resid = calibration_residuals[level]
                    if len(resid) > 0:
                        q = np.quantile(np.abs(resid), level)
                        label = f"{int(level * 100)}"
                        result[f"lower_{label}"] = np.maximum(point - q, 0.0)
                        result[f"upper_{label}"] = point + q

        return result

    @property
    def used_fallback(self) -> bool:
        return self._used_fallback

    def feature_importance(self) -> pd.DataFrame | None:
        """Return feature importances from the h=1 model."""
        if self._used_fallback or not self._models or self._models[0] is None:
            return None
        model = self._models[0]
        return pd.DataFrame(
            {"feature": self._feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
