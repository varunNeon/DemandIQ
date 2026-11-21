from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Point the app's engine at a fresh SQLite file for one test."""
    from demandiq.config import settings
    from demandiq.database import connection

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(connection, "_engine", None)
    monkeypatch.setattr(connection, "_SessionLocal", None)
    connection.init_db()
    yield
    if connection._engine is not None:
        connection._engine.dispose()


@pytest.fixture()
def weekly_series_df() -> pd.DataFrame:
    """Two years of daily demand with a weekly pattern, promotions and noise."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2014-01-01", periods=400, freq="D")
    weekly = np.array([0.9, 0.85, 0.9, 1.0, 1.1, 1.4, 1.25])[dates.dayofweek]
    promo = (rng.random(len(dates)) < 0.1).astype(int)
    mean = 80 * weekly * (1 + 0.5 * promo)
    units = rng.poisson(mean).astype(float)
    return pd.DataFrame({
        "date": dates, "units_sold": units, "onpromotion": promo, "is_holiday": 0,
    })
