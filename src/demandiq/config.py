"""
DemandIQ Configuration Module.

All settings are loaded from environment variables (with .env file support).
This module provides a single Config object used throughout the application.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=False)


class Config:
    """Central configuration object. Reads from environment variables."""

    # --- Paths ---
    ROOT_DIR: Path = _ROOT
    DATA_DIR: Path = _ROOT / "data"
    DEMO_DATA_DIR: Path = _ROOT / "data" / "demo"
    RAW_DATA_DIR: Path = _ROOT / "data" / "raw"
    MODELS_DIR: Path = _ROOT / "models"
    LOGS_DIR: Path = _ROOT / "logs"

    # --- Database ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_ROOT / 'demandiq.db'}")

    # --- Application mode ---
    DATA_MODE: str = os.getenv("DATA_MODE", "demo")  # "demo" | "real"

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Forecasting ---
    ENABLED_MODELS: list[str] = [
        m.strip()
        for m in os.getenv("ENABLED_MODELS", "naive,seasonal_naive,ets,lgbm").split(",")
    ]
    DEFAULT_HORIZON: int = int(os.getenv("DEFAULT_HORIZON", "14"))
    BACKTEST_SPLITS: int = int(os.getenv("BACKTEST_SPLITS", "3"))
    MIN_TRAIN_PERIODS: int = int(os.getenv("MIN_TRAIN_PERIODS", "60"))  # days

    # --- Reproducibility ---
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

    # --- Prediction intervals ---
    PI_LEVELS: list[float] = [0.80, 0.95]  # nominal coverage targets

    # --- Inventory defaults (can be overridden per product/store) ---
    DEFAULT_LEAD_TIME_DAYS: int = 7
    DEFAULT_SERVICE_LEVEL: float = 0.90
    DEFAULT_HOLDING_COST_PER_UNIT_DAY: float = 0.05
    DEFAULT_ORDERING_COST: float = 50.0
    DEFAULT_SHORTAGE_COST_PER_UNIT: float = 5.0
    DEFAULT_UNIT_COST: float = 10.0

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create required directories if they don't exist."""
        for d in [cls.DATA_DIR, cls.DEMO_DATA_DIR, cls.RAW_DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Config()
