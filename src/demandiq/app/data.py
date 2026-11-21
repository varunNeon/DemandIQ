"""Cached data access for the Streamlit app."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict

import pandas as pd
import streamlit as st

from demandiq.config import settings
from demandiq.data.ingestion import ingest_directory
from demandiq.database.connection import get_db, init_db
from demandiq.database.models import Product, SalesHistory, Store
from demandiq.database.repositories import get_sales_dataframe
from demandiq.services.forecast_service import ModelResult, forecast_series
from demandiq.services.inventory_service import (
    InventoryInputs,
    service_level_tradeoff,
    simulate_policy,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Preparing the demo database…")
def ensure_database() -> int:
    """Create tables and load the demo CSVs on first run. Returns the sales row count."""
    init_db()
    with get_db() as session:
        count = session.query(SalesHistory.id).count()
    if count == 0:
        ingest_directory(settings.DEMO_DATA_DIR, data_source="demo")
        with get_db() as session:
            count = session.query(SalesHistory.id).count()
    return count


@st.cache_data(show_spinner=False)
def load_catalog() -> tuple[pd.DataFrame, pd.DataFrame]:
    with get_db() as session:
        stores = pd.DataFrame(
            [{"store_id": s.store_id, "city": s.city, "store_type": s.store_type}
             for s in session.query(Store).order_by(Store.store_id)]
        )
        products = pd.DataFrame(
            [{"product_id": p.product_id, "family": p.family, "category": p.category,
              "perishable": p.perishable}
             for p in session.query(Product).order_by(Product.product_id)]
        )
    return stores, products


@st.cache_data(show_spinner=False)
def load_series(store_id: str, product_id: str) -> pd.DataFrame:
    with get_db() as session:
        df = get_sales_dataframe(session, store_id=store_id, product_id=product_id)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["onpromotion"] = df["onpromotion"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_all_sales() -> pd.DataFrame:
    """All demo sales, read from the tracked CSV for speed."""
    df = pd.read_csv(settings.DEMO_DATA_DIR / "sales.csv", parse_dates=["date"])
    return df


@st.cache_data(show_spinner=False)
def load_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    base = settings.DEMO_DATA_DIR
    try:
        bench = pd.read_csv(base / "benchmark.csv")
        by_h = pd.read_csv(base / "benchmark_by_horizon.csv")
        meta = json.loads((base / "benchmark_meta.json").read_text())
    except FileNotFoundError:
        return None
    return bench, by_h, meta


@st.cache_data(show_spinner=False, max_entries=64)
def run_forecasts(
    store_id: str,
    product_id: str,
    models: tuple[str, ...],
    horizon: int,
    promo_days: tuple[int, ...] = (),
) -> dict[str, ModelResult]:
    """Backtest and forecast one series. Cached per input combination."""
    warnings.filterwarnings("ignore")
    df = load_series(store_id, product_id)
    plan = None
    if promo_days:
        plan = pd.DataFrame({"onpromotion": [1 if d in promo_days else 0 for d in range(1, horizon + 1)]})
    return forecast_series(df, list(models), horizon, settings.BACKTEST_SPLITS, promo_plan=plan)


@st.cache_data(show_spinner=False, max_entries=64)
def run_simulation(store_id: str, product_id: str, inputs: dict, days: int = 182) -> pd.DataFrame:
    df = load_series(store_id, product_id)
    demand = df.set_index("date")["units_sold"]
    sim = simulate_policy(demand, InventoryInputs(**inputs), days=days)
    return sim.to_dataframe()


@st.cache_data(show_spinner=False, max_entries=64)
def run_tradeoff(store_id: str, product_id: str, inputs: dict, days: int = 182) -> pd.DataFrame:
    df = load_series(store_id, product_id)
    demand = df.set_index("date")["units_sold"]
    return service_level_tradeoff(demand, InventoryInputs(**inputs), days=days)


def inputs_dict(inputs: InventoryInputs) -> dict:
    return asdict(inputs)
