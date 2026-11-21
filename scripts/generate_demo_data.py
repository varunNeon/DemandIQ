"""
Demo data generator for DemandIQ.

Generates a small, reproducible synthetic dataset that mimics real retail sales patterns:
  - Weekly seasonality (higher sales on weekends)
  - Gentle trend and a slowly drifting local level
  - Promotion spikes (+25-60% lift)
  - Product-specific annual seasonality (e.g. a December peak for toys)
  - Holidays with sharply reduced demand
  - Whole-unit, over-dispersed (negative binomial) daily counts
  - Occasional stock-out days with zero sales

This data is CLEARLY LABELED as synthetic demo data.
It is NOT presented as real retail sales.

Usage:
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --output data/demo --stores 3 --products 10 --days 730
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure src is on path when running as script
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


STORE_METADATA = [
    {"store_id": "S001", "city": "Quito", "state": "Pichincha", "store_type": "A", "cluster": 1},
    {"store_id": "S002", "city": "Guayaquil", "state": "Guayas", "store_type": "B", "cluster": 2},
    {"store_id": "S003", "city": "Cuenca", "state": "Azuay", "store_type": "A", "cluster": 1},
    {"store_id": "S004", "city": "Ambato", "state": "Tungurahua", "store_type": "C", "cluster": 3},
    {"store_id": "S005", "city": "Loja", "state": "Loja", "store_type": "B", "cluster": 2},
]

PRODUCT_METADATA = [
    {"product_id": "GROCERY_001", "family": "GROCERY I", "category": "DRY GOODS", "perishable": False},
    {"product_id": "GROCERY_002", "family": "GROCERY I", "category": "BEVERAGES", "perishable": False},
    {"product_id": "PRODUCE_001", "family": "PRODUCE", "category": "FRESH VEGETABLES", "perishable": True},
    {"product_id": "PRODUCE_002", "family": "PRODUCE", "category": "FRESH FRUITS", "perishable": True},
    {"product_id": "DAIRY_001", "family": "DAIRY", "category": "MILK", "perishable": True},
    {"product_id": "BAKERY_001", "family": "BAKERY", "category": "BREAD", "perishable": True},
    {"product_id": "BEVERAGES_001", "family": "BEVERAGES", "category": "SOFT DRINKS", "perishable": False},
    {"product_id": "CLEANING_001", "family": "CLEANING", "category": "HOUSEHOLD", "perishable": False},
    {"product_id": "PERSONAL_001", "family": "PERSONAL CARE", "category": "HYGIENE", "perishable": False},
    {"product_id": "FROZEN_001", "family": "FROZEN FOODS", "category": "FROZEN MEALS", "perishable": True},
    {"product_id": "SNACKS_001", "family": "SNACKS", "category": "CHIPS", "perishable": False},
    {"product_id": "MEATS_001", "family": "MEATS", "category": "POULTRY", "perishable": True},
    {"product_id": "SEAFOOD_001", "family": "SEAFOOD", "category": "FRESH FISH", "perishable": True},
    {"product_id": "DELI_001", "family": "DELI", "category": "PREPARED FOODS", "perishable": True},
    {"product_id": "PET_001", "family": "PET SUPPLIES", "category": "PET FOOD", "perishable": False},
    {"product_id": "HARDWARE_001", "family": "HARDWARE", "category": "TOOLS", "perishable": False},
    {"product_id": "BEAUTY_001", "family": "BEAUTY", "category": "COSMETICS", "perishable": False},
    {"product_id": "AUTOMOTIVE_001", "family": "AUTOMOTIVE", "category": "ACCESSORIES", "perishable": False},
    {"product_id": "TOYS_001", "family": "TOYS", "category": "GAMES", "perishable": False},
    {"product_id": "ELECTRONICS_001", "family": "ELECTRONICS", "category": "ACCESSORIES", "perishable": False},
]

# Ecuadorian national holidays (simplified)
from demandiq.data.calendar import KNOWN_HOLIDAYS  # noqa: E402

ALL_HOLIDAYS = set(KNOWN_HOLIDAYS)

# Seasonal character per product family: (annual amplitude, peak day-of-year).
# Toys and electronics peak before Christmas; beverages peak in the warm season.
FAMILY_SEASONALITY = {
    "TOYS": (0.55, 355),
    "ELECTRONICS": (0.40, 352),
    "BEAUTY": (0.20, 350),
    "BEVERAGES": (0.25, 70),
    "GROCERY I": (0.10, 355),
    "FROZEN FOODS": (0.18, 60),
}
DEFAULT_SEASONALITY = (0.08, 355)

# Day-of-week shape (Mon..Sun): weekend shopping peak.
DOW_SHAPE = np.array([0.92, 0.85, 0.90, 1.00, 1.12, 1.32, 1.22])


def generate_sales(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    base_demand: float,
    store_factor: float,
    product_factor: float,
    promotion_dates: set[str],
    holiday_dates: set[str],
    weekly_strength: float = 1.0,
    annual_amplitude: float = 0.10,
    annual_peak_doy: int = 355,
) -> pd.Series:
    """
    Generate daily unit sales (whole units) for a single store-product series.

    Expected demand:
        mean(t) = base * store_factor * product_factor
                  * trend(t) * weekly(t) * annual(t)
                  * promotion(t) * holiday(t) * local_level(t)

    Realised sales are drawn from a negative binomial (gamma-Poisson) around
    that mean, so values are integers and over-dispersed like real retail
    counts. A small share of days are zeroed to mimic stock-outs.
    """
    n = len(dates)
    t = np.arange(n)

    # Trend: between -5% and +25% over the full period
    trend = 1.0 + rng.uniform(-0.05, 0.25) * t / max(n, 1)

    # Weekly pattern, with product-specific strength
    weekly = 1.0 + weekly_strength * (DOW_SHAPE[dates.dayofweek] - 1.0)

    # Annual pattern peaking on `annual_peak_doy`
    annual = 1.0 + annual_amplitude * np.cos(
        2 * np.pi * (np.asarray(dates.dayofyear, dtype=float) - annual_peak_doy) / 365.25
    )

    # Promotions lift demand by 25-60%
    promo_flags = np.array([d.strftime("%Y-%m-%d") in promotion_dates for d in dates], dtype=float)
    promo_effect = 1.0 + rng.uniform(0.25, 0.60) * promo_flags

    # Holidays cut demand to 10-40% of normal
    holiday_flags = np.array([d.strftime("%Y-%m-%d") in holiday_dates for d in dates], dtype=float)
    holiday_effect = 1.0 - (1.0 - rng.uniform(0.10, 0.40)) * holiday_flags

    # Slowly drifting local level (AR(1) in log space)
    log_level = np.zeros(n)
    shocks = rng.normal(0.0, 0.025, size=n)
    for i in range(1, n):
        log_level[i] = 0.97 * log_level[i - 1] + shocks[i]
    local_level = np.exp(log_level)

    mean = (
        base_demand * store_factor * product_factor
        * trend * weekly * annual * promo_effect * holiday_effect * local_level
    )
    mean = np.maximum(mean, 0.0)

    # Gamma-Poisson draw -> whole units with realistic over-dispersion
    dispersion = 25.0
    sales = rng.poisson(mean * rng.gamma(dispersion, 1.0 / dispersion, size=n)).astype(float)

    # Occasional stock-out days (0-3% of days)
    stockout_mask = rng.random(n) < rng.uniform(0.0, 0.03)
    sales[stockout_mask] = 0.0

    return pd.Series(sales.astype(int), index=dates, name="units_sold")


def generate_promotions(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    promo_rate: float = 0.08,
) -> set[str]:
    """Randomly select ~8% of dates as promotion days."""
    promo_mask = rng.random(len(dates)) < promo_rate
    return set(dates[promo_mask].strftime("%Y-%m-%d").tolist())


def run_generator(
    output_dir: Path,
    n_stores: int,
    n_products: int,
    n_days: int,
    seed: int = 42,
) -> None:
    """Generate demo CSVs and save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    end_date = pd.Timestamp("2015-12-31")
    start_date = end_date - pd.Timedelta(days=n_days - 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    stores = STORE_METADATA[:n_stores]
    products = PRODUCT_METADATA[:n_products]

    logger.info(
        "Generating demo data: %d stores × %d products × %d days (seed=%d)",
        n_stores, n_products, n_days, seed,
    )

    # --- 1. Stores CSV ---
    pd.DataFrame(stores).to_csv(output_dir / "stores.csv", index=False)
    logger.info("Wrote stores.csv (%d rows)", len(stores))

    # --- 2. Products CSV ---
    pd.DataFrame(products).to_csv(output_dir / "products.csv", index=False)
    logger.info("Wrote products.csv (%d rows)", len(products))

    # --- 3. Sales CSV ---
    all_rows = []
    store_factors = {s["store_id"]: rng.uniform(0.6, 1.4) for s in stores}
    product_factors = {p["product_id"]: rng.uniform(0.5, 2.0) for p in products}
    base_demands = {p["product_id"]: rng.uniform(8, 120) for p in products}
    weekly_strengths = {p["product_id"]: rng.uniform(0.5, 1.4) for p in products}

    for store in stores:
        promo_dates = generate_promotions(rng, dates)
        for product in products:
            amplitude, peak = FAMILY_SEASONALITY.get(product["family"], DEFAULT_SEASONALITY)
            sales = generate_sales(
                rng=rng,
                dates=dates,
                base_demand=base_demands[product["product_id"]],
                store_factor=store_factors[store["store_id"]],
                product_factor=product_factors[product["product_id"]],
                promotion_dates=promo_dates,
                holiday_dates=ALL_HOLIDAYS,
                weekly_strength=weekly_strengths[product["product_id"]],
                annual_amplitude=amplitude,
                annual_peak_doy=peak,
            )
            rows = pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "store_id": store["store_id"],
                    "product_id": product["product_id"],
                    "units_sold": sales.values,
                    "onpromotion": [d.strftime("%Y-%m-%d") in promo_dates for d in dates],
                    "is_holiday": [d.strftime("%Y-%m-%d") in ALL_HOLIDAYS for d in dates],
                }
            )
            all_rows.append(rows)

    sales_df = pd.concat(all_rows, ignore_index=True)

    # Sanity checks
    assert (sales_df["units_sold"] >= 0).all(), "Negative sales detected!"
    assert not sales_df.duplicated(["date", "store_id", "product_id"]).any(), "Duplicates detected!"

    sales_df.to_csv(output_dir / "sales.csv", index=False)

    total_rows = len(sales_df)
    zero_rows = (sales_df["units_sold"] == 0).sum()
    promo_rows = sales_df["onpromotion"].sum()
    holiday_rows = sales_df["is_holiday"].sum()

    logger.info(
        "Wrote sales.csv: %d rows | zeros=%d (%.1f%%) | promo days=%d | holiday days=%d",
        total_rows, zero_rows, 100 * zero_rows / total_rows, promo_rows, holiday_rows,
    )

    # --- 4. Metadata file (clearly labels this as synthetic) ---
    meta = {
        "data_type": "SYNTHETIC_DEMO",
        "warning": "This is synthetic data generated for demonstration purposes. It is NOT real retail data.",
        "seed": seed,
        "n_stores": n_stores,
        "n_products": n_products,
        "n_days": n_days,
        "date_range": f"{start_date.date()} to {end_date.date()}",
        "total_rows": total_rows,
        "signal_components": [
            "weekly_seasonality",
            "annual_seasonality",
            "linear_trend",
            "promotion_lift",
            "holiday_effect",
            "drifting_local_level",
            "negative_binomial_counts",
            "stockout_zero_days",
        ],
    }
    import json
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote metadata.json")
    logger.info("Demo data generation complete. Output: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic demo data for DemandIQ.")
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--stores", type=int, default=5)
    parser.add_argument("--products", type=int, default=20)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_generator(
        output_dir=args.output,
        n_stores=args.stores,
        n_products=args.products,
        n_days=args.days,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
