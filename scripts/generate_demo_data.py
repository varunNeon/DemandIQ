"""
Demo data generator for DemandIQ.

Generates a small, reproducible synthetic dataset that mimics real retail sales patterns:
  - Weekly seasonality (higher sales on weekends)
  - Slight upward trend over time
  - Promotion spikes (+30-50% lift)
  - Occasional holidays (zero or near-zero sales)
  - Some zero-demand days (stockouts or slow movers)
  - Log-normal noise for realistic skewness

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
HOLIDAYS_2013 = ["2013-01-01", "2013-02-11", "2013-04-01", "2013-05-01", "2013-05-24", "2013-07-24", "2013-08-10", "2013-10-09", "2013-11-02", "2013-11-03", "2013-12-25"]
HOLIDAYS_2014 = ["2014-01-01", "2014-03-03", "2014-04-18", "2014-05-01", "2014-05-24", "2014-07-24", "2014-08-10", "2014-10-09", "2014-11-02", "2014-11-03", "2014-12-25"]
HOLIDAYS_2015 = ["2015-01-01", "2015-02-16", "2015-04-03", "2015-05-01", "2015-05-24", "2015-07-24", "2015-08-10", "2015-10-09", "2015-11-02", "2015-11-03", "2015-12-25"]
ALL_HOLIDAYS = set(HOLIDAYS_2013 + HOLIDAYS_2014 + HOLIDAYS_2015)


def generate_sales(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    base_demand: float,
    store_factor: float,
    product_factor: float,
    promotion_dates: set[str],
    holiday_dates: set[str],
) -> pd.Series:
    """
    Generate daily sales for a single store-product series.

    Signal model:
        sales(t) = base * store_factor * product_factor
                   * trend(t)
                   * weekly_seasonality(t)
                   * promotion_effect(t)
                   * holiday_effect(t)
                   * noise(t)

    All parameters are seeded for reproducibility.
    """
    n = len(dates)
    t = np.arange(n)

    # Trend: 0-15% total growth over the period
    trend = 1.0 + rng.uniform(0.0, 0.15) * t / max(n, 1)

    # Weekly seasonality: day-of-week multipliers
    dow_weights = np.array([1.0, 0.9, 0.95, 1.05, 1.2, 1.4, 1.3])  # Mon-Sun
    weekly_season = np.array([dow_weights[d.weekday()] for d in dates])

    # Annual seasonality: mild December peak
    annual_season = np.asarray(
        1.0 + 0.1 * np.sin(2 * np.pi * (dates.dayofyear - 355) / 365),
        dtype=float,
    )

    # Promotion effect: +30-50% lift on promo days
    promo_flags = np.array([d.strftime("%Y-%m-%d") in promotion_dates for d in dates], dtype=float)
    promo_lift = rng.uniform(0.3, 0.5)
    promo_effect = 1.0 + promo_lift * promo_flags

    # Holiday effect: demand drops to 5-30% on holidays
    holiday_flags = np.array([d.strftime("%Y-%m-%d") in holiday_dates for d in dates], dtype=float)
    holiday_drop = rng.uniform(0.05, 0.30)
    holiday_effect = 1.0 - (1.0 - holiday_drop) * holiday_flags

    # Log-normal multiplicative noise
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=n)

    raw_sales = (
        base_demand
        * store_factor
        * product_factor
        * trend
        * weekly_season
        * annual_season
        * promo_effect
        * holiday_effect
        * noise
    )

    # Clip negatives (shouldn't happen, but defensive)
    raw_sales = np.maximum(raw_sales, 0.0)

    # Introduce genuine zeros for slow-moving products (sparse demand)
    # Work on numpy array explicitly before wrapping in pd.Series
    raw_sales = raw_sales.copy()
    zero_prob = max(0.0, rng.uniform(-0.05, 0.15))  # 0-15% chance of zero-day
    zero_mask = rng.random(n) < zero_prob
    raw_sales[zero_mask] = 0.0

    return pd.Series(np.round(raw_sales, 2), index=dates, name="units_sold")


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
    base_demands = {p["product_id"]: rng.uniform(10, 150) for p in products}

    for store in stores:
        promo_dates = generate_promotions(rng, dates)
        for product in products:
            sales = generate_sales(
                rng=rng,
                dates=dates,
                base_demand=base_demands[product["product_id"]],
                store_factor=store_factors[store["store_id"]],
                product_factor=product_factors[product["product_id"]],
                promotion_dates=promo_dates,
                holiday_dates=ALL_HOLIDAYS,
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
            "lognormal_noise",
            "sparse_zero_demand",
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
