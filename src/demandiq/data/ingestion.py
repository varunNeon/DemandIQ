"""
Data ingestion pipeline for DemandIQ.

Loads CSVs, validates them, and upserts into the database.
Idempotent: re-running ingestion will not create duplicate records.

Missing value treatment:
- Missing ROWS: treated as "unknown" (the record is simply absent).
  We do NOT fill missing rows with zero — a missing row ≠ zero sales.
  This distinction is critical for forecasting.
- Missing VALUES within a row: numeric NaN → dropped with DQ event logged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from demandiq.config import settings
from demandiq.data.validation import validate_products, validate_sales, validate_stores
from demandiq.database.connection import get_db, init_db
from demandiq.database.models import Product, SalesHistory, Store
from demandiq.database.repositories import log_dq_event

logger = logging.getLogger(__name__)


def _upsert_stores(session: Session, df: pd.DataFrame) -> int:
    """Insert stores, skip existing (idempotent)."""
    count = 0
    for _, row in df.iterrows():
        existing = session.query(Store).filter_by(store_id=str(row["store_id"])).first()
        if not existing:
            session.add(
                Store(
                    store_id=str(row["store_id"]),
                    city=str(row.get("city", "") or ""),
                    state=str(row.get("state", "") or ""),
                    store_type=str(row.get("store_type", "") or ""),
                    cluster=int(row["cluster"]) if pd.notna(row.get("cluster")) else None,
                )
            )
            count += 1
    return count


def _upsert_products(session: Session, df: pd.DataFrame) -> int:
    """Insert products, skip existing (idempotent)."""
    count = 0
    for _, row in df.iterrows():
        existing = session.query(Product).filter_by(product_id=str(row["product_id"])).first()
        if not existing:
            perishable = bool(row.get("perishable", False))
            if isinstance(perishable, str):
                perishable = perishable.lower() in ("true", "1", "yes")
            session.add(
                Product(
                    product_id=str(row["product_id"]),
                    family=str(row.get("family", "UNKNOWN")),
                    category=str(row.get("category", "") or ""),
                    perishable=perishable,
                )
            )
            count += 1
    return count


def _bulk_upsert_sales(session: Session, df: pd.DataFrame, data_source: str) -> tuple[int, int]:
    """
    Bulk upsert sales rows.

    Uses INSERT OR IGNORE for SQLite, or equivalent for PostgreSQL.
    Returns (inserted_count, skipped_count).
    """
    if df.empty:
        return 0, 0

    # Normalize
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["store_id"] = df["store_id"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df["onpromotion"] = df.get("onpromotion", False).astype(bool)
    df["is_holiday"] = df.get("is_holiday", False).astype(bool)
    df["data_source"] = data_source

    # Drop invalid rows
    df = df.dropna(subset=["date", "store_id", "product_id", "units_sold"])
    df = df[df["units_sold"] >= 0]
    df = df.drop_duplicates(subset=["date", "store_id", "product_id"], keep="first")

    total = len(df)
    inserted = 0

    # Batch insert using ORM (portable across SQLite and PostgreSQL)
    # For large datasets this should be replaced with bulk_insert_mappings
    BATCH_SIZE = 500
    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start : start + BATCH_SIZE]
        for _, row in batch.iterrows():
            existing = (
                session.query(SalesHistory)
                .filter_by(
                    date=row["date"],
                    store_id=row["store_id"],
                    product_id=row["product_id"],
                )
                .first()
            )
            if not existing:
                session.add(
                    SalesHistory(
                        date=row["date"],
                        store_id=row["store_id"],
                        product_id=row["product_id"],
                        units_sold=float(row["units_sold"]),
                        onpromotion=bool(row["onpromotion"]),
                        is_holiday=bool(row["is_holiday"]),
                        data_source=data_source,
                    )
                )
                inserted += 1
        session.flush()  # Flush each batch

    skipped = total - inserted
    return inserted, skipped


def ingest_directory(data_dir: Path, data_source: str = "demo") -> dict:
    """
    Ingest all CSVs from a data directory into the database.

    Expected files:
        - stores.csv
        - products.csv
        - sales.csv

    Returns a summary dictionary.
    """
    init_db()

    data_dir = Path(data_dir)
    summary = {
        "data_dir": str(data_dir),
        "data_source": data_source,
        "stores_inserted": 0,
        "products_inserted": 0,
        "sales_inserted": 0,
        "sales_skipped": 0,
        "validation_issues": [],
        "success": False,
    }

    # --- Stores ---
    stores_path = data_dir / "stores.csv"
    if stores_path.exists():
        stores_df = pd.read_csv(stores_path)
        val = validate_stores(stores_df)
        summary["validation_issues"].extend(val.issues)
        if not val.has_errors:
            with get_db() as session:
                for issue in val.issues:
                    log_dq_event(session, **issue)
                n = _upsert_stores(session, stores_df)
                summary["stores_inserted"] = n
            logger.info("Stores ingested: %d inserted", n)
        else:
            logger.error("Stores validation failed: %s", val.summary())
    else:
        logger.warning("stores.csv not found in %s", data_dir)

    # --- Products ---
    products_path = data_dir / "products.csv"
    if products_path.exists():
        products_df = pd.read_csv(products_path)
        val = validate_products(products_df)
        summary["validation_issues"].extend(val.issues)
        if not val.has_errors:
            with get_db() as session:
                for issue in val.issues:
                    log_dq_event(session, **issue)
                n = _upsert_products(session, products_df)
                summary["products_inserted"] = n
            logger.info("Products ingested: %d inserted", n)
    else:
        logger.warning("products.csv not found in %s", data_dir)

    # --- Sales ---
    sales_path = data_dir / "sales.csv"
    if sales_path.exists():
        logger.info("Loading sales.csv (this may take a moment for large files)...")
        sales_df = pd.read_csv(sales_path)
        val = validate_sales(sales_df)
        summary["validation_issues"].extend(val.issues)

        if val.has_errors:
            logger.error("Sales validation has errors: %s", val.summary())
        else:
            logger.info("Sales validation passed. %s", val.summary())
            with get_db() as session:
                for issue in val.issues:
                    log_dq_event(session, **issue)
                inserted, skipped = _bulk_upsert_sales(session, sales_df, data_source)
                summary["sales_inserted"] = inserted
                summary["sales_skipped"] = skipped
            logger.info(
                "Sales ingested: %d inserted, %d skipped (already existed)",
                inserted, skipped,
            )
    else:
        logger.warning("sales.csv not found in %s", data_dir)

    summary["success"] = True
    return summary
