"""
Query repository layer for DemandIQ.

All database reads go through this module. Uses parameterized queries only.
Never builds SQL strings by concatenation.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from demandiq.database.models import (
    DataQualityEvent,
    Forecast,
    ForecastRun,
    InventoryParameters,
    Product,
    ReplenishmentRecommendation,
    SalesHistory,
    Store,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Store / Product helpers
# ---------------------------------------------------------------------------

def get_all_stores(session: Session) -> list[Store]:
    return session.query(Store).order_by(Store.store_id).all()


def get_all_products(session: Session) -> list[Product]:
    return session.query(Product).order_by(Product.product_id).all()


def get_store_ids(session: Session) -> list[str]:
    return [r[0] for r in session.query(Store.store_id).order_by(Store.store_id).all()]


def get_product_ids(session: Session) -> list[str]:
    return [r[0] for r in session.query(Product.product_id).order_by(Product.product_id).all()]


# ---------------------------------------------------------------------------
# Sales history
# ---------------------------------------------------------------------------

def get_sales_dataframe(
    session: Session,
    store_id: str | None = None,
    product_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Return sales history as a DataFrame.

    Filters are optional; all conditions are AND-combined.
    """
    query = session.query(SalesHistory)
    if store_id:
        query = query.filter(SalesHistory.store_id == store_id)
    if product_id:
        query = query.filter(SalesHistory.product_id == product_id)
    if start_date:
        query = query.filter(SalesHistory.date >= start_date)
    if end_date:
        query = query.filter(SalesHistory.date <= end_date)

    query = query.order_by(SalesHistory.date)
    rows = query.all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "date": r.date,
                "store_id": r.store_id,
                "product_id": r.product_id,
                "units_sold": r.units_sold,
                "onpromotion": r.onpromotion,
                "is_holiday": r.is_holiday,
                "data_source": r.data_source,
            }
            for r in rows
        ]
    )


def count_sales_rows(session: Session) -> int:
    return session.query(SalesHistory).count()


def get_sales_date_range(session: Session) -> tuple[date | None, date | None]:
    from sqlalchemy import func as sqlfunc
    result = session.query(
        sqlfunc.min(SalesHistory.date),
        sqlfunc.max(SalesHistory.date),
    ).first()
    return result if result else (None, None)


# ---------------------------------------------------------------------------
# Forecast runs and forecasts
# ---------------------------------------------------------------------------

def get_forecast_runs(
    session: Session,
    store_id: str | None = None,
    product_id: str | None = None,
    model_name: str | None = None,
) -> list[ForecastRun]:
    q = session.query(ForecastRun)
    if store_id:
        q = q.filter(ForecastRun.store_id == store_id)
    if product_id:
        q = q.filter(ForecastRun.product_id == product_id)
    if model_name:
        q = q.filter(ForecastRun.model_name == model_name)
    return q.order_by(ForecastRun.created_at.desc()).all()


def get_latest_forecast_run(
    session: Session,
    store_id: str,
    product_id: str,
    model_name: str,
) -> ForecastRun | None:
    return (
        session.query(ForecastRun)
        .filter(
            ForecastRun.store_id == store_id,
            ForecastRun.product_id == product_id,
            ForecastRun.model_name == model_name,
        )
        .order_by(ForecastRun.created_at.desc())
        .first()
    )


def get_forecasts_dataframe(session: Session, forecast_run_id: int) -> pd.DataFrame:
    rows = (
        session.query(Forecast)
        .filter(Forecast.forecast_run_id == forecast_run_id)
        .order_by(Forecast.forecast_date)
        .all()
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "forecast_date": r.forecast_date,
                "store_id": r.store_id,
                "product_id": r.product_id,
                "point_forecast": r.point_forecast,
                "lower_80": r.lower_80,
                "upper_80": r.upper_80,
                "lower_95": r.lower_95,
                "upper_95": r.upper_95,
                "is_future": r.is_future,
            }
            for r in rows
        ]
    )


def get_backtest_summary(session: Session) -> pd.DataFrame:
    """Return one row per ForecastRun with evaluation metrics."""
    runs = session.query(ForecastRun).filter(ForecastRun.eval_mae.isnot(None)).all()
    if not runs:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "run_id": r.forecast_run_id,
                "model": r.model_name,
                "store_id": r.store_id,
                "product_id": r.product_id,
                "training_cutoff": r.training_cutoff,
                "horizon_days": r.horizon_days,
                "MAE": r.eval_mae,
                "RMSE": r.eval_rmse,
                "WAPE": r.eval_wape,
                "MASE": r.eval_mase,
                "Coverage_80": r.eval_coverage_80,
                "Coverage_95": r.eval_coverage_95,
                "created_at": r.created_at,
            }
            for r in runs
        ]
    )


# ---------------------------------------------------------------------------
# Inventory parameters
# ---------------------------------------------------------------------------

def get_inventory_params(
    session: Session, store_id: str, product_id: str
) -> InventoryParameters | None:
    return (
        session.query(InventoryParameters)
        .filter(
            InventoryParameters.store_id == store_id,
            InventoryParameters.product_id == product_id,
        )
        .first()
    )


def upsert_inventory_params(session: Session, params: dict[str, Any]) -> InventoryParameters:
    """Insert or update inventory parameters for a store-product pair."""
    existing = get_inventory_params(session, params["store_id"], params["product_id"])
    if existing:
        for k, v in params.items():
            if k not in ("store_id", "product_id"):
                setattr(existing, k, v)
        return existing
    else:
        obj = InventoryParameters(**params)
        session.add(obj)
        return obj


# ---------------------------------------------------------------------------
# Replenishment recommendations
# ---------------------------------------------------------------------------

def get_latest_recommendation(
    session: Session, store_id: str, product_id: str
) -> ReplenishmentRecommendation | None:
    return (
        session.query(ReplenishmentRecommendation)
        .filter(
            ReplenishmentRecommendation.store_id == store_id,
            ReplenishmentRecommendation.product_id == product_id,
        )
        .order_by(ReplenishmentRecommendation.created_at.desc())
        .first()
    )


def get_all_recommendations(session: Session) -> pd.DataFrame:
    rows = (
        session.query(ReplenishmentRecommendation)
        .order_by(ReplenishmentRecommendation.created_at.desc())
        .all()
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "store_id": r.store_id,
                "product_id": r.product_id,
                "created_at": r.created_at,
                "inventory_position": r.inventory_position,
                "reorder_point": r.reorder_point,
                "suggested_order_qty": r.suggested_order_qty,
                "expected_shortage_risk": r.expected_shortage_risk,
                "explanation": r.explanation,
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# Data quality events
# ---------------------------------------------------------------------------

def get_dq_events(session: Session, limit: int = 200) -> pd.DataFrame:
    rows = (
        session.query(DataQualityEvent)
        .order_by(DataQualityEvent.occurred_at.desc())
        .limit(limit)
        .all()
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "occurred_at": r.occurred_at,
                "dataset": r.dataset,
                "issue_type": r.issue_type,
                "severity": r.severity,
                "row_count": r.row_count,
                "details": r.details,
            }
            for r in rows
        ]
    )


def log_dq_event(
    session: Session,
    dataset: str,
    issue_type: str,
    severity: str,
    row_count: int | None = None,
    details: str | None = None,
) -> None:
    event = DataQualityEvent(
        dataset=dataset,
        issue_type=issue_type,
        severity=severity,
        row_count=row_count,
        details=details,
    )
    session.add(event)
