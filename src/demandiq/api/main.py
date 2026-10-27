"""Read-only API endpoints for the DemandIQ web client."""

from __future__ import annotations

import os
from datetime import date
from typing import Generator

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from demandiq.database.connection import get_session_factory
from demandiq.database.models import (
    Forecast,
    ForecastRun,
    Product,
    ReplenishmentRecommendation,
    SalesHistory,
    Store,
)

app = FastAPI(
    title="DemandIQ API",
    description="Data and forecasting results for the DemandIQ web application.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("DEMANDIQ_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def get_session() -> Generator[Session, None, None]:
    """Yield a short-lived database session for one request."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@app.get("/health")
def health(session: Session = Depends(get_session)) -> dict[str, str]:
    """Report API and database availability."""
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ok", "database": "connected"}


@app.get("/api/v1/overview")
def overview(session: Session = Depends(get_session)) -> dict:
    sales_count = session.query(func.count(SalesHistory.id)).scalar() or 0
    store_count = session.query(func.count(Store.store_id)).scalar() or 0
    product_count = session.query(func.count(Product.product_id)).scalar() or 0
    date_range = session.query(func.min(SalesHistory.date), func.max(SalesHistory.date)).one()
    latest_run = session.query(ForecastRun).order_by(ForecastRun.created_at.desc()).first()
    latest_recommendation = (
        session.query(ReplenishmentRecommendation)
        .order_by(ReplenishmentRecommendation.created_at.desc())
        .first()
    )
    return {
        "sales_rows": sales_count,
        "store_count": store_count,
        "product_count": product_count,
        "date_start": date_range[0].isoformat() if date_range[0] else None,
        "date_end": date_range[1].isoformat() if date_range[1] else None,
        "forecast_run_count": session.query(func.count(ForecastRun.forecast_run_id)).scalar() or 0,
        "recommendation_count": session.query(func.count(ReplenishmentRecommendation.id)).scalar() or 0,
        "latest_run": _run_payload(latest_run) if latest_run else None,
        "latest_recommendation": _recommendation_payload(latest_recommendation)
        if latest_recommendation
        else None,
    }


@app.get("/api/v1/stores")
def stores(session: Session = Depends(get_session)) -> list[dict]:
    return [
        {"store_id": row.store_id, "city": row.city, "state": row.state,
         "store_type": row.store_type, "cluster": row.cluster}
        for row in session.query(Store).order_by(Store.store_id).all()
    ]


@app.get("/api/v1/products")
def products(session: Session = Depends(get_session)) -> list[dict]:
    return [
        {"product_id": row.product_id, "family": row.family, "category": row.category,
         "perishable": row.perishable}
        for row in session.query(Product).order_by(Product.product_id).all()
    ]


@app.get("/api/v1/sales/{store_id}/{product_id}")
def sales_history(
    store_id: str,
    product_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = Query(default=730, ge=1, le=5000),
    session: Session = Depends(get_session),
) -> list[dict]:
    query = session.query(SalesHistory).filter(
        SalesHistory.store_id == store_id, SalesHistory.product_id == product_id
    )
    if start_date:
        query = query.filter(SalesHistory.date >= start_date)
    if end_date:
        query = query.filter(SalesHistory.date <= end_date)
    rows = query.order_by(SalesHistory.date.desc()).limit(limit).all()
    rows.reverse()
    return [{"date": row.date.isoformat(), "units_sold": row.units_sold,
             "onpromotion": row.onpromotion, "is_holiday": row.is_holiday} for row in rows]


@app.get("/api/v1/forecast-runs")
def forecast_runs(
    store_id: str | None = None,
    product_id: str | None = None,
    limit: int = Query(default=25, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[dict]:
    query = session.query(ForecastRun)
    if store_id:
        query = query.filter(ForecastRun.store_id == store_id)
    if product_id:
        query = query.filter(ForecastRun.product_id == product_id)
    return [_run_payload(row) for row in query.order_by(ForecastRun.created_at.desc()).limit(limit)]


@app.get("/api/v1/forecast-runs/{run_id}")
def forecast_run(run_id: int, session: Session = Depends(get_session)) -> dict:
    run = session.get(ForecastRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    payload = _run_payload(run)
    payload["forecasts"] = [
        {"date": row.forecast_date.isoformat(), "point": row.point_forecast,
         "lower_80": row.lower_80, "upper_80": row.upper_80,
         "lower_95": row.lower_95, "upper_95": row.upper_95}
        for row in session.query(Forecast)
        .filter(Forecast.forecast_run_id == run_id, Forecast.is_future.is_(True))
        .order_by(Forecast.forecast_date)
    ]
    return payload


@app.get("/api/v1/inventory/recommendations")
def recommendations(
    limit: int = Query(default=25, ge=1, le=100), session: Session = Depends(get_session)
) -> list[dict]:
    rows = (session.query(ReplenishmentRecommendation)
            .order_by(ReplenishmentRecommendation.created_at.desc()).limit(limit).all())
    return [_recommendation_payload(row) for row in rows]


def _run_payload(run: ForecastRun) -> dict:
    return {
        "run_id": run.forecast_run_id,
        "model_name": run.model_name,
        "training_cutoff": run.training_cutoff.isoformat() if run.training_cutoff else None,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "horizon_days": run.horizon_days,
        "store_id": run.store_id,
        "product_id": run.product_id,
        "metrics": {"mae": run.eval_mae, "rmse": run.eval_rmse, "wape": run.eval_wape,
                    "mase": run.eval_mase, "coverage_80": run.eval_coverage_80,
                    "coverage_95": run.eval_coverage_95},
    }


def _recommendation_payload(row: ReplenishmentRecommendation) -> dict:
    return {
        "store_id": row.store_id,
        "product_id": row.product_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "inventory_position": row.inventory_position,
        "lead_time_demand": row.lead_time_demand_estimate,
        "reorder_point": row.reorder_point,
        "suggested_order_qty": row.suggested_order_qty,
        "expected_shortage_risk": row.expected_shortage_risk,
        "explanation": row.explanation,
    }
