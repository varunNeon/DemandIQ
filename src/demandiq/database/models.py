"""
SQLAlchemy ORM models for DemandIQ.

All tables use explicit primary keys, foreign keys, and indexes.
Schema is designed to be backend-agnostic (SQLite and PostgreSQL compatible).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


class Store(Base):
    """Retail store metadata."""

    __tablename__ = "stores"

    store_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    city: Mapped[str | None] = mapped_column(String(100))
    state: Mapped[str | None] = mapped_column(String(100))
    store_type: Mapped[str | None] = mapped_column(String(50))
    cluster: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    sales: Mapped[list["SalesHistory"]] = relationship(back_populates="store")
    inventory_params: Mapped[list["InventoryParameters"]] = relationship(back_populates="store")


class Product(Base):
    """Product / SKU metadata."""

    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    family: Mapped[str] = mapped_column(String(100))
    category: Mapped[str | None] = mapped_column(String(100))
    perishable: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    sales: Mapped[list["SalesHistory"]] = relationship(back_populates="product")
    inventory_params: Mapped[list["InventoryParameters"]] = relationship(back_populates="product")


class SalesHistory(Base):
    """Daily unit sales by store and product (grain: date × store × product)."""

    __tablename__ = "sales_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    store_id: Mapped[str] = mapped_column(String(50), ForeignKey("stores.store_id"), nullable=False)
    product_id: Mapped[str] = mapped_column(String(100), ForeignKey("products.product_id"), nullable=False)
    units_sold: Mapped[float] = mapped_column(Float, nullable=False)
    onpromotion: Mapped[bool] = mapped_column(Boolean, default=False)
    is_holiday: Mapped[bool] = mapped_column(Boolean, default=False)
    data_source: Mapped[str] = mapped_column(String(50), default="demo")  # "demo" | "kaggle"
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    store: Mapped["Store"] = relationship(back_populates="sales")
    product: Mapped["Product"] = relationship(back_populates="sales")

    # Constraints & indexes
    __table_args__ = (
        UniqueConstraint("date", "store_id", "product_id", name="uq_sales_date_store_product"),
        Index("ix_sales_date", "date"),
        Index("ix_sales_store_product", "store_id", "product_id"),
    )


class ForecastRun(Base):
    """Metadata for a model training / forecast generation run."""

    __tablename__ = "forecast_runs"

    forecast_run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    training_cutoff: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    store_id: Mapped[str | None] = mapped_column(String(50))
    product_id: Mapped[str | None] = mapped_column(String(100))
    model_config: Mapped[str | None] = mapped_column(Text)  # JSON string
    eval_mae: Mapped[float | None] = mapped_column(Float)
    eval_rmse: Mapped[float | None] = mapped_column(Float)
    eval_wape: Mapped[float | None] = mapped_column(Float)
    eval_mase: Mapped[float | None] = mapped_column(Float)
    eval_coverage_80: Mapped[float | None] = mapped_column(Float)
    eval_coverage_95: Mapped[float | None] = mapped_column(Float)

    # Relationships
    forecasts: Mapped[list["Forecast"]] = relationship(back_populates="run")

    __table_args__ = (
        Index("ix_forecast_runs_model_cutoff", "model_name", "training_cutoff"),
    )


class Forecast(Base):
    """Individual point and interval forecasts linked to a ForecastRun."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("forecast_runs.forecast_run_id"), nullable=False)
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False)
    store_id: Mapped[str] = mapped_column(String(50), nullable=False)
    product_id: Mapped[str] = mapped_column(String(100), nullable=False)
    point_forecast: Mapped[float] = mapped_column(Float, nullable=False)
    lower_80: Mapped[float | None] = mapped_column(Float)
    upper_80: Mapped[float | None] = mapped_column(Float)
    lower_95: Mapped[float | None] = mapped_column(Float)
    upper_95: Mapped[float | None] = mapped_column(Float)
    is_future: Mapped[bool] = mapped_column(Boolean, default=True)  # False = backtest period

    # Relationships
    run: Mapped["ForecastRun"] = relationship(back_populates="forecasts")

    __table_args__ = (
        Index("ix_forecasts_run_date", "forecast_run_id", "forecast_date"),
        Index("ix_forecasts_store_product", "store_id", "product_id"),
    )


class InventoryParameters(Base):
    """
    Operational inventory parameters per store-product combination.

    These are ASSUMPTIONS, not facts inferred from data. Defaults are
    clearly labeled and editable by the user through the dashboard.
    """

    __tablename__ = "inventory_parameters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(50), ForeignKey("stores.store_id"), nullable=False)
    product_id: Mapped[str] = mapped_column(String(100), ForeignKey("products.product_id"), nullable=False)
    lead_time_days: Mapped[int] = mapped_column(Integer, default=7)
    unit_cost: Mapped[float] = mapped_column(Float, default=10.0)
    ordering_cost: Mapped[float] = mapped_column(Float, default=50.0)
    holding_cost_per_unit_day: Mapped[float] = mapped_column(Float, default=0.05)
    shortage_cost_per_unit: Mapped[float] = mapped_column(Float, default=5.0)
    service_level: Mapped[float] = mapped_column(Float, default=0.90)
    min_order_quantity: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    store: Mapped["Store"] = relationship(back_populates="inventory_params")
    product: Mapped["Product"] = relationship(back_populates="inventory_params")

    __table_args__ = (
        UniqueConstraint("store_id", "product_id", name="uq_inv_params_store_product"),
    )


class ReplenishmentRecommendation(Base):
    """Replenishment recommendation linked to a forecast run."""

    __tablename__ = "replenishment_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(50), nullable=False)
    product_id: Mapped[str] = mapped_column(String(100), nullable=False)
    forecast_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("forecast_runs.forecast_run_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    inventory_position: Mapped[float] = mapped_column(Float)
    lead_time_demand_estimate: Mapped[float] = mapped_column(Float)
    reorder_point: Mapped[float] = mapped_column(Float)
    suggested_order_qty: Mapped[float] = mapped_column(Float)
    expected_shortage_risk: Mapped[float | None] = mapped_column(Float)
    estimated_holding_cost: Mapped[float | None] = mapped_column(Float)
    estimated_ordering_cost: Mapped[float | None] = mapped_column(Float)
    explanation: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_reco_store_product", "store_id", "product_id"),
    )


class DataQualityEvent(Base):
    """Records data quality issues found during ingestion or validation."""

    __tablename__ = "data_quality_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    dataset: Mapped[str] = mapped_column(String(100))  # e.g. "sales_history"
    issue_type: Mapped[str] = mapped_column(String(100))  # e.g. "duplicate_record"
    severity: Mapped[str] = mapped_column(String(20))  # "info" | "warning" | "error"
    row_count: Mapped[int | None] = mapped_column(Integer)
    details: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_dq_events_dataset_type", "dataset", "issue_type"),
    )
