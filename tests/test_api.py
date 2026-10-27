from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from demandiq.api.main import app, get_session
from demandiq.database.models import Base, Forecast, ForecastRun, Product, SalesHistory, Store


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = override_session
    with factory() as session:
        session.add(Store(store_id="S001", city="Oakland"))
        session.add(Product(product_id="P001", family="Produce", category="Fresh"))
        session.add(SalesHistory(date=date(2025, 1, 1), store_id="S001", product_id="P001", units_sold=12))
        run = ForecastRun(
            model_name="naive", training_cutoff=date(2025, 1, 1), horizon_days=1,
            store_id="S001", product_id="P001", eval_mae=2.0, eval_wape=0.15,
        )
        session.add(run)
        session.flush()
        session.add(Forecast(
            forecast_run_id=run.forecast_run_id, forecast_date=date(2025, 1, 2),
            store_id="S001", product_id="P001", point_forecast=10,
            lower_80=8, upper_80=12, is_future=True,
        ))
        session.commit()
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_overview_and_series_endpoints(client: TestClient) -> None:
    response = client.get("/api/v1/overview")
    assert response.status_code == 200
    assert response.json()["sales_rows"] == 1
    assert response.json()["store_count"] == 1
    assert response.json()["product_count"] == 1

    history = client.get("/api/v1/sales/S001/P001")
    assert history.status_code == 200
    assert history.json() == [{"date": "2025-01-01", "units_sold": 12.0, "onpromotion": False, "is_holiday": False}]


def test_forecast_run_detail_returns_saved_forecast(client: TestClient) -> None:
    response = client.get("/api/v1/forecast-runs/1")
    assert response.status_code == 200
    assert response.json()["model_name"] == "naive"
    assert response.json()["forecasts"][0]["point"] == 10.0


def test_missing_forecast_run_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/forecast-runs/999")
    assert response.status_code == 404
