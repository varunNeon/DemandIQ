from __future__ import annotations

import pandas as pd

from demandiq.data.ingestion import ingest_directory
from demandiq.database.connection import get_db
from demandiq.database.models import (
    Forecast,
    ForecastRun,
    ReplenishmentRecommendation,
    SalesHistory,
)


def _write_demo(tmp_path, weekly_series_df: pd.DataFrame):
    data_dir = tmp_path / "demo"
    data_dir.mkdir()
    pd.DataFrame([{"store_id": "S001", "city": "Quito", "state": "P", "store_type": "A", "cluster": 1}]
                 ).to_csv(data_dir / "stores.csv", index=False)
    pd.DataFrame([{"product_id": "P001", "family": "TOYS", "category": "GAMES", "perishable": False}]
                 ).to_csv(data_dir / "products.csv", index=False)
    sales = weekly_series_df.assign(store_id="S001", product_id="P001")
    sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")
    sales.to_csv(data_dir / "sales.csv", index=False)
    return data_dir


def test_ingestion_is_idempotent(temp_db, tmp_path, weekly_series_df) -> None:
    data_dir = _write_demo(tmp_path, weekly_series_df)

    first = ingest_directory(data_dir)
    second = ingest_directory(data_dir)

    assert first["sales_inserted"] == len(weekly_series_df)
    assert second["sales_inserted"] == 0
    with get_db() as session:
        assert session.query(SalesHistory).count() == len(weekly_series_df)


def test_rerunning_the_pipeline_replaces_runs_and_creates_recommendations(
    temp_db, tmp_path, weekly_series_df
) -> None:
    from demandiq.services.forecast_service import run_forecast_pipeline
    from demandiq.services.inventory_service import generate_recommendations

    ingest_directory(_write_demo(tmp_path, weekly_series_df))
    for _ in range(2):
        out = run_forecast_pipeline("S001", "P001", ["naive", "seasonal_naive"], horizon=7, n_backtest_splits=5)
        assert all("run_id" in v for v in out.values())

    created = generate_recommendations(["S001"], ["P001"])

    with get_db() as session:
        assert session.query(ForecastRun).count() == 2  # one per model, not duplicated
        assert session.query(Forecast).count() == 14
        runs = session.query(ForecastRun).all()
        assert all(r.eval_coverage_80 is not None for r in runs)
        assert created == 1
        assert session.query(ReplenishmentRecommendation).count() == 1
