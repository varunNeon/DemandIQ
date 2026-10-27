"""
Main pipeline runner for DemandIQ.

Executes the full pipeline:
  generate/load data → ingest → train models → generate forecasts → recommendations

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --store S001 --product GROCERY_001 --horizon 14
    python scripts/run_pipeline.py --generate-demo  # regenerate demo data then run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import shutil

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from demandiq.config import settings
from demandiq.data.ingestion import ingest_directory
from demandiq.database.connection import init_db
from demandiq.database.repositories import (
    get_product_ids,
    get_sales_date_range,
    get_store_ids,
)
from demandiq.database.connection import get_db

console = Console()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demandiq.pipeline")


def cmd_generate_demo(args) -> None:
    """Generate fresh demo data."""
    from scripts.generate_demo_data import run_generator

    console.print(Panel("[bold cyan]Generating Demo Data[/]"))
    run_generator(
        output_dir=settings.DEMO_DATA_DIR,
        n_stores=args.stores,
        n_products=args.products,
        n_days=args.days,
        seed=settings.RANDOM_SEED,
    )
    console.print(f"[green]OK[/] Demo data written to {settings.DEMO_DATA_DIR}")


def cmd_ingest(args) -> None:
    """Ingest data from CSV files."""
    data_dir = settings.DEMO_DATA_DIR if settings.DATA_MODE == "demo" else settings.RAW_DATA_DIR
    console.print(Panel(f"[bold cyan]Ingesting Data from {data_dir}[/]"))

    summary = ingest_directory(data_dir, data_source=settings.DATA_MODE)

    table = Table(title="Ingestion Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Stores inserted", str(summary["stores_inserted"]))
    table.add_row("Products inserted", str(summary["products_inserted"]))
    table.add_row("Sales rows inserted", str(summary["sales_inserted"]))
    table.add_row("Sales rows skipped (already exist)", str(summary["sales_skipped"]))
    table.add_row("DQ issues", str(len(summary["validation_issues"])))
    console.print(table)

    if summary["validation_issues"]:
        for issue in summary["validation_issues"]:
            severity = issue.get("severity", "info")
            color = {"error": "red", "warning": "yellow", "info": "blue"}.get(severity, "white")
            console.print(f"  [{color}][{severity.upper()}][/] {issue['issue_type']}: {issue.get('details', '')}")


def cmd_train(args) -> None:
    """Train forecast models."""
    from demandiq.services.forecast_service import run_forecast_pipeline

    console.print(Panel("[bold cyan]Training Forecast Models[/]"))

    with get_db() as session:
        store_ids = get_store_ids(session)
        product_ids = get_product_ids(session)

    if not store_ids or not product_ids:
        console.print("[red]No data found. Run ingestion first.[/]")
        return

    # Limit to sample for quick runs
    if args.store:
        store_ids = [s for s in store_ids if s == args.store]
    if args.product:
        product_ids = [p for p in product_ids if p == args.product]

    if args.sample:
        store_ids = store_ids[: args.sample_stores]
        product_ids = product_ids[: args.sample_products]

    total = len(store_ids) * len(product_ids)
    console.print(f"Training on {len(store_ids)} stores × {len(product_ids)} products = {total} series")

    trained = 0
    errors = 0
    for store_id in store_ids:
        for product_id in product_ids:
            try:
                results = run_forecast_pipeline(
                    store_id=store_id,
                    product_id=product_id,
                    model_names=args.models.split(",") if args.models else None,
                    horizon=args.horizon,
                    n_backtest_splits=args.splits,
                )
                if results:
                    for model, res in results.items():
                        if "error" in res:
                            errors += 1
                            console.print(f"  [red]ERROR[/] {store_id}/{product_id} [{model}] {res['error']}")
                            continue
                        trained += 1
                        if "metrics" in res:
                            m = res["metrics"]
                            console.print(
                                f"  [green]OK[/] {store_id}/{product_id} [{model}] "
                                f"MAE={m.get('MAE', 'N/A'):.2f} WAPE={m.get('WAPE', 'N/A'):.3f}"
                            )
            except Exception as exc:
                errors += 1
                console.print(f"  [red]ERROR[/] {store_id}/{product_id}: {exc}")

    console.print(f"\n[bold]Done:[/] {trained} series trained, {errors} errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DemandIQ Pipeline Runner")
    parser.add_argument("--generate-demo", action="store_true", help="Regenerate demo data before running")
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion")
    parser.add_argument("--train", action="store_true", help="Train forecast models")
    parser.add_argument("--all", action="store_true", help="Run full pipeline (demo → ingest → train)")

    parser.add_argument("--store", type=str, help="Filter to specific store_id")
    parser.add_argument("--product", type=str, help="Filter to specific product_id")
    parser.add_argument("--models", type=str, help="Comma-separated model names")
    parser.add_argument("--horizon", type=int, default=settings.DEFAULT_HORIZON)
    parser.add_argument("--splits", type=int, default=settings.BACKTEST_SPLITS)
    parser.add_argument("--sample", action="store_true", help="Train on small sample for testing")
    parser.add_argument("--sample-stores", type=int, default=2)
    parser.add_argument("--sample-products", type=int, default=3)

    # Demo data options
    parser.add_argument("--stores", type=int, default=5)
    parser.add_argument("--products", type=int, default=20)
    parser.add_argument("--days", type=int, default=730)

    args = parser.parse_args()

    settings.ensure_dirs()
    init_db()

    if args.all or args.generate_demo:
        cmd_generate_demo(args)

    if args.all or args.ingest:
        cmd_ingest(args)

    if args.all or args.train:
        cmd_train(args)

    if not any([args.all, args.generate_demo, args.ingest, args.train]):
        console.print("[yellow]No action specified. Use --all, --generate-demo, --ingest, or --train.[/]")
        parser.print_help()


if __name__ == "__main__":
    main()
