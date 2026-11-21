"""
Benchmark every forecasting model on every demo series.

Runs the same rolling-origin backtest the app uses, in parallel, and writes:

    data/demo/benchmark.csv             one row per series x model
    data/demo/benchmark_by_horizon.csv  WAPE per model and days ahead
    data/demo/benchmark_meta.json       settings used for the run

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --workers 8 --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from demandiq.config import settings  # noqa: E402


def _run_one(args: tuple[str, str, pd.DataFrame, int, int]) -> tuple[list[dict], list[dict]]:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    from demandiq.services.forecast_service import AVAILABLE_MODELS, forecast_series

    store_id, product_id, df, horizon, n_splits = args
    results = forecast_series(df, AVAILABLE_MODELS, horizon, n_splits)
    rows, steps = [], []
    for name, res in results.items():
        m = res.metrics
        rows.append({
            "store_id": store_id,
            "product_id": product_id,
            "model": name,
            "n_folds": res.backtest.n_folds,
            **{k: m.get(k, np.nan) for k in (
                "MAE", "RMSE", "WAPE", "MASE", "MFE",
                "Coverage_80", "Coverage_95", "Width_80", "Width_95",
            )},
        })
        actual = np.vstack([f.actual for f in res.backtest.folds])
        predicted = np.vstack([f.predicted for f in res.backtest.folds])
        for h in range(actual.shape[1]):
            steps.append({
                "model": name,
                "horizon_step": h + 1,
                "abs_error": float(np.abs(actual[:, h] - predicted[:, h]).sum()),
                "actual": float(actual[:, h].sum()),
            })
    return rows, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N series")
    parser.add_argument("--horizon", type=int, default=settings.DEFAULT_HORIZON)
    parser.add_argument("--splits", type=int, default=settings.BACKTEST_SPLITS)
    args = parser.parse_args()

    data_dir = settings.DEMO_DATA_DIR
    sales = pd.read_csv(data_dir / "sales.csv")
    products = pd.read_csv(data_dir / "products.csv")
    groups = list(sales.groupby(["store_id", "product_id"]))
    if args.limit:
        groups = groups[: args.limit]

    jobs = [(s, p, g, args.horizon, args.splits) for (s, p), g in groups]
    print(f"Benchmarking {len(jobs)} series with {args.workers} workers...")
    started = time.time()
    all_rows, all_steps = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            rows, steps = fut.result()
            all_rows.extend(rows)
            all_steps.extend(steps)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} series done ({time.time() - started:.0f}s)")

    bench = pd.DataFrame(all_rows).merge(products[["product_id", "family"]], on="product_id", how="left")
    bench = bench.sort_values(["store_id", "product_id", "model"])
    bench.to_csv(data_dir / "benchmark.csv", index=False, float_format="%.5f")

    steps = pd.DataFrame(all_steps).groupby(["model", "horizon_step"], as_index=False).sum()
    steps["WAPE"] = steps["abs_error"] / steps["actual"]
    steps[["model", "horizon_step", "WAPE"]].to_csv(
        data_dir / "benchmark_by_horizon.csv", index=False, float_format="%.5f"
    )

    meta = {
        "series": len(jobs),
        "horizon_days": args.horizon,
        "backtest_folds": args.splits,
        "fold_step_days": 7,
        "interval_method": "horizon-aware conformal on rolling-origin residuals",
        "data": "synthetic demo data (see data/demo/metadata.json)",
        "runtime_seconds": round(time.time() - started, 1),
    }
    (data_dir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2))

    summary = bench.groupby("model")[["WAPE", "MASE", "Coverage_80", "Coverage_95"]].mean()
    print(summary.sort_values("WAPE").round(3).to_string())


if __name__ == "__main__":
    main()
