import sys; sys.path.insert(0, 'src')
import warnings; warnings.filterwarnings('ignore')
import logging; logging.basicConfig(level=logging.WARNING)

from demandiq.services.forecast_service import run_forecast_pipeline

print("Testing naive + seasonal_naive on S001/GROCERY_001...")
results = run_forecast_pipeline(
    store_id='S001', product_id='GROCERY_001',
    model_names=['naive', 'seasonal_naive'],
    horizon=14, n_backtest_splits=2,
)
for model, res in results.items():
    m = res.get('metrics', {})
    mae = m.get('MAE', 'err')
    wape = m.get('WAPE', 'err')
    n = len(res['future_forecast_df'])
    print(f"  {model}: MAE={mae:.2f}, WAPE={wape:.3f}, forecast_rows={n}")

print("\nTesting ETS...")
results_ets = run_forecast_pipeline(
    store_id='S001', product_id='GROCERY_001',
    model_names=['ets'], horizon=14, n_backtest_splits=2,
)
for model, res in results_ets.items():
    m = res.get('metrics', {})
    mae = m.get('MAE', 'err')
    print(f"  {model}: MAE={mae}")

print("\nTesting LGBM...")
results_lgbm = run_forecast_pipeline(
    store_id='S001', product_id='GROCERY_001',
    model_names=['lgbm'], horizon=14, n_backtest_splits=2,
)
for model, res in results_lgbm.items():
    m = res.get('metrics', {})
    mae = m.get('MAE', 'err')
    print(f"  {model}: MAE={mae}")

print("\nAll pipeline tests passed.")
