import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import logging; logging.basicConfig(level=logging.WARNING)

# Test 1: demo data generation (numpy mutation bug)
from scripts.generate_demo_data import run_generator
from pathlib import Path
run_generator(Path('data/demo'), n_stores=2, n_products=3, n_days=100, seed=42)
print("generate_demo_data: OK")

# Test 2: conformal calibration speed (O(n^2) bug)
import time
import numpy as np
import pandas as pd
from demandiq.forecasting.baselines import NaiveForecaster
from demandiq.forecasting.intervals import ConformalIntervalWrapper

series = pd.Series(np.random.default_rng(42).normal(50, 10, 730))
t0 = time.time()
wrapper = ConformalIntervalWrapper(NaiveForecaster(), calibration_frac=0.2)
wrapper.fit(series)
elapsed = time.time() - t0
print(f"Conformal calibration (730 pts): {elapsed:.3f}s  (should be <1s)")

result = wrapper.predict_with_intervals(14)
print(f"Interval keys: {list(result.keys())}")
assert "lower_80" in result
assert "upper_80" in result
assert (result["lower_80"] <= result["point"]).all()
assert (result["point"] <= result["upper_80"]).all()
print("Interval ordering: OK")

# Test 3: SeasonalNaive dead code removed
from demandiq.forecasting.baselines import SeasonalNaiveForecaster
sn = SeasonalNaiveForecaster(period=7)
data = pd.Series([float(i % 7) + 1 for i in range(21)])
sn.fit(data)
preds = sn.predict(7)
assert len(preds) == 7
print(f"SeasonalNaive preds: {preds}")
print("All fixes verified OK")
