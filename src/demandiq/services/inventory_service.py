"""
Inventory service for DemandIQ.

Turns probabilistic forecasts into replenishment decisions and stress-tests
policies on historical demand.

Inventory inputs (lead time, costs, service level, stock on hand) are
OPERATING ASSUMPTIONS supplied by the user. They are not inferred from sales.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from demandiq.config import settings
from demandiq.database.connection import get_db
from demandiq.database.models import ForecastRun, ReplenishmentRecommendation
from demandiq.database.repositories import get_forecasts_dataframe, get_sales_dataframe
from demandiq.inventory.policies import InventoryPolicy, ReorderPointPolicy, std_from_interval
from demandiq.inventory.simulator import SimulationResult, run_simulation

logger = logging.getLogger(__name__)


@dataclass
class InventoryInputs:
    """User-supplied operating assumptions."""

    lead_time_days: int = settings.DEFAULT_LEAD_TIME_DAYS
    service_level: float = settings.DEFAULT_SERVICE_LEVEL
    on_hand: float = 0.0
    on_order: float = 0.0
    holding_cost_per_unit_day: float = settings.DEFAULT_HOLDING_COST_PER_UNIT_DAY
    ordering_cost: float = settings.DEFAULT_ORDERING_COST
    shortage_cost_per_unit: float = settings.DEFAULT_SHORTAGE_COST_PER_UNIT
    unit_cost: float = settings.DEFAULT_UNIT_COST
    min_order_qty: int = 1


def _policy(inputs: InventoryInputs) -> ReorderPointPolicy:
    return ReorderPointPolicy(InventoryPolicy(
        lead_time_days=inputs.lead_time_days,
        service_level=inputs.service_level,
        on_hand=inputs.on_hand,
        on_order=inputs.on_order,
        min_order_qty=inputs.min_order_qty,
    ))


def recommend_from_forecast(future: pd.DataFrame, inputs: InventoryInputs) -> dict:
    """
    Reorder-point recommendation from a probabilistic forecast.

    `future` needs `point_forecast` and, ideally, `lower_80`/`upper_80`.
    The 80% interval is converted to a per-day forecast-error standard
    deviation, so safety stock reflects measured forecast uncertainty rather
    than a guessed coefficient of variation.
    """
    point = future["point_forecast"].to_numpy(dtype=float)
    std = None
    if {"lower_80", "upper_80"} <= set(future.columns) and future["lower_80"].notna().all():
        std = std_from_interval(future["lower_80"].to_numpy(), future["upper_80"].to_numpy(), 0.80)

    rec = _policy(inputs).recommend(
        point,
        ordering_cost=inputs.ordering_cost,
        holding_cost_per_unit_day=inputs.holding_cost_per_unit_day,
        unit_cost=inputs.unit_cost,
        forecast_std=std,
    )
    rec["uncertainty_source"] = "80% prediction interval" if std is not None else "assumed 30% CV"
    rec["inputs"] = asdict(inputs)
    return rec


def project_stock(future: pd.DataFrame, inputs: InventoryInputs, rec: dict) -> pd.DataFrame:
    """
    Expected stock over the forecast horizon if the plan is followed.

    Each day: deliveries arrive, expected demand is served (stock cannot go
    below zero), then an order of the suggested size is placed whenever the
    inventory position (stock + on order) is at or below the reorder point.
    The existing `on_order` quantity is assumed to arrive after the lead time.
    Also returns the path if nothing is ordered, for comparison.
    """
    demand = future["point_forecast"].to_numpy(dtype=float)
    rop, qty, lead = rec["reorder_point"], max(rec["suggested_order_qty"], 1.0), inputs.lead_time_days
    stock, stock_none = inputs.on_hand, inputs.on_hand
    pipeline: list[tuple[int, float]] = [(lead, inputs.on_order)] if inputs.on_order > 0 else []
    rows = []
    for day, d in enumerate(demand, start=1):
        arriving = sum(q for t, q in pipeline if t == day)
        pipeline = [(t, q) for t, q in pipeline if t != day]
        stock += arriving
        if day == lead:
            stock_none += inputs.on_order
        stock = max(stock - d, 0.0)
        stock_none = max(stock_none - d, 0.0)
        placed = 0.0
        if stock + sum(q for _, q in pipeline) <= rop:
            placed = qty
            pipeline.append((day + lead, qty))
        rows.append({
            "date": future["date"].iloc[day - 1], "stock": stock, "stock_no_order": stock_none,
            "order_placed": placed, "delivery": arriving,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Policy simulation on historical demand
# --------------------------------------------------------------------------- #
def profile_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    """
    Fast, leakage-free planning forecast used inside the simulator:
    the last 28 days' mean, shaped by each weekday's share of demand over the
    last 8 weeks. Only `history` (data up to the decision day) is used.
    """
    values = history.to_numpy(dtype=float)
    if len(values) < 14:
        return np.full(horizon, float(np.mean(values)) if len(values) else 0.0)
    level = float(np.mean(values[-28:]))
    recent = values[-56:]
    n = len(recent)
    positions = np.arange(n) % 7  # position relative to the window start
    weekday_mean = np.array([recent[positions == d].mean() for d in range(7)])
    overall = weekday_mean.mean() or 1.0
    factors = weekday_mean / overall
    next_positions = (np.arange(1, horizon + 1) + n - 1) % 7
    return level * factors[next_positions]


class _IntervalAwarePolicy:
    """Adapts ReorderPointPolicy to the simulator with a fixed error std."""

    def __init__(self, inputs: InventoryInputs, daily_error_std: float) -> None:
        self.inputs = inputs
        self.daily_error_std = daily_error_std
        self.policy = _policy(inputs)

    def recommend(self, forecast: np.ndarray) -> dict:
        return self.policy.recommend(
            forecast,
            ordering_cost=self.inputs.ordering_cost,
            holding_cost_per_unit_day=self.inputs.holding_cost_per_unit_day,
            unit_cost=self.inputs.unit_cost,
            forecast_std=self.daily_error_std,
        )


def planning_error_std(demand: pd.Series, warmup: int = 56) -> float:
    """One-step error std of `profile_forecast`, measured on the warm-up history only."""
    values = demand.iloc[:warmup] if len(demand) > warmup else demand
    errors = []
    for t in range(28, len(values)):
        errors.append(values.iloc[t] - profile_forecast(values.iloc[:t], 1)[0])
    return float(np.std(errors, ddof=1)) if len(errors) > 1 else float(values.std() or 1.0)


def simulate_policy(
    demand: pd.Series,
    inputs: InventoryInputs,
    days: int = 182,
    starting_inventory: float | None = None,
) -> SimulationResult:
    """
    Replay the last `days` of actual demand under a reorder-point policy.

    Decisions use only demand observed up to each simulated day. Unmet
    demand is lost (lost-sales model).
    """
    demand = demand.astype(float)
    history_needed = 84
    window = demand.iloc[-(days + history_needed):]
    warmup, test = window.iloc[:history_needed], window.iloc[history_needed:]
    error_std = planning_error_std(warmup, warmup=history_needed)

    if starting_inventory is None:
        starting_inventory = float(warmup.iloc[-28:].mean() * (inputs.lead_time_days + 7))

    # The simulator forecasts from history; prepend warm-up so forecasts are informed.
    offset = len(warmup)

    def forecast_fn(history: pd.Series, horizon: int) -> np.ndarray:
        full = pd.concat([warmup, history])
        return profile_forecast(full, horizon)

    result = run_simulation(
        demand_series=test,
        forecast_fn=forecast_fn,
        policy=_IntervalAwarePolicy(inputs, error_std),
        starting_inventory=starting_inventory,
        lead_time_days=inputs.lead_time_days,
        holding_cost_per_unit_day=inputs.holding_cost_per_unit_day,
        ordering_cost=inputs.ordering_cost,
        shortage_cost_per_unit=inputs.shortage_cost_per_unit,
        policy_name=f"ROP @ {inputs.service_level:.0%}",
        min_order_qty=inputs.min_order_qty,
        forecast_horizon=max(inputs.lead_time_days, 14),
    )
    result.extra = {"offset": offset, "error_std": error_std}  # type: ignore[attr-defined]
    return result


def service_level_tradeoff(
    demand: pd.Series,
    inputs: InventoryInputs,
    levels: tuple[float, ...] = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99),
    days: int = 182,
) -> pd.DataFrame:
    """Simulate each target service level and report achieved fill rate and costs."""
    rows = []
    for level in levels:
        trial = InventoryInputs(**{**asdict(inputs), "service_level": level})
        sim = simulate_policy(demand, trial, days=days)
        s = sim.summary()
        rows.append({
            "target_service_level": level,
            "fill_rate": s["service_level"],
            "stockout_days": int(round(s["stockout_rate"] * s["n_days"])),
            "holding_cost": s["holding_cost"],
            "ordering_cost": s["ordering_cost"],
            "shortage_cost": s["shortage_cost"],
            "total_cost": s["total_cost"],
            "avg_inventory": s["avg_inventory"],
            "orders": s["n_orders"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Persistence for the pipeline / API
# --------------------------------------------------------------------------- #
def save_recommendation(store_id: str, product_id: str, forecast_run_id: int, rec: dict) -> int:
    """Store a recommendation, replacing any earlier one for the same run."""
    inputs = rec.get("inputs", {})
    with get_db() as session:
        session.query(ReplenishmentRecommendation).filter(
            ReplenishmentRecommendation.forecast_run_id == forecast_run_id
        ).delete()
        row = ReplenishmentRecommendation(
            store_id=store_id,
            product_id=product_id,
            forecast_run_id=forecast_run_id,
            inventory_position=rec["inventory_position"],
            lead_time_demand_estimate=rec["mu_LT"],
            reorder_point=rec["reorder_point"],
            suggested_order_qty=rec["suggested_order_qty"] if rec["should_order"] else 0.0,
            expected_shortage_risk=rec.get("stockout_risk"),
            estimated_holding_cost=rec["safety_stock"] * inputs.get("holding_cost_per_unit_day", 0.0),
            estimated_ordering_cost=inputs.get("ordering_cost") if rec["should_order"] else 0.0,
            explanation=rec["explanation"]
            + " Stock on hand is an assumed value (one week of recent demand) for the demo.",
        )
        session.add(row)
        session.flush()
        return row.id


def generate_recommendations(store_ids: list[str], product_ids: list[str]) -> int:
    """
    For each series, take the most accurate latest forecast run (lowest WAPE)
    and save a reorder-point recommendation using default operating inputs.
    Stock on hand is assumed to equal the last 7 days of demand.
    """
    created = 0
    for store_id in store_ids:
        for product_id in product_ids:
            with get_db() as session:
                runs = (
                    session.query(ForecastRun)
                    .filter(ForecastRun.store_id == store_id, ForecastRun.product_id == product_id)
                    .order_by(ForecastRun.training_cutoff.desc(), ForecastRun.created_at.desc())
                    .all()
                )
                if not runs:
                    continue
                latest_cutoff = runs[0].training_cutoff
                candidates = [r for r in runs if r.training_cutoff == latest_cutoff and r.eval_wape is not None]
                if not candidates:
                    continue
                best = min(candidates, key=lambda r: r.eval_wape)
                run_id = best.forecast_run_id
                future = get_forecasts_dataframe(session, run_id)
                sales = get_sales_dataframe(session, store_id=store_id, product_id=product_id)
            if future.empty or sales.empty:
                continue
            on_hand = float(sales.sort_values("date")["units_sold"].tail(7).sum())
            rec = recommend_from_forecast(future, InventoryInputs(on_hand=on_hand))
            save_recommendation(store_id, product_id, run_id, rec)
            created += 1
    return created
