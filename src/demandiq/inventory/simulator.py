"""
Discrete-time inventory simulator for DemandIQ.

Simulates daily inventory operations to compare replenishment policies
on historical demand data.

Event sequence per day (strictly enforced):
1. Receive any orders that have arrived (lead time elapsed)
2. Serve demand from available inventory
3. Record stockouts / lost sales
4. Make replenishment decision based on current inventory position
   and FORECAST (not actual future demand)
5. Record costs for the day

CRITICAL: Replenishment decisions use only information available
at the simulated decision time. Future actual demand is NEVER accessed
when making ordering decisions.

This simulator uses the LOST SALES model (unmet demand is lost, not backordered).
This is explicitly documented and is one of two common choices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Full daily trace and aggregate statistics from a simulation run."""

    policy_name: str
    store_id: str
    product_id: str
    start_date: object
    end_date: object

    # Daily trace
    dates: list = field(default_factory=list)
    inventory_eod: list[float] = field(default_factory=list)  # end-of-day inventory
    demand_actual: list[float] = field(default_factory=list)
    demand_fulfilled: list[float] = field(default_factory=list)
    lost_sales: list[float] = field(default_factory=list)
    orders_placed: list[float] = field(default_factory=list)
    orders_received: list[float] = field(default_factory=list)
    holding_cost_daily: list[float] = field(default_factory=list)
    ordering_cost_daily: list[float] = field(default_factory=list)
    shortage_cost_daily: list[float] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(self.holding_cost_daily) + sum(self.ordering_cost_daily) + sum(self.shortage_cost_daily)

    @property
    def total_holding_cost(self) -> float:
        return sum(self.holding_cost_daily)

    @property
    def total_ordering_cost(self) -> float:
        return sum(self.ordering_cost_daily)

    @property
    def total_shortage_cost(self) -> float:
        return sum(self.shortage_cost_daily)

    @property
    def service_level_achieved(self) -> float:
        total_demand = sum(self.demand_actual)
        if total_demand == 0:
            return 1.0
        return sum(self.demand_fulfilled) / total_demand

    @property
    def stockout_rate(self) -> float:
        n_days = len(self.demand_actual)
        if n_days == 0:
            return 0.0
        stockout_days = sum(1 for ls in self.lost_sales if ls > 0)
        return stockout_days / n_days

    @property
    def avg_inventory(self) -> float:
        return float(np.mean(self.inventory_eod)) if self.inventory_eod else 0.0

    @property
    def n_orders(self) -> int:
        return sum(1 for o in self.orders_placed if o > 0)

    @property
    def avg_order_size(self) -> float:
        orders = [o for o in self.orders_placed if o > 0]
        return float(np.mean(orders)) if orders else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "date": self.dates,
            "inventory_eod": self.inventory_eod,
            "demand_actual": self.demand_actual,
            "demand_fulfilled": self.demand_fulfilled,
            "lost_sales": self.lost_sales,
            "orders_placed": self.orders_placed,
            "orders_received": self.orders_received,
            "holding_cost": self.holding_cost_daily,
            "ordering_cost": self.ordering_cost_daily,
            "shortage_cost": self.shortage_cost_daily,
        })

    def summary(self) -> dict:
        return {
            "policy": self.policy_name,
            "total_cost": round(self.total_cost, 2),
            "holding_cost": round(self.total_holding_cost, 2),
            "ordering_cost": round(self.total_ordering_cost, 2),
            "shortage_cost": round(self.total_shortage_cost, 2),
            "service_level": round(self.service_level_achieved, 4),
            "stockout_rate": round(self.stockout_rate, 4),
            "avg_inventory": round(self.avg_inventory, 2),
            "n_orders": self.n_orders,
            "avg_order_size": round(self.avg_order_size, 2),
            "n_days": len(self.dates),
        }


def run_simulation(
    demand_series: pd.Series,
    forecast_fn,
    policy,
    starting_inventory: float,
    lead_time_days: int,
    holding_cost_per_unit_day: float,
    ordering_cost: float,
    shortage_cost_per_unit: float,
    policy_name: str = "policy",
    store_id: str = "unknown",
    product_id: str = "unknown",
    min_order_qty: int = 1,
    forecast_horizon: int = 28,
) -> SimulationResult:
    """
    Run a discrete-time inventory simulation.

    Parameters
    ----------
    demand_series : Actual historical demand (sorted by date). Used for demand realization only.
    forecast_fn : Callable(history: pd.Series, horizon: int) -> np.ndarray.
                  Called at each decision point. Must only use data up to that point.
    policy : Policy object with a .recommend(forecast, ...) method.
    starting_inventory : Initial on-hand inventory.
    lead_time_days : Days until an order arrives after placement.
    holding_cost_per_unit_day : Cost to hold 1 unit for 1 day.
    ordering_cost : Fixed cost per order placed.
    shortage_cost_per_unit : Cost per unit of lost sales.
    policy_name : Label for this policy in results.
    store_id, product_id : Identifiers.
    min_order_qty : Minimum order quantity.
    forecast_horizon : How many days ahead to forecast at each decision point.

    Returns
    -------
    SimulationResult with full daily trace and aggregate statistics.
    """
    n_days = len(demand_series)
    dates = demand_series.index if hasattr(demand_series.index, '__iter__') else list(range(n_days))

    result = SimulationResult(
        policy_name=policy_name,
        store_id=store_id,
        product_id=product_id,
        start_date=dates[0] if len(dates) > 0 else None,
        end_date=dates[-1] if len(dates) > 0 else None,
    )

    inventory = float(starting_inventory)
    on_order: float = 0.0
    # Pipeline of incoming orders: list of (arrival_day, quantity)
    order_pipeline: list[tuple[int, float]] = []

    for t in range(n_days):
        # --- Step 1: Receive orders that have arrived ---
        received = 0.0
        new_pipeline = []
        for arrival_day, qty in order_pipeline:
            if arrival_day <= t:
                received += qty
            else:
                new_pipeline.append((arrival_day, qty))
        order_pipeline = new_pipeline
        inventory += received
        on_order -= received

        # --- Step 2: Serve demand ---
        actual_demand = float(demand_series.iloc[t])
        fulfilled = min(inventory, actual_demand)
        lost = actual_demand - fulfilled
        inventory -= fulfilled

        # --- Step 3: Compute daily costs ---
        h_cost = inventory * holding_cost_per_unit_day
        s_cost = lost * shortage_cost_per_unit

        # --- Step 4: Replenishment decision ---
        # CRITICAL: forecast uses only history up to and including day t
        history = demand_series.iloc[: t + 1]
        order_qty = 0.0
        o_cost = 0.0

        try:
            # Get forecast for lead_time_days ahead
            forecast = forecast_fn(history, forecast_horizon)

            # Use policy to decide order quantity
            # Compute inventory position
            inv_pos = inventory + on_order  # lost sales model: no backorders

            # Simple ROP check: compare inv_pos to reorder point
            rec = policy.recommend(forecast)
            rop = rec.get("reorder_point", rec.get("order_up_to_level", 0))

            if inv_pos <= rop:
                raw_qty = rec.get("suggested_order_qty", 0)
                order_qty = max(raw_qty, float(min_order_qty)) if raw_qty > 0 else 0.0
                if order_qty > 0:
                    order_pipeline.append((t + lead_time_days, order_qty))
                    on_order += order_qty
                    o_cost = ordering_cost

        except Exception as exc:
            logger.debug("Simulation day %d: forecast/policy failed: %s", t, exc)

        # --- Record daily state ---
        result.dates.append(dates[t])
        result.inventory_eod.append(round(inventory, 4))
        result.demand_actual.append(actual_demand)
        result.demand_fulfilled.append(fulfilled)
        result.lost_sales.append(lost)
        result.orders_placed.append(order_qty)
        result.orders_received.append(received)
        result.holding_cost_daily.append(h_cost)
        result.ordering_cost_daily.append(o_cost)
        result.shortage_cost_daily.append(s_cost)

    return result
