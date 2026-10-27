"""
Inventory optimization policies for DemandIQ.

Implements two core replenishment policies:
1. Reorder Point (ROP) Policy — continuous review
2. Periodic Review Policy — review every R periods

Both policies use demand forecasts and forecast uncertainty
to set statistically justified reorder points and order quantities.

IMPORTANT: Inventory parameters (lead time, costs, service level) are
OPERATIONAL ASSUMPTIONS, not facts inferred from sales data alone.
All defaults should be clearly labeled and editable by the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class InventoryPolicy:
    """Base class for inventory policy parameters."""

    lead_time_days: int = 7
    service_level: float = 0.90  # target fill rate (e.g. 0.90 = 90%)
    on_hand: float = 0.0  # current on-hand inventory
    on_order: float = 0.0  # outstanding purchase orders
    backorders: float = 0.0  # unfulfilled demand
    min_order_qty: int = 1  # minimum order quantity


def inventory_position(on_hand: float, on_order: float, backorders: float) -> float:
    """
    Inventory position = on_hand + on_order - backorders.

    This is the correct measure of "how covered are we" because
    it includes incoming stock that hasn't arrived yet.
    """
    return on_hand + on_order - backorders


def lead_time_demand_stats(
    daily_demand_forecast: np.ndarray,
    lead_time_days: int,
    forecast_std: float | None = None,
) -> tuple[float, float]:
    """
    Estimate mean and standard deviation of demand during lead time.

    Parameters
    ----------
    daily_demand_forecast : Forecast for each day of the lead time.
    lead_time_days : Number of days until order arrives.
    forecast_std : Optional per-day forecast standard deviation.
                   If None, estimated from the coefficient of variation
                   of the forecast series.

    Returns
    -------
    (mu_LT, sigma_LT) : Mean and std of lead-time demand.
    """
    # Use the first lead_time_days of the forecast
    horizon = min(lead_time_days, len(daily_demand_forecast))
    lt_forecast = daily_demand_forecast[:horizon]

    mu_LT = float(np.sum(lt_forecast))

    if forecast_std is not None:
        # If per-day std provided, aggregate assuming independence
        sigma_LT = float(np.sqrt(lead_time_days * forecast_std**2))
    else:
        # Estimate from forecast variability (conservative)
        if len(lt_forecast) > 1:
            daily_std = float(np.std(lt_forecast, ddof=1))
        else:
            # Fallback: assume 30% CV of mean daily demand
            daily_mean = mu_LT / max(horizon, 1)
            daily_std = 0.30 * daily_mean
        sigma_LT = float(np.sqrt(lead_time_days) * daily_std)

    return mu_LT, sigma_LT


class ReorderPointPolicy:
    """
    Continuous-review (s, Q) reorder point policy.

    When inventory position drops to or below the reorder point s,
    an order of quantity Q is placed.

    Reorder point formula:
        s = mu_LT + z * sigma_LT
    where z is the safety factor corresponding to the target service level.

    Order quantity is based on Economic Order Quantity (EOQ) concept,
    but can be overridden or adjusted for minimum order quantities.
    """

    name: str = "reorder_point"

    def __init__(self, params: InventoryPolicy) -> None:
        self.params = params

    def compute_reorder_point(
        self,
        mu_LT: float,
        sigma_LT: float,
    ) -> float:
        """
        Compute reorder point using the normal approximation.

        ROP = mu_LT + z_alpha * sigma_LT

        where z_alpha is the inverse normal CDF at the service level.
        This assumes lead-time demand is approximately normally distributed.

        For intermittent or highly skewed demand, this is an approximation
        and should be noted as such.
        """
        if sigma_LT <= 0:
            # No uncertainty — use mean only
            return mu_LT

        z = float(stats.norm.ppf(self.params.service_level))
        safety_stock = z * sigma_LT
        rop = mu_LT + safety_stock

        logger.debug(
            "ROP: mu_LT=%.2f, sigma_LT=%.2f, z=%.2f, safety_stock=%.2f, ROP=%.2f",
            mu_LT, sigma_LT, z, safety_stock, rop,
        )
        return max(rop, 0.0)

    def compute_eoq(
        self,
        daily_demand: float,
        ordering_cost: float,
        holding_cost_per_unit_day: float,
        unit_cost: float,
    ) -> float:
        """
        Economic Order Quantity.

        EOQ = sqrt(2 * D * K / h)
        where:
            D = annual demand (daily_demand * 365)
            K = ordering cost per order
            h = annual holding cost per unit (holding_cost_per_unit_day * 365)

        Note: unit_cost is included in holding cost calculation
        (holding cost = holding_cost_rate * unit_cost * annual_demand_per_unit).
        """
        annual_demand = daily_demand * 365
        annual_holding = (holding_cost_per_unit_day + holding_cost_per_unit_day * unit_cost) * 365

        if annual_holding <= 0 or annual_demand <= 0:
            logger.warning("EOQ: invalid inputs (D=%.2f, h=%.2f). Returning 1.", annual_demand, annual_holding)
            return 1.0

        eoq = np.sqrt(2 * annual_demand * ordering_cost / annual_holding)
        return max(eoq, float(self.params.min_order_qty))

    def recommend(
        self,
        daily_demand_forecast: np.ndarray,
        ordering_cost: float = 50.0,
        holding_cost_per_unit_day: float = 0.05,
        unit_cost: float = 10.0,
        forecast_std: float | None = None,
    ) -> dict:
        """
        Compute full replenishment recommendation.

        Returns a dictionary with:
            - inventory_position
            - mu_LT (mean lead-time demand)
            - sigma_LT (std of lead-time demand)
            - safety_stock
            - reorder_point
            - suggested_order_qty
            - should_order (bool)
            - explanation
        """
        inv_pos = inventory_position(
            self.params.on_hand,
            self.params.on_order,
            self.params.backorders,
        )

        mu_LT, sigma_LT = lead_time_demand_stats(
            daily_demand_forecast,
            self.params.lead_time_days,
            forecast_std,
        )

        rop = self.compute_reorder_point(mu_LT, sigma_LT)
        z = float(stats.norm.ppf(self.params.service_level))
        safety_stock = z * sigma_LT if sigma_LT > 0 else 0.0

        daily_mean = float(np.mean(daily_demand_forecast)) if len(daily_demand_forecast) > 0 else 0.0
        order_qty = self.compute_eoq(daily_mean, ordering_cost, holding_cost_per_unit_day, unit_cost)
        order_qty = max(order_qty, float(self.params.min_order_qty))

        should_order = inv_pos <= rop

        if should_order:
            explanation = (
                f"Order recommended: inventory position ({inv_pos:.1f}) ≤ reorder point ({rop:.1f}). "
                f"Suggested order: {order_qty:.0f} units. "
                f"Lead-time demand estimate: {mu_LT:.1f} ± {sigma_LT:.1f} units "
                f"(safety stock: {safety_stock:.1f} units for {self.params.service_level:.0%} service level)."
            )
        else:
            shortfall = inv_pos - rop
            explanation = (
                f"No order needed: inventory position ({inv_pos:.1f}) > reorder point ({rop:.1f}) "
                f"by {shortfall:.1f} units. "
                f"Monitor when position approaches {rop:.1f}."
            )

        return {
            "inventory_position": round(inv_pos, 2),
            "on_hand": self.params.on_hand,
            "on_order": self.params.on_order,
            "backorders": self.params.backorders,
            "lead_time_days": self.params.lead_time_days,
            "mu_LT": round(mu_LT, 2),
            "sigma_LT": round(sigma_LT, 2),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(rop, 2),
            "suggested_order_qty": round(order_qty, 0),
            "should_order": should_order,
            "service_level": self.params.service_level,
            "explanation": explanation,
        }


class PeriodicReviewPolicy:
    """
    Periodic review (R, S) policy.

    Orders are placed every R days to raise inventory up to a target level S.
    Target level S covers demand during lead time + review period.

    Order quantity = max(S - inventory_position, min_order_qty)
    """

    name: str = "periodic_review"

    def __init__(self, params: InventoryPolicy, review_period_days: int = 7) -> None:
        self.params = params
        self.review_period_days = review_period_days

    def compute_order_up_to_level(
        self,
        daily_demand_forecast: np.ndarray,
        forecast_std: float | None = None,
    ) -> float:
        """
        Compute order-up-to level S.

        Protection period = lead_time + review_period.
        S = mu_protection + z * sigma_protection.
        """
        protection_days = self.params.lead_time_days + self.review_period_days
        protection_forecast = daily_demand_forecast[:protection_days]

        mu_P = float(np.sum(protection_forecast))

        if forecast_std is not None:
            sigma_P = float(np.sqrt(protection_days * forecast_std**2))
        else:
            daily_std = float(np.std(protection_forecast, ddof=1)) if len(protection_forecast) > 1 else 0.3 * mu_P / max(protection_days, 1)
            sigma_P = float(np.sqrt(protection_days) * daily_std)

        z = float(stats.norm.ppf(self.params.service_level))
        S = mu_P + z * sigma_P
        return max(S, 0.0)

    def recommend(
        self,
        daily_demand_forecast: np.ndarray,
        forecast_std: float | None = None,
    ) -> dict:
        """Compute periodic review recommendation."""
        inv_pos = inventory_position(
            self.params.on_hand,
            self.params.on_order,
            self.params.backorders,
        )

        S = self.compute_order_up_to_level(daily_demand_forecast, forecast_std)
        order_qty = max(S - inv_pos, 0.0)
        order_qty = max(order_qty, float(self.params.min_order_qty)) if order_qty > 0 else 0.0

        explanation = (
            f"Periodic review (every {self.review_period_days} days): "
            f"order up to {S:.1f} units. "
            f"Current inventory position: {inv_pos:.1f} units. "
            f"Suggested order: {order_qty:.0f} units."
        )

        return {
            "inventory_position": round(inv_pos, 2),
            "order_up_to_level": round(S, 2),
            "suggested_order_qty": round(order_qty, 0),
            "review_period_days": self.review_period_days,
            "protection_period_days": self.params.lead_time_days + self.review_period_days,
            "service_level": self.params.service_level,
            "explanation": explanation,
        }
