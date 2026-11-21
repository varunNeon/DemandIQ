"""Inventory planner: turn the probabilistic forecast into an order."""

from __future__ import annotations

import streamlit as st

from demandiq.app import charts, data, theme
from demandiq.services.inventory_service import (
    InventoryInputs,
    project_stock,
    recommend_from_forecast,
)


def render() -> None:
    store_id = st.session_state["store_id"]
    product_id = st.session_state["product_id"]
    series = data.load_series(store_id, product_id)
    name = product_id.replace("_", " ").title()
    recent_week = float(series["units_sold"].tail(7).sum())

    theme.masthead(
        "How much to order, and when.",
        f"A reorder-point plan for {name} at {store_id}, sized from the LightGBM forecast and its "
        "measured uncertainty. Costs, lead time and stock levels are your assumptions; adjust "
        "them and the plan updates.",
    )

    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.markdown("##### Your situation")
        service = st.slider("Target service level", 0.80, 0.99, 0.95, 0.01, format="%.2f",
                            help="Chance of not running out before the next delivery arrives.")
        lead = st.slider("Supplier lead time (days)", 1, 14, 5)
        on_hand = st.number_input("Units on the shelf now", min_value=0, value=int(recent_week), step=10,
                                  help="Defaults to last week's sales, as a stand-in.")
        on_order = st.number_input("Units already on order", min_value=0, value=0, step=10)
        with st.expander("Cost assumptions"):
            holding = st.number_input("Holding cost per unit per day", 0.0, 5.0, 0.05, 0.01, format="%.2f")
            ordering = st.number_input("Cost per order placed", 0.0, 1000.0, 50.0, 5.0)
            shortage = st.number_input("Cost per unit of lost sales", 0.0, 100.0, 5.0, 0.5)

    inputs = InventoryInputs(
        lead_time_days=lead, service_level=service, on_hand=float(on_hand), on_order=float(on_order),
        holding_cost_per_unit_day=holding, ordering_cost=ordering, shortage_cost_per_unit=shortage,
    )

    with st.spinner("Forecasting demand over the lead time…"):
        results = data.run_forecasts(store_id, product_id, ("lgbm",), 14)
    future = results["lgbm"].future
    rec = recommend_from_forecast(future, inputs)

    plan = project_stock(future, inputs, rec)
    first_order = plan.loc[plan["order_placed"] > 0, "date"]
    with right:
        if rec["should_order"]:
            theme.decision(
                f"Order {rec['suggested_order_qty']:,.0f} units today",
                f"Stock on hand plus on order ({rec['inventory_position']:,.0f}) is at or below the "
                f"reorder point of {rec['reorder_point']:,.0f}. Without an order there is a "
                f"{rec['stockout_risk']:.0%} chance of running out before a delivery could arrive.",
                urgent=True,
            )
        else:
            gap = rec["inventory_position"] - rec["reorder_point"]
            when = (
                f" At the forecast pace you reach it on {first_order.iloc[0]:%A %d %B}, so plan the "
                f"next order for then."
                if len(first_order) else " Stock covers the whole two-week forecast."
            )
            theme.decision(
                "No order needed today",
                f"You hold {gap:,.0f} units above the reorder point of {rec['reorder_point']:,.0f}."
                f"{when} The chance of running out within the {lead}-day lead time is "
                f"{rec['stockout_risk']:.1%}.",
            )
        theme.figures([
            (theme.num(rec["reorder_point"]), "reorder point (units)", "gold"),
            (theme.num(rec["safety_stock"]), f"safety stock for a {service:.0%} service level", ""),
            (f"{theme.num(rec['mu_LT'])} ± {theme.num(rec['sigma_LT'])}", f"expected demand over {lead} days", "ember"),
            (f"{rec['days_of_cover']:.1f}" if rec["days_of_cover"] else "—", "days of cover in stock", ""),
        ])
        st.markdown("##### Expected stock over the next two weeks")
        st.caption(
            f"Following the plan means ordering {rec['suggested_order_qty']:,.0f} units whenever "
            "stock plus open orders falls to the reorder point."
        )
        st.plotly_chart(charts.stock_projection(plan, rec["reorder_point"]),
                        use_container_width=True, config=theme.PLOTLY_CONFIG)
        theme.note(
            f"Safety stock uses forecast uncertainty from the {rec['uncertainty_source']}, assuming "
            "independent daily errors and roughly normal lead-time demand. Order size follows the "
            "economic order quantity for your cost inputs."
        )

    # ---- Stress test -------------------------------------------------------
    params = data.inputs_dict(inputs)
    with st.spinner("Simulating six months of ordering decisions…"):
        tradeoff = data.run_tradeoff(store_id, product_id, params)
        sim = data.run_simulation(store_id, product_id, params)

    cheapest = tradeoff.loc[tradeoff["total_cost"].idxmin()]
    ratio = shortage / holding if holding > 0 else float("inf")
    theme.section(
        "Choosing a service level",
        "The same policy replayed over the last six months of demand at targets from 80% to 99%. "
        "Each decision only sees sales up to that day, and unmet demand is lost. "
        f"With these costs a lost sale costs {ratio:,.0f} times as much as holding a unit for a "
        f"day, so the cheapest target is {cheapest['target_service_level']:.0%}. Raise the holding "
        "cost or lower the shortage cost to watch the balance shift.",
    )

    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.plotly_chart(charts.tradeoff_chart(tradeoff, service), use_container_width=True,
                        config=theme.PLOTLY_CONFIG)
    with c2:
        filled = sim["demand_fulfilled"].sum() / max(sim["demand_actual"].sum(), 1)
        stockout_days = int((sim["lost_sales"] > 0).sum())
        cost = (sim["holding_cost"] + sim["ordering_cost"] + sim["shortage_cost"]).sum()
        theme.figures([
            (theme.pct(filled), "of demand served from stock", "gold"),
            (str(stockout_days), "days with a stock-out", "ember"),
            (theme.num(cost), "total cost over six months", ""),
        ])
        st.plotly_chart(charts.simulation_chart(sim), use_container_width=True, config=theme.PLOTLY_CONFIG)
