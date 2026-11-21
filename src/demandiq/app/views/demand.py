"""Demand explorer: what drives this series."""

from __future__ import annotations

import numpy as np
import streamlit as st

from demandiq.app import charts, data, theme


def render() -> None:
    store_id = st.session_state["store_id"]
    product_id = st.session_state["product_id"]
    df = data.load_series(store_id, product_id)
    stores, products = data.load_catalog()
    product = products.set_index("product_id").loc[product_id]
    city = stores.set_index("store_id").loc[store_id, "city"]
    name = product_id.replace("_", " ").title()

    theme.masthead(
        f"{name} in {city}",
        f"{product['family'].title()}, {product['category'].lower()}. Two years of daily unit sales "
        f"at store {store_id}, with the patterns a forecast has to capture.",
    )

    regular = df[(df["onpromotion"] == 0) & (df["is_holiday"] == 0)]
    promo = df[(df["onpromotion"] == 1) & (df["is_holiday"] == 0)]
    holiday = df[df["is_holiday"] == 1]
    base = regular["units_sold"].mean()
    promo_lift = promo["units_sold"].mean() / base - 1 if len(promo) else np.nan
    holiday_drop = holiday["units_sold"].mean() / base - 1 if len(holiday) else np.nan
    weekend = regular[regular["date"].dt.dayofweek >= 5]["units_sold"].mean()
    weekday = regular[regular["date"].dt.dayofweek < 5]["units_sold"].mean()
    last_year = df[df["date"] > df["date"].max() - np.timedelta64(365, "D")]["units_sold"].sum()
    first_year = df[df["date"] <= df["date"].min() + np.timedelta64(364, "D")]["units_sold"].sum()

    theme.figures([
        (theme.num(base), "units on an ordinary day", "gold"),
        (f"{promo_lift:+.0%}", "on promotion days", "ember"),
        (f"{weekend / weekday - 1:+.0%}", "weekends against weekdays", ""),
        (f"{holiday_drop:+.0%}", "on public holidays", ""),
        (f"{last_year / first_year - 1:+.0%}", "year-on-year volume", ""),
    ])

    theme.section("Daily sales", "Daily units with a 28-day average. Diamonds mark public holidays.")
    st.plotly_chart(charts.history_chart(df), use_container_width=True, config=theme.PLOTLY_CONFIG)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Average day of the week")
        st.caption("Ordinary days only, so promotions and holidays do not distort the shape.")
        st.plotly_chart(charts.weekday_chart(df), use_container_width=True, config=theme.PLOTLY_CONFIG)
    with right:
        st.markdown("##### Weekday by month")
        st.caption("Average units. Bright cells are where demand concentrates through the year.")
        st.plotly_chart(charts.seasonality_heatmap(df), use_container_width=True, config=theme.PLOTLY_CONFIG)

    theme.section(
        "Across the whole range",
        "Year-two volume for every product in this store, so you can compare the selected series "
        "with its neighbours.",
    )
    sales = data.load_all_sales()
    store_sales = sales[(sales["store_id"] == store_id)
                        & (sales["date"] > sales["date"].max() - np.timedelta64(365, "D"))]
    totals = (
        store_sales.groupby("product_id")["units_sold"].agg(["sum", "mean"])
        .merge(products.set_index("product_id")[["family"]], left_index=True, right_index=True)
        .sort_values("sum", ascending=False)
        .reset_index()
    )
    peak = totals["sum"].max()
    totals["share"] = totals["sum"].map(
        lambda v: f"{v:,.0f}<span class='diq-bar'><b style='width:{v / peak * 100:.0f}%'></b></span>"
    )
    totals["Product"] = totals["product_id"].str.replace("_", " ").str.title()
    shown = totals[["Product", "family", "mean", "share"]]
    shown.columns = ["Product", "Family", "Units per day", "Units in the last year"]
    selected_first = shown.assign(_sel=totals["product_id"] == product_id)
    order = selected_first.sort_values("_sel", ascending=False, kind="stable").drop(columns="_sel")
    theme.table(
        order,
        formats={"Family": lambda v: str(v).title(), "Units per day": lambda v: theme.num(v, 1)},
        highlight_first=True,
        raw_html_cols=("Units in the last year",),
    )
    theme.note("The highlighted row is the series selected in the sidebar.")
