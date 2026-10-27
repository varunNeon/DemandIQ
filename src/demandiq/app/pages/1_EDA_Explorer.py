"""EDA Explorer — page 1."""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from demandiq.app.theme import (
    BLUE, BG_CARD, BORDER, MODEL_COLORS, MUTED, ORANGE, PLOTLY_TEMPLATE,
    apply_theme, page_header, render_sidebar_nav, stat_card,
)
from demandiq.database.connection import get_db, init_db
from demandiq.database.repositories import (
    get_all_products, get_all_stores, get_sales_dataframe, get_sales_date_range,
)

st.set_page_config(page_title="Explore · DemandIQ", layout="wide")
apply_theme()


@st.cache_data(ttl=120, show_spinner=False)
def _load_all_sales() -> pd.DataFrame:
    init_db()
    with get_db() as s:
        df = get_sales_dataframe(s)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=120, show_spinner=False)
def _load_series(store_id: str, product_id: str) -> pd.DataFrame:
    with get_db() as s:
        df = get_sales_dataframe(s, store_id=store_id, product_id=product_id)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_nav()
    st.divider()

with st.spinner("Loading data..."):
    all_df = _load_all_sales()

if all_df.empty:
    page_header("EDA Explorer", "Exploratory Data Analysis")
    st.info("No data ingested yet. Run: `python scripts/run_pipeline.py --generate-demo --ingest`")
    st.stop()

store_ids   = sorted(all_df["store_id"].unique())
product_ids = sorted(all_df["product_id"].unique())

with st.sidebar:
    selected_store   = st.selectbox("Store",   store_ids)
    selected_product = st.selectbox("Product", product_ids)

series_df = _load_series(selected_store, selected_product)

# ── Page header ───────────────────────────────────────────────────────────────
page_header("EDA Explorer", f"Exploring {selected_store} / {selected_product}")

# ── Series-level KPIs ─────────────────────────────────────────────────────────
if not series_df.empty:
    s = series_df["units_sold"]
    zero_pct = (s == 0).mean()
    cv = s.std() / s.mean() if s.mean() > 0 else float("nan")
    promo_lift_pct = None
    if "onpromotion" in series_df.columns:
        promo_mean  = series_df.loc[series_df["onpromotion"], "units_sold"].mean()
        normal_mean = series_df.loc[~series_df["onpromotion"], "units_sold"].mean()
        if normal_mean > 0:
            promo_lift_pct = (promo_mean - normal_mean) / normal_mean

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(stat_card("Observations",  f"{len(s):,}"),             unsafe_allow_html=True)
    c2.markdown(stat_card("Mean daily",    f"{s.mean():.1f}"),         unsafe_allow_html=True)
    c3.markdown(stat_card("Std dev",       f"{s.std():.1f}"),          unsafe_allow_html=True)
    c4.markdown(stat_card("Zero days",     f"{zero_pct:.1%}",          color="#E05B55" if zero_pct > 0.1 else "#D8B65C"), unsafe_allow_html=True)
    c5.markdown(stat_card("Coeff of var.", f"{cv:.2f}" if not np.isnan(cv) else "—"), unsafe_allow_html=True)

    # Intermittency classification
    if zero_pct > 0.3:
        st.warning(f"**Intermittent demand** — {zero_pct:.0%} zero days. Complex models may overfit; consider Croston's method or seasonal naive baseline.")
    elif cv > 1.5:
        st.warning(f"**High variability** — CV = {cv:.2f}. Prediction intervals will be wide.")

    st.markdown("<br>", unsafe_allow_html=True)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_ts, tab_dow, tab_dist, tab_promo, tab_heatmap, tab_global = st.tabs([
    "Time series", "Day of week", "Distribution",
    "Promotions", "Calendar", "Product comparison",
])

# ── Tab 1: Time series ────────────────────────────────────────────────────────
with tab_ts:
    if series_df.empty:
        st.info("No data for this series.")
    else:
        # 30-day rolling mean overlay
        ts = series_df.sort_values("date").copy()
        ts["roll_mean_30d"] = ts["units_sold"].rolling(30, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["date"], y=ts["units_sold"], mode="lines",
            name="Daily sales", line={"color": BLUE, "width": 1},
            opacity=0.6,
        ))
        fig.add_trace(go.Scatter(
            x=ts["date"], y=ts["roll_mean_30d"], mode="lines",
            name="30-day rolling mean", line={"color": ORANGE, "width": 2},
        ))
        # Promotion markers
        if "onpromotion" in ts.columns:
            promo = ts[ts["onpromotion"]]
            if not promo.empty:
                fig.add_trace(go.Scatter(
                    x=promo["date"], y=promo["units_sold"], mode="markers",
                    name="Promotion day", marker={"color": "#D8B65C", "size": 5, "symbol": "diamond"},
                ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            yaxis_title="Units sold", hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trend detection: compare first-half mean vs second-half mean
        mid = len(ts) // 2
        first_mean = ts["units_sold"].iloc[:mid].mean()
        second_mean = ts["units_sold"].iloc[mid:].mean()
        pct_change = (second_mean - first_mean) / first_mean * 100 if first_mean > 0 else 0
        direction = "↑" if pct_change > 2 else "↓" if pct_change < -2 else "→"
        st.caption(
            f"Trend: {direction} second half avg **{second_mean:.1f}** vs first half avg **{first_mean:.1f}** "
            f"({pct_change:+.1f}%). Computed from data — not assumed."
        )

# ── Tab 2: Day of week profile ────────────────────────────────────────────────
with tab_dow:
    if series_df.empty:
        st.info("No data.")
    else:
        ts = series_df.copy()
        ts["dow"] = ts["date"].dt.dayofweek
        ts["dow_name"] = ts["date"].dt.day_name()
        dow_stats = ts.groupby(["dow", "dow_name"])["units_sold"].agg(
            mean="mean", median="median", std="std"
        ).reset_index().sort_values("dow")

        overall_mean = ts["units_sold"].mean()
        dow_stats["lift_pct"] = (dow_stats["mean"] - overall_mean) / overall_mean * 100

        fig = go.Figure()
        fig.add_bar(
            x=dow_stats["dow_name"], y=dow_stats["mean"],
            name="Mean", marker_color=[
                "#D8B65C" if v > 5 else "#E05B55" if v < -5 else BLUE
                for v in dow_stats["lift_pct"]
            ],
            error_y={"type": "data", "array": dow_stats["std"], "visible": True, "color": MUTED},
        )
        fig.add_hline(y=overall_mean, line_dash="dot", line_color=MUTED,
                      annotation_text=f"Overall mean {overall_mean:.1f}")
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            yaxis_title="Mean daily units",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Error bars = ±1 std. Colour: green = +5%+ above average, red = −5%+ below average.")

# ── Tab 3: Distribution ───────────────────────────────────────────────────────
with tab_dist:
    if series_df.empty:
        st.info("No data.")
    else:
        s = series_df["units_sold"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=s, nbinsx=40, name="Sales distribution",
            marker_color=BLUE, opacity=0.75,
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=340,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            xaxis_title="Units sold", yaxis_title="Days",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary stats table
        desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        desc_df = pd.DataFrame({"Statistic": desc.index, "Value": desc.values.round(2)})
        st.dataframe(desc_df, hide_index=True, use_container_width=False)

# ── Tab 4: Promotions ─────────────────────────────────────────────────────────
with tab_promo:
    if series_df.empty or "onpromotion" not in series_df.columns:
        st.info("No promotion data available.")
    else:
        ts = series_df.copy()
        promo_df = ts.groupby("onpromotion")["units_sold"].agg(
            count="count", mean="mean", median="median", std="std"
        ).reset_index()
        promo_df["label"] = promo_df["onpromotion"].map({True: "Promotion", False: "Normal"})

        fig = px.box(
            ts, x="onpromotion", y="units_sold",
            color="onpromotion",
            color_discrete_map={True: "#D8B65C", False: BLUE},
            labels={"onpromotion": "On promotion", "units_sold": "Units sold"},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            height=340, margin={"l": 0, "r": 0, "t": 30, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        if promo_lift_pct is not None:
            direction = "higher" if promo_lift_pct > 0 else "lower"
            st.metric(
                "Promotion lift",
                f"{abs(promo_lift_pct):.1%} {direction} than non-promo days",
                delta=f"{promo_lift_pct:+.1%}",
            )
            st.caption("Promotion lift is computed from the data — not assumed. Interpret with caution: promotions may be placed on products already expected to sell well.")

# ── Tab 5: Heatmap ────────────────────────────────────────────────────────────
with tab_heatmap:
    if series_df.empty:
        st.info("No data.")
    else:
        ts = series_df.copy()
        ts["year"]  = ts["date"].dt.year
        ts["month"] = ts["date"].dt.month
        pivot = ts.pivot_table(values="units_sold", index="month", columns="year", aggfunc="mean")

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot.index = [month_names[m - 1] for m in pivot.index]

        fig = px.imshow(
            pivot, text_auto=".0f", aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Year", "y": "Month", "color": "Mean units"},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(
            height=360, margin={"l": 0, "r": 0, "t": 30, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Lighter = lower sales. Darker = higher sales. Empty cells = no data for that month/year.")

# ── Tab 6: All-product comparison ─────────────────────────────────────────────
with tab_global:
    with st.spinner("Computing product summaries..."):
        product_stats = all_df[all_df["store_id"] == selected_store].groupby("product_id").agg(
            mean_daily=("units_sold", "mean"),
            total=("units_sold", "sum"),
            zero_rate=("units_sold", lambda x: (x == 0).mean()),
            cv=("units_sold", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
            n_days=("units_sold", "count"),
        ).reset_index().sort_values("mean_daily", ascending=False)

    fig = px.scatter(
        product_stats,
        x="mean_daily", y="cv",
        size="total", color="zero_rate",
        hover_name="product_id",
        labels={
            "mean_daily": "Mean daily demand",
            "cv": "Coefficient of variation",
            "zero_rate": "Zero-day rate",
            "total": "Total units",
        },
        color_continuous_scale="RdYlGn_r",
        template=PLOTLY_TEMPLATE,
        title=f"Product demand profile — {selected_store}",
    )
    fig.update_layout(
        height=420, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bubble size = total volume. Colour = zero-day rate (green = few zeros, red = many zeros). "
        "High CV + high zero-rate = intermittent demand — harder to forecast."
    )

    st.dataframe(
        product_stats.rename(columns={
            "product_id": "Product", "mean_daily": "Mean daily",
            "total": "Total units", "zero_rate": "Zero-day rate",
            "cv": "CV", "n_days": "Obs.",
        }).style.format({
            "Mean daily": "{:.1f}", "Total units": "{:,.0f}",
            "Zero-day rate": "{:.1%}", "CV": "{:.2f}",
        }),
        hide_index=True, use_container_width=True,
    )
