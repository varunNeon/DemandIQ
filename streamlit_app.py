"""
DemandIQ Streamlit app.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="DemandIQ · Demand forecasting & inventory",
    page_icon="src/demandiq/app/icon.svg",
    layout="wide",
    initial_sidebar_state="auto",
)

from demandiq.app import data  # noqa: E402
from demandiq.app.theme import apply_theme  # noqa: E402
from demandiq.app.views import demand, forecast, inventory, method, overview  # noqa: E402

apply_theme()
data.ensure_database()

stores, products = data.load_catalog()
store_labels = {r.store_id: f"{r.store_id} — {r.city}" for r in stores.itertuples()}
product_labels = {r.product_id: f"{r.product_id.replace('_', ' ').title()} ({r.family.title()})"
                  for r in products.itertuples()}

st.session_state.setdefault("store_id", "S001")
st.session_state.setdefault("product_id", "TOYS_001" if "TOYS_001" in product_labels else products.product_id.iloc[0])

pages = [
    st.Page(overview.render, title="Overview", icon=":material/insights:", default=True),
    st.Page(demand.render, title="Demand explorer", icon=":material/stacked_line_chart:", url_path="demand"),
    st.Page(forecast.render, title="Forecast lab", icon=":material/online_prediction:", url_path="forecast"),
    st.Page(inventory.render, title="Inventory planner", icon=":material/inventory_2:", url_path="inventory"),
    st.Page(method.render, title="How it works", icon=":material/menu_book:", url_path="method"),
]
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown("#### Series")
    st.selectbox("Store", options=list(store_labels), format_func=store_labels.get, key="store_id")
    st.selectbox("Product", options=list(product_labels), format_func=product_labels.get, key="product_id")
    st.caption(
        "Synthetic demo data: 5 stores, 20 products, two years of daily sales. "
        "Every page follows the series chosen here."
    )

nav.run()
