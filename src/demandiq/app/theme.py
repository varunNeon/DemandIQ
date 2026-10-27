"""DemandIQ's shared dark, gold, and red interface system."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK = "#08090B"
SURFACE = "#101114"
SURFACE_RAISED = "#17181C"
GOLD = "#D8B65C"
GOLD_BRIGHT = "#F2D27A"
RED = "#E05B55"
TEXT = "#F1EFE8"
MUTED = "#93918A"
BORDER = "#2A2B2F"

# Keep old import names working while making every chart follow the new palette.
BLUE = "#BEB9AC"
ORANGE = GOLD_BRIGHT
GREEN = GOLD
BG_CARD = SURFACE
BG_PAGE = INK

MODEL_COLORS = {
    "naive": "#C2BCA9",
    "seasonal_naive": "#A98D48",
    "ets": GOLD_BRIGHT,
    "lgbm": RED,
}

_TEMPLATE_NAME = "demandiq_dark"
pio.templates[_TEMPLATE_NAME] = go.layout.Template(
    layout={
        "paper_bgcolor": INK,
        "plot_bgcolor": INK,
        "font": {"family": "Aptos, Segoe UI, sans-serif", "color": TEXT, "size": 12},
        "colorway": [GOLD, RED, "#C2BCA9", GOLD_BRIGHT],
        "xaxis": {
            "gridcolor": "#25262A",
            "linecolor": BORDER,
            "zerolinecolor": BORDER,
            "tickfont": {"color": MUTED},
        },
        "yaxis": {
            "gridcolor": "#25262A",
            "linecolor": BORDER,
            "zerolinecolor": BORDER,
            "tickfont": {"color": MUTED},
        },
        "legend": {"bgcolor": "rgba(8,9,11,0.75)", "borderwidth": 0},
        "hoverlabel": {
            "bgcolor": SURFACE_RAISED,
            "bordercolor": GOLD,
            "font": {"color": TEXT},
        },
    }
)
PLOTLY_TEMPLATE = _TEMPLATE_NAME


def apply_theme() -> None:
    """Apply the shared interface styling to the active Streamlit page."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --diq-ink: #08090B;
            --diq-surface: #101114;
            --diq-raised: #17181C;
            --diq-gold: #D8B65C;
            --diq-gold-bright: #F2D27A;
            --diq-red: #E05B55;
            --diq-text: #F1EFE8;
            --diq-muted: #93918A;
            --diq-border: #2A2B2F;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 78% -18%, rgba(216,182,92,.075), transparent 36%),
                var(--diq-ink);
            color: var(--diq-text);
            cursor: default;
        }
        .stApp * { cursor: default !important; }
        .stApp a, .stApp button, .stApp [role="button"], .stApp [role="slider"],
        .stApp select, .stApp input[type="checkbox"], .stApp input[type="radio"] {
            cursor: pointer !important;
        }
        .stApp input:not([type="checkbox"]):not([type="radio"]),
        .stApp textarea { cursor: text !important; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
        .main .block-container { max-width: 1600px; padding: 2.3rem 3.1rem 4rem; }

        section[data-testid="stSidebar"] {
            background: #0B0C0E;
            border-right: 1px solid var(--diq-border);
        }
        section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
        [data-testid="stSidebarNav"] { display: none; }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--diq-muted);
        }

        h1, h2, h3 {
            color: var(--diq-text) !important;
            letter-spacing: -.035em;
        }
        h1 { font-size: 2.15rem !important; font-weight: 620 !important; }
        h2 { font-size: 1.35rem !important; font-weight: 580 !important; }
        h3 { font-size: 1.05rem !important; font-weight: 580 !important; }
        a[data-testid="stHeaderLink"] { display: none; }
        p, li, label, [data-testid="stCaptionContainer"] { color: #C2C0B8; }
        [data-testid="stCaptionContainer"] { color: var(--diq-muted); }

        .diq-brand {
            display: flex; align-items: center; gap: .7rem;
            padding: .2rem .1rem 1rem;
            border-bottom: 1px solid var(--diq-border);
            margin-bottom: .8rem;
        }
        .diq-mark {
            display: grid; place-items: center; width: 2rem; height: 2rem;
            border: 1px solid rgba(216,182,92,.65); color: var(--diq-gold-bright);
            font-size: 1.1rem; transform: rotate(45deg);
        }
        .diq-mark span { transform: rotate(-45deg); color: var(--diq-gold-bright); }
        .diq-brand-name { color: var(--diq-text); font-size: 1rem; font-weight: 650; letter-spacing: .04em; }
        .diq-brand-sub { color: var(--diq-muted); font-size: .63rem; letter-spacing: .13em; text-transform: uppercase; }
        .diq-nav-label {
            color: var(--diq-muted); font-size: .64rem; letter-spacing: .14em;
            text-transform: uppercase; margin: 1.15rem 0 .35rem;
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"] {
            border: 1px solid transparent; border-radius: 5px; padding: .48rem .65rem;
            color: #B9B7AF; transition: all .16s ease;
        }
        section[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]:hover {
            background: rgba(216,182,92,.07); border-color: rgba(216,182,92,.2);
            color: var(--diq-gold-bright);
        }
        section[data-testid="stSidebar"] a[aria-current="page"] {
            background: rgba(216,182,92,.1); border-color: rgba(216,182,92,.28);
            color: var(--diq-gold-bright);
        }
        section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
            background: rgba(216,182,92,.1) !important;
            border-color: rgba(216,182,92,.28) !important;
            color: var(--diq-gold-bright) !important;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #121316, #0C0D0F);
            border: 1px solid var(--diq-border); border-top: 1px solid #48402B;
            border-radius: 6px; padding: 1rem 1.1rem;
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--diq-muted) !important; font-size: .68rem;
            letter-spacing: .1em; text-transform: uppercase;
        }
        div[data-testid="stMetricValue"] { color: var(--diq-gold-bright) !important; }
        .diq-stat {
            min-height: 5.8rem; padding: .8rem 1rem;
            background: linear-gradient(145deg, #121316, #0C0D0F);
            border: 1px solid var(--diq-border); border-top: 1px solid #48402B;
            border-radius: 5px;
        }
        .diq-stat-label { color: var(--diq-muted); font-size: .65rem; font-weight: 600;
            letter-spacing: .12em; text-transform: uppercase; }
        .diq-stat-value { color: var(--diq-text); font-size: 1.55rem; font-weight: 620;
            letter-spacing: -.035em; margin-top: .35rem; }

        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: #101114; border-color: #333439; border-radius: 5px;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {
            border-color: var(--diq-gold) !important;
            box-shadow: 0 0 0 1px rgba(216,182,92,.18);
        }
        button[kind="primary"] {
            background: var(--diq-gold); color: #13110A; border: 1px solid var(--diq-gold);
            border-radius: 5px; font-weight: 650;
        }
        button[kind="primary"]:hover { background: var(--diq-gold-bright); border-color: var(--diq-gold-bright); }
        button[kind="secondary"], [data-testid="stPageLink-NavLink"] {
            border-radius: 5px;
        }
        hr { border-color: var(--diq-border); }
        details, div[data-testid="stDataFrame"] { border-color: var(--diq-border); border-radius: 5px; }
        div[data-testid="stAlert"] { border-radius: 5px; }
        .stPlotlyChart { border: 1px solid #202125; border-radius: 6px; padding: .35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav() -> None:
    """Render consistent product navigation in place of Streamlit's default list."""
    st.markdown(
        """
        <div class="diq-brand">
            <div class="diq-mark"><span>◈</span></div>
            <div><div class="diq-brand-name">DEMANDIQ</div>
            <div class="diq-brand-sub">Forecast intelligence</div></div>
        </div>
        <div class="diq-nav-label">Workspace</div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("main.py", label="Overview")
    st.page_link("pages/1_EDA_Explorer.py", label="Explore demand")
    st.page_link("pages/2_Forecast.py", label="Forecast studio")
    st.page_link("pages/3_Inventory_Advisor.py", label="Inventory advisor")


def page_header(title: str, subtitle: str = "") -> None:
    """Render a restrained, product-level page heading."""
    st.markdown(
        f"""
        <div style="padding:0 0 1.25rem; border-bottom:1px solid {BORDER}; margin-bottom:1.4rem;">
            <div style="width:2rem; height:2px; background:{GOLD}; margin-bottom:.8rem;"></div>
            <h1 style="margin:0; font-size:2.15rem; font-weight:620; color:{TEXT};">{title}</h1>
            {f'<p style="margin:.35rem 0 0; color:{MUTED}; font-size:.9rem;">{subtitle}</p>' if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, delta: str = "", color: str = TEXT) -> str:
    """Return markup for a compact, high-contrast metric panel."""
    delta_html = f'<div style="color:{GOLD};font-size:.72rem;margin-top:.25rem">{delta}</div>' if delta else ""
    return (
        '<div class="diq-stat">'
        f'<div class="diq-stat-label">{label}</div>'
        f'<div class="diq-stat-value" style="color:{color}">{value}</div>'
        f"{delta_html}</div>"
    )
