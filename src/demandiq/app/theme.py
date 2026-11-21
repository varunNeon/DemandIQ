"""
DemandIQ visual system for the Streamlit app.

Palette
    coal      #0D0C0B  page background (warm charcoal)
    graphite  #161513  raised surfaces
    seam      #2B2824  hairlines and grid
    bone      #EEE8DA  primary text
    ash       #9A9385  secondary text
    gold      #D9B65D  observed demand, primary actions
    ember     #E0533D  anything forecast: point forecasts and uncertainty

Type
    Cormorant Garamond for headlines and large figures,
    IBM Plex Sans for interface text and data.
"""

from __future__ import annotations

import html
from typing import Callable, Iterable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

COAL = "#0D0C0B"
GRAPHITE = "#161513"
GRAPHITE_HI = "#1E1C19"
SEAM = "#2B2824"
GRID = "#211F1C"
BONE = "#EEE8DA"
ASH = "#9A9385"
DIM = "#6B655A"
GOLD = "#D9B65D"
GOLD_SOFT = "#8C7638"
EMBER = "#E0533D"
EMBER_SOFT = "#7A2E22"

MODEL_COLORS = {
    "naive": "#7D776B",
    "seasonal_naive": "#B3AA98",
    "ets": "#C9A24A",
    "lgbm": EMBER,
}

DISPLAY_FONT = "'Cormorant Garamond', 'Iowan Old Style', Georgia, serif"
BODY_FONT = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"

pio.templates["demandiq"] = go.layout.Template(
    layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": BODY_FONT, "color": ASH, "size": 12},
        "colorway": [GOLD, EMBER, "#B3AA98", "#7D776B"],
        "margin": {"l": 8, "r": 8, "t": 16, "b": 8},
        "xaxis": {
            "gridcolor": GRID, "linecolor": SEAM, "zeroline": False,
            "tickfont": {"color": ASH}, "showspikes": True, "spikecolor": SEAM,
            "spikethickness": 1, "spikedash": "solid", "spikemode": "across",
        },
        "yaxis": {
            "gridcolor": GRID, "linecolor": SEAM, "zeroline": False,
            "tickfont": {"color": ASH}, "rangemode": "tozero",
        },
        "legend": {
            "orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0,
            "bgcolor": "rgba(0,0,0,0)", "font": {"color": BONE},
        },
        "hoverlabel": {
            "bgcolor": GRAPHITE_HI, "bordercolor": SEAM,
            "font": {"color": BONE, "family": BODY_FONT},
        },
        "hovermode": "x unified",
    }
)
pio.templates.default = "demandiq"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
  --coal: {COAL}; --graphite: {GRAPHITE}; --graphite-hi: {GRAPHITE_HI};
  --seam: {SEAM}; --bone: {BONE}; --ash: {ASH}; --dim: {DIM};
  --gold: {GOLD}; --ember: {EMBER};
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
  background: var(--coal);
  color: var(--bone);
  font-family: {BODY_FONT};
  font-feature-settings: "tnum" 1, "lnum" 1;
}}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button,
[data-testid="stMarkdownContainer"], [data-baseweb="select"] {{ font-family: {BODY_FONT}; }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }}

h1, h2, h3, .stApp h1, .stApp h2, .stApp h3 {{
  font-family: {DISPLAY_FONT} !important;
  color: var(--bone);
  letter-spacing: -0.01em;
  font-weight: 600;
}}
.stApp h2 {{ font-size: 2rem; line-height: 1.1; margin: 0.4rem 0 0.2rem; }}
.stApp h3 {{ font-size: 1.45rem; line-height: 1.15; margin: 0.2rem 0 0.1rem; }}

/* Sidebar */
[data-testid="stSidebar"] {{
  background: var(--graphite);
  border-right: 1px solid var(--seam);
}}
[data-testid="stSidebarNav"] a span, [data-testid="stSidebarNavLink"] span {{ font-size: 0.95rem; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(217,182,93,0.10); }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--gold); }}

/* Widgets */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stNumberInput input {{
  background: var(--graphite-hi) !important; border-color: var(--seam) !important;
}}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  background: var(--gold); color: #17140C; border: 0; font-weight: 600;
  border-radius: 6px; padding: 0.5rem 1.1rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{ background: #E6C672; color: #17140C; }}
.stButton > button p, .stDownloadButton > button p, [data-testid="stFormSubmitButton"] > button p {{ color: #17140C; font-weight: 600; }}
.stButton > button:focus-visible {{ outline: 2px solid var(--bone); outline-offset: 2px; }}
[data-testid="stExpander"] {{ border: 1px solid var(--seam); border-radius: 8px; background: var(--graphite); }}
[data-testid="stTabs"] button p {{ font-size: 0.95rem; }}
[data-testid="stTabs"] [aria-selected="true"] p {{ color: var(--gold); }}
hr {{ border-color: var(--seam); }}

/* Masthead */
.diq-mast {{ margin: 0 0 1.6rem; }}
.diq-mark {{ font-family: {DISPLAY_FONT}; font-size: 1.05rem; color: var(--ash); margin-bottom: 0.9rem; }}
.diq-mark b {{ color: var(--gold); font-weight: 700; }}
.diq-mast h1 {{
  font-size: clamp(2.4rem, 4.6vw, 3.9rem) !important; line-height: 1.02 !important;
  margin: 0 0 0.7rem !important; max-width: 17ch; font-weight: 600 !important;
}}
.diq-lede {{ color: var(--ash); font-size: 1.05rem; line-height: 1.6; max-width: 62ch; margin: 0; }}

/* Headline result */
.diq-result {{
  display: grid; grid-template-columns: auto 1fr; gap: 1.6rem; align-items: center;
  padding: 1.4rem 0 1.5rem; border-top: 1px solid var(--seam); border-bottom: 1px solid var(--seam);
  margin: 0.4rem 0 1.6rem;
}}
.diq-result .num {{
  font-family: {DISPLAY_FONT}; font-size: clamp(3.6rem, 8vw, 6.2rem); line-height: 0.9;
  color: var(--ember); font-weight: 600;
}}
.diq-result .num small {{ font-size: 0.45em; color: var(--ember); }}
.diq-result p {{ margin: 0; font-size: 1.12rem; line-height: 1.55; color: var(--bone); max-width: 58ch; }}
.diq-result p span {{ color: var(--ash); display: block; font-size: 0.92rem; margin-top: 0.35rem; }}

/* Figures row */
.diq-figures {{ display: flex; flex-wrap: wrap; gap: 0.5rem 2.6rem; margin: 0.3rem 0 1.2rem; }}
.diq-fig .v {{ font-family: {DISPLAY_FONT}; font-size: 2.3rem; line-height: 1; color: var(--bone); font-weight: 600; }}
.diq-fig .v.gold {{ color: var(--gold); }} .diq-fig .v.ember {{ color: var(--ember); }}
.diq-fig .l {{ color: var(--ash); font-size: 0.84rem; margin-top: 0.3rem; max-width: 22ch; line-height: 1.35; }}

/* Section intro */
.diq-section {{ margin: 2.2rem 0 0.6rem; }}
.diq-section p {{ color: var(--ash); margin: 0.2rem 0 0; max-width: 70ch; line-height: 1.55; }}

/* Decision panel */
.diq-decision {{
  border-left: 3px solid var(--gold); background: var(--graphite);
  padding: 1.1rem 1.3rem; border-radius: 0 8px 8px 0; margin: 0.2rem 0 1rem;
}}
.diq-decision.urgent {{ border-left-color: var(--ember); }}
.diq-decision h3 {{ margin: 0 0 0.35rem !important; font-size: 1.9rem !important; }}
.diq-decision p {{ margin: 0; color: var(--ash); line-height: 1.55; }}

.diq-note {{ color: var(--dim); font-size: 0.85rem; line-height: 1.5; margin-top: 0.4rem; }}
.diq-legend {{ display: flex; gap: 1.4rem; flex-wrap: wrap; color: var(--ash); font-size: 0.85rem; margin: 0.2rem 0 0.4rem; }}
.diq-legend i {{ display: inline-block; width: 18px; height: 3px; vertical-align: middle; margin-right: 0.45rem; border-radius: 2px; }}
.diq-legend i.band {{ height: 10px; opacity: 0.55; }}

/* Tables */
.diq-table {{ width: 100%; border-collapse: collapse; font-size: 0.93rem; }}
.diq-table th {{
  text-align: right; color: var(--ash); font-weight: 500; padding: 0.55rem 0.7rem;
  border-bottom: 1px solid var(--seam);
}}
.diq-table th:first-child, .diq-table td:first-child {{ text-align: left; }}
.diq-table td {{ text-align: right; padding: 0.6rem 0.7rem; border-bottom: 1px solid {GRID}; color: var(--bone); white-space: nowrap; }}
.diq-table th {{ white-space: nowrap; }}
.diq-scroll {{ overflow-x: auto; }}
.diq-table tr.top td {{ color: var(--bone); background: rgba(224,83,61,0.07); }}
.diq-table tr.top td:first-child {{ box-shadow: inset 3px 0 0 var(--ember); }}
.diq-table td .sub {{ color: var(--dim); font-size: 0.8rem; margin-left: 0.4rem; }}
.diq-bar {{ display: inline-block; height: 6px; border-radius: 3px; background: var(--seam); width: 90px; vertical-align: middle; margin-left: 0.6rem; position: relative; }}
.diq-bar b {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--gold); }}

/* Steps (a real sequence) */
.diq-steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.4rem; margin-top: 0.8rem; }}
.diq-step .n {{ font-family: {DISPLAY_FONT}; color: var(--gold); font-size: 1.5rem; }}
.diq-step h4 {{ margin: 0.1rem 0 0.3rem; font-size: 1rem; color: var(--bone); font-weight: 600; }}
.diq-step p {{ margin: 0; color: var(--ash); font-size: 0.9rem; line-height: 1.5; }}

@media (max-width: 820px) {{
  .diq-result {{ grid-template-columns: 1fr; gap: 0.6rem; }}
  .diq-steps {{ grid-template-columns: 1fr 1fr; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def apply_theme() -> None:
    """Inject fonts and styles. Call once per run from the entry point."""
    st.html(_CSS)


def _e(text: object) -> str:
    return html.escape(str(text))


def masthead(title: str, lede: str) -> None:
    st.html(
        f'<div class="diq-mast"><div class="diq-mark">Demand<b>IQ</b></div>'
        f"<h1>{_e(title)}</h1><p class='diq-lede'>{_e(lede)}</p></div>"
    )


def section(title: str, intro: str | None = None) -> None:
    body = f"<p>{_e(intro)}</p>" if intro else ""
    st.html(f'<div class="diq-section"><h2>{_e(title)}</h2>{body}</div>')


def figures(items: Iterable[tuple[str, str, str]]) -> None:
    """Row of (value, label, tone) figures; tone is '', 'gold' or 'ember'."""
    cells = "".join(
        f'<div class="diq-fig"><div class="v {tone}">{_e(v)}</div><div class="l">{_e(label)}</div></div>'
        for v, label, tone in items
    )
    st.html(f'<div class="diq-figures">{cells}</div>')


def result_banner(number: str, suffix: str, sentence: str, detail: str) -> None:
    st.html(
        f'<div class="diq-result"><div class="num">{_e(number)}<small>{_e(suffix)}</small></div>'
        f"<p>{_e(sentence)}<span>{_e(detail)}</span></p></div>"
    )


def decision(title: str, body: str, urgent: bool = False) -> None:
    cls = "diq-decision urgent" if urgent else "diq-decision"
    st.html(f'<div class="{cls}"><h3>{_e(title)}</h3><p>{_e(body)}</p></div>')


def note(text: str) -> None:
    st.html(f'<p class="diq-note">{_e(text)}</p>')


def chart_legend(items: Iterable[tuple[str, str, bool]]) -> None:
    """(label, color, is_band) swatches shown above a chart."""
    cells = "".join(
        f'<span><i class="{"band" if band else ""}" style="background:{color}"></i>{_e(label)}</span>'
        for label, color, band in items
    )
    st.html(f'<div class="diq-legend">{cells}</div>')


def steps(items: Iterable[tuple[str, str]]) -> None:
    cells = "".join(
        f'<div class="diq-step"><div class="n">{i}</div><h4>{_e(t)}</h4><p>{_e(b)}</p></div>'
        for i, (t, b) in enumerate(items, start=1)
    )
    st.html(f'<div class="diq-steps">{cells}</div>')


def table(
    df: pd.DataFrame,
    formats: dict[str, Callable[[object], str]] | None = None,
    highlight_first: bool = False,
    raw_html_cols: tuple[str, ...] = (),
) -> None:
    """Render a DataFrame as a styled HTML table."""
    formats = formats or {}
    head = "".join(f"<th>{_e(c)}</th>" for c in df.columns)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for col in df.columns:
            value = row[col]
            text = formats[col](value) if col in formats else value
            cells.append(f"<td>{text if col in raw_html_cols else _e(text)}</td>")
        cls = ' class="top"' if highlight_first and i == 0 else ""
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    st.html(
        f'<div class="diq-scroll"><table class="diq-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def pct(value: object, digits: int = 1) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if pd.isna(v) else f"{v * 100:.{digits}f}%"


def num(value: object, digits: int = 0) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if pd.isna(v) else f"{v:,.{digits}f}"
