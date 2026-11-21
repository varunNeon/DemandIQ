"""Plotly figures for the DemandIQ app (uses the 'demandiq' template)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from demandiq.app.theme import ASH, BONE, EMBER, GOLD, GOLD_SOFT, MODEL_COLORS, SEAM
from demandiq.data.calendar import KNOWN_HOLIDAYS
from demandiq.services.forecast_service import MODEL_LABELS


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _band(fig: go.Figure, x, lower, upper, color: str, alpha: float, name: str) -> None:
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself", fillcolor=_rgba(color, alpha), line={"width": 0},
        hoverinfo="skip", name=name, showlegend=False,
    ))


def fan_chart(
    history: pd.DataFrame,
    future: pd.DataFrame,
    history_days: int = 120,
    height: int = 420,
    show_promos: bool = True,
) -> go.Figure:
    """Observed demand (gold) running into a forecast with 80%/95% bands (ember)."""
    hist = history.tail(history_days)
    fig = go.Figure()

    if "lower_95" in future and future["lower_95"].notna().all():
        _band(fig, future["date"], future["lower_95"], future["upper_95"], EMBER, 0.13, "95% interval")
    if "lower_80" in future and future["lower_80"].notna().all():
        _band(fig, future["date"], future["lower_80"], future["upper_80"], EMBER, 0.26, "80% interval")

    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["units_sold"], mode="lines", name="Observed",
        line={"color": GOLD, "width": 1.6},
        hovertemplate="%{y:,.0f} units<extra>Observed</extra>",
    ))
    if show_promos and "onpromotion" in hist:
        promo = hist[hist["onpromotion"] == 1]
        fig.add_trace(go.Scatter(
            x=promo["date"], y=promo["units_sold"], mode="markers", name="Promotion day",
            marker={"color": GOLD, "size": 6, "symbol": "circle-open", "line": {"width": 1.4}},
            hovertemplate="Promotion<extra></extra>",
        ))

    # Bridge the last observation to the first forecast so the line reads continuously
    bridge_x = [hist["date"].iloc[-1], future["date"].iloc[0]]
    bridge_y = [hist["units_sold"].iloc[-1], future["point_forecast"].iloc[0]]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y, mode="lines", line={"color": EMBER, "width": 1.2, "dash": "dot"},
        hoverinfo="skip", showlegend=False,
    ))
    custom = np.column_stack([
        future.get("lower_80", pd.Series(np.nan, index=future.index)),
        future.get("upper_80", pd.Series(np.nan, index=future.index)),
    ])
    fig.add_trace(go.Scatter(
        x=future["date"], y=future["point_forecast"], mode="lines+markers", name="Forecast",
        line={"color": EMBER, "width": 2.2}, marker={"size": 4, "color": EMBER},
        customdata=custom,
        hovertemplate="%{y:,.0f} units (80%: %{customdata[0]:,.0f}–%{customdata[1]:,.0f})<extra>Forecast</extra>",
    ))

    holidays = future[[d.strftime("%Y-%m-%d") in KNOWN_HOLIDAYS for d in future["date"]]]
    if not holidays.empty:
        fig.add_trace(go.Scatter(
            x=holidays["date"], y=holidays["point_forecast"], mode="markers+text",
            marker={"color": BONE, "size": 7, "symbol": "diamond"},
            text=["Holiday"] * len(holidays), textposition="bottom center",
            textfont={"color": ASH, "size": 10},
            hovertemplate="Public holiday: lower demand expected<extra></extra>", showlegend=False,
        ))

    origin = hist["date"].iloc[-1]
    fig.add_vline(x=origin, line={"color": SEAM, "width": 1})
    fig.add_annotation(
        x=origin, y=1, yref="paper", text="Forecast origin", showarrow=False,
        xanchor="right", yanchor="top", xshift=-6, font={"color": ASH, "size": 11},
    )
    fig.update_layout(height=height, showlegend=False, hovermode="x unified")
    fig.update_yaxes(title=None, tickformat=",.0f")
    return fig


def backtest_replay(history: pd.DataFrame, result, height: int = 360) -> go.Figure:
    """Backtest forecasts laid over what actually happened.

    Fold origins are a week apart and each covers two weeks, so only every
    other fold (ending with the latest) is drawn to keep the windows from overlapping.
    """
    folds = result.backtest.folds[::-1][::2][::-1]
    start = folds[0].val_dates[0] - pd.Timedelta(days=21)
    hist = history[history["date"] >= start]
    fig = go.Figure()
    for fold in folds:
        if fold.lower_80 is not None:
            _band(fig, fold.val_dates, fold.lower_80, fold.upper_80, EMBER, 0.16, "80% interval")
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["units_sold"], mode="lines", name="Actual",
        line={"color": GOLD, "width": 1.4},
        hovertemplate="%{y:,.0f}<extra>Actual</extra>",
    ))
    for i, fold in enumerate(folds):
        fig.add_trace(go.Scatter(
            x=fold.val_dates, y=fold.predicted, mode="lines", name="Backtest forecast",
            line={"color": EMBER, "width": 1.6}, showlegend=False, opacity=0.9,
            hovertemplate=f"Fold {i + 1}: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(height=height, showlegend=False, hovermode="closest")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def coverage_chart(results: dict, height: int = 300) -> go.Figure:
    """Observed interval coverage against the nominal target, per model."""
    names = [n for n in results if np.isfinite(results[n].metrics.get("Coverage_80", np.nan))]
    labels = [MODEL_LABELS[n] for n in names]
    fig = go.Figure()
    for level, color, alpha in (("80", EMBER, 0.55), ("95", EMBER, 0.95)):
        vals = [results[n].metrics.get(f"Coverage_{level}", np.nan) for n in names]
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=f"{level}% interval",
            marker={"color": _rgba(color, alpha)},
            hovertemplate=f"{level}% interval: %{{y:.0%}}<extra>%{{x}}</extra>",
        ))
        fig.add_hline(y=int(level) / 100, line={"color": BONE if level == "95" else ASH,
                                                "width": 1, "dash": "dot"})
    fig.update_layout(height=height, barmode="group", bargap=0.35, hovermode="closest",
                      legend={"y": 1.08})
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return fig


def importance_chart(importance: pd.DataFrame, top: int = 10, height: int = 340) -> go.Figure:
    df = importance.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["label"], orientation="h",
        marker={"color": [GOLD if i == len(df) - 1 else GOLD_SOFT for i in range(len(df))]},
        hovertemplate="%{x:.1%} of total gain<extra></extra>",
    ))
    fig.update_layout(height=height, hovermode="closest", margin={"l": 8, "r": 8, "t": 8, "b": 8})
    fig.update_xaxes(tickformat=".0%")
    return fig


def horizon_error_chart(by_horizon: pd.DataFrame, height: int = 300) -> go.Figure:
    fig = go.Figure()
    for model in ["naive", "seasonal_naive", "ets", "lgbm"]:
        d = by_horizon[by_horizon["model"] == model]
        if d.empty:
            continue
        fig.add_trace(go.Scatter(
            x=d["horizon_step"], y=d["WAPE"], mode="lines+markers", name=MODEL_LABELS[model],
            line={"color": MODEL_COLORS[model], "width": 2.4 if model == "lgbm" else 1.6},
            marker={"size": 5},
            hovertemplate="%{y:.1%}<extra>" + MODEL_LABELS[model] + "</extra>",
        ))
    fig.update_layout(height=height, hovermode="x unified", legend={"y": 1.1})
    fig.update_xaxes(title="Days ahead", dtick=1)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def weekday_chart(df: pd.DataFrame, height: int = 260) -> go.Figure:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    regular = df[(df["onpromotion"] == 0) & (df["is_holiday"] == 0)]
    by_dow = regular.groupby(regular["date"].dt.dayofweek)["units_sold"].mean().reindex(range(7))
    peak = int(by_dow.idxmax())
    fig = go.Figure(go.Bar(
        x=names, y=by_dow.values,
        marker={"color": [GOLD if i == peak else GOLD_SOFT for i in range(7)]},
        hovertemplate="%{y:,.0f} units on an average %{x}<extra></extra>",
    ))
    fig.update_layout(height=height, hovermode="closest")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def seasonality_heatmap(df: pd.DataFrame, height: int = 260) -> go.Figure:
    d = df.assign(month=df["date"].dt.month, dow=df["date"].dt.dayofweek)
    grid = d.pivot_table(index="dow", columns="month", values="units_sold", aggfunc="mean")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[months[m - 1] for m in grid.columns],
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        colorscale=[[0, "#1A1815"], [0.5, GOLD_SOFT], [1, "#F0D488"]],
        hovertemplate="%{y} in %{x}: %{z:,.0f} units<extra></extra>",
        colorbar={"thickness": 8, "outlinewidth": 0, "tickfont": {"color": ASH}},
        xgap=2, ygap=2,
    ))
    fig.update_layout(height=height, hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, showspikes=False)
    return fig


def history_chart(df: pd.DataFrame, height: int = 380) -> go.Figure:
    smooth = df["units_sold"].rolling(28, min_periods=7).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["units_sold"], mode="lines", name="Daily units",
        line={"color": _rgba(GOLD, 0.35), "width": 1},
        hovertemplate="%{y:,.0f} units<extra>Daily</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=smooth, mode="lines", name="28-day average",
        line={"color": GOLD, "width": 2.2},
        hovertemplate="%{y:,.0f}<extra>28-day average</extra>",
    ))
    hol = df[df["is_holiday"] == 1]
    fig.add_trace(go.Scatter(
        x=hol["date"], y=hol["units_sold"], mode="markers", name="Holiday",
        marker={"color": BONE, "size": 5, "symbol": "diamond"},
        hovertemplate="Holiday<extra></extra>",
    ))
    fig.update_layout(height=height, legend={"y": 1.06})
    fig.update_xaxes(rangeslider={"visible": False})
    fig.update_yaxes(tickformat=",.0f")
    return fig


def stock_projection(plan: pd.DataFrame, reorder_point: float, height: int = 340) -> go.Figure:
    """Expected stock if the plan is followed, against doing nothing."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plan["date"], y=plan["stock_no_order"], mode="lines", name="If nothing is ordered",
        line={"color": EMBER, "width": 1.8, "dash": "dash"},
        hovertemplate="%{y:,.0f} units<extra>No orders</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=plan["date"], y=plan["stock"], mode="lines", name="Following the plan",
        line={"color": GOLD, "width": 2.4, "shape": "hv"},
        hovertemplate="%{y:,.0f} units<extra>With plan</extra>",
    ))
    orders = plan[plan["order_placed"] > 0]
    fig.add_trace(go.Scatter(
        x=orders["date"], y=orders["stock"], mode="markers", name="Order placed",
        marker={"color": BONE, "size": 8, "symbol": "circle-open", "line": {"width": 2}},
        customdata=orders["order_placed"],
        hovertemplate="Order %{customdata:,.0f} units<extra></extra>",
    ))
    deliveries = plan[plan["delivery"] > 0]
    fig.add_trace(go.Scatter(
        x=deliveries["date"], y=deliveries["stock"], mode="markers", name="Delivery arrives",
        marker={"color": GOLD, "size": 8, "symbol": "triangle-up"},
        customdata=deliveries["delivery"],
        hovertemplate="Delivery of %{customdata:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=reorder_point, line={"color": ASH, "width": 1, "dash": "dot"},
                  annotation_text="Reorder point", annotation_font_color=ASH,
                  annotation_position="top left")
    fig.update_layout(height=height, legend={"y": 1.08}, hovermode="x unified")
    fig.update_yaxes(rangemode="tozero", tickformat=",.0f")
    return fig


def tradeoff_chart(tradeoff: pd.DataFrame, current: float, height: int = 340) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tradeoff["fill_rate"], y=tradeoff["total_cost"], mode="lines+markers",
        line={"color": GOLD, "width": 2}, marker={"size": 7, "color": GOLD},
        customdata=tradeoff[["target_service_level", "stockout_days", "avg_inventory"]],
        hovertemplate=(
            "Target %{customdata[0]:.0%}<br>Fill rate %{x:.1%}<br>Total cost %{y:,.0f}"
            "<br>Stock-out days %{customdata[1]}<br>Avg stock %{customdata[2]:,.0f}<extra></extra>"
        ),
    ))
    sel = tradeoff.iloc[(tradeoff["target_service_level"] - current).abs().argsort()[:1]]
    fig.add_trace(go.Scatter(
        x=sel["fill_rate"], y=sel["total_cost"], mode="markers",
        marker={"size": 16, "color": "rgba(0,0,0,0)", "line": {"color": EMBER, "width": 2}},
        hoverinfo="skip",
    ))
    fig.add_annotation(
        x=sel["fill_rate"].iloc[0], y=sel["total_cost"].iloc[0],
        text=f"Your target ({current:.0%})", showarrow=False, yshift=18,
        font={"color": EMBER, "size": 11},
    )
    for _, row in tradeoff.iloc[[0, -1]].iterrows():
        fig.add_annotation(
            x=row["fill_rate"], y=row["total_cost"], text=f"{row['target_service_level']:.0%} target",
            showarrow=False, yshift=-16, font={"color": ASH, "size": 10},
        )
    fig.update_layout(height=height, showlegend=False, hovermode="closest")
    fig.update_xaxes(title="Achieved fill rate", tickformat=".0%")
    fig.update_yaxes(title="Total cost over the period", tickformat=",.0f", rangemode="normal")
    return fig


def simulation_chart(sim: pd.DataFrame, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim["date"], y=sim["inventory_eod"], mode="lines", name="Stock at close",
        line={"color": GOLD, "width": 1.6}, fill="tozeroy", fillcolor=_rgba(GOLD, 0.08),
        hovertemplate="%{y:,.0f} units<extra>Stock</extra>",
    ))
    lost = sim[sim["lost_sales"] > 0]
    fig.add_trace(go.Bar(
        x=lost["date"], y=lost["lost_sales"], name="Lost sales", marker={"color": EMBER},
        hovertemplate="%{y:,.0f} units lost<extra></extra>",
    ))
    fig.update_layout(height=height, legend={"y": 1.1}, bargap=0)
    fig.update_yaxes(tickformat=",.0f")
    return fig
