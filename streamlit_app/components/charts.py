"""
streamlit_app/components/charts.py
==================================
Reusable chart components.  All functions take data as input -- no DB calls,
no computation.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Safe import: plotly is optional but preferred
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# PnL curve
# ---------------------------------------------------------------------------

def pnl_curve(daily_pnl: list[dict], title: str = "Cumulative P&L") -> None:
    """Render a cumulative P&L line chart.

    Parameters
    ----------
    daily_pnl : list[dict]
        Each dict has ``date`` (str) and ``pnl`` (float).
    """
    if not daily_pnl:
        st.info("No P&L data to display.")
        return

    dates = [d["date"] for d in daily_pnl]
    cum_pnl = []
    running = 0.0
    for d in daily_pnl:
        running += float(d.get("pnl", 0))
        cum_pnl.append(round(running, 2))

    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=cum_pnl, mode="lines+markers",
            line=dict(color="#00FFB2", width=2),
            marker=dict(size=4),
            name="Cum P&L",
        ))
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="#04080F",
            plot_bgcolor="#04080F",
            font=dict(family="Fira Code, monospace", size=11),
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import pandas as pd
        df = pd.DataFrame({"date": dates, "cumulative_pnl": cum_pnl})
        st.line_chart(df.set_index("date")["cumulative_pnl"])


# ---------------------------------------------------------------------------
# Calibration chart
# ---------------------------------------------------------------------------

def calibration_chart(buckets: list[dict], title: str = "Model Calibration") -> None:
    """Render a predicted-vs-actual calibration plot.

    Parameters
    ----------
    buckets : list[dict]
        Each dict has ``predicted_avg`` and ``actual_rate``.
    """
    if not buckets:
        st.info("No calibration data available.")
        return

    predicted = [float(b["predicted_avg"]) for b in buckets]
    actual = [float(b["actual_rate"]) for b in buckets]
    labels = [b.get("bucket_label", f"{b['predicted_avg']:.0%}") for b in buckets]
    n_bets = [int(b.get("n_bets", 0)) for b in buckets]

    if _HAS_PLOTLY:
        fig = go.Figure()
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#2A4060", dash="dash", width=1),
            name="Perfect",
            showlegend=True,
        ))
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=predicted, y=actual, mode="lines+markers",
            line=dict(color="#00FFB2", width=2),
            marker=dict(size=8, color="#00FFB2"),
            text=[f"{lbl} (n={n})" for lbl, n in zip(labels, n_bets)],
            hovertemplate="Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}",
            name="Model",
        ))
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="#04080F",
            plot_bgcolor="#04080F",
            font=dict(family="Fira Code, monospace", size=11),
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import pandas as pd
        df = pd.DataFrame({"predicted": predicted, "actual": actual})
        st.line_chart(df.set_index("predicted")["actual"])


# ---------------------------------------------------------------------------
# CLV distribution chart
# ---------------------------------------------------------------------------

def clv_chart(clv_data: list[dict], title: str = "CLV Distribution") -> None:
    """Render a CLV histogram or bar chart.

    Parameters
    ----------
    clv_data : list[dict]
        Each dict has ``clv_cents`` (float).
    """
    if not clv_data:
        st.info("No CLV data to display.")
        return

    cents = [float(d.get("clv_cents", 0)) for d in clv_data]

    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=cents, nbinsx=30,
            marker_color="#00FFB2",
            opacity=0.8,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#FF3358", line_width=1)
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="#04080F",
            plot_bgcolor="#04080F",
            font=dict(family="Fira Code, monospace", size=11),
            xaxis_title="CLV (cents)",
            yaxis_title="Count",
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import pandas as pd
        st.bar_chart(pd.Series(cents, name="CLV cents"))


# ---------------------------------------------------------------------------
# Edge distribution bar chart
# ---------------------------------------------------------------------------

def edge_distribution_chart(edge_dist: dict, title: str = "Edge Distribution") -> None:
    """Render edge category distribution as a horizontal bar chart.

    Parameters
    ----------
    edge_dist : dict
        Keys like ``no_edge``, ``lean``, ``solid``, ``strong`` with int counts.
    """
    if not edge_dist:
        st.info("No edge distribution data.")
        return

    categories = list(edge_dist.keys())
    counts = list(edge_dist.values())
    colors = {"no_edge": "#FF3358", "lean": "#FFB800", "solid": "#00AAFF", "strong": "#00FFB2"}

    if _HAS_PLOTLY:
        bar_colors = [colors.get(c, "#4A607A") for c in categories]
        fig = go.Figure(go.Bar(
            x=counts, y=categories, orientation="h",
            marker_color=bar_colors,
            text=counts, textposition="auto",
        ))
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="#04080F",
            plot_bgcolor="#04080F",
            font=dict(family="Fira Code, monospace", size=11),
            height=250,
            margin=dict(l=80, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import pandas as pd
        st.bar_chart(pd.Series(counts, index=categories, name="Count"))


# ---------------------------------------------------------------------------
# Bankroll history sparkline
# ---------------------------------------------------------------------------

def bankroll_sparkline(history: list[dict]) -> None:
    """Render a mini bankroll history chart.

    Parameters
    ----------
    history : list[dict]
        Each dict has ``amount`` (float) and ``ts`` (str).
    """
    if not history or len(history) < 2:
        return

    amounts = [float(h["amount"]) for h in history]

    if _HAS_PLOTLY:
        fig = go.Figure(go.Scatter(
            y=amounts, mode="lines",
            line=dict(color="#00FFB2", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0, 255, 178, 0.05)",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=80,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        import pandas as pd
        st.line_chart(pd.Series(amounts, name="Bankroll"))


# ---------------------------------------------------------------------------
# Generic metric card (HTML)
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str | None = None,
                color: str = "#00FFB2") -> str:
    """Return an HTML string for a styled metric card."""
    delta_html = ""
    if delta is not None:
        d_color = "#00FFB2" if not delta.startswith("-") else "#FF3358"
        delta_html = (
            f"<div style='font-family:Fira Code,monospace;font-size:0.6rem;"
            f"color:{d_color};margin-top:2px;'>{delta}</div>"
        )
    return f"""
<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;
            padding:0.5rem 0.7rem;text-align:center;'>
    <div style='font-family:Fira Code,monospace;font-size:0.55rem;color:#2A6080;
                letter-spacing:0.1em;text-transform:uppercase;'>{label}</div>
    <div style='font-family:Chakra Petch,monospace;font-size:1.1rem;font-weight:700;
                color:{color};margin-top:2px;'>{value}</div>
    {delta_html}
</div>
"""
