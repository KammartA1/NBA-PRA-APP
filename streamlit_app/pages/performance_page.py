"""
streamlit_app/pages/performance_page.py
=======================================
PAGE 3: PERFORMANCE -- Bloomberg Terminal-style performance dashboard
for the NBA Prop Alpha Engine.

Four charts in a 2x2 grid:
  - CLV Over Time
  - ROI Curve
  - Drawdown
  - Calibration

Plus a summary metric row beneath.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from services.ui_bridge import UIBridge
from streamlit_app.design import metric_card, card_container, PLOTLY_TEMPLATE, COLORS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHART_HEIGHT = 300
_CHART_CONFIG = {"displayModeBar": False}
_FONT = dict(family="JetBrains Mono, monospace", color=COLORS["text_secondary"])
_TRANSPARENT = "rgba(0,0,0,0)"

_HEADER_STYLE = (
    "font-family:JetBrains Mono;font-size:13px;color:#8B8B96;"
    "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:8px;"
)


def _chart_header(label: str) -> str:
    return f"<div style='{_HEADER_STYLE}'>{label}</div>"


def _base_figure() -> go.Figure:
    """Create a Plotly figure with the shared template and transparent bg."""
    fig = go.Figure()
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font=_FONT,
        height=_CHART_HEIGHT,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    return fig


def _no_data_annotation(fig: go.Figure) -> go.Figure:
    """Add a centered 'NO DATA' annotation to an empty chart."""
    fig.add_annotation(
        text="NO DATA",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(
            family="JetBrains Mono, monospace",
            size=18,
            color=COLORS["text_secondary"],
        ),
        opacity=0.5,
    )
    return fig


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _build_clv_chart(clv_history: list[dict]) -> go.Figure:
    fig = _base_figure()

    if not clv_history:
        return _no_data_annotation(fig)

    x = [d["bet_num"] for d in clv_history]
    y = [d["cumulative_clv"] for d in clv_history]

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        name="Cumulative CLV",
    ))

    # Zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color=COLORS["border"],
        line_width=1,
    )

    # Rolling average annotation (last 20 bets or all if fewer)
    if len(y) >= 2:
        window = min(20, len(y))
        rolling_avg = sum(y[-window:]) / window
        fig.add_annotation(
            x=x[-1], y=y[-1],
            text=f"Avg(20): {rolling_avg:.1f}\u00a2",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLORS["text_secondary"],
            font=dict(
                family="JetBrains Mono, monospace",
                size=10,
                color=COLORS["text_secondary"],
            ),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            borderwidth=1,
            borderpad=4,
        )

    fig.update_xaxes(title_text="Bet #")
    fig.update_yaxes(title_text="Cumulative CLV (\u00a2)")
    return fig


def _build_roi_chart(roi_history: list[dict]) -> go.Figure:
    fig = _base_figure()

    if not roi_history:
        return _no_data_annotation(fig)

    x = [d["bet_num"] for d in roi_history]
    y = [d["cumulative_roi"] for d in roi_history]

    latest_roi = y[-1] if y else 0
    line_color = COLORS["green"] if latest_roi >= 0 else COLORS["red"]

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=line_color, width=2),
        name="Cumulative ROI",
    ))

    # Confidence band if enough data (30+ bets)
    if len(y) >= 30:
        import statistics
        std = statistics.stdev(y)
        upper = [v + std for v in y]
        lower = [v - std for v in y]

        fig.add_trace(go.Scatter(
            x=x, y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(139,139,150,0.12)",
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_xaxes(title_text="Bet #")
    fig.update_yaxes(title_text="Cumulative ROI (%)")
    return fig


def _build_drawdown_chart(drawdown_history: list[dict]) -> go.Figure:
    fig = _base_figure()

    if not drawdown_history:
        return _no_data_annotation(fig)

    x = [d["bet_num"] for d in drawdown_history]
    y = [d["drawdown_pct"] for d in drawdown_history]

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=COLORS["red"], width=2),
        fill="tozeroy",
        fillcolor="rgba(255,71,87,0.20)",
        name="Drawdown",
    ))

    # Max acceptable drawdown line at -25%
    fig.add_hline(
        y=-25,
        line_dash="dash",
        line_color=COLORS["amber"],
        line_width=1,
        annotation_text="Max DD -25%",
        annotation_font=dict(
            family="JetBrains Mono, monospace",
            size=10,
            color=COLORS["amber"],
        ),
        annotation_position="top left",
    )

    fig.update_xaxes(title_text="Bet #")
    fig.update_yaxes(title_text="Drawdown (%)")
    return fig


def _build_calibration_chart(calibration_data: list[dict]) -> go.Figure:
    fig = _base_figure()

    if not calibration_data:
        return _no_data_annotation(fig)

    predicted = [d["predicted"] for d in calibration_data]
    actual = [d["actual"] for d in calibration_data]
    counts = [d["count"] for d in calibration_data]

    # Normalize dot sizes: min 6, max 20
    max_count = max(counts) if counts else 1
    min_count = min(counts) if counts else 1
    range_count = max_count - min_count if max_count != min_count else 1
    sizes = [6 + 14 * ((c - min_count) / range_count) for c in counts]

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=COLORS["text_secondary"], width=1, dash="dash"),
        name="Perfect",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Actual calibration data
    fig.add_trace(go.Scatter(
        x=predicted,
        y=actual,
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(
            color=COLORS["blue"],
            size=sizes,
            line=dict(width=1, color=COLORS["surface"]),
        ),
        name="Actual",
        text=[f"n={c}" for c in counts],
        hovertemplate="Predicted: %{x:.0%}<br>Actual: %{y:.0%}<br>%{text}<extra></extra>",
    ))

    fig.update_xaxes(title_text="Predicted Probability", tickformat=".0%")
    fig.update_yaxes(title_text="Actual Win Rate", tickformat=".0%")
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Performance page."""
    bridge = UIBridge()
    data = bridge.get_performance_data(days=90)

    clv_history = data.get("clv_history", [])
    roi_history = data.get("roi_history", [])
    drawdown_history = data.get("drawdown_history", [])
    calibration_data = data.get("calibration_data", [])
    summary = data.get("summary", {})

    # Check if all data lists are empty
    all_empty = (
        not clv_history
        and not roi_history
        and not drawdown_history
        and not calibration_data
    )

    if all_empty:
        st.markdown(
            "<div style='text-align:center;color:#8B8B96;padding:60px 0;"
            "font-family:JetBrains Mono,monospace;font-size:14px;'>"
            "No performance data available. Place bets to see results."
            "</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------
    # 2x2 Chart Grid -- Top Row
    # -------------------------------------------------------------------
    top_left, top_right = st.columns(2)

    with top_left:
        st.markdown(
            card_container(_chart_header("CLV OVER TIME")),
            unsafe_allow_html=True,
        )
        fig_clv = _build_clv_chart(clv_history)
        st.plotly_chart(fig_clv, use_container_width=True, config=_CHART_CONFIG)

    with top_right:
        st.markdown(
            card_container(_chart_header("ROI CURVE")),
            unsafe_allow_html=True,
        )
        fig_roi = _build_roi_chart(roi_history)
        st.plotly_chart(fig_roi, use_container_width=True, config=_CHART_CONFIG)

    # -------------------------------------------------------------------
    # 2x2 Chart Grid -- Bottom Row
    # -------------------------------------------------------------------
    bot_left, bot_right = st.columns(2)

    with bot_left:
        st.markdown(
            card_container(_chart_header("DRAWDOWN")),
            unsafe_allow_html=True,
        )
        fig_dd = _build_drawdown_chart(drawdown_history)
        st.plotly_chart(fig_dd, use_container_width=True, config=_CHART_CONFIG)

    with bot_right:
        st.markdown(
            card_container(_chart_header("CALIBRATION")),
            unsafe_allow_html=True,
        )
        fig_cal = _build_calibration_chart(calibration_data)
        st.plotly_chart(fig_cal, use_container_width=True, config=_CHART_CONFIG)

    # -------------------------------------------------------------------
    # Summary Row
    # -------------------------------------------------------------------
    total_bets = summary.get("total_bets", 0)
    win_rate = summary.get("win_rate", 0)
    roi = summary.get("roi", 0)
    avg_clv = summary.get("avg_clv", 0)
    sharpe = summary.get("sharpe", 0)
    max_dd = summary.get("max_drawdown", 0)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown(
            metric_card("Total Bets", str(total_bets)),
            unsafe_allow_html=True,
        )
    with m2:
        wr_color = "green" if win_rate >= 50 else "red"
        st.markdown(
            metric_card("Win Rate", f"{win_rate:.1f}%", delta_color=wr_color),
            unsafe_allow_html=True,
        )
    with m3:
        roi_color = "green" if roi >= 0 else "red"
        st.markdown(
            metric_card("ROI", f"{roi:.1f}%", delta_color=roi_color),
            unsafe_allow_html=True,
        )
    with m4:
        clv_color = "green" if avg_clv >= 0 else "red"
        st.markdown(
            metric_card("Avg CLV", f"{avg_clv:.1f}\u00a2", delta_color=clv_color),
            unsafe_allow_html=True,
        )
    with m5:
        sharpe_color = "green" if sharpe >= 1.0 else ("amber" if sharpe >= 0 else "red")
        st.markdown(
            metric_card("Sharpe", f"{sharpe:.2f}", delta_color=sharpe_color),
            unsafe_allow_html=True,
        )
    with m6:
        dd_color = "green" if max_dd > -10 else ("amber" if max_dd > -25 else "red")
        st.markdown(
            metric_card("Max Drawdown", f"{max_dd:.1f}%", delta_color=dd_color),
            unsafe_allow_html=True,
        )

    # Smaller metric font override for this row
    st.markdown(
        "<style>"
        ".gqe-metric-value { font-size: 32px !important; }"
        "</style>",
        unsafe_allow_html=True,
    )
