"""
streamlit_app/pages/edge_decomposition_tab.py
===============================================
EDGE DECOMPOSITION — Atomic-level performance attribution.

Decomposes total system performance into exactly 5 components:
  1. Predictive Edge — Brier score, log loss, calibration
  2. Informational Edge — Signal timing vs line movement
  3. Market Inefficiency — CLV analysis
  4. Execution Edge — Slippage tracking
  5. Structural Edge — Correlation, Kelly, diversification

Final verdict: "Which component is doing the heavy lifting — and which are illusions?"
"""

from __future__ import annotations

import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

log = logging.getLogger(__name__)

# Theme constants
_GREEN = "#00FFB2"
_RED = "#FF3358"
_YELLOW = "#FFB800"
_MUTED = "#4A607A"
_BG = "rgba(0,0,0,0)"


def _get_report():
    """Generate or retrieve cached edge decomposition report."""
    from edge_analysis.report import EdgeReportGenerator
    gen = EdgeReportGenerator(sport="nba")
    return gen.generate(), gen


def render():
    """Render the Edge Decomposition tab."""
    st.markdown("""
    <h2 style='font-family: Chakra Petch, monospace; color: #00FFB2; margin-bottom: 0;
    font-size: 1.3rem; letter-spacing: 0.1em;'>
    EDGE DECOMPOSITION — ATOMIC LEVEL
    </h2>
    <p style='font-family: Fira Code, monospace; color: #4A607A; font-size: 0.65rem;
    margin-top: 0.2rem;'>
    Which component is doing the heavy lifting — and which are illusions?
    </p>
    """, unsafe_allow_html=True)

    tab_overview, tab_predictive, tab_market, tab_execution, tab_structural, tab_verdict = st.tabs([
        "OVERVIEW", "PREDICTIVE", "CLV / MARKET", "EXECUTION", "STRUCTURAL", "VERDICT",
    ])

    try:
        report, gen = _get_report()
    except Exception as exc:
        st.error(f"Edge decomposition unavailable: {exc}")
        _render_empty_state()
        return

    if report.total_bets < 10:
        _render_empty_state()
        return

    with tab_overview:
        _render_overview(report)

    with tab_predictive:
        _render_predictive(report)

    with tab_market:
        _render_market(report)

    with tab_execution:
        _render_execution(report)

    with tab_structural:
        _render_structural(report)

    with tab_verdict:
        _render_verdict(report)


# ── OVERVIEW ─────────────────────────────────────────────────────────────────

def _render_overview(report):
    """Top-level attribution pie chart and metrics."""
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = _GREEN if report.total_roi > 0 else _RED
        st.metric("TOTAL ROI", f"{report.total_roi:+.2f}%")
    with col2:
        st.metric("TOTAL BETS", f"{report.total_bets}")
    with col3:
        st.metric("TOTAL P&L", f"${report.total_pnl:+,.2f}")
    with col4:
        st.metric("HEAVY LIFTER", report.heavy_lifter)

    st.markdown("---")

    # Attribution chart — donut
    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        st.markdown("##### Edge Attribution")
        labels = ["Predictive", "Informational", "Market Inefficiency", "Execution", "Structural"]
        values = [
            abs(report.predictive_pct),
            abs(report.informational_pct),
            abs(report.market_pct),
            abs(report.execution_pct),
            abs(report.structural_pct),
        ]
        colors_list = []
        for comp_name in ["predictive", "informational", "market_inefficiency", "execution", "structural"]:
            comp = getattr(report, comp_name, None)
            if comp and comp.is_positive and comp.is_significant:
                colors_list.append(_GREEN)
            elif comp and comp.is_positive:
                colors_list.append(_YELLOW)
            else:
                colors_list.append(_RED)

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors_list),
            textinfo="label+percent",
            textfont=dict(size=11, family="Fira Code, monospace"),
        )])
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{report.total_roi:+.1f}%</b><br>ROI",
                x=0.5, y=0.5, font_size=16, showarrow=False,
                font=dict(family="Chakra Petch, monospace", color=_GREEN if report.total_roi > 0 else _RED),
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("##### Component Status")
        rows = []
        for comp_name, label in [
            ("predictive", "Predictive"),
            ("informational", "Informational"),
            ("market_inefficiency", "Market Inefficiency"),
            ("execution", "Execution"),
            ("structural", "Structural"),
        ]:
            comp = getattr(report, comp_name, None)
            if comp:
                status = "REAL" if comp.is_positive and comp.is_significant else (
                    "POSSIBLE" if comp.is_positive else "ILLUSION"
                )
                rows.append({
                    "Component": label,
                    "Share": f"{comp.edge_pct_of_roi:+.1f}%",
                    "Status": status,
                    "P-Value": f"{comp.p_value:.4f}",
                    "Significant": comp.is_significant,
                    "N": comp.sample_size,
                })

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Illusions warning
        if report.illusions:
            st.warning(f"Illusions detected: **{', '.join(report.illusions)}**")
        else:
            st.success("All components show genuine contribution")

    # Bar chart — attribution with sign
    st.markdown("##### Attribution Breakdown (signed)")
    fig_bar = go.Figure()
    names = ["Predictive", "Informational", "Market\nInefficiency", "Execution", "Structural"]
    vals = [report.predictive_pct, report.informational_pct, report.market_pct,
            report.execution_pct, report.structural_pct]
    bar_colors = [_GREEN if v > 0 else _RED for v in vals]
    fig_bar.add_trace(go.Bar(
        x=names, y=vals,
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in vals],
        textposition="outside",
    ))
    fig_bar.add_hline(y=0, line_dash="dash", line_color=_MUTED, opacity=0.5)
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        yaxis_title="% of Total Edge",
        height=320,
        margin=dict(l=40, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── PREDICTIVE ───────────────────────────────────────────────────────────────

def _render_predictive(report):
    """Predictive edge: Brier, log loss, calibration curve."""
    comp = report.predictive
    if not comp:
        st.info("Predictive edge analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    # Scoring metrics
    col1, col2, col3, col4 = st.columns(4)
    d = comp.details
    with col1:
        st.metric("Brier Score (Model)", f"{d.get('brier_model', 0):.4f}")
    with col2:
        st.metric("Brier Score (Market)", f"{d.get('brier_market', 0):.4f}")
    with col3:
        st.metric("Brier Skill", f"{d.get('brier_skill', 0):.1%}")
    with col4:
        st.metric("Mean Cal Error", f"{d.get('mean_calibration_error', 0):.1%}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Log Loss (Model)", f"{d.get('logloss_model', 0):.4f}")
    with col6:
        st.metric("Log Loss (Market)", f"{d.get('logloss_market', 0):.4f}")
    with col7:
        st.metric("Log Loss Skill", f"{d.get('logloss_skill', 0):.1%}")
    with col8:
        st.metric("P-Value", f"{comp.p_value:.4f}")

    st.markdown("---")

    # Calibration curve
    cal_data = d.get("calibration_curve", [])
    if cal_data:
        st.markdown("##### Calibration Curve")
        predicted = [p["predicted"] for p in cal_data]
        actual = [p["actual"] for p in cal_data]
        n_bets = [p["n"] for p in cal_data]

        fig = go.Figure()
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0.4, 0.95], y=[0.4, 0.95],
            mode="lines", name="Perfect",
            line=dict(color=_MUTED, dash="dash", width=1),
        ))
        # Model calibration
        fig.add_trace(go.Scatter(
            x=predicted, y=actual,
            mode="lines+markers", name="Model",
            line=dict(color=_GREEN, width=2),
            marker=dict(size=[max(4, min(15, n / 3)) for n in n_bets]),
            text=[f"n={n}" for n in n_bets],
            hovertemplate="Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            height=400,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(range=[0.35, 1.0]),
            yaxis=dict(range=[0.35, 1.0]),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Calibration table
        df_cal = pd.DataFrame(cal_data)
        st.dataframe(df_cal, use_container_width=True, hide_index=True)


# ── MARKET / CLV ─────────────────────────────────────────────────────────────

def _render_market(report):
    """Market inefficiency edge: CLV analysis."""
    comp = report.market_inefficiency
    if not comp:
        st.info("Market inefficiency analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    d = comp.details
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        clv = d.get("avg_clv_cents", 0)
        st.metric("Avg CLV (cents)", f"{clv:+.2f}")
    with col2:
        st.metric("Beat Close Rate", f"{d.get('beat_close_rate', 0):.1%}")
    with col3:
        st.metric("Median CLV", f"{d.get('median_clv_cents', 0):+.2f}")
    with col4:
        st.metric("P-Value", f"{comp.p_value:.4f}")

    st.markdown("---")

    # CLV distribution histogram
    clv_dist = d.get("clv_distribution", [])
    if clv_dist:
        st.markdown("##### CLV Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=clv_dist,
            nbinsx=40,
            marker_color=_GREEN,
            opacity=0.7,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=_RED, line_width=2)
        avg = d.get("avg_clv_cents", 0)
        fig.add_vline(x=avg, line_dash="dot", line_color=_YELLOW, line_width=2,
                      annotation_text=f"Avg: {avg:+.2f}")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            xaxis_title="CLV (cents)",
            yaxis_title="Count",
            height=350,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # CLV by market type
    clv_by_market = d.get("clv_by_market", {})
    if clv_by_market:
        st.markdown("##### CLV by Market Type")
        mkt_rows = []
        for seg, metrics in sorted(clv_by_market.items(), key=lambda x: x[1]["avg_clv"], reverse=True):
            mkt_rows.append({
                "Market": seg,
                "Avg CLV": f"{metrics['avg_clv']:+.2f}",
                "Beat Close %": f"{metrics['beat_close_pct']:.1%}",
                "N Bets": metrics["n_bets"],
                "Positive": metrics["is_positive"],
            })
        df_mkt = pd.DataFrame(mkt_rows)
        st.dataframe(df_mkt, use_container_width=True, hide_index=True)

        # Bar chart
        markets = [r["Market"] for r in mkt_rows]
        clv_vals = [clv_by_market[m]["avg_clv"] for m in markets]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=markets, y=clv_vals,
            marker_color=[_GREEN if v > 0 else _RED for v in clv_vals],
        ))
        fig_bar.add_hline(y=0, line_dash="dash", line_color=_MUTED)
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            yaxis_title="Avg CLV (cents)",
            height=300,
            margin=dict(l=40, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ── EXECUTION ────────────────────────────────────────────────────────────────

def _render_execution(report):
    """Execution edge: slippage tracking."""
    comp = report.execution
    if not comp:
        st.info("Execution analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    d = comp.details
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        slip = d.get("avg_slippage_cents", 0)
        st.metric("Avg Slippage", f"{slip:+.2f} cents")
    with col2:
        st.metric("Price Improved %", f"{d.get('pct_price_improved', 0):.1%}")
    with col3:
        st.metric("Capture Rate", f"{d.get('avg_capture_rate', 0):.1%}")
    with col4:
        st.metric("Total Drag", f"{d.get('total_drag_cents', 0):+.1f} cents")

    # Cost by market
    cost_by = d.get("cost_by_market", {})
    if cost_by:
        st.markdown("##### Execution Cost by Market")
        rows = []
        for seg, metrics in sorted(cost_by.items(), key=lambda x: x[1]["avg_slippage"]):
            rows.append({
                "Market": seg,
                "Avg Slippage": f"{metrics['avg_slippage']:+.3f}",
                "Price Improved %": f"{metrics['pct_improved']:.1%}",
                "N Bets": metrics["n_bets"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── STRUCTURAL ───────────────────────────────────────────────────────────────

def _render_structural(report):
    """Structural edge: correlation, Kelly, diversification."""
    comp = report.structural
    if not comp:
        st.info("Structural analysis not available")
        return

    st.markdown(f"**Verdict:** {comp.verdict}")

    d = comp.details
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Correlation (rho)", f"{d.get('correlation_rho', 0):.4f}")
    with col2:
        st.metric("Variance Ratio", f"{d.get('variance_ratio', 1):.3f}")
    with col3:
        st.metric("Sharpe Ratio", f"{d.get('annualized_sharpe', 0):.3f}")
    with col4:
        st.metric("Max Drawdown", f"${d.get('max_drawdown', 0):.2f}")

    st.markdown("---")

    # Kelly analysis
    kelly = d.get("kelly", {})
    if kelly.get("sufficient_data"):
        st.markdown("##### Kelly Criterion Adherence")
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Avg Kelly Fraction", f"{kelly.get('avg_kelly_fraction', 0):.4f}")
        with col_k2:
            st.metric("Kelly Efficiency", f"{kelly.get('kelly_efficiency', 0):.1%}")
        with col_k3:
            st.metric("Actual Growth", f"{kelly.get('actual_growth_rate', 0):.6f}")
        with col_k4:
            st.metric("Full Kelly Growth", f"{kelly.get('full_kelly_growth_rate', 0):.6f}")

    # Diversification
    div = d.get("diversification", {})
    if div:
        st.markdown("##### Portfolio Diversification")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Unique Markets", div.get("n_unique_markets", 0))
        with col_d2:
            st.metric("Unique Players", div.get("n_unique_players", 0))
        with col_d3:
            hhi = div.get("hhi_market", 1.0)
            label = "Excellent" if hhi < 0.15 else ("Acceptable" if hhi < 0.3 else "Concentrated")
            st.metric("Market HHI", f"{hhi:.4f} ({label})")

        # Market distribution
        mkt_dist = div.get("market_distribution", {})
        if mkt_dist:
            fig = go.Figure(data=[go.Bar(
                x=list(mkt_dist.keys()),
                y=list(mkt_dist.values()),
                marker_color=_GREEN,
            )])
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor=_BG, plot_bgcolor=_BG,
                yaxis_title="Bet Count",
                height=280,
                margin=dict(l=40, r=20, t=20, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── VERDICT ──────────────────────────────────────────────────────────────────

def _render_verdict(report):
    """Final verdict: the definitive answer."""
    st.markdown("##### FINAL VERDICT")

    # Status banner
    if report.market_inefficiency and report.market_inefficiency.is_positive and report.market_inefficiency.is_significant:
        color = _GREEN
        banner = "GENUINE EDGE DETECTED"
    elif report.total_roi > 0:
        color = _YELLOW
        banner = "PROFITABLE BUT UNCONFIRMED"
    else:
        color = _RED
        banner = "NO CONFIRMED EDGE"

    st.markdown(f"""
    <div style='text-align: center; padding: 20px; border: 2px solid {color};
    border-radius: 12px; margin: 10px 0;'>
        <div style='font-size: 1.8rem; font-weight: bold; color: {color};
        font-family: Chakra Petch, monospace;'>{banner}</div>
        <div style='font-size: 0.85rem; color: {_MUTED};
        font-family: Fira Code, monospace; margin-top: 8px;'>
        Heavy Lifter: <span style="color: {_GREEN}">{report.heavy_lifter}</span>
        &nbsp;|&nbsp;
        Illusions: <span style="color: {_RED}">{', '.join(report.illusions) if report.illusions else 'None'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Full text report
    st.code(report.verdict, language="text")

    # Individual component verdicts
    st.markdown("---")
    st.markdown("##### Component Verdicts")
    for comp_name, label in [
        ("predictive", "1. PREDICTIVE EDGE"),
        ("informational", "2. INFORMATIONAL EDGE"),
        ("market_inefficiency", "3. MARKET INEFFICIENCY"),
        ("execution", "4. EXECUTION EDGE"),
        ("structural", "5. STRUCTURAL EDGE"),
    ]:
        comp = getattr(report, comp_name, None)
        if comp:
            status = "REAL" if comp.is_positive and comp.is_significant else (
                "POSSIBLE" if comp.is_positive else "ILLUSION"
            )
            icon_color = _GREEN if status == "REAL" else (_YELLOW if status == "POSSIBLE" else _RED)
            st.markdown(f"""
            <div style='margin: 8px 0; padding: 10px 15px; border-left: 3px solid {icon_color};
            background: rgba(255,255,255,0.02); border-radius: 0 8px 8px 0;'>
                <div style='font-family: Chakra Petch, monospace; font-size: 0.8rem;
                color: {icon_color}; font-weight: 600;'>
                {label} — {status} ({comp.edge_pct_of_roi:+.1f}% of ROI, p={comp.p_value:.4f})
                </div>
                <div style='font-family: Fira Code, monospace; font-size: 0.7rem;
                color: #8A9AAA; margin-top: 4px;'>
                {comp.verdict}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Empty state ──────────────────────────────────────────────────────────────

def _render_empty_state():
    """Shown when there's not enough data."""
    st.markdown("""
    <div style='text-align: center; padding: 40px; border: 1px dashed #1E2D3D;
    border-radius: 12px; margin: 20px 0;'>
        <div style='font-size: 1.2rem; color: #4A607A; font-family: Chakra Petch, monospace;
        margin-bottom: 10px;'>EDGE DECOMPOSITION INITIALIZING</div>
        <div style='font-size: 0.75rem; color: #2A3A4A; font-family: Fira Code, monospace;'>
            The edge decomposition system needs at least 10 settled bets to produce analysis.<br><br>
            <b>5 Components Analyzed:</b><br>
            1. Predictive Edge — Brier score & calibration vs market<br>
            2. Informational Edge — Signal timing vs line movement<br>
            3. Market Inefficiency — CLV (closing line value)<br>
            4. Execution Edge — Slippage tracking<br>
            5. Structural Edge — Correlation, Kelly sizing, diversification<br><br>
            Once enough bets settle, this page will show which components carry real weight
            and which are illusions.
        </div>
    </div>
    """, unsafe_allow_html=True)
