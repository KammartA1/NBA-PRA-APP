"""
streamlit_app/pages/attribution_tab.py
=======================================
EDGE ATTRIBUTION tab — answers "Where does profit ACTUALLY come from?"

Displays:
  - Profit decomposition pie chart (4 buckets)
  - Bootstrap distribution histogram with CI bands
  - Counterfactual comparison chart (actual vs closing-line profit)
  - Market inefficiency segment table
  - CLV diagnostic
  - Final verdict

Pure frontend — reads from database via services, delegates all computation
to edge_analysis.attribution.EdgeAttributionEngine.
"""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

import numpy as np

from edge_analysis.attribution import EdgeAttributionEngine, AttributionReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme constants (match app theme)
# ---------------------------------------------------------------------------
_BG = "#04080F"
_FONT = "Fira Code, monospace"
_GREEN = "#00FFB2"
_YELLOW = "#FFB800"
_RED = "#FF3358"
_BLUE = "#00AAFF"
_MUTED = "#4A607A"
_PURPLE = "#A855F7"

_BUCKET_COLORS = {
    "Prediction Edge": _GREEN,
    "CLV/Timing Edge": _BLUE,
    "Market Inefficiency": _YELLOW,
    "Variance": _RED,
}


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def _load_settled_bets() -> list:
    """Load settled bets from the database."""
    try:
        from database.connection import get_session
        from database.models import Bet
        with get_session() as session:
            rows = (
                session.query(Bet)
                .filter(Bet.status.in_(["won", "lost", "push"]))
                .order_by(Bet.timestamp.desc())
                .all()
            )
            bets = []
            for r in rows:
                bets.append({
                    "bet_id": str(r.id),
                    "timestamp": r.timestamp or datetime.now(),
                    "player": r.player or "",
                    "market_type": r.market or r.stat_type or "unknown",
                    "stat_type": getattr(r, "stat_type", "unknown"),
                    "direction": r.direction or "over",
                    "bet_line": float(r.line or 0),
                    "line": float(r.line or 0),
                    "closing_line": float(r.closing_line) if r.closing_line else None,
                    "predicted_prob": float(r.model_prob or 0.5),
                    "model_prob": float(r.model_prob or 0.5),
                    "market_prob_at_bet": float(r.market_prob or 0.5),
                    "market_prob": float(r.market_prob or 0.5),
                    "won": r.status == "won",
                    "stake": float(r.stake or 0),
                    "pnl": float(r.pnl or 0),
                    "odds_decimal": float(r.odds_decimal or 1.909),
                    "odds_american": int(r.odds_american or -110),
                    "model_projection": float(getattr(r, "model_projection", 0) or 0),
                    "confidence_score": float(getattr(r, "confidence_score", 0) or 0),
                    "book": getattr(r, "sportsbook", "unknown") or "unknown",
                })
            return bets
    except Exception as e:
        logger.warning("Could not load bets from DB: %s", e)
        return []


# ---------------------------------------------------------------------------
# Chart renderers
# ---------------------------------------------------------------------------

def _render_pie_chart(report: AttributionReport) -> None:
    """Profit decomposition pie chart."""
    buckets = [
        report.prediction_edge,
        report.clv_timing_edge,
        report.market_inefficiency,
        report.variance_component,
    ]
    labels = [b.label for b in buckets]
    values = [abs(b.dollar_value) for b in buckets]
    colors = [_BUCKET_COLORS.get(b.label, _MUTED) for b in buckets]
    text_info = [
        f"${b.dollar_value:+,.2f} ({b.pct_of_total:+.1f}%)" for b in buckets
    ]

    if _HAS_PLOTLY:
        fig = go.Figure(go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color=_BG, width=2)),
            textinfo="label+percent",
            hovertext=text_info,
            hoverinfo="text",
            hole=0.45,
        ))
        fig.update_layout(
            title="Profit Decomposition",
            template="plotly_dark",
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            font=dict(family=_FONT, size=11),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        # Add center annotation
        fig.add_annotation(
            text=f"${report.total_profit:+,.0f}",
            x=0.5, y=0.5, font=dict(size=18, color=_GREEN, family=_FONT),
            showarrow=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        for b in buckets:
            st.write(f"**{b.label}**: ${b.dollar_value:+,.2f} ({b.pct_of_total:+.1f}%)")


def _render_bootstrap_histogram(report: AttributionReport) -> None:
    """Bootstrap distribution histogram with CI bands."""
    dist = report.bootstrap.profit_distribution
    if not dist:
        st.info("No bootstrap data — need more settled bets.")
        return

    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=dist,
            nbinsx=80,
            marker_color=_BLUE,
            opacity=0.7,
            name="Bootstrap P&L",
        ))
        # Zero line
        fig.add_vline(x=0, line_dash="dash", line_color=_RED, line_width=2,
                      annotation_text="Break Even", annotation_position="top")
        # Actual profit
        fig.add_vline(x=report.total_profit, line_dash="solid",
                      line_color=_GREEN, line_width=2,
                      annotation_text=f"Actual: ${report.total_profit:,.0f}",
                      annotation_position="top right")
        # 95% CI
        fig.add_vrect(
            x0=report.bootstrap.ci_lower_95,
            x1=report.bootstrap.ci_upper_95,
            fillcolor=_BLUE, opacity=0.08,
            line_width=0,
            annotation_text="95% CI",
            annotation_position="top left",
        )

        fig.update_layout(
            title=f"Bootstrap Profit Distribution ({report.bootstrap.n_simulations:,} sims)",
            template="plotly_dark",
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            font=dict(family=_FONT, size=11),
            xaxis_title="Simulated Total Profit ($)",
            yaxis_title="Frequency",
            height=380,
            margin=dict(l=40, r=20, t=50, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"Mean: ${report.bootstrap.mean_profit:,.2f}")
        st.write(f"95% CI: [${report.bootstrap.ci_lower_95:,.2f}, ${report.bootstrap.ci_upper_95:,.2f}]")


def _render_counterfactual_chart(report: AttributionReport) -> None:
    """Counterfactual comparison: actual vs closing-line profit."""
    cf = report.counterfactual

    if _HAS_PLOTLY:
        categories = ["Actual Profit", "If Bet at Close", "Timing Edge", "Prediction Edge"]
        values = [
            cf.actual_profit,
            cf.closing_line_profit,
            cf.timing_edge_dollars,
            cf.prediction_edge_dollars,
        ]
        colors = [_GREEN, _MUTED, _BLUE, _PURPLE]

        fig = go.Figure(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"${v:+,.2f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, family=_FONT),
        ))
        fig.update_layout(
            title="Counterfactual Analysis: Actual vs Closing Line",
            template="plotly_dark",
            paper_bgcolor=_BG,
            plot_bgcolor=_BG,
            font=dict(family=_FONT, size=11),
            yaxis_title="Profit ($)",
            height=380,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"Actual Profit: ${cf.actual_profit:+,.2f}")
        st.write(f"Closing-Line Profit: ${cf.closing_line_profit:+,.2f}")
        st.write(f"Timing Edge: ${cf.timing_edge_dollars:+,.2f}")
        st.write(f"Prediction Edge: ${cf.prediction_edge_dollars:+,.2f}")


def _render_segment_table(report: AttributionReport) -> None:
    """Market inefficiency segment table."""
    segments = report.inefficiency.segments
    if not segments:
        st.info("No segment data available.")
        return

    # Sort by PnL descending
    segments_sorted = sorted(segments, key=lambda s: s.total_pnl, reverse=True)

    if _HAS_PANDAS:
        rows = []
        for s in segments_sorted:
            rows.append({
                "Segment": s.segment_name,
                "Value": s.segment_value,
                "Bets": s.n_bets,
                "P&L": f"${s.total_pnl:+,.2f}",
                "ROI": f"{s.roi_pct:+.1f}%",
                "Avg Edge": f"{s.avg_edge:+.4f}",
                "% of Profit": f"{s.pct_of_total_profit:.1f}%",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        for s in segments_sorted:
            st.write(
                f"**{s.segment_name}: {s.segment_value}** — "
                f"{s.n_bets} bets, ${s.total_pnl:+,.2f} P&L, "
                f"{s.roi_pct:+.1f}% ROI"
            )


def _render_verdict_card(report: AttributionReport) -> None:
    """Render the final verdict as a styled card."""
    verdict_colors = {
        "REAL EDGE": _GREEN,
        "REAL EDGE (CLV-DRIVEN)": _GREEN,
        "PROBABLE EDGE": _BLUE,
        "PROBABLE EDGE (NEEDS MORE DATA)": _BLUE,
        "POSSIBLE EDGE (NEEDS MORE DATA)": _YELLOW,
        "LIKELY VARIANCE": _RED,
        "INSUFFICIENT DATA": _MUTED,
    }
    color = verdict_colors.get(report.verdict, _MUTED)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #060D18, #0A1628);
                border: 2px solid {color}; border-radius: 8px;
                padding: 1.2rem; text-align: center; margin: 1rem 0;">
        <div style="font-family: {_FONT}; font-size: 0.7rem; color: {_MUTED};
                    letter-spacing: 0.15em; text-transform: uppercase;">
            EDGE ATTRIBUTION VERDICT
        </div>
        <div style="font-family: {_FONT}; font-size: 1.6rem; font-weight: 700;
                    color: {color}; margin: 0.5rem 0;">
            {report.verdict}
        </div>
        <div style="font-family: {_FONT}; font-size: 0.75rem; color: {_MUTED};">
            Confidence: {report.confidence_level} |
            P(profit > 0): {report.bootstrap.p_profit_positive:.1%} |
            {report.total_bets} bets analyzed
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Edge Attribution tab."""
    st.markdown(
        "<h2 style='font-family: Chakra Petch, monospace; color: #00FFB2; "
        "margin-bottom: 0;'>EDGE ATTRIBUTION</h2>"
        "<p style='font-family: Fira Code, monospace; font-size: 0.7rem; "
        "color: #4A607A; margin-top: 0;'>"
        "Where does profit ACTUALLY come from?</p>",
        unsafe_allow_html=True,
    )

    # Load data
    bets = _load_settled_bets()

    if not bets:
        st.warning(
            "No settled bets found. Place and settle bets to see attribution analysis."
        )
        st.info(
            "The Edge Attribution Engine decomposes every dollar of profit into:\n"
            "1. **Prediction Edge** — being right about probabilities\n"
            "2. **CLV/Timing Edge** — betting before the line moves\n"
            "3. **Market Inefficiency** — structural mispricing\n"
            "4. **Variance** — pure luck (the enemy)"
        )
        return

    # Run engine
    engine = EdgeAttributionEngine(bets)
    report = engine.decompose()

    # Verdict card at the top
    _render_verdict_card(report)

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Profit", f"${report.total_profit:+,.2f}")
    with c2:
        st.metric("ROI", f"{report.roi_pct:+.2f}%")
    with c3:
        st.metric("Total Bets", f"{report.total_bets}")
    with c4:
        clv_label = "CLV BOT" if report.is_clv_bot else "PREDICTION ENGINE"
        st.metric("System Type", clv_label)

    st.markdown("---")

    # Charts row 1: Pie + Bootstrap
    col_pie, col_boot = st.columns(2)
    with col_pie:
        _render_pie_chart(report)
    with col_boot:
        _render_bootstrap_histogram(report)

    # Charts row 2: Counterfactual
    _render_counterfactual_chart(report)

    st.markdown("---")

    # Attribution detail metrics
    st.markdown(
        "<h4 style='font-family: Chakra Petch, monospace; color: #00AAFF;'>"
        "PROFIT ATTRIBUTION BREAKDOWN</h4>",
        unsafe_allow_html=True,
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric(
            "Prediction Edge",
            f"${report.prediction_edge.dollar_value:+,.2f}",
            f"{report.prediction_edge.pct_of_total:+.1f}%",
        )
    with mc2:
        st.metric(
            "CLV/Timing Edge",
            f"${report.clv_timing_edge.dollar_value:+,.2f}",
            f"{report.clv_timing_edge.pct_of_total:+.1f}%",
        )
    with mc3:
        st.metric(
            "Market Inefficiency",
            f"${report.market_inefficiency.dollar_value:+,.2f}",
            f"{report.market_inefficiency.pct_of_total:+.1f}%",
        )
    with mc4:
        st.metric(
            "Variance (Luck)",
            f"${report.variance_component.dollar_value:+,.2f}",
            f"{report.variance_component.pct_of_total:+.1f}%",
        )

    st.markdown("---")

    # Statistical significance section
    st.markdown(
        "<h4 style='font-family: Chakra Petch, monospace; color: #00AAFF;'>"
        "STATISTICAL SIGNIFICANCE</h4>",
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        p_val = report.bootstrap.p_profit_positive
        color = "normal" if p_val >= 0.95 else ("off" if p_val >= 0.80 else "inverse")
        st.metric("P(Profit > 0)", f"{p_val:.1%}", delta_color=color)
    with s2:
        st.metric("95% CI Lower", f"${report.bootstrap.ci_lower_95:+,.2f}")
    with s3:
        st.metric("95% CI Upper", f"${report.bootstrap.ci_upper_95:+,.2f}")
    with s4:
        st.metric("Bets Needed", f"{report.bootstrap.required_sample_for_significance}")

    st.markdown("---")

    # CLV diagnostic
    st.markdown(
        "<h4 style='font-family: Chakra Petch, monospace; color: #00AAFF;'>"
        "CLV DIAGNOSTIC</h4>",
        unsafe_allow_html=True,
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("CLV % of Profit", f"{report.clv_pct_of_profit:.1f}%")
    with d2:
        cf = report.counterfactual
        st.metric("Bets Still Profitable at Close", f"{cf.n_bets_better_at_close}")
    with d3:
        st.metric("Bets Profitable ONLY from Timing", f"{cf.n_bets_only_timing}")

    if report.is_clv_bot:
        st.warning(
            "CLV accounts for >80% of profit. This system is a CLV bot — "
            "profit comes from timing, not prediction accuracy. "
            "Both are valid edges, but know which one you have."
        )

    st.markdown("---")

    # Market Inefficiency segments
    st.markdown(
        "<h4 style='font-family: Chakra Petch, monospace; color: #00AAFF;'>"
        "MARKET INEFFICIENCY SEGMENTS</h4>",
        unsafe_allow_html=True,
    )
    if report.inefficiency.concentrated:
        st.warning(
            f"Profit is concentrated in: **{report.inefficiency.dominant_segment}** — "
            f"this suggests structural inefficiency capture rather than broad prediction skill."
        )

    _render_segment_table(report)

    # Raw report (collapsible)
    with st.expander("Raw Attribution Report"):
        st.code(engine.generate_report(), language="text")
