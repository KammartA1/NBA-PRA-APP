"""
streamlit_app/pages/edge_monitor_tab.py
=========================================
Edge Monitor dashboard — THE home page feel.

Daily edge verdict (EDGE = YES/NO), CLV trends, Brier trends,
ROI evolution, regime classification, alerts.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING


def render() -> None:
    st.markdown(
        f"<h2 style='font-family:{FONT_DISPLAY};color:{COLOR_PRIMARY};font-size:1.3rem;"
        f"letter-spacing:0.1em;margin-bottom:0;'>EDGE MONITOR</h2>"
        f"<p style='font-family:{FONT_MONO};color:#4A607A;font-size:0.65rem;margin-top:0.2rem;'>"
        f"Daily verdict: do I still have edge?</p>",
        unsafe_allow_html=True,
    )

    # Generate verdict
    try:
        from services.edge_monitor.alert_system import EdgeAlertSystem
        alert_sys = EdgeAlertSystem(sport="NBA")
        verdict = alert_sys.generate_daily_verdict()
    except Exception as e:
        st.warning(f"Edge monitor not fully operational: {e}")
        verdict = None

    if verdict:
        _render_verdict_banner(verdict)

    tabs = st.tabs(["DAILY VERDICT", "CLV TRENDS", "CALIBRATION", "ALERTS", "HISTORY"])

    with tabs[0]:
        _render_daily_verdict(verdict)
    with tabs[1]:
        _render_clv_trends()
    with tabs[2]:
        _render_calibration(verdict)
    with tabs[3]:
        _render_alerts(verdict)
    with tabs[4]:
        _render_history()


def _render_verdict_banner(verdict) -> None:
    has_edge = verdict.has_edge
    color = COLOR_PRIMARY if has_edge else COLOR_DANGER
    edge_text = "YES" if has_edge else "NO"

    banner_style = (
        f"background:linear-gradient(135deg,{'#00FFB210' if has_edge else '#FF335810'},"
        f"{'#00AAFF08' if has_edge else '#FF000008'});"
        f"border:2px solid {color};border-radius:12px;padding:1.5rem;margin-bottom:1rem;"
        f"text-align:center;"
    )

    st.markdown(f"""
    <div style='{banner_style}'>
        <div style='font-family:{FONT_DISPLAY};font-size:0.7rem;color:#6A8AAA;
                    letter-spacing:0.2em;margin-bottom:0.3rem;'>DAILY EDGE VERDICT</div>
        <div style='font-family:{FONT_DISPLAY};font-size:2.5rem;font-weight:700;
                    color:{color};letter-spacing:0.1em;'>EDGE = {edge_text}</div>
        <div style='font-family:{FONT_MONO};font-size:0.75rem;color:#8BA8BF;margin-top:0.3rem;'>
            Score: {verdict.edge_score:.0f}/100 | Regime: {verdict.regime.upper()} |
            Trend: {verdict.trend.upper()} |
            Action: {verdict.recommended_action}</div>
    </div>
    """, unsafe_allow_html=True)

    if verdict.is_critical:
        st.error(f"CRITICAL: No edge detected for {verdict.consecutive_no_edge_days} consecutive days. "
                 f"Action: {verdict.recommended_action}")


def _render_daily_verdict(verdict) -> None:
    if not verdict:
        st.info("No verdict data available.")
        return

    m = verdict.metrics
    if not m:
        st.info("No metrics data available.")
        return

    st.markdown("### Core Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _clv_delta = round(m.clv_50_avg - m.clv_200_avg, 2)
        st.metric("CLV (50-bet)", f"{m.clv_50_avg:.2f}c",
                  delta=_clv_delta, delta_color="normal")
    with c2:
        st.metric("Brier Score", f"{m.brier_score:.4f}",
                  delta=round(m.brier_advantage, 4), delta_color="inverse")
    with c3:
        st.metric("ROI (50-bet)", f"{m.roi_50_pct:.1f}%")
    with c4:
        st.metric("Win Rate (50)", f"{m.win_rate_50:.1%}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("CLV Beat Rate", f"{m.clv_beat_rate:.1%}")
    with c6:
        st.metric("Log Loss", f"{m.log_loss:.4f}")
    with c7:
        st.metric("Variance Ratio", f"{m.variance_ratio:.2f}x")
    with c8:
        st.metric("Max Drawdown", f"{m.max_drawdown_pct:.1f}%")

    # Edge score gauge
    score = verdict.edge_score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Edge Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLOR_PRIMARY if score >= 50 else COLOR_DANGER},
            "steps": [
                {"range": [0, 30], "color": "#FF335820"},
                {"range": [30, 50], "color": "#FFB80020"},
                {"range": [50, 70], "color": "#00AAFF20"},
                {"range": [70, 100], "color": "#00FFB220"},
            ],
            "threshold": {"line": {"color": "#FFFFFF", "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(height=250, paper_bgcolor="transparent", plot_bgcolor="transparent",
                      font=dict(color="#8BA8BF"))
    st.plotly_chart(fig, use_container_width=True)


def _render_clv_trends() -> None:
    st.markdown("### CLV & ROI Trends")

    try:
        from services.edge_monitor.daily_metrics import DailyEdgeMetrics
        engine = DailyEdgeMetrics(sport="NBA")
        window = st.selectbox("Rolling Window", [25, 50, 100, 200], index=1, key="clv_window")
        rolling = engine.compute_rolling_series(window=window)
    except Exception as e:
        st.warning(f"Could not compute trends: {e}")
        return

    if not rolling.get("clv"):
        st.info("Not enough data for rolling trends.")
        return

    # CLV trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rolling["clv"], mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2), name="CLV (cents)"))
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_DANGER, line_width=1)
    fig.update_layout(
        title=f"{window}-Bet Rolling CLV",
        yaxis_title="CLV (cents)", template="plotly_dark", height=300,
        paper_bgcolor="transparent", plot_bgcolor="transparent",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ROI trend
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=rolling["roi"], mode="lines",
        line=dict(color="#4C9AFF", width=2), name="ROI %"))
    fig2.add_hline(y=0, line_dash="dash", line_color=COLOR_DANGER, line_width=1)
    fig2.update_layout(
        title=f"{window}-Bet Rolling ROI",
        yaxis_title="ROI %", template="plotly_dark", height=300,
        paper_bgcolor="transparent", plot_bgcolor="transparent",
    )
    st.plotly_chart(fig2, use_container_width=True)


def _render_calibration(verdict) -> None:
    st.markdown("### Calibration & Brier Score")
    if not verdict:
        st.info("No calibration data available.")
        return

    m = verdict.metrics
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Our Brier Score | {m.brier_score:.4f} |
    | Market Brier Score | {m.market_brier_score:.4f} |
    | Brier Advantage | {m.brier_advantage:+.4f} |
    | Calibration Error | {m.calibration_error:.4f} |
    | Log Loss | {m.log_loss:.4f} |
    """)

    better = m.brier_advantage < 0
    st.markdown(
        f"**Verdict:** Model is **{'better' if better else 'worse'}** than market "
        f"by {abs(m.brier_advantage):.4f} Brier points."
    )


def _render_alerts(verdict) -> None:
    st.markdown("### Active Alerts")
    if not verdict or not verdict.alerts:
        st.success("No active alerts.")
        return

    for alert in verdict.alerts:
        level = alert.level
        if level == "SYSTEM_HALT":
            st.error(f"[{level}] {alert.message}")
        elif level == "CRITICAL":
            st.error(f"[{level}] {alert.message}")
        elif level == "WARNING":
            st.warning(f"[{level}] {alert.message}")
        else:
            st.info(f"[{level}] {alert.message}")
        st.caption(f"Recommendation: {alert.recommendation}")


def _render_history() -> None:
    st.markdown("### Verdict History")
    try:
        from services.edge_monitor.alert_system import EdgeAlertSystem
        history = EdgeAlertSystem(sport="NBA").get_verdict_history(30)
        if history:
            import pandas as pd
            df = pd.DataFrame(history)
            cols = ["date", "has_edge", "edge_score", "regime", "recommended_action", "clv_50", "brier", "roi_50"]
            available = [c for c in cols if c in df.columns]
            if available:
                st.dataframe(df[available], hide_index=True, use_container_width=True)
            else:
                st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No verdict history yet.")
    except Exception as e:
        st.warning(f"Could not load history: {e}")
