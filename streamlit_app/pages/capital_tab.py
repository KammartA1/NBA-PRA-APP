"""
streamlit_app/pages/capital_tab.py
====================================
Capital management dashboard — Kelly calculator, risk metrics,
portfolio optimizer, concentration alerts.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING


def render() -> None:
    st.markdown(
        f"<h2 style='font-family:{FONT_DISPLAY};color:{COLOR_PRIMARY};font-size:1.3rem;"
        f"letter-spacing:0.1em;margin-bottom:0;'>CAPITAL MANAGEMENT</h2>"
        f"<p style='font-family:{FONT_MONO};color:#4A607A;font-size:0.65rem;margin-top:0.2rem;'>"
        f"Kelly sizing, risk metrics, portfolio optimization</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["KELLY CALCULATOR", "RISK METRICS", "PORTFOLIO", "OPTIMIZER"])

    with tabs[0]:
        _render_kelly()
    with tabs[1]:
        _render_risk_metrics()
    with tabs[2]:
        _render_portfolio()
    with tabs[3]:
        _render_optimizer()


def _render_kelly() -> None:
    st.markdown("### Kelly Criterion Calculator")

    from services.capital.kelly import KellyCalculator

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        win_prob = st.slider("Win Probability", 0.40, 0.75, 0.55, 0.01, key="kc_wp")
    with col2:
        odds = st.number_input("Decimal Odds", 1.50, 3.00, 1.91, 0.01, key="kc_odds")
    with col3:
        bankroll = st.number_input("Bankroll ($)", 100, 100000, 5000, 100, key="kc_br")
    with col4:
        fraction = st.selectbox("Kelly Fraction", [0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 1.0], index=3, key="kc_frac")

    prob_std = st.slider("Probability Uncertainty (std)", 0.0, 0.10, 0.03, 0.01, key="kc_std")

    calc = KellyCalculator(default_fraction=fraction)
    result = calc.compute(win_prob, odds, bankroll, prob_std, fraction)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Full Kelly", f"{result.full_kelly_pct:.2f}%")
    with c2:
        st.metric(f"{fraction:.0%} Kelly", f"{result.fractional_kelly_pct:.2f}%")
    with c3:
        st.metric("Uncertainty Adj", f"{result.uncertainty_adjusted_pct:.2f}%")
    with c4:
        st.metric("Final Stake", f"${result.final_stake_dollars:.2f}")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Edge", f"{result.edge_pct:.2f}%")
    with m2:
        st.metric("Growth Rate", f"{result.growth_rate:.4f}")
    with m3:
        color = COLOR_PRIMARY if result.is_positive_ev else COLOR_DANGER
        st.metric("+EV?", "YES" if result.is_positive_ev else "NO")

    # Sensitivity chart: Kelly stake vs win probability
    probs = np.linspace(0.45, 0.70, 50)
    stakes = [max(calc.full_kelly(p, odds) * fraction * 100, 0) for p in probs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=probs * 100, y=stakes, mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2), name="Kelly Stake %"))
    fig.add_vline(x=win_prob * 100, line_dash="dash", line_color=COLOR_WARNING)
    fig.update_layout(
        title="Stake % vs Win Probability",
        xaxis_title="Win Probability (%)", yaxis_title="Stake (% of bankroll)",
        template="plotly_dark", height=300,
        paper_bgcolor="transparent", plot_bgcolor="transparent",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_metrics() -> None:
    st.markdown("### Risk-Adjusted Performance")

    from services.capital.risk_adjusted import RiskMetrics
    from database.connection import session_scope
    from database.models import Bet

    try:
        with session_scope() as session:
            bets = session.query(Bet).filter(
                Bet.sport == "NBA", Bet.status == "settled"
            ).order_by(Bet.timestamp.asc()).all()
            pnl = np.array([b.profit or 0 for b in bets])
            stakes = np.array([b.stake or 1 for b in bets])
    except Exception:
        pnl = np.array([])
        stakes = np.array([])

    if len(pnl) < 10:
        st.info("Need at least 10 settled bets for risk metrics.")
        return

    rm = RiskMetrics()
    report = rm.compute(pnl, stakes=stakes)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Sharpe Ratio", f"{report.sharpe_ratio:.2f}")
    with c2:
        st.metric("Sortino Ratio", f"{report.sortino_ratio:.2f}")
    with c3:
        st.metric("Calmar Ratio", f"{report.calmar_ratio:.2f}")
    with c4:
        st.metric("Profit Factor", f"{report.profit_factor:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Max Drawdown", f"{report.max_drawdown_pct:.1f}%")
    with c6:
        st.metric("95% VaR", f"{report.var_95_daily_pct:.2f}%")
    with c7:
        st.metric("95% CVaR", f"{report.cvar_95_daily_pct:.2f}%")
    with c8:
        st.metric("Variance Ratio", f"{report.variance_ratio:.2f}x")

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        st.metric("Win Rate", f"{report.win_rate:.1%}")
    with c10:
        st.metric("Total ROI", f"{report.total_return_pct:.1f}%")
    with c11:
        st.metric("Expectancy/Bet", f"${report.expectancy_per_bet:.2f}")
    with c12:
        st.metric("Total Bets", str(report.n_bets))

    # Equity curve
    equity = np.cumsum(pnl) + 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode="lines",
        line=dict(color=COLOR_PRIMARY, width=2), name="Equity"))
    peak = np.maximum.accumulate(equity)
    fig.add_trace(go.Scatter(y=peak, mode="lines",
        line=dict(color="#4A607A", width=1, dash="dot"), name="Peak"))
    fig.update_layout(
        title="Equity Curve", xaxis_title="Bet #", yaxis_title="Bankroll ($)",
        template="plotly_dark", height=300,
        paper_bgcolor="transparent", plot_bgcolor="transparent",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_portfolio() -> None:
    st.markdown("### Portfolio Correlation & Concentration")
    st.info("Portfolio analysis will populate when concurrent pending bets exist. "
            "Use the Capital Optimizer tab to simulate portfolio allocation.")


def _render_optimizer() -> None:
    st.markdown("### Capital Optimizer")

    from services.capital.optimizer import CapitalOptimizer, OptimizationConstraints

    bankroll = st.number_input("Bankroll ($)", 500, 100000, 5000, 500, key="co_br")
    kelly_frac = st.slider("Kelly Fraction", 0.10, 0.50, 0.25, 0.05, key="co_kf")

    st.markdown("#### Add Bets to Optimize")
    n_bets = st.number_input("Number of bets", 1, 10, 3, key="co_nb")

    bets = []
    for i in range(int(n_bets)):
        with st.expander(f"Bet {i+1}", expanded=(i == 0)):
            c1, c2, c3 = st.columns(3)
            with c1:
                wp = st.slider(f"Win Prob", 0.45, 0.70, 0.55, 0.01, key=f"co_wp_{i}")
            with c2:
                od = st.number_input(f"Odds (decimal)", 1.50, 3.00, 1.91, 0.01, key=f"co_od_{i}")
            with c3:
                player = st.text_input(f"Player", f"Player {i+1}", key=f"co_pl_{i}")
            bets.append({
                "win_prob": wp, "odds_decimal": od, "player": player,
                "market": "points", "direction": "over", "game": f"game_{i // 2}",
            })

    if st.button("Optimize Portfolio", type="primary"):
        constraints = OptimizationConstraints(kelly_fraction=kelly_frac)
        optimizer = CapitalOptimizer(bankroll=bankroll, constraints=constraints)
        result = optimizer.optimize_batch(bets)

        summary = result["summary"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Approved", f"{summary['approved']}/{summary['total_bets']}")
        with c2:
            st.metric("Total Exposure", f"{summary['total_exposure_pct']:.1f}%")
        with c3:
            st.metric("Portfolio Edge", f"{summary['portfolio_edge_pct']:.2f}%")

        import pandas as pd
        bet_rows = []
        for b in result["bets"]:
            bet_rows.append({
                "Player": b["player"],
                "Edge %": f"{b['edge_pct']:.2f}",
                "Stake %": f"{b['stake_pct']:.2f}",
                "Stake $": f"${b['stake_dollars']:.2f}",
                "Approved": "YES" if b["approved"] else "NO",
                "Notes": b.get("rejection_reason", "") or ", ".join(b.get("adjustments", [])),
            })
        st.dataframe(pd.DataFrame(bet_rows), hide_index=True, use_container_width=True)

        if result["alerts"]:
            st.warning("Concentration Alerts:")
            for a in result["alerts"]:
                st.markdown(f"- **{a['severity'].upper()}**: {a['recommendation']}")
