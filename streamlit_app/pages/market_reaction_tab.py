"""
streamlit_app/pages/market_reaction_tab.py
============================================
Market Reaction dashboard — survival curves, limit timeline, edge half-life.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING


def render() -> None:
    st.markdown(
        f"<h2 style='font-family:{FONT_DISPLAY};color:{COLOR_PRIMARY};font-size:1.3rem;"
        f"letter-spacing:0.1em;margin-bottom:0;'>MARKET REACTION</h2>"
        f"<p style='font-family:{FONT_MONO};color:#4A607A;font-size:0.65rem;margin-top:0.2rem;'>"
        f"How sportsbooks will react to your edge over time</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["SURVIVAL SIM", "LIMIT TIMELINE", "EDGE DECAY", "BOOK PROFILES", "SHADING"])

    with tabs[0]:
        _render_survival_sim()
    with tabs[1]:
        _render_limit_timeline()
    with tabs[2]:
        _render_edge_decay()
    with tabs[3]:
        _render_book_profiles()
    with tabs[4]:
        _render_shading()


def _render_survival_sim() -> None:
    st.markdown("### 12-Month Survival Simulation")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        initial_bankroll = st.number_input("Initial Bankroll ($)", 1000, 100000, 5000, 1000)
    with col2:
        initial_edge = st.slider("Initial Edge (%)", 1.0, 10.0, 4.0, 0.5)
    with col3:
        bets_per_day = st.slider("Bets/Day", 1.0, 10.0, 3.0, 0.5)
    with col4:
        n_sims = st.selectbox("Simulations", [100, 500, 1000], index=0)

    if st.button("Run Survival Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            from services.market_reaction.survival_simulator import SurvivalSimulator, SurvivalConfig
            config = SurvivalConfig(
                initial_bankroll=float(initial_bankroll),
                initial_avg_edge_pct=float(initial_edge),
                bets_per_day=float(bets_per_day),
                n_simulations=int(n_sims),
            )
            sim = SurvivalSimulator(config=config)
            results = sim.run_monte_carlo()

        # Verdict
        verdict = results.get("verdict", {})
        assessment = verdict.get("assessment", "UNKNOWN")
        color = COLOR_PRIMARY if assessment in ("STRONG", "VIABLE") else COLOR_DANGER
        st.markdown(
            f"<div style='background:#04080F;border:2px solid {color};border-radius:8px;"
            f"padding:1rem;margin:1rem 0;text-align:center;'>"
            f"<div style='font-family:{FONT_DISPLAY};font-size:1.5rem;color:{color};'>"
            f"{assessment}</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.7rem;color:#8BA8BF;margin-top:0.3rem;'>"
            f"{verdict.get('detail', '')}</div></div>",
            unsafe_allow_html=True,
        )

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("P(Profitable)", f"{verdict.get('pct_profitable', 0):.0f}%")
        with m2:
            st.metric("Median ROI", f"{verdict.get('median_roi_pct', 0):.1f}%")
        with m3:
            st.metric("Survived Full Term", f"{verdict.get('pct_survived_full_term', 0):.0f}%")
        with m4:
            st.metric("Ruin Probability", f"{verdict.get('ruin_probability_pct', 0):.1f}%")

        # Survival curve
        surv = results.get("survival", {}).get("curve", [])
        if surv:
            fig = go.Figure()
            months = [s["month"] for s in surv]
            pcts = [s["survival_pct"] for s in surv]
            fig.add_trace(go.Scatter(
                x=months, y=pcts, mode="lines+markers",
                line=dict(color=COLOR_PRIMARY, width=2),
                marker=dict(size=4),
                name="Survival %",
            ))
            fig.update_layout(
                title="Survival Curve", xaxis_title="Month", yaxis_title="% Still Active",
                template="plotly_dark", height=350,
                paper_bgcolor="transparent", plot_bgcolor="transparent",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Bankroll fan chart
        mp = results.get("monthly_percentiles", [])
        if mp:
            fig2 = go.Figure()
            months2 = [p["month"] for p in mp]
            fig2.add_trace(go.Scatter(x=months2, y=[p["p95"] for p in mp], mode="lines",
                line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=months2, y=[p["p5"] for p in mp], mode="lines",
                fill="tonexty", fillcolor="rgba(0,255,178,0.1)",
                line=dict(width=0), name="5th-95th"))
            fig2.add_trace(go.Scatter(x=months2, y=[p["p75"] for p in mp], mode="lines",
                line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=months2, y=[p["p25"] for p in mp], mode="lines",
                fill="tonexty", fillcolor="rgba(0,255,178,0.2)",
                line=dict(width=0), name="25th-75th"))
            fig2.add_trace(go.Scatter(x=months2, y=[p["p50"] for p in mp], mode="lines",
                line=dict(color=COLOR_PRIMARY, width=2), name="Median"))
            fig2.update_layout(
                title="Bankroll Distribution Over Time",
                xaxis_title="Month", yaxis_title="Bankroll ($)",
                template="plotly_dark", height=350,
                paper_bgcolor="transparent", plot_bgcolor="transparent",
            )
            st.plotly_chart(fig2, use_container_width=True)


def _render_limit_timeline() -> None:
    st.markdown("### Sportsbook Limit Progression")

    from services.market_reaction.limit_progression import LimitProgressionModel, LimitProgressionConfig
    config = LimitProgressionConfig(
        bets_per_day=st.slider("Bets per day", 1.0, 10.0, 3.0, 0.5, key="lt_bpd"),
        avg_edge_pct=st.slider("Average edge %", 1.0, 8.0, 4.0, 0.5, key="lt_edge"),
    )
    model = LimitProgressionModel(config=config)
    timelines = model.simulate_all_books()

    for name, timeline in sorted(timelines.items(), key=lambda x: x[1].total_lifetime_ev, reverse=True):
        with st.expander(f"{name.upper()} — Lifetime EV: ${timeline.total_lifetime_ev:,.0f}", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Profitable Months", f"{timeline.total_profitable_months:.1f}")
            with c2:
                st.metric("Months to Ban", f"{timeline.months_to_ban:.1f}")
            with c3:
                st.metric("Optimal Exit", f"Month {timeline.optimal_exit_month:.1f}")

            # Stage table
            if timeline.stages:
                import pandas as pd
                df = pd.DataFrame([{
                    "Stage": s.stage_name.upper(),
                    "Entry Month": s.entry_month,
                    "Max Bet": f"${s.max_bet:.0f}",
                    "Edge %": f"{s.effective_edge_pct:.1f}%",
                    "Monthly EV": f"${s.monthly_ev:.0f}",
                    "Cumulative EV": f"${s.cumulative_ev:.0f}",
                } for s in timeline.stages])
                st.dataframe(df, hide_index=True, use_container_width=True)


def _render_edge_decay() -> None:
    st.markdown("### Edge Decay by Source")

    from services.market_reaction.edge_decay import EdgeDecayModel
    model = EdgeDecayModel()

    days = st.slider("Days to project", 30, 730, 365, 30, key="ed_days")
    trajectory = model.total_edge_over_time(days=days, resolution_days=max(days // 100, 1))

    fig = go.Figure()
    time_pts = trajectory["time_points"]
    fig.add_trace(go.Scatter(
        x=time_pts, y=trajectory["total_edge"],
        mode="lines", name="Total Edge",
        line=dict(color=COLOR_PRIMARY, width=3),
    ))

    # Top sources
    source_edges = trajectory["source_edges"]
    colors = ["#4C9AFF", "#FF3358", "#FFB800", "#00AAFF", "#FF6B9D"]
    for i, (name, vals) in enumerate(sorted(source_edges.items(),
            key=lambda x: x[1][0] if x[1] else 0, reverse=True)[:5]):
        fig.add_trace(go.Scatter(
            x=time_pts, y=vals, mode="lines", name=name,
            line=dict(color=colors[i % len(colors)], width=1, dash="dot"),
        ))

    fig.update_layout(
        title="Edge Decay Over Time",
        xaxis_title="Days", yaxis_title="Edge %",
        template="plotly_dark", height=400,
        paper_bgcolor="transparent", plot_bgcolor="transparent",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Composition at current time
    comp = model.edge_composition_at(float(days))
    st.markdown(f"**Day {days} Composition:** Total = {comp['total_edge_pct']:.2f}%, "
                f"Active sources = {comp['n_active_sources']}, "
                f"Depleted = {comp['n_depleted_sources']}")

    if comp["by_category"]:
        import pandas as pd
        cat_df = pd.DataFrame(comp["by_category"])
        st.dataframe(cat_df, hide_index=True, use_container_width=True)


def _render_book_profiles() -> None:
    st.markdown("### Sportsbook Classification")

    from services.market_reaction.book_behavior import BookBehaviorModel, BettorProfile

    col1, col2, col3 = st.columns(3)
    with col1:
        total_bets = st.number_input("Total Bets", 10, 5000, 200, 50, key="bp_bets")
    with col2:
        avg_clv = st.slider("Avg CLV (cents)", -2.0, 5.0, 1.5, 0.1, key="bp_clv")
    with col3:
        win_rate = st.slider("Win Rate", 0.45, 0.65, 0.54, 0.01, key="bp_wr")

    bettor = BettorProfile(total_bets=total_bets, avg_clv_cents=avg_clv, win_rate=win_rate, clv_beat_rate=0.55)
    model = BookBehaviorModel()
    results = model.classify_all_books(bettor)

    for book, result in sorted(results.items(), key=lambda x: x[1]["trigger_score"], reverse=True):
        classif = result["classification"]
        color = {
            "unrestricted": COLOR_PRIMARY, "watched": COLOR_WARNING,
            "restricted": "#FF6B00", "banned": COLOR_DANGER,
        }.get(classif, "#4A607A")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:0.5rem 1rem;border:1px solid {color}30;border-radius:6px;margin:0.3rem 0;'>"
            f"<span style='font-family:{FONT_DISPLAY};color:#EEF4FF;font-weight:600;'>{book.upper()}</span>"
            f"<span style='font-family:{FONT_MONO};color:{color};font-weight:700;'>{classif.upper()}</span>"
            f"<span style='font-family:{FONT_MONO};font-size:0.7rem;color:#6A8AAA;'>"
            f"Max: ${result['max_bet']:.0f} | Score: {result['sharp_score']:.0f}</span></div>",
            unsafe_allow_html=True,
        )


def _render_shading() -> None:
    st.markdown("### Line Shading Analysis")

    from services.market_reaction.line_shading import LineShadingModel

    model = LineShadingModel()
    sharp_score = st.slider("Sharp Score (0-100)", 0.0, 100.0, 40.0, 5.0, key="sh_score")

    trajectory = model.shading_trajectory(sharp_score=sharp_score, months=24)

    fig = go.Figure()
    months = [t["month"] for t in trajectory["trajectory"]]
    shading = [t["shading_cents"] for t in trajectory["trajectory"]]
    fig.add_trace(go.Scatter(
        x=months, y=shading, mode="lines+markers",
        line=dict(color=COLOR_DANGER, width=2),
        name="Shading (cents)",
    ))
    fig.update_layout(
        title=f"Projected Shading (Sharp Score = {sharp_score:.0f})",
        xaxis_title="Month", yaxis_title="Shading (cents)",
        template="plotly_dark", height=300,
        paper_bgcolor="transparent", plot_bgcolor="transparent",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Net edge calculator
    st.markdown("#### Net Edge After Shading")
    gross_edge = st.slider("Gross Edge %", 1.0, 10.0, 4.0, 0.5, key="sh_edge")
    shading_val = st.slider("Shading (cents)", 0.0, 3.0, 1.0, 0.1, key="sh_val")
    result = model.compute_net_edge(gross_edge, shading_val)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Gross Edge", f"{result['gross_edge_pct']:.2f}%")
    with c2:
        st.metric("Shading Cost", f"{result['shading_cost_pct']:.2f}%")
    with c3:
        color = COLOR_PRIMARY if result["still_profitable"] else COLOR_DANGER
        st.metric("Net Edge", f"{result['net_edge_pct']:.2f}%")
