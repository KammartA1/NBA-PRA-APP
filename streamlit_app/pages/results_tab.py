"""
streamlit_app/pages/results_tab.py
==================================
RESULTS tab -- displays the last model run results, edge comparisons,
and AI analysis.  All data from session state (transient last run) or
from the signals table via report_service.
"""
from __future__ import annotations

import streamlit as st

from services import report_service, projection_service
from streamlit_app.config import (
    FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING,
    get_anthropic_key, get_user_id,
)
from streamlit_app.state import get_ui_state


def render() -> None:
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"PROJECTION RESULTS & ANALYSIS</div>",
        unsafe_allow_html=True,
    )

    # Try transient results first, then fall back to DB signals
    results = get_ui_state("last_results", [])
    if not results:
        st.info(
            "No model results in current session. Run the MODEL tab first, "
            "or view historical signals below."
        )

    # -- Display current results -------------------------------------------
    if results:
        _display_edge_summary(results)
        st.markdown("---")
        _display_signal_cards(results)

    # -- Recent signals from DB --------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:#2A5070;"
        f"letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.6rem;'>"
        f"RECENT SIGNALS FROM DATABASE</div>",
        unsafe_allow_html=True,
    )
    edge_summary = report_service.get_edge_summary()
    if edge_summary and edge_summary.get("total_signals", 0) > 0:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Signals (7d)", edge_summary.get("total_signals", 0))
        mc2.metric("Avg Edge", f"{edge_summary.get('avg_edge', 0):.1f}%")
        mc3.metric("Positive Edge %", f"{edge_summary.get('gated_pct', 0):.0f}%")
        mc4.metric("Avg Sharpness", f"{edge_summary.get('avg_sharpness', 0):.0f}")

        # Top markets
        top_markets = edge_summary.get("top_markets", [])
        if top_markets:
            st.markdown(
                f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
                f"letter-spacing:0.12em;margin-top:0.8rem;margin-bottom:0.4rem;'>"
                f"TOP MARKETS BY VOLUME</div>",
                unsafe_allow_html=True,
            )
            import pandas as pd
            df = pd.DataFrame(top_markets)
            df.columns = ["Market", "Count", "Avg Edge %"]
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No signals recorded in the last 7 days.")


def _display_edge_summary(results: list[dict]) -> None:
    """Summary bar across all legs."""
    valid = [r for r in results if r.get("gate_ok")]
    total = len(results)
    gated = total - len(valid)

    c1, c2, c3 = st.columns(3)
    c1.metric("Legs Analyzed", total)
    c2.metric("Passed Gate", len(valid))
    c3.metric("Gated", gated)

    if valid:
        avg_ev = sum(float(r.get("ev_adj", 0) or 0) for r in valid) / len(valid) * 100
        avg_prob = sum(float(r.get("p_cal", 0) or 0) for r in valid) / len(valid) * 100
        total_stake = sum(float(r.get("stake", 0) or 0) for r in valid)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Avg EV", f"{avg_ev:+.1f}%")
        mc2.metric("Avg Win Prob", f"{avg_prob:.1f}%")
        mc3.metric("Total Stake", f"${total_stake:.2f}")


def _display_signal_cards(results: list[dict]) -> None:
    """Display each result as a card with optional AI analysis."""
    for i, leg in enumerate(results):
        player = leg.get("player", "Unknown")
        market = leg.get("market", "")
        line = leg.get("line", 0)
        proj = leg.get("proj")
        p_cal = leg.get("p_cal")
        ev_adj = leg.get("ev_adj")
        gate_ok = leg.get("gate_ok", False)
        side = leg.get("side", "over")

        if not gate_ok:
            with st.expander(f"GATED: {player} | {market} {side.upper()} {line}", expanded=False):
                st.warning(f"Gate reason: {leg.get('gate_reason', 'unknown')}")
                errors = leg.get("errors", [])
                for e in errors:
                    st.caption(e)
            continue

        # Passed-gate card
        ev_pct = float(ev_adj or 0) * 100
        prob_pct = float(p_cal or 0) * 100
        edge_cat = leg.get("edge_cat", "")

        edge_color = COLOR_PRIMARY if ev_pct >= 5 else (COLOR_WARNING if ev_pct >= 2 else "#4A607A")
        with st.expander(
            f"{player} | {market} {side.upper()} {line} | "
            f"EV {ev_pct:+.1f}% | Prob {prob_pct:.0f}% | {edge_cat}",
            expanded=True,
        ):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Projection", f"{proj:.1f}" if proj else "--")
            c2.metric("Win Prob", f"{prob_pct:.1f}%")
            c3.metric("EV", f"{ev_pct:+.1f}%")
            c4.metric("Stake", f"${leg.get('stake', 0):.2f}")
            c5.metric("Sharpness", f"{_safe(leg.get('sharpness_score'), 0)}")

            # Trend / context row
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.caption(f"Trend: {leg.get('trend_label', '--')}")
            tc2.caption(f"Hot/Cold: {leg.get('hot_cold', '--')}")
            tc3.caption(f"Regime: {leg.get('regime', '--')}")
            tc4.caption(f"Fatigue: {leg.get('fatigue_label', 'Normal')}")

            # AI analysis
            if get_anthropic_key():
                if st.button(f"AI Analysis", key=f"ai_btn_{i}"):
                    with st.spinner("Generating AI analysis..."):
                        analysis = _get_ai_analysis(leg)
                    if analysis:
                        st.markdown(
                            f"<div style='background:#060D18;border-left:3px solid {COLOR_PRIMARY};"
                            f"padding:0.6rem 0.8rem;margin-top:0.5rem;font-family:{FONT_MONO};"
                            f"font-size:0.7rem;color:#B0C8E0;line-height:1.6;'>{analysis}</div>",
                            unsafe_allow_html=True,
                        )


def _get_ai_analysis(leg: dict) -> str | None:
    """Call AI to explain a leg.  Uses cached version from app.py if available."""
    try:
        import nba_engine as _app
        return _app.ai_explain_edge(
            player=leg.get("player", ""),
            market=leg.get("market", ""),
            line=leg.get("line", 0),
            side=leg.get("side", "over"),
            proj=leg.get("proj"),
            p_cal=leg.get("p_cal"),
            ev_pct=float(leg.get("ev_adj", 0) or 0) * 100,
            edge_cat=leg.get("edge_cat", ""),
            hot_cold=leg.get("hot_cold", "Average"),
            rest_days=leg.get("rest_days", 2),
            dnp_risk=leg.get("dnp_risk", False),
            b2b=leg.get("b2b", False),
            opp=leg.get("opp", ""),
            vol_cv=leg.get("volatility_cv"),
            n_games=leg.get("n_games_used", 0),
            errors_str=", ".join(leg.get("errors", [])),
            trend_label=leg.get("trend_label", "Neutral"),
            sharpness_score=leg.get("sharpness_score"),
            sharpness_tier=leg.get("sharpness_tier"),
            game_total=leg.get("game_total"),
            fatigue_label=leg.get("fatigue_label", "Normal"),
        )
    except Exception:
        return None


def _safe(val, decimals=2) -> str:
    if val is None:
        return "--"
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)
