"""
streamlit_app/pages/kill_switch_tab.py
=======================================
Kill Switch dashboard — status of all 6 hard kill conditions
with big red banners when triggered.
"""

from __future__ import annotations

import streamlit as st
from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING


def render() -> None:
    st.markdown(
        f"<h2 style='font-family:{FONT_DISPLAY};color:{COLOR_PRIMARY};font-size:1.3rem;"
        f"letter-spacing:0.1em;margin-bottom:0;'>KILL SWITCH</h2>"
        f"<p style='font-family:{FONT_MONO};color:#4A607A;font-size:0.65rem;margin-top:0.2rem;'>"
        f"6 non-negotiable halt conditions — no overrides</p>",
        unsafe_allow_html=True,
    )

    bankroll = float(st.session_state.get("bankroll", 1000))
    peak = float(st.session_state.get("peak_bankroll", bankroll))
    if bankroll > peak:
        peak = bankroll
        st.session_state["peak_bankroll"] = peak

    try:
        from services.kill_switch import KillSwitch
        ks = KillSwitch(sport="NBA")
        status = ks.check_all(bankroll=bankroll, peak_bankroll=peak)
    except Exception as e:
        st.warning(f"Kill switch check failed: {e}")
        return

    # Main status banner
    if status.severity == "halted":
        st.markdown(f"""
        <div style='background:#FF335815;border:3px solid {COLOR_DANGER};border-radius:12px;
                    padding:2rem;text-align:center;margin-bottom:1.5rem;'>
            <div style='font-family:{FONT_DISPLAY};font-size:2rem;font-weight:700;
                        color:{COLOR_DANGER};letter-spacing:0.15em;'>SYSTEM HALTED</div>
            <div style='font-family:{FONT_MONO};font-size:0.75rem;color:#FF6B8A;margin-top:0.5rem;'>
                {status.halt_reason}</div>
        </div>
        """, unsafe_allow_html=True)
    elif status.severity == "reduced":
        st.markdown(f"""
        <div style='background:#FFB80015;border:3px solid {COLOR_WARNING};border-radius:12px;
                    padding:2rem;text-align:center;margin-bottom:1.5rem;'>
            <div style='font-family:{FONT_DISPLAY};font-size:2rem;font-weight:700;
                        color:{COLOR_WARNING};letter-spacing:0.15em;'>REDUCED MODE</div>
            <div style='font-family:{FONT_MONO};font-size:0.75rem;color:#FFD060;margin-top:0.5rem;'>
                Kelly reduced to {status.recommended_kelly_mult:.0%} — {status.halt_reason}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#00FFB210;border:3px solid {COLOR_PRIMARY};border-radius:12px;
                    padding:2rem;text-align:center;margin-bottom:1.5rem;'>
            <div style='font-family:{FONT_DISPLAY};font-size:2rem;font-weight:700;
                        color:{COLOR_PRIMARY};letter-spacing:0.15em;'>ALL CLEAR</div>
            <div style='font-family:{FONT_MONO};font-size:0.75rem;color:#80FFD0;margin-top:0.5rem;'>
                All kill conditions passed — system is cleared for betting</div>
        </div>
        """, unsafe_allow_html=True)

    # Individual conditions
    st.markdown("### Kill Conditions")

    for cond in status.conditions:
        triggered = cond.triggered
        color = COLOR_DANGER if triggered and cond.severity == "halt" else (
            COLOR_WARNING if triggered else COLOR_PRIMARY
        )
        icon = "X" if triggered else "OK"
        bg = f"{color}10"

        st.markdown(f"""
        <div style='background:{bg};border:1px solid {color}40;border-radius:8px;
                    padding:1rem 1.5rem;margin:0.5rem 0;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='font-family:{FONT_DISPLAY};font-size:0.85rem;font-weight:700;
                                color:{color};letter-spacing:0.08em;'>
                        [{icon}] {cond.name.upper().replace("_", " ")}</div>
                    <div style='font-family:{FONT_MONO};font-size:0.65rem;color:#8BA8BF;
                                margin-top:0.2rem;'>{cond.description}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-family:{FONT_MONO};font-size:0.75rem;color:{color};
                                font-weight:700;'>
                        {"TRIGGERED" if triggered else "PASSED"}</div>
                    <div style='font-family:{FONT_MONO};font-size:0.6rem;color:#6A8AAA;'>
                        Severity: {cond.severity.upper()}</div>
                </div>
            </div>
            {"<div style='font-family:" + FONT_MONO + ";font-size:0.6rem;color:#FFB800;margin-top:0.4rem;border-top:1px solid " + color + "20;padding-top:0.4rem;'>Action: " + cond.action + "</div>" if triggered else ""}
        </div>
        """, unsafe_allow_html=True)

    # Summary
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Bets Analyzed", str(status.total_bets_analyzed))
    with c2:
        st.metric("Conditions Triggered", str(len(status.triggered_conditions)))
    with c3:
        st.metric("Kelly Multiplier", f"{status.recommended_kelly_mult:.0%}")
