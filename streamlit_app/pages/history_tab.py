"""
streamlit_app/pages/history_tab.py
==================================
HISTORY tab -- bet history, settlement, P&L display.
All data from bet_service (DB).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from services import bet_service
from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, get_user_id
from streamlit_app.components.charts import pnl_curve


def render() -> None:
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"BET HISTORY & P&L</div>",
        unsafe_allow_html=True,
    )

    # -- P&L Summary -------------------------------------------------------
    period_tabs = st.tabs(["Daily", "Weekly", "Monthly", "All Time"])

    periods = ["daily", "weekly", "monthly", "all"]
    for idx, period in enumerate(periods):
        with period_tabs[idx]:
            summary = bet_service.get_pnl_summary(period=period)
            if summary:
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                pnl = summary.get("total_pnl", 0)
                pnl_color = COLOR_PRIMARY if pnl >= 0 else COLOR_DANGER
                mc1.metric("Total P&L", f"${pnl:+.2f}")
                mc2.metric("Bets", summary.get("n_bets", 0))
                mc3.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
                mc4.metric("ROI", f"{summary.get('roi_pct', 0):.1f}%")
                mc5.metric("Avg Stake", f"${summary.get('avg_stake', 0):.2f}")

                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("Wins", summary.get("wins", 0))
                wc2.metric("Losses", summary.get("losses", 0))
                wc3.metric("Pushes", summary.get("pushes", 0))
            else:
                st.caption("No betting data for this period.")

    # -- P&L Chart ---------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:#2A5070;"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>CUMULATIVE P&L</div>",
        unsafe_allow_html=True,
    )
    from services import report_service
    pnl_history = report_service.get_report_history("daily", days=90)
    if pnl_history:
        pnl_curve(pnl_history, title="90-Day Cumulative P&L")
    else:
        st.caption("No P&L history data available.")

    # -- Pending Bets ------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>PENDING BETS</div>",
        unsafe_allow_html=True,
    )
    pending = bet_service.get_pending_bets()
    if pending:
        st.info(f"{len(pending)} pending bets awaiting settlement.")
        df = pd.DataFrame(pending)
        display_cols = [
            "id", "player", "stat_type", "line", "direction",
            "model_prob", "edge", "stake", "odds_decimal", "timestamp",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, hide_index=True)

        # -- Settle a bet --------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.12em;margin-top:0.8rem;margin-bottom:0.4rem;'>"
            f"SETTLE A BET</div>",
            unsafe_allow_html=True,
        )
        sc1, sc2, sc3 = st.columns([2, 2, 1])
        with sc1:
            bet_ids = [int(b["id"]) for b in pending]
            settle_id = st.selectbox("Bet ID", options=bet_ids, key="settle_bet_id")
        with sc2:
            actual_result = st.number_input(
                "Actual stat result", min_value=0.0, step=0.5, key="settle_actual",
            )
        with sc3:
            st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
            if st.button("Settle", key="settle_btn", type="primary"):
                result = bet_service.settle_bet(int(settle_id), float(actual_result))
                if "error" in result:
                    st.error(f"Settlement failed: {result['error']}")
                else:
                    status = result.get("status", "?")
                    pnl = result.get("pnl", 0)
                    st.success(
                        f"Bet {settle_id} settled: {status.upper()} | P&L: ${pnl:+.2f}"
                    )
                    if result.get("beat_close"):
                        st.info(f"Beat closing line by {result.get('clv_raw', 0):.1f} points")
                    st.rerun()
    else:
        st.caption("No pending bets.")

    # -- Full History with Filters -----------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:#2A5070;"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>FULL BET HISTORY</div>",
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        filter_player = st.text_input("Player filter", key="hist_filter_player")
    with fc2:
        filter_status = st.selectbox(
            "Status", ["all", "won", "lost", "push", "pending", "signal"],
            key="hist_filter_status",
        )
    with fc3:
        filter_direction = st.selectbox(
            "Direction", ["all", "over", "under"], key="hist_filter_direction",
        )
    with fc4:
        filter_days = st.number_input("Days back", 1, 365, 30, key="hist_filter_days")

    filters = {}
    if filter_player:
        filters["player"] = filter_player
    if filter_status != "all":
        filters["status"] = filter_status
    if filter_direction != "all":
        filters["direction"] = filter_direction
    if filter_days:
        from datetime import datetime
        filters["min_date"] = datetime.utcnow() - timedelta(days=int(filter_days))

    history = bet_service.get_bet_history(filters=filters)
    if history:
        df = pd.DataFrame(history)
        display_cols = [
            "id", "player", "stat_type", "line", "direction", "status",
            "model_prob", "edge", "stake", "pnl", "actual_result",
            "odds_decimal", "timestamp",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, hide_index=True, height=400)
        st.caption(f"Showing {len(history)} bets.")
    else:
        st.caption("No bets found matching filters.")
