"""
streamlit_app/pages/insights_tab.py
===================================
INSIGHTS tab -- CLV Leaderboard, Book Efficiency, Prop Breakdown,
Bayesian Priors.  All data from report_service / bet_service.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import report_service, bet_service
from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY
from streamlit_app.components.charts import edge_distribution_chart, clv_chart


def render() -> None:
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"MARKET INSIGHTS & ANALYTICS</div>",
        unsafe_allow_html=True,
    )

    ins_tabs = st.tabs(["CLV Leaderboard", "Book Efficiency", "Prop Breakdown", "Bayesian Priors"])

    # -- CLV Leaderboard ---------------------------------------------------
    with ins_tabs[0]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>CLV PERFORMANCE</div>",
            unsafe_allow_html=True,
        )
        clv_summary = bet_service.get_clv_summary(window=100)
        if clv_summary and clv_summary.get("n_bets", 0) > 0:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Avg CLV (cents)", f"{clv_summary.get('avg_clv_cents', 0):.1f}")
            mc2.metric("Beat Close %", f"{clv_summary.get('beat_close_pct', 0):.1f}%")
            mc3.metric("Bets Analyzed", clv_summary.get("n_bets", 0))
            mc4.metric("Total CLV Raw", f"{clv_summary.get('total_clv_raw', 0):.1f}")

            # CLV leaders by player
            history = bet_service.get_bet_history(filters={"status": "won"})
            history += bet_service.get_bet_history(filters={"status": "lost"})
            if history:
                df = pd.DataFrame(history)
                if "player" in df.columns and "pnl" in df.columns:
                    leaderboard = (
                        df.groupby("player")
                        .agg(
                            bets=("id", "count"),
                            total_pnl=("pnl", "sum"),
                            avg_edge=("edge", "mean"),
                        )
                        .sort_values("total_pnl", ascending=False)
                        .head(20)
                        .reset_index()
                    )
                    leaderboard["total_pnl"] = leaderboard["total_pnl"].round(2)
                    leaderboard["avg_edge"] = (leaderboard["avg_edge"] * 100).round(2)
                    leaderboard.columns = ["Player", "Bets", "Total P&L", "Avg Edge %"]
                    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        else:
            st.caption("No CLV data available yet. Settle some bets with closing line data first.")

    # -- Book Efficiency ---------------------------------------------------
    with ins_tabs[1]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>BOOK EFFICIENCY ANALYSIS</div>",
            unsafe_allow_html=True,
        )
        # Analyze by source/book from bet history
        all_bets = bet_service.get_bet_history()
        if all_bets:
            df = pd.DataFrame(all_bets)
            if "notes" in df.columns:
                # Extract book from notes or bet_id pattern
                df["book"] = df["notes"].apply(
                    lambda x: _extract_book(str(x)) if x else "unknown"
                )
                book_stats = (
                    df[df["status"].isin(["won", "lost"])]
                    .groupby("book")
                    .agg(
                        bets=("id", "count"),
                        wins=("status", lambda x: (x == "won").sum()),
                        avg_edge=("edge", "mean"),
                        total_pnl=("pnl", "sum"),
                    )
                    .sort_values("total_pnl", ascending=False)
                    .reset_index()
                )
                if not book_stats.empty:
                    book_stats["win_rate"] = (
                        book_stats["wins"] / book_stats["bets"] * 100
                    ).round(1)
                    book_stats["avg_edge"] = (book_stats["avg_edge"] * 100).round(2)
                    book_stats["total_pnl"] = book_stats["total_pnl"].round(2)
                    st.dataframe(book_stats, use_container_width=True, hide_index=True)
                else:
                    st.caption("Not enough settled bets for book analysis.")
            else:
                st.caption("Bet history doesn't contain book identification data.")
        else:
            st.caption("No bet history available for analysis.")

    # -- Prop Breakdown ---------------------------------------------------
    with ins_tabs[2]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>PERFORMANCE BY PROP TYPE</div>",
            unsafe_allow_html=True,
        )
        edge_summary = report_service.get_edge_summary()
        if edge_summary:
            # Edge distribution chart
            edge_dist = edge_summary.get("edge_distribution", {})
            if edge_dist:
                edge_distribution_chart(edge_dist)

            # Top markets table
            top_markets = edge_summary.get("top_markets", [])
            if top_markets:
                df = pd.DataFrame(top_markets)
                df.columns = ["Market", "Count", "Avg Edge %"]
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Settled performance
            if edge_summary.get("settled_count", 0) > 0:
                st.metric(
                    "Settled P&L (7d)",
                    f"${edge_summary.get('settled_pnl', 0):.2f}",
                )
        else:
            st.caption("No edge data available.")

    # -- Bayesian Priors ---------------------------------------------------
    with ins_tabs[3]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>BAYESIAN PRIOR ANALYSIS</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-family:{FONT_MONO};font-size:0.65rem;color:#6A8AAA;"
            f"line-height:1.8;'>"
            f"The model uses a Bayesian prior to blend the statistical projection "
            f"with the market-implied probability.<br><br>"
            f"<b>Market Prior Weight</b> controls this blend: "
            f"1.0 = pure statistical model, 0.0 = pure market-implied.<br>"
            f"Current setting: <b>{st.session_state.get('market_prior_weight', 0.65):.2f}</b>"
            f"<br><br>"
            f"Lower weights are more conservative (trust the market more). "
            f"Higher weights are more aggressive (trust the model more).<br>"
            f"Recommended range: 0.55 - 0.75 depending on model calibration quality.</div>",
            unsafe_allow_html=True,
        )

        # Show calibration error as a quality indicator
        cal_data = report_service.get_calibration_data()
        if cal_data:
            total_bets = sum(int(b.get("n_bets", 0)) for b in cal_data)
            errors = [abs(float(b.get("calibration_error", 0))) for b in cal_data]
            n_bets_list = [int(b.get("n_bets", 0)) for b in cal_data]
            avg_error = sum(e * n for e, n in zip(errors, n_bets_list)) / max(total_bets, 1)

            if avg_error < 0.03:
                recommendation = "Excellent calibration. Consider raising market_prior_weight to 0.70-0.80."
                rec_color = COLOR_PRIMARY
            elif avg_error < 0.06:
                recommendation = "Good calibration. Current weight is likely appropriate."
                rec_color = "#FFB800"
            else:
                recommendation = "Model needs recalibration. Consider lowering market_prior_weight to 0.50-0.60."
                rec_color = "#FF3358"

            st.markdown(
                f"<div style='background:#060D18;border-left:3px solid {rec_color};"
                f"padding:0.6rem 0.8rem;margin-top:0.8rem;font-family:{FONT_MONO};"
                f"font-size:0.65rem;color:#B0C8E0;'>"
                f"Calibration Error: {avg_error:.4f}<br>{recommendation}</div>",
                unsafe_allow_html=True,
            )


def _extract_book(notes: str) -> str:
    """Try to extract sportsbook name from bet notes."""
    notes_lower = notes.lower()
    for book in ["fanduel", "draftkings", "betmgm", "bet365", "caesars",
                  "pointsbet", "prizepicks", "underdog", "sleeper"]:
        if book in notes_lower:
            return book
    return "unknown"
