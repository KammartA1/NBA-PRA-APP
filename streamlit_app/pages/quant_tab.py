"""
streamlit_app/pages/quant_tab.py
================================
QUANT SYSTEM tab -- dashboard, bet logger, CLV tracker, risk & sizing,
backtest display.  All data from services / quant_system read-only.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import bet_service, report_service, settings_service
from streamlit_app.config import (
    FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_DANGER, COLOR_WARNING,
    get_user_id,
)
from streamlit_app.components.charts import pnl_curve, clv_chart


def render() -> None:
    user_id = get_user_id()

    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"QUANT SYSTEM</div>",
        unsafe_allow_html=True,
    )

    # Check quant system availability
    quant_available = False
    quant_error = ""
    try:
        from quant_system.engine import QuantEngine
        from quant_system.core.types import Sport, BetType, SystemState
        quant_available = True
    except Exception as e:
        quant_error = str(e)

    if not quant_available:
        st.warning(f"Quant System not available: {quant_error}")

    qs_tabs = st.tabs(["Dashboard", "Bet Logger", "CLV Tracker", "Risk & Sizing", "Backtest"])

    # -- Dashboard ---------------------------------------------------------
    with qs_tabs[0]:
        _render_dashboard(user_id)

    # -- Bet Logger --------------------------------------------------------
    with qs_tabs[1]:
        _render_bet_logger(user_id)

    # -- CLV Tracker -------------------------------------------------------
    with qs_tabs[2]:
        _render_clv_tracker()

    # -- Risk & Sizing -----------------------------------------------------
    with qs_tabs[3]:
        _render_risk_sizing(user_id)

    # -- Backtest ----------------------------------------------------------
    with qs_tabs[4]:
        _render_backtest(quant_available)


def _render_dashboard(user_id: str) -> None:
    """System dashboard with key metrics from DB."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>SYSTEM DASHBOARD</div>",
        unsafe_allow_html=True,
    )

    # System state
    sys_state = report_service.get_system_state()
    state_name = sys_state.get("state", "unknown").upper()
    state_colors = {
        "ACTIVE": COLOR_PRIMARY,
        "REDUCED": COLOR_WARNING,
        "SUSPENDED": "#FF6B3D",
        "KILLED": COLOR_DANGER,
    }
    state_color = state_colors.get(state_name, "#4A607A")

    st.markdown(
        f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
        f"padding:0.6rem 0.8rem;margin-bottom:0.8rem;display:flex;"
        f"align-items:center;gap:0.8rem;'>"
        f"<div style='width:10px;height:10px;border-radius:50%;background:{state_color};"
        f"box-shadow:0 0 8px {state_color};'></div>"
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.85rem;font-weight:700;"
        f"color:{state_color};'>{state_name}</div>"
        f"<div style='margin-left:auto;font-family:{FONT_MONO};font-size:0.55rem;"
        f"color:#4A607A;'>{sys_state.get('reason', '')}</div></div>",
        unsafe_allow_html=True,
    )

    # Key metrics from summaries
    daily = report_service.get_latest_report("daily")
    weekly = report_service.get_latest_report("weekly")
    clv = bet_service.get_clv_summary()

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Today's Signals", daily.get("total_signals", 0) if daily else 0)
    mc2.metric("Today's P&L", f"${daily.get('pnl', 0):.2f}" if daily else "$0.00")
    mc3.metric("Week's P&L", f"${weekly.get('pnl', 0):.2f}" if weekly else "$0.00")
    mc4.metric("Avg CLV", f"{clv.get('avg_clv_cents', 0):.1f}c" if clv else "0.0c")

    bankroll = settings_service.get_bankroll(user_id)
    drawdown = sys_state.get("drawdown")

    mc5, mc6, mc7, mc8 = st.columns(4)
    mc5.metric("Bankroll", f"${bankroll:,.2f}")
    mc6.metric(
        "Drawdown",
        f"{drawdown:.1f}%" if drawdown is not None else "--",
    )
    mc7.metric(
        "Weekly Win Rate",
        f"{weekly.get('win_rate', 0):.1f}%" if weekly else "--",
    )
    mc8.metric(
        "Weekly ROI",
        f"{weekly.get('roi_pct', 0):.1f}%" if weekly else "--",
    )

    # P&L chart
    pnl_history = report_service.get_report_history("daily", days=30)
    if pnl_history:
        pnl_curve(pnl_history, title="30-Day P&L")


def _render_bet_logger(user_id: str) -> None:
    """Manual bet logging form."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>LOG A BET</div>",
        unsafe_allow_html=True,
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        log_player = st.text_input("Player", key="ql_player", placeholder="LeBron James")
        log_market = st.text_input("Market (stat type)", key="ql_market", placeholder="Points")
        log_line = st.number_input("Line", min_value=0.0, step=0.5, key="ql_line")
        log_direction = st.selectbox("Direction", ["over", "under"], key="ql_direction")
    with bc2:
        log_stake = st.number_input("Stake ($)", min_value=0.0, step=5.0, key="ql_stake")
        log_odds = st.number_input(
            "Decimal Odds", min_value=1.01, value=1.909, step=0.01, key="ql_odds",
        )
        log_model_prob = st.number_input(
            "Model Prob (%)", 0.0, 100.0, 55.0, 0.5, key="ql_model_prob",
        )
        log_notes = st.text_input("Notes", key="ql_notes")

    if st.button("LOG BET", use_container_width=True, key="ql_log_btn", type="primary"):
        if not log_player or log_stake <= 0:
            st.warning("Enter player name and stake amount.")
        else:
            bet_data = {
                "player": log_player,
                "market": log_market,
                "line": float(log_line),
                "direction": log_direction,
                "stake": float(log_stake),
                "price_decimal": float(log_odds),
                "model_prob": float(log_model_prob) / 100.0,
                "market_prob": 1.0 / float(log_odds) if log_odds > 1 else 0.5,
                "edge": (float(log_model_prob) / 100.0) - (1.0 / float(log_odds) if log_odds > 1 else 0.5),
                "notes": log_notes,
            }
            row_id = bet_service.place_bet(bet_data)
            if row_id > 0:
                st.success(f"Bet logged (ID: {row_id}).")
            else:
                st.error("Failed to log bet. Check service logs.")

    # Recent bets
    st.markdown("---")
    pending = bet_service.get_pending_bets()
    if pending:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.12em;margin-bottom:0.4rem;'>"
            f"RECENT PENDING ({len(pending)})</div>",
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(pending[:10])
        cols = ["player", "stat_type", "line", "direction", "stake", "odds_decimal", "timestamp"]
        available = [c for c in cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True, hide_index=True)


def _render_clv_tracker() -> None:
    """CLV performance display."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>CLOSING LINE VALUE TRACKER</div>",
        unsafe_allow_html=True,
    )

    window = st.slider("Analysis window (bets)", 20, 500, 100, 10, key="clv_window")
    clv_data = bet_service.get_clv_summary(window=int(window))

    if clv_data and clv_data.get("n_bets", 0) > 0:
        mc1, mc2, mc3, mc4 = st.columns(4)
        avg_clv = clv_data.get("avg_clv_cents", 0)
        clv_color = COLOR_PRIMARY if avg_clv > 0 else COLOR_DANGER
        mc1.metric("Avg CLV (cents)", f"{avg_clv:.1f}")
        mc2.metric("Beat Close %", f"{clv_data.get('beat_close_pct', 0):.1f}%")
        mc3.metric("Bets", clv_data.get("n_bets", 0))
        mc4.metric("Total CLV Raw", f"{clv_data.get('total_clv_raw', 0):.1f}")

        st.markdown(
            f"<div style='background:#060D18;border-left:3px solid {clv_color};"
            f"padding:0.5rem 0.7rem;margin-top:0.5rem;font-family:{FONT_MONO};"
            f"font-size:0.62rem;color:#8BA8BF;'>"
            f"{'Positive CLV = you are consistently beating the closing line. This is the #1 indicator of long-term profitability.' if avg_clv > 0 else 'Negative CLV = the market is closing against you. Review bet selection criteria and timing.'}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No CLV data yet. CLV is calculated when bets are settled with closing line information.")


def _render_risk_sizing(user_id: str) -> None:
    """Risk limits and position sizing display."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>RISK MANAGEMENT & SIZING</div>",
        unsafe_allow_html=True,
    )

    limits = settings_service.get_risk_limits(user_id)

    # Display limits
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.markdown(
            f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
            f"padding:0.5rem;text-align:center;'>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;'>DAILY STOP</div>"
            f"<div style='font-family:{FONT_MONO};font-size:1rem;color:#FFB800;font-weight:600;'>"
            f"{limits.get('max_daily_loss', 15)}%</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#4A607A;'>"
            f"${limits.get('max_daily_loss_amount', 0):.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with lc2:
        st.markdown(
            f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
            f"padding:0.5rem;text-align:center;'>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;'>WEEKLY STOP</div>"
            f"<div style='font-family:{FONT_MONO};font-size:1rem;color:#FFB800;font-weight:600;'>"
            f"{limits.get('max_weekly_loss', 25)}%</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#4A607A;'>"
            f"${limits.get('max_weekly_loss_amount', 0):.2f}</div></div>",
            unsafe_allow_html=True,
        )
    with lc3:
        mrpb = float(limits.get("max_risk_per_bet", 3.0))
        rclr = COLOR_DANGER if mrpb >= 7 else (COLOR_WARNING if mrpb >= 4 else COLOR_PRIMARY)
        st.markdown(
            f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
            f"padding:0.5rem;text-align:center;'>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;'>MAX BET</div>"
            f"<div style='font-family:{FONT_MONO};font-size:1rem;color:{rclr};font-weight:600;'>"
            f"{mrpb:.1f}%</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#4A607A;'>"
            f"${limits.get('max_single_bet_amount', 0):.2f}</div></div>",
            unsafe_allow_html=True,
        )

    # Current exposure
    st.markdown("---")
    pending = bet_service.get_pending_bets()
    total_exposure = sum(float(b.get("stake", 0)) for b in pending)
    bankroll = limits.get("bankroll", 1000)
    exposure_pct = total_exposure / max(bankroll, 1) * 100

    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
        f"letter-spacing:0.12em;margin-bottom:0.4rem;'>CURRENT EXPOSURE</div>",
        unsafe_allow_html=True,
    )
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Pending Bets", len(pending))
    ec2.metric("Total Exposure", f"${total_exposure:.2f}")
    ec3.metric("Exposure %", f"{exposure_pct:.1f}%")

    max_exposure = float(limits.get("max_concurrent_exposure", 10.0))
    if exposure_pct > max_exposure:
        st.error(
            f"Exposure ({exposure_pct:.1f}%) exceeds limit ({max_exposure:.0f}%). "
            f"Reduce position sizes or wait for settlements."
        )


def _render_backtest(quant_available: bool) -> None:
    """Backtest results display (read-only from quant_system)."""
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>BACKTEST RESULTS</div>",
        unsafe_allow_html=True,
    )

    if not quant_available:
        st.info("Quant system not available for backtesting.")
        return

    try:
        from quant_system.backtest.runner import BacktestRunner
        from quant_system.backtest.data_loader import BacktestDataLoader

        st.markdown(
            f"<div style='font-family:{FONT_MONO};font-size:0.62rem;color:#6A8AAA;"
            f"line-height:1.8;margin-bottom:0.8rem;'>"
            f"Run backtests from the command line:<br>"
            f"<code>python -m quant_system.backtest.runner --days 90</code><br><br>"
            f"Results will be displayed here once available.</div>",
            unsafe_allow_html=True,
        )

        # Try to load most recent backtest results from DB
        try:
            from quant_system.db.schema import get_session
            session = get_session()
            # Check for backtest results table
            result = session.execute(
                __import__("sqlalchemy").text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_results'"
                )
            ).fetchone()
            if result:
                rows = session.execute(
                    __import__("sqlalchemy").text(
                        "SELECT * FROM backtest_results ORDER BY run_date DESC LIMIT 1"
                    )
                ).fetchone()
                if rows:
                    st.success("Most recent backtest results found.")
                    st.json(dict(rows._mapping) if hasattr(rows, "_mapping") else {})
            session.close()
        except Exception:
            pass

    except ImportError:
        st.info(
            "Backtest module not found. Ensure quant_system.backtest is installed."
        )
    except Exception as e:
        st.caption(f"Backtest loader error: {e}")
