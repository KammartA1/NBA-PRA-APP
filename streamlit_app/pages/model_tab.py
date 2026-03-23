"""
streamlit_app/pages/model_tab.py
================================
MODEL tab -- user configures up to 4 legs, clicks RUN, results display.
All computation delegated to projection_service.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from services import projection_service, settings_service
from streamlit_app.config import MARKET_OPTIONS, get_user_id, FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY
from streamlit_app.state import get_ui_state, set_ui_state, get_model_settings


def render() -> None:
    user_id = get_user_id()
    bankroll = float(st.session_state.get("bankroll", settings_service.get_bankroll(user_id)))
    model_settings = get_model_settings(user_id)

    # Loss stop check
    loss_stop_hit = _check_loss_stops(user_id, bankroll)

    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"CONFIGURE UP TO 4 LEGS</div>",
        unsafe_allow_html=True,
    )

    # -- Date and book selection -------------------------------------------
    date_col, book_col = st.columns([2, 2])
    with date_col:
        scan_date = st.date_input(
            "Lines Date", value=date.today(), key="model_date",
        )
    with book_col:
        # Get available sportsbooks (lazy import from app.py to avoid duplication)
        book_choices = _get_book_choices(scan_date)
        # Consume staged sportsbook from scanner -> model flow
        if "_staged_sportsbook" in st.session_state:
            sb = st.session_state.pop("_staged_sportsbook")
            if sb in book_choices:
                st.session_state["model_sportsbook"] = sb
        sportsbook = st.selectbox(
            "Sportsbook", options=book_choices, index=0, key="model_sportsbook",
        )

    # -- Consume staged scanner inputs ------------------------------------
    for si in range(1, 5):
        for prefix_src, prefix_dst in [
            ("_staged_pname_", "pname_"),
            ("_staged_mkt_", "mkt_"),
            ("_staged_mline_", "mline_"),
            ("_staged_manual_", "manual_"),
            ("_staged_out_", "out_"),
        ]:
            key_src = f"{prefix_src}{si}"
            key_dst = f"{prefix_dst}{si}"
            if key_src in st.session_state:
                st.session_state[key_dst] = st.session_state.pop(key_src)
    if "_staged_model_date" in st.session_state:
        st.session_state["model_date"] = st.session_state.pop("_staged_model_date")

    # -- Leg configuration UI -----------------------------------------------
    leg_configs = []
    for row_idx in range(2):
        cols = st.columns(2)
        for col_idx in range(2):
            leg_n = row_idx * 2 + col_idx + 1
            with cols[col_idx]:
                st.markdown(
                    f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;"
                    f"color:{COLOR_PRIMARY};letter-spacing:0.14em;text-transform:uppercase;"
                    f"margin-bottom:0.4rem;'>LEG {leg_n}</div>",
                    unsafe_allow_html=True,
                )
                pname = st.text_input(
                    "Player", key=f"pname_{leg_n}", placeholder="e.g. LeBron James",
                )
                mkt = st.selectbox("Market", options=MARKET_OPTIONS, key=f"mkt_{leg_n}")
                manual = st.checkbox("Manual line", key=f"manual_{leg_n}")
                if manual:
                    st.markdown(
                        "<div style='font-family:{};font-size:0.60rem;color:#FFB800;"
                        "letter-spacing:0.10em;margin:-4px 0 2px 0;'>"
                        "MANUAL -- not from Odds API</div>".format(FONT_DISPLAY),
                        unsafe_allow_html=True,
                    )
                mline = st.number_input(
                    "Line", min_value=0.0,
                    value=float(st.session_state.get(f"line_{leg_n}", 22.5)),
                    step=0.5, key=f"mline_{leg_n}",
                )
                out_cb = st.checkbox("Key teammate OUT?", key=f"out_{leg_n}")
                leg_configs.append((f"P{leg_n}", pname, mkt, manual, mline, out_cb))

    # Auto-run from scanner
    auto_run = bool(st.session_state.pop("_auto_run_model", False))
    if auto_run:
        st.info("Legs loaded from Live Scanner -- running model automatically...")

    run_btn = st.button(
        "RUN MODEL", use_container_width=True, disabled=loss_stop_hit,
    ) or auto_run

    # -- Run projections ---------------------------------------------------
    if run_btn:
        legs_to_run = []
        for tag, pname, mkt, manual, mline, out_cb in leg_configs:
            pname = pname.strip()
            if not pname:
                continue
            legs_to_run.append({
                "player_name": pname,
                "market": mkt,
                "line": float(mline),
                "book": None if manual else sportsbook,
                "game_date": scan_date,
                "key_teammate_out": out_cb,
            })

        if not legs_to_run:
            st.warning("Enter at least one player name to run the model.")
            return

        with st.spinner("Running projections..."):
            results = projection_service.run_multi_leg_projection(
                legs=legs_to_run,
                settings=model_settings,
            )

        # Store results in transient session state for display
        set_ui_state("last_results", results)

    # -- Display results ---------------------------------------------------
    results = get_ui_state("last_results", [])
    if results:
        _display_results(results)


def _display_results(results: list[dict]) -> None:
    """Render projection results."""
    for i, leg in enumerate(results):
        player = leg.get("player", "Unknown")
        market = leg.get("market", "")
        line = leg.get("line", 0)
        proj = leg.get("proj")
        p_cal = leg.get("p_cal")
        ev_adj = leg.get("ev_adj")
        gate_ok = leg.get("gate_ok", False)
        side = leg.get("side", "over")
        errors = leg.get("errors", [])

        st.markdown(f"---")
        header_color = COLOR_PRIMARY if gate_ok else "#FF3358"
        gate_label = "PASS" if gate_ok else "GATED"
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.75rem;font-weight:700;"
            f"color:{header_color};letter-spacing:0.08em;'>"
            f"{player} | {market} {side.upper()} {line} | {gate_label}</div>",
            unsafe_allow_html=True,
        )

        if errors:
            for err in errors:
                st.warning(err)
            continue

        if proj is None or p_cal is None:
            st.info("No projection data returned.")
            continue

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Projection", f"{proj:.1f}" if proj else "--")
        c2.metric("Win Prob", f"{p_cal * 100:.1f}%" if p_cal else "--")
        c3.metric(
            "EV",
            f"{ev_adj * 100:+.1f}%" if ev_adj else "--",
        )
        c4.metric(
            "Stake",
            f"${leg.get('stake', 0):.2f}" if leg.get("stake") else "--",
        )

        # Detail expander
        with st.expander("Full details", expanded=False):
            detail_cols = st.columns(3)
            detail_cols[0].markdown(
                f"**Edge Category:** {leg.get('edge_cat', '--')}\n\n"
                f"**Regime:** {leg.get('regime', '--')}\n\n"
                f"**Hot/Cold:** {leg.get('hot_cold', '--')}",
            )
            detail_cols[1].markdown(
                f"**Sharpness:** {_fmt(leg.get('sharpness_score'), 0)}/100 "
                f"({leg.get('sharpness_tier', '--')})\n\n"
                f"**Trend:** {leg.get('trend_label', '--')}\n\n"
                f"**Fatigue:** {leg.get('fatigue_label', 'Normal')}",
            )
            detail_cols[2].markdown(
                f"**Sigma:** {_fmt(leg.get('sigma'), 2)}\n\n"
                f"**Rest Days:** {leg.get('rest_days', '--')}\n\n"
                f"**B2B:** {'Yes' if leg.get('b2b') else 'No'}",
            )

            # Gate reason if gated
            if not gate_ok:
                st.error(f"Gate reason: {leg.get('gate_reason', 'unknown')}")


def _check_loss_stops(user_id: str, bankroll: float) -> bool:
    """Check daily loss stops via bet_service."""
    try:
        from services import bet_service
        pnl_summary = bet_service.get_pnl_summary(period="daily")
        total_pnl = pnl_summary.get("total_pnl", 0)
        max_daily_loss_pct = float(st.session_state.get("max_daily_loss", 15))
        if bankroll > 0 and total_pnl < 0 and abs(total_pnl) / bankroll * 100 > max_daily_loss_pct:
            st.error(
                f"DAILY LOSS STOP HIT ({abs(total_pnl) / bankroll * 100:.1f}%). "
                f"No new bets recommended today."
            )
            return True
    except Exception:
        pass
    return False


def _get_book_choices(scan_date) -> list[str]:
    """Return sportsbook choices.  Tries to load from app.py, falls back to defaults."""
    fallback = ["consensus", "fanduel", "draftkings", "betmgm", "bet365",
                "pointsbet", "caesars", "prizepicks"]
    try:
        import nba_engine as _app
        choices, err = _app.get_sportsbook_choices(scan_date.isoformat())
        if choices:
            return choices
    except Exception:
        pass
    return fallback


def _fmt(val, decimals: int = 2) -> str:
    if val is None:
        return "--"
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)
