"""
streamlit_app/pages/scanner_tab.py
==================================
LIVE SCANNER tab -- calls scanner_service.scan_all_books() or reads
cached scan results.  Displays value opportunities sorted by edge.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from services import scanner_service, odds_service, settings_service
from streamlit_app.config import (
    MARKET_OPTIONS, FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_WARNING,
    get_user_id,
)
from streamlit_app.state import get_ui_state, set_ui_state, get_model_settings


def render() -> None:
    user_id = get_user_id()
    model_settings = get_model_settings(user_id)

    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"LIVE SCANNER -- VALUE OPPORTUNITIES</div>",
        unsafe_allow_html=True,
    )

    # -- Background worker freshness banner ---------------------------------
    # Shows when the always-on GitHub Actions worker last refreshed the data.
    # Green = fresh, yellow = getting stale, red = worker may be down.
    _render_freshness_banner()

    # -- Controls -----------------------------------------------------------
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 1])
    with ctrl1:
        scan_source = st.selectbox(
            "Source", ["All sources", "Odds API only", "PP + UD only"],
            key="scanner_source",
        )
    with ctrl2:
        selected_markets = st.multiselect(
            "Markets",
            options=MARKET_OPTIONS,
            default=["Points", "Rebounds", "Assists", "3PM", "PRA"],
            key="scanner_markets",
        )
    with ctrl3:
        scan_date = st.date_input("Date", value=date.today(), key="scanner_date")

    # Threshold controls
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        min_prob = st.slider("Min Probability", 0.50, 0.75, 0.53, 0.01, key="scanner_min_prob")
    with tc2:
        min_ev = st.slider("Min EV %", 0.0, 15.0, 1.0, 0.5, key="scanner_min_ev")
    with tc3:
        max_results = st.number_input("Max Results", 10, 200, 60, 10, key="scanner_max_results")

    # -- Run scan -----------------------------------------------------------
    if st.button("SCAN NOW", use_container_width=True, type="primary"):
        with st.spinner("Scanning lines across all books..."):
            results = scanner_service.scan_all_books(
                markets=selected_markets,
                book="all",
                game_date=scan_date,
                settings=model_settings,
                min_prob=min_prob,
                min_adv=0.01,
                min_ev=min_ev / 100.0,
                max_results=int(max_results),
                scan_source=scan_source,
            )
        set_ui_state("scanner_results", results)

    # -- Sharp movements alert bar -----------------------------------------
    movements = scanner_service.scan_sharp_movements()
    if movements:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:{COLOR_WARNING};"
            f"letter-spacing:0.12em;margin-bottom:0.4rem;'>"
            f"SHARP LINE MOVEMENTS ({len(movements)} detected)</div>",
            unsafe_allow_html=True,
        )
        with st.expander("View sharp movements", expanded=False):
            for mv in movements[:10]:
                direction_icon = "^" if mv["direction"] == "UP" else "v"
                st.caption(
                    f"{mv['player']} | {mv['stat_type']} | "
                    f"{mv['source']} | {mv['opening_line']} -> {mv['current_line']} "
                    f"({direction_icon} {abs(mv['delta']):.1f})"
                )

    # -- Display results ---------------------------------------------------
    results = get_ui_state("scanner_results")

    # Fall back to cached results from disk / DB
    if results is None:
        results = scanner_service.get_scanner_results()
        if results:
            set_ui_state("scanner_results", results)

    if not results:
        st.info("No scan results yet. Click SCAN NOW to find value opportunities.")
        return

    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
        f"letter-spacing:0.12em;margin-bottom:0.6rem;'>"
        f"{len(results)} OPPORTUNITIES FOUND</div>",
        unsafe_allow_html=True,
    )

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    avg_ev = sum(float(r.get("ev_adj_pct", 0)) for r in results) / max(len(results), 1)
    avg_sharp = sum(float(r.get("sharp", 0) or 0) for r in results) / max(len(results), 1)
    m1.metric("Opportunities", len(results))
    m2.metric("Avg EV %", f"{avg_ev:.1f}%")
    m3.metric("Avg Sharpness", f"{avg_sharp:.0f}")
    m4.metric("Sources", ", ".join(set(str(r.get("src", "?")) for r in results)))

    # Results table
    display_cols = [
        "src", "player", "market", "line", "p_cal", "p_implied",
        "advantage", "ev_adj_pct", "proj", "edge_cat", "sharp",
        "sharp_tier", "trend", "team", "opp", "stake_$",
    ]
    df = pd.DataFrame(results)
    available_cols = [c for c in display_cols if c in df.columns]
    if available_cols:
        st.dataframe(
            df[available_cols],
            use_container_width=True,
            hide_index=True,
            height=500,
        )

    # -- Send to MODEL tab --------------------------------------------------
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
        f"letter-spacing:0.12em;margin-top:1rem;margin-bottom:0.4rem;'>"
        f"SEND TO MODEL TAB</div>",
        unsafe_allow_html=True,
    )
    for i, row in enumerate(results[:20]):
        col_btn, col_info = st.columns([1, 5])
        with col_btn:
            if st.button("Load", key=f"scan_load_{i}"):
                st.session_state["_staged_pname_1"] = row.get("player", "")
                st.session_state["_staged_mkt_1"] = row.get("market", "Points")
                st.session_state["_staged_mline_1"] = float(row.get("line", 22.5))
                st.session_state["_staged_manual_1"] = False
                if row.get("src") in ("PP", "UD", "SL"):
                    st.session_state["_staged_manual_1"] = True
                st.session_state["_auto_run_model"] = True
                st.rerun()
        with col_info:
            st.caption(
                f"{row.get('player', '?')} | {row.get('market', '?')} "
                f"{row.get('line', '?')} | EV {row.get('ev_adj_pct', 0):.1f}% | "
                f"Sharp {row.get('sharp', 0)}"
            )


def _render_freshness_banner() -> None:
    """Show how long ago the always-on background worker last refreshed data."""
    try:
        from core import db
        age = db.last_scan_age_minutes("NBA")
    except Exception:
        age = None

    if age is None:
        st.warning(
            "⚠️ No background scan on record yet. The automated worker hasn't "
            "completed a scan, or the database isn't reachable."
        )
        return

    if age < 90:
        color, icon, label = "#00FFB2", "🟢", "fresh"
    elif age < 180:
        color, icon, label = "#FFC857", "🟡", "getting stale"
    else:
        color, icon, label = "#FF5C5C", "🔴", "stale — worker may be down"

    if age < 60:
        ago = f"{age:.0f} min ago"
    else:
        ago = f"{age/60:.1f} hr ago"

    st.markdown(
        f"<div style='font-family:{FONT_MONO};font-size:0.62rem;color:{color};"
        f"letter-spacing:0.06em;margin-bottom:0.8rem;'>"
        f"{icon} Background worker last scan: {ago} ({label})</div>",
        unsafe_allow_html=True,
    )
