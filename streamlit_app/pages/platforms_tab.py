"""
streamlit_app/pages/platforms_tab.py
====================================
PLATFORMS tab -- PrizePicks, Underdog, Sleeper lines display.
Line history from odds_service.  Best Available comparison.
No scraping here -- workers handle that.  We only read from DB.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services import odds_service
from streamlit_app.config import FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY


def render() -> None:
    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"PLATFORM LINES & COMPARISON</div>",
        unsafe_allow_html=True,
    )

    plat_tabs = st.tabs(["PrizePicks", "Underdog", "Sleeper", "Line History", "Best Available"])

    # -- PrizePicks ---------------------------------------------------------
    with plat_tabs[0]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;'>PRIZEPICKS NBA LINES</div>",
            unsafe_allow_html=True,
        )
        pp_lines = odds_service.get_prizepicks_lines()
        if pp_lines:
            st.success(f"{len(pp_lines)} PrizePicks lines loaded from database.")
            df = pd.DataFrame(pp_lines)
            display_cols = ["player", "stat_type", "line", "odds_type", "start_time", "fetched_at"]
            available = [c for c in display_cols if c in df.columns]
            if available:
                # Filter controls
                fc1, fc2 = st.columns(2)
                with fc1:
                    stat_filter = st.multiselect(
                        "Filter by stat",
                        options=sorted(df["stat_type"].unique().tolist()) if "stat_type" in df.columns else [],
                        key="pp_stat_filter",
                    )
                with fc2:
                    player_search = st.text_input("Search player", key="pp_player_search")

                filtered = df
                if stat_filter:
                    filtered = filtered[filtered["stat_type"].isin(stat_filter)]
                if player_search:
                    filtered = filtered[
                        filtered["player"].str.contains(player_search, case=False, na=False)
                    ]

                st.dataframe(filtered[available], use_container_width=True, hide_index=True, height=400)
        else:
            st.info(
                "No PrizePicks lines in database. Lines are fetched by background workers. "
                "Check that the PrizePicks scraper is running."
            )

        # Manual import fallback
        with st.expander("Manual Import (paste JSON)", expanded=False):
            raw = st.text_area("Paste PrizePicks JSON", key="pp_manual_json", height=150)
            if st.button("Import", key="pp_manual_import"):
                _import_manual_lines(raw, "prizepicks")

    # -- Underdog -----------------------------------------------------------
    with plat_tabs[1]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;'>UNDERDOG FANTASY LINES</div>",
            unsafe_allow_html=True,
        )
        ud_lines = odds_service.get_underdog_lines()
        if ud_lines:
            st.success(f"{len(ud_lines)} Underdog lines loaded from database.")
            df = pd.DataFrame(ud_lines)
            display_cols = ["player", "stat_type", "line", "source", "captured_at"]
            available = [c for c in display_cols if c in df.columns]
            if available:
                player_search = st.text_input("Search player", key="ud_player_search")
                filtered = df
                if player_search:
                    filtered = filtered[
                        filtered["player"].str.contains(player_search, case=False, na=False)
                    ]
                st.dataframe(filtered[available], use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No Underdog lines in database.")

        with st.expander("Manual Import (paste JSON)", expanded=False):
            raw = st.text_area("Paste Underdog JSON", key="ud_manual_json", height=150)
            if st.button("Import", key="ud_manual_import"):
                _import_manual_lines(raw, "underdog")

    # -- Sleeper ------------------------------------------------------------
    with plat_tabs[2]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;'>SLEEPER LINES</div>",
            unsafe_allow_html=True,
        )
        sl_lines = odds_service.get_sleeper_lines()
        if sl_lines:
            st.success(f"{len(sl_lines)} Sleeper lines loaded from database.")
            df = pd.DataFrame(sl_lines)
            display_cols = ["player", "stat_type", "line", "source", "captured_at"]
            available = [c for c in display_cols if c in df.columns]
            if available:
                st.dataframe(df[available], use_container_width=True, hide_index=True, height=400)
        else:
            st.info(
                "No Sleeper lines in database. Use Manual Import to paste lines "
                "exported from the Sleeper app."
            )

        with st.expander("Manual Import (paste JSON)", expanded=False):
            raw = st.text_area("Paste Sleeper JSON", key="sl_manual_json", height=150)
            if st.button("Import", key="sl_manual_import"):
                _import_manual_lines(raw, "sleeper")

    # -- Line History -------------------------------------------------------
    with plat_tabs[3]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:#2A5070;"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>LINE MOVEMENT HISTORY</div>",
            unsafe_allow_html=True,
        )
        lh1, lh2, lh3 = st.columns(3)
        with lh1:
            lh_player = st.text_input("Player", key="lh_player", placeholder="LeBron James")
        with lh2:
            lh_market = st.text_input("Market", key="lh_market", placeholder="points")
        with lh3:
            lh_hours = st.number_input("Hours back", 1, 168, 24, key="lh_hours")

        if lh_player:
            history = odds_service.get_line_history(
                player=lh_player,
                market=lh_market or "points",
                hours=int(lh_hours),
            )
            if history:
                df = pd.DataFrame(history)
                st.dataframe(df, use_container_width=True, hide_index=True)
                # Line chart
                if "captured_at" in df.columns and "line" in df.columns:
                    try:
                        chart_df = df.copy()
                        chart_df["captured_at"] = pd.to_datetime(chart_df["captured_at"])
                        chart_df = chart_df.set_index("captured_at")
                        st.line_chart(chart_df["line"])
                    except Exception:
                        pass
            else:
                st.caption("No line history found for this player/market.")
        else:
            st.caption("Enter a player name to view line history.")

    # -- Best Available -----------------------------------------------------
    with plat_tabs[4]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:#2A5070;"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>BEST AVAILABLE ACROSS BOOKS</div>",
            unsafe_allow_html=True,
        )
        ba1, ba2 = st.columns(2)
        with ba1:
            ba_player = st.text_input("Player", key="ba_player", placeholder="Jayson Tatum")
        with ba2:
            ba_market = st.text_input("Market", key="ba_market", placeholder="points")

        if ba_player and ba_market:
            best = odds_service.get_best_available(ba_player, ba_market)
            if best:
                st.markdown(
                    f"<div style='background:#060D18;border:1px solid #0E1E30;"
                    f"border-radius:4px;padding:0.7rem;margin-top:0.5rem;'>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.7rem;color:{COLOR_PRIMARY};'>"
                    f"Best line: <b>{best.get('line', '--')}</b> "
                    f"@ {best.get('source', 'unknown')}</div>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.6rem;color:#6A8AAA;'>"
                    f"Over implied: {_pct(best.get('over_prob_implied'))} | "
                    f"Captured: {best.get('captured_at', '--')}</div></div>",
                    unsafe_allow_html=True,
                )

                # All lines for comparison
                all_lines = odds_service.get_lines_for_player(ba_player, ba_market)
                if all_lines:
                    st.markdown("**All available lines:**")
                    df = pd.DataFrame(all_lines)
                    st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.caption("No lines found for this player/market.")
        else:
            st.caption("Enter player and market to compare lines.")


def _import_manual_lines(raw: str, source: str) -> None:
    """Parse manually pasted JSON lines and store via odds_service."""
    if not raw.strip():
        st.warning("No data pasted.")
        return
    import json
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            st.error("Expected a JSON array of line objects.")
            return

        lines = []
        for item in data:
            lines.append({
                "player": item.get("player", item.get("player_name", "")),
                "stat_type": item.get("stat_type", item.get("market", "")),
                "source": source,
                "line": float(item.get("line", item.get("line_score", 0))),
                "is_opening": False,
            })

        if lines:
            odds_service.store_lines(lines)
            st.success(f"Imported {len(lines)} {source} lines into database.")
        else:
            st.warning("No valid lines found in JSON.")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
    except Exception as e:
        st.error(f"Import failed: {e}")


def _pct(val) -> str:
    if val is None:
        return "--"
    try:
        return f"{float(val) * 100:.1f}%"
    except Exception:
        return str(val)
