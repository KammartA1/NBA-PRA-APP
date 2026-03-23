"""
streamlit_app/pages/alerts_tab.py
=================================
ALERTS tab -- alert rules stored in user_settings via settings_service.
Injury data from player_service.  News/RSS integration.
"""
from __future__ import annotations

import streamlit as st

from services import settings_service, player_service
from streamlit_app.config import (
    FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_WARNING, COLOR_DANGER,
    get_user_id,
)


def render() -> None:
    user_id = get_user_id()

    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>"
        f"ALERTS & INJURY MONITOR</div>",
        unsafe_allow_html=True,
    )

    alert_tabs = st.tabs(["Injury Report", "Alert Rules", "News Feed"])

    # -- Injury Report -----------------------------------------------------
    with alert_tabs[0]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_DANGER};"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>NBA INJURY REPORT</div>",
            unsafe_allow_html=True,
        )

        if st.button("Refresh Injury Report", key="refresh_injuries"):
            st.session_state.pop("_injury_cache", None)

        # Cache injury data in transient session state to avoid repeated API calls
        if "_injury_cache" not in st.session_state:
            with st.spinner("Fetching injury report..."):
                st.session_state["_injury_cache"] = player_service.get_injury_report()

        injuries = st.session_state.get("_injury_cache", {})

        if injuries:
            # Summary
            total_out = sum(
                sum(1 for p in players if p.get("status", "").upper() == "OUT")
                for players in injuries.values()
            )
            total_gtd = sum(
                sum(1 for p in players if p.get("status", "").upper() in ("QUESTIONABLE", "DOUBTFUL"))
                for players in injuries.values()
            )
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Teams Affected", len(injuries))
            mc2.metric("Players OUT", total_out)
            mc3.metric("Game-Time Decisions", total_gtd)

            # Team-by-team display
            for team_abbr in sorted(injuries.keys()):
                players = injuries[team_abbr]
                with st.expander(f"{team_abbr} ({len(players)} injured)", expanded=False):
                    for p in players:
                        status = p.get("status", "").upper()
                        status_color = {
                            "OUT": COLOR_DANGER,
                            "DOUBTFUL": "#FF6B3D",
                            "QUESTIONABLE": COLOR_WARNING,
                            "PROBABLE": COLOR_PRIMARY,
                            "DAY-TO-DAY": COLOR_WARNING,
                        }.get(status, "#4A607A")
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"padding:2px 0;font-family:{FONT_MONO};font-size:0.65rem;'>"
                            f"<span style='color:#B0C8E0;'>{p.get('player', 'Unknown')}</span>"
                            f"<span style='color:{status_color};font-weight:600;'>{status}</span>"
                            f"</div>"
                            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;"
                            f"color:#4A607A;margin-bottom:4px;'>{p.get('reason', '')}</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.caption("No injury data available or API unavailable.")

    # -- Alert Rules -------------------------------------------------------
    with alert_tabs[1]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>CUSTOM ALERT RULES</div>",
            unsafe_allow_html=True,
        )

        # Load existing rules from settings
        alert_rules = settings_service.get_setting(user_id, "alert_rules", [])

        # Display existing rules
        if alert_rules:
            for i, rule in enumerate(alert_rules):
                rc1, rc2 = st.columns([5, 1])
                with rc1:
                    rule_type = rule.get("type", "unknown")
                    rule_desc = _format_rule(rule)
                    st.markdown(
                        f"<div style='background:#060D18;border:1px solid #0E1E30;"
                        f"border-radius:3px;padding:0.4rem 0.6rem;margin-bottom:0.3rem;"
                        f"font-family:{FONT_MONO};font-size:0.62rem;color:#8BA8BF;'>"
                        f"{rule_desc}</div>",
                        unsafe_allow_html=True,
                    )
                with rc2:
                    if st.button("X", key=f"del_rule_{i}"):
                        alert_rules.pop(i)
                        settings_service.set_setting(user_id, "alert_rules", alert_rules)
                        st.rerun()
        else:
            st.caption("No alert rules configured.")

        # Add new rule
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.12em;margin-top:0.8rem;margin-bottom:0.4rem;'>"
            f"ADD NEW RULE</div>",
            unsafe_allow_html=True,
        )
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            rule_type = st.selectbox(
                "Type",
                ["player_injury", "line_movement", "ev_threshold", "sharp_movement"],
                key="new_rule_type",
            )
        with nc2:
            rule_target = st.text_input("Target (player/market)", key="new_rule_target")
        with nc3:
            rule_threshold = st.number_input(
                "Threshold", value=0.0, step=0.5, key="new_rule_threshold",
            )

        if st.button("Add Rule", key="add_rule_btn"):
            if rule_target:
                new_rule = {
                    "type": rule_type,
                    "target": rule_target,
                    "threshold": float(rule_threshold),
                    "active": True,
                }
                alert_rules.append(new_rule)
                settings_service.set_setting(user_id, "alert_rules", alert_rules)
                st.success(f"Added {rule_type} alert for {rule_target}")
                st.rerun()
            else:
                st.warning("Enter a target for the alert rule.")

    # -- News Feed ---------------------------------------------------------
    with alert_tabs[2]:
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.62rem;color:{COLOR_PRIMARY};"
            f"letter-spacing:0.12em;margin-bottom:0.6rem;'>NBA NEWS FEED</div>",
            unsafe_allow_html=True,
        )
        _render_news_feed()


def _format_rule(rule: dict) -> str:
    """Format an alert rule for display."""
    rule_type = rule.get("type", "?")
    target = rule.get("target", "?")
    threshold = rule.get("threshold", 0)

    type_labels = {
        "player_injury": "INJURY ALERT",
        "line_movement": "LINE MOVE",
        "ev_threshold": "EV THRESHOLD",
        "sharp_movement": "SHARP MOVE",
    }
    label = type_labels.get(rule_type, rule_type.upper())

    if rule_type == "ev_threshold":
        return f"[{label}] {target} | Min EV: {threshold:.1f}%"
    elif rule_type == "line_movement":
        return f"[{label}] {target} | Move >= {threshold:.1f} pts"
    elif rule_type == "sharp_movement":
        return f"[{label}] {target} | Sharp >= {threshold:.1f} pts"
    else:
        return f"[{label}] {target}"


def _render_news_feed() -> None:
    """Render a simple NBA news feed from ESPN RSS or similar."""
    try:
        import requests
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.ok:
            data = resp.json()
            articles = data.get("articles", [])[:15]
            for article in articles:
                headline = article.get("headline", "")
                description = article.get("description", "")
                published = article.get("published", "")
                link = ""
                links = article.get("links", {})
                if isinstance(links, dict):
                    web = links.get("web", {})
                    link = web.get("href", "") if isinstance(web, dict) else ""

                st.markdown(
                    f"<div style='border-bottom:1px solid #0E1E30;padding:0.4rem 0;'>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.68rem;"
                    f"color:#B0C8E0;font-weight:600;'>{headline}</div>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.58rem;"
                    f"color:#4A607A;margin-top:2px;'>{description[:200]}</div>"
                    f"<div style='font-family:{FONT_MONO};font-size:0.50rem;"
                    f"color:#2A4060;margin-top:2px;'>{published[:16]}"
                    + ('  |  <a href="' + link + '" target="_blank" style="color:#00AAFF;">Read more</a>' if link else '') +
                    "</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Could not load news feed.")
    except Exception:
        st.caption("News feed unavailable (network error).")
