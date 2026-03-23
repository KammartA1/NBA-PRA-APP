"""
streamlit_app/pages/settings_tab.py
===================================
SETTINGS tab -- all settings read/written via settings_service.
Save button writes to DB immediately.
"""
from __future__ import annotations

import os

import streamlit as st

from services import settings_service
from streamlit_app.config import (
    FONT_DISPLAY, FONT_MONO, COLOR_PRIMARY, COLOR_WARNING, COLOR_DANGER,
    get_user_id, get_anthropic_key,
)
from streamlit_app.state import save_user_settings


def render() -> None:
    user_id = get_user_id()

    st.markdown(
        f"<div style='font-family:{FONT_DISPLAY};font-size:0.68rem;color:#4A607A;"
        f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1.2rem;'>"
        f"ACCOUNT & ENGINE SETTINGS</div>",
        unsafe_allow_html=True,
    )

    # Load current settings from DB
    all_settings = settings_service.get_all_settings(user_id)

    col1, col2 = st.columns(2)

    with col1:
        # -- Account -------------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;"
            f"padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; ACCOUNT</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='background:#060D18;border:1px solid #0E1E30;border-radius:4px;"
            f"padding:0.7rem 0.9rem;margin-bottom:0.8rem;'>"
            f"<div style='font-family:{FONT_MONO};font-size:0.58rem;color:#2A6080;"
            f"margin-bottom:4px;'>USERNAME</div>"
            f"<div style='font-family:{FONT_DISPLAY};font-size:1rem;font-weight:700;"
            f"color:{COLOR_PRIMARY};'>{user_id}</div></div>",
            unsafe_allow_html=True,
        )

        # -- Bankroll -------------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;"
            f"margin-top:0.8rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; BANKROLL</div>",
            unsafe_allow_html=True,
        )
        bankroll_val = float(all_settings.get("bankroll", 1000.0))
        set_br = st.number_input(
            "Bankroll ($)", min_value=0.0,
            value=bankroll_val, step=50.0,
            key="settings_bankroll",
        )

        # -- Model Parameters -----------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;"
            f"margin-top:0.8rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; MODEL PARAMETERS</div>",
            unsafe_allow_html=True,
        )
        set_mpw = st.slider(
            "Model Weight vs Market", 0.0, 1.0,
            float(all_settings.get("market_prior_weight", 0.65)), 0.05,
            help="1.0 = pure statistical model. 0.0 = pure market implied prob.",
            key="settings_mpw",
        )
        set_ng = st.slider(
            "Sample Window (games)", 5, 30,
            int(all_settings.get("n_games", 10)),
            key="settings_ng",
        )
        set_fk = st.slider(
            "Fractional Kelly", 0.0, 1.0,
            float(all_settings.get("frac_kelly", 0.25)), 0.05,
            key="settings_fk",
        )
        set_pm = st.number_input(
            "Parlay Payout (x)", min_value=1.0,
            value=float(all_settings.get("payout_multi", 3.0)), step=0.1,
            key="settings_pm",
        )

        # -- Filters --------------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;"
            f"margin-top:0.8rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; FILTERS</div>",
            unsafe_allow_html=True,
        )
        set_exc = st.checkbox(
            "Block Chaotic Regime",
            value=bool(all_settings.get("exclude_chaotic", True)),
            help="Filters high-CV / blowout-risk environments",
            key="settings_exc",
        )
        set_su = st.checkbox(
            "Show Under Opportunities",
            value=bool(all_settings.get("show_unders", False)),
            help="Surface high-probability Unders alongside each leg",
            key="settings_su",
        )

    with col2:
        # -- Risk Controls --------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;"
            f"padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; RISK CONTROLS</div>",
            unsafe_allow_html=True,
        )
        set_mrpb = st.slider(
            "Max Bet Size (% Bankroll)", 0.0, 10.0,
            float(all_settings.get("max_risk_per_bet", 3.0)), 0.5,
            key="settings_mrpb",
        )
        set_mdl = st.slider(
            "Daily Loss Stop (%)", 0, 50,
            int(all_settings.get("max_daily_loss", 15)),
            key="settings_mdl",
        )
        set_mwl = st.slider(
            "Weekly Loss Stop (%)", 0, 50,
            int(all_settings.get("max_weekly_loss", 25)),
            key="settings_mwl",
        )

        # Risk summary display
        rclr = COLOR_DANGER if set_mrpb >= 7 else (COLOR_WARNING if set_mrpb >= 4 else COLOR_PRIMARY)
        st.markdown(
            f"<div style='background:#04080F;border:1px solid #0E1E30;border-radius:4px;"
            f"padding:0.5rem 0.7rem;margin:0.4rem 0 0.8rem 0;display:flex;gap:1rem;"
            f"align-items:center;'>"
            f"<div style='flex:1;text-align:center;'>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;'>DAILY STOP</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.85rem;color:#FFB800;"
            f"font-weight:600;'>{set_mdl}%</div></div>"
            f"<div style='width:1px;height:28px;background:#0E1E30;'></div>"
            f"<div style='flex:1;text-align:center;'>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;'>WEEKLY STOP</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.85rem;color:#FFB800;"
            f"font-weight:600;'>{set_mwl}%</div></div>"
            f"<div style='width:1px;height:28px;background:#0E1E30;'></div>"
            f"<div style='flex:1;text-align:center;'>"
            f"<div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;'>MAX BET</div>"
            f"<div style='font-family:{FONT_MONO};font-size:0.85rem;color:{rclr};"
            f"font-weight:600;'>{set_mrpb:.1f}%</div></div></div>",
            unsafe_allow_html=True,
        )

        # -- Odds API -------------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.6rem;"
            f"padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; ODDS API</div>",
            unsafe_allow_html=True,
        )
        set_mrd = st.number_input(
            "Max API requests/day", 1, 500,
            int(all_settings.get("max_req_day", 100)), 10,
            key="settings_mrd",
        )

        # -- Claude AI -------------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-top:0.8rem;"
            f"margin-bottom:0.6rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; CLAUDE AI ENGINE</div>",
            unsafe_allow_html=True,
        )
        ai_key = get_anthropic_key()
        if ai_key:
            st.markdown(
                f"<div style='background:linear-gradient(90deg,#00FFB208,transparent);"
                f"border:1px solid #00FFB230;border-radius:3px;padding:0.4rem 0.7rem;"
                f"display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;'>"
                f"<div style='width:6px;height:6px;border-radius:50%;background:{COLOR_PRIMARY};"
                f"box-shadow:0 0 6px {COLOR_PRIMARY};'></div>"
                f"<div style='font-family:{FONT_MONO};font-size:0.62rem;color:{COLOR_PRIMARY};'>"
                f"CLAUDE AI ACTIVE</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Claude AI key not configured. Add ANTHROPIC_API_KEY to "
                "Streamlit secrets to enable AI for all users."
            )

        with st.expander("Override API Key (optional)", expanded=False):
            ai_override = st.text_input(
                "Personal Anthropic API Key",
                value=st.session_state.get("_anthropic_key_override", ""),
                type="password",
                help="Leave blank to use the app's shared key.",
                key="settings_ai_key_input",
            )
            if ai_override != st.session_state.get("_anthropic_key_override", ""):
                st.session_state["_anthropic_key_override"] = ai_override
                if ai_override:
                    os.environ["ANTHROPIC_API_KEY"] = ai_override

        # -- Watchlist -------------------------------------------------------
        st.markdown(
            f"<div style='font-family:{FONT_DISPLAY};font-size:0.60rem;color:#2A5070;"
            f"letter-spacing:0.18em;text-transform:uppercase;margin-top:0.8rem;"
            f"margin-bottom:0.6rem;padding-bottom:0.4rem;border-bottom:1px solid #0E1E30;'>"
            f"&#9656; WATCHLIST</div>",
            unsafe_allow_html=True,
        )
        watchlist = settings_service.get_setting(user_id, "watchlist", [])
        wl_add = st.text_input(
            "Add player to watchlist", placeholder="LeBron James",
            key="settings_wl_add",
        )
        wl_c1, wl_c2 = st.columns(2)
        if wl_c1.button("Add", key="settings_wl_add_btn"):
            if wl_add.strip() and wl_add.strip() not in watchlist:
                watchlist.append(wl_add.strip())
                settings_service.set_setting(user_id, "watchlist", watchlist)
                st.rerun()

        if watchlist:
            wl_rm = st.selectbox("Remove player", ["--"] + watchlist, key="settings_wl_rm")
            if wl_c2.button("Remove", key="settings_wl_rm_btn") and wl_rm != "--":
                watchlist = [p for p in watchlist if p != wl_rm]
                settings_service.set_setting(user_id, "watchlist", watchlist)
                st.rerun()

            pills = "".join([
                f"<div style='display:inline-block;background:#0A1828;border:1px solid #1A3050;"
                f"border-radius:3px;padding:0.15rem 0.5rem;margin:0.15rem;"
                f"font-family:{FONT_MONO};font-size:0.62rem;color:#6AABCF;'>{p}</div>"
                for p in watchlist
            ])
            st.markdown(
                f"<div style='margin-top:0.4rem;'>{pills}</div>",
                unsafe_allow_html=True,
            )

    # -- Save button --------------------------------------------------------
    st.markdown("---")
    if st.button("SAVE ALL SETTINGS", use_container_width=True, type="primary"):
        # Collect all values from widget keys and persist to DB
        settings_to_save = {
            "bankroll": float(set_br),
            "market_prior_weight": float(set_mpw),
            "n_games": int(set_ng),
            "frac_kelly": float(set_fk),
            "payout_multi": float(set_pm),
            "exclude_chaotic": bool(set_exc),
            "show_unders": bool(set_su),
            "max_risk_per_bet": float(set_mrpb),
            "max_daily_loss": int(set_mdl),
            "max_weekly_loss": int(set_mwl),
            "max_req_day": int(set_mrd),
        }

        # Write bankroll with history tracking
        settings_service.set_bankroll(user_id, float(set_br))

        # Write model/risk params
        settings_service.set_model_params(user_id, settings_to_save)

        # Also update session state so other tabs see changes immediately
        for k, v in settings_to_save.items():
            st.session_state[k] = v

        st.success("All settings saved to database.")

    st.markdown(
        f"<div style='margin-top:1rem;font-family:{FONT_MONO};font-size:0.60rem;"
        f"color:#2A5070;border-top:1px solid #0E1E30;padding-top:0.6rem;'>"
        f"Settings are persisted to the database. If Streamlit crashes, "
        f"your settings will be restored on next startup.</div>",
        unsafe_allow_html=True,
    )
