"""
streamlit_app/components/sidebar.py
===================================
Sidebar: user info, bankroll, system status, quick actions.
All data comes from service layer -- no computation here.
"""
from __future__ import annotations

import streamlit as st

from services import settings_service
from streamlit_app.config import (
    APP_VERSION, COLOR_PRIMARY, COLOR_MUTED, FONT_MONO, FONT_DISPLAY,
    get_anthropic_key, get_user_id,
)
from streamlit_app.state import save_user_settings


def render_sidebar() -> str:
    """Render the sidebar and return the authenticated ``user_id``."""
    with st.sidebar:
        # -- Brand header --------------------------------------------------
        st.markdown(f"""
<div style='padding:1rem 0.2rem 0.8rem 0.2rem;border-bottom:1px solid #0E1E30;
            margin-bottom:0.8rem;'>
    <div style='font-family:{FONT_DISPLAY};font-size:0.58rem;color:#1E4060;
                letter-spacing:0.25em;text-transform:uppercase;margin-bottom:0.2rem;'>
        QUANTITATIVE ENGINE</div>
    <div style='font-family:{FONT_DISPLAY};font-size:1.05rem;font-weight:700;
                color:#EEF4FF;letter-spacing:0.08em;line-height:1.1;'>
        NBA PROP<br><span style='color:{COLOR_PRIMARY};'>ALPHA</span> ENGINE</div>
    <div style='display:flex;align-items:center;gap:0.5rem;margin-top:0.45rem;'>
        <div style='width:6px;height:6px;border-radius:50%;background:{COLOR_PRIMARY};
                    box-shadow:0 0 6px {COLOR_PRIMARY};'></div>
        <div style='font-family:{FONT_MONO};font-size:0.58rem;color:{COLOR_PRIMARY};
                    letter-spacing:0.1em;'>LIVE  &#183;  {APP_VERSION}</div>
        <div style='margin-left:auto;font-family:{FONT_MONO};font-size:0.55rem;
                    color:#2A4060;'>NBA 2024-25</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # -- Account info --------------------------------------------------
        user_id = get_user_id()
        st.markdown(f"""
<div style='background:linear-gradient(135deg,#00FFB208,#00AAFF06);
            border:1px solid #0E2840;border-radius:4px;
            padding:0.5rem 0.7rem;margin-bottom:0.6rem;'>
    <div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;
                letter-spacing:0.10em;margin-bottom:3px;'>SIGNED IN AS</div>
    <div style='font-family:{FONT_DISPLAY};font-size:0.85rem;font-weight:700;
                color:{COLOR_PRIMARY};letter-spacing:0.06em;'>{user_id}</div>
</div>
""", unsafe_allow_html=True)

        # -- Bankroll (read from DB, write on change) -----------------------
        st.markdown(f"""<div style='font-family:{FONT_DISPLAY};font-size:0.55rem;
color:#2A5070;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.4rem;'>
&#9656; BANKROLL</div>""", unsafe_allow_html=True)

        db_bankroll = settings_service.get_bankroll(user_id)
        display_val = float(st.session_state.get("bankroll", db_bankroll))

        bankroll = st.number_input(
            "Bankroll ($)", min_value=0.0, value=display_val, step=50.0,
            key="sidebar_bankroll",
        )
        # Sync to session state and persist if changed
        if float(bankroll) != float(st.session_state.get("bankroll", db_bankroll)):
            st.session_state["bankroll"] = float(bankroll)
            settings_service.set_bankroll(user_id, float(bankroll))

        br_val = float(st.session_state.get("bankroll", bankroll))
        st.markdown(f"""
<div style='background:#04080F;border:1px solid #0E2840;border-radius:4px;
            padding:0.35rem 0.7rem;margin:0.2rem 0 0.8rem 0;
            display:flex;justify-content:space-between;align-items:center;'>
    <div style='font-family:{FONT_MONO};font-size:0.55rem;color:#2A6080;'>CAPITAL</div>
    <div style='font-family:{FONT_MONO};font-size:0.95rem;font-weight:700;
                color:{COLOR_PRIMARY};'>${br_val:,.2f}</div>
</div>
""", unsafe_allow_html=True)

        # -- Claude AI status -----------------------------------------------
        ai_key = get_anthropic_key()
        if ai_key:
            st.markdown(f"""
<div style='background:linear-gradient(90deg,#00FFB208,transparent);
            border:1px solid #00FFB230;border-radius:3px;
            padding:0.3rem 0.6rem;display:flex;align-items:center;gap:0.5rem;
            margin-bottom:0.4rem;'>
    <div style='width:5px;height:5px;border-radius:50%;background:{COLOR_PRIMARY};
                box-shadow:0 0 5px {COLOR_PRIMARY};'></div>
    <div style='font-family:{FONT_MONO};font-size:0.60rem;color:{COLOR_PRIMARY};
                letter-spacing:0.06em;'>CLAUDE AI  &#183;  ACTIVE</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style='background:#04080F;border:1px solid #1A2030;border-radius:3px;
            padding:0.3rem 0.6rem;display:flex;align-items:center;gap:0.5rem;
            margin-bottom:0.4rem;'>
    <div style='width:5px;height:5px;border-radius:50%;background:#2A4060;'></div>
    <div style='font-family:{FONT_MONO};font-size:0.60rem;color:#2A4060;
                letter-spacing:0.06em;'>CLAUDE AI  &#183;  OFFLINE</div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style='font-family:{FONT_MONO};font-size:0.58rem;
color:#2A5070;margin-bottom:0.6rem;'>&#8594; Configure in <b style='color:#8BA8BF;'>SETTINGS</b> tab</div>""",
            unsafe_allow_html=True)

        # -- Worker / system status -----------------------------------------
        _render_worker_status()

        # -- Sign out -------------------------------------------------------
        st.markdown("""<div style='margin-top:0.4rem;padding-top:0.6rem;
border-top:1px solid #0E1E30;'></div>""", unsafe_allow_html=True)
        if st.button("SIGN OUT", use_container_width=True, key="sidebar_signout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        # -- Footer ---------------------------------------------------------
        st.markdown(f"""
<div style='margin-top:1.5rem;padding-top:0.6rem;border-top:1px solid #080F1A;
            text-align:center;'>
    <div style='font-family:{FONT_MONO};font-size:0.52rem;color:#1A2F45;
                letter-spacing:0.08em;line-height:1.8;'>
        SPORTSBOOK MODEL &#183; -110 VIG REMOVED<br>
        KELLY STAKING &#183; MC CORRELATED<br>
        <span style='color:#0E2840;'>&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;&#9473;</span><br>
        <span style='color:#1A3050;'>NBA QUANT ENGINE {APP_VERSION}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    return user_id


def _render_worker_status() -> None:
    """Show background worker health from the worker_status table."""
    try:
        from database.connection import session_scope
        from database.models import WorkerStatus
        with session_scope() as session:
            workers = session.query(WorkerStatus).order_by(WorkerStatus.worker_name).all()
            if not workers:
                return
            rows_html = ""
            for w in workers:
                status = w.status or "idle"
                color = {
                    "running": COLOR_PRIMARY,
                    "idle": COLOR_MUTED,
                    "error": "#FF3358",
                }.get(status, COLOR_MUTED)
                rows_html += (
                    f"<div style='display:flex;justify-content:space-between;padding:1px 0;'>"
                    f"<span style='color:#6A8AAA;font-size:0.55rem;'>{w.worker_name}</span>"
                    f"<span style='color:{color};font-size:0.55rem;font-weight:600;'>{status.upper()}</span>"
                    f"</div>"
                )
            st.markdown(f"""
<div style='margin-top:0.6rem;padding-top:0.5rem;border-top:1px solid #0E1E30;'>
    <div style='font-family:{FONT_DISPLAY};font-size:0.55rem;color:#2A5070;
                letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.3rem;'>
        &#9656; WORKERS</div>
    <div style='font-family:{FONT_MONO};'>{rows_html}</div>
</div>""", unsafe_allow_html=True)
    except Exception:
        pass
