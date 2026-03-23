"""
PAGE 1: COMMAND CENTER
Bloomberg Terminal-style overview for the NBA Prop Alpha Engine.
"""

from datetime import datetime

import streamlit as st

from services.ui_bridge import UIBridge
from streamlit_app.design import (
    metric_card,
    status_dot,
    badge,
    card_container,
    kill_switch_bar,
    table_html,
    COLORS,
)


def render():
    bridge = UIBridge()
    data = bridge.get_command_center_data()

    bankroll = data.get("bankroll", 0)
    today_pnl = data.get("today_pnl", 0)
    today_pnl_delta = data.get("today_pnl_delta", 0)
    edge_status = data.get("edge_status", False)
    clv_50bet = data.get("clv_50bet", 0.0)
    kill_switches = data.get("kill_switches", [])
    worker_statuses = data.get("worker_statuses", [])
    todays_signals = data.get("todays_signals", [])
    exposure = data.get("exposure", {})
    system_state = data.get("system_state", "ACTIVE")
    model_version = data.get("model_version", "N/A")
    data_quality_score = data.get("data_quality_score", 0.0)
    simulation_status = data.get("simulation_status", "UNKNOWN")

    # ------------------------------------------------------------------
    # Kill-switch alert banner (must be first visible element)
    # ------------------------------------------------------------------
    triggered = [
        sw for sw in kill_switches if sw.get("status") != "active"
    ]
    if triggered:
        reasons = ", ".join(
            f"{sw.get('name', '?')}: {sw.get('reason', 'triggered')}"
            for sw in triggered
        )
        st.error(f"KILL SWITCH TRIGGERED  --  {reasons}")

    # ------------------------------------------------------------------
    # TOP BAR
    # ------------------------------------------------------------------
    top_left, top_center, top_right = st.columns([3, 4, 3])

    with top_left:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:16px;'
            f"font-weight:600;text-transform:uppercase;letter-spacing:0.12em;"
            f'color:{COLORS["text_primary"]};line-height:64px;">'
            f"NBA PROP ALPHA ENGINE</div>",
            unsafe_allow_html=True,
        )

    with top_center:
        if system_state == "ACTIVE":
            dot_html = status_dot("active")
            state_label = "ACTIVE"
            state_color = COLORS["green"]
        else:
            dot_html = status_dot("dead")
            state_label = f"HALTED: {system_state}"
            state_color = COLORS["red"]
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:center;'
            f'gap:8px;line-height:64px;height:64px;">'
            f'{dot_html}'
            f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;'
            f'font-weight:600;color:{state_color};text-transform:uppercase;'
            f'letter-spacing:0.08em;">{state_label}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with top_right:
        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        st.markdown(
            f'<div style="text-align:right;font-family:JetBrains Mono,monospace;'
            f"font-size:12px;color:{COLORS['text_secondary']};"
            f'line-height:64px;">{now_str}</div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # ROW 1 -- Hero Metrics
    # ------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        pnl_sign = "+" if today_pnl >= 0 else ""
        st.markdown(
            metric_card(
                "BANKROLL",
                f"${bankroll:,.0f}",
                delta=f"{pnl_sign}${today_pnl:,.0f} today",
                delta_color="green" if today_pnl >= 0 else "red",
            ),
            unsafe_allow_html=True,
        )

    with m2:
        pnl_str = f"+${today_pnl:,.0f}" if today_pnl >= 0 else f"-${abs(today_pnl):,.0f}"
        st.markdown(
            metric_card(
                "TODAY P/L",
                pnl_str,
                delta=f"{today_pnl_delta:+.1f}% ROI" if today_pnl_delta else None,
                delta_color="green" if today_pnl >= 0 else "red",
            ),
            unsafe_allow_html=True,
        )

    with m3:
        edge_val = "YES" if edge_status else "NO"
        edge_color = COLORS["green"] if edge_status else COLORS["red"]
        st.markdown(
            metric_card(
                "EDGE STATUS",
                f'<span style="color:{edge_color}">{edge_val}</span>',
            ),
            unsafe_allow_html=True,
        )

    with m4:
        clv_cents = clv_50bet * 100
        clv_str = f"{clv_cents:+.1f}\u00a2"
        st.markdown(
            metric_card(
                "CLV (50-BET)",
                clv_str,
                delta_color="green" if clv_cents >= 0 else "red",
            ),
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # ROW 2 -- Kill Switch Status Bar
    # ------------------------------------------------------------------
    if kill_switches:
        st.markdown(kill_switch_bar(kill_switches), unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # ROW 3 -- Signals + Exposure
    # ------------------------------------------------------------------
    sig_col, exp_col = st.columns([3, 2])

    with sig_col:
        st.markdown(
            f'<h2 style="margin-top:16px;">TODAY\'S SIGNALS</h2>',
            unsafe_allow_html=True,
        )
        if todays_signals:
            rows = []
            for sig in todays_signals[:10]:
                edge_pct = sig.get("edge_pct", 0)
                action_badge = (
                    badge("TAKE", "green") if edge_pct > 0 else badge("SKIP", "red")
                )
                rows.append([
                    sig.get("player", ""),
                    sig.get("market", ""),
                    f"{edge_pct:+.1f}%",
                    f"${sig.get('rec_stake', 0):,.0f}",
                    action_badge,
                ])
            st.markdown(
                table_html(
                    ["Player", "Market", "Edge %", "Stake", "Action"],
                    rows,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="color:{COLORS["text_secondary"]};'
                f'font-family:IBM Plex Sans,sans-serif;font-size:12px;'
                f'padding:24px 0;">No signals generated today</div>',
                unsafe_allow_html=True,
            )

    with exp_col:
        st.markdown(
            '<h2 style="margin-top:16px;">EXPOSURE</h2>',
            unsafe_allow_html=True,
        )
        total_wagered = exposure.get("total_wagered", 0)
        by_confidence = exposure.get("by_confidence", {})

        total_html = (
            f'<div style="margin-bottom:8px;">'
            f'<span style="color:{COLORS["text_secondary"]};font-size:11px;'
            f'text-transform:uppercase;letter-spacing:0.06em;">Total Wagered</span>'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:22px;'
            f'font-weight:600;color:{COLORS["text_primary"]};margin-top:4px;">'
            f'${total_wagered:,.0f}</div></div>'
        )
        st.markdown(card_container(total_html), unsafe_allow_html=True)

        tiers_html = ""
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            amt = by_confidence.get(tier, 0)
            tier_color = {
                "HIGH": COLORS["green"],
                "MEDIUM": COLORS["amber"],
                "LOW": COLORS["text_secondary"],
            }.get(tier, COLORS["text_secondary"])
            tiers_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:6px 0;'
                f'border-bottom:1px solid {COLORS["border"]};">'
                f'<span style="color:{tier_color};font-size:11px;font-weight:600;'
                f'letter-spacing:0.04em;">{tier}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:12px;'
                f'color:{COLORS["text_primary"]};">${amt:,.0f}</span>'
                f"</div>"
            )
        st.markdown(
            card_container(
                f'<div style="margin-bottom:4px;color:{COLORS["text_secondary"]};'
                f'font-size:11px;text-transform:uppercase;letter-spacing:0.06em;">'
                f"By Confidence</div>{tiers_html}"
            ),
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # ROW 4 -- System Health
    # ------------------------------------------------------------------
    w_col, mod_col, dq_col = st.columns(3)

    with w_col:
        st.markdown('<h2 style="margin-top:16px;">WORKERS</h2>', unsafe_allow_html=True)
        worker_html = ""
        if worker_statuses:
            for w in worker_statuses:
                dot = status_dot(w.get("status", "unknown"))
                name = w.get("name", "unknown")
                last_run = w.get("last_run", "never")
                worker_html += (
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f"padding:6px 0;border-bottom:1px solid {COLORS['border']};\">"
                    f"{dot}"
                    f'<span style="font-size:11px;font-weight:500;'
                    f'color:{COLORS["text_primary"]};flex:1;">{name}</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;'
                    f'font-size:10px;color:{COLORS["text_secondary"]};">{last_run}</span>'
                    f"</div>"
                )
        else:
            worker_html = (
                f'<div style="color:{COLORS["text_secondary"]};font-size:11px;'
                f'padding:8px 0;">No workers reporting</div>'
            )
        st.markdown(card_container(worker_html), unsafe_allow_html=True)

    with mod_col:
        st.markdown('<h2 style="margin-top:16px;">MODEL INFO</h2>', unsafe_allow_html=True)
        sim_color = {
            "PASS": COLORS["green"],
            "FAIL": COLORS["red"],
        }.get(simulation_status, COLORS["amber"])
        model_html = (
            f'<div style="padding:4px 0;">'
            f'<div style="color:{COLORS["text_secondary"]};font-size:11px;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
            f"Version</div>"
            f'<div style="font-family:JetBrains Mono,monospace;font-size:14px;'
            f'font-weight:600;color:{COLORS["text_primary"]};">{model_version}</div>'
            f"</div>"
            f'<div style="padding:8px 0 4px 0;border-top:1px solid {COLORS["border"]};'
            f'margin-top:8px;">'
            f'<div style="color:{COLORS["text_secondary"]};font-size:11px;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">'
            f"Simulation</div>"
            f'<div style="font-family:JetBrains Mono,monospace;font-size:14px;'
            f'font-weight:600;color:{sim_color};">{simulation_status}</div>'
            f"</div>"
        )
        st.markdown(card_container(model_html), unsafe_allow_html=True)

    with dq_col:
        st.markdown(
            '<h2 style="margin-top:16px;">DATA QUALITY</h2>',
            unsafe_allow_html=True,
        )
        if data_quality_score >= 80:
            dq_color = COLORS["green"]
        elif data_quality_score >= 50:
            dq_color = COLORS["amber"]
        else:
            dq_color = COLORS["red"]
        dq_html = (
            f'<div style="text-align:center;padding:8px 0;">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:36px;'
            f'font-weight:700;color:{dq_color};">{data_quality_score:.0f}</div>'
            f'<div style="color:{COLORS["text_secondary"]};font-size:11px;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">'
            f"/ 100</div></div>"
        )
        st.markdown(card_container(dq_html), unsafe_allow_html=True)
