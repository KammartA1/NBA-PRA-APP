"""
core/notify.py — Telegram notifications for background workers.
No Streamlit dependency. Uses env vars TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import requests

log = logging.getLogger(__name__)


def _get_config() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    return token, chat_id


def is_configured() -> bool:
    token, chat_id = _get_config()
    return bool(token and chat_id)


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    token, chat_id = _get_config()
    if not token or not chat_id:
        log.warning("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }, timeout=15)
        if resp.status_code == 200:
            log.info("Telegram message sent (%d chars)", len(text))
            return True
        log.warning("Telegram API error %d: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as e:
        log.error("Telegram send failed: %s", e)
        return False


def send_edge_alert(edges: list[dict], sport: str = "NBA") -> bool:
    if not edges:
        return False
    header = f"🏀 <b>{sport} Edge Alert</b> — {len(edges)} high-EV props found\n"
    lines = []
    for e in edges[:8]:
        ev = e.get("ev_pct", 0)
        prob = (e.get("p_cal", 0) or 0) * 100
        lines.append(
            f"• <b>{e.get('player','?')}</b> {e.get('market','?')} "
            f"{e.get('side','Over')} {e.get('line','?')} "
            f"— Proj: {e.get('proj', 0):.1f}, P(hit): {prob:.0f}%, EV: {ev:+.1f}%"
        )
    body = "\n".join(lines)
    footer = "\n\n<i>Open the app for full analysis →</i>"
    return send_message(header + body + footer)


def send_worker_status(status: str, details: str = "") -> bool:
    msg = f"⚙️ <b>Worker Status:</b> {status}"
    if details:
        msg += f"\n{details}"
    return send_message(msg)


def test_connection() -> bool:
    return send_message("✅ NBA-PRA Worker notification test — connection verified!")
