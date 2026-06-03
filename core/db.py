"""
core/db.py — Standalone Supabase client for background workers.
No Streamlit dependency. Uses environment variables only.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

_client = None
_checked = False


def _get_client():
    global _client, _checked
    if _checked:
        return _client
    _checked = True
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        log.warning("SUPABASE_URL/KEY not set — DB writes disabled")
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        log.info("Connected to Supabase")
        return _client
    except Exception as e:
        log.error("Supabase init failed: %s", e)
        return None


def _retry(fn, retries=2, delay=1.0):
    last_err = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(delay * (2 ** i))
    raise last_err


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


# --- Scan Results ---

def write_scan_results(scan_id: str, sport: str, edges: list[dict]) -> int:
    sb = _get_client()
    if not sb:
        return 0
    rows = []
    for e in edges:
        rows.append({
            "scan_id": scan_id,
            "sport": sport,
            "scanned_at": _utcnow(),
            "player": e.get("player", ""),
            "team": e.get("team", ""),
            "opp": e.get("opp", ""),
            "market": e.get("market", ""),
            "line": e.get("line"),
            "side": e.get("side", "Over"),
            "proj": e.get("proj"),
            "p_cal": e.get("p_cal"),
            "ev_pct": e.get("ev_pct"),
            "edge_cat": e.get("edge_cat", ""),
            "l5_avg": e.get("l5_avg"),
            "l10_avg": e.get("l10_avg"),
            "src": e.get("src", "PrizePicks"),
            "is_active": True,
        })
    if not rows:
        return 0
    try:
        _retry(lambda: sb.table("scan_results").insert(rows).execute())
        log.info("Wrote %d scan results (scan_id=%s)", len(rows), scan_id)
        return len(rows)
    except Exception as e:
        log.error("write_scan_results failed: %s", e)
        return 0


def deactivate_old_scans(sport: str, keep_scan_id: str):
    sb = _get_client()
    if not sb:
        return
    try:
        _retry(lambda: (
            sb.table("scan_results")
            .update({"is_active": False})
            .eq("sport", sport)
            .eq("is_active", True)
            .neq("scan_id", keep_scan_id)
            .execute()
        ))
    except Exception as e:
        log.warning("deactivate_old_scans failed: %s", e)


def load_active_scan(sport: str = "NBA") -> list[dict]:
    sb = _get_client()
    if not sb:
        return []
    try:
        resp = _retry(lambda: (
            sb.table("scan_results")
            .select("*")
            .eq("sport", sport)
            .eq("is_active", True)
            .order("ev_pct", desc=True)
            .limit(100)
            .execute()
        ))
        return resp.data or []
    except Exception as e:
        log.warning("load_active_scan failed: %s", e)
        return []


# --- Worker Run Log ---

def log_worker_run(worker_name: str, status: str, details: dict | None = None) -> bool:
    sb = _get_client()
    if not sb:
        return False
    try:
        data = {
            "worker_name": worker_name,
            "status": status,
            "ran_at": _utcnow(),
            "details": details or {},
        }
        _retry(lambda: sb.table("worker_runs").insert(data).execute())
        return True
    except Exception as e:
        log.warning("log_worker_run failed: %s", e)
        return False


def get_recent_worker_runs(hours: int = 24, worker_name: str | None = None) -> list[dict]:
    """Return worker_runs from the last `hours` hours, newest first."""
    sb = _get_client()
    if not sb:
        return []
    try:
        from datetime import timedelta
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        q = sb.table("worker_runs").select("*").gte("ran_at", since)
        if worker_name:
            q = q.eq("worker_name", worker_name)
        resp = _retry(lambda: q.order("ran_at", desc=True).limit(500).execute())
        return resp.data or []
    except Exception as e:
        log.warning("get_recent_worker_runs failed: %s", e)
        return []


def last_scan_age_minutes(sport: str = "NBA") -> float | None:
    """Minutes since the last successful/partial scanner run. None if never."""
    runs = get_recent_worker_runs(hours=48, worker_name="scanner")
    for r in runs:
        if r.get("status") in ("success", "partial"):
            try:
                ran = datetime.fromisoformat(r["ran_at"].replace("Z", "+00:00"))
                delta = datetime.now(timezone.utc) - ran
                return round(delta.total_seconds() / 60.0, 1)
            except Exception:
                continue
    return None


# --- Notification Log ---

def log_notification(ntype: str, message: str, delivered: bool = True) -> bool:
    sb = _get_client()
    if not sb:
        return False
    try:
        data = {
            "sent_at": _utcnow(),
            "type": ntype,
            "message": message[:2000],
            "delivered": delivered,
        }
        _retry(lambda: sb.table("notification_log").insert(data).execute())
        return True
    except Exception as e:
        log.warning("log_notification failed: %s", e)
        return False


def was_notified_today(ntype: str) -> bool:
    """True if a notification of `ntype` was already logged today (UTC)."""
    sb = _get_client()
    if not sb:
        return False
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        resp = _retry(lambda: (
            sb.table("notification_log")
            .select("id")
            .eq("type", ntype)
            .gte("sent_at", f"{today}T00:00:00+00:00")
            .limit(1)
            .execute()
        ))
        return bool(resp.data)
    except Exception as e:
        log.warning("was_notified_today failed: %s", e)
        return False


# --- Prop Line History (CLV tracking) ---

def append_pp_lines(lines: list[dict]) -> int:
    sb = _get_client()
    if not sb:
        return 0
    rows = []
    for ln in lines:
        rows.append({
            "ts": _utcnow(),
            "player": ln.get("player", ""),
            "market": ln.get("market", ""),
            "line": ln.get("line"),
            "price": None,
            "book": "PrizePicks",
            "event_id": ln.get("event_id"),
        })
    if not rows:
        return 0
    try:
        for chunk in [rows[i:i+50] for i in range(0, len(rows), 50)]:
            _retry(lambda c=chunk: sb.table("prop_line_history").insert(c).execute())
        return len(rows)
    except Exception as e:
        log.warning("append_pp_lines failed: %s", e)
        return 0
