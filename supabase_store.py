"""
supabase_store.py
=================
Persistent storage layer backed by Supabase (free Postgres).
Falls back to local files when Supabase is not configured.

All public functions are drop-in replacements for the local file
operations in app.py.  The module initialises lazily on first call
so the import itself never blocks or raises.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Supabase client singleton
# ---------------------------------------------------------------------------

_client = None
_client_checked = False


def _get_client():
    """Return the Supabase client, or None if not configured."""
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
        key = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
        if not url or not key:
            log.info("[supabase_store] No SUPABASE_URL/KEY configured — using local files")
            return None
        from supabase import create_client
        _client = create_client(url, key)
        log.info("[supabase_store] Connected to Supabase")
        return _client
    except Exception as e:
        log.warning("[supabase_store] Supabase init failed: %s — falling back to local files", e)
        return None


def is_available() -> bool:
    """Return True if Supabase is configured and reachable."""
    return _get_client() is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _retry(fn, retries=2, delay=1.0):
    """Retry a Supabase operation with exponential backoff."""
    last_err = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < retries:
                time.sleep(delay * (2 ** i))
    raise last_err


# ===================================================================
# BET HISTORY  (replaces history_{uid}.csv)
# ===================================================================

def append_history(uid: str, row: dict) -> bool:
    """Append a bet to history.  Returns True on success."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        data = {
            "ts": row.get("ts", datetime.utcnow().isoformat()),
            "user_id": uid or "default",
            "legs": json.loads(row["legs"]) if isinstance(row.get("legs"), str) else (row.get("legs") or []),
            "n_legs": int(row.get("n_legs", 1)),
            "leg_results": json.loads(row["leg_results"]) if isinstance(row.get("leg_results"), str) else (row.get("leg_results") or ["Pending"]),
            "result": str(row.get("result", "Pending")),
            "decision": str(row.get("decision", "BET")),
            "notes": str(row.get("notes", "")),
        }
        _retry(lambda: sb.table("bet_history").insert(data).execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] append_history failed: %s", e)
        return False


def load_history(uid: str) -> pd.DataFrame:
    """Load all bet history for a user.  Returns DataFrame matching CSV schema."""
    sb = _get_client()
    if sb is None:
        return pd.DataFrame()
    try:
        resp = _retry(lambda: (
            sb.table("bet_history")
            .select("*")
            .eq("user_id", uid or "default")
            .order("ts", desc=True)
            .execute()
        ))
        rows = resp.data or []
        if not rows:
            return pd.DataFrame()
        df_rows = []
        for r in rows:
            df_rows.append({
                "ts": r.get("ts", ""),
                "user_id": r.get("user_id", ""),
                "legs": json.dumps(r["legs"]) if not isinstance(r.get("legs"), str) else r["legs"],
                "n_legs": r.get("n_legs", 1),
                "leg_results": json.dumps(r["leg_results"]) if not isinstance(r.get("leg_results"), str) else r.get("leg_results", ""),
                "result": r.get("result", "Pending"),
                "decision": r.get("decision", "BET"),
                "notes": r.get("notes", ""),
                "_supa_id": r.get("id"),
            })
        return pd.DataFrame(df_rows)
    except Exception as e:
        log.warning("[supabase_store] load_history failed: %s", e)
        return pd.DataFrame()


def update_history_row(uid: str, row_id: int, updates: dict) -> bool:
    """Update a specific history row by its Supabase id."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        clean = {}
        for k, v in updates.items():
            if k in ("legs", "leg_results") and not isinstance(v, str):
                clean[k] = v  # Supabase accepts native JSON
            else:
                clean[k] = v
        _retry(lambda: sb.table("bet_history").update(clean).eq("id", row_id).execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] update_history_row failed: %s", e)
        return False


def delete_history_row(uid: str, row_id: int) -> bool:
    """Delete a single history row."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        _retry(lambda: sb.table("bet_history").delete().eq("id", row_id).execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] delete_history_row failed: %s", e)
        return False


def load_all_pending_bets() -> list:
    """Load ALL pending bets across all users (for background CLV updater)."""
    sb = _get_client()
    if sb is None:
        return []
    try:
        resp = _retry(lambda: (
            sb.table("bet_history")
            .select("*")
            .eq("result", "Pending")
            .execute()
        ))
        return resp.data or []
    except Exception as e:
        log.warning("[supabase_store] load_all_pending_bets failed: %s", e)
        return []


def clear_history(uid: str) -> bool:
    """Delete ALL history for a user."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        _retry(lambda: sb.table("bet_history").delete().eq("user_id", uid or "default").execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] clear_history failed: %s", e)
        return False


# ===================================================================
# USER SETTINGS  (replaces user_state_{uid}.json)
# ===================================================================

def load_user_settings(uid: str) -> dict:
    """Load user settings.  Returns dict or empty dict."""
    sb = _get_client()
    if sb is None:
        return {}
    try:
        resp = _retry(lambda: (
            sb.table("user_settings")
            .select("settings")
            .eq("user_id", uid or "default")
            .limit(1)
            .execute()
        ))
        if resp.data:
            return resp.data[0].get("settings", {})
        return {}
    except Exception as e:
        log.warning("[supabase_store] load_user_settings failed: %s", e)
        return {}


def save_user_settings(uid: str, settings: dict) -> bool:
    """Save user settings (upsert)."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        data = {
            "user_id": uid or "default",
            "settings": settings,
            "updated_at": datetime.utcnow().isoformat(),
        }
        _retry(lambda: sb.table("user_settings").upsert(data, on_conflict="user_id").execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] save_user_settings failed: %s", e)
        return False


# ===================================================================
# PROP LINE HISTORY  (replaces prop_line_history.jsonl)
# ===================================================================

def append_prop_line(record: dict) -> bool:
    """Append a single prop line snapshot."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        data = {
            "ts": record.get("ts", datetime.utcnow().isoformat()),
            "player": str(record.get("player", "")),
            "market": str(record.get("market", "")),
            "line": record.get("line"),
            "price": record.get("price"),
            "book": record.get("book"),
            "event_id": record.get("event_id"),
        }
        _retry(lambda: sb.table("prop_line_history").insert(data).execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] append_prop_line failed: %s", e)
        return False


def load_prop_line_history(player: str = None, market: str = None,
                           limit: int = 500) -> List[dict]:
    """Load prop line history, optionally filtered."""
    sb = _get_client()
    if sb is None:
        return []
    try:
        q = sb.table("prop_line_history").select("*").order("ts", desc=True).limit(limit)
        if player:
            q = q.eq("player", player)
        if market:
            q = q.eq("market", market)
        resp = _retry(lambda: q.execute())
        return resp.data or []
    except Exception as e:
        log.warning("[supabase_store] load_prop_line_history failed: %s", e)
        return []


# ===================================================================
# OPENING LINES  (replaces opening_lines.json)
# ===================================================================

def get_opening_line(key: str) -> Optional[dict]:
    """Get a single opening line by lookup key."""
    sb = _get_client()
    if sb is None:
        return None
    try:
        resp = _retry(lambda: (
            sb.table("opening_lines")
            .select("*")
            .eq("lookup_key", key)
            .limit(1)
            .execute()
        ))
        if resp.data:
            r = resp.data[0]
            return {"line": r.get("line"), "price": r.get("price"),
                    "ts": r.get("ts"), "date": r.get("date_str")}
        return None
    except Exception as e:
        log.warning("[supabase_store] get_opening_line failed: %s", e)
        return None


def set_opening_line(key: str, line: float, price: float,
                     ts: str, date_str: str) -> bool:
    """Set an opening line (insert if not exists)."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        data = {
            "lookup_key": key,
            "line": line,
            "price": price,
            "ts": ts,
            "date_str": date_str,
        }
        _retry(lambda: sb.table("opening_lines").upsert(data, on_conflict="lookup_key").execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] set_opening_line failed: %s", e)
        return False


def load_all_opening_lines() -> dict:
    """Load all opening lines as {key: {line, price, ts, date}} dict."""
    sb = _get_client()
    if sb is None:
        return {}
    try:
        resp = _retry(lambda: sb.table("opening_lines").select("*").execute())
        result = {}
        for r in (resp.data or []):
            result[r["lookup_key"]] = {
                "line": r.get("line"), "price": r.get("price"),
                "ts": r.get("ts"), "date": r.get("date_str"),
            }
        return result
    except Exception as e:
        log.warning("[supabase_store] load_all_opening_lines failed: %s", e)
        return {}


# ===================================================================
# USER AUTH  (replaces users_auth.json)
# ===================================================================

def load_auth_db() -> dict:
    """Load all auth records as {username: {pw_hash, email, created}}."""
    sb = _get_client()
    if sb is None:
        return {}
    try:
        resp = _retry(lambda: sb.table("user_auth").select("*").execute())
        result = {}
        for r in (resp.data or []):
            result[r["username"]] = {
                "pw_hash": r.get("pw_hash", ""),
                "email": r.get("email", ""),
                "created": r.get("created_at", ""),
            }
        return result
    except Exception as e:
        log.warning("[supabase_store] load_auth_db failed: %s", e)
        return {}


def save_auth_user(username: str, pw_hash: str, email: str = "") -> bool:
    """Create or update a user auth record."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        data = {
            "username": username,
            "pw_hash": pw_hash,
            "email": email,
            "created_at": datetime.utcnow().isoformat(),
        }
        _retry(lambda: sb.table("user_auth").upsert(data, on_conflict="username").execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] save_auth_user failed: %s", e)
        return False


# ===================================================================
# WATCHLIST  (replaces watchlist_{uid}.json)
# ===================================================================

def load_watchlist(uid: str) -> list:
    """Load player watchlist for a user."""
    sb = _get_client()
    if sb is None:
        return []
    try:
        resp = _retry(lambda: (
            sb.table("watchlist")
            .select("players")
            .eq("user_id", uid or "default")
            .limit(1)
            .execute()
        ))
        if resp.data:
            return resp.data[0].get("players", [])
        return []
    except Exception as e:
        log.warning("[supabase_store] load_watchlist failed: %s", e)
        return []


def save_watchlist(uid: str, players: list) -> bool:
    """Save player watchlist (upsert)."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        data = {
            "user_id": uid or "default",
            "players": players,
            "updated_at": datetime.utcnow().isoformat(),
        }
        _retry(lambda: sb.table("watchlist").upsert(data, on_conflict="user_id").execute())
        return True
    except Exception as e:
        log.warning("[supabase_store] save_watchlist failed: %s", e)
        return False


# ===================================================================
# MIGRATION HELPER — import existing local files into Supabase
# ===================================================================

def migrate_local_history(uid: str, local_csv_path: str) -> int:
    """Import a local history CSV into Supabase.  Returns rows imported."""
    sb = _get_client()
    if sb is None:
        return 0
    try:
        df = pd.read_csv(local_csv_path)
    except Exception:
        return 0
    count = 0
    for _, r in df.iterrows():
        try:
            legs_val = r.get("legs", "[]")
            if isinstance(legs_val, str):
                legs_val = json.loads(legs_val)
            lr_val = r.get("leg_results", '["Pending"]')
            if isinstance(lr_val, str):
                lr_val = json.loads(lr_val)
            data = {
                "ts": str(r.get("ts", datetime.utcnow().isoformat())),
                "user_id": str(r.get("user_id", uid)),
                "legs": legs_val,
                "n_legs": int(r.get("n_legs", 1)),
                "leg_results": lr_val,
                "result": str(r.get("result", "Pending")),
                "decision": str(r.get("decision", "BET")),
                "notes": str(r.get("notes", "")),
            }
            sb.table("bet_history").insert(data).execute()
            count += 1
        except Exception as e:
            log.warning("[supabase_store] migrate row failed: %s", e)
    return count


# ===================================================================
# HEALTH PING — keeps Supabase free tier awake
# ===================================================================

def health_ping() -> bool:
    """Lightweight query to prevent Supabase free-tier auto-pause."""
    sb = _get_client()
    if sb is None:
        return False
    try:
        sb.table("user_settings").select("id").limit(1).execute()
        return True
    except Exception:
        return False
