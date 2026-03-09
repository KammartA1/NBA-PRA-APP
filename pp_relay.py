#!/usr/bin/env python3
"""
pp_relay.py — PrizePicks Local Relay Server
============================================
Run this on YOUR LOCAL MACHINE (residential IP) so the Streamlit app
can get PrizePicks lines even when deployed on Streamlit Cloud.

Usage:
    pip install curl_cffi requests
    python pp_relay.py

Then either:
  • If app is local too:  set Relay URL to http://localhost:8765/lines
  • If app is on cloud:   install ngrok (https://ngrok.com), run:
        ngrok http 8765
    and paste the https://xxxx.ngrok.io/lines URL into the app's Relay URL field.

Endpoints:
  GET /lines       → latest PP lines as JSON list (or {"rows":[...], "ts":..., "count":...})
  GET /status      → health + last-fetch metadata
  POST /refresh    → trigger immediate re-fetch
"""

import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

PORT = int(os.environ.get("PP_RELAY_PORT", 8765))
FETCH_INTERVAL = int(os.environ.get("PP_RELAY_INTERVAL", 600))   # seconds between auto-fetches
LEAGUE_FILTER  = ("NBA", "NBA 1Q", "NBA 1H", "NBA 2H")   # set to None to get all sports

PRIZEPICKS_API = "https://api.prizepicks.com/projections"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/vnd.api+json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://app.prizepicks.com/",
    "Origin": "https://app.prizepicks.com",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Connection": "keep-alive",
    "DNT": "1",
}

# ── State shared between the fetch thread and HTTP handler ──
_state = {
    "rows": [],
    "ts": 0.0,
    "err": None,
    "lock": threading.Lock(),
    "refresh_evt": threading.Event(),
}


def _parse_response(data: dict) -> list:
    _VALID_TYPES = {"projection", "new_player_projection", "boardprojection"}
    included = {
        item["id"]: item
        for item in data.get("included", [])
        if isinstance(item, dict) and "id" in item
    }
    rows = []
    for proj in data.get("data", []):
        if not isinstance(proj, dict):
            continue
        _type = str(proj.get("type", "")).lower()
        attrs = proj.get("attributes", {}) or {}
        _has_fields = bool(attrs.get("stat_type") and attrs.get("line_score") is not None)
        if _type not in _VALID_TYPES and not _has_fields:
            continue
        if LEAGUE_FILTER:
            league = str(attrs.get("league", "") or "").upper()
            if league and league not in LEAGUE_FILTER:
                continue
        rels = proj.get("relationships", {}) or {}
        player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
        if not player_id:
            player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
        player_attrs = included.get(player_id, {}).get("attributes", {}) if player_id else {}
        player_name = (
            player_attrs.get("name", "") or attrs.get("name", "") or attrs.get("display_name", "")
        )
        stat_type  = attrs.get("stat_type", "")
        line_score = attrs.get("line_score")
        odds_type  = str(attrs.get("odds_type", "") or "").lower().strip() or "standard"
        if player_name and stat_type and line_score is not None:
            try:
                rows.append({
                    "player":    player_name,
                    "stat_type": stat_type,
                    "line":      float(line_score),
                    "start_time": attrs.get("start_time", ""),
                    "source":    "prizepicks",
                    "odds_type": odds_type,
                })
            except (TypeError, ValueError):
                pass
    return rows


def _do_request(per_page: int = 500, single_stat: str = "true"):
    params = {
        "per_page":    str(per_page),
        "single_stat": single_stat,
        "in_play":     "false",
    }
    # Try curl_cffi (best — Chrome TLS impersonation, bypasses PerimeterX)
    try:
        from curl_cffi import requests as cffi_req
        r = cffi_req.get(
            PRIZEPICKS_API, params=params, headers=HEADERS,
            impersonate="chrome120", timeout=25,
        )
        if r.status_code not in (403, 429):
            return r, None
        print(f"  curl_cffi got {r.status_code}")
    except ImportError:
        pass
    except Exception as e:
        print(f"  curl_cffi error: {e}")

    # Fallback: cloudscraper
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        r = scraper.get(PRIZEPICKS_API, params=params, headers=HEADERS, timeout=25)
        if r.status_code not in (403, 429):
            return r, None
        print(f"  cloudscraper got {r.status_code}")
    except ImportError:
        pass
    except Exception as e:
        print(f"  cloudscraper error: {e}")

    # Fallback: plain requests
    import requests as _req
    try:
        r = _req.get(PRIZEPICKS_API, params=params, headers=HEADERS, timeout=20)
        return r, None
    except Exception as e:
        return None, str(e)


def fetch_all():
    """Fetch standard + combo markets, deduplicate, return (rows, error)."""
    all_rows: list = []
    seen: set = set()
    last_err = None
    for single_stat in ("true", "false"):
        r, err = _do_request(500, single_stat)
        if err:
            last_err = err
            continue
        if r is None:
            last_err = "No response"
            continue
        if not r.ok:
            last_err = f"HTTP {r.status_code}"
            if r.status_code in (403, 429):
                return [], last_err
            continue
        try:
            rows = _parse_response(r.json())
        except Exception as e:
            last_err = f"Parse error: {e}"
            continue
        for row in rows:
            k = (row["player"], row["stat_type"])
            if k not in seen:
                seen.add(k)
                all_rows.append(row)
    if all_rows:
        return all_rows, None
    return [], last_err or "No NBA props found"


def _fetch_loop():
    """Background thread: fetch on interval or when refresh_evt is set."""
    print(f"[relay] Fetch thread started — interval={FETCH_INTERVAL}s")
    while True:
        print("[relay] Fetching PrizePicks lines...")
        rows, err = fetch_all()
        with _state["lock"]:
            if rows:
                _state["rows"] = rows
                _state["ts"]   = time.time()
                _state["err"]  = None
                print(f"[relay] OK — {len(rows)} props cached")
            else:
                _state["err"] = err
                print(f"[relay] Error: {err}")
        _state["refresh_evt"].wait(FETCH_INTERVAL)
        _state["refresh_evt"].clear()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default Apache-style logs

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/lines", "/lines/"):
            with _state["lock"]:
                rows = list(_state["rows"])
                ts   = _state["ts"]
                err  = _state["err"]
            age = int(time.time() - ts) if ts else None
            if rows:
                self._send_json({"rows": rows, "count": len(rows), "ts": ts, "age_sec": age})
            else:
                self._send_json({"rows": [], "count": 0, "err": err}, status=503)
        elif path in ("/status", "/status/"):
            with _state["lock"]:
                ts  = _state["ts"]
                err = _state["err"]
                cnt = len(_state["rows"])
            self._send_json({
                "ok":        cnt > 0,
                "count":     cnt,
                "last_fetch": ts,
                "age_sec":   int(time.time() - ts) if ts else None,
                "error":     err,
                "interval":  FETCH_INTERVAL,
            })
        else:
            self._send_json({"error": "Not found. Use /lines or /status"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path in ("/refresh", "/refresh/"):
            _state["refresh_evt"].set()
            self._send_json({"ok": True, "message": "Refresh triggered"})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()


if __name__ == "__main__":
    t = threading.Thread(target=_fetch_loop, daemon=True, name="pp_relay_fetcher")
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"[relay] Listening on http://localhost:{PORT}")
    print(f"[relay] Endpoints: /lines  /status  /refresh")
    print(f"[relay] Streamlit relay URL: http://localhost:{PORT}/lines")
    print(f"[relay] For cloud access run:  ngrok http {PORT}")
    print(f"[relay] Press Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[relay] Stopped.")
        sys.exit(0)
