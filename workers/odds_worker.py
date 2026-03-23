"""
workers/odds_worker.py
======================
Fetches odds / prop lines from multiple sources on a 5-minute cadence,
normalises them into a unified format, stores ``LineMovement`` rows, and
detects sharp line movements.

Sources:
  1. PrizePicks (existing scraper logic)
  2. Underdog Fantasy
  3. Sleeper (unofficial endpoints)
  4. The Odds API (DraftKings, FanDuel, etc.)

Run standalone:
    python -m workers.odds_worker          # one-shot
    python -m workers.odds_worker --loop   # 5-min loop
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from database.connection import session_scope, init_db
from database.models import LineMovement, Player, Event
from workers.base import BaseWorker, standalone_main

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA = "basketball_nba"
REGION_US = "us"

PRIZEPICKS_API = "https://api.prizepicks.com/projections"
NBA_LEAGUES = {"NBA", "NBA 1Q", "NBA 1H", "NBA 2H"}

UNDERDOG_ENDPOINTS = [
    "https://api.underdogfantasy.com/v3/over_under_lines",
    "https://api.underdogfantasy.com/v4/over_under_lines",
    "https://api.underdogfantasy.com/v2/over_under_lines",
]

SLEEPER_ENDPOINTS = [
    "https://api.sleeper.app/v1/stats/nba/projections/regular/2025/1",
    "https://api.sleeper.app/projections/nba",
    "https://api.sleeper.com/picks/nba",
]

UD_BASKETBALL_SPORT_IDS = {
    "nba", "basketball", "5", "4", "nba_basketball", "basketball_nba", "",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# PrizePicks stat type normalisation (subset -- full map lives in app.py)
_PP_STAT_MAP: Dict[str, str] = {
    "Points": "Points", "Pts": "Points", "PTS": "Points",
    "Rebounds": "Rebounds", "Reb": "Rebounds", "REB": "Rebounds",
    "Total Rebounds": "Rebounds",
    "Assists": "Assists", "Ast": "Assists", "AST": "Assists",
    "3-Pointers Made": "3PM", "3 Pointers Made": "3PM", "3PM": "3PM",
    "3-PT Made": "3PM", "3PT Made": "3PM",
    "Pts+Reb+Ast": "PRA", "Points+Rebounds+Assists": "PRA", "PRA": "PRA",
    "Pts+Reb": "PR", "Points+Rebounds": "PR",
    "Pts+Ast": "PA", "Points+Assists": "PA",
    "Reb+Ast": "RA", "Rebounds+Assists": "RA",
    "Blocked Shots": "Blocks", "Blocks": "Blocks", "Blk": "Blocks",
    "Steals": "Steals", "Stl": "Steals",
    "Turnovers": "Turnovers", "Tov": "Turnovers",
    "Blks+Stls": "Stocks", "Stocks": "Stocks",
    "Fantasy Score": "Fantasy Score",
    "H1 Points": "H1 Points", "1H Points": "H1 Points",
    "1st Half Points": "H1 Points",
    "H2 Points": "H2 Points", "2H Points": "H2 Points",
}

ODDS_API_MARKET_KEYS = [
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_points_rebounds_assists",
    "player_points_rebounds", "player_points_assists",
    "player_rebounds_assists", "player_blocks", "player_steals",
    "player_turnovers",
]

# Sharp movement: >1 point in <30 minutes
SHARP_MOVE_THRESHOLD_PTS = 1.0
SHARP_MOVE_WINDOW_MINUTES = 30


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _odds_api_key() -> str:
    return os.environ.get("ODDS_API_KEY", "")


def _normalize_stat(raw: str) -> str:
    """Best-effort stat-type normalisation."""
    s = str(raw).strip()
    for k, v in _PP_STAT_MAP.items():
        if k.lower() == s.lower():
            return v
    return s


def _http_get(url: str, params: Optional[dict] = None,
              headers: Optional[dict] = None, timeout: int = 20,
              retries: int = 3) -> Tuple[Optional[Any], Optional[str]]:
    """GET with retry + backoff.  Returns (json_body | None, error | None)."""
    hdrs = dict(_HEADERS)
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            # Prefer curl_cffi for anti-bot bypass
            try:
                from curl_cffi import requests as cffi_req
                r = cffi_req.get(url, params=params, headers=hdrs,
                                 impersonate="chrome120", timeout=timeout)
                if r.status_code not in (403, 429):
                    r.raise_for_status()
                    return r.json(), None
                if r.status_code == 429:
                    wait = 8 * (attempt + 1)
                    log.warning("Rate limited %s, waiting %ds", url, wait)
                    time.sleep(wait)
                    continue
            except ImportError:
                pass
            except Exception:
                pass

            r = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json(), None
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return None, f"{type(exc).__name__}: {exc}"
    return None, "All retries exhausted"


# ===================================================================
# Source fetchers
# ===================================================================

def fetch_prizepicks() -> List[Dict[str, Any]]:
    """Fetch NBA props from PrizePicks API, return unified rows."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for single_stat in ("true", "false"):
        params = {
            "per_page": "500",
            "single_stat": single_stat,
            "in_play": "false",
        }
        pp_headers = dict(_HEADERS)
        pp_headers["Referer"] = "https://app.prizepicks.com/"
        pp_headers["Origin"] = "https://app.prizepicks.com"

        data, err = _http_get(PRIZEPICKS_API, params=params, headers=pp_headers)
        if err or data is None:
            log.warning("PrizePicks fetch error: %s", err)
            continue

        included = {
            item["id"]: item
            for item in data.get("included", [])
            if isinstance(item, dict) and "id" in item
        }

        for proj in data.get("data", []):
            if not isinstance(proj, dict):
                continue
            attrs = proj.get("attributes", {}) or {}
            league = str(attrs.get("league", "") or "").upper().strip()
            if league and league not in NBA_LEAGUES:
                continue

            stat_type = attrs.get("stat_type", "")
            line_score = attrs.get("line_score")
            if not stat_type or line_score is None:
                continue

            rels = proj.get("relationships", {}) or {}
            player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
            if not player_id:
                player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
            player_attrs = (
                included.get(player_id, {}).get("attributes", {})
                if player_id else {}
            )
            player_name = (
                player_attrs.get("name", "")
                or player_attrs.get("display_name", "")
                or attrs.get("name", "")
                or attrs.get("description", "")
            )
            if not player_name:
                continue

            key = (player_name.lower(), _normalize_stat(stat_type))
            if key in seen:
                continue
            seen.add(key)

            start_time = attrs.get("start_time", "")
            try:
                rows.append({
                    "player": player_name,
                    "market": _normalize_stat(stat_type),
                    "line": float(line_score),
                    "book": "prizepicks",
                    "event_name": "",
                    "start_time": start_time,
                })
            except (TypeError, ValueError):
                pass

        time.sleep(2)

    return rows


def fetch_underdog() -> List[Dict[str, Any]]:
    """Fetch NBA props from Underdog Fantasy, return unified rows."""
    rows: List[Dict[str, Any]] = []
    for url in UNDERDOG_ENDPOINTS:
        data, err = _http_get(url, headers={
            **_HEADERS,
            "Referer": "https://underdogfantasy.com/",
            "Origin": "https://underdogfantasy.com",
        })
        if err or data is None:
            continue

        appearances = {}
        for a in data.get("appearances", []):
            appearances[str(a.get("id", ""))] = a

        players_map = {}
        for p in data.get("players", []):
            players_map[str(p.get("id", ""))] = p

        lines_list = data.get("over_under_lines", data.get("lines", []))
        for line_item in lines_list:
            try:
                ou = line_item.get("over_under", line_item)
                app_stat = ou.get("appearance_stat", {})
                app_id = str(app_stat.get("appearance_id", ""))
                app = appearances.get(app_id, {})
                sport = str(
                    app.get("sport_id", app.get("sport", ""))
                ).lower().strip()
                if sport and sport not in UD_BASKETBALL_SPORT_IDS:
                    continue
                player_id = str(app.get("player_id", ""))
                player = players_map.get(player_id, {})
                player_name = (
                    f"{player.get('first_name', '')} "
                    f"{player.get('last_name', '')}".strip()
                    or player.get("name", "")
                )
                stat_type = app_stat.get(
                    "display_stat", app_stat.get("stat", "")
                )
                stat_value = line_item.get(
                    "stat_value", ou.get("stat_value")
                )
                if player_name and stat_type and stat_value is not None:
                    rows.append({
                        "player": player_name,
                        "market": _normalize_stat(stat_type),
                        "line": float(stat_value),
                        "book": "underdog",
                        "event_name": "",
                        "start_time": "",
                    })
            except Exception:
                continue

        if rows:
            break
        time.sleep(2)

    return rows


def fetch_sleeper() -> List[Dict[str, Any]]:
    """Attempt to fetch NBA pick lines from Sleeper (unofficial)."""
    rows: List[Dict[str, Any]] = []
    for url in SLEEPER_ENDPOINTS:
        data, err = _http_get(url, timeout=10)
        if err or data is None:
            continue
        try:
            if isinstance(data, dict):
                items = data.get("picks", data.get("lines",
                         data.get("projections", [])))
            elif isinstance(data, list):
                items = data
            else:
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue
                player_name = (
                    item.get("player_name")
                    or item.get("name")
                    or item.get("player", "")
                ).strip()
                stat_type = (
                    item.get("stat_type")
                    or item.get("stat")
                    or item.get("category", "")
                ).strip()
                line_val = item.get("line") or item.get("line_score") or item.get("value")
                if player_name and stat_type and line_val is not None:
                    rows.append({
                        "player": player_name,
                        "market": _normalize_stat(stat_type),
                        "line": float(line_val),
                        "book": "sleeper",
                        "event_name": "",
                        "start_time": "",
                    })

            if rows:
                break
        except Exception:
            continue

    return rows


def fetch_odds_api() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch props + events from The Odds API.

    Returns (line_rows, event_rows).
    """
    key = _odds_api_key()
    if not key:
        log.info("No ODDS_API_KEY set -- skipping The Odds API")
        return [], []

    line_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    # 1. Get events
    events_data, err = _http_get(
        f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events",
        params={"apiKey": key},
    )
    if err or not isinstance(events_data, list):
        log.warning("Odds API events error: %s", err)
        return [], []

    for ev in events_data:
        event_rows.append({
            "event_id": ev.get("id", ""),
            "event_name": f"{ev.get('away_team', '')} @ {ev.get('home_team', '')}",
            "start_time": ev.get("commence_time", ""),
            "home_team": ev.get("home_team", ""),
            "away_team": ev.get("away_team", ""),
        })

    # 2. Get player props for each event (batch by market keys)
    market_str = ",".join(ODDS_API_MARKET_KEYS)
    for ev in events_data:
        eid = ev.get("id", "")
        if not eid:
            continue
        event_name = f"{ev.get('away_team', '')} @ {ev.get('home_team', '')}"
        odds_data, err2 = _http_get(
            f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{eid}/odds",
            params={
                "apiKey": key,
                "regions": REGION_US,
                "markets": market_str,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
        )
        if err2 or not isinstance(odds_data, dict):
            continue

        for bookmaker in odds_data.get("bookmakers", []):
            book_key = bookmaker.get("key", "unknown")
            for market_obj in bookmaker.get("markets", []):
                market_key = market_obj.get("key", "")
                for outcome in market_obj.get("outcomes", []):
                    desc = outcome.get("description", "")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if desc and point is not None:
                        line_rows.append({
                            "player": desc,
                            "market": market_key,
                            "line": float(point),
                            "book": book_key,
                            "event_name": event_name,
                            "start_time": ev.get("commence_time", ""),
                            "odds_decimal": float(price) if price else None,
                        })

        time.sleep(0.5)  # Rate limit between event requests

    return line_rows, event_rows


# ===================================================================
# Sharp movement detection
# ===================================================================

def detect_sharp_movements(session, new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare new lines against recent DB entries.  Flag lines that moved
    more than SHARP_MOVE_THRESHOLD_PTS within the last SHARP_MOVE_WINDOW_MINUTES."""
    sharp_alerts: List[Dict[str, Any]] = []
    cutoff = _utcnow() - timedelta(minutes=SHARP_MOVE_WINDOW_MINUTES)

    for row in new_rows:
        player = row["player"]
        market = row["market"]
        new_line = row["line"]
        book = row.get("book", "")

        recent = (
            session.query(LineMovement)
            .filter(
                LineMovement.player == player,
                LineMovement.market == market,
                LineMovement.book == book,
                LineMovement.timestamp >= cutoff,
            )
            .order_by(LineMovement.timestamp.desc())
            .first()
        )
        if recent is not None:
            move = abs(new_line - recent.line)
            if move >= SHARP_MOVE_THRESHOLD_PTS:
                sharp_alerts.append({
                    "player": player,
                    "market": market,
                    "book": book,
                    "old_line": recent.line,
                    "new_line": new_line,
                    "move": move,
                    "minutes_elapsed": (
                        (_utcnow() - recent.timestamp).total_seconds() / 60
                        if recent.timestamp.tzinfo
                        else (_utcnow().replace(tzinfo=None) - recent.timestamp).total_seconds() / 60
                    ),
                })

    return sharp_alerts


# ===================================================================
# Player / Event upsert
# ===================================================================

def _upsert_players(session, rows: List[Dict[str, Any]]) -> int:
    """Insert new players discovered in fetched lines.  Returns count of newly created."""
    created = 0
    seen_names = set()
    for row in rows:
        name = row.get("player", "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        existing = (
            session.query(Player)
            .filter(Player.name == name, Player.sport == "NBA")
            .first()
        )
        if existing is None:
            session.add(Player(name=name, sport="NBA", active=True))
            created += 1
    return created


def _upsert_events(session, event_rows: List[Dict[str, Any]]) -> int:
    """Insert new events from Odds API data.  Returns count of newly created."""
    created = 0
    seen = set()
    for ev in event_rows:
        ename = ev.get("event_name", "").strip()
        if not ename or ename in seen:
            continue
        seen.add(ename)

        start_str = ev.get("start_time", "")
        start_dt = None
        if start_str:
            try:
                start_dt = datetime.fromisoformat(
                    start_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        existing = (
            session.query(Event)
            .filter(Event.event_name == ename, Event.sport == "NBA")
            .first()
        )
        if existing is None:
            session.add(Event(
                event_name=ename,
                sport="NBA",
                start_time=start_dt,
                status="scheduled",
                metadata_json=json.dumps(ev, default=str),
            ))
            created += 1
        elif start_dt and existing.start_time != start_dt:
            existing.start_time = start_dt
    return created


# ===================================================================
# Worker class
# ===================================================================

class OddsWorker(BaseWorker):
    """Fetches and stores odds from all sources every 5 minutes."""

    def __init__(self, **kwargs):
        super().__init__(
            name="odds_worker",
            interval_seconds=int(os.environ.get("ODDS_INTERVAL", "300")),
            max_retries=3,
            retry_delay=10.0,
            **kwargs,
        )

    def execute(self) -> Dict[str, Any]:
        all_rows: List[Dict[str, Any]] = []
        source_counts: Dict[str, int] = {}
        errors: List[str] = []

        # 1. PrizePicks
        try:
            pp_rows = fetch_prizepicks()
            all_rows.extend(pp_rows)
            source_counts["prizepicks"] = len(pp_rows)
            self.logger.info("PrizePicks: %d lines", len(pp_rows))
        except Exception as exc:
            errors.append(f"PrizePicks: {exc}")
            self.logger.warning("PrizePicks failed: %s", exc)

        # 2. Underdog
        try:
            ud_rows = fetch_underdog()
            all_rows.extend(ud_rows)
            source_counts["underdog"] = len(ud_rows)
            self.logger.info("Underdog: %d lines", len(ud_rows))
        except Exception as exc:
            errors.append(f"Underdog: {exc}")
            self.logger.warning("Underdog failed: %s", exc)

        # 3. Sleeper
        try:
            sl_rows = fetch_sleeper()
            all_rows.extend(sl_rows)
            source_counts["sleeper"] = len(sl_rows)
            self.logger.info("Sleeper: %d lines", len(sl_rows))
        except Exception as exc:
            errors.append(f"Sleeper: {exc}")
            self.logger.warning("Sleeper failed: %s", exc)

        # 4. The Odds API
        try:
            oa_lines, oa_events = fetch_odds_api()
            all_rows.extend(oa_lines)
            source_counts["odds_api"] = len(oa_lines)
            self.logger.info("Odds API: %d lines, %d events", len(oa_lines), len(oa_events))
        except Exception as exc:
            oa_events = []
            errors.append(f"Odds API: {exc}")
            self.logger.warning("Odds API failed: %s", exc)

        if not all_rows:
            return {
                "ok": False,
                "error": "No lines fetched from any source",
                "source_counts": source_counts,
                "errors": errors,
            }

        # Store to database
        now = _utcnow()
        sharp_alerts: List[Dict[str, Any]] = []
        stored_count = 0
        new_players = 0
        new_events = 0

        with session_scope() as session:
            # Detect sharp movements before inserting new data
            sharp_alerts = detect_sharp_movements(session, all_rows)

            # Upsert players and events
            new_players = _upsert_players(session, all_rows)
            if oa_events:
                new_events = _upsert_events(session, oa_events)

            # Insert line_movements
            for row in all_rows:
                lm = LineMovement(
                    sport="NBA",
                    event=row.get("event_name", ""),
                    market=row.get("market", ""),
                    book=row.get("book", ""),
                    player=row.get("player", ""),
                    line=row["line"],
                    odds=None,
                    timestamp=now,
                    is_opening=False,
                    is_closing=False,
                )
                session.add(lm)
                stored_count += 1

        if sharp_alerts:
            self.logger.warning(
                "SHARP MOVEMENTS DETECTED: %d alerts", len(sharp_alerts)
            )
            for sa in sharp_alerts[:5]:
                self.logger.warning(
                    "  SHARP: %s %s [%s] %.1f -> %.1f (%.1f pts in %.0f min)",
                    sa["player"], sa["market"], sa["book"],
                    sa["old_line"], sa["new_line"],
                    sa["move"], sa.get("minutes_elapsed", 0),
                )

        # Run post-ingestion data quality audit (non-blocking)
        audit_result = {}
        try:
            from workers.data_audit_worker import run_post_ingestion_audit
            audit_result = run_post_ingestion_audit()
        except Exception as exc:
            self.logger.warning("Post-ingestion audit skipped: %s", exc)

        return {
            "ok": True,
            "lines_stored": stored_count,
            "source_counts": source_counts,
            "new_players": new_players,
            "new_events": new_events,
            "sharp_alerts": len(sharp_alerts),
            "errors": errors if errors else None,
            "audit_score": audit_result.get("composite_score"),
        }


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    standalone_main(OddsWorker)
