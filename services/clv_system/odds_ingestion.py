"""
services/clv_system/odds_ingestion.py
======================================
Continuous odds capture from multiple sources for CLV tracking.

Ingests lines from:
  - PrizePicks (NBA props)
  - Underdog Fantasy
  - Sleeper
  - The Odds API (DraftKings, FanDuel, etc.)

Stores EVERY line movement with millisecond-precision timestamps in the
clv_line_movements table.  Runs on a 5-minute schedule for NBA.

This module wraps the existing workers/odds_worker.py fetch functions and
pipes the results into the CLV-specific line_movements table for
high-resolution tracking separate from the main app's line_movements table.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from quant_system.db.schema import get_engine, get_session
from services.clv_system.models import CLVLineMovement, Base

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

ODDS_API_MARKET_KEYS = [
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_points_rebounds_assists",
    "player_points_rebounds", "player_points_assists",
    "player_rebounds_assists", "player_blocks", "player_steals",
    "player_turnovers",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _odds_api_key() -> str:
    return os.environ.get("ODDS_API_KEY", "")


def _normalize_stat(raw: str) -> str:
    s = str(raw).strip()
    for k, v in _PP_STAT_MAP.items():
        if k.lower() == s.lower():
            return v
    return s


def _http_get(url: str, params: Optional[dict] = None,
              headers: Optional[dict] = None, timeout: int = 20,
              retries: int = 3) -> Tuple[Optional[Any], Optional[str]]:
    """GET with retry + backoff."""
    hdrs = dict(_HEADERS)
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
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


def _american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.5
    elif odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


class CLVOddsIngestion:
    """Continuous odds capture engine for CLV tracking.

    Ingests from all available sources, stores every line movement with
    millisecond timestamps in the clv_line_movements table.
    """

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create CLV tables if they don't exist."""
        engine = get_engine(self._db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    # ── Master ingest method ──────────────────────────────────────────

    def ingest_all(self) -> Dict[str, Any]:
        """Fetch odds from all sources and store in clv_line_movements.

        Returns summary dict with counts per source and any errors.
        """
        results: Dict[str, Any] = {
            "ok": True,
            "timestamp": _utcnow().isoformat(),
            "source_counts": {},
            "total_lines": 0,
            "errors": [],
        }

        # 1. PrizePicks
        try:
            pp_count = self._ingest_prizepicks()
            results["source_counts"]["prizepicks"] = pp_count
            results["total_lines"] += pp_count
            log.info("CLV ingestion — PrizePicks: %d lines", pp_count)
        except Exception as exc:
            results["errors"].append(f"PrizePicks: {exc}")
            log.warning("CLV PrizePicks failed: %s", exc)

        # 2. Underdog Fantasy
        try:
            ud_count = self._ingest_underdog()
            results["source_counts"]["underdog"] = ud_count
            results["total_lines"] += ud_count
            log.info("CLV ingestion — Underdog: %d lines", ud_count)
        except Exception as exc:
            results["errors"].append(f"Underdog: {exc}")
            log.warning("CLV Underdog failed: %s", exc)

        # 3. Sleeper
        try:
            sl_count = self._ingest_sleeper()
            results["source_counts"]["sleeper"] = sl_count
            results["total_lines"] += sl_count
            log.info("CLV ingestion — Sleeper: %d lines", sl_count)
        except Exception as exc:
            results["errors"].append(f"Sleeper: {exc}")
            log.warning("CLV Sleeper failed: %s", exc)

        # 4. The Odds API
        try:
            oa_count = self._ingest_odds_api()
            results["source_counts"]["odds_api"] = oa_count
            results["total_lines"] += oa_count
            log.info("CLV ingestion — Odds API: %d lines", oa_count)
        except Exception as exc:
            results["errors"].append(f"Odds API: {exc}")
            log.warning("CLV Odds API failed: %s", exc)

        if results["total_lines"] == 0:
            results["ok"] = False

        return results

    # ── PrizePicks ────────────────────────────────────────────────────

    def _ingest_prizepicks(self) -> int:
        """Fetch and store PrizePicks NBA lines."""
        rows = self._fetch_prizepicks_raw()
        return self._store_rows(rows, book="prizepicks")

    def _fetch_prizepicks_raw(self) -> List[Dict[str, Any]]:
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
                        "market_type": _normalize_stat(stat_type),
                        "line": float(line_score),
                        "event_id": start_time,
                        "odds_american": -110,
                    })
                except (TypeError, ValueError):
                    pass

            time.sleep(2)

        return rows

    # ── Underdog Fantasy ──────────────────────────────────────────────

    def _ingest_underdog(self) -> int:
        rows = self._fetch_underdog_raw()
        return self._store_rows(rows, book="underdog")

    def _fetch_underdog_raw(self) -> List[Dict[str, Any]]:
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
                    sport_id = str(
                        app.get("sport_id", app.get("sport", ""))
                    ).lower().strip()
                    if sport_id and sport_id not in UD_BASKETBALL_SPORT_IDS:
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
                            "market_type": _normalize_stat(stat_type),
                            "line": float(stat_value),
                            "event_id": "",
                            "odds_american": -110,
                        })
                except Exception:
                    continue

            if rows:
                break
            time.sleep(2)

        return rows

    # ── Sleeper ───────────────────────────────────────────────────────

    def _ingest_sleeper(self) -> int:
        rows = self._fetch_sleeper_raw()
        return self._store_rows(rows, book="sleeper")

    def _fetch_sleeper_raw(self) -> List[Dict[str, Any]]:
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
                            "market_type": _normalize_stat(stat_type),
                            "line": float(line_val),
                            "event_id": "",
                            "odds_american": -110,
                        })

                if rows:
                    break
            except Exception:
                continue

        return rows

    # ── The Odds API ──────────────────────────────────────────────────

    def _ingest_odds_api(self) -> int:
        rows = self._fetch_odds_api_raw()
        return self._store_rows_bulk(rows)

    def _fetch_odds_api_raw(self) -> List[Dict[str, Any]]:
        key = _odds_api_key()
        if not key:
            log.info("No ODDS_API_KEY — skipping The Odds API")
            return []

        rows: List[Dict[str, Any]] = []

        events_data, err = _http_get(
            f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events",
            params={"apiKey": key},
        )
        if err or not isinstance(events_data, list):
            log.warning("Odds API events error: %s", err)
            return []

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
                            odds_dec = float(price) if price else None
                            odds_am = None
                            if odds_dec and odds_dec > 0:
                                if odds_dec >= 2.0:
                                    odds_am = int(round((odds_dec - 1) * 100))
                                else:
                                    odds_am = int(round(-100 / (odds_dec - 1)))

                            rows.append({
                                "player": desc,
                                "market_type": market_key,
                                "line": float(point),
                                "book": book_key,
                                "event_id": event_name,
                                "odds_american": odds_am,
                                "odds_decimal": odds_dec,
                            })

            time.sleep(0.5)

        return rows

    # ── Storage helpers ───────────────────────────────────────────────

    def _store_rows(self, rows: List[Dict[str, Any]], book: str) -> int:
        """Store a batch of line rows from a single book."""
        if not rows:
            return 0

        now = _utcnow()
        session = self._session()
        count = 0
        try:
            for row in rows:
                odds_am = row.get("odds_american")
                implied = _american_to_implied_prob(odds_am) if odds_am else None

                lm = CLVLineMovement(
                    sport=self.sport,
                    event_id=row.get("event_id", ""),
                    market_type=row.get("market_type", ""),
                    book=book,
                    player=row.get("player", ""),
                    line=row["line"],
                    odds_american=odds_am,
                    odds_decimal=row.get("odds_decimal"),
                    implied_prob=implied,
                    timestamp=now,
                )
                session.add(lm)
                count += 1
            session.commit()
        except Exception:
            session.rollback()
            log.exception("Failed to store CLV lines for book=%s", book)
            raise
        finally:
            session.close()

        return count

    def _store_rows_bulk(self, rows: List[Dict[str, Any]]) -> int:
        """Store a batch of line rows with per-row book info (Odds API)."""
        if not rows:
            return 0

        now = _utcnow()
        session = self._session()
        count = 0
        try:
            for row in rows:
                odds_am = row.get("odds_american")
                implied = _american_to_implied_prob(odds_am) if odds_am else None

                lm = CLVLineMovement(
                    sport=self.sport,
                    event_id=row.get("event_id", ""),
                    market_type=row.get("market_type", ""),
                    book=row.get("book", "odds_api"),
                    player=row.get("player", ""),
                    line=row["line"],
                    odds_american=odds_am,
                    odds_decimal=row.get("odds_decimal"),
                    implied_prob=implied,
                    timestamp=now,
                )
                session.add(lm)
                count += 1
            session.commit()
        except Exception:
            session.rollback()
            log.exception("Failed to store CLV Odds API lines")
            raise
        finally:
            session.close()

        return count

    # ── Query helpers (used by other CLV modules) ─────────────────────

    def get_latest_lines(self, player: str, market_type: str,
                         limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent line observations for a player/market."""
        session = self._session()
        try:
            rows = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                )
                .order_by(CLVLineMovement.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def get_line_history(self, player: str, market_type: str,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Get line history for a player/market over the last N hours."""
        from datetime import timedelta
        cutoff = _utcnow() - timedelta(hours=hours)
        session = self._session()
        try:
            rows = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                    CLVLineMovement.timestamp >= cutoff,
                )
                .order_by(CLVLineMovement.timestamp.asc())
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def get_all_current_lines(self) -> List[Dict[str, Any]]:
        """Get the most recent line for each player/market/book combination."""
        from sqlalchemy import func as sa_func

        session = self._session()
        try:
            subq = (
                session.query(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                    sa_func.max(CLVLineMovement.timestamp).label("max_ts"),
                )
                .filter(CLVLineMovement.sport == self.sport)
                .group_by(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                )
                .subquery()
            )

            rows = (
                session.query(CLVLineMovement)
                .join(
                    subq,
                    (CLVLineMovement.player == subq.c.player)
                    & (CLVLineMovement.market_type == subq.c.market_type)
                    & (CLVLineMovement.book == subq.c.book)
                    & (CLVLineMovement.timestamp == subq.c.max_ts),
                )
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()
