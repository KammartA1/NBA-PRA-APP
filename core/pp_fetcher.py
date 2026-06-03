"""
core/pp_fetcher.py — Standalone PrizePicks fetcher.
No Streamlit dependency. Session-cookie approach proven to work from datacenter IPs.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

PRIZEPICKS_API = "https://api.prizepicks.com/projections"

MARKET_MAP = {
    "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
    "Pts+Rebs+Asts": "PRA", "Pts+Asts": "PA", "Pts+Rebs": "PR",
    "Rebs+Asts": "RA", "3-Pt Made": "FG3M", "Blocked Shots": "BLK",
    "Steals": "STL", "Turnovers": "TOV", "Fantasy Score": "FS",
    "FG Attempted": "FGA", "FT Made": "FTM",
    "Blks+Stls": "BLST", "Double Doubles": "DD", "Triple Doubles": "TD",
}


def _pp_request(per_page: int = 500, single_stat: str = "true") -> tuple[Optional[requests.Response], Optional[str]]:
    last_status = None

    # Session-cookie approach: hit app page → get tokens → use for API
    try:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                          "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                          "Mobile/15E148 Safari/604.1",
        })
        s.get("https://app.prizepicks.com", timeout=15)
        r = s.get(PRIZEPICKS_API, params={
            "per_page": str(per_page),
            "single_stat": single_stat,
            "in_play": "false",
        }, headers={
            "Accept": "application/json",
            "Referer": "https://app.prizepicks.com/",
            "Origin": "https://app.prizepicks.com",
        }, timeout=20)
        if r.ok:
            return r, None
        last_status = r.status_code
    except Exception as e:
        log.warning("PP session-cookie method failed: %s", e)

    return None, f"HTTP {last_status}" if last_status else "All methods failed"


def _parse_response(data: dict, league_filter: set | None = None) -> list[dict]:
    included = {
        item["id"]: item
        for item in data.get("included", [])
        if isinstance(item, dict) and "id" in item
    }
    league_map = {}
    for iid, item in included.items():
        if item.get("type") == "league":
            league_map[iid] = (item.get("attributes", {}) or {}).get("name", "")

    rows = []
    valid_types = {"projection", "new_player_projection", "boardprojection"}

    for proj in data.get("data", []):
        if not isinstance(proj, dict):
            continue
        ptype = str(proj.get("type", "")).lower()
        attrs = proj.get("attributes", {}) or {}
        has_fields = bool(attrs.get("stat_type") and attrs.get("line_score") is not None)
        if ptype not in valid_types and not has_fields:
            continue

        rels = proj.get("relationships", {}) or {}
        league_rel = (rels.get("league", {}).get("data", {}) or {}).get("id")
        league = league_map.get(league_rel, "") if league_rel else ""
        if not league:
            league = str(attrs.get("league", "") or "")

        if league_filter:
            league_upper = league.upper().replace(" ", "")
            is_nba = league_upper.startswith("NBA")
            if not is_nba:
                continue

        player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
        if not player_id:
            player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
        player_attrs = included.get(player_id, {}).get("attributes", {}) if player_id else {}
        player_name = player_attrs.get("name", "") or attrs.get("name", "") or attrs.get("display_name", "")
        team = player_attrs.get("team", "") or ""
        stat_type = attrs.get("stat_type", "")
        line_score = attrs.get("line_score")
        odds_type = str(attrs.get("odds_type", "") or "").lower().strip()

        if player_name and stat_type and line_score is not None:
            try:
                market = MARKET_MAP.get(stat_type)
                rows.append({
                    "player": player_name,
                    "stat_type": stat_type,
                    "market": market,
                    "line": float(line_score),
                    "team": team,
                    "start_time": attrs.get("start_time", ""),
                    "source": "PrizePicks",
                    "odds_type": odds_type or "standard",
                    "league": league,
                })
            except (TypeError, ValueError):
                pass
    return rows


def fetch_prizepicks_nba() -> tuple[list[dict], Optional[str]]:
    all_rows = []
    seen = set()

    for single_stat in ["true", "false"]:
        r, err = _pp_request(per_page=500, single_stat=single_stat)
        if err:
            if not all_rows:
                return [], err
            continue
        try:
            parsed = _parse_response(r.json(), league_filter={"NBA"})
            for row in parsed:
                key = (row["player"], row["stat_type"], row["line"])
                if key not in seen:
                    seen.add(key)
                    all_rows.append(row)
        except Exception as e:
            if not all_rows:
                return [], f"Parse error: {e}"

    log.info("Fetched %d NBA PrizePicks lines", len(all_rows))
    return all_rows, None


def fetch_prizepicks_all_sports() -> tuple[list[dict], Optional[str]]:
    all_rows = []
    seen = set()

    for single_stat in ["true", "false"]:
        r, err = _pp_request(per_page=500, single_stat=single_stat)
        if err:
            if not all_rows:
                return [], err
            continue
        try:
            parsed = _parse_response(r.json(), league_filter=None)
            for row in parsed:
                key = (row["player"], row["stat_type"], row["line"], row["league"])
                if key not in seen:
                    seen.add(key)
                    all_rows.append(row)
        except Exception as e:
            if not all_rows:
                return [], f"Parse error: {e}"

    log.info("Fetched %d total PrizePicks lines (all sports)", len(all_rows))
    return all_rows, None
