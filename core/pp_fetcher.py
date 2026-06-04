"""
core/pp_fetcher.py — Standalone PrizePicks fetcher.
No Streamlit dependency. Session-cookie approach with retry + UA rotation
for reliability from datacenter IPs (PerimeterX blocks are intermittent).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

PRIZEPICKS_API = "https://api.prizepicks.com/projections"

MARKET_MAP = {
    # NBA
    "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
    "Pts+Rebs+Asts": "PRA", "Pts+Asts": "PA", "Pts+Rebs": "PR",
    "Rebs+Asts": "RA", "3-Pt Made": "FG3M", "Blocked Shots": "BLK",
    "Steals": "STL", "Turnovers": "TOV", "Fantasy Score": "FS",
    "FG Attempted": "FGA", "FT Made": "FTM",
    "Blks+Stls": "BLST", "Double Doubles": "DD", "Triple Doubles": "TD",
    # MLB
    "Pitcher Strikeouts": "K", "Pitching Outs": "OUTS",
    "Earned Runs": "ER", "Hits Allowed": "HA", "Walks Allowed": "BB_A",
    "Total Bases": "TB", "Hits": "H", "Runs": "R", "RBIs": "RBI",
    "Stolen Bases": "SB", "Home Runs": "HR", "Hits+Runs+RBIs": "HRR",
    "Walks": "BB", "Singles": "1B", "Doubles": "2B", "Triples": "3B",
}

_USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.113 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-S928U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.6478.71 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
]

MAX_RETRIES = 4
BASE_DELAY = 3.0


def _pp_request(per_page: int = 500, single_stat: str = "true") -> tuple[Optional[requests.Response], Optional[str]]:
    last_status = None
    last_err_msg = None

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(1, 5)
            log.info("PP retry %d/%d — waiting %.1fs", attempt + 1, MAX_RETRIES, delay)
            time.sleep(delay)

        ua = random.choice(_USER_AGENTS)
        try:
            s = requests.Session()
            s.headers.update({"User-Agent": ua})

            # Random pre-delay on first attempt to avoid hitting exactly on the cron minute
            if attempt == 0:
                jitter = random.uniform(1, 8)
                time.sleep(jitter)

            s.get("https://app.prizepicks.com", timeout=15)
            time.sleep(random.uniform(0.5, 2.0))

            r = s.get(PRIZEPICKS_API, params={
                "per_page": str(per_page),
                "single_stat": single_stat,
                "in_play": "false",
            }, headers={
                "Accept": "application/json",
                "Referer": "https://app.prizepicks.com/",
                "Origin": "https://app.prizepicks.com",
                "X-Requested-With": "XMLHttpRequest",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            }, timeout=20)

            if r.ok:
                log.info("PP fetch OK on attempt %d (UA=%s…)", attempt + 1, ua[:30])
                return r, None

            last_status = r.status_code
            last_err_msg = f"HTTP {last_status}"
            log.warning("PP attempt %d/%d — HTTP %d", attempt + 1, MAX_RETRIES, last_status)

        except requests.exceptions.Timeout:
            last_err_msg = "Timeout"
            log.warning("PP attempt %d/%d — timeout", attempt + 1, MAX_RETRIES)
        except Exception as e:
            last_err_msg = str(e)
            log.warning("PP attempt %d/%d — %s", attempt + 1, MAX_RETRIES, e)

    return None, last_err_msg or "All attempts failed"


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
            # Goblin/demon lines have adjusted multipliers (~0.85x / ~1.25x)
            # and shifted lines that corrupt EV calculations. Only ingest
            # standard lines — the app UI already does this filtering.
            effective_odds = odds_type or "standard"
            if effective_odds not in ("standard", ""):
                continue
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
                    "odds_type": effective_odds,
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
