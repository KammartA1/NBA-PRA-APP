"""
Player Service — player data, stats, injury reports, metadata.

Uses NBA API for live data with DB caching.
Returns plain dicts.
"""
import json
import logging
import os
import re
import time
from datetime import datetime, date, timedelta

log = logging.getLogger(__name__)

# In-memory caches (populated lazily from NBA API)
_PLAYER_CACHE = {}        # {normalized_name: {id, full_name, ...}}
_STATS_CACHE = {}         # {(player_name, n_games): {stats_dict, ts}}
_STATS_CACHE_TTL = 1800   # 30 minutes

# Path for persisting player metadata on disk
_PLAYER_META_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "player_metadata.json",
)


def _normalize_name(name: str) -> str:
    """Normalize player name for matching (mirrors app.py normalize_name)."""
    if not name:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    s = re.sub(r"[.\'\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_player_meta() -> dict:
    try:
        if os.path.exists(_PLAYER_META_PATH):
            with open(_PLAYER_META_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_player_meta(data: dict):
    try:
        os.makedirs(os.path.dirname(_PLAYER_META_PATH), exist_ok=True)
        with open(_PLAYER_META_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        log.warning("Failed to save player metadata: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_player(
    name: str,
    team: str | None = None,
    position: str | None = None,
) -> dict:
    """
    Resolve a player by name. Returns player info dict with id, full_name,
    team, position. Creates a metadata entry if not found in NBA API.
    """
    norm = _normalize_name(name)
    if not norm:
        return {}

    # Check in-memory cache first
    if norm in _PLAYER_CACHE:
        return dict(_PLAYER_CACHE[norm])

    # Try NBA API lookup
    try:
        from nba_api.stats.static import players as nba_players
        plist = nba_players.get_players()
        for p in plist:
            if _normalize_name(p.get("full_name", "")) == norm:
                result = {
                    "id": p.get("id"),
                    "full_name": p.get("full_name", ""),
                    "first_name": p.get("first_name", ""),
                    "last_name": p.get("last_name", ""),
                    "is_active": p.get("is_active", False),
                    "team": team,
                    "position": position,
                }
                _PLAYER_CACHE[norm] = result
                return dict(result)

        # Fuzzy match fallback
        import difflib
        names = [p.get("full_name", "") for p in plist]
        candidates = difflib.get_close_matches(name, names, n=1, cutoff=0.75)
        if candidates:
            for p in plist:
                if p.get("full_name") == candidates[0]:
                    result = {
                        "id": p.get("id"),
                        "full_name": p.get("full_name", ""),
                        "first_name": p.get("first_name", ""),
                        "last_name": p.get("last_name", ""),
                        "is_active": p.get("is_active", False),
                        "team": team,
                        "position": position,
                        "fuzzy_match": True,
                        "original_query": name,
                    }
                    _PLAYER_CACHE[norm] = result
                    return dict(result)
    except Exception as exc:
        log.warning("NBA API player lookup failed: %s", exc)

    # Fallback: create entry in local metadata store
    meta = _load_player_meta()
    if norm in meta:
        cached = meta[norm]
        if team:
            cached["team"] = team
        if position:
            cached["position"] = position
        _PLAYER_CACHE[norm] = cached
        return dict(cached)

    # Create new entry
    new_entry = {
        "id": None,
        "full_name": name.strip(),
        "team": team,
        "position": position,
        "created_at": datetime.utcnow().isoformat(),
        "source": "manual",
    }
    meta[norm] = new_entry
    _save_player_meta(meta)
    _PLAYER_CACHE[norm] = new_entry
    return dict(new_entry)


def search_players(query: str) -> list[dict]:
    """
    Search players by partial name match.
    Returns up to 20 matching players.
    """
    if not query or len(query) < 2:
        return []
    query_lower = query.strip().lower()
    results = []
    try:
        from nba_api.stats.static import players as nba_players
        plist = nba_players.get_players()
        for p in plist:
            full = p.get("full_name", "")
            if query_lower in full.lower():
                results.append({
                    "id": p.get("id"),
                    "full_name": full,
                    "first_name": p.get("first_name", ""),
                    "last_name": p.get("last_name", ""),
                    "is_active": p.get("is_active", False),
                })
                if len(results) >= 20:
                    break
    except Exception as exc:
        log.warning("search_players NBA API failed: %s", exc)

    # Also search local metadata
    meta = _load_player_meta()
    for norm, entry in meta.items():
        if query_lower in norm:
            if not any(r.get("full_name") == entry.get("full_name") for r in results):
                results.append(entry)
                if len(results) >= 20:
                    break

    return results


def get_player_stats(player_name: str, n_games: int = 20) -> dict:
    """
    Fetch player game stats from NBA API. Caches results for 30 minutes.

    Returns dict with:
        player_id, games (list of game dicts), averages,
        position, team, n_games_returned
    """
    cache_key = (_normalize_name(player_name), n_games)
    # Check cache
    if cache_key in _STATS_CACHE:
        cached = _STATS_CACHE[cache_key]
        if time.time() - cached["ts"] < _STATS_CACHE_TTL:
            return cached["data"]

    result = {
        "player_name": player_name,
        "player_id": None,
        "games": [],
        "averages": {},
        "position": "",
        "team": None,
        "n_games_returned": 0,
        "errors": [],
    }

    # Resolve player ID
    player_info = get_or_create_player(player_name)
    player_id = player_info.get("id")
    if not player_id:
        result["errors"].append("Could not resolve NBA player ID")
        return result

    result["player_id"] = player_id
    result["position"] = player_info.get("position", "")

    # Fetch game log
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import teams as nba_teams
        import pandas as pd

        # Determine current season
        today = date.today()
        start = today.year if today.month >= 10 else today.year - 1
        season = f"{start}-{(start + 1) % 100:02d}"

        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
            timeout=15,
        )
        df = gl.get_data_frames()[0]
        if df.empty:
            result["errors"].append("Empty game log")
            return result

        df = df.head(n_games)
        result["n_games_returned"] = len(df)

        # Extract team from most recent game
        if "MATCHUP" in df.columns and not df.empty:
            matchup = str(df.iloc[0].get("MATCHUP", ""))
            parts = re.split(r'\s+(?:vs\.?|@)\s+', matchup, flags=re.IGNORECASE)
            if parts:
                result["team"] = parts[0].strip()

        # Convert game log to list of dicts
        stat_cols = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV",
                     "FGM", "FGA", "FTM", "FTA", "FG3A", "MIN", "PLUS_MINUS"]
        games_list = []
        for _, row in df.iterrows():
            game = {
                "game_date": str(row.get("GAME_DATE", "")),
                "matchup": str(row.get("MATCHUP", "")),
                "wl": str(row.get("WL", "")),
            }
            for col in stat_cols:
                if col in row.index:
                    val = row[col]
                    if col == "MIN" and isinstance(val, str) and ":" in val:
                        try:
                            val = float(val.split(":")[0])
                        except Exception:
                            val = 0.0
                    try:
                        game[col.lower()] = float(val) if val is not None else 0.0
                    except (ValueError, TypeError):
                        game[col.lower()] = 0.0
            games_list.append(game)

        result["games"] = games_list

        # Compute averages
        avgs = {}
        for col in ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov",
                     "fgm", "fga", "ftm", "fta", "fg3a", "min"]:
            vals = [g.get(col, 0.0) for g in games_list if g.get(col) is not None]
            if vals:
                avgs[col] = round(sum(vals) / len(vals), 1)
        # Computed combos
        if "pts" in avgs and "reb" in avgs and "ast" in avgs:
            avgs["pra"] = round(avgs["pts"] + avgs["reb"] + avgs["ast"], 1)
            avgs["pr"] = round(avgs["pts"] + avgs["reb"], 1)
            avgs["pa"] = round(avgs["pts"] + avgs["ast"], 1)
            avgs["ra"] = round(avgs["reb"] + avgs["ast"], 1)
        if "blk" in avgs and "stl" in avgs:
            avgs["stocks"] = round(avgs["blk"] + avgs["stl"], 1)

        result["averages"] = avgs

    except Exception as exc:
        log.error("get_player_stats failed: %s", exc)
        result["errors"].append(f"NBA API: {type(exc).__name__}: {exc}")

    # Cache result
    _STATS_CACHE[cache_key] = {"data": result, "ts": time.time()}
    return result


def get_injury_report() -> dict:
    """
    Get current NBA injury report.

    Returns: {team_abbr: [{"player": str, "status": str, "reason": str}, ...]}
    """
    try:
        import requests
        ESPN_INJURY_URL = "https://site.api.espn.com/apis/fantasy/v2/games/fba/injuries"
        r = requests.get(
            ESPN_INJURY_URL,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=15,
        )
        if not r.ok:
            return _injury_report_nba_api_fallback()

        data = r.json()
        out = {}
        for entry in data.get("injuries", []):
            team_info = entry.get("team", {})
            abbr = str(team_info.get("abbreviation", "")).upper()
            for inj in entry.get("injuries", []):
                athlete = inj.get("athlete", {})
                pname = athlete.get("fullName", "") or athlete.get("displayName", "")
                status_raw = str(inj.get("status", "")).upper()
                reason_detail = inj.get("details", {}) or {}
                reason = reason_detail.get("type", "") or reason_detail.get("returnDate", "")
                if pname and status_raw:
                    out.setdefault(abbr, []).append({
                        "player": pname,
                        "status": status_raw,
                        "reason": reason,
                    })
        if out:
            return out
        return _injury_report_nba_api_fallback()
    except Exception as exc:
        log.warning("ESPN injury fetch failed: %s", exc)
        return _injury_report_nba_api_fallback()


def _injury_report_nba_api_fallback() -> dict:
    """Fallback injury report via NBA API."""
    try:
        from nba_api.stats.endpoints import InjuryReport as NBAInjuryReport
        df = NBAInjuryReport(
            game_date=date.today().strftime("%m/%d/%Y")
        ).get_data_frames()[0]
        out = {}
        for _, r in df.iterrows():
            team = str(r.get("TEAM_TRICODE", "")).upper()
            status = str(r.get("PLAYER_STATUS", "")).upper()
            if status in ("OUT", "DOUBTFUL", "QUESTIONABLE"):
                out.setdefault(team, []).append({
                    "player": r.get("PLAYER_NAME", ""),
                    "status": status,
                    "reason": r.get("RETURN_FROM_INJURY", ""),
                })
        return out
    except Exception:
        return {}


def update_player_metadata(player_id: int | None, metadata: dict):
    """
    Update or set metadata for a player (team, position, custom fields).

    player_id is optional — if None, uses player name for lookup.
    metadata: dict with keys like team, position, notes, etc.
    """
    name = metadata.get("full_name", metadata.get("player_name", ""))
    if not name and player_id:
        # Try to resolve name from NBA API
        try:
            from nba_api.stats.static import players as nba_players
            for p in nba_players.get_players():
                if p.get("id") == player_id:
                    name = p.get("full_name", "")
                    break
        except Exception:
            pass

    if not name:
        log.warning("update_player_metadata: no player name available")
        return

    norm = _normalize_name(name)
    meta_db = _load_player_meta()
    existing = meta_db.get(norm, {})
    existing.update(metadata)
    existing["updated_at"] = datetime.utcnow().isoformat()
    if player_id:
        existing["id"] = player_id
    meta_db[norm] = existing
    _save_player_meta(meta_db)

    # Update in-memory cache too
    if norm in _PLAYER_CACHE:
        _PLAYER_CACHE[norm].update(metadata)
    log.info("Updated metadata for %s", name)
