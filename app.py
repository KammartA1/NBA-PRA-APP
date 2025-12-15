
# ============================================================
# NBA PROP MODEL — QUANT ENGINE (The Odds API + NBA API)
# Single-file Streamlit app
# - Live lines (player props) via The Odds API v4 (events/{id}/odds)
# - Opponent/context via nba_api (enhancement only; never blocks)
# - Live Scanner finds single-leg edges above threshold
# - History logging to local CSV (per user id)
# ============================================================

import os
import re
import math
import time
import json
import difflib
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import pandas as pd
import numpy as np
import requests
import streamlit as st

# nba_api (stats endpoints)
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import playergamelog, scoreboardv2




def current_nba_season_str(d: date) -> str:
    """Return season string like '2025-26' as expected by nba_api endpoints."""
    start_year = d.year if d.month >= 10 else d.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"



@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_player_gamelog_df(player_id: int, season_str: str, n_games: int = 15) -> pd.DataFrame:
    """
    Robust wrapper around nba_api PlayerGameLog across nba_api versions.
    - Tries season_nullable then season.
    - Returns most recent n_games (already sorted by date desc in response).
    """
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season_nullable=season_str)
        df = gl.get_data_frames()[0]
    except TypeError:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season_str)
        df = gl.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df.head(int(max(1, n_games))).copy()
def infer_team_opp_from_gamelog(gl: pd.DataFrame):
    """Infer team/opponent abbreviations from nba_api PlayerGameLog MATCHUP column."""
    try:
        if gl is None or gl.empty:
            return None, None
        matchup = str(gl.iloc[0].get("MATCHUP", "")).strip()
        # Common formats: 'BKN vs. MIA' or 'BKN @ MIA'
        parts = matchup.split()
        if len(parts) >= 3:
            team = parts[0].strip()
            opp = parts[2].strip()
            return team if team else None, opp if opp else None
    except Exception:
        pass
    return None, None


# ------------------------------
# Global constants
# ------------------------------
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA = "basketball_nba"
REGION_US = "us"

# The Odds API market keys (player props) — from The Odds API docs
# See: betting markets list (player_points, player_rebounds, player_assists, etc.)
ODDS_MARKETS = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
    "3PM": "player_threes",
    "PRA": "player_points_rebounds_assists",
    "PR": "player_points_rebounds",
    "PA": "player_points_assists",
    "RA": "player_rebounds_assists",
}

# Optional additional markets toggled in sidebar
ODDS_MARKETS_OPTIONAL = {
    "Blocks": "player_blocks",
    "Steals": "player_steals",
    "Turnovers": "player_turnovers",
}

# Simple stat mapping for projection from nba_api logs
STAT_FIELDS = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "3PM": "FG3M",
    "PRA": ("PTS", "REB", "AST"),
    "PR": ("PTS", "REB"),
    "PA": ("PTS", "AST"),
    "RA": ("REB", "AST"),
    "Blocks": "BLK",
    "Steals": "STL",
    "Turnovers": "TOV",
}

# ------------------------------
# Utilities
# ------------------------------

def _safe_rerun():
    """Compatibility wrapper for Streamlit rerun across versions."""
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def safe_float(x, default=None):
    """Best-effort float coercion for UI rendering and calculations."""
    try:
        if x is None or (isinstance(x, str) and x.strip()==""):
            return default
        return float(x)
    except Exception:
        return default

def safe_round(x, nd=2, default=None):
    v = safe_float(x, default=None)
    if v is None:
        return default
    try:
        return round(v, nd)
    except Exception:
        return default

def normalize_name(name: str) -> str:
    """Canonical player name normalization shared across:
       - line lookup
       - scanner
       - projections
       - cards
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\.\'\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def odds_api_key() -> str:
    # Never crash if missing; warn later
    return (
        st.secrets.get("ODDS_API_KEY", "")
        if hasattr(st, "secrets")
        else ""
    ) or os.getenv("ODDS_API_KEY", "")

def http_get_json(url: str, params: dict, timeout: int = 25):
    """Requests wrapper with robust error handling; never throws upstream without context."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        # store headers for credit tracking
        remaining = r.headers.get("x-requests-remaining") or r.headers.get("x-requests-remaining".title())
        used = r.headers.get("x-requests-used") or r.headers.get("x-requests-used".title())
        st.session_state["_odds_headers_last"] = {"remaining": remaining, "used": used, "ts": _now_iso()}
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        msg = f"HTTPError {getattr(e.response,'status_code',None)} on {url}"
        try:
            detail = e.response.text[:3000] if e.response is not None else ""
        except Exception:
            detail = ""
        return None, msg + (f"\n{detail}" if detail else "")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

@st.cache_data(ttl=60*10, show_spinner=False)
def get_team_maps():
    """Map between team full names and abbreviations/ids using nba_api static teams."""
    teams = nba_teams.get_teams()
    by_name = {}
    for t in teams:
        full = t.get("full_name", "")
        abbr = t.get("abbreviation", "")
        tid = t.get("id")
        by_name[normalize_name(full)] = {"abbr": abbr, "id": tid, "full_name": full}
    # common aliases (Odds API uses standard NBA names, but keep a few)
    alias = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        "gs warriors": "golden state warriors",
    }
    for a, tgt in alias.items():
        if normalize_name(tgt) in by_name:
            by_name[normalize_name(a)] = by_name[normalize_name(tgt)]
    return by_name

def map_team_name_to_abbr(team_name: str) -> str | None:
    if not team_name:
        return None
    m = get_team_maps()
    rec = m.get(normalize_name(team_name))
    return rec["abbr"] if rec else None

@st.cache_data(ttl=60*60*24, show_spinner=False)
def lookup_player_id(full_name: str):
    """Best-effort NBA player id lookup (does not block)."""
    if not full_name:
        return None
    nm = normalize_name(full_name)
    # exact / close matching via nba_api static list
    plist = nba_players.get_players()
    # first exact normalized match
    for p in plist:
        if normalize_name(p.get("full_name", "")) == nm:
            return p.get("id")
    # fallback: difflib
    names = [p.get("full_name", "") for p in plist]
    cand = difflib.get_close_matches(full_name, names, n=1, cutoff=0.75)
    if cand:
        for p in plist:
            if p.get("full_name") == cand[0]:
                return p.get("id")
    return None

def nba_headshot_url(player_id: int | None) -> str | None:
    if not player_id:
        return None
    # NBA CDN pattern used broadly
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

# ------------------------------
# Odds API ingestion
# ------------------------------
@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_events(date_iso: str | None = None):
    """Fetch NBA events. date_iso (YYYY-MM-DD) optional; if provided filters by that date on backend when supported."""
    key = odds_api_key()
    if not key:
        return [], "Missing ODDS_API_KEY in secrets or environment."
    url = f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events"
    params = {"apiKey": key}
    # Some Odds API deployments support dateFormat / commenceTimeFrom/To; keep minimal to reduce risk
    data, err = http_get_json(url, params=params)
    if err or not isinstance(data, list):
        return [], err or "Unexpected response for events."
    # Optional client-side filter by date if date_iso provided
    if date_iso:
        out = []
        for ev in data:
            ct = safe_get(ev, "commence_time")
            if not ct:
                continue
            try:
                d = ct[:10]
            except Exception:
                d = None
            if d == date_iso:
                out.append(ev)
        return out, None
    return data, None

def _markets_param(selected_market_keys: list[str]) -> str:
    # Odds API expects comma-separated market keys
    return ",".join(selected_market_keys)

@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_event_odds(event_id: str, market_keys: tuple, regions: str = REGION_US):
    """Fetch odds (including player props) for a given event."""
    key = odds_api_key()
    if not key:
        return None, "Missing ODDS_API_KEY in secrets or environment."
    url = f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{event_id}/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": _markets_param(list(market_keys)),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    data, err = http_get_json(url, params=params)
    return data, err

def extract_books_from_event_odds(event_odds: dict) -> list[str]:
    books = []
    for b in (event_odds or {}).get("bookmakers", []) or []:
        k = b.get("key")
        if k:
            books.append(k)
    books = sorted(list(dict.fromkeys(books)))
    return books

def _parse_player_prop_outcomes(event_odds: dict, market_key: str, book_filter: str | None = None):
    """
    Returns list of dict rows:
      {player, line, odds_price, book, side, market_key, event_id, home_team, away_team, commence_time}
    Supports:
      - book_filter=None => include all books
      - book_filter="consensus" => compute consensus line using median across books for each player/side
      - book_filter="<bookkey>" => only that book
    """
    if not event_odds:
        return [], "Empty event odds."
    event_id = event_odds.get("id")
    home = event_odds.get("home_team")
    away = event_odds.get("away_team")
    ct = event_odds.get("commence_time")
    books = event_odds.get("bookmakers", []) or []

    rows = []
    for b in books:
        bkey = b.get("key")
        if book_filter and book_filter not in ("consensus", "all") and bkey != book_filter:
            continue
        for mk in b.get("markets", []) or []:
            if mk.get("key") != market_key:
                continue
            for out in mk.get("outcomes", []) or []:
                player = out.get("description") or out.get("name")  # props use description for player in many payloads
                # For Over/Under props, name is typically "Over"/"Under", description is player
                side = out.get("name")
                point = out.get("point")
                price = out.get("price")
                if player and point is not None:
                    rows.append({
                        "player": player,
                        "player_norm": normalize_name(player),
                        "line": float(point),
                        "price": price,
                        "book": bkey or "",
                        "side": side or "",
                        "market_key": market_key,
                        "event_id": event_id,
                        "home_team": home,
                        "away_team": away,
                        "commence_time": ct,
                    })
    if book_filter == "consensus":
        # Build consensus by median line across books for each (player_norm, side)
        if not rows:
            return [], None
        df = pd.DataFrame(rows)
        # Keep only numeric lines
        df = df[pd.to_numeric(df["line"], errors="coerce").notna()].copy()
        if df.empty:
            return [], None
        g = df.groupby(["player_norm", "side"], dropna=False)["line"].median().reset_index()
        # use one representative player string (most common)
        name_map = df.groupby("player_norm")["player"].agg(lambda x: x.value_counts().index[0]).to_dict()
        out_rows = []
        for _, r in g.iterrows():
            out_rows.append({
                "player": name_map.get(r["player_norm"], r["player_norm"]),
                "player_norm": r["player_norm"],
                "line": float(r["line"]),
                "price": None,
                "book": "consensus",
                "side": r["side"],
                "market_key": market_key,
                "event_id": event_id,
                "home_team": home,
                "away_team": away,
                "commence_time": ct,
            })
        return out_rows, None

    return rows, None

def find_player_line_from_events(player_name: str, market_key: str, scan_date_iso: str, book_choice: str):
    """Find a player's line by searching events on a date and parsing event odds.
       Returns: (line, meta_dict, err)
    """
    evs, err = odds_get_events(scan_date_iso)
    if err:
        return None, None, err
    if not evs:
        return None, None, "No events returned for that date."
    target = normalize_name(player_name)
    # iterate events and look for player
    for ev in evs:
        eid = ev.get("id")
        if not eid:
            continue
        odds, oerr = odds_get_event_odds(eid, (market_key,))
        if oerr or not odds:
            continue
        rows, _ = _parse_player_prop_outcomes(odds, market_key, book_filter=book_choice)
        for r in rows:
            if r.get("player_norm") == target:
                return float(r["line"]), r, None
        # close match fallback
        norms = [r.get("player_norm","") for r in rows]
        close = difflib.get_close_matches(target, norms, n=1, cutoff=0.88)
        if close:
            rr = next((x for x in rows if x.get("player_norm")==close[0]), None)
            if rr:
                return float(rr["line"]), rr, None
    return None, None, "Player/market not found in Odds API props for that date/book."

# ------------------------------
# NBA API context (best-effort)
# ------------------------------
@st.cache_data(ttl=60*10, show_spinner=False)
def nba_scoreboard_games(game_date: date):
    """Fetch NBA scoreboard games and derive matchup mapping by team abbrev."""
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y"))
        df = sb.get_data_frames()[0]  # game header
        # Columns include HOME_TEAM_ID, VISITOR_TEAM_ID, GAME_ID
        games = []
        for _, row in df.iterrows():
            games.append({
                "game_id": str(row.get("GAME_ID")),
                "home_team_id": int(row.get("HOME_TEAM_ID")),
                "away_team_id": int(row.get("VISITOR_TEAM_ID")),
            })
        # map team id -> opponent id for that day (if multiple games, keep first)
        return games, None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

@st.cache_data(ttl=60*60*24, show_spinner=False)
def team_id_to_abbr_map():
    m = {}
    for t in nba_teams.get_teams():
        m[int(t["id"])] = t["abbreviation"]
    return m

def resolve_opponent_from_odds_event(meta: dict) -> tuple[str|None, str|None]:
    """From Odds API event meta (home_team, away_team), return (team_abbr, opp_abbr) if resolvable."""
    home = meta.get("home_team") if meta else None
    away = meta.get("away_team") if meta else None
    if not home or not away:
        return None, None
    home_abbr = map_team_name_to_abbr(home)
    away_abbr = map_team_name_to_abbr(away)
    # If player team is unknown here, return both; caller can pick
    return home_abbr, away_abbr

def resolve_player_team_abbr_from_nba(player_id: int, game_date: date) -> str | None:
    """Best-effort: infer player's team abbreviation from most recent game log up to date."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season_nullable=current_nba_season_str(game_date))
        df = gl.get_data_frames()[0]
        if df.empty:
            return None
        # MATCHUP like "PHI vs. MIL" or "PHI @ MIL"
        matchup = str(df.iloc[0].get("MATCHUP",""))
        team_abbr = matchup.split(" ")[0].strip() if matchup else None
        return team_abbr or None
    except Exception:
        return None

def opponent_from_team_abbr(team_abbr: str, game_date: date) -> str | None:
    games, _ = nba_scoreboard_games(game_date)
    if not games or not team_abbr:
        return None
    tid_map = team_id_to_abbr_map()
    for g in games:
        ha = tid_map.get(g["home_team_id"])
        aa = tid_map.get(g["away_team_id"])
        if ha == team_abbr:
            return aa
        if aa == team_abbr:
            return ha
    return None


# ------------------------------
# Advanced context engines (legacy stack)
# ------------------------------

@st.cache_data(ttl=60*60, show_spinner=False)
def get_team_context_current_season():
    """Team opponent context (pace/defense/assist/reb profiles). Best-effort; never blocks the app."""
    try:
        # Determine current NBA season in NBA API format YYYY-YY (rolls in Oct)
        today = datetime.now()
        year = today.year if today.month >= 10 else today.year - 1
        season = f"{year}-{str(year+1)[-2:]}"
        from nba_api.stats.endpoints import leaguedashteamstats
        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season, per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","REB_PCT","OREB_PCT","DREB_PCT","AST_PCT","PACE"
        ]]

        defn = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"
        ]]

        df = base.merge(adv, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left").merge(defn, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")

        league_avg = {col: float(df[col].mean()) for col in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]}
        team_ctx = {
            str(r["TEAM_ABBREVIATION"]).upper(): {
                "PACE": float(r.get("PACE", np.nan)),
                "DEF_RATING": float(r.get("DEF_RATING", np.nan)),
                "REB_PCT": float(r.get("REB_PCT", np.nan)),
                "DREB_PCT": float(r.get("DREB_PCT", np.nan)),
                "AST_PCT": float(r.get("AST_PCT", np.nan)),
            } for _, r in df.iterrows()
        }
        return team_ctx, league_avg, None
    except Exception as e:
        return {}, {}, f"{type(e).__name__}: {e}"

TEAM_CTX, LEAGUE_CTX, _TEAM_CTX_ERR = get_team_context_current_season()

PLAYER_POSITION_CACHE: dict[str, str] = {}

def get_player_position(name: str) -> str:
    """Resolve player position with nba_api (cached)."""
    key = normalize_name(name)
    if not key:
        return "Unknown"
    if key in PLAYER_POSITION_CACHE:
        return PLAYER_POSITION_CACHE[key]
    try:
        pid = lookup_player_id(name)
        pos = "Unknown"
        if pid:
            from nba_api.stats.endpoints import commonplayerinfo
            info = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
            raw = str(info.get("POSITION", "") or "")
            pos = raw if raw else "Unknown"
        PLAYER_POSITION_CACHE[key] = pos
        return pos
    except Exception:
        PLAYER_POSITION_CACHE[key] = "Unknown"
        return "Unknown"

def get_position_bucket(pos: str) -> str:
    if not pos:
        return "Unknown"
    p = pos.upper()
    if p.startswith("G"):
        return "Guard"
    if p.startswith("F"):
        return "Wing"
    if p.startswith("C"):
        return "Big"
    if "G" in p and "F" in p:
        return "Wing"
    if "F" in p and "C" in p:
        return "Big"
    return "Unknown"

def get_context_multiplier(opp_abbrev: str | None, market: str, position: str | None = None) -> float:
    """Opponent/pace/defense/positional context multiplier. Deterministic fallback if TEAM_CTX unavailable."""
    def _fallback(opp: str | None) -> float:
        base = 1.0
        bucket = get_position_bucket(position or "")
        if bucket == "Guard" and market in ["Assists","PA","PRA","RA"]:
            base *= 1.03
        elif bucket == "Big" and market in ["Rebounds","PR","PRA","RA","Blocks"]:
            base *= 1.04
        if opp:
            h = sum(ord(c) for c in opp.strip().upper())
            offset = ((h % 15) - 7) / 200.0
            base *= (1.0 + offset)
        return float(np.clip(base, 0.90, 1.10))

    if not opp_abbrev:
        return _fallback(None)
    opp_key = opp_abbrev.strip().upper()

    if not TEAM_CTX or not LEAGUE_CTX or opp_key not in TEAM_CTX:
        return _fallback(opp_key)

    opp = TEAM_CTX[opp_key]
    pace_f = float(opp["PACE"] / max(LEAGUE_CTX["PACE"], 1e-6))
    def_f = float(max(LEAGUE_CTX["DEF_RATING"], 1e-6) / max(opp["DEF_RATING"], 1e-6))

    reb_adj = float(max(LEAGUE_CTX.get("REB_PCT",1.0),1e-6) / max(opp.get("DREB_PCT",1.0),1e-6)) if market in ["Rebounds","PR","PRA","RA"] else 1.0
    ast_adj = float(max(LEAGUE_CTX.get("AST_PCT",1.0),1e-6) / max(opp.get("AST_PCT",1.0),1e-6)) if market in ["Assists","PA","PRA","RA"] else 1.0

    bucket = get_position_bucket(position or "")
    pos_factor = 1.0
    if bucket == "Guard":
        pos_factor = 0.5 * float(opp["AST_PCT"] / max(LEAGUE_CTX["AST_PCT"],1e-6)) + 0.5 * pace_f
    elif bucket == "Wing":
        pos_factor = 0.5 * def_f + 0.5 * pace_f
    elif bucket == "Big":
        pos_factor = 0.6 * float(max(LEAGUE_CTX["REB_PCT"],1e-6) / max(opp["DREB_PCT"],1e-6)) + 0.4 * def_f

    if market in ["Rebounds"]:
        mult = 0.30*pace_f + 0.25*def_f + 0.30*reb_adj + 0.15*pos_factor
    elif market in ["Assists"]:
        mult = 0.30*pace_f + 0.25*def_f + 0.30*ast_adj + 0.15*pos_factor
    elif market in ["RA"]:  # rebs+asts
        mult = 0.25*pace_f + 0.20*def_f + 0.25*reb_adj + 0.20*ast_adj + 0.10*pos_factor
    else:
        mult = 0.45*pace_f + 0.40*def_f + 0.15*pos_factor

    return float(np.clip(mult, 0.80, 1.30))

# ------------------------------
# Calibration (self-learning)
# ------------------------------
def calibration_file_path(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", (user_id or "default"))
    return f"calibration_state_{safe}.json"

def load_calibration_state(user_id: str) -> dict:
    fp = calibration_file_path(user_id)
    if not os.path.exists(fp):
        return {"prob_scale": 1.0, "last_updated": None}
    try:
        with open(fp, "r") as f:
            return json.load(f)
    except Exception:
        return {"prob_scale": 1.0, "last_updated": None}

def save_calibration_state(user_id: str, prob_scale: float):
    fp = calibration_file_path(user_id)
    state = {"prob_scale": float(prob_scale), "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")}
    with open(fp, "w") as f:
        json.dump(state, f)

def apply_calibration_to_prob(p: float, cal_state: dict) -> float:
    scale = float(cal_state.get("prob_scale", 1.0))
    centered = float(p) - 0.5
    adj = 0.5 + centered * scale
    return float(np.clip(adj, 0.02, 0.98))

# ------------------------------
# Blowout risk from Odds API spreads
# ------------------------------
@st.cache_data(ttl=60*10, show_spinner=False)
def odds_get_event_spread_abs(event_id: str) -> float | None:
    """Returns median absolute spread across books for the event (if available)."""
    odds, err = odds_get_event_odds(event_id, ("spreads",))
    if err or not odds:
        return None
    spreads = []
    for b in odds.get("bookmakers", []) or []:
        for mk in b.get("markets", []) or []:
            if mk.get("key") != "spreads":
                continue
            for out in mk.get("outcomes", []) or []:
                pt = out.get("point")
                if pt is None:
                    continue
                try:
                    spreads.append(abs(float(pt)))
                except Exception:
                    pass
    if not spreads:
        return None
    return float(np.median(spreads))

def estimate_blowout_risk_from_spread(spread_abs: float | None) -> float:
    if spread_abs is None:
        return 0.10
    s = float(abs(spread_abs))
    if s < 5:
        return 0.08
    if s < 10:
        return 0.14
    if s < 15:
        return 0.22
    return 0.30

def estimate_minutes_distribution(minutes: np.ndarray, blowout_prob: float, key_teammate_out: bool) -> tuple[float, float]:
    mins = np.array(minutes, dtype=float)
    if mins.size == 0:
        base_mean, base_sd = 32.0, 4.0
    else:
        base_mean = float(mins.mean())
        base_sd = float(max(mins.std(ddof=1), 2.0))
    if key_teammate_out:
        base_mean *= 1.05
    base_mean *= float(1.0 - 0.35 * np.clip(blowout_prob, 0.0, 0.8))
    return base_mean, base_sd

def role_based_bayesian_prior(samples: np.ndarray, line: float) -> np.ndarray:
    s = np.array(samples, dtype=float)
    if s.size == 0:
        return np.array([float(line)])
    prior_mean = float(line)
    prior_sd = float(max(np.std(s), 4.0))
    prior_draws = np.random.default_rng(13).normal(prior_mean, prior_sd, size=2000)
    return np.concatenate([s, prior_draws])

def player_similarity_factor(samples: np.ndarray) -> float:
    s = np.array(samples, dtype=float)
    if s.size < 4:
        return 1.0
    sd = float(np.std(s))
    mu = float(np.mean(s))
    if mu <= 0:
        return 1.0
    cv = sd / mu
    if cv < 0.25:
        return 0.9
    if cv > 0.6:
        return 1.1
    return 1.0

def game_script_simulation(samples: np.ndarray, minutes_params: tuple[float,float], line: float, ctx_mult: float, blowout_prob: float, n_sims: int = 6000):
    s = np.array(samples, dtype=float)
    if s.size == 0:
        return 0.5, 0.0, 1.0
    m_mean, m_sd = minutes_params
    m_sd = float(max(m_sd, 1.5))
    blowout_prob = float(np.clip(blowout_prob, 0.0, 0.8))

    base_comp = 0.70 - 0.4 * blowout_prob
    base_mild = 0.20 + 0.25 * blowout_prob
    base_severe = 0.10 + 0.15 * blowout_prob
    total = base_comp + base_mild + base_severe
    w_comp, w_mild, w_severe = base_comp/total, base_mild/total, base_severe/total

    rng = np.random.default_rng(17)
    draws = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        r = rng.random()
        if r < w_comp:
            min_draw = rng.normal(m_mean, m_sd*0.6)
        elif r < w_comp + w_mild:
            min_draw = rng.normal(m_mean*0.9, m_sd)
        else:
            min_draw = rng.normal(m_mean*0.8, m_sd*1.2)
        min_mult = float(min_draw / max(m_mean, 1e-6))
        base_sample = float(rng.choice(s))
        draws[i] = base_sample * min_mult * float(ctx_mult)

    mu = float(draws.mean())
    sd = float(max(draws.std(ddof=1), 0.75))
    p_over = float((draws > float(line)).mean())
    return p_over, mu, sd


# ------------------------------
# Projection engine (full legacy stack + Odds API lines)
# ------------------------------
def compute_leg_projection(player_name: str, market_name: str, line: float, meta: dict | None, n_games: int, key_teammate_out: bool, user_id: str):
    """Compute single leg outputs using:
       - Empirical bootstrap Monte Carlo (10,000)
       - Game-script minutes simulation (blowout-adjusted)
       - Defensive/pace/positional context multiplier
       - Bayesian prior around the market line
       - Volatility labeling
       - Self-learning calibration (per user)
    """
    errors = []

    # Resolve player id and pull game log
    player_id = lookup_player_id(player_name)
    if not player_id:
        errors.append("Could not resolve NBA player id (name mismatch).")
        return {
            "player": player_name, "market": market_name, "line": float(line),
            "proj": None, "p_over": None, "edge": None,
            "team": None, "opp": None, "headshot": None,
            "blowout_prob": 0.10, "ctx_mult": 1.00, "usage_boost": 1.00,
            "matchup_text": "Unknown matchup.", "position": "Unknown", "pos_bucket": "Unknown",
            "volatility_cv": None, "volatility_label": None,
            "errors": errors,
        }

    # Fetch full season gamelog but keep last n_games
    try:
        gl = fetch_player_gamelog_df(player_id, current_nba_season_str(date.today()), n_games=n_games)
    except Exception as e:
        errors.append(f"NBA API gamelog error: {type(e).__name__}")
        gl = pd.DataFrame()

    if gl is None or gl.empty:
        errors.append("No recent game logs returned.")
        return {
            "player": player_name, "market": market_name, "line": float(line),
            "proj": None, "p_over": None, "edge": None,
            "team": None, "opp": None, "headshot": nba_headshot_url(player_id),
            "blowout_prob": 0.10, "ctx_mult": 1.00, "usage_boost": 1.00,
            "matchup_text": "No data.", "position": get_player_position(player_name), "pos_bucket": get_position_bucket(get_player_position(player_name)),
            "volatility_cv": None, "volatility_label": None,
            "errors": errors,
        }

    # Normalize ordering newest first
    try:
        gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce")
        gl = gl.sort_values("GAME_DATE", ascending=False)
    except Exception:
        pass
    gl = gl.head(int(max(5, n_games))).copy()

    # Minutes parsing
    mins = []
    for v in gl.get("MIN", []):
        try:
            if isinstance(v, str) and ":" in v:
                mm, ss = v.split(":")
                mins.append(float(mm) + float(ss) / 60.0)
            else:
                mins.append(float(v))
        except Exception:
            pass
    minutes_arr = np.array(mins, dtype=float)

    # Stat samples for this market
    stat_series = compute_stat_from_gamelog(gl, market_name)
    samples = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if samples.size < 3:
        errors.append("Insufficient stat samples for robust simulation (using fallback).")

    # Team + opponent inference
    gdate = date.today()
    team_abbr, opp_abbr = infer_team_opp_from_gamelog(gl)

    # Prefer Odds API event metadata when available
    if meta:
        home_abbr, away_abbr = resolve_opponent_from_odds_event(meta)
        if home_abbr and away_abbr:
            if team_abbr:
                if team_abbr == home_abbr:
                    opp_abbr = away_abbr
                elif team_abbr == away_abbr:
                    opp_abbr = home_abbr
            else:
                # If we cannot infer the player's team, at least attach the event teams
                team_abbr, opp_abbr = home_abbr, away_abbr

    # Fallback to nba_api team resolver (non-blocking)
    if not team_abbr:
        try:
            team_abbr = resolve_player_team_abbr_from_nba(player_id, gdate)
        except Exception:
            team_abbr = None

    if team_abbr and not opp_abbr:
        try:
            opp_abbr = opponent_from_team_abbr(team_abbr, gdate)
        except Exception:
            opp_abbr = None

# Position + context multiplier
    position_raw = get_player_position(player_name)
    pos_bucket = get_position_bucket(position_raw)
    ctx_mult = get_context_multiplier(opp_abbr, market_name, position_raw)

    # Usage boost (recent minutes trend)
    avg_min = float(minutes_arr.mean()) if minutes_arr.size else 32.0
    recent_min = float(np.mean(minutes_arr[: min(3, minutes_arr.size)])) if minutes_arr.size else avg_min
    usage_boost = 1.0
    if recent_min >= avg_min + 4:
        usage_boost = 1.06
    elif recent_min <= max(10.0, avg_min - 5):
        usage_boost = 0.94

    # Blowout risk from spreads if we have event id
    spread_abs = None
    try:
        eid = (meta or {}).get("event_id") or (meta or {}).get("id")
        if eid:
            spread_abs = odds_get_event_spread_abs(str(eid))
    except Exception:
        spread_abs = None
    blowout_prob = estimate_blowout_risk_from_spread(spread_abs)

    # Minutes distribution engine
    min_mean, min_sd = estimate_minutes_distribution(minutes_arr, blowout_prob, key_teammate_out=bool(key_teammate_out))

    # Build base samples with context + usage; mild blowout shrink
    base_samples = samples.astype(float) if samples.size else np.array([float(line)])
    base_samples = base_samples * float(ctx_mult) * float(usage_boost)
    if blowout_prob >= 0.20:
        base_samples *= 0.96

    # Bayesian prior around market line + volatility shaping
    rng = np.random.default_rng(23)
    bayes = role_based_bayesian_prior(base_samples, float(line))
    sim_factor = player_similarity_factor(bayes)
    bayes = (bayes - bayes.mean()) * sim_factor + bayes.mean()

    # Game-script simulation
    gs_prob, mu_gs, sd_gs = game_script_simulation(bayes, (min_mean, min_sd), float(line), float(ctx_mult), blowout_prob)

    # Empirical bootstrap MC (10k)
    n_sims = 10000
    draws_emp = rng.choice(base_samples, size=n_sims, replace=True) if base_samples.size else rng.normal(float(line), 4.0, size=n_sims)
    p_over_emp = float((draws_emp > float(line)).mean())
    mu_emp = float(draws_emp.mean())
    sd_emp = float(max(draws_emp.std(ddof=1), 0.75))

    # Blend
    p_over_raw = 0.6 * p_over_emp + 0.4 * gs_prob
    mu = 0.6 * mu_emp + 0.4 * mu_gs
    sd = 0.6 * sd_emp + 0.4 * sd_gs

    # Volatility label
    vol_sd = float(np.std(bayes))
    vol_mu = float(max(np.mean(bayes), 1e-6))
    vol_cv = float(vol_sd / vol_mu)
    if vol_cv < 0.25:
        vol_label = "Low"
    elif vol_cv < 0.45:
        vol_label = "Medium"
    else:
        vol_label = "High"

    # Calibration
    cal_state = load_calibration_state(user_id)
    p_over = apply_calibration_to_prob(p_over_raw, cal_state)

    # Edge baseline vs 50/50
    edge = float(p_over - 0.5)

    # Projection shown is calibrated mean with context (mu already includes ctx_mult usage)
    proj = float(mu)

    # Matchup text
    matchup_text = "Neutral matchup."
    if opp_abbr and TEAM_CTX and LEAGUE_CTX and opp_abbr.strip().upper() in TEAM_CTX:
        opp_ctx = TEAM_CTX[opp_abbr.strip().upper()]
        pace_rel = float(opp_ctx["PACE"] / max(LEAGUE_CTX["PACE"], 1e-6))
        def_rel = float(opp_ctx["DEF_RATING"] / max(LEAGUE_CTX["DEF_RATING"], 1e-6))
        pieces = []
        if def_rel > 1.06:
            pieces.append("strong team defense")
        elif def_rel < 0.94:
            pieces.append("soft team defense")
        if pace_rel > 1.05:
            pieces.append("very fast pace")
        elif pace_rel < 0.95:
            pieces.append("slow tempo")
        if pos_bucket == "Guard":
            pieces.append("guard-centric matchup")
        elif pos_bucket == "Wing":
            pieces.append("wing matchup emphasis")
        elif pos_bucket == "Big":
            pieces.append("paint/big-man emphasis")
        matchup_text = (", ".join(pieces).capitalize() + ".") if pieces else "Slightly neutral matchup with no extreme flags."

    return {
        "player": player_name,
        "market": market_name,
        "line": float(line),
        "proj": float(proj),
        "p_over": float(p_over),
        "p_over_raw": float(p_over_raw),
        "edge": float(edge),
        "mu": float(mu),
        "sd": float(sd),
        "team": team_abbr,
        "opp": opp_abbr,
        "headshot": nba_headshot_url(player_id),
        "blowout_prob": float(blowout_prob),
        "spread_abs": None if spread_abs is None else float(spread_abs),
        "ctx_mult": float(ctx_mult),
        "usage_boost": float(usage_boost),
        "matchup_text": matchup_text,
        "position": position_raw,
        "pos_bucket": pos_bucket,
        "volatility_cv": float(vol_cv),
        "volatility_label": vol_label,
        "errors": errors,
    }



# ------------------------------
# History persistence (per user) — supports PENDING + HIT/MISS/PUSH + stake
# ------------------------------
HISTORY_COLUMNS = [
    "bet_id","ts_utc","date_local","user_id",
    "entry_type","stake","payout_mult",
    "legs_json",
    "status","pnl","notes",
    "avg_prob","avg_edge","avg_volatility_cv",
    "clv","kelly_frac",
]

def history_path(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", (user_id or "default").strip() or "default")
    return f"history_{safe}.csv"

def _ensure_history_file(user_id: str):
    fp = history_path(user_id)
    if not os.path.exists(fp):
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(fp, index=False)

def load_history(user_id: str) -> pd.DataFrame:
    _ensure_history_file(user_id)
    fp = history_path(user_id)
    try:
        df = pd.read_csv(fp)
        # normalize status casing
        if "status" in df.columns:
            df["status"] = df["status"].astype(str).str.upper()
        return df
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

def save_history(user_id: str, df: pd.DataFrame):
    fp = history_path(user_id)
    df.to_csv(fp, index=False)

def append_history(user_id: str, entry: dict):
    df = load_history(user_id)
    df2 = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    save_history(user_id, df2)

def update_history_status(user_id: str, bet_id: str, new_status: str):
    df = load_history(user_id)
    if df.empty or "bet_id" not in df.columns:
        return
    new_status_u = (new_status or "").strip().upper()
    if new_status_u not in ("PENDING","HIT","MISS","PUSH"):
        return
    idx = df.index[df["bet_id"].astype(str) == str(bet_id)]
    if len(idx) == 0:
        return
    i = idx[0]
    stake = float(df.loc[i, "stake"]) if "stake" in df.columns and pd.notna(df.loc[i, "stake"]) else 0.0
    payout_mult = float(df.loc[i, "payout_mult"]) if "payout_mult" in df.columns and pd.notna(df.loc[i, "payout_mult"]) else 2.0
    pnl = 0.0
    if new_status_u == "HIT":
        pnl = stake * (payout_mult - 1.0)
    elif new_status_u == "MISS":
        pnl = -stake
    else:
        pnl = 0.0

    df.loc[i, "status"] = new_status_u
    df.loc[i, "pnl"] = float(pnl)
    save_history(user_id, df)


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="NBA Prop Model — Quant Engine", layout="wide")

# Styling close to prior look
st.markdown("""
<style>
/* App background */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(110,0,20,0.55), rgba(10,10,16,0.95) 55%, rgba(0,0,0,0.98) 100%);
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(110,0,20,0.92), rgba(40,0,12,0.92));
}
div[data-testid="stSidebarContent"] * { color: #f3f3f3; }
/* Cards */
.block-container { padding-top: 4rem !important; }
/* Prevent custom headers from covering tabs */
div[data-testid="stHeader"] { z-index: 0 !important; }
div[data-testid="stTabs"] { position: relative; z-index: 5 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## User & Bankroll")
user_id = st.sidebar.text_input("Your ID (for personal history)", value=st.session_state.get("user_id", "Me"))
st.session_state["user_id"] = user_id

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=0.0, value=float(st.session_state.get("bankroll", 100.0)), step=10.0)
st.session_state["bankroll"] = bankroll

payout_2pick = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.0, value=float(st.session_state.get("payout_2pick", 3.0)), step=0.1)
st.session_state["payout_2pick"] = payout_2pick

frac_kelly = st.sidebar.slider("Fractional Kelly", min_value=0.0, max_value=1.0, value=float(st.session_state.get("frac_kelly", 0.25)), step=0.05)
st.session_state["frac_kelly"] = frac_kelly

n_games = st.sidebar.slider("Recent Games Sample (N)", min_value=5, max_value=30, value=int(st.session_state.get("n_games", 10)))
st.session_state["n_games"] = n_games

compact = st.sidebar.checkbox("Compact Mode (mobile)", value=bool(st.session_state.get("compact", False)))
st.session_state["compact"] = compact

max_daily_loss = st.sidebar.slider("Max Daily Loss % (stop)", 0, 50, int(st.session_state.get("max_daily_loss", 15)))
max_weekly_loss = st.sidebar.slider("Max Weekly Loss % (stop)", 0, 50, int(st.session_state.get("max_weekly_loss", 25)))
st.session_state["max_daily_loss"] = max_daily_loss
st.session_state["max_weekly_loss"] = max_weekly_loss

with st.sidebar.expander("The Odds API Settings", expanded=True):
    include_optional = st.checkbox("Include Rebs+Asts market", value=bool(st.session_state.get("include_ra", True)), key="include_ra")
    max_requests_per_day = st.number_input("Max Odds API requests per day", min_value=1, value=int(st.session_state.get("max_req_day", 60)), step=5)
    st.session_state["max_req_day"] = int(max_requests_per_day)

# Credits display
hdr = st.session_state.get("_odds_headers_last", {})
if hdr:
    st.sidebar.caption(f"Odds API usage — used: {hdr.get('used','?')} | remaining: {hdr.get('remaining','?')} | last: {hdr.get('ts','')}")

# Market list for UI
MARKET_OPTIONS = ["Points", "Rebounds", "Assists", "PRA", "PR", "PA", "RA", "3PM"]
if include_optional:
    MARKET_OPTIONS.append("RA")  # already included; harmless
# add optional
MARKET_OPTIONS += ["Blocks", "Steals", "Turnovers"]

# Tabs
tabs = st.tabs(["📊 Model", "📈 Results", "⚡ Live Scanner", "🗂️ History", "🧪 Calibration"])

# Shared session results container
if "last_results" not in st.session_state:
    st.session_state["last_results"] = []
if "last_board_meta" not in st.session_state:
    st.session_state["last_board_meta"] = {}

def get_sportsbook_choices(scan_date_iso: str):
    # derive from events -> first event odds (minimal)
    evs, err = odds_get_events(scan_date_iso)
    if err or not evs:
        return ["consensus"], err
    # use first event with odds
    for ev in evs[:6]:
        eid = ev.get("id")
        if not eid:
            continue
        mk = tuple({ODDS_MARKETS["Points"]})  # cheap call
        odds, oerr = odds_get_event_odds(eid, mk)
        if odds and not oerr:
            books = extract_books_from_event_odds(odds)
            return ["consensus"] + books, None
    return ["consensus"], "Could not enumerate sportsbooks (Odds API response missing bookmakers)."

with tabs[0]:
    st.markdown("### Up to 4-Leg Projection & Edge (Bootstrap + Context)")
    scan_date = st.date_input("Lines date", value=date.today())
    scan_date_iso = scan_date.isoformat()

    # sportsbook dropdown
    book_choices, book_err = get_sportsbook_choices(scan_date_iso)
    if book_err:
        st.info(book_err)
    sportsbook = st.selectbox("Sportsbook (live lines)", options=book_choices, index=0, help="Choose a book key or consensus (median across books).")

    colA, colB = st.columns(2)
    players = []
    with colA:
        p1 = st.text_input("Player 1 Name", value=st.session_state.get("p1",""))
        m1 = st.selectbox("P1 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m1","Points")) if st.session_state.get("m1","Points") in MARKET_OPTIONS else 0)
        o1 = st.checkbox("P1: Manual line override", value=False)
        l1 = st.number_input("P1 Line", min_value=0.0, value=float(st.session_state.get("l1", 25.0)), step=0.5)
        t1 = st.text_input("P1 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp1",""))
        out1 = st.checkbox("P1 Key teammate OUT?", value=False)
        players.append(("P1", p1, m1, o1, l1, t1, out1))

        p3 = st.text_input("Player 3 Name", value=st.session_state.get("p3",""))
        m3 = st.selectbox("P3 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m3","PRA")) if st.session_state.get("m3","PRA") in MARKET_OPTIONS else 0)
        o3 = st.checkbox("P3: Manual line override", value=False)
        l3 = st.number_input("P3 Line", min_value=0.0, value=float(st.session_state.get("l3", 25.0)), step=0.5)
        t3 = st.text_input("P3 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp3",""))
        out3 = st.checkbox("P3 Key teammate OUT?", value=False)
        players.append(("P3", p3, m3, o3, l3, t3, out3))

    with colB:
        p2 = st.text_input("Player 2 Name", value=st.session_state.get("p2",""))
        m2 = st.selectbox("P2 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m2","Points")) if st.session_state.get("m2","Points") in MARKET_OPTIONS else 0)
        o2 = st.checkbox("P2: Manual line override", value=False)
        l2 = st.number_input("P2 Line", min_value=0.0, value=float(st.session_state.get("l2", 25.0)), step=0.5)
        t2 = st.text_input("P2 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp2",""))
        out2 = st.checkbox("P2 Key teammate OUT?", value=False)
        players.append(("P2", p2, m2, o2, l2, t2, out2))

        p4 = st.text_input("Player 4 Name", value=st.session_state.get("p4",""))
        m4 = st.selectbox("P4 Market", options=MARKET_OPTIONS, index=MARKET_OPTIONS.index(st.session_state.get("m4","PRA")) if st.session_state.get("m4","PRA") in MARKET_OPTIONS else 0)
        o4 = st.checkbox("P4: Manual line override", value=False)
        l4 = st.number_input("P4 Line", min_value=0.0, value=float(st.session_state.get("l4", 25.0)), step=0.5)
        t4 = st.text_input("P4 Opponent (Team Abbrev, optional)", value=st.session_state.get("opp4",""))
        out4 = st.checkbox("P4 Key teammate OUT?", value=False)
        players.append(("P4", p4, m4, o4, l4, t4, out4))

    run = st.button("Run Model ⚡")

    if run:
        results = []
        warnings = []
        for tag, pname, mkt, manual, manual_line, opp_in, teammate_out in players:
            pname = (pname or "").strip()
            if not pname:
                continue

            market_key = ODDS_MARKETS.get(mkt) or ODDS_MARKETS_OPTIONAL.get(mkt)
            if not market_key:
                warnings.append(f"{tag}: Unsupported market {mkt}")
                continue

            line = manual_line
            meta = None

            if not manual:
                line_found, meta, ferr = find_player_line_from_events(pname, market_key, scan_date_iso, sportsbook)
                if ferr:
                    warnings.append(f"{tag}: Could not auto-fetch The Odds API line ({ferr}). Enable manual override.")
                else:
                    line = float(line_found)
                    st.info(f"{tag} auto The Odds API line detected: {line}")

            if line is None or float(line) <= 0:
                warnings.append(f"{tag}: does not have a valid line.")
                continue

            leg = compute_leg_projection(
                player_name=pname,
                market_name=mkt,
                line=float(line),
                meta=meta,
                n_games=n_games,
                key_teammate_out=bool(teammate_out),
                user_id=user_id,
            )

            # user-provided opponent overrides do not block; they only override display
            if opp_in:
                leg["opp"] = opp_in.strip().upper()

            results.append(leg)

        if warnings:
            for w in warnings:
                st.warning(w)

        if not results:
            st.warning("No valid legs configured.")
        else:
            st.session_state["last_results"] = results

            st.session_state["model_ran"] = True
            st.success(f"Computed {len(results)} legs.")

            # persist inputs
            st.session_state["p1"], st.session_state["m1"], st.session_state["l1"], st.session_state["opp1"] = p1, m1, l1, t1
            st.session_state["p2"], st.session_state["m2"], st.session_state["l2"], st.session_state["opp2"] = p2, m2, l2, t2
            st.session_state["p3"], st.session_state["m3"], st.session_state["l3"], st.session_state["opp3"] = p3, m3, l3, t3
            st.session_state["p4"], st.session_state["m4"], st.session_state["l4"], st.session_state["opp4"] = p4, m4, l4, t4


    # Inline Results (no forced navigation)
    if st.session_state.get("model_ran") and (st.session_state.get("last_results") or []):
        st.markdown("#### Results Cards")
        res_inline = st.session_state.get("last_results") or []
        cols = st.columns(min(4, len(res_inline)))
        for i, leg in enumerate(res_inline):
            c = cols[i % len(cols)]
            with c:
                st.markdown(f"**{leg['player']} — {leg['market']}**")
                if leg.get("headshot"):
                    st.image(leg["headshot"], use_container_width=True)
                st.write(f"Line: {leg.get('line')}")
                st.write(f"Proj: {safe_round(leg.get('proj'),2,default=None)}")
                st.write(f"P(Over): {safe_round(leg.get('p_over'),3,default=None)}")
                st.write(f"Edge: {safe_round(leg.get('edge'),3,default=None)}")
                tm = leg.get("team") or "?"
                op = leg.get("opp") or "?"
                st.caption(f"Matchup: {tm} vs {op}")
                if leg.get("spread_abs") is not None:
                    st.caption(f"Spread |abs|: {safe_round(leg.get('spread_abs'),1,default=None)}")
                st.caption(f"Ctx: {safe_round(leg.get('ctx_mult'),3,default=None)} | Blowout: {safe_round(leg.get('blowout_prob'),3,default=None)}")
                st.caption(f"Volatility: {leg.get('volatility_label','?')} (CV={safe_round(leg.get('volatility_cv'),2,default=None)})")
                if leg.get("matchup_text"):
                    st.caption(leg["matchup_text"])
                if leg.get("errors"):
                    st.caption("Notes: " + "; ".join(leg["errors"]))

    # Logging block (doesn't throw)
    st.markdown("---")
    
    # Logging (entry-level) — stake + pending + status updates in History tab
    st.markdown("---")
    st.markdown("#### 🗂️ Log Bet Entry")

    res_to_log = st.session_state.get("last_results") or []
    if not res_to_log:
        st.info("Run the model first to log an entry.")
    else:
        entry_type = st.selectbox("Entry Type", options=["Single", "2-Pick", "3-Pick", "4-Pick"], index=min(len(res_to_log)-1,3))
        default_payout = 2.0 if entry_type=="Single" else (payout_2pick if entry_type=="2-Pick" else 5.0)
        payout_mult = st.number_input("Payout Multiplier (x)", min_value=1.0, value=float(default_payout), step=0.1)
        stake = st.number_input("Stake ($)", min_value=0.0, value=float(st.session_state.get("stake_last", 10.0)), step=1.0)
        st.session_state["stake_last"] = float(stake)
        notes = st.text_input("Notes (optional)", value="")

        # Summary metrics
        probs = [float(l.get("p_over", 0.0)) for l in res_to_log if l.get("p_over") is not None]
        edges = [float(l.get("edge", 0.0)) for l in res_to_log if l.get("edge") is not None]
        vols = [float(l.get("volatility_cv", 0.0)) for l in res_to_log if l.get("volatility_cv") is not None]
        avg_prob = float(np.mean(probs)) if probs else None
        avg_edge = float(np.mean(edges)) if edges else None
        avg_vol = float(np.mean(vols)) if vols else None

        if st.button("Log as PENDING"):
            bet_id = f"{int(time.time())}_{abs(hash(user_id))%10000}"
            entry = {
                "bet_id": bet_id,
                "ts_utc": _now_iso(),
                "date_local": datetime.now().strftime("%Y-%m-%d"),
                "user_id": user_id,
                "entry_type": entry_type,
                "stake": float(stake),
                "payout_mult": float(payout_mult),
                "legs_json": json.dumps(res_to_log),
                "status": "PENDING",
                "pnl": 0.0,
                "notes": notes,
                "avg_prob": None if avg_prob is None else float(avg_prob),
                "avg_edge": None if avg_edge is None else float(avg_edge),
                "avg_volatility_cv": None if avg_vol is None else float(avg_vol),
                "clv": None,
                "kelly_frac": None,
            }
            append_history(user_id, entry)
            st.success("Logged to History as PENDING.")


with tabs[1]:
    st.markdown("### Results")
    res = st.session_state.get("last_results") or []
    if not res:
        st.info("Run the model to see projections.")
    else:
        # Player cards
        cols = st.columns(min(4, len(res)))
        for i, leg in enumerate(res):
            c = cols[i % len(cols)]
            with c:
                st.markdown(f"**{leg['player']} — {leg['market']}**")
                if leg.get("headshot"):
                    st.image(leg["headshot"], use_container_width=True)
                st.write(f"Line: {leg.get('line')}")
                st.write(f"Proj (ctx): {None if leg.get('proj') is None else round(leg['proj'],2)}")
                st.write(f"P(Over): {None if leg.get('p_over') is None else round(leg['p_over'],3)}")
                st.write(f"Edge vs 50%: {None if leg.get('edge') is None else round(leg['edge'],3)}")
                tm = leg.get("team") or "?"
                op = leg.get("opp") or "?"
                st.caption(f"Matchup: {tm} vs {op}")
                st.caption(f"Ctx mult: {round(leg.get('context_mult',1.0),3)} | Blowout risk: {round(leg.get('blowout_prob',0.1),3)}")
                if leg.get("errors"):
                    st.caption("Notes: " + "; ".join(leg["errors"]))

        st.markdown("---")
        st.dataframe(pd.DataFrame(res), use_container_width=True)

with tabs[2]:
    st.markdown("### ⚡ Live Scanner — Single-Leg Edges (The Odds API)")
    st.caption("Pulls player prop lines from The Odds API and ranks legs by model win probability (bootstrap).")

    scan_date2 = st.date_input("Scan date", value=date.today(), key="scan_date2")
    scan_date2_iso = scan_date2.isoformat()

    markets_sel = st.multiselect(
        "Markets",
        options=["Points", "Rebounds", "Assists", "PRA", "PR", "PA", "RA", "3PM", "Blocks", "Steals", "Turnovers"],
        default=["Points", "Rebounds", "Assists"],
    )

    min_prob = st.slider("Min model win prob", min_value=0.50, max_value=0.80, value=0.65, step=0.01)
    max_rows = st.slider("Max rows shown", min_value=10, max_value=200, value=60, step=10)

    # sportsbook choices
    book_choices2, book_err2 = get_sportsbook_choices(scan_date2_iso)
    if book_err2:
        st.info(book_err2)
    sportsbook2 = st.selectbox("Sportsbook (Scanner)", options=["all"] + book_choices2, index=0)

    # Fetch offers button
    if st.button("Fetch Live Lines"):
        st.session_state["scanner_offers"] = None
        st.session_state["scanner_reason"] = None

        selected_keys = []
        for m in markets_sel:
            mk = ODDS_MARKETS.get(m) or ODDS_MARKETS_OPTIONAL.get(m)
            if mk:
                selected_keys.append(mk)
        selected_keys = list(dict.fromkeys(selected_keys))
        if not selected_keys:
            st.warning("Select at least one supported market.")
        else:
            evs, err = odds_get_events(scan_date2_iso)
            if err:
                st.error(err)
            elif not evs:
                st.warning("No events returned for that date.")
            else:
                offers = []
                reasons = []
                for ev in evs:
                    eid = ev.get("id")
                    if not eid:
                        continue
                    odds, oerr = odds_get_event_odds(eid, tuple(selected_keys))
                    if oerr or not odds:
                        reasons.append({"event_id": eid, "reason": oerr or "No odds"})
                        continue
                    for m in markets_sel:
                        mk = ODDS_MARKETS.get(m) or ODDS_MARKETS_OPTIONAL.get(m)
                        if not mk:
                            continue
                        bf = sportsbook2 if sportsbook2 != "all" else None
                        parsed, perr = _parse_player_prop_outcomes(odds, mk, book_filter=bf)
                        if perr:
                            reasons.append({"event_id": eid, "reason": perr})
                        offers.extend([{**r, "market": m} for r in parsed])

                if not offers:
                    st.session_state["scanner_reason"] = reasons
                    st.warning("No offers returned. Confirm your plan includes NBA player props and that props are posted for the selected date.")
                else:
                    df = pd.DataFrame(offers)
                    st.session_state["scanner_offers"] = df
                    st.success(f"Fetched {len(df)} player prop outcomes (raw).")

    # Run scan button
    if st.button("Run Live Scan 🚀"):
        df = st.session_state.get("scanner_offers")
        if df is None or df.empty:
            st.warning("Fetch live lines first.")
        else:
            # Build per-player consensus line if sportsbook2 == all or consensus not used
            # For scanning, we treat each distinct (player, market, line) as a candidate
            out_rows = []
            dropped = []
            # Reduce duplicated Over/Under; we scan Over side by default (standard)
            df2 = df.copy()
            df2["side_norm"] = df2["side"].astype(str).str.lower()
            df2 = df2[df2["side_norm"].isin(["over", "o", "over "]) | (df2["side_norm"]=="over") | (df2["side"]=="Over")].copy()

            if df2.empty:
                st.warning("No Over-side outcomes found in offers. (Some books may name sides differently.)")
                df2 = df.copy()

            for _, r in df2.iterrows():
                pname = r.get("player")
                mkt = r.get("market")
                line = r.get("line")
                if not pname or pd.isna(line) or not mkt:
                    dropped.append({"player": pname, "market": mkt, "reason": "Missing player/market/line"})
                    continue
                meta = {
                    "event_id": r.get("event_id"),
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "commence_time": r.get("commence_time"),
                }
                leg = compute_leg_projection(pname, mkt, float(line), meta, n_games=n_games, key_teammate_out=False, user_id=user_id)
                if leg.get("p_over") is None:
                    dropped.append({"player": pname, "market": mkt, "reason": "Projection failed"})
                    continue
                if leg["p_over"] >= float(min_prob):
                    out_rows.append({
                        "player": pname,
                        "market": mkt,
                        "line": float(line),
                        "p_over": float(leg["p_over"]),
                        "proj": leg.get("proj"),
                        "team": leg.get("team"),
                        "opp": leg.get("opp"),
                        "book": r.get("book"),
                        "event_id": r.get("event_id"),
                    })
                else:
                    dropped.append({"player": pname, "market": mkt, "reason": f"p_over<{min_prob:.2f}"})

            out_df = pd.DataFrame(out_rows)
            if out_df.empty:
                st.markdown("#### Scanner Results")
                st.warning("No legs met the threshold (or projections failed).")
            else:
                # Defensive: older cached frames or different build paths could rename columns
                if "p_over" not in out_df.columns and "P_over" in out_df.columns:
                    out_df = out_df.rename(columns={"P_over":"p_over"})
                if "p_over" in out_df.columns:
                    out_df = out_df.sort_values("p_over", ascending=False).head(int(max_rows))
                st.markdown("#### Scanner Results")
                st.dataframe(out_df, use_container_width=True)

            with st.expander("Why legs were excluded (debug)", expanded=False):
                if dropped:
                    st.dataframe(pd.DataFrame(dropped).head(200), use_container_width=True)
                else:
                    st.write("No exclusions.")


with tabs[3]:
    st.markdown("### 🗂️ History")
    h = load_history(user_id)
    if h.empty:
        st.info("No bets logged yet.")
    else:
        # Display summary table
        disp_cols = ["bet_id","date_local","entry_type","stake","payout_mult","status","pnl","avg_prob","avg_edge","avg_volatility_cv","notes"]
        for c in disp_cols:
            if c not in h.columns:
                h[c] = None
        st.dataframe(h[disp_cols].sort_values(["date_local","ts_utc"], ascending=[False, False]), use_container_width=True)

        st.markdown("#### Update Status (HIT / MISS / PUSH)")
        pending = h[h["status"].astype(str).str.upper() == "PENDING"].copy()
        if pending.empty:
            st.info("No PENDING entries.")
        else:
            bet_choice = st.selectbox("Select a pending bet_id", options=pending["bet_id"].astype(str).tolist())
            new_status = st.selectbox("New status", options=["HIT","MISS","PUSH"], index=0)
            if st.button("Apply Status Update"):
                update_history_status(user_id, bet_choice, new_status)
                st.success("Updated. Refreshing...")
                _safe_rerun()

        with st.expander("View legs for a bet_id", expanded=False):
            bet_view = st.selectbox("bet_id", options=h["bet_id"].astype(str).tolist(), key="bet_view_id")
            row = h[h["bet_id"].astype(str) == str(bet_view)]
            if not row.empty:
                legs_json = row.iloc[0].get("legs_json")
                try:
                    legs = json.loads(legs_json) if isinstance(legs_json, str) else legs_json
                except Exception:
                    legs = None
                if legs:
                    st.dataframe(pd.DataFrame(legs), use_container_width=True)
                else:
                    st.write("No legs payload.")


with tabs[4]:
    st.markdown("### 🧪 Calibration (stub)")
    st.info("Calibration UI is reserved for future iterations (CLV learning, drift calibration).")

# Footer
st.caption("© 2025 NBA Prop Quant Engine — Powered by Kamal")
