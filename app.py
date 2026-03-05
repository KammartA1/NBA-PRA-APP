# ============================================================
# NBA PROP QUANT ENGINE v2.1 — Audit-Hardened + Enhanced
# Premium Quant Terminal UI + Hardened Model
# All audit fixes applied: Spearman, adaptive k, minutes filter,
# season-phase rest, skewness gate, no-vig CLV, retry backoff,
# negative-edge guard, calibrator OOD, alert dedup, PASS logging,
# export button, persistent scanner, week-ahead, permanent IDs
# ============================================================
import os, re, math, time, json, difflib, hashlib, logging, threading
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import requests
import streamlit as st

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    playergamelog, scoreboardv2, LeagueDashTeamStats, CommonPlayerInfo
)

# ──────────────────────────────────────────────
# SAFE HELPERS
# ──────────────────────────────────────────────
def safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def safe_round(v, d=2):
    try:
        return round(float(v), d) if v is not None else None
    except Exception:
        return None

def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\.\'\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ODDS_BASE       = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA   = "basketball_nba"
REGION_US       = "us"
MIN_MINUTES_THRESHOLD = 10  # [FIX 3] filter DNP/garbage-time

ODDS_MARKETS = {
    # ── Full-game standard ──────────────────────
    "Points":          "player_points",
    "Rebounds":        "player_rebounds",
    "Assists":         "player_assists",
    "3PM":             "player_threes",
    "PRA":             "player_points_rebounds_assists",
    "PR":              "player_points_rebounds",
    "PA":              "player_points_assists",
    "RA":              "player_rebounds_assists",
    "Blocks":          "player_blocks",
    "Steals":          "player_steals",
    "Turnovers":       "player_turnovers",
    "Stocks":          "player_blocks_steals",
    # ── 1st Half markets ────────────────────────
    # Odds API supports both q1q2 and first_half variants; q1q2 is most common
    "H1 Points":       "player_points_q1q2",
    "H1 Rebounds":     "player_rebounds_q1q2",
    "H1 Assists":      "player_assists_q1q2",
    "H1 3PM":          "player_threes_q1q2",
    "H1 PRA":          "player_points_rebounds_assists_q1q2",
    # ── 2nd Half markets ────────────────────────
    "H2 Points":       "player_points_q3q4",
    "H2 Rebounds":     "player_rebounds_q3q4",
    "H2 Assists":      "player_assists_q3q4",
    # ── 1st Quarter markets ─────────────────────
    "Q1 Points":       "player_points_q1",
    "Q1 Rebounds":     "player_rebounds_q1",
    "Q1 Assists":      "player_assists_q1",
    # ── Alternate lines ─────────────────────────
    "Alt Points":      "player_points_alternate",
    "Alt Rebounds":    "player_rebounds_alternate",
    "Alt Assists":     "player_assists_alternate",
    "Alt 3PM":         "player_threes_alternate",
    # ── Fantasy score ────────────────────────────
    "Fantasy Score":   "player_fantasy_points",
    # ── Combo / special ─────────────────────────
    "Double Double":   "player_double_double",
    "Triple Double":   "player_triple_double",
    "First Basket":    "player_first_basket",
    # ── Shooting volume (high-probability) ──────
    "FGM":             "player_field_goals_made",
    "FGA":             "player_field_goals_attempted",
    "3PA":             "player_three_point_field_goals_attempted",
    "FTM":             "player_free_throws_made",
    "FTA":             "player_free_throws_attempted",
}

# Markets that require batching separately (not all books offer these)
SPECIALTY_MARKET_KEYS = {
    # Half-game markets (only DK/FD/etc offer these)
    "player_points_q1q2", "player_rebounds_q1q2",
    "player_assists_q1q2", "player_threes_q1q2",
    "player_points_rebounds_assists_q1q2",
    "player_points_q3q4", "player_rebounds_q3q4", "player_assists_q3q4",
    # 1Q markets
    "player_points_q1", "player_rebounds_q1", "player_assists_q1",
    # Alt lines
    "player_points_alternate", "player_rebounds_alternate",
    "player_assists_alternate", "player_threes_alternate",
    # Fantasy
    "player_fantasy_points",
    # Shooting volume
    "player_field_goals_made", "player_field_goals_attempted",
    "player_three_point_field_goals_attempted",
    "player_free_throws_made", "player_free_throws_attempted",
}

STAT_FIELDS = {
    "Points":          "PTS",
    "Rebounds":        "REB",
    "Assists":         "AST",
    "3PM":             "FG3M",
    "PRA":             ("PTS","REB","AST"),
    "PR":              ("PTS","REB"),
    "PA":              ("PTS","AST"),
    "RA":              ("REB","AST"),
    "Blocks":          "BLK",
    "Steals":          "STL",
    "Turnovers":       "TOV",
    "Stocks":          ("BLK","STL"),
    # Half markets map to full-game fields (adjusted via HALF_FACTOR)
    "H1 Points":       "PTS",
    "H1 Rebounds":     "REB",
    "H1 Assists":      "AST",
    "H1 3PM":          "FG3M",
    "H1 PRA":          ("PTS","REB","AST"),
    "H2 Points":       "PTS",
    "H2 Rebounds":     "REB",
    "H2 Assists":      "AST",
    # 1Q markets map to full-game fields (adjusted via Q1_FACTOR)
    "Q1 Points":       "PTS",
    "Q1 Rebounds":     "REB",
    "Q1 Assists":      "AST",
    # Alt lines use same fields as base
    "Alt Points":      "PTS",
    "Alt Rebounds":    "REB",
    "Alt Assists":     "AST",
    "Alt 3PM":         "FG3M",
    # Fantasy score: PTS + 1.2*REB + 1.5*AST + 3*(BLK+STL) - TOV (DK-style)
    "Fantasy Score":   ("PTS","REB","AST","BLK","STL","TOV"),
    # Combo / special
    "Double Double":   ("PTS","REB","AST","BLK","STL"),
    "Triple Double":   ("PTS","REB","AST","BLK","STL"),
    "First Basket":    "PTS",
    # Shooting volume
    "FGM":             "FGM",
    "FGA":             "FGA",
    "3PA":             "FG3A",
    "FTM":             "FTM",
    "FTA":             "FTA",
}

# Half-game projection scale factors
HALF_FACTOR = {
    "H1 Points": 0.52, "H1 Rebounds": 0.52, "H1 Assists": 0.52,
    "H1 3PM": 0.52, "H1 PRA": 0.52, "H1 PR": 0.52, "H1 PA": 0.52,
    "H1 FTM": 0.52, "H1 FTA": 0.52, "H1 FGM": 0.52, "H1 FGA": 0.52,
    "H2 Points": 0.48, "H2 Rebounds": 0.48, "H2 Assists": 0.48,
    "H2 3PM": 0.48, "H2 PRA": 0.48, "H2 PR": 0.48, "H2 PA": 0.48,
    "H2 FTM": 0.48, "H2 FTA": 0.48, "H2 FGM": 0.48, "H2 FGA": 0.48,
    # 1Q markets: ~25% of full-game (first quarter only)
    "Q1 Points": 0.25, "Q1 Rebounds": 0.24, "Q1 Assists": 0.24,
    "Q1 3PM": 0.24, "Q1 PRA": 0.25, "Q1 FTM": 0.23, "Q1 FGA": 0.25,
}

# Alt markets — same engine, different API key
ALT_MARKETS = {"Alt Points","Alt Rebounds","Alt Assists","Alt 3PM"}

# Fantasy score markets need custom stat computation
FANTASY_MARKETS = {"Fantasy Score"}

# DD/TD markets — probability from game log, not bootstrap
DD_TD_MARKETS = {"Double Double","Triple Double"}

BOOK_SHARPNESS = {
    "pinnacle":0.99,"circa":0.95,"bookmaker":0.90,"betcris":0.85,
    "draftkings":0.70,"fanduel":0.70,"betmgm":0.65,"caesars":0.65,
    "betrivers":0.60,"pointsbetus":0.55,
    "betonlineag":0.45,"bovada":0.40,"mybookieag":0.30,
}

def book_sharpness(k):
    return float(BOOK_SHARPNESS.get((k or "").strip().lower(), 0.55))

POSITIONAL_PRIORS = {
    "Guard": {"Points":16.5,"Rebounds":3.4,"Assists":5.8,"3PM":2.1,
              "PRA":25.7,"PR":19.9,"PA":22.3,"RA":9.2,"Blocks":0.4,"Steals":1.2,"Turnovers":2.2,
              "Q1 Points":4.1,"Q1 Rebounds":0.9,"Q1 Assists":1.5,"Fantasy Score":31.2,
              "FGM":5.8,"FGA":13.5,"3PA":6.2,"FTM":3.2,"FTA":3.8},
    "Wing":  {"Points":14.8,"Rebounds":5.9,"Assists":2.9,"3PM":1.6,
              "PRA":23.6,"PR":20.7,"PA":17.7,"RA":8.8,"Blocks":0.8,"Steals":1.0,"Turnovers":1.7,
              "Q1 Points":3.7,"Q1 Rebounds":1.5,"Q1 Assists":0.7,"Fantasy Score":27.4,
              "FGM":5.4,"FGA":12.0,"3PA":4.5,"FTM":2.6,"FTA":3.2},
    "Big":   {"Points":13.2,"Rebounds":8.8,"Assists":2.1,"3PM":0.5,
              "PRA":24.1,"PR":22.0,"PA":15.3,"RA":10.9,"Blocks":1.4,"Steals":0.7,"Turnovers":2.0,
              "Q1 Points":3.3,"Q1 Rebounds":2.2,"Q1 Assists":0.5,"Fantasy Score":30.5,
              "FGM":5.0,"FGA":10.5,"3PA":1.4,"FTM":3.0,"FTA":4.0},
    "Unknown":{"Points":14.8,"Rebounds":5.5,"Assists":3.5,"3PM":1.4,
              "PRA":23.8,"PR":20.3,"PA":18.3,"RA":9.0,"Blocks":0.8,"Steals":0.9,"Turnovers":1.9,
              "Q1 Points":3.7,"Q1 Rebounds":1.5,"Q1 Assists":0.9,"Fantasy Score":29.7,
              "FGM":5.4,"FGA":12.0,"3PA":4.0,"FTM":2.9,"FTA":3.6},
}

REST_MULTIPLIERS = {0: 0.93, 1: 0.97, 2: 1.00, 3: 1.01, 4: 1.02}

# Exponential recency decay per stat (assists autocorrelate longer than blocks)
LAMBDA_DECAY_BY_STAT = {
    "Points": 0.88, "Rebounds": 0.85, "Assists": 0.88,
    "3PM": 0.84, "PRA": 0.87, "PR": 0.86, "PA": 0.87, "RA": 0.85,
    "Blocks": 0.83, "Steals": 0.84, "Turnovers": 0.86, "Stocks": 0.83,
    "H1 Points": 0.88, "H1 Rebounds": 0.85, "H1 Assists": 0.88,
    "H2 Points": 0.88, "H2 Rebounds": 0.85, "H2 Assists": 0.88,
    "Q1 Points": 0.87, "Q1 Rebounds": 0.84, "Q1 Assists": 0.87,
    "Alt Points": 0.88, "Alt Rebounds": 0.85, "Alt Assists": 0.88, "Alt 3PM": 0.84,
    "Fantasy Score": 0.88,
    "default": 0.88,
}

# Persistent file paths
OPENING_LINES_PATH = "opening_lines.json"
WATCHLIST_PATH_TPL  = "watchlist_{uid}.json"

# ──────────────────────────────────────────────
# PLAYER POSITION CACHE
# ──────────────────────────────────────────────
_POSITION_CACHE = {}  # {name_lower: pos_str}
_PID_POSITION_MAP = {}  # {player_id: pos_str} — populated by bulk fetch

@st.cache_data(ttl=60*60*6, show_spinner=False)
def _bulk_player_position_map():
    """Fetch ALL active player positions in one LeagueDashPlayerStats call.
    Returns {player_id: position_string}.  Falls back to {} on error.
    """
    try:
        from nba_api.stats.endpoints import LeagueDashPlayerStats
        df = LeagueDashPlayerStats(
            season=get_season_string(),
            per_mode_simple="PerGame",
            measure_type_detailed_defense="Base",
        ).get_data_frames()[0]
        if df.empty or "PLAYER_ID" not in df.columns:
            return {}
        # PLAYER_POSITION column if present
        pos_col = next((c for c in df.columns if "POSITION" in c.upper()), None)
        if pos_col:
            return {int(r["PLAYER_ID"]): str(r[pos_col]) for _, r in df.iterrows()}
        return {}
    except Exception:
        return {}

def _ensure_pid_position_map():
    global _PID_POSITION_MAP
    if not _PID_POSITION_MAP:
        _PID_POSITION_MAP = _bulk_player_position_map()

def get_player_position(name):
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key in _POSITION_CACHE:
        return _POSITION_CACHE[key]
    # Try bulk map first (fast — no extra API call)
    try:
        matches = nba_players.find_players_by_full_name(name)
    except Exception:
        matches = []
    pos = ""
    if matches:
        pid = matches[0].get("id")
        if pid:
            _ensure_pid_position_map()
            if int(pid) in _PID_POSITION_MAP:
                pos = _PID_POSITION_MAP[int(pid)]
            else:
                # Fallback: single CommonPlayerInfo call (slow, only if not in bulk map)
                try:
                    info = CommonPlayerInfo(player_id=pid).get_data_frames()[0]
                    raw = str(info.get("POSITION", "") or info.get("POSITION_SHORT","") or "")
                    pos = raw or ""
                except Exception:
                    pos = ""
    _POSITION_CACHE[key] = pos
    return pos

def get_position_bucket(pos):
    if not pos:
        return "Unknown"
    p = str(pos).upper()
    if p.startswith("G"): return "Guard"
    if p.startswith("F"): return "Wing"
    if p.startswith("C"): return "Big"
    if "G" in p and "F" in p: return "Wing"
    if "F" in p and "C" in p: return "Big"
    return "Unknown"

# ──────────────────────────────────────────────
# SEASON HELPER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)
def get_season_string(today=None):
    d = today or date.today()
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start+1)%100:02d}"

# ──────────────────────────────────────────────
# BULK GAME LOG — one API call for ALL players
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*30, show_spinner=False)
def _fetch_bulk_gamelogs():
    """LeagueGameLog: ONE call returns every player's game log for the season.
    Replaces ~200 individual PlayerGameLog calls for a full-slate scan.
    Returns a DataFrame sorted newest-first per player, or None on failure.
    """
    try:
        from nba_api.stats.endpoints import LeagueGameLog
        df = LeagueGameLog(
            player_or_team_abbreviation="P",
            season=get_season_string(),
            season_type_all_star="Regular Season",
            timeout=45,
        ).get_data_frames()[0]
        if df.empty:
            return None
        # Normalize player ID column (LeagueGameLog uses PLAYER_ID)
        if "PLAYER_ID" not in df.columns and "Player_ID" in df.columns:
            df = df.rename(columns={"Player_ID": "PLAYER_ID"})
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        # Sort newest → oldest within each player
        df = df.sort_values(["PLAYER_ID", "GAME_DATE"], ascending=[True, False])
        # LeagueGameLog returns MIN as float; convert to match PlayerGameLog "MM:SS" format
        if "MIN" in df.columns:
            df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
        return df
    except Exception:
        return None

# ──────────────────────────────────────────────
# GAME LOG FETCHER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_player_gamelog(player_id, max_games=15):
    """Per-player game log. Tries the bulk cache first (instant), falls back to individual API call."""
    # ── Fast path: bulk dataframe already in cache ──────────────
    bulk = _fetch_bulk_gamelogs()
    if bulk is not None:
        pid = int(player_id)
        player_df = bulk[bulk["PLAYER_ID"] == pid].head(int(max_games)).copy()
        if not player_df.empty:
            # Convert GAME_DATE back to string so downstream code (pd.to_datetime) still works
            player_df["GAME_DATE"] = player_df["GAME_DATE"].dt.strftime("%b %d, %Y")
            return player_df, []

    # ── Slow path: individual PlayerGameLog call (10s timeout) ──
    errs = []
    season_str = get_season_string()
    for params in [{"season_nullable": season_str}, {"season": season_str}, {}]:
        try:
            gl = playergamelog.PlayerGameLog(player_id=player_id, timeout=10, **params)
            df = gl.get_data_frames()[0]
            if not df.empty:
                return df.head(int(max_games)).copy(), errs
            errs.append(f"Empty log with params {params}")
        except TypeError as te:
            errs.append(f"TypeError {params}: {te}")
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")
    return pd.DataFrame(), errs

# ──────────────────────────────────────────────
# REAL HALF-GAME BOXSCORE FETCHER
# Per-period stats via BoxScoreTraditionalV2.
# start_period/end_period parameters give cumulative splits
# (H1=1-2, H2=3-4, Q1=1-1) without extra joins.
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*12, show_spinner=False)
def _fetch_boxscore_halfgame(game_id, player_id_int, start_period, end_period):
    """Return {PTS, REB, AST, ...} for a player over a specific period range, or None."""
    try:
        from nba_api.stats.endpoints import BoxScoreTraditionalV2
        bx = BoxScoreTraditionalV2(
            game_id=str(game_id),
            start_period=int(start_period),
            end_period=int(end_period),
            timeout=20,
        )
        pstats = bx.get_data_frames()[0]
        if pstats.empty:
            return None
        pid_col = next((c for c in ["PLAYER_ID", "Player_ID"] if c in pstats.columns), None)
        if not pid_col:
            return None
        row = pstats[pstats[pid_col] == int(player_id_int)]
        if row.empty:
            return None
        row = row.iloc[0]
        result = {}
        for col in ["PTS","REB","AST","FG3M","FGM","FGA","FG3A","FTM","FTA","BLK","STL","TOV","OREB","DREB"]:
            if col in row.index:
                result[col] = safe_float(row[col], default=0.0)
        return result if result else None
    except Exception:
        return None

def fetch_player_halfgame_log(player_id, game_log_df, market_name, n_games=10):
    """
    Build a real H1/H2/Q1 stat series from per-period boxscores.
    Returns pd.Series (newest-first) if >=3 games fetched, else None (falls back to scaled full-game).
    """
    if game_log_df is None or game_log_df.empty or not player_id:
        return None
    # Period boundaries
    if market_name.startswith("H1"):
        start_p, end_p = 1, 2
    elif market_name.startswith("H2"):
        start_p, end_p = 3, 4
    elif market_name.startswith("Q1"):
        start_p, end_p = 1, 1
    else:
        return None
    # Get game IDs from log
    game_id_col = next((c for c in ["GAME_ID", "Game_ID"] if c in game_log_df.columns), None)
    if not game_id_col:
        return None
    game_ids = game_log_df.head(n_games)[game_id_col].dropna().astype(str).tolist()
    if not game_ids:
        return None
    # Fetch in parallel (3 workers to stay under rate limits)
    stats_list = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_fetch_boxscore_halfgame, gid, int(player_id), start_p, end_p)
                   for gid in game_ids]
        for fut in futures:
            try:
                result = fut.result(timeout=25)
                if result:
                    stats_list.append(result)
            except Exception:
                pass
    if len(stats_list) < 3:
        return None   # Not enough half-game data; caller falls back to scaled full-game
    stats_df = pd.DataFrame(stats_list)
    # Resolve base market (strip H1/H2/Q1 prefix)
    base_mkt = market_name.replace("H1 ","").replace("H2 ","").replace("Q1 ","")
    stat_field = STAT_FIELDS.get(base_mkt)
    if stat_field is None:
        return None
    if isinstance(stat_field, tuple):
        s = pd.Series(0.0, index=range(len(stats_df)))
        for col in stat_field:
            if col in stats_df.columns:
                s = s + pd.to_numeric(stats_df[col], errors="coerce").fillna(0)
        return s.reset_index(drop=True)
    if stat_field in stats_df.columns:
        return pd.to_numeric(stats_df[stat_field], errors="coerce").reset_index(drop=True)
    return None

# ──────────────────────────────────────────────
# REST / B2B FACTOR  [FIX 4: season-phase scaling]
# ──────────────────────────────────────────────
def compute_rest_factor(game_log_df, game_date):
    if game_log_df is None or game_log_df.empty:
        return 1.00, 2
    try:
        dates_raw = pd.to_datetime(game_log_df["GAME_DATE"], errors="coerce").dropna()
        if dates_raw.empty:
            return 1.00, 2
        last_game = dates_raw.max().date()
        rest = (game_date - last_game).days - 1
        rest = max(0, min(rest, 4))
        base_mult = REST_MULTIPLIERS.get(rest, 1.02)
        # [FIX 4] Season-phase fatigue: B2B hits harder late in season
        try:
            season_year = int(get_season_string()[:4])
            season_start = date(season_year, 10, 1)
            days_in = max(0, (game_date - season_start).days)
            games_approx = min(82, days_in // 2)
            # Coefficient 0.015: late-season B2B should dent output ~1-2%
            # (0.0008 was negligible at ~0.08%; 0.015 gives realistic 1.5% max)
            fatigue = 1.0 - 0.015 * (games_approx / 82.0) * max(0, 2 - rest)
            base_mult *= fatigue
        except Exception:
            pass
        return float(base_mult), int(rest)
    except Exception:
        return 1.00, 2

# ──────────────────────────────────────────────
# HOME / AWAY SPLIT
# ──────────────────────────────────────────────
def compute_home_away_factor(game_log_df, market, is_home):
    if game_log_df is None or game_log_df.empty or is_home is None:
        return 1.00
    try:
        df = game_log_df.copy()
        df["_home"] = df["MATCHUP"].str.contains("vs", case=False, na=False)
        stat_col = STAT_FIELDS.get(market)
        if stat_col is None:
            return 1.00
        if isinstance(stat_col, tuple):
            df["_stat"] = sum(pd.to_numeric(df.get(c), errors="coerce").fillna(0) for c in stat_col)
        else:
            df["_stat"] = pd.to_numeric(df.get(stat_col), errors="coerce")
        home_avg = df[df["_home"]]["_stat"].mean()
        away_avg = df[~df["_home"]]["_stat"].mean()
        if pd.isna(home_avg) or pd.isna(away_avg) or away_avg == 0 or home_avg == 0:
            return 1.00
        ratio = (home_avg / away_avg) if is_home else (away_avg / home_avg)
        return float(np.clip(ratio, 0.88, 1.12))
    except Exception:
        return 1.00

# ──────────────────────────────────────────────
# BAYESIAN SHRINKAGE  [FIX 2: adaptive k]
# ──────────────────────────────────────────────
def bayesian_shrink(observed_mu, n_obs, market, position_bucket):
    prior = POSITIONAL_PRIORS.get(position_bucket, POSITIONAL_PRIORS["Unknown"]).get(market)
    if prior is None or observed_mu is None:
        return observed_mu
    # [FIX 2] Adaptive k: shrinks less for experienced players
    k = max(2.0, 8.0 / (1.0 + math.log1p(max(n_obs, 1) / 5.0)))
    w_prior = k / (k + max(n_obs, 1))
    w_obs   = 1.0 - w_prior
    return float(w_prior * prior + w_obs * observed_mu)

# ──────────────────────────────────────────────
# STAT SERIES COMPUTATION
# ──────────────────────────────────────────────
def compute_stat_from_gamelog(df, market):
    # [UPGRADE] Fantasy Score: weighted DraftKings formula
    if market == "Fantasy Score":
        try:
            pts = pd.to_numeric(df.get("PTS"), errors="coerce").fillna(0)
            reb = pd.to_numeric(df.get("REB"), errors="coerce").fillna(0)
            ast = pd.to_numeric(df.get("AST"), errors="coerce").fillna(0)
            blk = pd.to_numeric(df.get("BLK"), errors="coerce").fillna(0)
            stl = pd.to_numeric(df.get("STL"), errors="coerce").fillna(0)
            tov = pd.to_numeric(df.get("TOV"), errors="coerce").fillna(0)
            return pts + 1.2*reb + 1.5*ast + 3.0*(blk+stl) - tov
        except Exception:
            return pd.Series([], dtype=float)
    f = STAT_FIELDS.get(market)
    if f is None:
        return pd.Series([], dtype=float)
    if isinstance(f, tuple):
        s = pd.Series(0.0, index=df.index)
        for col in f:
            s = s + pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
        return s
    return pd.to_numeric(df.get(f), errors="coerce")

# ──────────────────────────────────────────────
# VOLATILITY ENGINE  [FIX 5: skewness helper]
# ──────────────────────────────────────────────
def compute_volatility(series):
    try:
        arr = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
    except Exception:
        return None, None
    if arr.size < 2:
        return None, None
    mean = arr.mean()
    if mean == 0:
        return None, None
    cv = float(arr.std(ddof=1) / mean)
    label = "Low" if cv < 0.15 else ("Moderate" if cv < 0.30 else "High")
    return cv, label

# [FIX 5/13] Skewness helper
def compute_skewness(series):
    try:
        arr = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
        if arr.size < 4:
            return None
        n = len(arr)
        m = arr.mean()
        s = arr.std(ddof=1)
        if s == 0:
            return 0.0
        return float((n / ((n-1)*(n-2))) * np.sum(((arr - m) / s)**3))
    except Exception:
        return None

# ──────────────────────────────────────────────
# [UPGRADE 3] OPPONENT-SPECIFIC HISTORICAL FACTOR
# ──────────────────────────────────────────────
def compute_opp_specific_factor(game_log_df, opp_abbr, market, n_min=3):
    """Return multiplier based on player's recorded performance vs this specific opponent."""
    if game_log_df is None or game_log_df.empty or not opp_abbr:
        return 1.0, 0
    try:
        df = game_log_df.copy()
        opp_upper = str(opp_abbr).upper()
        mask = df["MATCHUP"].str.upper().str.contains(opp_upper, na=False)
        opp_df = df[mask]; rest_df = df[~mask]
        n_opp = len(opp_df)
        if n_opp < n_min or len(rest_df) < n_min:
            return 1.0, n_opp
        stat_col = STAT_FIELDS.get(market)
        if stat_col is None:
            return 1.0, n_opp
        if isinstance(stat_col, tuple):
            def _sum_cols(d):
                return sum(pd.to_numeric(d.get(c), errors="coerce").fillna(0) for c in stat_col).mean()
            opp_avg = _sum_cols(opp_df); rest_avg = _sum_cols(rest_df)
        else:
            opp_avg = pd.to_numeric(opp_df.get(stat_col), errors="coerce").mean()
            rest_avg = pd.to_numeric(rest_df.get(stat_col), errors="coerce").mean()
        if pd.isna(opp_avg) or pd.isna(rest_avg) or rest_avg == 0:
            return 1.0, n_opp
        ratio = opp_avg / rest_avg
        # 40% weight on opponent-specific (dampen for small sample)
        weight = min(0.40, n_opp * 0.10)
        factor = 1.0 + weight * (ratio - 1.0)
        return float(np.clip(factor, 0.88, 1.12)), n_opp
    except Exception:
        return 1.0, 0

# ──────────────────────────────────────────────
# [UPGRADE 5] PLAYER REGIME — HOT / COLD / AVERAGE
# ──────────────────────────────────────────────
def compute_player_regime_hot_cold(stat_series, n_recent=5):
    """Tag player as Hot / Cold / Average based on last N-game z-score vs season."""
    try:
        arr = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
        if len(arr) < n_recent + 3:
            return "Average", 0.0
        season_mu = arr.mean(); season_sigma = arr.std(ddof=1)
        if season_sigma < 1e-6:
            return "Average", 0.0
        recent_mu = arr[:n_recent].mean()  # array is newest-first
        z = (recent_mu - season_mu) / season_sigma
        if z > 0.8:  return "Hot",  float(z)
        if z < -0.8: return "Cold", float(z)
        return "Average", float(z)
    except Exception:
        return "Average", 0.0

# ──────────────────────────────────────────────
# [UPGRADE 2] PROJECTED MINUTES
# ──────────────────────────────────────────────
def compute_projected_minutes(game_log_df, n_games=10):
    """Return rolling average of minutes played (DNP-aware) and a DNP-risk flag."""
    if game_log_df is None or game_log_df.empty or "MIN" not in game_log_df.columns:
        return None, False
    try:
        df = game_log_df.head(n_games).copy()
        mins = df["MIN"].apply(lambda v:
            float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
            else safe_float(v, default=0.0))
        active = mins[mins >= 5]
        if active.empty:
            return None, True
        avg_min = float(active.mean())
        # DNP risk: player was actually held out (0-4 min) in 50%+ of recent games,
        # or averaged <8 min active (deep rotation / G-League shuttle).
        # 19 min/game is NOT dnp risk — that's a normal rotation player.
        true_dnps = (mins <= 4).sum()
        dnp_risk = avg_min < 8.0 or (true_dnps >= len(df) * 0.50)
        return avg_min, bool(dnp_risk)
    except Exception:
        return None, False

def volatility_penalty_factor(cv):
    if cv is None: return 0.0
    v = float(cv)
    if v <= 0.20: return 1.00
    if v <= 0.25: return 0.85
    if v <= 0.30: return 0.65
    if v <= 0.35: return 0.45
    return 0.0

# [FIX 5] Skewness-adjusted volatility gate
def passes_volatility_gate(cv, ev_raw, skew=None, bet_type="Over"):
    if cv is None:
        return False, "no stat history (CV unavailable)"
    v = float(cv)
    if v > 0.35:
        return False, "CV>0.35 (too volatile)"
    if v > 0.25 and (ev_raw is None or float(ev_raw) < 0.06):
        return False, "CV>0.25 needs EV>=6%"
    # [FIX 5] Skewness-adjusted threshold
    if skew is not None and v > 0.20:
        is_over = "over" in str(bet_type).lower()
        # Negative skew + Over bet = tail risk of low games
        if float(skew) < -0.5 and is_over:
            tightened = 0.30
            if v > tightened and (ev_raw is None or float(ev_raw) < 0.08):
                return False, f"CV>{tightened:.2f} (neg-skew+Over tightened, needs EV>=8%)"
        # Positive skew + Under bet = tail risk of blow-up games
        elif float(skew) > 0.5 and not is_over:
            tightened = 0.30
            if v > tightened and (ev_raw is None or float(ev_raw) < 0.08):
                return False, f"CV>{tightened:.2f} (pos-skew+Under tightened, needs EV>=8%)"
    return True, ""

# ──────────────────────────────────────────────
# BOOTSTRAP WITH PER-PLAYER NOISE
# ──────────────────────────────────────────────
def bootstrap_prob_over(stat_series, line, n_sims=8000, cv_override=None, market="default"):
    x = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if x.size < 4:
        mu = float(np.nanmean(x)) if x.size else None
        sigma = float(np.nanstd(x, ddof=1)) if x.size > 1 else None
        return None, mu, sigma
    # [UPGRADE 4] Principled exponential decay (stat-specific λ, not arbitrary linear)
    if x.size >= 6:
        lam = LAMBDA_DECAY_BY_STAT.get(market, LAMBDA_DECAY_BY_STAT["default"])
        w = np.array([lam ** i for i in range(x.size)], dtype=float)
        w = w / w.sum()
    else:
        w = None
    rng = np.random.default_rng(42)
    sims = rng.choice(x, size=int(n_sims), replace=True, p=w)
    cv = cv_override or (float(x.std(ddof=1) / x.mean()) if x.mean() != 0 else 0.20)
    noise_scale = max(0.05, min(cv * 0.40, 0.25))
    noise = rng.normal(0, float(x.std(ddof=1) * noise_scale), int(n_sims))
    sims_noisy = np.clip(sims + noise, 0, None)
    p_over = float((sims_noisy > float(line)).mean())
    mu_w = float(np.average(x, weights=w) if w is not None else x.mean())
    sigma_w = float(np.sqrt(np.average((x - mu_w)**2, weights=w)) if w is not None else x.std(ddof=1))
    return float(np.clip(p_over, 1e-4, 1-1e-4)), mu_w, max(1e-9, sigma_w)

# ──────────────────────────────────────────────
# EMPIRICAL CORRELATION  [FIX 1: Spearman]
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)
def empirical_leg_correlation(pid1, pid2, mkt1, mkt2, n_games=20):
    try:
        gl1, _ = fetch_player_gamelog(pid1, max_games=n_games)
        gl2, _ = fetch_player_gamelog(pid2, max_games=n_games)
        if gl1.empty or gl2.empty:
            return None
        s1 = compute_stat_from_gamelog(gl1, mkt1).rename("s1")
        s2 = compute_stat_from_gamelog(gl2, mkt2).rename("s2")
        df1 = pd.concat([gl1["GAME_DATE"].reset_index(drop=True), s1.reset_index(drop=True)], axis=1)
        df2 = pd.concat([gl2["GAME_DATE"].reset_index(drop=True), s2.reset_index(drop=True)], axis=1)
        merged = df1.merge(df2, on="GAME_DATE", how="inner")
        if len(merged) < 6:
            return None
        # [FIX 1] Spearman rank correlation for count data
        corr = float(merged["s1"].corr(merged["s2"], method="spearman"))
        return float(np.clip(corr, -0.50, 0.70)) if not np.isnan(corr) else None
    except Exception:
        return None

def estimate_player_correlation(leg1, leg2):
    pid1 = leg1.get("player_id")
    pid2 = leg2.get("player_id")
    # Same player + same market = identical bet (perfect correlation)
    if pid1 and pid2 and int(pid1) == int(pid2) and leg1.get("market") == leg2.get("market"):
        return 1.0
    if pid1 and pid2:
        emp = empirical_leg_correlation(
            int(pid1), int(pid2), leg1.get("market","Points"), leg2.get("market","Points")
        )
        if emp is not None:
            return float(emp)
    corr = 0.0
    if leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"]:
        corr += 0.15
    m1, m2 = leg1.get("market"), leg2.get("market")
    if m1 == m2: corr += 0.10
    if set([m1,m2]) == {"Points","PRA"}: corr += 0.14
    if m1 in ["Rebounds","RA"] and m2 in ["Rebounds","RA"]: corr += 0.06
    if m1 in ["Assists","RA"] and m2 in ["Assists","RA"]: corr += 0.05
    ctx1, ctx2 = float(leg1.get("context_mult",1.0)), float(leg2.get("context_mult",1.0))
    if ctx1>1.03 and ctx2>1.03: corr += 0.04
    if ctx1<0.97 and ctx2<0.97: corr += 0.03
    if (ctx1>1.03 and ctx2<0.97) or (ctx1<0.97 and ctx2>1.03): corr -= 0.05
    return float(np.clip(corr, -0.25, 0.45))

# ──────────────────────────────────────────────
# LINE MOVEMENT ALERT  [FIX 10: dedup]
# ──────────────────────────────────────────────
def get_line_movement_signal(player_norm, market_key, current_line, side="Over"):
    store_key = f"open_line_{player_norm}_{market_key}_{side}"
    opening = st.session_state.get(store_key)
    if opening is None:
        st.session_state[store_key] = float(current_line)
        return {"direction": "-", "pips": 0.0, "steam": False, "fade": False, "msg": "Opening line recorded"}
    delta = float(current_line) - float(opening)
    is_over = "over" in str(side).lower()
    steam = (delta > 0.5 and is_over) or (delta < -0.5 and not is_over)
    fade  = (delta < -0.5 and is_over) or (delta > 0.5 and not is_over)
    msg = ""
    if abs(delta) >= 0.5:
        direction = "UP" if delta > 0 else "DOWN"
        msg = f"Line moved {direction} {abs(delta):.1f} pts from open ({opening:.1f} -> {current_line:.1f})"
        # [FIX 10] Alert deduplication
        alert_hash = f"{player_norm}_{market_key}_{side}_{direction}_{round(abs(delta),1)}"
        issued = st.session_state.get("_issued_mv_alerts", set())
        if alert_hash not in issued:
            if steam: msg += " STEAM (confirms your side)"
            if fade:  msg += " FADE (sharps vs your side)"
            issued.add(alert_hash)
            st.session_state["_issued_mv_alerts"] = issued
        else:
            msg = ""  # suppress duplicate alert
    return {
        "direction": "UP" if delta > 0 else ("DOWN" if delta < 0 else "FLAT"),
        "pips": float(delta),
        "steam": steam,
        "fade": fade,
        "msg": msg,
        "opening": float(opening),
    }

# ──────────────────────────────────────────────
# REGIME CLASSIFIER
# ──────────────────────────────────────────────
def classify_regime(cv, blowout_prob, ctx_mult):
    try: v = float(cv) if cv is not None else None
    except: v = None
    try: b = float(blowout_prob) if blowout_prob is not None else 0.10
    except: b = 0.10
    try: c = float(ctx_mult) if ctx_mult is not None else 1.0
    except: c = 1.0
    v_score = 0.0 if v is None else float(np.clip((v-0.15)/0.25, 0.0, 1.0))
    b_score = float(np.clip((b-0.08)/0.20, 0.0, 1.0))
    c_score = float(np.clip(abs(c-1.0)/0.20, 0.0, 1.0))
    score = float(np.clip(0.55*v_score + 0.30*b_score + 0.15*c_score, 0.0, 1.0))
    if score >= 0.65: return "Chaotic", score
    if score >= 0.40: return "Mixed", score
    return "Stable", score

# ──────────────────────────────────────────────
# MARKET PRICING  [FIX 6: remove_vig] [FIX 8: neg-edge guard]
# ──────────────────────────────────────────────
def implied_prob_from_decimal(price):
    if price is None: return None
    try: return float(np.clip(1.0/float(price), 1e-6, 1.0-1e-6))
    except: return None

def ev_per_dollar(p_win, price):
    if p_win is None or price is None: return None
    try:
        p, o = float(p_win), float(price)
        if o <= 1.0: return None
        return float(p*(o-1.0) - (1.0-p))
    except: return None

# [FIX 6] No-vig price calculation
def remove_vig(price_over, price_under):
    """Return no-vig (fair) decimal prices for a two-sided market."""
    try:
        ip_o = 1.0 / max(float(price_over), 1.0001)
        ip_u = 1.0 / max(float(price_under), 1.0001)
        overround = ip_o + ip_u
        if overround <= 0:
            return float(price_over), float(price_under)
        return float(1.0 / (ip_o / overround)), float(1.0 / (ip_u / overround))
    except Exception:
        return float(price_over), float(price_under)

def classify_edge(ev):
    if ev is None: return None
    e = float(ev)
    if e <= 0.0:   return "No Edge"
    if e < 0.04:   return "Lean Edge"
    if e < 0.08:   return "Solid Edge"
    return "Strong Edge"

def kelly_fraction(p, price):
    try:
        p, o = float(p), float(price)
        if o<=1.0 or p<=0 or p>=1: return 0.0
        b=o-1.0; q=1.0-p
        return float(max(0.0, (b*p-q)/b))
    except: return 0.0

# [FIX 8] Hard negative-edge guard
def recommended_stake(bankroll, p, price_decimal, frac_kelly, cap_frac=0.05):
    try: br=float(bankroll)
    except: br=0.0
    if br<=0 or p is None or price_decimal is None: return 0.0, 0.0, "bankroll<=0"
    k = kelly_fraction(float(p), float(price_decimal))
    if k <= 0:
        return 0.0, 0.0, "negative edge - hard blocked"
    f = max(0.0, min(1.0, float(frac_kelly))) * k
    f = min(f, float(cap_frac))
    stake = br * f
    if stake <= 0: return 0.0, 0.0, "kelly<=0"
    return float(stake), float(f), "ok"

# ──────────────────────────────────────────────
# TEAM CONTEXT
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*3, show_spinner=False)
def get_team_context():
    try:
        ss = get_season_string()
        adv = LeagueDashTeamStats(season=ss, measure_type_detailed="Advanced",
                                  per_mode_detailed="PerGame").get_data_frames()[0][
              ["TEAM_ID","TEAM_ABBREVIATION","PACE","REB_PCT","AST_PCT"]]
        try:
            defn = LeagueDashTeamStats(season=ss, measure_type_detailed_defense="Defense",
                                       per_mode_detailed="PerGame").get_data_frames()[0][
                   ["TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"]]
            df = adv.merge(defn, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
        except Exception:
            df = adv.copy()
            df["DEF_RATING"] = 113.0
        league_avg = {c: df[c].mean() for c in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]}
        ctx = {}
        for _, r in df.iterrows():
            ctx[str(r["TEAM_ABBREVIATION"]).upper()] = {
                "PACE": float(r.get("PACE",0)),
                "DEF_RATING": float(r.get("DEF_RATING",113)),
                "REB_PCT": float(r.get("REB_PCT",0)),
                "AST_PCT": float(r.get("AST_PCT",0)),
            }
        return ctx, league_avg
    except Exception:
        return {}, {}

TEAM_CTX, LEAGUE_CTX = get_team_context()

def get_context_multiplier(opp, market, position):
    def _hash_fallback(o):
        base = 1.0
        bucket = get_position_bucket(position or "")
        if bucket=="Guard" and market in ["Assists","PA","RA"]: base *= 1.03
        elif bucket=="Big" and market in ["Rebounds","PR","RA"]: base *= 1.04
        if o:
            h = sum(ord(c) for c in str(o).upper())
            base *= (1 + ((h%15)-7)/200.0)
        return float(np.clip(base, 0.90, 1.10))
    if not LEAGUE_CTX or not TEAM_CTX or not opp:
        return _hash_fallback(opp)
    ok = str(opp).upper()
    if ok not in TEAM_CTX:
        return _hash_fallback(opp)
    o = TEAM_CTX[ok]
    lg = LEAGUE_CTX
    pace_f = o["PACE"] / (lg.get("PACE",100) or 1)
    def_f  = (lg.get("DEF_RATING",113) or 1) / (o["DEF_RATING"] or 1)
    reb_adj = (lg.get("REB_PCT",0.5) or 1) / (o.get("REB_PCT",0.5) or 1)
    ast_adj = (lg.get("AST_PCT",0.6) or 1) / (o.get("AST_PCT",0.6) or 1)
    bucket  = get_position_bucket(position or "")
    if bucket=="Guard":   pos_f = 0.5*ast_adj + 0.5*pace_f
    elif bucket=="Wing":  pos_f = 0.5*def_f + 0.5*pace_f
    elif bucket=="Big":   pos_f = 0.6*reb_adj + 0.4*def_f
    else:                 pos_f = pace_f
    if market=="Rebounds":  mult = 0.30*pace_f+0.25*def_f+0.30*reb_adj+0.15*pos_f
    elif market=="Assists": mult = 0.30*pace_f+0.25*def_f+0.30*ast_adj+0.15*pos_f
    elif market in ("RA","PA","PR"): mult = 0.25*pace_f+0.20*def_f+0.25*reb_adj+0.20*ast_adj+0.10*pos_f
    else:                   mult = 0.45*pace_f+0.40*def_f+0.15*pos_f
    return float(np.clip(mult, 0.80, 1.30))

def advanced_context_multiplier(player_name, market, opp, teammate_out):
    pos = get_player_position(player_name) or ""
    base = get_context_multiplier(opp, market, pos)
    if teammate_out: base *= 1.05
    return float(base)

def estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=None):
    if spread_abs is not None:
        s = abs(float(spread_abs))
        if s < 5: return 0.05
        if s < 8: return 0.10
        if s < 12: return 0.18
        if s < 16: return 0.26
        return 0.33
    if TEAM_CTX and LEAGUE_CTX and team_abbr and opp_abbr:
        tk, ok = str(team_abbr).upper(), str(opp_abbr).upper()
        if tk in TEAM_CTX and ok in TEAM_CTX:
            def_gap = abs(TEAM_CTX[ok].get("DEF_RATING",113)-TEAM_CTX[tk].get("DEF_RATING",113)) / (LEAGUE_CTX.get("DEF_RATING",113) or 1)
            if def_gap<0.05: return 0.06
            if def_gap<0.10: return 0.10
            if def_gap<0.18: return 0.18
            return 0.24
    return 0.10

# ──────────────────────────────────────────────
# ODDS API  [FIX 7: retry on 429]
# ──────────────────────────────────────────────
def odds_api_key():
    return (st.secrets.get("ODDS_API_KEY","") if hasattr(st,"secrets") else "") or os.getenv("ODDS_API_KEY","")

# [FIX 7] Retry with exponential backoff on 429
def http_get_json(url, params, timeout=25):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            rem = r.headers.get("x-requests-remaining")
            used = r.headers.get("x-requests-used")
            st.session_state["_odds_headers_last"] = {"remaining":rem,"used":used,"ts":_now_iso()}
            if r.status_code == 429:
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                return None, "Rate limited (429) - quota exhausted"
            r.raise_for_status()
            return r.json(), None
        except requests.exceptions.HTTPError as e:
            detail = ""
            try: detail = e.response.text[:2000]
            except: pass
            return None, f"HTTP {getattr(e.response,'status_code',None)}: {detail}"
        except requests.exceptions.ConnectionError:
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return None, "Connection failed after retries"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    return None, "All retries failed"

@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_events(date_iso=None):
    key = odds_api_key()
    if not key: return [], "Missing ODDS_API_KEY"
    data, err = http_get_json(f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events", {"apiKey":key})
    if err or not isinstance(data, list): return [], err or "Unexpected events response"
    if date_iso:
        return [ev for ev in data if (ev.get("commence_time") or "")[:10] == date_iso], None
    return data, None

# [FIX 14] Week-ahead: fetch events for a date range
@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_events_range(start_iso, end_iso):
    key = odds_api_key()
    if not key: return [], "Missing ODDS_API_KEY"
    data, err = http_get_json(f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events", {"apiKey":key})
    if err or not isinstance(data, list): return [], err or "Unexpected events response"
    filtered = []
    for ev in data:
        ct = (ev.get("commence_time") or "")[:10]
        if start_iso <= ct <= end_iso:
            filtered.append(ev)
    return filtered, None

@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_event_odds(event_id, market_keys, regions=REGION_US):
    key = odds_api_key()
    if not key: return None, "Missing ODDS_API_KEY"
    data, err = http_get_json(
        f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{event_id}/odds",
        {"apiKey":key,"regions":regions,"markets":",".join(market_keys),
         "oddsFormat":"decimal","dateFormat":"iso"}
    )
    return data, err

@st.cache_data(ttl=60*60*24, show_spinner=False)
def lookup_player_id(name):
    if not name: return None
    nm = normalize_name(name)
    plist = nba_players.get_players()
    for p in plist:
        if normalize_name(p.get("full_name","")) == nm:
            return p.get("id")
    names = [p.get("full_name","") for p in plist]
    cand = difflib.get_close_matches(name, names, n=1, cutoff=0.75)
    if cand:
        for p in plist:
            if p.get("full_name") == cand[0]: return p.get("id")
    return None

def nba_headshot_url(pid):
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" if pid else None

@st.cache_data(ttl=60*60*24, show_spinner=False)
def get_team_maps():
    teams = nba_teams.get_teams()
    by_name = {}
    for t in teams:
        full = t.get("full_name","")
        by_name[normalize_name(full)] = {"abbr":t.get("abbreviation",""),"id":t.get("id"),"full_name":full}
    for a,tgt in [("la clippers","los angeles clippers"),("la lakers","los angeles lakers")]:
        if normalize_name(tgt) in by_name: by_name[normalize_name(a)] = by_name[normalize_name(tgt)]
    return by_name

def map_team_name_to_abbr(name):
    m = get_team_maps()
    rec = m.get(normalize_name(name))
    return rec["abbr"] if rec else None

@st.cache_data(ttl=60*60*24, show_spinner=False)
def team_id_to_abbr_map():
    return {int(t["id"]): t["abbreviation"] for t in nba_teams.get_teams()}

@st.cache_data(ttl=60*10, show_spinner=False)
def nba_scoreboard_games(game_date):
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y"))
        df = sb.get_data_frames()[0]
        return [{"game_id":str(r.get("GAME_ID")),"home_team_id":int(r.get("HOME_TEAM_ID")),
                 "away_team_id":int(r.get("VISITOR_TEAM_ID"))} for _,r in df.iterrows()], None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

def opponent_from_team_abbr(team_abbr, game_date):
    games, _ = nba_scoreboard_games(game_date)
    tid_map = team_id_to_abbr_map()
    for g in (games or []):
        ha = tid_map.get(g["home_team_id"])
        aa = tid_map.get(g["away_team_id"])
        if ha == team_abbr: return aa, True
        if aa == team_abbr: return ha, False
    return None, None

def _parse_player_prop_outcomes(event_odds, market_key, book_filter=None):
    if not event_odds: return [], None
    eid = event_odds.get("id"); home = event_odds.get("home_team"); away = event_odds.get("away_team")
    ct = event_odds.get("commence_time"); books = event_odds.get("bookmakers",[]) or []
    rows = []
    for b in books:
        bkey = b.get("key")
        if book_filter and book_filter not in ("consensus","all") and bkey != book_filter: continue
        for mk in b.get("markets",[]) or []:
            if mk.get("key") != market_key: continue
            for out in mk.get("outcomes",[]) or []:
                player = out.get("description") or out.get("name")
                if player and out.get("point") is not None:
                    rows.append({"player":player,"player_norm":normalize_name(player),
                                 "line":float(out.get("point")),"price":out.get("price"),
                                 "book":(bkey or ""),"side":(out.get("name") or ""),
                                 "market_key":market_key,"event_id":eid,
                                 "home_team":home,"away_team":away,"commence_time":ct})
    if book_filter == "consensus" and rows:
        df = pd.DataFrame(rows)
        df = df[pd.to_numeric(df["line"],errors="coerce").notna()].copy()
        if df.empty: return [], None
        df["w"] = df["book"].apply(lambda k: book_sharpness(str(k)))
        name_map = df.groupby("player_norm")["player"].agg(lambda x: x.value_counts().index[0]).to_dict()
        out_rows = []
        for (pn, side), sub in df.groupby(["player_norm","side"], dropna=False):
            sub = sub.copy().sort_values("line")
            if float(sub["w"].sum() or 0)>0:
                cw = sub["w"].cumsum(); cutoff=0.5*float(sub["w"].sum())
                line_med = float(sub.loc[cw>=cutoff,"line"].iloc[0])
            else:
                line_med = float(sub["line"].median())
            price_syn = None
            try:
                v = sub[pd.to_numeric(sub["price"],errors="coerce").notna()].copy()
                if not v.empty:
                    v["pf"] = pd.to_numeric(v["price"],errors="coerce").astype(float)
                    v = v[v["pf"]>1.0001]
                    if not v.empty:
                        v["pi"] = 1.0/v["pf"]; ws=float(v["w"].sum() or 0)
                        ps = float((v["pi"]*v["w"]).sum()/max(ws,1e-9))
                        ps = float(np.clip(ps,1e-6,1-1e-6)); price_syn=float(1.0/ps)
            except Exception: price_syn=None
            out_rows.append({"player":name_map.get(pn,pn),"player_norm":pn,"line":float(line_med),
                             "price":price_syn,"book":"consensus","side":side,"market_key":market_key,
                             "event_id":eid,"home_team":home,"away_team":away,"commence_time":ct})
        return out_rows, None
    return rows, None

def find_player_line_from_events(player_name, market_key, date_iso, book_choice):
    evs, err = odds_get_events(date_iso)
    if err: return None, None, err
    if not evs: return None, None, "No events for that date"
    target = normalize_name(player_name)
    # [FIX H1/H2/ALT] Use broader regions for specialty markets
    regions = "us,us2,eu,uk" if market_key in SPECIALTY_MARKET_KEYS else REGION_US
    for ev in evs:
        eid = ev.get("id")
        if not eid: continue
        odds, oerr = odds_get_event_odds(eid, (market_key,), regions=regions)
        if oerr or not odds: continue
        rows, _ = _parse_player_prop_outcomes(odds, market_key, book_filter=book_choice)
        for r in rows:
            if r.get("player_norm") == target: return float(r["line"]), r, None
        norms = [r.get("player_norm","") for r in rows]
        close = difflib.get_close_matches(target, norms, n=1, cutoff=0.88)
        if close:
            rr = next((x for x in rows if x.get("player_norm")==close[0]), None)
            if rr: return float(rr["line"]), rr, None
    return None, None, "Player/market not found in Odds API props"

def get_sportsbook_choices(date_iso):
    evs, err = odds_get_events(date_iso)
    if err or not evs: return ["consensus"], err
    for ev in evs[:6]:
        eid = ev.get("id")
        if not eid: continue
        odds, oerr = odds_get_event_odds(eid, (ODDS_MARKETS["Points"],))
        if odds and not oerr:
            books = sorted(list(dict.fromkeys(
                b.get("key") for b in odds.get("bookmakers",[]) if b.get("key"))))
            return ["consensus"] + books, None
    return ["consensus"], None

# ──────────────────────────────────────────────
# SHARP BOOK DIVERGENCE ALERT
# ──────────────────────────────────────────────
def sharp_divergence_alert(event_id, market_key, player_norm, side, model_side="Over"):
    try:
        sharp_books = ["pinnacle","circa","bookmaker"]
        soft_odds, _ = odds_get_event_odds(str(event_id), (str(market_key),))
        if not soft_odds: return {}
        sharp_lines, soft_lines = [], []
        for b in soft_odds.get("bookmakers",[]) or []:
            bk = (b.get("key") or "").lower()
            for mk in b.get("markets",[]) or []:
                if mk.get("key") != market_key: continue
                for out in mk.get("outcomes",[]) or []:
                    pn = normalize_name(out.get("description") or out.get("name") or "")
                    if pn == player_norm and (out.get("name") or "").lower() == side.lower():
                        line = out.get("point")
                        if line is not None:
                            if bk in sharp_books: sharp_lines.append(float(line))
                            else: soft_lines.append(float(line))
        if not sharp_lines or not soft_lines: return {}
        sl = np.mean(sharp_lines); softl = np.mean(soft_lines)
        diff = sl - softl
        return {"sharp_line": sl, "soft_line": softl, "diff": diff,
                "confirm": abs(diff) < 0.3, "fade_model": (diff < -0.5 if "over" in model_side.lower() else diff > 0.5)}
    except Exception:
        return {}

# ──────────────────────────────────────────────
# CLV TRACKING  [FIX 6: no-vig CLV]
# ──────────────────────────────────────────────
def fetch_latest_market_for_leg(leg):
    try:
        eid = leg.get("event_id"); mk = leg.get("market_key")
        if not eid or not mk: return None, None, None, "missing event_id/market_key"
        pn = leg.get("player_norm") or normalize_name(leg.get("player",""))
        side = (leg.get("side") or "Over").strip()
        for bf in [(leg.get("book") or "consensus").strip().lower(), "consensus"]:
            odds, oerr = odds_get_event_odds(str(eid), (str(mk),))
            if oerr or not odds: return None, None, None, oerr or "fetch failed"
            rows, _ = _parse_player_prop_outcomes(odds, str(mk), book_filter=(bf if bf!="all" else None))
            m = next((r for r in rows if r.get("player_norm")==pn and str(r.get("side","")).strip()==side), None)
            if m:
                ln = safe_float(m.get("line")); pr = m.get("price")
                try: pr = float(pr) if pr is not None else None
                except: pr = None
                return ln, pr, (m.get("book") or bf), None
        return None, None, None, "player/side not found"
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"

def apply_clv_update_to_legs(legs):
    errs, out = [], []
    for leg in legs:
        leg2 = dict(leg)
        line0 = safe_float(leg2.get("line")); price0 = safe_float(leg2.get("price_decimal"))
        line1, price1, book_used, err = fetch_latest_market_for_leg(leg2)
        leg2["close_ts"]=_now_iso(); leg2["line_close"]=line1
        leg2["price_close"]=price1; leg2["book_close"]=book_used
        side = (leg2.get("side") or "Over").strip().lower()
        if line0 is not None and line1 is not None:
            leg2["clv_line"]=float(line1-line0)
            leg2["clv_line_fav"]=bool(line1<line0 if "under" not in side else line1>line0)
        else:
            leg2["clv_line"]=None; leg2["clv_line_fav"]=None
        # [FIX 6] No-vig CLV for prices
        if price0 is not None and price1 is not None and price0 > 1 and price1 > 1:
            # Approximate other side as complement
            other0 = max(1.01, 1.0 / max(0.01, 1.0 - 1.0/price0))
            other1 = max(1.01, 1.0 / max(0.01, 1.0 - 1.0/price1))
            nv0, _ = remove_vig(price0, other0)
            nv1, _ = remove_vig(price1, other1)
            leg2["clv_price"]=float(nv1-nv0)
            leg2["clv_price_fav"]=bool(nv1>nv0)
            leg2["clv_price_novig_open"]=float(nv0)
            leg2["clv_price_novig_close"]=float(nv1)
        else:
            leg2["clv_price"]=None; leg2["clv_price_fav"]=None
        if err: errs.append(f"{leg2.get('player')} {leg2.get('market')}: {err}")
        out.append(leg2)
    return out, errs

# ──────────────────────────────────────────────
# USAGE RATE FROM BOX SCORE
# ──────────────────────────────────────────────
def compute_usage_rate(game_log_df, n_games=10):
    """Approximate possession usage: FGA + 0.44*FTA + TOV per game."""
    if game_log_df is None or game_log_df.empty:
        return None
    df = game_log_df.head(n_games).copy()
    if not all(c in df.columns for c in ["FGA","FTA","TOV"]):
        return None
    try:
        fga = pd.to_numeric(df["FGA"], errors="coerce").fillna(0)
        fta = pd.to_numeric(df["FTA"], errors="coerce").fillna(0)
        tov = pd.to_numeric(df["TOV"], errors="coerce").fillna(0)
        return float((fga + 0.44*fta + tov).mean())
    except Exception:
        return None

# ──────────────────────────────────────────────
# PACE-ADJUSTED STAT SERIES
# ──────────────────────────────────────────────
def compute_pace_adjusted_series(stat_series, opp_team):
    """Scale stat series by opponent-vs-league pace ratio (dampened 50%)."""
    if stat_series is None or len(stat_series.dropna()) < 4:
        return stat_series
    if not LEAGUE_CTX or not TEAM_CTX:
        return stat_series
    opp_key = str(opp_team or "").upper()
    if opp_key not in TEAM_CTX:
        return stat_series
    try:
        opp_pace = TEAM_CTX[opp_key].get("PACE", 100)
        league_pace = LEAGUE_CTX.get("PACE", 100) or 1.0
        pace_adj = opp_pace / league_pace
        adj_factor = float(np.clip(1.0 + 0.5*(pace_adj - 1.0), 0.88, 1.12))
        return stat_series * adj_factor
    except Exception:
        return stat_series

# ──────────────────────────────────────────────
# PER-POSITION DEFENSIVE GRADES  (one call, all teams)
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*6, show_spinner=False)
def _fetch_positional_def_full(position_bucket):
    """One LeagueDashPtDefend call → {TEAM_ABBR: pts_allowed} + league_avg."""
    try:
        from nba_api.stats.endpoints import LeagueDashPtDefend
        cat_map = {"Guard": "Guards", "Wing": "Forwards", "Big": "Centers", "Unknown": "Overall"}
        category = cat_map.get(position_bucket, "Overall")
        df = LeagueDashPtDefend(
            league_id="00", per_mode_simple="PerGame",
            defense_category=category, season=get_season_string(),
        ).get_data_frames()[0]
        if df.empty:
            return {}, None
        league_avg = float(df["PTS_ALLOWED"].mean())
        team_map = {str(r["TEAM_ABBREVIATION"]).upper(): float(r["PTS_ALLOWED"])
                    for _, r in df.iterrows()}
        return team_map, league_avg
    except Exception:
        return {}, None

def get_opp_positional_pts_allowed(opp_abbr, position_bucket):
    """Returns (opp_pts_allowed, league_avg) using cached bulk table."""
    team_map, league_avg = _fetch_positional_def_full(position_bucket)
    opp = str(opp_abbr or "").upper()
    return team_map.get(opp), league_avg

def positional_def_multiplier(opp_abbr, position_bucket, market):
    """Return a multiplier based on opponent's positional defensive strength."""
    if market not in ("Points","PRA","PR","PA","H1 Points","H2 Points","Alt Points"):
        return 1.0
    try:
        opp_pts, league_avg = get_opp_positional_pts_allowed(opp_abbr, position_bucket)
        if opp_pts is None or league_avg is None or league_avg == 0:
            return 1.0
        ratio = opp_pts / league_avg
        return float(np.clip(ratio, 0.82, 1.18))
    except Exception:
        return 1.0

# ──────────────────────────────────────────────
# INJURY REPORT  (ESPN public API — works every day, not just game days)
# ──────────────────────────────────────────────
_ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_injury_report():
    """Fetch NBA injury report from ESPN (primary) with NBA API fallback.

    Returns:
        dict: {team_abbr_upper: [{"player": str, "status": str, "reason": str}, ...]}
        AND sets st.session_state["injury_team_map"] = {team_abbr: [player_name, ...]} for OUT/DOUBTFUL only.
    """
    out = {}
    try:
        r = requests.get(
            _ESPN_INJURY_URL,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=15,
        )
        if r.ok:
            data = r.json()
            for entry in data.get("injuries", []):
                team_info = entry.get("team", {})
                abbr = str(team_info.get("abbreviation", "")).upper()
                for inj in entry.get("injuries", []):
                    athlete = inj.get("athlete", {})
                    pname = athlete.get("fullName", "") or athlete.get("displayName", "")
                    status_raw = str(inj.get("status", "")).upper()
                    status = status_raw if status_raw in ("OUT","DOUBTFUL","QUESTIONABLE") else status_raw
                    reason_detail = inj.get("details", {}) or {}
                    reason = reason_detail.get("type","") or reason_detail.get("returnDate","")
                    if pname and status:
                        out.setdefault(abbr, []).append({
                            "player": pname, "status": status, "reason": reason,
                        })
    except Exception:
        pass

    # NBA API fallback (only on game days)
    if not out:
        try:
            from nba_api.stats.endpoints import InjuryReport as NBAInjuryReport
            df = NBAInjuryReport(game_date=date.today().strftime("%m/%d/%Y")).get_data_frames()[0]
            for _, r in df.iterrows():
                team = str(r.get("TEAM_TRICODE","")).upper()
                status = str(r.get("PLAYER_STATUS","")).upper()
                if status in ("OUT","DOUBTFUL","QUESTIONABLE"):
                    out.setdefault(team, []).append({
                        "player": r.get("PLAYER_NAME",""),
                        "status": status,
                        "reason": r.get("RETURN_FROM_INJURY",""),
                    })
        except Exception:
            pass

    return out

# ──────────────────────────────────────────────
# [UPGRADE 9] ROTOWIRE NEWS SCRAPER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*5, show_spinner=False)
def fetch_rotowire_news():
    """Scrape Rotowire NBA injury page. Returns (rows, error)."""
    try:
        url = "https://www.rotowire.com/basketball/injury-report.php"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=15)
        if not r.ok:
            return [], f"HTTP {r.status_code}"
        text = r.text
        rows = []
        # Each player row contains: player link, team, position, status, return date, injury detail
        player_blocks = re.findall(
            r'<td[^>]*class="[^"]*player[^"]*"[^>]*>.*?<a[^>]+>([^<]+)</a>.*?</td>'
            r'.*?<td[^>]*>([A-Z]{2,4})</td>'      # team abbr
            r'.*?<td[^>]*>([A-Z]+)</td>'            # position
            r'.*?<td[^>]*>([^<]{2,30})</td>'        # status
            r'.*?<td[^>]*>([^<]{0,30})</td>',       # return date
            text, re.DOTALL
        )
        for m in player_blocks:
            pname = m[0].strip(); team = m[1].strip()
            pos = m[2].strip(); status = m[3].strip(); ret = m[4].strip()
            if pname and status:
                rows.append({"player": pname, "team": team, "pos": pos,
                             "status": status, "return": ret})
        # Fallback: simpler pattern if the above misses things
        if not rows:
            simple = re.findall(
                r'href="/basketball/player-profile[^"]*">([^<]+)</a>.*?'
                r'class="[^"]*status[^"]*"[^>]*>([^<]+)<',
                text, re.DOTALL
            )
            for pname, status in simple[:60]:
                rows.append({"player": pname.strip(), "status": status.strip(),
                             "team": "", "pos": "", "return": ""})
        return rows[:80], None
    except Exception as ex:
        return [], f"{type(ex).__name__}: {ex}"

def build_injury_team_map(injury_dict):
    """Build {team_abbr: [player_name_lower]} for OUT/DOUBTFUL players only — used for auto key_teammate_out."""
    result = {}
    for team, players in (injury_dict or {}).items():
        out_players = [p["player"].lower() for p in players
                       if str(p.get("status","")).upper() in ("OUT","DOUBTFUL")]
        if out_players:
            result[str(team).upper()] = out_players
    return result

# ──────────────────────────────────────────────
# DD / TD PROBABILITY
# ──────────────────────────────────────────────
def compute_dd_prob(game_log_df, n_games=10):
    """Historical frequency of double-doubles from game log."""
    if game_log_df is None or game_log_df.empty:
        return None
    df = game_log_df.head(n_games).copy()
    try:
        dd = sum(
            1 for _, row in df.iterrows()
            if sum(1 for c in ["PTS","REB","AST","BLK","STL"]
                   if safe_float(row.get(c)) >= 10) >= 2
        )
        return float(dd / len(df)) if len(df) > 0 else None
    except Exception:
        return None

def compute_td_prob(game_log_df, n_games=10):
    """Historical frequency of triple-doubles from game log."""
    if game_log_df is None or game_log_df.empty:
        return None
    df = game_log_df.head(n_games).copy()
    try:
        td = sum(
            1 for _, row in df.iterrows()
            if sum(1 for c in ["PTS","REB","AST","BLK","STL"]
                   if safe_float(row.get(c)) >= 10) >= 3
        )
        return float(td / len(df)) if len(df) > 0 else None
    except Exception:
        return None

# ──────────────────────────────────────────────
# PRIZEPICKS INGESTION
# ──────────────────────────────────────────────
PRIZEPICKS_API = "https://api.prizepicks.com/projections"

def _parse_pp_response(data):
    """Parse PrizePicks JSON response into list of prop dicts."""
    included = {item["id"]: item for item in data.get("included", [])}
    rows = []
    for proj in data.get("data", []):
        if proj.get("type") != "Projection":
            continue
        attrs = proj.get("attributes", {})
        league = str(attrs.get("league", "") or "").upper()
        if league and league not in ("NBA",):
            continue
        rels = proj.get("relationships", {})
        player_id = (rels.get("new_player", {}).get("data", {}) or {}).get("id")
        if not player_id:
            player_id = (rels.get("player", {}).get("data", {}) or {}).get("id")
        player_attrs = included.get(player_id, {}).get("attributes", {}) if player_id else {}
        player_name = player_attrs.get("name", "") or attrs.get("name", "")
        stat_type = attrs.get("stat_type", "")
        line_score = attrs.get("line_score")
        if player_name and stat_type and line_score is not None:
            rows.append({
                "player": player_name,
                "stat_type": stat_type,
                "line": float(line_score),
                "start_time": attrs.get("start_time", ""),
                "source": "prizepicks",
            })
    return rows

def _pp_request(per_page=500, cookies_str=""):
    """Make one PrizePicks API request.
    Tries curl_cffi (Chrome TLS impersonation) first, then plain requests.
    Returns (response_object_or_None, error_str_or_None).
    """
    url = PRIZEPICKS_API
    params = {"league_id": "7", "per_page": str(per_page),
              "single_stat": "true", "in_play": "false"}
    headers = {
        "Accept": "application/vnd.api+json",
        "Referer": "https://app.prizepicks.com/",
        "Origin": "https://app.prizepicks.com",
    }
    # Parse optional cookie string from user settings
    cookie_dict = {}
    if cookies_str:
        for part in cookies_str.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                cookie_dict[k.strip()] = v.strip()

    # ── Attempt 1: curl_cffi Chrome TLS impersonation (bypasses PerimeterX) ──
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(
            url, params=params, headers=headers,
            cookies=cookie_dict or None,
            impersonate="chrome120",
            timeout=25,
        )
        return r, None
    except ImportError:
        pass
    except Exception as e:
        pass  # fall through to plain requests

    # ── Attempt 2: plain requests (works locally, blocked on cloud IPs) ──
    try:
        r = requests.get(url, params=params, headers=headers,
                         cookies=cookie_dict or None, timeout=20)
        return r, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

@st.cache_data(ttl=60*10, show_spinner=False)
def _fetch_prizepicks_lines_cached(cookies_str=""):
    for per_page in (500, 250):
        # Retry up to 3 times on 429 with backoff
        for attempt in range(3):
            r, err = _pp_request(per_page=per_page, cookies_str=cookies_str)
            if err:
                return [], err
            if r is None:
                return [], "No response from PrizePicks"
            if r.status_code == 429:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))  # 2s, 4s
                    continue
                return [], (
                    "PrizePicks rate-limited (429) — Streamlit Cloud IP is throttled. "
                    "Wait 60s and retry, or run the app locally."
                )
            if r.status_code == 403:
                return [], (
                    "HTTP 403 — PerimeterX block. Fix: paste your PrizePicks browser "
                    "cookies into Settings → PrizePicks Cookies, then retry."
                )
            if not r.ok:
                return [], f"HTTP {r.status_code}: {r.text[:300]}"
            try:
                rows = _parse_pp_response(r.json())
            except Exception as e:
                return [], f"Parse error: {e}"
            if rows:
                return rows, None
            break  # non-429, non-error, empty — try next per_page
    return [], "No NBA props found — slate may not be posted yet"

def fetch_prizepicks_lines():
    cookies_str = st.session_state.get("pp_cookies", "")
    # Only clear cache when cookies changed — avoids redundant API hits
    if st.session_state.get("_pp_last_cookies_used") != cookies_str:
        _fetch_prizepicks_lines_cached.clear()
        st.session_state["_pp_last_cookies_used"] = cookies_str
    return _fetch_prizepicks_lines_cached(cookies_str=cookies_str)

# ──────────────────────────────────────────────
# UNDERDOG INGESTION
# ──────────────────────────────────────────────
# Try v3 then v4 endpoint
_UNDERDOG_ENDPOINTS = [
    "https://api.underdogfantasy.com/v3/over_under_lines",
    "https://api.underdogfantasy.com/v4/over_under_lines",
    "https://api.underdogfantasy.com/v2/over_under_lines",
]
_UD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://underdogfantasy.com/",
    "Origin": "https://underdogfantasy.com",
}
# Basketball sport IDs that Underdog may use (numeric or string)
_UD_BASKETBALL_SPORT_IDS = {"nba", "basketball", "5", "4", "nba_basketball", "basketball_nba", ""}

@st.cache_data(ttl=60*10, show_spinner=False)
def _fetch_underdog_lines_cached():
    """Inner cached fetch — call via fetch_underdog_lines() which clears cache first."""
    for url in _UNDERDOG_ENDPOINTS:
        try:
            r = requests.get(url, headers=_UD_HEADERS, timeout=20)
            if r.status_code == 429:
                return [], "Underdog rate-limited (429) — try again in 30s"
            if not r.ok:
                continue
            data = r.json()
            appearances = {a["id"]: a for a in data.get("appearances", [])}
            players_map = {p["id"]: p for p in data.get("players", [])}
            rows = []
            for line in data.get("over_under_lines", []):
                app_stat = line.get("over_under", {}).get("appearance_stat", {})
                app_id = str(app_stat.get("appearance_id", ""))
                app = appearances.get(app_id, {})
                sport = str(app.get("sport_id", "")).lower()
                if sport and sport not in _UD_BASKETBALL_SPORT_IDS:
                    continue
                player_id = str(app.get("player_id", ""))
                player = players_map.get(player_id, {})
                player_name = f"{player.get('first_name','')} {player.get('last_name','')}".strip()
                stat_type = app_stat.get("display_stat", "")
                stat_value = line.get("stat_value")
                if player_name and stat_type and stat_value is not None:
                    rows.append({
                        "player": player_name,
                        "stat_type": stat_type,
                        "line": float(stat_value),
                        "source": "underdog",
                    })
            if rows:
                return rows, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return [], locals().get("last_err", "No Underdog props found — slate may not be posted yet")

def fetch_underdog_lines():
    """Clear cache then fetch fresh Underdog lines."""
    _fetch_underdog_lines_cached.clear()
    return _fetch_underdog_lines_cached()

def map_platform_stat_to_market(stat_type):
    """Map PrizePicks/Underdog stat label to internal market name."""
    mapping = {
        "Points": "Points", "Pts": "Points",
        "Rebounds": "Rebounds", "Reb": "Rebounds",
        "Assists": "Assists", "Ast": "Assists",
        "3-Pointers Made": "3PM", "3 Pointers Made": "3PM", "3PM": "3PM",
        "Pts+Reb+Ast": "PRA", "Pts+Reb": "PR", "Pts+Ast": "PA", "Reb+Ast": "RA",
        "Blocked Shots": "Blocks", "Blocks": "Blocks", "Blk": "Blocks",
        "Steals": "Steals", "Stl": "Steals",
        "Turnovers": "Turnovers", "Tov": "Turnovers",
        "Blks+Stls": "Stocks", "Stocks": "Stocks",
    }
    for k, v in mapping.items():
        if k.lower() == str(stat_type).strip().lower():
            return v
    return None

# ──────────────────────────────────────────────
# LINE SHOPPING — BEST AVAILABLE PRICE
# ──────────────────────────────────────────────
def get_best_available_price(event_id, market_key, player_norm, side):
    """Return (best_price, best_book) across all books for a given player/market/side."""
    try:
        odds, err = odds_get_event_odds(str(event_id), (str(market_key),))
        if err or not odds:
            return None, None
        best_price, best_book = None, None
        for b in odds.get("bookmakers", []) or []:
            bkey = b.get("key", "")
            for mk in b.get("markets", []) or []:
                if mk.get("key") != market_key:
                    continue
                for out in mk.get("outcomes", []) or []:
                    pn = normalize_name(out.get("description") or out.get("name") or "")
                    if pn == player_norm and (out.get("name") or "").lower() == str(side).lower():
                        p = safe_float(out.get("price"))
                        if p > 1.0 and (best_price is None or p > best_price):
                            best_price = p
                            best_book = bkey
        return best_price, best_book
    except Exception:
        return None, None

# ──────────────────────────────────────────────
# PROP LINE HISTORY DATABASE
# ──────────────────────────────────────────────
PROP_HISTORY_PATH = "prop_line_history.jsonl"

def save_prop_line(player, market, line, price, book, event_id=None):
    """Append a prop line snapshot to JSONL history file."""
    try:
        with open(PROP_HISTORY_PATH, "a") as f:
            f.write(json.dumps({
                "ts": _now_iso(), "player": player, "market": market,
                "line": float(line) if line is not None else None,
                "price": float(price) if price is not None else None,
                "book": book, "event_id": event_id,
            }) + "\n")
    except Exception:
        pass

# ──────────────────────────────────────────────
# [UPGRADE 10] OPENING LINE CAPTURE
# ──────────────────────────────────────────────
def save_opening_line(player_norm, market_key, side, line, price):
    """Persist the first-seen line for a player/market/side as the 'opening' line."""
    try:
        data = {}
        if os.path.exists(OPENING_LINES_PATH):
            with open(OPENING_LINES_PATH) as f:
                data = json.load(f)
        # Include date in key so each day gets its own independent opening line
        today = date.today().isoformat()
        key = f"{player_norm}|{market_key}|{side}|{today}"
        if key not in data:  # Only write once per key (true opening)
            data[key] = {"line": float(line), "price": price, "ts": _now_iso(), "date": today}
            with open(OPENING_LINES_PATH, "w") as f:
                json.dump(data, f)
    except Exception as ex:
        logging.warning(f"save_opening_line: {ex}")

def get_opening_line(player_norm, market_key, side):
    """Return (opening_line, opening_price) or (None, None) if not recorded."""
    try:
        if not os.path.exists(OPENING_LINES_PATH):
            return None, None
        with open(OPENING_LINES_PATH) as f:
            data = json.load(f)
        today = date.today().isoformat()
        key = f"{player_norm}|{market_key}|{side}|{today}"
        rec = data.get(key)
        if rec:
            return rec.get("line"), rec.get("price")
    except Exception as ex:
        logging.warning(f"get_opening_line: {ex}")
    return None, None

def clear_opening_lines():
    try:
        if os.path.exists(OPENING_LINES_PATH):
            os.remove(OPENING_LINES_PATH)
    except Exception:
        pass

# ──────────────────────────────────────────────
# WATCHLIST PERSISTENCE
# ──────────────────────────────────────────────
def load_watchlist(uid):
    path = WATCHLIST_PATH_TPL.format(uid=re.sub(r"[^a-zA-Z0-9_-]", "_", uid or "default"))
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f) or []
    except Exception:
        pass
    return []

def save_watchlist(uid, players):
    path = WATCHLIST_PATH_TPL.format(uid=re.sub(r"[^a-zA-Z0-9_-]", "_", uid or "default"))
    try:
        with open(path, "w") as f:
            json.dump(list(players), f)
    except Exception:
        pass

def load_prop_line_history(player=None, market=None, limit=500):
    """Load prop line history filtered by player/market."""
    try:
        if not os.path.exists(PROP_HISTORY_PATH):
            return pd.DataFrame()
        rows = []
        with open(PROP_HISTORY_PATH) as f:
            for raw in f:
                try:
                    r = json.loads(raw.strip())
                    if player and normalize_name(r.get("player","")) != normalize_name(player):
                        continue
                    if market and r.get("market","").lower() != market.lower():
                        continue
                    rows.append(r)
                except Exception:
                    continue
        return pd.DataFrame(rows[-limit:]) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ──────────────────────────────────────────────
# DISCORD / TELEGRAM ALERTS
# ──────────────────────────────────────────────
def send_discord_alert(webhook_url, message):
    if not webhook_url:
        return False, "No webhook URL"
    try:
        r = requests.post(webhook_url, json={"content": message, "username": "NBA Quant Engine"}, timeout=10)
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return False, "Token/chat_id missing"
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def format_edge_alert(leg):
    p_cal = leg.get("p_cal") or 0
    ev = (leg.get("ev_adj_pct") or (leg.get("ev_adj", 0) * 100 if leg.get("ev_adj") else 0))
    proj = leg.get("proj")
    return (
        f"**{leg.get('player','?')}** — {leg.get('market','?')} O{leg.get('line','?')}\n"
        f"Proj: {proj:.1f} | P: {p_cal*100:.1f}% | EV: {ev:.1f}%\n"
        f"Book: {leg.get('book','?')} | {leg.get('edge_cat','')}"
    ) if proj else (
        f"**{leg.get('player','?')}** — {leg.get('market','?')} O{leg.get('line','?')}\n"
        f"P: {p_cal*100:.1f}% | EV: {ev:.1f}% | {leg.get('edge_cat','')}"
    )

# ──────────────────────────────────────────────
# [UPGRADE 23] ALERT DIGEST FORMATTER
# ──────────────────────────────────────────────
def format_digest_message(edges, as_of=None):
    """Format a ranked daily digest of top edges for Discord/Telegram."""
    ts = (as_of or datetime.utcnow()).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"**NBA QUANT ENGINE — DAILY DIGEST** ({ts})\n"]
    for i, leg in enumerate(edges[:15], 1):
        ev = leg.get("ev_adj_pct") or (float(leg.get("ev_adj", 0) or 0) * 100)
        p  = float(leg.get("p_cal") or 0) * 100
        proj = leg.get("proj")
        proj_str = f"{proj:.1f}" if proj is not None else "--"
        lines.append(
            f"{i}. **{leg.get('player','?')}** {leg.get('market','?')} "
            f"O{leg.get('line','?')} | Proj: {proj_str} | "
            f"P: {p:.0f}% | EV: {ev:+.1f}% | {leg.get('edge_cat','')}"
        )
    return "\n".join(lines)

# ──────────────────────────────────────────────
# [UPGRADE 31] CLV LEADERBOARD
# ──────────────────────────────────────────────
def compute_clv_leaderboard(history_df, top_n=20):
    """Return top bets ranked by no-vig CLV price improvement."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in history_df.iterrows():
        try:
            legs = json.loads(r.get("legs", "[]")) if isinstance(r.get("legs"), str) else []
        except Exception:
            legs = []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            clv_p = leg.get("clv_price")
            clv_l = leg.get("clv_line")
            if clv_p is None and clv_l is None:
                continue
            rows.append({
                "ts":     r.get("ts", ""),
                "player": leg.get("player", "?"),
                "market": leg.get("market", "?"),
                "line":   leg.get("line"),
                "side":   leg.get("side", "Over"),
                "result": r.get("result", "Pending"),
                "clv_line":  safe_round(clv_l, 2),
                "clv_price": safe_round(clv_p, 4),
                "clv_line_fav":  bool(leg.get("clv_line_fav")),
                "clv_price_fav": bool(leg.get("clv_price_fav")),
                "novig_open":  safe_round(leg.get("clv_price_novig_open"), 4),
                "novig_close": safe_round(leg.get("clv_price_novig_close"), 4),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("clv_price", ascending=False).head(top_n)
    return df.reset_index(drop=True)

# ──────────────────────────────────────────────
# [UPGRADE 32] PER-BOOK MARKET EFFICIENCY SCORE
# ──────────────────────────────────────────────
def compute_book_efficiency(history_df):
    """Track per-book win rate and CLV rate to rank market efficiency."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in history_df.iterrows():
        res = r.get("result", "Pending")
        if res not in ("HIT", "MISS"):
            continue
        y = 1 if res == "HIT" else 0
        try:
            legs = json.loads(r.get("legs", "[]")) if isinstance(r.get("legs"), str) else []
        except Exception:
            legs = []
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            book = leg.get("book") or "unknown"
            rows.append({
                "book":    book,
                "y":       y,
                "clv_fav": int(bool(leg.get("clv_price_fav"))),
                "ev_adj":  safe_float(leg.get("ev_adj"), 0.0),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    result = df.groupby("book").agg(
        bets=("y", "size"),
        hit_rate=("y", "mean"),
        clv_fav_rate=("clv_fav", "mean"),
        avg_ev=("ev_adj", "mean"),
    ).reset_index()
    result["hit_rate_%"] = (result["hit_rate"] * 100).round(1)
    result["clv_fav_%"]  = (result["clv_fav_rate"] * 100).round(1)
    result["avg_ev_%"]   = (result["avg_ev"] * 100).round(2)
    result = result[result["bets"] >= 3].sort_values("hit_rate_%", ascending=False)
    return result[["book", "bets", "hit_rate_%", "clv_fav_%", "avg_ev_%"]].reset_index(drop=True)

# ──────────────────────────────────────────────
# [UPGRADE 34] BAYESIAN PRIOR UPDATE FROM HISTORY
# ──────────────────────────────────────────────
def compute_history_based_priors(legs_df, position_bucket):
    """Blend positional priors with personal hit/miss data by market."""
    base = dict(POSITIONAL_PRIORS.get(position_bucket, POSITIONAL_PRIORS["Unknown"]))
    if legs_df is None or legs_df.empty:
        return base
    settled = legs_df[legs_df["y"].notna() & legs_df["market"].notna()].copy()
    if len(settled) < 20:
        return base
    mkt_stats = settled.groupby("market").agg(
        hit_rate=("y", "mean"), n=("y", "size")
    ).reset_index()
    for _, row in mkt_stats.iterrows():
        mkt = row["market"]; n = row["n"]
        if n < 8 or mkt not in base:
            continue
        # Bayesian blend: personal history weight grows with sample size (max 35%)
        w_personal = min(n / (n + 30), 0.35)
        # Positive hit rate → market is offering value → effectively lower the prior (easier line)
        adj = 1.0 + w_personal * (row["hit_rate"] - 0.50) * 0.25
        base[mkt] = base[mkt] * float(np.clip(adj, 0.88, 1.12))
    return base

# ──────────────────────────────────────────────
# KELLY PARLAY OPTIMIZER
# ──────────────────────────────────────────────
def kelly_parlay_optimizer(legs, payout_mult, max_legs=4, bankroll=1000.0, frac_kelly=0.25):
    """Find best 2-N leg combos using PSD covariance matrix + Gaussian copula MC simulation."""
    from itertools import combinations
    import scipy.stats as _sc

    N_SIMS_PARLAY = 3000

    valid = [l for l in legs if l.get("gate_ok") and float(l.get("p_cal") or 0) > 0.50]
    if len(valid) < 2:
        return []

    # Pre-build full N×N correlation matrix once (avoid redundant pairwise calls in inner loop)
    nv = len(valid)
    full_corr = np.eye(nv)
    for _i in range(nv):
        for _j in range(_i + 1, nv):
            c = float(estimate_player_correlation(valid[_i], valid[_j]) or 0.0)
            full_corr[_i, _j] = full_corr[_j, _i] = c

    rng = np.random.default_rng(42)
    results = []

    for n in range(2, min(max_legs + 1, nv + 1)):
        for combo in combinations(range(nv), n):
            combo_legs = [valid[i] for i in combo]
            probs = np.array([float(l["p_cal"]) for l in combo_legs])
            naive_joint = float(np.prod(probs))

            # Extract n×n sub-matrix and make PSD via eigenvalue clipping
            sub_corr = full_corr[np.ix_(list(combo), list(combo))]
            evals, evecs = np.linalg.eigh(sub_corr)
            evals = np.clip(evals, 1e-6, None)
            corr_psd = evecs @ np.diag(evals) @ evecs.T

            # Gaussian copula MC
            z = rng.multivariate_normal(np.zeros(n), corr_psd, N_SIMS_PARLAY)
            u = _sc.norm.cdf(z)
            hits = u < probs  # shape (N_SIMS_PARLAY, n)
            joint = float(hits.all(axis=1).mean())
            joint = float(np.clip(joint, 1e-6, 1.0))

            ev = payout_mult * joint - 1.0
            kelly_f = max(0.0, ev / (payout_mult - 1.0)) if payout_mult > 1 else 0.0
            stake = min(bankroll * frac_kelly * kelly_f, bankroll * 0.05)
            results.append({
                "combo": " + ".join(f"{l['player']} {l['market']}" for l in combo_legs),
                "n_legs": n,
                "joint_prob_%": round(joint * 100, 1),
                "naive_prob_%": round(naive_joint * 100, 1),
                "ev_%": round(ev * 100, 1),
                "payout_x": payout_mult,
                "kelly_stake_$": round(stake, 2),
            })
    return sorted(results, key=lambda x: x["ev_%"], reverse=True)[:25]

# ──────────────────────────────────────────────
# MONTE CARLO GAME SIMULATION
# ──────────────────────────────────────────────
def monte_carlo_game_sim(legs, n_sims=20000, payout_mult=3.0):
    """Correlated MC simulation across all legs."""
    try:
        import scipy.stats as _sc
        valid = [l for l in legs if l.get("p_cal")]
        if not valid:
            return None
        n = len(valid)
        probs = np.array([float(l["p_cal"]) for l in valid])
        corr_mat = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                c = estimate_player_correlation(valid[i], valid[j])
                corr_mat[i,j] = corr_mat[j,i] = float(c or 0.0)
        evals, evecs = np.linalg.eigh(corr_mat)
        evals = np.clip(evals, 1e-6, None)
        corr_psd = evecs @ np.diag(evals) @ evecs.T
        rng = np.random.default_rng(42)
        z = rng.multivariate_normal(np.zeros(n), corr_psd, n_sims)
        u = _sc.norm.cdf(z)
        hits = u < probs
        joint_hits = hits.all(axis=1)
        joint_prob = float(joint_hits.mean())
        ev = payout_mult * joint_prob - 1.0
        return {
            "joint_prob_%": round(joint_prob * 100, 2),
            "naive_joint_%": round(float(np.prod(probs)) * 100, 2),
            "ev_%": round(ev * 100, 2),
            "per_leg_sim_%": [round(float(hits[:,i].mean()) * 100, 1) for i in range(n)],
            "n_sims": n_sims,
        }
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────────────────────────
# ROLLING BRIER SCORE
# ──────────────────────────────────────────────
def compute_rolling_brier(legs_df, windows=(25, 50, 100)):
    """Compute Brier scores over trailing windows and a rolling series."""
    if legs_df is None or legs_df.empty:
        return {}
    d = legs_df[legs_df["y"].notna()].copy().reset_index(drop=True)
    if len(d) < 10:
        return {}
    result = {}
    for w in windows:
        if len(d) >= w:
            tail = d.tail(w)
            result[f"last_{w}"] = float(np.mean((tail["p_raw"].values.astype(float) - tail["y"].values.astype(float))**2))
    if len(d) >= 10:
        series = []
        for i in range(9, len(d)):
            window_d = d.iloc[max(0, i-24):i+1]
            series.append(float(np.mean((window_d["p_raw"].values.astype(float) - window_d["y"].values.astype(float))**2)))
        result["rolling_series"] = series
    return result

# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# CACHE PRE-WARMER  (eliminates NBA API I/O from threads)
# ──────────────────────────────────────────────
def pre_warm_scanner_caches(candidates, n_games):
    """
    Pre-fetch player IDs, game logs, and positions for all unique players in
    parallel (4 workers) before the main scan threads start.  Uses a session-
    state set to skip players already warmed this session.
    """
    unique_names = list(dict.fromkeys(pname for pname, *_ in candidates))
    already_warm = st.session_state.get("_prewarm_done", set())
    to_warm = [n for n in unique_names if n not in already_warm]
    if not to_warm:
        return set(unique_names)   # everything already cached

    # Bulk game logs + bulk position map (each = ONE API call for all players)
    _fetch_bulk_gamelogs()
    _ensure_pid_position_map()

    def _warm_one(name):
        try:
            pid = lookup_player_id(name)
            if pid:
                fetch_player_gamelog(player_id=pid, max_games=max(6, n_games + 5))
                get_player_position(name)
                return name
        except Exception:
            pass
        return None

    resolved = set()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for result in ex.map(_warm_one, to_warm):
            if result:
                resolved.add(result)

    already_warm.update(resolved)
    st.session_state["_prewarm_done"] = already_warm
    return resolved

# MAIN PROJECTION ENGINE  [FIX 3: minutes filter]
# ──────────────────────────────────────────────
def compute_leg_projection(
    player_name, market_name, line, meta,
    n_games, key_teammate_out,
    bankroll=0.0, frac_kelly=0.25, max_risk_frac=0.05,
    market_prior_weight=0.65, exclude_chaotic=True,
    game_date=None, is_home=None,
    injury_team_map=None,   # {team_abbr_upper: [player_name_lower, ...]} for OUT/DOUBTFUL players
):
    errors = []
    game_date = game_date or date.today()
    player_id = lookup_player_id(player_name)
    if not player_id:
        errors.append("Could not resolve NBA player id.")
        return {"player":player_name,"market":market_name,"line":float(line),
                "proj":None,"p_over":None,"p_cal":None,"edge":None,
                "team":None,"opp":None,"headshot":None,"errors":errors,
                "player_id":None,"gate_ok":False,"gate_reason":"no player id"}

    gldf, gl_errs = fetch_player_gamelog(player_id=player_id, max_games=max(6, n_games+5))
    if gl_errs: errors.extend([f"NBA API: {m}" for m in gl_errs])

    gldf_n = gldf.head(n_games) if not gldf.empty else gldf

    # [FIX 3] Filter out DNP/garbage-time games (MIN < threshold)
    if not gldf_n.empty and "MIN" in gldf_n.columns:
        try:
            min_vals = gldf_n["MIN"].apply(lambda v:
                float(str(v).split(":")[0]) if isinstance(v, str) and ":" in str(v)
                else safe_float(v, default=0.0))
            mask = min_vals >= MIN_MINUTES_THRESHOLD
            n_excluded = int((~mask).sum())
            gldf_filtered = gldf_n[mask]
            if len(gldf_filtered) >= 4:
                gldf_n = gldf_filtered
                if n_excluded > 0:
                    errors.append(f"Excluded {n_excluded} low-minute games (<{MIN_MINUTES_THRESHOLD} min)")
        except Exception:
            pass

    # ── Detect special market types ──────────────────────────────
    half_factor = HALF_FACTOR.get(market_name, 1.0)
    is_half_market = market_name in HALF_FACTOR
    is_dd_td = market_name in DD_TD_MARKETS
    is_fantasy = market_name in FANTASY_MARKETS
    # For half/Q1 markets, find effective full-game stat field
    base_market = market_name
    if is_half_market:
        base_market = (market_name.replace("H1 ","").replace("H2 ","").replace("Q1 ",""))

    stat_series = compute_stat_from_gamelog(gldf_n, base_market) if not gldf_n.empty else pd.Series([], dtype=float)

    # [AUDIT FIX] For half/Q1 markets: try real per-period boxscore data.
    # If successful, stat_series is replaced with actual H1/H2/Q1 values so
    # CV, skewness, and bootstrap all operate on real half-game distributions.
    _orig_half_factor = half_factor   # save for Bayesian prior scaling below
    if is_half_market and player_id:
        _hg_series = fetch_player_halfgame_log(player_id, gldf_n, market_name, n_games=n_games)
        if _hg_series is not None and len(_hg_series.dropna()) >= 3:
            stat_series = _hg_series
            half_factor = 1.0   # Real half-game data; no scaling needed anywhere
            errors.append(f"Real {market_name} split: {len(_hg_series.dropna())} games from boxscores")

    vol_cv, vol_label = compute_volatility(stat_series)
    stat_skew = compute_skewness(stat_series)
    # [AUDIT FIX] Skewness heuristic when n<4: market-type prior so gate isn't fully bypassed
    if stat_skew is None and not stat_series.dropna().empty:
        if base_market in ("3PM", "Blocks", "Steals", "Stocks", "FTM", "FTA"):
            stat_skew = 0.70   # Count/rate stats dominated by zeros: strong right skew
        elif base_market in ("Rebounds", "RA", "PR"):
            stat_skew = 0.25   # Mildly right-skewed (occasional bigs blowups)
    rest_mult, rest_days = compute_rest_factor(gldf, game_date)

    # [UPGRADE 2] Projected minutes + DNP risk
    proj_minutes, dnp_risk = compute_projected_minutes(gldf_n, n_games=n_games)

    # Usage rate signal
    usage_rate = compute_usage_rate(gldf_n, n_games=n_games)

    # [UPGRADE 1] Explicit B2B flag (rest_days computed later but extract early)
    _rest_calc, _rest_d_early = compute_rest_factor(gldf, game_date)
    b2b_flag = (_rest_d_early == 0)

    # Resolve team/opponent
    team_abbr, opp_abbr, is_home_resolved = None, None, is_home
    if not gldf_n.empty:
        try:
            matchup = str(gldf_n.iloc[0].get("MATCHUP","")).strip()
            if " vs " in matchup.lower().replace("vs.","vs"):
                pts = re.split(r'\s+vs\.?\s+', matchup, flags=re.IGNORECASE)
                if len(pts)==2: team_abbr, opp_abbr = pts[0].strip(), pts[1].strip()
            elif " @ " in matchup:
                pts = matchup.split(" @ ")
                if len(pts)==2: team_abbr, opp_abbr = pts[0].strip(), pts[1].strip()
        except Exception: pass

    if team_abbr and not opp_abbr:
        try:
            opp_abbr, is_home_resolved = opponent_from_team_abbr(team_abbr, game_date)
        except Exception: pass

    if meta:
        try:
            home_abbr = map_team_name_to_abbr(meta.get("home_team","") or "")
            away_abbr = map_team_name_to_abbr(meta.get("away_team","") or "")
            if team_abbr and home_abbr and away_abbr:
                if team_abbr == home_abbr: opp_abbr, is_home_resolved = away_abbr, True
                elif team_abbr == away_abbr: opp_abbr, is_home_resolved = home_abbr, False
        except Exception: pass

    # ── Auto key_teammate_out from injury map ─────────────────────
    auto_inj_triggered = False
    auto_inj_player = None
    if not key_teammate_out and injury_team_map and team_abbr:
        team_key = str(team_abbr).upper()
        out_players = injury_team_map.get(team_key, [])
        player_lower = (player_name or "").lower()
        # Any OUT/DOUBTFUL teammate (exclude the player themselves)
        for op in out_players:
            if op and op != player_lower:
                key_teammate_out = True
                auto_inj_triggered = True
                auto_inj_player = op
                break

    ha_mult = compute_home_away_factor(gldf_n, base_market, is_home_resolved)
    # [UPGRADE 8] Injury boost scaled by usage capacity
    # Low-usage players absorb the most opportunity when a teammate goes out;
    # high-usage players already near their ceiling, so smaller boost.
    _usage_boost = 1.05
    if key_teammate_out and usage_rate is not None:
        if usage_rate >= 18:   _usage_boost = 1.03   # near ceiling, small room to expand
        elif usage_rate >= 12: _usage_boost = 1.05   # moderate usage, moderate upside
        else:                  _usage_boost = 1.08   # low baseline usage, highest opportunity gain
    ctx_mult = advanced_context_multiplier(player_name, base_market, opp_abbr, False)
    if key_teammate_out:
        ctx_mult *= _usage_boost
    blowout_prob = estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=None)

    # Blowout risk trims expected minutes: high-blowout games see starters sit early.
    # Cap at 12% reduction (blowout_prob=0.80 → ×0.88) to avoid over-penalizing.
    if proj_minutes is not None and blowout_prob is not None and blowout_prob > 0.20:
        blowout_min_adj = 1.0 - min(0.12, (float(blowout_prob) - 0.20) * 0.20)
        proj_minutes = proj_minutes * blowout_min_adj

    pos_str = get_player_position(player_name) or ""
    pos_bucket = get_position_bucket(pos_str)

    # Positional defensive grade multiplier
    pos_def_mult = positional_def_multiplier(opp_abbr, pos_bucket, base_market)

    # Pace-adjusted stat series
    pace_adj_series = compute_pace_adjusted_series(stat_series, opp_abbr)

    # [UPGRADE 3] Opponent-specific historical factor
    opp_specific_factor, n_vs_opp = compute_opp_specific_factor(gldf_n, opp_abbr, base_market)

    # [UPGRADE 5] Hot / cold regime (uses full season log for broader z-score)
    hot_cold_label, hot_cold_z = compute_player_regime_hot_cold(stat_series)

    # DD / TD: short-circuit to frequency-based probability
    if is_dd_td:
        prob_fn = compute_dd_prob if market_name == "Double Double" else compute_td_prob
        dd_prob = prob_fn(gldf_n, n_games=n_games)
        p_over_raw = dd_prob
        mu_raw = dd_prob
        sigma = None
        if p_over_raw is None:
            errors.append("Insufficient history for DD/TD probability.")
    else:
        # For half markets: convert line to equivalent full-game threshold
        effective_line = float(line) / half_factor if is_half_market and half_factor > 0 else float(line)
        # [UPGRADE 4] Pass market for stat-specific λ decay
        p_over_raw, mu_raw, sigma = bootstrap_prob_over(
            pace_adj_series, effective_line, cv_override=vol_cv, market=base_market
        )
        if p_over_raw is None:
            errors.append(f"Insufficient history (need >=4 games, have {len(stat_series.dropna())})")

    n_valid = int(stat_series.dropna().count())
    # [AUDIT FIX] Half-market with real boxscore data: scale positional prior by _orig_half_factor
    # so shrinkage operates in half-game units (not full-game units)
    if is_half_market and _orig_half_factor != half_factor and mu_raw is not None:
        orig_prior_val = POSITIONAL_PRIORS.get(pos_bucket, POSITIONAL_PRIORS["Unknown"]).get(base_market)
        if orig_prior_val is not None:
            k = max(2.0, 8.0 / (1.0 + math.log1p(max(n_valid, 1) / 5.0)))
            w_p = k / (k + max(n_valid, 1))
            mu_shrunk = float(w_p * orig_prior_val * _orig_half_factor + (1.0 - w_p) * mu_raw)
        else:
            mu_shrunk = mu_raw  # No prior for this market: use observed directly
    else:
        mu_shrunk = bayesian_shrink(mu_raw, n_valid, base_market, pos_bucket) if mu_raw is not None else None

    # Apply half factor and positional D to projection — include opp-specific factor
    proj_full = (mu_shrunk * ctx_mult * rest_mult * ha_mult * pos_def_mult * opp_specific_factor
                 if mu_shrunk is not None else None)
    proj = proj_full * half_factor if (proj_full is not None and is_half_market) else proj_full
    regime_label, regime_score = classify_regime(vol_cv, blowout_prob, ctx_mult)

    price_decimal = None
    try:
        if meta and meta.get("price") is not None:
            price_decimal = float(meta.get("price"))
    except Exception: pass
    p_implied = implied_prob_from_decimal(price_decimal)

    # Determine side early for skewness gate
    side_str = (meta.get("side") if meta else "Over") or "Over"

    # [UPGRADE 6] Over/Under asymmetry: model the correct side independently
    _is_under = "under" in str(side_str).lower()
    p_model = (1.0 - p_over_raw if _is_under and p_over_raw is not None else p_over_raw)
    sharp = book_sharpness(meta.get("book") if meta else None)
    w_model = float(market_prior_weight)
    w_eff = float(np.clip(w_model*(1.0-0.60*sharp)+0.15, 0.10, 0.95))
    if p_model is not None and p_implied is not None:
        p_raw = float(np.clip(w_eff*p_model + (1.0-w_eff)*p_implied, 1e-4, 1-1e-4))
    else:
        p_raw = p_model

    p_cal = p_raw

    ev_raw = ev_per_dollar(p_cal, price_decimal) if (p_cal is not None and price_decimal is not None) else None
    pen = volatility_penalty_factor(vol_cv)
    # [AUDIT FIX] Asymmetric vol penalty: adjust based on skew-side alignment
    # When skew favors our bet direction (tail points our way), volatility is less harmful → soften penalty.
    # When skew opposes us (tail points against us), volatility is more harmful → tighten penalty.
    if stat_skew is not None and vol_cv is not None and float(vol_cv) > 0.20:
        _sk = float(stat_skew)
        _is_under_pen = "under" in str(side_str).lower()
        _skew_helps = (_sk < -0.4 and _is_under_pen) or (_sk > 0.4 and not _is_under_pen)
        _skew_hurts = (_sk < -0.4 and not _is_under_pen) or (_sk > 0.4 and _is_under_pen)
        if _skew_helps:
            pen = min(1.0, pen * 1.12)   # Skew aligned with our side: soften penalty by up to 12%
        elif _skew_hurts:
            pen = pen * 0.88             # Skew opposed to our side: tighten penalty by 12%
    ev_adj = float(ev_raw * pen) if ev_raw is not None else None
    # [FIX 5] Pass skewness to volatility gate
    gate_ok, gate_reason = passes_volatility_gate(vol_cv, ev_raw, skew=stat_skew, bet_type=side_str)
    if exclude_chaotic and regime_label=="Chaotic":
        gate_ok, gate_reason = False, "chaotic regime (high volatility + blowout risk)"
    if not gate_ok: ev_adj = None

    stake_dollars, stake_frac, stake_reason = 0.0, 0.0, "gated"
    if gate_ok and p_cal is not None and price_decimal is not None and ev_adj is not None and ev_adj > 0:
        stake_dollars, stake_frac, stake_reason = recommended_stake(
            bankroll, float(p_cal), float(price_decimal), frac_kelly, max_risk_frac)

    mk_key = meta.get("market_key") if meta else ODDS_MARKETS.get(market_name,"")
    player_norm = normalize_name(player_name)
    mv_signal = get_line_movement_signal(player_norm, str(mk_key), float(line), side_str)

    sharp_div = {}
    if meta and meta.get("event_id"):
        try:
            sharp_div = sharp_divergence_alert(meta["event_id"], mk_key, player_norm, side_str, side_str) or {}
        except Exception: sharp_div = {}

    return {
        "player":           player_name,
        "player_norm":      player_norm,
        "player_id":        player_id,
        "market":           market_name,
        "line":             float(line),
        "proj":             float(proj) if proj is not None else None,
        "proj_vs_line":     float(proj - line) if proj is not None else None,
        "p_over":           float(p_raw) if p_raw is not None else None,
        "p_raw":            float(p_raw) if p_raw is not None else None,
        "p_model":          float(p_model) if p_model is not None else None,
        "p_cal":            float(p_cal) if p_cal is not None else None,
        "p_implied":        float(p_implied) if p_implied is not None else None,
        "advantage":        float(p_cal - p_implied) if (p_cal and p_implied) else None,
        "price_decimal":    float(price_decimal) if price_decimal is not None else None,
        "book":             meta.get("book") if meta else None,
        "event_id":         meta.get("event_id") if meta else None,
        "market_key":       meta.get("market_key") if meta else None,
        "side":             side_str,
        "commence_time":    meta.get("commence_time") if meta else None,
        "regime":           regime_label,
        "regime_score":     float(regime_score),
        "ev_raw":           float(ev_raw) if ev_raw is not None else None,
        "ev_adj":           float(ev_adj) if ev_adj is not None else None,
        "ev_pct":           float(ev_adj*100) if ev_adj is not None else None,
        "stake":            float(stake_dollars),
        "stake_frac":       float(stake_frac),
        "stake_reason":     stake_reason,
        "vol_penalty":      float(pen),
        "gate_ok":          bool(gate_ok),
        "gate_reason":      gate_reason,
        "edge":             float(ev_adj) if ev_adj is not None else None,
        "edge_cat":         classify_edge(ev_adj),
        "team":             team_abbr,
        "opp":              opp_abbr,
        "is_home":          is_home_resolved,
        "headshot":         nba_headshot_url(player_id),
        "blowout_prob":     float(blowout_prob),
        "context_mult":     float(ctx_mult),
        "rest_mult":        float(rest_mult),
        "rest_days":        int(rest_days),
        "ha_mult":          float(ha_mult),
        "volatility_cv":    vol_cv,
        "volatility_label": vol_label,
        "stat_skewness":    stat_skew,
        "position":         pos_str,
        "position_bucket":  pos_bucket,
        "n_games_used":     n_valid,
        "mu_raw":           float(mu_raw) if mu_raw is not None else None,
        "mu_shrunk":        float(mu_shrunk) if mu_shrunk is not None else None,
        "sigma":            float(sigma) if sigma is not None else None,
        "line_movement":    mv_signal,
        "sharp_div":        sharp_div,
        "usage_rate":       float(usage_rate) if usage_rate is not None else None,
        "pos_def_mult":     float(pos_def_mult),
        "half_factor":      float(half_factor),
        "pace_adj":         True if opp_abbr and TEAM_CTX.get(str(opp_abbr).upper()) else False,
        "auto_inj":          auto_inj_triggered,
        "auto_inj_player":   auto_inj_player,
        "key_teammate_out":  key_teammate_out,
        # [UPGRADE 1] Explicit B2B flag
        "b2b":               b2b_flag,
        # [UPGRADE 2] Projected minutes
        "proj_minutes":      float(proj_minutes) if proj_minutes is not None else None,
        "dnp_risk":          bool(dnp_risk),
        # [UPGRADE 3] Opponent-specific factor
        "opp_specific_factor": float(opp_specific_factor),
        "n_vs_opp":          int(n_vs_opp),
        # [UPGRADE 5] Hot/cold regime
        "hot_cold":          hot_cold_label,
        "hot_cold_z":        float(hot_cold_z),
        "errors":            errors,
    }

# ──────────────────────────────────────────────
# CALIBRATION ENGINE  [FIX 9: training range + OOD]
# ──────────────────────────────────────────────
def _expand_history_legs(history_df):
    if history_df is None or history_df.empty: return pd.DataFrame()
    rows = []
    for _, r in history_df.iterrows():
        bet_res = str(r.get("result","Pending"))
        # [FIX 11] Include PASS decisions for calibration analysis (no outcome)
        if bet_res not in ("HIT","MISS","PUSH","SKIP"): continue
        try:
            legs = json.loads(r.get("legs","[]")) if isinstance(r.get("legs"),str) else (r.get("legs") or [])
        except: legs = []
        # Per-leg results: if stored use them; otherwise fall back to parent result only for single-leg bets
        try:
            leg_results_list = json.loads(r.get("leg_results","[]")) if isinstance(r.get("leg_results"),str) else []
            if not isinstance(leg_results_list, list): leg_results_list = []
        except: leg_results_list = []
        n_legs = len(legs)
        for i, leg in enumerate(legs):
            if not isinstance(leg,dict): continue
            # Determine this leg's individual result
            if i < len(leg_results_list) and leg_results_list[i] in ("HIT","MISS","PUSH"):
                leg_res = leg_results_list[i]
            elif n_legs == 1 and bet_res in ("HIT","MISS","PUSH"):
                # Single-leg bet: parent result is the leg result
                leg_res = bet_res
            else:
                # Multi-leg without individual results logged — skip for calibration
                # to avoid incorrectly assigning the parlay outcome to all legs
                leg_res = "UNKNOWN"
            row = {
                "ts":r.get("ts"),"market":leg.get("market"),"player":leg.get("player"),
                "p_raw":safe_float(leg.get("p_raw") or leg.get("p_over"), default=np.nan),
                "price_decimal":safe_float(leg.get("price_decimal"), default=np.nan),
                "cv":safe_float(leg.get("volatility_cv"), default=np.nan),
                "ev_adj":safe_float(leg.get("ev_adj"), default=np.nan),
                "result":leg_res,
                "bet_result":bet_res,
                "decision":str(r.get("decision","BET")),
                "clv_line_fav":leg.get("clv_line_fav"),
                "clv_price_fav":leg.get("clv_price_fav"),
            }
            if leg_res in ("HIT","MISS","PUSH"):
                row["y"] = 1.0 if leg_res=="HIT" else 0.0
            elif bet_res == "SKIP":
                row["y"] = np.nan  # PASS has no outcome
            else:
                row["y"] = np.nan  # multi-leg leg with no individual result
            rows.append(row)
    df = pd.DataFrame(rows)
    return df[pd.to_numeric(df["p_raw"],errors="coerce").notna()].copy() if not df.empty else df

# [FIX 9] Store training range in calibrator
def fit_monotone_calibrator(df_legs, n_bins=12):
    if df_legs is None or df_legs.empty: return None
    # Only use settled legs (with outcomes) for fitting
    d = df_legs[df_legs["y"].notna()].copy()
    d = d[(d["p_raw"]>=0.01)&(d["p_raw"]<=0.99)]
    if len(d) < 80: return None
    d["bin"] = pd.cut(d["p_raw"], bins=n_bins, labels=False, include_lowest=True)
    g = d.groupby("bin",dropna=True).agg(p_mid=("p_raw","mean"),win=("y","mean"),n=("y","size")).reset_index()
    g = g[g["n"]>=5].sort_values("p_mid")
    if g.empty or len(g)<4: return None
    win_mono = np.maximum.accumulate(g["win"].values.astype(float))
    win_mono = np.clip(win_mono, 0.01, 0.99)
    return {
        "x":g["p_mid"].values.astype(float).tolist(),
        "y":win_mono.tolist(),
        "n":int(len(d)),
        "training_min": float(d["p_raw"].min()),
        "training_max": float(d["p_raw"].max()),
    }

# [FIX 9] OOD detection in calibrator
def apply_calibrator(p_raw, calib):
    if p_raw is None: return None
    try: p = float(p_raw)
    except: return None
    if calib is None: return float(np.clip(p, 0.0, 1.0))
    xs = calib.get("x") or []; ys = calib.get("y") or []
    if len(xs)<2 or len(xs)!=len(ys): return float(np.clip(p, 0.0, 1.0))
    try:
        result = float(np.clip(np.interp(p, xs, ys), 0.0, 1.0))
        # [FIX 9] Flag OOD but still return interpolated value
        t_min = calib.get("training_min", 0.0)
        t_max = calib.get("training_max", 1.0)
        if p < t_min * 0.85 or p > t_max * 1.15:
            # OOD: extrapolation warning (surfaced in UI)
            pass
        return result
    except: return float(np.clip(p, 0.0, 1.0))

def recompute_pricing_fields(leg, calib):
    p_raw = leg.get("p_raw")
    p_cal = apply_calibrator(p_raw, calib)
    leg["p_cal"] = p_cal
    price = leg.get("price_decimal")
    p_imp = implied_prob_from_decimal(price) if price is not None else None
    leg["p_implied"] = p_imp
    leg["advantage"] = float(p_cal-p_imp) if (p_cal and p_imp) else None
    ev_raw = ev_per_dollar(p_cal, price) if (p_cal and price) else None
    leg["ev_raw"] = ev_raw
    pen = volatility_penalty_factor(leg.get("volatility_cv"))
    leg["vol_penalty"] = pen
    side_str = leg.get("side", "Over") or "Over"
    gate_ok, gate_reason = passes_volatility_gate(
        leg.get("volatility_cv"), ev_raw,
        skew=leg.get("stat_skewness"), bet_type=side_str)
    regime_label, regime_score = classify_regime(leg.get("volatility_cv"),leg.get("blowout_prob"),leg.get("context_mult"))
    leg["regime"]=regime_label; leg["regime_score"]=float(regime_score)
    if bool(st.session_state.get("exclude_chaotic",True)) and regime_label=="Chaotic":
        gate_ok, gate_reason = False, "chaotic regime"
    leg["gate_ok"]=gate_ok; leg["gate_reason"]=gate_reason
    leg["ev_adj"] = (ev_raw*pen) if (ev_raw is not None and gate_ok) else None
    leg["ev_pct"] = float(leg["ev_adj"]*100) if leg["ev_adj"] is not None else None
    leg["edge"] = leg["ev_adj"]
    leg["edge_cat"] = classify_edge(leg["ev_adj"])
    bankroll = float(st.session_state.get("bankroll",0.0) or 0)
    frac_k = float(st.session_state.get("frac_kelly",0.25) or 0.25)
    cap_frac = float(st.session_state.get("max_risk_per_bet",5.0) or 5.0)/100.0
    if gate_ok and p_cal and price and leg.get("ev_adj") and float(leg["ev_adj"])>0 and bankroll>0:
        sd, sf, sr = recommended_stake(bankroll, float(p_cal), float(price), frac_k, cap_frac)
        leg["stake"]=float(sd); leg["stake_frac"]=float(sf); leg["stake_reason"]=sr
    else:
        leg["stake"]=float(leg.get("stake",0) or 0)
        leg["stake_frac"]=float(leg.get("stake_frac",0) or 0)
        leg["stake_reason"]=leg.get("stake_reason") or "gated"
    return leg

# ──────────────────────────────────────────────
# HISTORY PERSISTENCE  [FIX 15: no expiry]
# ──────────────────────────────────────────────
def user_state_path(uid): return f"user_state_{re.sub(r'[^a-zA-Z0-9_-]','_',uid or 'default')}.json"
def history_path(uid):    return f"history_{re.sub(r'[^a-zA-Z0-9_-]','_',uid or 'default')}.csv"

# [FIX 15] No TTL on user data - persists forever on disk
def load_user_state(uid):
    fp = user_state_path(uid)
    try:
        if os.path.exists(fp):
            with open(fp) as f: return json.load(f) or {}
    except Exception: pass
    return {}

def save_user_state(uid, state):
    try:
        with open(user_state_path(uid),"w") as f: json.dump(state or {}, f)
    except Exception: pass

def load_history(uid):
    fp = history_path(uid)
    try:
        if os.path.exists(fp): return pd.read_csv(fp)
    except Exception: pass
    return pd.DataFrame()

def append_history(uid, row):
    df = load_history(uid)
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(history_path(uid), index=False)

# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="NBA QUANT ENGINE",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── FONTS + GLOBAL STYLES ───────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;600;700&family=Fira+Code:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#070B10;--bg2:#0D1117;--bg3:#111820;--panel:#0F1620;--border:#1E2D3D;
  --green:#00FFB2;--green-dim:#00C88A;--blue:#00AAFF;--red:#FF3358;--amber:#FFB800;
  --muted:#4A607A;--text:#C8D8E8;--text-hi:#EEF4FF;
  --font-head:'Chakra Petch',monospace;--font-mono:'Fira Code',monospace;
}
.stApp{background:var(--bg)!important;font-family:var(--font-mono)!important;color:var(--text)!important;}
.block-container{padding-top:1.2rem!important;max-width:1400px!important;}
section[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;font-family:var(--font-mono)!important;}
h1,h2,h3{font-family:var(--font-head)!important;color:var(--text-hi)!important;letter-spacing:0.04em;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;border-bottom:1px solid var(--border)!important;gap:0px;}
.stTabs [data-baseweb="tab"]{font-family:var(--font-head)!important;font-size:0.72rem!important;letter-spacing:0.08em;color:var(--muted)!important;padding:0.6rem 1.2rem!important;border-bottom:2px solid transparent!important;text-transform:uppercase;}
.stTabs [aria-selected="true"]{color:var(--green)!important;border-bottom-color:var(--green)!important;background:transparent!important;}
.stButton>button{background:transparent!important;border:1px solid var(--green)!important;color:var(--green)!important;font-family:var(--font-head)!important;font-size:0.75rem!important;letter-spacing:0.10em;text-transform:uppercase;padding:0.5rem 1.4rem!important;transition:all 0.2s;border-radius:2px!important;}
.stButton>button:hover{background:var(--green)!important;color:var(--bg)!important;box-shadow:0 0 18px rgba(0,255,178,0.35)!important;}
.stTextInput input,.stNumberInput input,.stSelectbox select{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--text-hi)!important;font-family:var(--font-mono)!important;border-radius:2px!important;}
.stDataFrame{background:var(--panel)!important;border:1px solid var(--border)!important;}
.stDataFrame thead th{background:var(--bg3)!important;color:var(--green)!important;font-family:var(--font-head)!important;font-size:0.68rem!important;text-transform:uppercase;}
.stDataFrame td{font-family:var(--font-mono)!important;font-size:0.72rem!important;color:var(--text)!important;}
[data-testid="stMetric"]{background:var(--panel)!important;border:1px solid var(--border)!important;border-left:3px solid var(--green)!important;padding:0.8rem 1rem!important;border-radius:2px!important;}
[data-testid="stMetricLabel"]{font-family:var(--font-head)!important;font-size:0.65rem!important;letter-spacing:0.10em;text-transform:uppercase;color:var(--muted)!important;}
[data-testid="stMetricValue"]{font-family:var(--font-mono)!important;font-size:1.4rem!important;color:var(--text-hi)!important;}
.stAlert{background:var(--panel)!important;border:1px solid var(--border)!important;font-family:var(--font-mono)!important;font-size:0.74rem!important;}
::-webkit-scrollbar{width:5px;height:5px;}::-webkit-scrollbar-track{background:var(--bg2);}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.stApp::after{content:"";position:fixed;top:0;left:0;right:0;bottom:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);pointer-events:none;z-index:9999;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:1.5rem;padding:0.8rem 0 1.2rem;border-bottom:1px solid #1E2D3D;margin-bottom:1.2rem;">
  <div style="font-family:'Chakra Petch',monospace;font-size:1.7rem;font-weight:700;color:#00FFB2;letter-spacing:0.06em;">
    NBA QUANT ENGINE <span style="font-size:0.75rem;color:#4A607A;vertical-align:middle;margin-left:0.5rem;">v4.0</span>
  </div>
  <div style="flex:1;height:1px;background:linear-gradient(90deg,#00FFB2,transparent);"></div>
  <div style="font-family:'Fira Code',monospace;font-size:0.65rem;color:#4A607A;text-align:right;">
    BOOTSTRAP &middot; BAYESIAN &middot; KELLY &middot; LIVE ODDS
  </div>
</div>
""", unsafe_allow_html=True)

# ─── CARD HELPERS ─────────────────────────────────────────────
def make_card(content_html, border_color="#1E2D3D", glow=False):
    glow_css = f"box-shadow:0 0 20px {border_color}40;" if glow else ""
    return f"""<div style="background:#0F1620;border:1px solid {border_color};border-radius:3px;padding:1.1rem 1.2rem;margin-bottom:0.8rem;{glow_css}font-family:'Fira Code',monospace;">{content_html}</div>"""

def color_for_edge(cat):
    if cat == "Strong Edge": return "#00FFB2"
    if cat == "Solid Edge":  return "#00AAFF"
    if cat == "Lean Edge":   return "#FFB800"
    return "#4A607A"

def prob_bar_html(p, line_pct=0.50, label=""):
    if p is None: return "<span style='color:#4A607A;font-size:0.72rem;'>--</span>"
    pct = int(round(p * 100))
    color = "#00FFB2" if p > 0.57 else ("#FFB800" if p > 0.52 else "#FF3358")
    return f"""<div style="margin:0.35rem 0;"><div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#4A607A;margin-bottom:2px;"><span>{label}</span><span style="color:{color};font-weight:600;">{pct}%</span></div><div style="background:#111820;border-radius:1px;height:6px;overflow:hidden;"><div style="width:{pct}%;height:100%;background:{color};border-radius:1px;transition:width 0.4s;"></div></div></div>"""

def regime_badge(label):
    colors = {"Stable":"#00FFB2","Mixed":"#FFB800","Chaotic":"#FF3358"}
    c = colors.get(label, "#4A607A")
    return f"<span style='background:{c}18;border:1px solid {c};color:{c};padding:1px 7px;border-radius:1px;font-size:0.60rem;letter-spacing:0.08em;font-family:Chakra Petch,monospace;'>{label.upper()}</span>"

def hot_cold_badge(label):
    colors = {"Hot": "#FF6B35", "Cold": "#00AAFF", "Average": "#4A607A"}
    c = colors.get(label, "#4A607A")
    return f"<span style='background:{c}18;border:1px solid {c};color:{c};padding:1px 7px;border-radius:1px;font-size:0.60rem;letter-spacing:0.08em;font-family:Chakra Petch,monospace;'>{label.upper()}</span>"

def confidence_tier_color(p_cal):
    """Return border color based on calibrated probability confidence tier."""
    if p_cal is None: return "#1E2D3D"
    p = float(p_cal)
    if p >= 0.65: return "#00FFB2"   # Green — high confidence
    if p >= 0.58: return "#00AAFF"   # Blue — solid
    if p >= 0.52: return "#FFB800"   # Amber — moderate
    return "#FF3358"                  # Red — marginal

def mv_badge(mv):
    if not mv or abs(mv.get("pips",0)) < 0.25: return ""
    pips = mv.get("pips",0)
    if mv.get("steam"): col,icon = "#00FFB2","STEAM"
    elif mv.get("fade"): col,icon = "#FF3358","FADE"
    else: col,icon = "#FFB800","MOVE"
    arrow = "UP" if pips > 0 else "DN"
    return f"<span style='color:{col};font-size:0.65rem;'>{icon} {arrow} {abs(pips):.1f}</span>"

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;border-bottom:1px solid #1E2D3D;padding-bottom:0.5rem;'>CONTROL PANEL</div>""", unsafe_allow_html=True)
    user_id = st.text_input("Personal ID", value=st.session_state.get("user_id","trader"))
    st.session_state["user_id"] = user_id
    _active = st.session_state.get("_active_user_id")
    if _active != user_id:
        state = load_user_state(user_id)
        st.session_state["bankroll"] = safe_float(state.get("bankroll"), default=st.session_state.get("bankroll",1000.0))
        st.session_state["_active_user_id"] = user_id
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=float(st.session_state.get("bankroll",1000.0)), step=50.0)
    st.session_state["bankroll"] = float(bankroll)
    _lb = st.session_state.get("_last_saved_bankroll")
    if _lb is None or float(_lb) != float(bankroll):
        state = load_user_state(user_id); state["bankroll"]=float(bankroll)
        save_user_state(user_id, state); st.session_state["_last_saved_bankroll"]=float(bankroll)
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
    payout_multi = st.number_input("Multi-Leg Payout (x)", min_value=1.0, value=float(st.session_state.get("payout_multi",3.0)), step=0.1)
    st.session_state["payout_multi"] = payout_multi
    frac_kelly = st.slider("Fractional Kelly", 0.0, 1.0, float(st.session_state.get("frac_kelly",0.25)), 0.05)
    st.session_state["frac_kelly"] = frac_kelly
    market_prior_weight = st.slider("Model Weight (vs Market)", 0.0, 1.0, float(st.session_state.get("market_prior_weight",0.65)), 0.05, help="1.0 = pure model; 0.0 = pure market implied prob")
    st.session_state["market_prior_weight"] = float(market_prior_weight)
    max_risk_per_bet = st.slider("Max Bet Size (% BR)", 0.0, 10.0, float(st.session_state.get("max_risk_per_bet",3.0)), 0.5)
    st.session_state["max_risk_per_bet"] = float(max_risk_per_bet)
    n_games = st.slider("Sample Window (games)", 5, 30, int(st.session_state.get("n_games",10)))
    st.session_state["n_games"] = n_games
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
    exclude_chaotic = st.checkbox("Block Chaotic Regime", value=bool(st.session_state.get("exclude_chaotic",True)), help="Filters high-CV / blowout-risk environments")
    st.session_state["exclude_chaotic"] = bool(exclude_chaotic)
    max_daily_loss = st.slider("Daily Loss Stop (%)", 0, 50, int(st.session_state.get("max_daily_loss",15)))
    max_weekly_loss = st.slider("Weekly Loss Stop (%)", 0, 50, int(st.session_state.get("max_weekly_loss",25)))
    st.session_state["max_daily_loss"] = max_daily_loss
    st.session_state["max_weekly_loss"] = max_weekly_loss
    with st.expander("Odds API", expanded=False):
        scan_book_override = st.text_input("Book override (blank=auto)", value="")
        max_req_day = st.number_input("Max requests/day", 1, 500, int(st.session_state.get("max_req_day",100)), 10)
        st.session_state["max_req_day"] = int(max_req_day)

    # [UPGRADE 24] Always-visible quota tracker
    hdr = st.session_state.get("_odds_headers_last", {})
    rem  = hdr.get("remaining", "?")
    used = hdr.get("used", "?")
    try:
        rem_int = int(rem)
        rem_color = "#FF3358" if rem_int < 10 else ("#FFB800" if rem_int < 30 else "#00FFB2")
        rem_label = f"<span style='color:{rem_color};font-weight:600;'>{rem}</span>"
    except Exception:
        rem_label = f"<span style='color:#4A607A;'>{rem}</span>"
    st.markdown(
        f"<div style='font-family:Fira Code,monospace;font-size:0.62rem;color:#4A607A;margin-top:0.4rem;'>"
        f"API QUOTA: used {used} | rem {rem_label}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if hdr.get("remaining") == "0":
        st.error("Odds API quota exhausted — all fetches paused.")

    # [UPGRADE 22] Watchlist quick view
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
    with st.expander("Watchlist", expanded=False):
        wl = load_watchlist(st.session_state.get("user_id","trader"))
        wl_input = st.text_input("Add player", placeholder="LeBron James", key="wl_add")
        wl_col1, wl_col2 = st.columns(2)
        if wl_col1.button("Add", key="wl_add_btn"):
            if wl_input.strip() and wl_input.strip() not in wl:
                wl.append(wl_input.strip())
                save_watchlist(st.session_state.get("user_id","trader"), wl)
                st.rerun()
        if wl:
            rm = st.selectbox("Remove", ["--"] + wl, key="wl_rm")
            if wl_col2.button("Remove", key="wl_rm_btn") and rm != "--":
                wl = [p for p in wl if p != rm]
                save_watchlist(st.session_state.get("user_id","trader"), wl)
                st.rerun()
            for p in wl:
                st.markdown(f"<div style='font-size:0.68rem;color:#00AAFF;'>· {p}</div>", unsafe_allow_html=True)
        else:
            st.caption("No players on watchlist.")

# ─── SESSION STATE INIT ────────────────────────────────────────
for k in ["last_results","calibrator_map","scanner_offers","scanner_results"]:
    if k not in st.session_state: st.session_state[k] = None if k != "last_results" else []
MARKET_OPTIONS = list(ODDS_MARKETS.keys())

def _daily_pnl(uid):
    h = load_history(uid)
    if h.empty: return 0.0
    try:
        h["ts_d"] = pd.to_datetime(h["ts"],errors="coerce").dt.date
        today_rows = h[h["ts_d"]==date.today()].copy()
        if today_rows.empty: return 0.0
        hits = (today_rows["result"]=="HIT").sum(); miss = (today_rows["result"]=="MISS").sum()
        return float(hits - miss)
    except: return 0.0

def _check_loss_stops(uid, bankroll):
    pnl = _daily_pnl(uid)
    if bankroll > 0 and pnl < 0 and abs(pnl)/bankroll*100 > float(st.session_state.get("max_daily_loss",15)):
        st.error(f"DAILY LOSS STOP HIT ({abs(pnl)/bankroll*100:.1f}%). No new bets recommended today.")
        return True
    return False

# ─── TABS ─────────────────────────────────────────────────────
tabs = st.tabs(["MODEL", "RESULTS", "LIVE SCANNER", "PLATFORMS", "HISTORY", "CALIBRATION", "INSIGHTS", "ALERTS"])

with tabs[0]:
    _check_loss_stops(user_id, bankroll)
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>CONFIGURE UP TO 4 LEGS</div>""", unsafe_allow_html=True)
    date_col, book_col = st.columns([2,2])
    with date_col:
        scan_date = st.date_input("Lines Date", value=date.today(), key="model_date")
    with book_col:
        book_choices, book_err = get_sportsbook_choices(scan_date.isoformat())
        if book_err: st.caption(book_err)
        sportsbook = st.selectbox("Sportsbook", options=book_choices, index=0)
    leg_configs = []
    for row_idx in range(2):
        cols = st.columns(2)
        for col_idx in range(2):
            leg_n = row_idx * 2 + col_idx + 1
            tag = f"P{leg_n}"
            with cols[col_idx]:
                st.markdown(f"<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;'>LEG {leg_n}</div>", unsafe_allow_html=True)
                pname = st.text_input(f"Player", key=f"pname_{leg_n}", placeholder="e.g. LeBron James")
                mkt = st.selectbox(f"Market", options=MARKET_OPTIONS, key=f"mkt_{leg_n}")
                manual = st.checkbox(f"Manual line", key=f"manual_{leg_n}")
                mline = st.number_input(f"Line", min_value=0.0, value=float(st.session_state.get(f"line_{leg_n}",22.5)), step=0.5, key=f"mline_{leg_n}")
                out_cb = st.checkbox(f"Key teammate OUT?", key=f"out_{leg_n}")
                leg_configs.append((tag, pname, mkt, manual, mline, out_cb))
    run_btn = st.button("RUN MODEL", use_container_width=True)
    if run_btn:
        results = []; warnings = []; tasks = []
        for (tag, pname, mkt, manual, mline, teammate_out) in leg_configs:
            pname = (pname or "").strip()
            if not pname: continue
            market_key = ODDS_MARKETS.get(mkt)
            if not market_key: warnings.append(f"{tag}: unsupported market {mkt}"); continue
            line = mline; meta = None
            if not manual:
                val, m_meta, ferr = find_player_line_from_events(pname, market_key, scan_date.isoformat(), sportsbook)
                if val is not None:
                    line = float(val); meta = m_meta
                    st.success(f"{tag} - {pname} {mkt}: line {line:.1f} ({sportsbook})")
                else:
                    st.warning(f"{tag} auto-line failed ({ferr}). Using manual {line:.1f}.")
            if not line or float(line) <= 0:
                warnings.append(f"{tag}: invalid line"); continue
            tasks.append((tag, pname, mkt, float(line), meta, bool(teammate_out)))
        if tasks:
            _inj_map = st.session_state.get("injury_team_map", {})
            with st.spinner("Computing projections..."):
                with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as ex:
                    futs = [ex.submit(compute_leg_projection, pname, mkt, line, meta,
                                      n_games=n_games, key_teammate_out=to,
                                      bankroll=bankroll, frac_kelly=frac_kelly,
                                      max_risk_frac=float(st.session_state.get("max_risk_per_bet",3.0))/100.0,
                                      market_prior_weight=market_prior_weight,
                                      exclude_chaotic=bool(exclude_chaotic),
                                      game_date=scan_date,
                                      injury_team_map=_inj_map)
                            for (tag, pname, mkt, line, meta, to) in tasks]
                    results = []
                    for f in futs:
                        try:
                            results.append(f.result())
                        except Exception as _te:
                            results.append({"player": "Error", "market": "?", "line": 0.0,
                                            "errors": [f"thread error: {type(_te).__name__}: {_te}"],
                                            "gate_ok": False, "gate_reason": "thread error"})
            calib = st.session_state.get("calibrator_map")
            results = [recompute_pricing_fields(dict(leg), calib) for leg in results]
            st.session_state["last_results"] = results
            if warnings:
                for w in warnings: st.warning(w)

    # [FIX 11] Log ALL evaluations (BET and PASS) for calibration
    st.markdown("<hr style='border-color:#1E2D3D;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#4A607A;letter-spacing:0.12em;'>LOG THIS SLATE</span>", unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1: placed = st.radio("Did you place?", ["No","Yes"], horizontal=True, index=0)
    with c2:
        if st.button("Confirm Log"):
            res = st.session_state.get("last_results") or []
            if not res:
                st.warning("Run model first.")
            else:
                decision = "BET" if placed == "Yes" else "PASS"
                result_val = "Pending" if placed == "Yes" else "SKIP"
                append_history(user_id, {
                    "ts":_now_iso(),"user_id":user_id,
                    "legs":json.dumps(res),"n_legs":len(res),
                    "leg_results":json.dumps(["Pending"]*len(res)),
                    "result":result_val,"decision":decision,"notes":""
                })
                st.success(f"Logged ({decision})")

with tabs[1]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>PROJECTION RESULTS & EDGE ANALYSIS</div>""", unsafe_allow_html=True)
    res = st.session_state.get("last_results") or []
    if not res:
        st.markdown(make_card("<span style='color:#4A607A;font-size:0.8rem;'>Run the model to see projections.</span>"), unsafe_allow_html=True)
    else:
        n_gated = sum(1 for l in res if l.get("gate_ok"))
        n_edge  = sum(1 for l in res if (l.get("ev_adj") or 0) > 0.01)
        total_stake = sum(l.get("stake",0) for l in res)
        m1c, m2c, m3c, m4c = st.columns(4)
        m1c.metric("Legs Analyzed", len(res))
        m2c.metric("Passed Gate", n_gated)
        m3c.metric("Positive EV", n_edge)
        m4c.metric("Total Rec. Stake", f"${total_stake:.2f}")
        st.markdown("")
        cols = st.columns(min(4, len(res)))
        for i, leg in enumerate(res):
            c = cols[i % len(cols)]
            with c:
                ec = color_for_edge(leg.get("edge_cat"))
                ev_pct = leg.get("ev_pct")
                ev_str = f"{ev_pct:+.1f}%" if ev_pct is not None else "--"
                proj_disp = f"{leg['proj']:.1f}" if leg.get("proj") is not None else "--"
                p_cal_v = leg.get("p_cal") or leg.get("p_over")
                n_used = leg.get("n_games_used",0)
                rest_d = leg.get("rest_days",2)
                rest_tag = "B2B" if rest_d==0 else f"{rest_d}d rest"
                mv = leg.get("line_movement") or {}
                mv_html = mv_badge(mv)
                sharp = leg.get("sharp_div") or {}
                sharp_html = ""
                if sharp.get("fade_model"):
                    sharp_html = "<div style='color:#FF3358;font-size:0.62rem;'>SHARP FADE</div>"
                elif sharp.get("confirm") == True:
                    sharp_html = "<div style='color:#00FFB2;font-size:0.62rem;'>SHARP CONFIRM</div>"
                hs = leg.get("headshot","")
                hs_html = f"<img src='{hs}' style='width:52px;height:38px;object-fit:cover;border-radius:2px;border:1px solid #1E2D3D;float:right;'>" if hs else ""
                card_html = f"""
<div style='margin-bottom:0.5rem;'>
  {hs_html}
  <div style='font-family:Chakra Petch,monospace;font-size:0.82rem;font-weight:700;color:#EEF4FF;'>{leg["player"]}</div>
  <div style='font-size:0.65rem;color:#4A607A;letter-spacing:0.08em;'>{leg.get("team","??")} vs {leg.get("opp","??")}</div>
  <div style='clear:both;'></div>
</div>
<div style='font-size:0.70rem;color:#4A607A;margin:0.15rem 0;text-transform:uppercase;letter-spacing:0.06em;'>{leg["market"]} | {rest_tag} | {leg.get("position_bucket","?")}</div>
<div style='display:flex;justify-content:space-between;margin:0.6rem 0;'>
  <div style='text-align:center;'><div style='font-size:0.60rem;color:#4A607A;'>LINE</div><div style='font-family:Fira Code,monospace;font-size:1.1rem;color:#EEF4FF;font-weight:500;'>{leg["line"]:.1f}</div></div>
  <div style='text-align:center;'><div style='font-size:0.60rem;color:#4A607A;'>PROJ</div><div style='font-family:Fira Code,monospace;font-size:1.1rem;color:#00AAFF;font-weight:500;'>{proj_disp}</div></div>
  <div style='text-align:center;'><div style='font-size:0.60rem;color:#4A607A;'>EV</div><div style='font-family:Fira Code,monospace;font-size:1.1rem;color:{ec};font-weight:600;'>{ev_str}</div></div>
</div>
{prob_bar_html(p_cal_v, label="P(OVER)")}
{prob_bar_html(leg.get("p_implied"), label="IMPLIED")}
<div style='margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;'>
  {regime_badge(leg.get("regime","?"))}
  {hot_cold_badge(leg.get("hot_cold","Average"))}
  {mv_html}
</div>
{sharp_html}
{"<div style='color:#FFA500;font-size:0.62rem;'>🏥 AUTO TEAMMATE OUT: " + (leg.get("auto_inj_player") or "").title() + "</div>" if leg.get("auto_inj") else ""}
<div style='margin-top:0.7rem;font-size:0.64rem;color:#4A607A;'>
  ctx x{leg.get("context_mult",1):.3f} | rest x{leg.get("rest_mult",1):.2f} | ha x{leg.get("ha_mult",1):.2f}<br>
  CV={f"{leg['volatility_cv']:.2f}" if leg.get("volatility_cv") else "--"} | N={n_used} games<br>
  Shrunk mu: {f"{leg['mu_shrunk']:.1f}" if leg.get("mu_shrunk") else "--"}
</div>"""
                stake = safe_float(leg.get("stake"))
                if stake > 0:
                    card_html += f"<div style='margin-top:0.6rem;background:#00FFB218;border:1px solid #00FFB230;border-radius:2px;padding:0.4rem 0.6rem;font-size:0.72rem;color:#00FFB2;font-family:Fira Code,monospace;'>REC STAKE: ${stake:.2f} ({leg.get('stake_frac',0)*100:.1f}% BR)</div>"
                elif not leg.get("gate_ok"):
                    card_html += f"<div style='margin-top:0.6rem;background:#FF335818;border:1px solid #FF335830;border-radius:2px;padding:0.4rem 0.6rem;font-size:0.65rem;color:#FF3358;'>GATED: {leg.get('gate_reason','')}</div>"
                if leg.get("errors"):
                    card_html += "<div style='margin-top:0.4rem;font-size:0.60rem;color:#FFB800;'>" + "<br>".join(leg["errors"][:2]) + "</div>"
                # [UPGRADE 20] Confidence-tier border overrides edge color
                tier_border = confidence_tier_color(leg.get("p_cal"))
                st.markdown(make_card(card_html, border_color=tier_border, glow=(leg.get("edge_cat") in ["Strong Edge","Solid Edge"])), unsafe_allow_html=True)

                # [UPGRADE 19] Player card expander with per-game bar chart
                with st.expander(f"Drill-down: {leg['player']}", expanded=False):
                    pid = leg.get("player_id")
                    _mkt_exp = leg.get("market", "Points")
                    if pid:
                        gl_exp, _ = fetch_player_gamelog(player_id=pid, max_games=10)
                        if not gl_exp.empty:
                            s_exp = compute_stat_from_gamelog(gl_exp, _mkt_exp.replace("H1 ","").replace("H2 ",""))
                            s_exp = pd.to_numeric(s_exp, errors="coerce").dropna().reset_index(drop=True)
                            if not s_exp.empty:
                                chart_df = pd.DataFrame({
                                    "Game": [f"G-{i+1}" for i in range(len(s_exp))],
                                    "Actual": s_exp.values,
                                    "Line":   [float(leg.get("line", 0))] * len(s_exp),
                                })
                                chart_df = chart_df.set_index("Game")
                                st.caption(f"Last {len(s_exp)} games — {_mkt_exp}")
                                st.bar_chart(chart_df, use_container_width=True, height=180)
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Proj Minutes",
                              f"{leg.get('proj_minutes'):.0f}" if leg.get("proj_minutes") else "--",
                              delta="DNP risk" if leg.get("dnp_risk") else None,
                              delta_color="inverse")
                    d2.metric("vs Opponent", f"{leg.get('opp_specific_factor',1.0):.3f}x",
                              help=f"Based on {leg.get('n_vs_opp',0)} prior meetings")
                    d3.metric("Regime", leg.get("hot_cold","Average"),
                              delta=f"z={leg.get('hot_cold_z',0.0):+.2f}")
                    sigma = leg.get("sigma")
                    proj  = leg.get("proj")
                    if sigma and proj:
                        st.caption(f"Confidence band: {proj:.1f} ± {sigma:.1f} (μ±σ) — "
                                   f"{max(0,proj-sigma):.1f} to {proj+sigma:.1f}")

        # Multi-leg combo
        if len(res) >= 2:
            st.markdown("<hr style='border-color:#1E2D3D;margin:1rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.72rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.8rem;'>MULTI-LEG JOINT MONTE CARLO (CORRELATED)</div>", unsafe_allow_html=True)
            try:
                from scipy.stats import norm as _norm_mc
                valid_legs = [l for l in res if l.get("gate_ok") and l.get("p_cal") is not None]
                if len(valid_legs) < 2:
                    st.caption("Need >=2 gated legs for combo.")
                else:
                    n = len(valid_legs)
                    probs = np.array([float(l["p_cal"]) for l in valid_legs])
                    corr_mat = np.eye(n)
                    for i in range(n):
                        for j in range(i+1, n):
                            c = estimate_player_correlation(valid_legs[i], valid_legs[j])
                            corr_mat[i,j] = corr_mat[j,i] = c
                    evals, evecs = np.linalg.eigh(corr_mat)
                    evals = np.clip(evals, 1e-6, None)
                    corr_psd = evecs @ np.diag(evals) @ evecs.T
                    rng2 = np.random.default_rng(99)
                    z = rng2.multivariate_normal(np.zeros(n), corr_psd, 10000)
                    u = _norm_mc.cdf(z)
                    joint = float((u < probs).all(axis=1).mean())
                    payout_mult = float(st.session_state.get("payout_multi",3.0))
                    ev_combo = payout_mult * joint - 1.0
                    naive = float(np.prod(probs))
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Joint Hit Prob (MC)", f"{joint*100:.1f}%")
                    mc2.metric("Naive (uncorr)", f"{naive*100:.1f}%")
                    mc3.metric(f"Combo EV (x{payout_mult})", f"{ev_combo*100:+.1f}%")
            except ImportError:
                st.caption("scipy not available - joint MC skipped.")
            except Exception as e:
                st.caption(f"Joint MC error: {type(e).__name__}: {e}")

        with st.expander("Raw Data Table", expanded=False):
            display_cols = ["player","market","line","proj","p_cal","p_implied","advantage",
                            "ev_pct","edge_cat","gate_ok","stake","volatility_label","volatility_cv",
                            "regime","rest_days","position_bucket","context_mult","n_games_used",
                            "usage_rate","pos_def_mult","half_factor","pace_adj"]
            disp_df = pd.DataFrame([{k:l.get(k) for k in display_cols} for l in res])
            st.dataframe(disp_df, use_container_width=True)

        # ── Parlay Optimizer ──────────────────────────────────
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>PARLAY OPTIMIZER</div>", unsafe_allow_html=True)
        po_col1, po_col2 = st.columns(2)
        with po_col1:
            po_max_legs = st.slider("Max combo legs", 2, 4, 3, key="po_max_legs")
        with po_col2:
            po_payout = st.number_input("Payout multiplier (x)", 1.5, 20.0, float(st.session_state.get("payout_multi",3.0)), 0.5, key="po_payout")
        if st.button("Optimize Parlay Combos", use_container_width=True):
            combos = kelly_parlay_optimizer(res, po_payout, max_legs=po_max_legs, bankroll=bankroll, frac_kelly=frac_kelly)
            if combos:
                st.dataframe(pd.DataFrame(combos), use_container_width=True)
            else:
                st.warning("Need 2+ gated legs with P > 50% to generate combos.")

        # ── Monte Carlo Simulation ────────────────────────────
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>MONTE CARLO GAME SIMULATION</div>", unsafe_allow_html=True)
        if st.button("Run MC Simulation (all legs)", use_container_width=True):
            mc = monte_carlo_game_sim(res, n_sims=20000, payout_mult=float(po_payout))
            if mc and "error" not in mc:
                mc_c1, mc_c2, mc_c3 = st.columns(3)
                mc_c1.metric("Joint Hit Prob (MC)", f"{mc['joint_prob_%']:.2f}%")
                mc_c2.metric("Naive (uncorr)", f"{mc['naive_joint_%']:.2f}%")
                mc_c3.metric(f"Combo EV (x{po_payout:.1f})", f"{mc['ev_%']:+.2f}%")
                if mc.get("per_leg_sim_%"):
                    st.caption("Per-leg simulated hit rates: " + " | ".join(
                        f"{res[i].get('player','?')} {p}%" for i, p in enumerate(mc["per_leg_sim_%"]) if i < len(res)
                    ))
            elif mc and "error" in mc:
                st.warning(f"MC error: {mc['error']}")

# ─── LIVE SCANNER TAB [FIX 13: persistent] [FIX 14: week-ahead] ───
with tabs[2]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>LIVE SCANNER - SWEEP ALL PLAYER PROPS FOR EDGES</div>""", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([2,2,2])
    with sc1:
        # [FIX 14] Week-ahead: date range selection
        scan_start = st.date_input("Start Date", value=date.today(), key="scan_start")
        scan_days = st.slider("Days ahead", 0, 7, 0, key="scan_days_ahead")
        scan_end = scan_start + timedelta(days=scan_days)
        if scan_days > 0:
            st.caption(f"Scanning {scan_start.isoformat()} to {scan_end.isoformat()}")
    with sc2:
        markets_sel = st.multiselect("Markets", options=MARKET_OPTIONS, default=["Points","Rebounds","Assists"])
    with sc3:
        book_choices2, _ = get_sportsbook_choices(scan_start.isoformat())
        sportsbook2 = st.selectbox("Book", options=["all"]+book_choices2, index=0)
    sf1, sf2, sf3 = st.columns(3)
    with sf1: min_prob = st.slider("Min P(Over)", 0.50, 0.80, 0.57, 0.01)
    with sf2: min_adv  = st.slider("Min Advantage vs Implied", 0.00, 0.12, 0.02, 0.005)
    with sf3: min_ev   = st.slider("Min EV (adj)", -0.05, 0.25, 0.01, 0.005)
    max_rows = st.slider("Max Results", 10, 200, 60, 10)

    fetch_col, scan_col = st.columns(2)
    if fetch_col.button("Fetch Live Lines", use_container_width=True):
        selected_keys = list(dict.fromkeys(ODDS_MARKETS.get(m) for m in markets_sel if ODDS_MARKETS.get(m)))
        if not selected_keys:
            st.warning("Select at least one market.")
        else:
            # [FIX 14] Fetch across date range
            if scan_days > 0:
                evs, err = odds_get_events_range(scan_start.isoformat(), scan_end.isoformat())
            else:
                evs, err = odds_get_events(scan_start.isoformat())
            if err: st.error(err)
            elif not evs: st.warning("No events for that date range.")
            else:
                offers = []
                # [FIX H1/H2/ALT] Split market keys into standard and specialty batches
                # Odds API supports max ~6 markets per call reliably; specialty markets
                # (H1/H2/Q1/Alt/Fantasy) need separate calls as not all books offer them
                std_keys     = [k for k in selected_keys if k not in SPECIALTY_MARKET_KEYS]
                spec_keys    = [k for k in selected_keys if k in SPECIALTY_MARKET_KEYS]
                market_batches = []
                BATCH_SIZE = 6
                if std_keys:
                    for i in range(0, len(std_keys), BATCH_SIZE):
                        market_batches.append(std_keys[i:i+BATCH_SIZE])
                # Specialty markets individually (each book may only carry a subset)
                for sk in spec_keys:
                    market_batches.append([sk])

                for ev in evs:
                    eid = ev.get("id")
                    if not eid: continue
                    for batch_keys in market_batches:
                        # For specialty markets try all regions for better coverage
                        regions = "us,us2,eu,uk" if any(k in SPECIALTY_MARKET_KEYS for k in batch_keys) else REGION_US
                        odds, oerr = odds_get_event_odds(eid, tuple(batch_keys), regions=regions)
                        if oerr or not odds: continue
                        for m in markets_sel:
                            mk = ODDS_MARKETS.get(m)
                            if not mk or mk not in batch_keys: continue
                            bf = sportsbook2 if sportsbook2 != "all" else None
                            parsed, _ = _parse_player_prop_outcomes(odds, mk, book_filter=bf)
                            offers.extend([{**r,"market":m} for r in parsed])
                if offers:
                    # [FIX 13] Store in session state - persists across tab switches
                    st.session_state["scanner_offers"] = pd.DataFrame(offers)
                    # Auto-save to prop line history DB + [UPGRADE 10] opening line capture
                    for r2 in offers:
                        save_prop_line(r2.get("player",""), r2.get("market",""),
                                       r2.get("line"), r2.get("price"), r2.get("book"),
                                       event_id=r2.get("event_id"))
                        pn2 = normalize_name(r2.get("player",""))
                        mk2 = r2.get("market_key", ODDS_MARKETS.get(r2.get("market",""), ""))
                        side2 = r2.get("side", "Over")
                        save_opening_line(pn2, mk2, side2, r2.get("line", 0), r2.get("price"))
                    st.success(f"Fetched {len(offers)} raw prop outcomes — opening lines captured.")
                else:
                    st.warning("No offers returned.")

    # ── Bulk game log loader (recommended before large scans) ──────
    bulk_loaded = _fetch_bulk_gamelogs() is not None
    _bulk_label = "✓ All Game Logs Loaded" if bulk_loaded else "Load All Game Logs (Recommended)"
    if scan_col.button(_bulk_label, use_container_width=True, disabled=bulk_loaded):
        with st.spinner("Loading all NBA player game logs (one-time, cached 6h)..."):
            _fetch_bulk_gamelogs.clear()
            result = _fetch_bulk_gamelogs()
        if result is not None:
            st.success(f"Loaded {len(result):,} game log rows — scans are now near-instant.")
        else:
            st.warning("Bulk load failed — scanner will fall back to per-player fetches.")

    if scan_col.button("Run Scan", use_container_width=True):
        df = st.session_state.get("scanner_offers")
        if df is None or (hasattr(df, 'empty') and df.empty):
            st.warning("Fetch lines first.")
        else:
            df2 = df[df["side"].str.lower().isin(["over","o"])].copy()
            if df2.empty: df2 = df.copy()
            candidates = []
            for _, r in df2.iterrows():
                pname = r.get("player"); mkt = r.get("market"); line = r.get("line")
                if not pname or pd.isna(line) or not mkt: continue
                meta = {"event_id":r.get("event_id"),"home_team":r.get("home_team"),
                        "away_team":r.get("away_team"),"commence_time":r.get("commence_time"),
                        "price":r.get("price"),"book":r.get("book"),
                        "market_key":ODDS_MARKETS.get(mkt),"side":r.get("side","Over")}
                candidates.append((pname, mkt, float(line), meta))
            out_rows, dropped = [], []
            if candidates:
                _inj_map = st.session_state.get("injury_team_map", {})
                # Auto-load bulk game logs if not already cached (one-time, ~15-30s)
                bulk_ready = _fetch_bulk_gamelogs() is not None
                if not bulk_ready:
                    with st.spinner("Loading all NBA game logs (one-time ~20s)..."):
                        _fetch_bulk_gamelogs.clear()
                        bulk_ready = _fetch_bulk_gamelogs() is not None
                _scan_workers = 16 if bulk_ready else 6
                if not bulk_ready:
                    st.warning(
                        f"Bulk game log load failed — scanning with {_scan_workers} workers "
                        f"(individual NBA API calls). Click **Load All Game Logs** above for faster scans."
                    )
                with st.spinner(f"Scanning {len(candidates)} candidates ({_scan_workers} workers)..."):
                    with ThreadPoolExecutor(max_workers=_scan_workers) as ex:
                        futs = [ex.submit(compute_leg_projection, pname, mkt, line, meta,
                                          n_games=n_games, key_teammate_out=False,
                                          bankroll=bankroll, frac_kelly=frac_kelly,
                                          max_risk_frac=float(st.session_state.get("max_risk_per_bet",3.0))/100.0,
                                          market_prior_weight=market_prior_weight,
                                          exclude_chaotic=bool(exclude_chaotic),
                                          game_date=scan_start,
                                          injury_team_map=_inj_map)
                                for pname, mkt, line, meta in candidates]
                        for (pname, mkt, line, meta), fut in zip(candidates, futs):
                            try:
                                leg = fut.result()
                            except Exception as _te:
                                dropped.append({"player": pname, "market": mkt, "reason": f"thread error: {type(_te).__name__}: {_te}"})
                                continue
                            leg = recompute_pricing_fields(leg, st.session_state.get("calibrator_map"))
                            if not leg.get("gate_ok"):
                                dropped.append({"player":pname,"market":mkt,"reason":leg.get("gate_reason","gated")}); continue
                            pc = float(leg.get("p_cal") or leg.get("p_over") or 0)
                            pi = leg.get("p_implied")
                            ev = leg.get("ev_adj")
                            if pi is None or ev is None:
                                dropped.append({"player":pname,"market":mkt,"reason":"no price/EV"}); continue
                            adv = pc - float(pi)
                            if pc < min_prob: dropped.append({"player":pname,"market":mkt,"reason":f"p_cal<{min_prob:.2f}"}); continue
                            if adv < min_adv: dropped.append({"player":pname,"market":mkt,"reason":f"adv<{min_adv:.3f}"}); continue
                            if float(ev) < min_ev: dropped.append({"player":pname,"market":mkt,"reason":f"ev<{min_ev:.3f}"}); continue
                            mv = leg.get("line_movement") or {}
                            inj_flag = ("🏥 " + (leg.get("auto_inj_player") or "").title()
                                        if leg.get("auto_inj") else "")
                            out_rows.append({
                                "player":pname,"market":mkt,"line":line,
                                "p_cal":round(pc,3),"p_implied":round(float(pi),3),
                                "advantage":round(adv,3),"ev_adj_pct":round(float(ev)*100,2),
                                "proj":safe_round(leg.get("proj")),
                                "edge_cat":leg.get("edge_cat",""),"regime":leg.get("regime",""),
                                "hot_cold":leg.get("hot_cold","Average"),
                                "team":leg.get("team",""),"opp":leg.get("opp",""),
                                "b2b": "B2B" if leg.get("b2b") else "",
                                "dnp_risk": "DNP?" if leg.get("dnp_risk") else "",
                                "vol_cv":safe_round(leg.get("volatility_cv")),
                                "rest_d":leg.get("rest_days",2),
                                "line_mv":mv.get("direction","--"),
                                "mv_pips":mv.get("pips",0.0),
                                "steam": "STEAM" if mv.get("steam") else ("FADE" if mv.get("fade") else ""),
                                "stake_$":round(leg.get("stake",0),2),
                                "n_games":leg.get("n_games_used",0),
                                "inj_boost": inj_flag,
                                "min_proj": safe_round(leg.get("proj_minutes"),0),
                            })
            out_df = pd.DataFrame(out_rows)
            if not out_df.empty:
                out_df = out_df.sort_values("ev_adj_pct", ascending=False).head(max_rows)
                # [FIX 13] Persist scanner results in session state
                st.session_state["scanner_results"] = out_df
                st.session_state["scanner_dropped"] = dropped
                # Auto-send Discord/Telegram alerts for strong edges
                _dw = st.session_state.get("discord_webhook","")
                _tt = st.session_state.get("tg_token","")
                _tc = st.session_state.get("tg_chat","")
                _et = float(st.session_state.get("discord_ev_thresh", 5.0))
                if (_dw or (_tt and _tc)):
                    strong = [r for _, r in out_df.iterrows() if float(r.get("ev_adj_pct") or 0) >= _et]
                    for r in strong:
                        _msg = format_edge_alert(dict(r))
                        if _dw: send_discord_alert(_dw, _msg)
                        if _tt and _tc: send_telegram_alert(_tt, _tc, _msg)
                    if strong:
                        st.success(f"Auto-sent {len(strong)} alerts.")
            else:
                st.session_state["scanner_results"] = pd.DataFrame()
                st.session_state["scanner_dropped"] = dropped
                st.warning("No legs met threshold criteria.")

    # [FIX 13] Always show last scanner results (persists across tab switches)
    scanner_out = st.session_state.get("scanner_results")
    if scanner_out is not None and not scanner_out.empty:
        st.markdown(f"<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.10em;margin-bottom:0.6rem;'>{len(scanner_out)} EDGES FOUND</div>", unsafe_allow_html=True)

        # [UPGRADE 20] Color-code rows by confidence tier
        def _style_scanner_row(row):
            p = float(row.get("p_cal") or 0)
            if p >= 0.65:   bg = "background-color:#00FFB215;"
            elif p >= 0.58: bg = "background-color:#00AAFF12;"
            elif p >= 0.52: bg = "background-color:#FFB80010;"
            else:           bg = "background-color:#FF335810;"
            return [bg] * len(row)

        try:
            styled = scanner_out.style.apply(_style_scanner_row, axis=1)
            st.dataframe(styled, use_container_width=True)
        except Exception:
            st.dataframe(scanner_out, use_container_width=True)

        # [UPGRADE 21] One-click parlay builder from scanner results
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>ONE-CLICK PARLAY BUILDER</div>", unsafe_allow_html=True)
        player_labels = [f"{r['player']} — {r['market']} O{r['line']} ({r.get('ev_adj_pct',0):+.1f}%)"
                         for _, r in scanner_out.iterrows()]
        selected_legs = st.multiselect("Select legs to parlay", options=player_labels, key="parlay_picker")
        if selected_legs:
            sel_indices = [player_labels.index(s) for s in selected_legs if s in player_labels]
            sel_rows = [scanner_out.iloc[i] for i in sel_indices]
            probs = [float(r.get("p_cal") or 0) for r in sel_rows]
            naive_joint = float(np.prod(probs)) if probs else 0.0
            pm = float(st.session_state.get("payout_multi", 3.0))
            ev_parlay = pm * naive_joint - 1.0
            pb1, pb2, pb3 = st.columns(3)
            pb1.metric("Joint Prob (naive)", f"{naive_joint*100:.1f}%")
            pb2.metric(f"EV @ {pm:.1f}x payout", f"{ev_parlay*100:+.1f}%")
            rec_stake_parlay = float(st.session_state.get("bankroll", 1000)) * float(st.session_state.get("frac_kelly", 0.25)) * max(0, ev_parlay / (pm - 1)) if pm > 1 else 0
            pb3.metric("Rec Stake", f"${min(rec_stake_parlay, float(st.session_state.get('bankroll',1000))*0.05):.2f}")
            if st.button("Log This Parlay to History", use_container_width=True, key="log_parlay_btn"):
                parlay_legs = []
                for r in sel_rows:
                    parlay_legs.append({k: v for k, v in r.items()})
                append_history(st.session_state.get("user_id","trader"), {
                    "ts": _now_iso(), "user_id": st.session_state.get("user_id","trader"),
                    "legs": json.dumps(parlay_legs), "n_legs": len(parlay_legs),
                    "result": "Pending", "decision": "BET", "notes": "parlay-builder",
                })
                st.success(f"Logged {len(parlay_legs)}-leg parlay to history.")

        # [UPGRADE 35] Live Steam Check
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#FFB800;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem;'>STEAM DETECTOR — CHECK LINE MOVES VS OPENING</div>", unsafe_allow_html=True)
        sc_steam1, sc_steam2 = st.columns([3,1])
        with sc_steam2:
            steam_thresh = st.number_input("Move threshold", 0.1, 2.0, 0.5, 0.1, key="steam_thresh")
        if sc_steam1.button("Run Steam Check (vs Opening Lines)", use_container_width=True, key="steam_check_btn"):
            steam_alerts = []
            for _, row in scanner_out.iterrows():
                pn = normalize_name(str(row.get("player","")))
                mk = ODDS_MARKETS.get(str(row.get("market","")), "")
                cur_line = float(row.get("line", 0) or 0)
                open_line, _ = get_opening_line(pn, mk, "Over")
                if open_line is not None:
                    delta = cur_line - float(open_line)
                    if abs(delta) >= steam_thresh:
                        direction = "UP" if delta > 0 else "DOWN"
                        steam_type = "STEAM" if (delta > 0) else "FADE"
                        steam_alerts.append({
                            "player": row.get("player"), "market": row.get("market"),
                            "open": open_line, "current": cur_line,
                            "move": f"{direction} {abs(delta):.1f}",
                            "type": steam_type,
                            "ev_%": row.get("ev_adj_pct"),
                        })
            if steam_alerts:
                st.warning(f"**{len(steam_alerts)} line move(s) detected vs opening:**")
                st.dataframe(pd.DataFrame(steam_alerts), use_container_width=True)
                # Auto-alert significant steam
                _dw2 = st.session_state.get("discord_webhook","")
                _tt2 = st.session_state.get("tg_token","")
                _tc2 = st.session_state.get("tg_chat","")
                for sa in steam_alerts:
                    msg = (f"**STEAM ALERT** — {sa['player']} {sa['market']}\n"
                           f"Line moved {sa['move']} ({sa['open']} → {sa['current']}) [{sa['type']}]\n"
                           f"Current EV: {sa.get('ev_%',0):+.1f}%")
                    if _dw2: send_discord_alert(_dw2, msg)
                    if _tt2 and _tc2: send_telegram_alert(_tt2, _tc2, msg)
                if _dw2 or (_tt2 and _tc2):
                    st.success(f"Steam alerts fired to Discord/Telegram.")
            else:
                st.success("No significant line moves vs opening. Lines are stable.")

    scanner_dropped = st.session_state.get("scanner_dropped", [])
    if scanner_dropped:
        with st.expander(f"Excluded ({len(scanner_dropped)})", expanded=False):
            st.dataframe(pd.DataFrame(scanner_dropped).head(200), use_container_width=True)

# ─── PLATFORMS TAB ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>PLATFORMS — PRIZEPICKS / UNDERDOG / LINE SHOPPING</div>""", unsafe_allow_html=True)
    plat_tabs = st.tabs(["PrizePicks", "Underdog", "Line History", "Best Available"])

    with plat_tabs[0]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>PRIZEPICKS NBA LINES</div>", unsafe_allow_html=True)

        pp_load_tab, pp_manual_tab = st.tabs(["Auto Fetch", "Manual Import"])

        with pp_load_tab:
            if st.button("Fetch PrizePicks Lines", use_container_width=True):
                with st.spinner("Fetching PrizePicks..."):
                    pp_lines, pp_err = fetch_prizepicks_lines()
                if pp_err:
                    st.error(f"PrizePicks: {pp_err}")
                    st.info(
                        "PrizePicks blocks cloud server requests (PerimeterX). "
                        "Use **Manual Import** instead:\n\n"
                        "1. Open [PrizePicks](https://app.prizepicks.com) in your browser\n"
                        "2. Open DevTools → Network tab → filter for `projections`\n"
                        "3. Copy the **Response** JSON and paste it in Manual Import\n\n"
                        "Or upload a CSV with columns: `player, stat_type, line`"
                    )
                elif not pp_lines:
                    st.warning("No lines returned.")
                else:
                    pp_df = pd.DataFrame(pp_lines)
                    st.session_state["pp_lines"] = pp_df
                    st.success(f"Fetched {len(pp_df)} PrizePicks props.")

        with pp_manual_tab:
            st.markdown("""<div style='font-size:0.68rem;color:#4A607A;margin-bottom:0.5rem;'>
            <b>How to get the JSON:</b> Open prizepicks.com → DevTools (F12) → Network tab → filter
            <code>projections</code> → click the largest request (~200-300 kB) →
            <b>Response</b> tab → right-click body → Copy &rarr; Copy response. Paste below.<br>
            The JSON starts with <code>{"data":[</code>
            </div>""", unsafe_allow_html=True)
            pp_upload = st.file_uploader("Upload CSV", type=["csv"], key="pp_csv_upload")
            pp_paste = st.text_area("Or paste PrizePicks API JSON response", height=120, key="pp_json_paste",
                                    placeholder='{"data":[{"id":"...","type":"Projection",...}],"included":[...]}')
            if st.button("Load Data", use_container_width=True, key="pp_manual_load"):
                rows = []
                err_msg = None
                if pp_upload is not None:
                    try:
                        df_up = pd.read_csv(pp_upload)
                        df_up.columns = [c.strip().lower() for c in df_up.columns]
                        col_map = {}
                        for need, alts in [("player",["player","name","player_name"]),
                                           ("stat_type",["stat_type","stat","market","type"]),
                                           ("line",["line","line_score","value","projection"])]:
                            for a in alts:
                                if a in df_up.columns:
                                    col_map[need] = a; break
                        if all(k in col_map for k in ("player","stat_type","line")):
                            for _, r in df_up.iterrows():
                                rows.append({"player": str(r[col_map["player"]]),
                                             "stat_type": str(r[col_map["stat_type"]]),
                                             "line": float(r[col_map["line"]]),
                                             "source": "prizepicks"})
                        else:
                            err_msg = f"CSV must have player, stat_type, line columns. Found: {list(df_up.columns)}"
                    except Exception as e:
                        err_msg = f"CSV parse error: {e}"
                elif pp_paste.strip():
                    try:
                        raw = pp_paste.strip()
                        data = json.loads(raw)
                        if not isinstance(data, dict):
                            raise ValueError("Expected a JSON object starting with {. Make sure you copied the full Response body, not just part of it.")
                        if "data" not in data:
                            raise ValueError('JSON is missing a "data" key. Copy the Response tab (not Preview or Headers).')
                        included = {item["id"]: item for item in data.get("included", []) if isinstance(item, dict) and "id" in item}
                        for proj in data.get("data", []):
                            if not isinstance(proj, dict): continue
                            if proj.get("type") != "Projection": continue
                            attrs = proj.get("attributes", {}) or {}
                            rels = proj.get("relationships", {}) or {}
                            pid = (rels.get("new_player",{}).get("data",{}) or {}).get("id")
                            if not pid:
                                pid = (rels.get("player",{}).get("data",{}) or {}).get("id")
                            pattrs = included.get(pid,{}).get("attributes",{}) if pid else {}
                            pname = pattrs.get("name","") or attrs.get("name","")
                            stat_type = attrs.get("stat_type","")
                            line_score = attrs.get("line_score")
                            if pname and stat_type and line_score is not None:
                                try:
                                    rows.append({"player": pname, "stat_type": stat_type,
                                                 "line": float(line_score), "source": "prizepicks"})
                                except (TypeError, ValueError):
                                    pass
                        if not rows:
                            err_msg = "No NBA projections found. The JSON parsed OK but contained no NBA props — check you selected the NBA board before copying."
                    except json.JSONDecodeError as e:
                        err_msg = f'Invalid JSON: {e}. Make sure you copied the entire response (it should start with {{"data":[).'
                    except ValueError as e:
                        err_msg = str(e)
                    except Exception as e:
                        err_msg = f"Parse error: {type(e).__name__}: {e}"
                else:
                    err_msg = "Upload a CSV or paste JSON."
                if err_msg:
                    st.error(err_msg)
                elif rows:
                    pp_df_manual = pd.DataFrame(rows)
                    st.session_state["pp_lines"] = pp_df_manual
                    st.success(f"Loaded {len(pp_df_manual)} PrizePicks props.")

        pp_df = st.session_state.get("pp_lines")
        if pp_df is not None and not pp_df.empty:
            # Run model on PrizePicks lines
            pp_col1, pp_col2 = st.columns([3,1])
            with pp_col1:
                pp_filter = st.text_input("Filter player", key="pp_filter")
            with pp_col2:
                pp_min_ev = st.number_input("Min EV%", -5.0, 30.0, 2.0, 0.5, key="pp_min_ev")
            display_df = pp_df
            if pp_filter:
                display_df = pp_df[pp_df["player"].str.contains(pp_filter, case=False, na=False)]
            st.dataframe(display_df, use_container_width=True)
            if st.button("Scan PrizePicks vs Model", use_container_width=True):
                pp_candidates = []
                for _, r in display_df.iterrows():
                    mkt = map_platform_stat_to_market(r.get("stat_type",""))
                    if mkt and r.get("line"):
                        pp_candidates.append((r["player"], mkt, float(r["line"]), None))
                if pp_candidates:
                    _inj_map = st.session_state.get("injury_team_map", {})
                    with st.spinner(f"Scanning {len(pp_candidates)} PrizePicks props..."):
                        with ThreadPoolExecutor(max_workers=16) as ex:
                            futs_pp = [ex.submit(compute_leg_projection, pn, mk, ln, mt,
                                                 n_games=n_games, key_teammate_out=False,
                                                 bankroll=bankroll, frac_kelly=frac_kelly,
                                                 market_prior_weight=market_prior_weight,
                                                 exclude_chaotic=bool(exclude_chaotic),
                                                 game_date=date.today(),
                                                 injury_team_map=_inj_map)
                                       for pn, mk, ln, mt in pp_candidates]
                            pp_results = []
                            for fut in futs_pp:
                                try:
                                    pp_results.append(fut.result())
                                except Exception as _e:
                                    pass
                    calib = st.session_state.get("calibrator_map")
                    pp_results = [recompute_pricing_fields(dict(l), calib) for l in pp_results]
                    pp_edges = [l for l in pp_results if l.get("gate_ok") and float(l.get("ev_adj",0) or 0)*100 >= pp_min_ev]
                    if pp_edges:
                        st.success(f"{len(pp_edges)} edges vs PrizePicks lines")
                        pp_out = pd.DataFrame([{
                            "player": l["player"], "market": l["market"], "line": l["line"],
                            "proj": safe_round(l.get("proj")), "p_cal": safe_round(l.get("p_cal"),3),
                            "ev_%": safe_round(l.get("ev_adj",0)*100,1), "edge_cat": l.get("edge_cat",""),
                        } for l in pp_edges])
                        st.dataframe(pp_out.sort_values("ev_%", ascending=False), use_container_width=True)
                    else:
                        st.warning("No edges found vs PrizePicks lines.")

    with plat_tabs[1]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>UNDERDOG FANTASY NBA LINES</div>", unsafe_allow_html=True)
        if st.button("Fetch Underdog Lines", use_container_width=True):
            with st.spinner("Fetching Underdog..."):
                ud_lines, ud_err = fetch_underdog_lines()
            if ud_err:
                st.error(f"Underdog: {ud_err}")
            elif not ud_lines:
                st.warning("No lines returned.")
            else:
                ud_df = pd.DataFrame(ud_lines)
                st.session_state["ud_lines"] = ud_df
                st.success(f"Fetched {len(ud_df)} Underdog props.")
        ud_df = st.session_state.get("ud_lines")
        if ud_df is not None and not ud_df.empty:
            ud_filter = st.text_input("Filter player", key="ud_filter")
            display_ud = ud_df
            if ud_filter:
                display_ud = ud_df[ud_df["player"].str.contains(ud_filter, case=False, na=False)]
            st.dataframe(display_ud, use_container_width=True)
            if st.button("Scan Underdog vs Model", use_container_width=True):
                ud_candidates = []
                for _, r in display_ud.iterrows():
                    mkt = map_platform_stat_to_market(r.get("stat_type",""))
                    if mkt and r.get("line"):
                        ud_candidates.append((r["player"], mkt, float(r["line"]), None))
                if ud_candidates:
                    _inj_map = st.session_state.get("injury_team_map", {})
                    with st.spinner(f"Scanning {len(ud_candidates)} Underdog props..."):
                        with ThreadPoolExecutor(max_workers=16) as ex:
                            futs_ud = [ex.submit(compute_leg_projection, pn, mk, ln, mt,
                                                 n_games=n_games, key_teammate_out=False,
                                                 bankroll=bankroll, frac_kelly=frac_kelly,
                                                 market_prior_weight=market_prior_weight,
                                                 exclude_chaotic=bool(exclude_chaotic),
                                                 game_date=date.today(),
                                                 injury_team_map=_inj_map)
                                       for pn, mk, ln, mt in ud_candidates]
                            ud_results = []
                            for fut in futs_ud:
                                try:
                                    ud_results.append(fut.result())
                                except Exception:
                                    pass
                    calib = st.session_state.get("calibrator_map")
                    ud_results = [recompute_pricing_fields(dict(l), calib) for l in ud_results]
                    ud_edges = [l for l in ud_results if l.get("gate_ok") and float(l.get("ev_adj",0) or 0) > 0]
                    if ud_edges:
                        st.success(f"{len(ud_edges)} edges vs Underdog lines")
                        ud_out = pd.DataFrame([{
                            "player": l["player"], "market": l["market"], "line": l["line"],
                            "proj": safe_round(l.get("proj")), "p_cal": safe_round(l.get("p_cal"),3),
                            "ev_%": safe_round(l.get("ev_adj",0)*100,1),
                        } for l in ud_edges])
                        st.dataframe(ud_out.sort_values("ev_%", ascending=False), use_container_width=True)
                    else:
                        st.warning("No edges found vs Underdog lines.")

    with plat_tabs[2]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>PROP LINE HISTORY</div>", unsafe_allow_html=True)
        ph_col1, ph_col2 = st.columns(2)
        with ph_col1:
            ph_player = st.text_input("Player name", key="ph_player")
        with ph_col2:
            ph_market = st.selectbox("Market", [""] + list(ODDS_MARKETS.keys()), key="ph_market")
        ph_df = load_prop_line_history(
            player=ph_player if ph_player else None,
            market=ph_market if ph_market else None,
        )
        if ph_df.empty:
            st.info("No prop line history yet. Lines are auto-saved when you run the Live Scanner.")
        else:
            st.dataframe(ph_df, use_container_width=True)
            ph_csv = ph_df.to_csv(index=False)
            st.download_button("Export Line History CSV", ph_csv, "prop_line_history.csv", "text/csv")

    with plat_tabs[3]:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>BEST AVAILABLE LINE SHOPPING</div>", unsafe_allow_html=True)
        st.caption("Checks all available books for the highest price on any scanner result. Requires a valid Odds API key.")
        scanner_out_shop = st.session_state.get("scanner_results")
        if scanner_out_shop is None or scanner_out_shop.empty:
            st.info("Run the Live Scanner first to populate results.")
        else:
            has_odds_key = bool(odds_api_key())
            if not has_odds_key:
                st.warning("No Odds API key configured — best_price column will be empty. Add ODDS_API_KEY in Settings to enable price lookups.")
            if st.button("Find Best Lines for Scanner Results", use_container_width=True):
                shop_rows = []
                for _, r in scanner_out_shop.iterrows():
                    eid = r.get("event_id") if hasattr(r,"get") else None
                    mk = ODDS_MARKETS.get(r.get("market","") if hasattr(r,"get") else "")
                    pn = normalize_name(str(r.get("player","") if hasattr(r,"get") else ""))
                    best_p, best_b = None, None
                    if eid and mk and pn and has_odds_key:
                        best_p, best_b = get_best_available_price(eid, mk, pn, "Over")
                    shop_rows.append({
                        "player": r.get("player",""), "market": r.get("market",""),
                        "line": r.get("line"), "book": r.get("book",""),
                        "ev_%": r.get("ev_adj_pct"),
                        "best_price": safe_round(best_p) if best_p else "—",
                        "best_book": best_b or ("no key" if not has_odds_key else "not found"),
                    })
                if shop_rows:
                    st.dataframe(pd.DataFrame(shop_rows), use_container_width=True)
                else:
                    st.info("No scanner results to display.")

# ─── HISTORY TAB [FIX 12: export button] ──────────────────────
with tabs[4]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>BET HISTORY & CLV TRACKER</div>""", unsafe_allow_html=True)
    h = load_history(user_id)
    if h.empty:
        st.markdown(make_card("<span style='color:#4A607A;'>No bets logged yet. Log from the Model tab.</span>"), unsafe_allow_html=True)
    else:
        settled = h[h["result"].isin(["HIT","MISS","PUSH"])].copy() if not h.empty else pd.DataFrame()
        n_hit = (settled["result"]=="HIT").sum() if not settled.empty else 0
        n_miss = (settled["result"]=="MISS").sum() if not settled.empty else 0
        n_pend = (h["result"]=="Pending").sum() if not h.empty else 0
        hit_rate = n_hit/(n_hit+n_miss) if (n_hit+n_miss)>0 else None
        hc1,hc2,hc3,hc4 = st.columns(4)
        hc1.metric("Parlay Hit Rate", f"{hit_rate*100:.1f}%" if hit_rate else "--")
        hc2.metric("Parlay Wins", n_hit)
        hc3.metric("Parlay Losses", n_miss)
        hc4.metric("Pending", n_pend)

        # Per-leg accuracy (only legs with individual results logged)
        h_legs_df = _expand_history_legs(h)
        settled_legs_df = h_legs_df[h_legs_df["y"].notna()].copy() if not h_legs_df.empty else pd.DataFrame()
        if not settled_legs_df.empty:
            n_leg_hit = int((settled_legs_df["y"]==1).sum())
            n_leg_miss = int((settled_legs_df["y"]==0).sum())
            leg_hr = n_leg_hit/(n_leg_hit+n_leg_miss) if (n_leg_hit+n_leg_miss)>0 else None
            lc1,lc2,lc3 = st.columns(3)
            lc1.metric("Per-Leg Hit Rate", f"{leg_hr*100:.1f}%" if leg_hr else "--",
                       help="Individual leg accuracy — requires per-leg results to be marked below")
            lc2.metric("Leg Hits", n_leg_hit)
            lc3.metric("Leg Misses", n_leg_miss)
        else:
            st.caption("Per-leg accuracy will appear once you mark individual leg results below.")

        st.dataframe(h, use_container_width=True)

        # [FIX 12] Export button
        csv_data = h.to_csv(index=False)
        st.download_button("Export History CSV", data=csv_data,
                           file_name=f"history_{user_id}.csv", mime="text/csv",
                           use_container_width=True)

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)

        # ── Per-leg result update (fixes calibration skew for multi-leg bets) ──
        st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.10em;'>UPDATE PER-LEG RESULTS</span>", unsafe_allow_html=True)
        st.caption("For multi-leg bets, mark each leg individually so calibration uses accurate per-leg outcomes.")
        leg_row_idx = st.number_input("Bet row to update legs", 0, max(0, len(h)-1), 0, 1, key="leg_row_idx")
        try:
            _legs_to_show = json.loads(h.loc[int(leg_row_idx), "legs"]) if isinstance(h.loc[int(leg_row_idx), "legs"], str) else []
            _leg_res_stored = json.loads(h.loc[int(leg_row_idx), "leg_results"]) if "leg_results" in h.columns and isinstance(h.loc[int(leg_row_idx), "leg_results"], str) else ["Pending"]*len(_legs_to_show)
            if len(_leg_res_stored) < len(_legs_to_show):
                _leg_res_stored = _leg_res_stored + ["Pending"]*(len(_legs_to_show)-len(_leg_res_stored))
        except Exception:
            _legs_to_show = []
            _leg_res_stored = []
        if _legs_to_show:
            new_leg_results = []
            for _li, _lleg in enumerate(_legs_to_show):
                _default = _leg_res_stored[_li] if _li < len(_leg_res_stored) else "Pending"
                _opts = ["Pending","HIT","MISS","PUSH"]
                _di = _opts.index(_default) if _default in _opts else 0
                _lcols = st.columns([3,2])
                _lcols[0].markdown(f"<span style='font-size:0.75rem;color:#A0B8C8;'>{_lleg.get('player','?')} — {_lleg.get('market','?')} {_lleg.get('line','')}</span>", unsafe_allow_html=True)
                _sel = _lcols[1].selectbox("", _opts, index=_di, key=f"legres_{leg_row_idx}_{_li}", label_visibility="collapsed")
                new_leg_results.append(_sel)
            if st.button("Save Leg Results", key="save_leg_results"):
                h2 = h.copy()
                h2.loc[int(leg_row_idx), "leg_results"] = json.dumps(new_leg_results)
                # Auto-derive parlay result: HIT only if all legs HIT, MISS if any MISS, else Pending
                if all(r=="HIT" for r in new_leg_results):
                    h2.loc[int(leg_row_idx), "result"] = "HIT"
                elif any(r=="MISS" for r in new_leg_results):
                    h2.loc[int(leg_row_idx), "result"] = "MISS"
                elif all(r in ("HIT","PUSH") for r in new_leg_results):
                    h2.loc[int(leg_row_idx), "result"] = "PUSH"
                h2.to_csv(history_path(user_id), index=False)
                st.success("Leg results saved — calibration will now use individual leg outcomes.")
        else:
            st.caption("No legs found for this row.")

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        uc1, uc2 = st.columns(2)
        with uc1:
            st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#4A607A;letter-spacing:0.10em;'>UPDATE PARLAY RESULT</span>", unsafe_allow_html=True)
            idx = st.number_input("Row index", 0, max(0,len(h)-1), 0, 1)
            new_res = st.selectbox("Result", ["Pending","HIT","MISS","PUSH"])
            if st.button("Update Result"):
                h2 = h.copy(); h2.loc[int(idx),"result"] = new_res
                h2.to_csv(history_path(user_id), index=False)
                st.success("Updated")
        with uc2:
            st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#4A607A;letter-spacing:0.10em;'>CLV UPDATE</span>", unsafe_allow_html=True)
            idx2 = st.number_input("CLV Row index", 0, max(0,len(h)-1), 0, 1, key="clv_idx")
            if st.button("Fetch & Update CLV"):
                try:
                    h2 = h.copy()
                    legs = json.loads(h2.loc[int(idx2),"legs"])
                    if not isinstance(legs,list) or not legs:
                        st.warning("No legs on that row.")
                    else:
                        legs2, errs = apply_clv_update_to_legs(legs)
                        h2.loc[int(idx2),"legs"] = json.dumps(legs2)
                        h2.to_csv(history_path(user_id), index=False)
                        for e in errs[:5]: st.warning(e)
                        st.success("CLV updated")
                except Exception as e:
                    st.error(f"CLV update failed: {e}")

# ─── CALIBRATION TAB ─────────────────────────────────────────
with tabs[5]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>CALIBRATION ENGINE</div>""", unsafe_allow_html=True)
    h = load_history(user_id)
    legs_df = _expand_history_legs(h)
    # Only use settled legs for calibration metrics
    settled_df = legs_df[legs_df["y"].notna()].copy() if not legs_df.empty else pd.DataFrame()
    if settled_df.empty:
        st.markdown(make_card("<span style='color:#4A607A;font-size:0.78rem;'>No settled bets yet. Log bets and mark results to enable calibration.<br><span style='font-size:0.65rem;'>Minimum ~80 settled legs needed.</span></span>"), unsafe_allow_html=True)
    else:
        y = settled_df["y"].values.astype(float)
        p_raw = settled_df["p_raw"].values.astype(float)
        brier = float(np.mean((p_raw - y)**2))
        hit_rate_cal = float(y.mean())
        n_settled = len(settled_df)
        n_pass_logged = len(legs_df[legs_df.get("decision","")=="PASS"]) if "decision" in legs_df.columns else 0
        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("Settled Legs", n_settled)
        cc2.metric("Actual Hit Rate", f"{hit_rate_cal*100:.1f}%")
        cc3.metric("Brier Score (raw)", f"{brier:.4f}")
        cc4.metric("Calibrator Fitted", "Yes" if st.session_state.get("calibrator_map") else "No")

        if "clv_line_fav" in settled_df.columns and settled_df["clv_line_fav"].notna().any():
            clv_line_rate = float(settled_df["clv_line_fav"].dropna().astype(int).mean())
            st.metric("CLV (line) favorable %", f"{clv_line_rate*100:.1f}%",
                      delta="Edge exists" if clv_line_rate > 0.52 else "No edge vs closing line",
                      delta_color="normal" if clv_line_rate > 0.52 else "inverse")

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>ROI BY MARKET</div>", unsafe_allow_html=True)
        if "market" in settled_df.columns:
            mkt_grp = settled_df.groupby("market").agg(
                bets=("y","size"), hit_rate=("y","mean"),
                avg_ev=("ev_adj","mean")
            ).reset_index()
            mkt_grp["hit_rate_pct"] = (mkt_grp["hit_rate"]*100).round(1)
            mkt_grp["avg_ev_pct"] = (mkt_grp["avg_ev"]*100).round(2)
            mkt_grp = mkt_grp.sort_values("hit_rate_pct", ascending=False)
            st.dataframe(mkt_grp, use_container_width=True)

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>RELIABILITY TABLE (p_raw vs actual)</div>", unsafe_allow_html=True)
        n_bins = st.slider("Bins", 6, 20, 10)
        settled_df["bin"] = pd.cut(settled_df["p_raw"], bins=n_bins, labels=False, include_lowest=True)
        rel = settled_df.groupby("bin",dropna=True).agg(
            p_mean=("p_raw","mean"), win_rate=("y","mean"), n=("y","size")).reset_index()
        rel["calibration_error"] = (rel["p_mean"] - rel["win_rate"]).abs().round(3)
        st.dataframe(rel, use_container_width=True)

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>FIT CALIBRATOR</div>", unsafe_allow_html=True)
        st.caption("Monotone isotonic calibration maps p_raw -> p_cal using your settled history.")
        if st.button("Fit Calibrator from History", use_container_width=True):
            calib = fit_monotone_calibrator(settled_df, n_bins=int(n_bins))
            if calib is None:
                st.warning(f"Need ~80+ quality legs (currently {n_settled}). Identity calibration used.")
                st.session_state["calibrator_map"] = None
            else:
                st.session_state["calibrator_map"] = calib
                st.success(f"Calibrator fitted on {calib.get('n','?')} legs (range: {calib.get('training_min',0):.2f}-{calib.get('training_max',1):.2f})")

        calib = st.session_state.get("calibrator_map")
        if calib:
            settled_df["p_cal_fit"] = settled_df["p_raw"].apply(lambda p: apply_calibrator(p, calib))
            brier_cal = float(np.mean((settled_df["p_cal_fit"].values.astype(float)-y)**2))
            st.metric("Brier Score (calibrated)", f"{brier_cal:.4f}",
                      delta=f"{(brier_cal-brier)*100:.2f}% vs raw",
                      delta_color="inverse")
            # [FIX 9] Show training range
            st.caption(f"Training range: [{calib.get('training_min',0):.3f}, {calib.get('training_max',1):.3f}]")

            settled_df["bin2"] = pd.cut(settled_df["p_cal_fit"], bins=n_bins, labels=False, include_lowest=True)
            rel2 = settled_df.groupby("bin2",dropna=True).agg(
                p_mean=("p_cal_fit","mean"),win_rate=("y","mean"),n=("y","size")).reset_index()
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00AAFF;letter-spacing:0.12em;text-transform:uppercase;margin:0.6rem 0;'>POST-CALIBRATION RELIABILITY</div>", unsafe_allow_html=True)
            st.dataframe(rel2, use_container_width=True)

            st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#4A607A;'>POLICY AUDIT</div>", unsafe_allow_html=True)
            if hit_rate_cal < 0.48:
                st.error("Hit rate below 48% - cut volume, tighten EV threshold, review market selection.")
            elif hit_rate_cal > 0.58:
                st.success("Strong hit rate - consider increasing Kelly fraction gradually.")
            else:
                st.info("Moderate hit rate. Continue collecting data, focus on CLV tracking.")
            if brier_cal > brier:
                st.warning("Calibrator is WORSENING Brier score - needs more data. Reset to identity.")

    # ── Rolling Brier Score ────────────────────────────────────
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>ROLLING BRIER SCORE (TRAILING WINDOWS)</div>", unsafe_allow_html=True)
    rb = compute_rolling_brier(settled_df if not settled_df.empty else pd.DataFrame())
    if rb:
        rb_cols = st.columns(3)
        for i, w in enumerate([25, 50, 100]):
            key = f"last_{w}"
            if key in rb:
                rb_cols[i].metric(f"Last {w} Legs", f"{rb[key]:.4f}", help="Brier score: lower = better calibrated")
        if "rolling_series" in rb and len(rb["rolling_series"]) > 5:
            st.caption("Trailing 25-leg rolling Brier (lower = better):")
            import pandas as _pd_rb
            series_df = _pd_rb.DataFrame({"Brier": rb["rolling_series"]})
            st.line_chart(series_df, use_container_width=True, height=150)
    else:
        st.caption("Need 10+ settled legs for rolling Brier.")

# ─── INSIGHTS TAB (CLV leaderboard, book efficiency, prop breakdown, Bayesian priors) ───
with tabs[6]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>INSIGHTS — EDGE ANALYTICS & INTELLIGENCE</div>""", unsafe_allow_html=True)
    h_ins = load_history(st.session_state.get("user_id","trader"))
    legs_ins = _expand_history_legs(h_ins)

    ins_tabs = st.tabs(["CLV Leaderboard", "Book Efficiency", "Prop Breakdown", "Bayesian Priors"])

    with ins_tabs[0]:
        # [UPGRADE 31] CLV Leaderboard
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>TOP CLOSING LINE VALUE PLAYS</div>", unsafe_allow_html=True)
        st.caption("Ranks your best bets by no-vig price CLV. Positive = you beat the closing line. This is the gold-standard long-term edge indicator.")
        clv_lb = compute_clv_leaderboard(h_ins, top_n=25)
        if clv_lb.empty:
            st.info("No CLV data yet. Fetch CLV updates from the History tab after games close.")
        else:
            clv_pos = clv_lb[clv_lb["clv_price"].notna() & (clv_lb["clv_price"] > 0)]
            clv_neg = clv_lb[clv_lb["clv_price"].notna() & (clv_lb["clv_price"] <= 0)]
            rate = len(clv_pos) / max(len(clv_lb), 1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Beats Closing Line", f"{rate*100:.0f}%", help="CLV price > 0 = bought above close")
            c2.metric("Avg CLV (price)", f"{clv_lb['clv_price'].mean():.4f}" if not clv_lb.empty else "--")
            c3.metric("Avg Line CLV", f"{clv_lb['clv_line'].mean():.2f}" if not clv_lb.empty else "--")
            st.dataframe(clv_lb, use_container_width=True)
            if not clv_pos.empty:
                st.markdown("<div style='font-size:0.65rem;color:#00FFB2;margin-top:0.4rem;'>Positive CLV = model found real edge vs closing price. Keep targeting these players/markets.</div>", unsafe_allow_html=True)

    with ins_tabs[1]:
        # [UPGRADE 32] Book Efficiency Score
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>PER-BOOK MARKET EFFICIENCY</div>", unsafe_allow_html=True)
        st.caption("Books with higher hit rate = your model has edge there. Books that never move = likely pricing your bets correctly. Focus on soft books with high hit rate.")
        book_eff = compute_book_efficiency(h_ins)
        if book_eff.empty:
            st.info("Need settled bets with CLV data to compute book efficiency.")
        else:
            st.dataframe(book_eff, use_container_width=True)
            best_book = book_eff.iloc[0]["book"] if not book_eff.empty else None
            if best_book:
                st.markdown(f"<div style='color:#00FFB2;font-size:0.68rem;margin-top:0.4rem;'>Best book by hit rate: <b>{best_book}</b> — prioritize this book when line shopping.</div>", unsafe_allow_html=True)

    with ins_tabs[2]:
        # [UPGRADE 33] Prop-type edge breakdown
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>EDGE BREAKDOWN BY MARKET TYPE</div>", unsafe_allow_html=True)
        st.caption("Hit rate and EV by stat type. If rebounds hit at 58% and points at 48%, focus on rebounds.")
        if legs_ins.empty:
            st.info("No settled legs yet.")
        else:
            settled_ins = legs_ins[legs_ins["y"].notna()].copy()
            if settled_ins.empty:
                st.info("No settled legs yet.")
            else:
                mkt_breakdown = settled_ins.groupby("market").agg(
                    bets=("y","size"),
                    hit_rate=("y","mean"),
                    avg_ev_adj=("ev_adj","mean"),
                    avg_cv=("cv","mean"),
                ).reset_index()
                mkt_breakdown["hit_%"]  = (mkt_breakdown["hit_rate"] * 100).round(1)
                mkt_breakdown["ev_%"]   = (mkt_breakdown["avg_ev_adj"] * 100).round(2)
                mkt_breakdown["avg_cv"] = mkt_breakdown["avg_cv"].round(3)
                mkt_breakdown = mkt_breakdown.sort_values("hit_%", ascending=False)
                st.dataframe(mkt_breakdown[["market","bets","hit_%","ev_%","avg_cv"]], use_container_width=True)
                # Highlight best market
                best_mkt = mkt_breakdown.iloc[0]["market"] if not mkt_breakdown.empty else None
                if best_mkt:
                    st.markdown(f"<div style='color:#00FFB2;font-size:0.68rem;'>Best market: <b>{best_mkt}</b> ({mkt_breakdown.iloc[0]['hit_%']:.1f}% hit rate). Increase allocation here.</div>", unsafe_allow_html=True)
                # Worst market
                worst_mkt = mkt_breakdown.iloc[-1]["market"] if len(mkt_breakdown) > 1 else None
                if worst_mkt and mkt_breakdown.iloc[-1]["hit_%"] < 48:
                    st.markdown(f"<div style='color:#FF3358;font-size:0.68rem;'>Weakest market: <b>{worst_mkt}</b> ({mkt_breakdown.iloc[-1]['hit_%']:.1f}% hit rate). Consider avoiding.</div>", unsafe_allow_html=True)

    with ins_tabs[3]:
        # [UPGRADE 34] Bayesian Prior Update from history
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;'>BAYESIAN PRIOR UPDATE FROM YOUR HISTORY</div>", unsafe_allow_html=True)
        st.caption("Updates positional priors (the baseline expected stat means) using your personal hit/miss record. Markets you consistently beat get slightly higher priors.")
        pos_for_prior = st.selectbox("Position bucket", ["Guard","Wing","Big","Unknown"], key="prior_pos_sel")
        if not legs_ins.empty and len(legs_ins[legs_ins["y"].notna()]) >= 20:
            updated_priors = compute_history_based_priors(legs_ins, pos_for_prior)
            base_priors    = POSITIONAL_PRIORS.get(pos_for_prior, POSITIONAL_PRIORS["Unknown"])
            prior_rows = []
            for mkt in base_priors:
                orig = base_priors[mkt]
                upd  = updated_priors.get(mkt, orig)
                prior_rows.append({
                    "market": mkt,
                    "original_prior": round(orig, 2),
                    "updated_prior":  round(upd, 2),
                    "delta_%": round((upd/orig - 1)*100, 1) if orig else 0,
                })
            prior_df = pd.DataFrame(prior_rows).sort_values("delta_%", ascending=False)
            st.dataframe(prior_df, use_container_width=True)
            if st.button("Apply Updated Priors to Session", use_container_width=True, key="apply_priors_btn"):
                st.session_state["_custom_priors"] = {pos_for_prior: updated_priors}
                st.success(f"Updated {pos_for_prior} priors applied to this session. Model will use these for the next run.")
        else:
            st.info("Need 20+ settled legs to compute personalised prior updates.")

# ─── ALERTS TAB ───────────────────────────────────────────────
with tabs[7]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>ALERTS — DISCORD / TELEGRAM</div>""", unsafe_allow_html=True)

    # ── PrizePicks cookie auth ─────────────────────────────────
    with st.expander("PrizePicks Cookie Auth (bypasses 403 block)", expanded=False):
        st.markdown("""<div style='font-size:0.68rem;color:#4A607A;line-height:1.6;'>
<b>How to get your cookies (one-time setup, lasts days):</b><br>
1. Open <a href='https://app.prizepicks.com' target='_blank' style='color:#00FFB2;'>app.prizepicks.com</a> in Chrome and log in<br>
2. Press <b>F12</b> → Network tab → refresh the page<br>
3. Click any <code>projections</code> request → Headers tab<br>
4. Find the <b>Cookie:</b> request header → copy the entire value<br>
5. Paste below and click Save
</div>""", unsafe_allow_html=True)
        pp_cookies_val = st.text_area(
            "PrizePicks Cookie String",
            value=st.session_state.get("pp_cookies", ""),
            height=80,
            placeholder="_pxmvid=...; _px3=...; __cf_bm=...",
            key="pp_cookies_input",
        )
        if st.button("Save PrizePicks Cookies", use_container_width=True):
            st.session_state["pp_cookies"] = pp_cookies_val.strip()
            st.success("Cookies saved — Auto Fetch will now use these for auth.")
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
    al_col1, al_col2 = st.columns(2)
    with al_col1:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>DISCORD WEBHOOK</div>", unsafe_allow_html=True)
        discord_webhook = st.text_input("Webhook URL", value=st.session_state.get("discord_webhook",""), type="password", key="discord_wh_input")
        st.session_state["discord_webhook"] = discord_webhook
        if st.button("Test Discord", use_container_width=True):
            ok, err = send_discord_alert(discord_webhook, "NBA Quant Engine — Discord alert test ✅")
            st.success("Discord OK") if ok else st.error(f"Discord failed: {err}")
        st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
        st.caption("Auto-alert on strong edges (EV > threshold):")
        discord_ev_thresh = st.slider("Min EV% for auto-alert", 0.0, 25.0, float(st.session_state.get("discord_ev_thresh",5.0)), 0.5, key="d_ev_thresh")
        st.session_state["discord_ev_thresh"] = float(discord_ev_thresh)
    with al_col2:
        st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#00FFB2;letter-spacing:0.12em;'>TELEGRAM BOT</div>", unsafe_allow_html=True)
        tg_token = st.text_input("Bot Token", value=st.session_state.get("tg_token",""), type="password", key="tg_token_input")
        st.session_state["tg_token"] = tg_token
        tg_chat = st.text_input("Chat ID", value=st.session_state.get("tg_chat",""), key="tg_chat_input")
        st.session_state["tg_chat"] = tg_chat
        if st.button("Test Telegram", use_container_width=True):
            ok, err = send_telegram_alert(tg_token, tg_chat, "NBA Quant Engine — Telegram alert test ✅")
            st.success("Telegram OK") if ok else st.error(f"Telegram failed: {err}")
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>SEND SCANNER EDGES AS ALERTS</div>", unsafe_allow_html=True)
    scanner_for_alerts = st.session_state.get("scanner_results")
    if scanner_for_alerts is None or scanner_for_alerts.empty:
        st.info("Run Live Scanner first.")
    else:
        alert_thresh = st.slider("Min EV% to include", 0.0, 20.0, 3.0, 0.5, key="alert_thresh_send")
        alerted = [r for _, r in scanner_for_alerts.iterrows() if float(r.get("ev_adj_pct") or 0) >= alert_thresh]
        st.write(f"{len(alerted)} edges above {alert_thresh:.1f}% EV threshold")
        if st.button(f"Send {len(alerted)} alerts to Discord + Telegram", use_container_width=True):
            sent_d, sent_t, errs_d, errs_t = 0, 0, [], []
            for r in alerted:
                msg = format_edge_alert(dict(r))
                if discord_webhook:
                    ok, e = send_discord_alert(discord_webhook, msg)
                    if ok: sent_d += 1
                    elif e: errs_d.append(e)
                if tg_token and tg_chat:
                    ok, e = send_telegram_alert(tg_token, tg_chat, msg)
                    if ok: sent_t += 1
                    elif e: errs_t.append(e)
            st.success(f"Sent — Discord: {sent_d} | Telegram: {sent_t}")
            for e in (errs_d + errs_t)[:5]:
                st.warning(e)
    # [UPGRADE 23] Alert Digest Mode
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>DAILY DIGEST MODE</div>", unsafe_allow_html=True)
    scanner_for_digest = st.session_state.get("scanner_results")
    if scanner_for_digest is None or scanner_for_digest.empty:
        st.info("Run Live Scanner first to generate digest content.")
    else:
        digest_ev_thresh = st.slider("Min EV% for digest", 0.0, 20.0, 3.0, 0.5, key="digest_thresh")
        digest_legs = [dict(r) for _, r in scanner_for_digest.iterrows()
                       if float(r.get("ev_adj_pct") or 0) >= digest_ev_thresh]
        digest_msg  = format_digest_message(digest_legs)
        st.text_area("Digest Preview", value=digest_msg, height=200, key="digest_preview")
        dig_c1, dig_c2 = st.columns(2)
        if dig_c1.button("Send Digest to Discord", use_container_width=True, key="send_digest_discord"):
            _dw3 = st.session_state.get("discord_webhook","")
            if _dw3:
                ok, err = send_discord_alert(_dw3, digest_msg)
                st.success("Digest sent to Discord.") if ok else st.error(f"Discord error: {err}")
            else:
                st.warning("No Discord webhook configured.")
        if dig_c2.button("Send Digest to Telegram", use_container_width=True, key="send_digest_tg"):
            _tt3 = st.session_state.get("tg_token",""); _tc3 = st.session_state.get("tg_chat","")
            if _tt3 and _tc3:
                ok, err = send_telegram_alert(_tt3, _tc3, digest_msg)
                st.success("Digest sent to Telegram.") if ok else st.error(f"Telegram error: {err}")
            else:
                st.warning("No Telegram bot configured.")

    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#4A607A;letter-spacing:0.10em;'>INJURY REPORT</div>", unsafe_allow_html=True)
    if st.button("Fetch Today's Injury Report", use_container_width=True):
        fetch_injury_report.clear()   # force fresh data
        with st.spinner("Fetching from ESPN..."):
            injuries = fetch_injury_report()
        if not injuries:
            st.info("No injury data found — ESPN may not have posted today's report yet.")
        else:
            # Build and store the map for auto key_teammate_out
            inj_map = build_injury_team_map(injuries)
            st.session_state["injury_team_map"] = inj_map
            out_count = sum(len(v) for v in inj_map.values())
            st.success(f"Loaded {out_count} OUT/DOUBTFUL player(s) — auto teammate-out active in scanner & model.")
            inj_rows = []
            for team, players in injuries.items():
                for p in players:
                    inj_rows.append({"team": team, **p})
            st.dataframe(
                pd.DataFrame(inj_rows).sort_values(["team","status"]),
                use_container_width=True,
            )
    # Show current injury map status
    cur_inj = st.session_state.get("injury_team_map", {})
    if cur_inj:
        n_out = sum(len(v) for v in cur_inj.values())
        st.caption(f"Auto-injury active: {n_out} OUT/DOUBTFUL players across {len(cur_inj)} teams")

    # [UPGRADE 9] Rotowire News Feed
    st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;color:#FFB800;letter-spacing:0.10em;text-transform:uppercase;'>ROTOWIRE NBA NEWS — FAST INJURY INTEL</div>", unsafe_allow_html=True)
    st.caption("Scrapes Rotowire's live NBA injury page for faster intel than ESPN's 15-min cache.")
    rw_col1, rw_col2 = st.columns([3,1])
    with rw_col2:
        rw_filter = st.text_input("Filter player", key="rw_filter_input")
    if rw_col1.button("Fetch Rotowire News", use_container_width=True, key="rw_fetch_btn"):
        fetch_rotowire_news.clear()
        with st.spinner("Scraping Rotowire..."):
            rw_rows, rw_err = fetch_rotowire_news()
        if rw_err:
            st.error(f"Rotowire: {rw_err}")
        elif not rw_rows:
            st.warning("No news found — Rotowire page structure may have changed.")
        else:
            st.session_state["rw_news"] = rw_rows
            st.success(f"Fetched {len(rw_rows)} Rotowire reports.")
    rw_news = st.session_state.get("rw_news", [])
    if rw_news:
        rw_df = pd.DataFrame(rw_news)
        if rw_filter:
            rw_df = rw_df[rw_df["player"].str.contains(rw_filter, case=False, na=False)]
        st.dataframe(rw_df, use_container_width=True)
        # Cross-reference with watchlist
        wl_cur = load_watchlist(st.session_state.get("user_id","trader"))
        if wl_cur:
            wl_norms = [normalize_name(p) for p in wl_cur]
            rw_wl = rw_df[rw_df["player"].apply(lambda p: normalize_name(p) in wl_norms)]
            if not rw_wl.empty:
                st.markdown("<div style='color:#FF3358;font-size:0.68rem;font-weight:600;margin-top:0.4rem;'>WATCHLIST ALERT — these players have Rotowire news:</div>", unsafe_allow_html=True)
                st.dataframe(rw_wl, use_container_width=True)

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:2rem;padding-top:0.8rem;border-top:1px solid #1E2D3D;
font-family:Fira Code,monospace;font-size:0.60rem;color:#2A3A4A;
display:flex;justify-content:space-between;'>
  <span>NBA QUANT ENGINE v4.0</span>
  <span>EXP DECAY | OPP SPLITS | HOT/COLD | O/U ASYMMETRY | CLV LEADERBOARD | BOOK EFFICIENCY | STEAM DETECTOR | ROTOWIRE | PARLAY BUILDER | DIGEST ALERTS | Q1/FANTASY MKTS</span>
  <span>Powered by Kamal</span>
</div>
""", unsafe_allow_html=True)
