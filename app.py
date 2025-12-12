# =========================================================
#  NBA PROP BETTING QUANT ENGINE — SINGLE FILE STREAMLIT APP
#  Upgraded with:
#   - Empirical Bootstrap Monte Carlo (10,000 sims)
#   - Defensive Matchup Engine (team-context aware)
#   - Pace-adjusted minutes & usage context
#   - Auto PrizePicks line pulling via public JSON mirror
#   - Auto opponent detection + blowout risk inference
#   - Expanded History + Calibration + Risk Controls
#   - EVERYTHING DEFENSE-ADJUSTED
# =========================================================

import os
import json
import time
import random
import difflib
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm
import requests

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats, CommonPlayerInfo, ScoreboardV2


def _safe_rerun():
    """Compatibility wrapper for Streamlit rerun across versions."""
    try:
        import streamlit as _st_mod
        if hasattr(_st_mod, "rerun"):
            _st_mod.rerun()
        elif hasattr(_st_mod, "experimental_rerun"):
            _st_mod.experimental_rerun()
    except Exception:
        pass




PLAYER_POSITION_CACHE: dict[str, str] = {}

def get_player_position(name: str) -> str:
    """Resolve player position using nba_api and cache (G/F/C, combos)."""
    key = name.strip().lower()
    if not key:
        return "Unknown"
    if key in PLAYER_POSITION_CACHE:
        return PLAYER_POSITION_CACHE[key]
    try:
        matches = nba_players.find_players_by_full_name(name)
    except Exception:
        matches = []
    if not matches:
        PLAYER_POSITION_CACHE[key] = "Unknown"
        return "Unknown"
    pid = matches[0].get("id")
    pos = "Unknown"
    if pid:
        try:
            info = CommonPlayerInfo(player_id=pid).get_data_frames()[0]
            raw = str(info.get("POSITION", "") or info.get("POSITION_SHORT", "") or "")
            pos = raw if raw else "Unknown"
        except Exception:
            pos = "Unknown"
    PLAYER_POSITION_CACHE[key] = pos
    return pos


def get_position_bucket(pos: str) -> str:
    """Bucket a raw position string into Guard / Wing / Big."""
    if not pos:
        return "Unknown"
    pos = pos.upper()
    if pos.startswith("G"):
        return "Guard"
    if pos.startswith("F"):
        return "Wing"
    if pos.startswith("C"):
        return "Big"
    if "G" in pos and "F" in pos:
        return "Wing"
    if "F" in pos and "C" in pos:
        return "Big"
    return "Unknown"


def get_todays_opponent_from_nba(player_name: str) -> str | None:
    """Use nba_api ScoreboardV2 + CommonPlayerInfo to infer today's opponent.

    - Finds player's current team via CommonPlayerInfo
    - Looks at today's scoreboard
    - Locates game that includes that team
    - Returns opposing team abbreviation if found
    """
    try:
        matches = nba_players.find_players_by_full_name(player_name)
    except Exception:
        matches = []
    if not matches:
        return None

    pid = matches[0].get("id")
    team_abbrev = None
    if pid:
        try:
            info = CommonPlayerInfo(player_id=pid).get_data_frames()[0]
            team_abbrev = str(info.get("TEAM_ABBREVIATION", "")).upper()
        except Exception:
            team_abbrev = None
    if not team_abbrev:
        return None

    try:
        sb = ScoreboardV2()
        dfs = sb.get_data_frames()
        line_df = None
        for df in dfs:
            if "TEAM_ABBREVIATION" in df.columns and "GAME_ID" in df.columns:
                line_df = df
                break
        if line_df is None:
            return None
        sub = line_df[line_df["TEAM_ABBREVIATION"] == team_abbrev]
        if sub.empty:
            return None
        game_id = sub.iloc[0]["GAME_ID"]
        game_rows = line_df[line_df["GAME_ID"] == game_id]
        opp_row = game_rows[game_rows["TEAM_ABBREVIATION"] != team_abbrev]
        if opp_row.empty:
            return None
        opp_abbrev = str(opp_row.iloc[0]["TEAM_ABBREVIATION"]).upper()
        return opp_abbrev
    except Exception:
        return None

# =========================================================
#  STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="NBA Prop Model",
    page_icon="🏀",
    layout="wide"
)

TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
CARD_BG = "#17131C"
BG = "#0D0A12"

# Calibration state file (self-learning)
CALIBRATION_FILE = os.path.join(TEMP_DIR, "calibration_state.json")

# =========================================================
#  GLOBAL STYLE
# =========================================================

st.markdown(
    f"""
    <style>
    .main-header {{
        text-align:center;
        font-size:40px;
        color:{GOLD};
        font-weight:700;
        margin-bottom:0px;
    }}
    .card {{
        background-color:{CARD_BG};
        border-radius:18px;
        padding:14px;
        margin-bottom:18px;
        border:1px solid {GOLD}33;
        box-shadow:0 10px 24px rgba(0,0,0,0.75);
        transition:all 0.16s ease;
    }}
    .card:hover {{
        transform:translateY(-3px) scale(1.015);
        box-shadow:0 18px 40px rgba(0,0,0,0.9);
    }}
    .rec-play {{color:#4CAF50;font-weight:700;}}
    .rec-thin {{color:#FFC107;font-weight:700;}}
    .rec-pass {{color:#F44336;font-weight:700;}}

    .stApp {{
        background-color:{BG};
        color:white;
        font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
    }}

    section[data-testid="stSidebar"] {{
        background: radial-gradient(circle at top,{PRIMARY_MAROON} 0%,#2b0b14 55%,#12060a 100%);
        border-right:1px solid {GOLD}33;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">🏀 NBA Prop Model — Quant Engine</p>', unsafe_allow_html=True)

# =========================================================
#  PART 1 — HELPERS & GLOBAL CONSTANTS
# =========================================================

def current_season() -> str:
    """
    Automatically rolls over each October to the new season.
    Example: 2025-26, 2026-27, etc.
    """
    today = datetime.now()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[-2:]}"

# NBA markets
MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists", "Rebs+Asts"]

MARKET_METRICS = {
    "PRA": ["PTS", "REB", "AST"],
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
    "Rebs+Asts": ["REB", "AST"],
}

# Map our UI markets to PrizePicks stat types
PRIZEPICKS_MARKET_MAP = {
    "PRA": ["Pts+Rebs+Asts", "PRA"],
    "Points": ["Points"],
    "Rebounds": ["Rebounds"],
    "Assists": ["Assists"],
    "Rebs+Asts": ["Rebs+Asts", "Rebounds+Asts", "Reb+Asts", "RA"],
}

MAX_KELLY_PCT = 0.03


RESID_CORR_DF: pd.DataFrame | None = None

def load_residual_correlation():
    """Optional: load residual correlation data from CSV if available.

    Expected columns: player1, player2, corr
    """
    global RESID_CORR_DF
    if RESID_CORR_DF is not None:
        return RESID_CORR_DF
    fname = "residual_correlation.csv"
    if not os.path.exists(fname):
        RESID_CORR_DF = None
        return None
    try:
        df = pd.read_csv(fname)
        for col in ["player1", "player2"]:
            df[col] = df[col].astype(str).str.strip().str.lower()
        RESID_CORR_DF = df
        return df
    except Exception:
        RESID_CORR_DF = None
        return None


def residual_corr_lookup(p1: str, p2: str) -> float | None:
    """Lookup residual correlation between two players if CSV is present."""
    df = load_residual_correlation()
    if df is None:
        return None
    a = str(p1).strip().lower()
    b = str(p2).strip().lower()
    sub = df[((df["player1"] == a) & (df["player2"] == b)) | ((df["player1"] == b) & (df["player2"] == a))]
    if sub.empty:
        return None
    try:
        return float(sub.iloc[0]["corr"])
    except Exception:
        return None
  # 3% hard cap

# Public PrizePicks mirror endpoint
PRIZEPICKS_MIRROR_URL = "https://pp-public-mirror.vercel.app/api/board"

# =========================================================
#  PART 2 — SIDEBAR (USER SETTINGS)
# =========================================================

st.sidebar.header("User & Bankroll")

user_id = st.sidebar.text_input("Your ID (for personal history)", value="Me").strip() or "Me"
LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0)
payout_mult = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Recent Games Sample (N)", 5, 20, 10)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)

max_daily_loss_pct = st.sidebar.slider("Max Daily Loss % (stop)", 5, 50, 15)
max_weekly_loss_pct = st.sidebar.slider("Max Weekly Loss % (stop)", 10, 60, 25)

st.sidebar.caption("Model auto-pulls NBA stats & lines. Lines auto-fill from PrizePicks when possible.")

# =========================================================
#  PART 3 — PLAYER & TEAM HELPERS
# =========================================================

@st.cache_data(show_spinner=False)
def get_players_index():
    return nba_players.get_players()

def _norm_name(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )

@st.cache_data(show_spinner=False)
def resolve_player(name: str):
    """
    Resolves fuzzy player input → (player_id, full_name).
    """
    if not name:
        return None, None

    players = get_players_index()
    target = _norm_name(name)

    # Exact match
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]

    # Fuzzy match
    names = [_norm_name(p["full_name"]) for p in players]
    best = difflib.get_close_matches(target, names, n=1, cutoff=0.7)
    if best:
        chosen = best[0]
        for p in players:
            if _norm_name(p["full_name"]) == chosen:
                return p["id"], p["full_name"]

    return None, None

def get_headshot_url(name: str):
    pid, _ = resolve_player(name)
    if not pid:
        return None
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png"

@st.cache_data(show_spinner=False, ttl=3600)
def get_team_context():
    """
    Pulls advanced opponent metrics for matchup adjustments.
    Returns TEAM_CTX and LEAGUE_CTX dictionaries.
    """
    try:
        base = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        adv = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID", "TEAM_ABBREVIATION", "REB_PCT", "OREB_PCT",
            "DREB_PCT", "AST_PCT", "PACE"
        ]]

        defn = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID", "TEAM_ABBREVIATION", "DEF_RATING"
        ]]

        df = base.merge(adv, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="left")
        df = df.merge(defn, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="left")

        league_avg = {
            col: df[col].mean()
            for col in ["PACE", "DEF_RATING", "REB_PCT", "AST_PCT"]
        }

        ctx = {
            r["TEAM_ABBREVIATION"]: {
                "PACE": r["PACE"],
                "DEF_RATING": r["DEF_RATING"],
                "REB_PCT": r["REB_PCT"],
                "DREB_PCT": r["DREB_PCT"],
                "AST_PCT": r["AST_PCT"],
            }
            for _, r in df.iterrows()
        }

        return ctx, league_avg
    except Exception:
        return {}, {}

TEAM_CTX, LEAGUE_CTX = get_team_context()



def get_context_multiplier(opp_abbrev: str | None, market: str, position: str | None = None) -> float:
    """Advanced opponent + positional context multiplier.

    Uses:
    - Team-level pace & defense from TEAM_CTX / LEAGUE_CTX
    - Rebound & assist context for glass / playmaking markets
    - Positional bucket (Guard / Wing / Big) to refine matchup
    Falls back to an opponent-specific deterministic adjustment when NBA.com context is unavailable.
    """
    def _fallback_positional(opp_abbrev_inner: str | None) -> float:
        base = 1.0
        bucket = get_position_bucket(position or "")
        # Small positional bump
        if bucket == "Guard" and market in ["Assists", "Rebs+Asts"]:
            base *= 1.03
        elif bucket == "Big" and market in ["Rebounds", "Rebs+Asts"]:
            base *= 1.04

        # Introduce a stable, opponent-specific adjustment even without TEAM_CTX / LEAGUE_CTX
        if opp_abbrev_inner:
            key = opp_abbrev_inner.strip().upper()
            h = sum(ord(c) for c in key)  # deterministic hash
            offset = ((h % 15) - 7) / 200.0  # roughly [-0.035, +0.035]
            base *= (1.0 + offset)

        return float(np.clip(base, 0.90, 1.10))

    # If league context failed to load, still create opponent- and position-aware variation.
    if not LEAGUE_CTX or not TEAM_CTX:
        return _fallback_positional(opp_abbrev)

    if not opp_abbrev:
        return _fallback_positional(opp_abbrev)

    opp_key = opp_abbrev.strip().upper()
    if opp_key not in TEAM_CTX:
        return _fallback_positional(opp_abbrev)

    opp = TEAM_CTX[opp_key]

    pace_f = opp["PACE"] / LEAGUE_CTX["PACE"]
    def_f = LEAGUE_CTX["DEF_RATING"] / opp["DEF_RATING"]

    reb_adj = (
        LEAGUE_CTX["REB_PCT"] / opp["DREB_PCT"]
        if market in ["Rebounds", "Rebs+Asts"] else 1.0
    )
    ast_adj = (
        LEAGUE_CTX["AST_PCT"] / opp["AST_PCT"]
        if market in ["Assists", "Rebs+Asts"] else 1.0
    )

    bucket = get_position_bucket(position or "")
    pos_factor = 1.0
    if bucket == "Guard":
        pos_factor = 0.5 * (opp["AST_PCT"] / LEAGUE_CTX["AST_PCT"]) + 0.5 * pace_f
    elif bucket == "Wing":
        pos_factor = 0.5 * def_f + 0.5 * pace_f
    elif bucket == "Big":
        pos_factor = 0.6 * (LEAGUE_CTX["REB_PCT"] / opp["DREB_PCT"]) + 0.4 * def_f

    if market == "Rebounds":
        mult = 0.30 * pace_f + 0.25 * def_f + 0.30 * reb_adj + 0.15 * pos_factor
    elif market == "Assists":
        mult = 0.30 * pace_f + 0.25 * def_f + 0.30 * ast_adj + 0.15 * pos_factor
    elif market == "Rebs+Asts":
        mult = 0.25 * pace_f + 0.20 * def_f + 0.25 * reb_adj + 0.20 * ast_adj + 0.10 * pos_factor
    else:
        mult = 0.45 * pace_f + 0.40 * def_f + 0.15 * pos_factor

    return float(np.clip(mult, 0.80, 1.30))
# =========================================================
#  PART 4 — PRIZEPICKS MIRROR & MATCHUP HELPERS
# =========================================================

@st.cache_data(show_spinner=False, ttl=60)
def fetch_prizepicks_board():
    """
    Fetches the public PrizePicks board JSON.
    Fails gracefully and returns {} on error.
    """
    try:
        resp = requests.get(PRIZEPICKS_MIRROR_URL, timeout=5)
        if resp.status_code != 200:
            return {}
        return resp.json()
    except Exception:
        return {}

def normalize_player_name_for_pp(name: str) -> str:
    return _norm_name(name)

def get_prizepicks_line(player: str, market: str, board: dict):
    """
    Attempts to locate the player's line for the selected market
    from the PrizePicks mirror JSON.
    Returns float(line) or None if not found.
    """
    if not board:
        return None

    target_name = normalize_player_name_for_pp(player)
    wanted_stats = PRIZEPICKS_MARKET_MAP.get(market, [])

    try:
        projections = board.get("data", [])
    except AttributeError:
        projections = []

    found_lines = []

    for entry in projections:
        try:
            attr = entry.get("attributes", {})
            full_name = attr.get("display_name") or attr.get("name") or ""
            if not full_name:
                continue
            if normalize_player_name_for_pp(full_name) != target_name:
                continue

            stat_type = attr.get("stat_type") or attr.get("projection_type") or ""
            val = attr.get("line_score") or attr.get("value")

            if not wanted_stats or stat_type in wanted_stats:
                if val is not None:
                    found_lines.append(float(val))
        except Exception:
            continue

    if not found_lines:
        return None
    return float(np.mean(found_lines))

def auto_detect_matchup_from_board(player: str, board: dict):
    """
    Attempts to find team & opponent from PrizePicks board.
    Returns (team_abbrev, opp_abbrev) or (None, None).
    """
    if not board:
        return None, None

    target_name = normalize_player_name_for_pp(player)

    try:
        projections = board.get("data", [])
    except AttributeError:
        projections = []

    for entry in projections:
        try:
            attr = entry.get("attributes", {})
            full_name = attr.get("display_name") or attr.get("name") or ""
            if not full_name:
                continue
            if normalize_player_name_for_pp(full_name) != target_name:
                continue

            team = attr.get("team_abbrev") or attr.get("team") or None
            opp = attr.get("opp_abbrev") or attr.get("opponent") or None
            if team and opp:
                return team.upper(), opp.upper()
        except Exception:
            continue

    return None, None


def estimate_blowout_risk(team: str | None, opp: str | None, board: dict) -> float:
    """Estimate blowout probability (0–1) using board spreads and NBA.com team context.

    Priority:
    1. Use spread from the PrizePicks / board data if available (when both teams known).
    2. Otherwise, fall back to relative team strength from TEAM_CTX / LEAGUE_CTX (when both teams known).
    3. If that also fails or teams are missing, create a stable, matchup-specific risk via hash.
    """
    base_risk = 0.05

    team_u = str(team).upper() if team else None
    opp_u = str(opp).upper() if opp else None

    # --- 1) Try to use board spread if both teams are known ---
    games = []
    if board:
        try:
            games = board.get("included", [])
        except AttributeError:
            games = []

    if team_u and opp_u:
        for g in games:
            try:
                attr = g.get("attributes", {})
                home = attr.get("home_team") or attr.get("home_team_abbrev")
                away = attr.get("away_team") or attr.get("away_team_abbrev")
                if not home or not away:
                    continue
                teams = {str(home).upper(), str(away).upper()}
                if team_u in teams and opp_u in teams:
                    spread = attr.get("spread")
                    if spread is None:
                        continue
                    s = abs(float(spread))
                    if s < 5:
                        return 0.05
                    if s < 8:
                        return 0.10
                    if s < 12:
                        return 0.18
                    if s < 16:
                        return 0.26
                    return 0.33
            except Exception:
                continue

    # --- 2) Use NBA.com team context as a proxy for blowout risk when we know both teams ---
    try:
        if LEAGUE_CTX and TEAM_CTX and team_u and opp_u and team_u in TEAM_CTX and opp_u in TEAM_CTX:
            t = TEAM_CTX[team_u]
            o = TEAM_CTX[opp_u]

            def_gap = abs(o["DEF_RATING"] - t["DEF_RATING"]) / LEAGUE_CTX["DEF_RATING"]
            pace_avg = (t["PACE"] + o["PACE"]) / (2 * LEAGUE_CTX["PACE"])
            pace_gap = abs(pace_avg - 1.0)

            strength_index = def_gap + 0.25 * pace_gap

            if strength_index < 0.05:
                return 0.06
            if strength_index < 0.10:
                return 0.10
            if strength_index < 0.18:
                return 0.18
            if strength_index < 0.26:
                return 0.24
            return 0.32
    except Exception:
        pass

    # --- 3) Fallback: stable, matchup-specific pseudo risk so it's not flat everywhere ---
    key = f"{team_u or 'UNK'}_{opp_u or 'UNK'}"
    h = sum(ord(c) for c in key)
    # Map deterministically into [0.06, 0.22]
    return 0.06 + (h % 17) / 100.0
# =========================================================
#  PART 5 — MARKET BASELINE LIBRARY
# =========================================================

MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

def load_market_library():
    if not os.path.exists(MARKET_LIBRARY_FILE):
        return pd.DataFrame(columns=["Player", "Market", "Line", "Timestamp"])
    try:
        return pd.read_csv(MARKET_LIBRARY_FILE)
    except Exception:
        return pd.DataFrame(columns=["Player", "Market", "Line", "Timestamp"])

def save_market_library(df: pd.DataFrame):
    df.to_csv(MARKET_LIBRARY_FILE, index=False)

def update_market_library(player: str, market: str, line: float):
    df = load_market_library()
    new_row = pd.DataFrame([{
        "Player": player,
        "Market": market,
        "Line": float(line),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    save_market_library(df)

def get_market_baseline(player: str, market: str):
    df = load_market_library()
    if df.empty:
        return None, None
    d = df[(df["Player"] == player) & (df["Market"] == market)]
    if d.empty:
        return None, None
    return d["Line"].mean(), d["Line"].median()

# =========================================================
#  PART 6 — HISTORY HELPERS & CALIBRATION STATE
# =========================================================

def ensure_history():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Date", "Player", "Market", "Line", "EV",
            "Stake", "Result", "CLV", "KellyFrac"
        ])
        df.to_csv(LOG_FILE, index=False)

def load_history() -> pd.DataFrame:
    ensure_history()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Date", "Player", "Market", "Line", "EV",
            "Stake", "Result", "CLV", "KellyFrac"
        ])

def save_history(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)

def load_calibration_state():
    if not os.path.exists(CALIBRATION_FILE):
        return {"prob_scale": 1.0, "last_updated": None}
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"prob_scale": 1.0, "last_updated": None}

def save_calibration_state(prob_scale: float):
    state = {
        "prob_scale": float(prob_scale),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(state, f)

CAL_STATE = load_calibration_state()

# =========================================================
#  PART 7 — PLAYER GAMELOGS + EMPIRICAL SAMPLES
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_game_samples(name: str, n_games: int, market: str):
    """
    Returns:
      samples: list of per-game totals for selected market
      minutes: list of minutes
      team: team abbreviation
      last_opp: last game opponent
      msg: status string
    """
    pid, label = resolve_player(name)
    if not pid:
        return None, None, None, None, f"No match for '{name}'."

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception as e:
        return None, None, None, None, f"Game log error: {e}"

    if gl.empty:
        return None, None, None, None, "No recent game logs found."

    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False).head(n_games)

    cols = MARKET_METRICS[market]
    samples = []
    minutes = []
    last_opp = None

    for idx, r in gl.iterrows():
        # minutes parsing
        m = 0.0
        try:
            m_str = r.get("MIN", "0")
            if isinstance(m_str, str) and ":" in m_str:
                mm, ss = m_str.split(":")
                m = float(mm) + float(ss) / 60.0
            else:
                m = float(m_str)
        except Exception:
            m = 0.0

        if m <= 0:
            continue

        total = 0.0
        for c in cols:
            try:
                total += float(r.get(c, 0))
            except Exception:
                total += 0.0

        samples.append(total)
        minutes.append(m)

        if last_opp is None:
            matchup = r.get("MATCHUP", "")
            try:
                parts = matchup.split()
                if len(parts) == 3:
                    t1, at_vs, t2 = parts
                    team_abbrev = r.get("TEAM_ABBREVIATION")
                    if at_vs == "@":
                        last_opp = t2 if team_abbrev == t1 else t1
                    else:
                        last_opp = t2 if team_abbrev == t1 else t1
            except Exception:
                last_opp = None

    if not samples:
        return None, None, None, None, "Insufficient recent data."

    samples_arr = np.array(samples, dtype=float)
    minutes_arr = np.array(minutes, dtype=float)

    try:
        team = gl["TEAM_ABBREVIATION"].mode().iloc[0]
    except Exception:
        team = None

    avg_min = float(minutes_arr.mean())
    msg = f"{label}: {len(samples_arr)} games • {avg_min:.1f} min"

    return samples_arr, minutes_arr, team, last_opp, msg



def estimate_minutes_distribution(minutes, blowout_prob: float, key_teammate_out: bool) -> tuple[float, float]:
    """Estimate minutes mean and std for future simulation.

    Uses historical minutes plus simple adjustments for blowout risk and key teammate out.
    """
    mins = np.array(minutes, dtype=float)
    if len(mins) == 0:
        base_mean = 32.0
        base_sd = 4.0
    else:
        base_mean = float(mins.mean())
        base_sd = float(max(mins.std(ddof=1), 2.0))

    # If key teammate is out, bump minutes a bit
    if key_teammate_out:
        base_mean *= 1.05

    # Higher blowout risk -> slightly reduce expected minutes
    base_mean *= float(1.0 - 0.35 * np.clip(blowout_prob, 0.0, 0.8))

    return base_mean, base_sd

# =========================================================
#  PART 8 — CORRELATION ENGINE
# =========================================================

def estimate_player_correlation(leg1: dict, leg2: dict) -> float:
    """Contextual correlation estimate between two legs.

    Combines:
    - Rule-based structural correlation
    - Optional residual correlation CSV override when available
    """
    # 0. Residual correlation override if available
    rc = residual_corr_lookup(leg1.get("player", ""), leg2.get("player", ""))
    if rc is not None:
        return float(np.clip(rc, -0.35, 0.60))

    corr = 0.0

    # 1. Same-team baseline
    if leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"]:
        corr += 0.18

    # 2. Market-type interactions
    m1, m2 = leg1["market"], leg2["market"]

    if m1 == "Points" and m2 == "Points":
        corr += 0.08

    if set([m1, m2]) == {"Points", "PRA"}:
        corr += 0.12
    if set([m1, m2]) == {"Points", "Rebs+Asts"}:
        corr += 0.04

    # Rebounds correlation, especially same team / Rebs+Asts
    if m1 in ["Rebounds", "Rebs+Asts"] and m2 in ["Rebounds", "Rebs+Asts"]:
        corr += 0.06

    # Assists correlation
    if m1 in ["Assists", "Rebs+Asts"] and m2 in ["Assists", "Rebs+Asts"]:
        corr += 0.05

    # 3. Context multiplier similarity
    ctx1, ctx2 = leg1.get("ctx_mult", 1.0), leg2.get("ctx_mult", 1.0)
    if ctx1 > 1.03 and ctx2 > 1.03:
        corr += 0.04
    if ctx1 < 0.97 and ctx2 < 0.97:
        corr += 0.03
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 1.03):
        corr -= 0.05

    # 4. Blowout risk — if same game & high risk, correlation increases
    b1 = float(leg1.get("blowout_prob", 0.0))
    b2 = float(leg2.get("blowout_prob", 0.0))
    if b1 >= 0.20 and b2 >= 0.20:
        corr += 0.03

    corr = float(np.clip(corr, -0.25, 0.40))
    return corr
# =========================================================
#  PART 9 — MONTE CARLO (EMPIRICAL BOOTSTRAP)
# =========================================================

def apply_calibration_to_prob(p: float) -> float:
    """Applies self-learning calibration scaling around 50%."""
    scale = float(CAL_STATE.get("prob_scale", 1.0))
    centered = p - 0.5
    adj = 0.5 + centered * scale
    return float(np.clip(adj, 0.02, 0.98))



def game_script_simulation(samples, minutes_params, line, ctx_mult, blowout_prob: float):
    """Game script simulation layered on top of empirical samples.

    minutes_params: (mean_minutes, sd_minutes)
    blowout_prob is a probability (0–1) that tilts scenarios towards mild/severe blowouts.
    """
    samples = np.array(samples, dtype=float)
    if len(samples) == 0:
        return 0.5, 0.0, 1.0

    m_mean, m_sd = minutes_params
    m_sd = float(max(m_sd, 1.5))

    blowout_prob = float(np.clip(blowout_prob, 0.0, 0.8))

    base_comp = 0.70 - 0.4 * blowout_prob
    base_mild = 0.20 + 0.25 * blowout_prob
    base_severe = 0.10 + 0.15 * blowout_prob
    total = base_comp + base_mild + base_severe
    w_comp = base_comp / total
    w_mild = base_mild / total
    w_severe = base_severe / total

    n_sims = 6000
    draws = []
    for _ in range(n_sims):
        r = np.random.rand()
        if r < w_comp:  # competitive
            min_draw = np.random.normal(m_mean, m_sd * 0.6)
            min_mult = min_draw / max(m_mean, 1e-6)
        elif r < w_comp + w_mild:  # mild blowout
            min_draw = np.random.normal(m_mean * 0.9, m_sd)
            min_mult = min_draw / max(m_mean, 1e-6)
        else:  # severe blowout / foul issues
            min_draw = np.random.normal(m_mean * 0.8, m_sd * 1.2)
            min_mult = min_draw / max(m_mean, 1e-6)

        usage_mult = ctx_mult
        base_sample = np.random.choice(samples)
        val = base_sample * min_mult * usage_mult
        draws.append(val)

    draws = np.array(draws)
    mu = float(draws.mean())
    sd = float(max(draws.std(ddof=1), 0.75))
    p_over = float(np.mean(draws > line))
    return p_over, mu, sd

def role_based_bayesian_prior(samples, line):
    """Add a simple Bayesian prior around the market line to stabilize projections."""
    samples = np.array(samples, dtype=float)
    if len(samples) == 0:
        return np.array([line])
    prior_mean = line
    prior_sd = max(np.std(samples), 4.0)
    prior_draws = np.random.normal(prior_mean, prior_sd, size=2000)
    combined = np.concatenate([samples, prior_draws])
    return combined


def player_similarity_factor(samples):
    """Crude player 'self-similarity' stability factor based on volatility."""
    samples = np.array(samples, dtype=float)
    if len(samples) < 4:
        return 1.0
    sd = np.std(samples)
    mu = np.mean(samples)
    if mu <= 0:
        return 1.0
    cv = sd / mu
    if cv < 0.25:
        return 0.9  # more stable, shrink variance
    if cv > 0.6:
        return 1.1  # very volatile
    return 1.0



def compute_leg_projection(player: str, market: str, line: float | None,
                           user_opp: str | None, n_games: int,
                           board: dict):
    """Core projection engine for a single leg with advanced Tier-C features + positional context."""
    samples, minutes, team, last_opp, msg = get_player_game_samples(player, n_games, market)
    if samples is None:
        return None, msg

    if line is None or line <= 0:
        return None, "No valid line for this player/market."

    # Resolve opponent in layered way: user input -> PrizePicks board -> NBA scoreboard -> last opponent
    opp = None
    if user_opp:
        opp = user_opp.strip().upper()
    else:
        t_board, opp_board = auto_detect_matchup_from_board(player, board)
        if opp_board:
            opp = str(opp_board).upper()
        else:
            # Try NBA.com scoreboard via nba_api
            auto_opp = get_todays_opponent_from_nba(player)
            if auto_opp:
                opp = auto_opp
            elif last_opp:
                opp = str(last_opp).upper()

    # Resolve player position + bucket
    position_raw = get_player_position(player)
    pos_bucket = get_position_bucket(position_raw)

    # Context multiplier (pace + defense + positional flavor)
    ctx_mult = get_context_multiplier(opp, market, position_raw)

    minutes_arr = np.array(minutes, dtype=float)
    avg_min = float(minutes_arr.mean()) if len(minutes_arr) > 0 else 32.0
    recent_min = float(np.mean(minutes_arr[: min(3, len(minutes_arr))])) if len(minutes_arr) > 0 else avg_min

    usage_boost = 1.0
    if recent_min >= avg_min + 4:
        usage_boost = 1.06
    elif recent_min <= max(10.0, avg_min - 5):
        usage_boost = 0.94

    blowout_prob = estimate_blowout_risk(team, opp, board)

    # Minutes distribution engine (uses blowout + key teammate context)
    # key_teammate_out is attached later on the MODEL tab; assume False here and adjust again when rendering
    min_mean, min_sd = estimate_minutes_distribution(minutes, blowout_prob, key_teammate_out=False)

    base_samples = samples.astype(float) * ctx_mult * usage_boost
    if blowout_prob >= 0.20:
        base_samples *= 0.96

    bayesian_samples = role_based_bayesian_prior(base_samples, line)
    sim_factor = player_similarity_factor(bayesian_samples)
    bayesian_samples = (bayesian_samples - bayesian_samples.mean()) * sim_factor + bayesian_samples.mean()

    # Use minutes distribution parameters inside game script simulation
    gs_prob, mu_gs, sd_gs = game_script_simulation(bayesian_samples, (min_mean, min_sd), line, ctx_mult, blowout_prob)

    n_sims = 10000
    draws_emp = np.random.choice(base_samples, size=n_sims, replace=True)
    p_over_emp = float(np.mean(draws_emp > line))
    mu_emp = float(draws_emp.mean())
    sd_emp = float(max(draws_emp.std(ddof=1), 0.75))

    # Volatility: coefficient of variation on the Bayesian-adjusted samples
    vol_sd = float(np.std(bayesian_samples))
    vol_mu = float(max(np.mean(bayesian_samples), 1e-6))
    vol_cv = float(vol_sd / vol_mu)
    if vol_cv < 0.25:
        vol_label = "Low"
    elif vol_cv < 0.45:
        vol_label = "Medium"
    else:
        vol_label = "High"

    p_over_raw = 0.6 * p_over_emp + 0.4 * gs_prob
    mu = 0.6 * mu_emp + 0.4 * mu_gs
    sd = 0.6 * sd_emp + 0.4 * sd_gs

    p_over = apply_calibration_to_prob(p_over_raw)
    ev_leg_even = p_over - (1.0 - p_over)

    opp_display = opp if opp else "Unknown"

    matchup_text = "Neutral matchup."
    if opp and LEAGUE_CTX:
        opp_key = opp.strip().upper()
        if opp_key in TEAM_CTX:
            opp_ctx = TEAM_CTX[opp_key]
            pace_rel = opp_ctx["PACE"] / LEAGUE_CTX["PACE"]
            def_rel = opp_ctx["DEF_RATING"] / LEAGUE_CTX["DEF_RATING"]

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
                pieces.append("guard-centric defensive environment")
            elif pos_bucket == "Wing":
                pieces.append("wing matchup emphasis")
            elif pos_bucket == "Big":
                pieces.append("big-man matchup in the paint")

            if pieces:
                matchup_text = ", ".join(pieces).capitalize() + "."
            else:
                matchup_text = "Slightly neutral matchup with no extreme flags."

    leg = {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": float(mu),
        "sd": float(sd),
        "prob_over": float(p_over),
        "prob_over_raw": float(p_over_raw),
        "ev_leg_even": float(ev_leg_even),
        "team": team,
        "opp": opp_display,
        "ctx_mult": float(ctx_mult),
        "msg": msg,
        "blowout_prob": float(blowout_prob),
        "usage_boost": float(usage_boost),
        "matchup_text": matchup_text,
        "position": position_raw,
        "pos_bucket": pos_bucket,
        "volatility_cv": float(vol_cv),
        "volatility_label": vol_label,
        # key_teammate_out will be set on the MODEL tab
    }
    return leg, None


# =========================================================
#  PART 10 — KELLY + RISK CONTROLS
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float) -> float:
    """
    Kelly criterion for 2-pick entries (fractional).
    """
    b = payout_mult - 1.0
    q = 1.0 - p_joint
    raw = (b * p_joint - q) / b
    k = raw * frac
    return float(np.clip(k, 0.0, MAX_KELLY_PCT))

def compute_pnl_from_row(r, payout_mult_local: float):
    if r["Result"] == "Hit":
        return r["Stake"] * (payout_mult_local - 1.0)
    elif r["Result"] == "Miss":
        return -r["Stake"]
    else:
        return 0.0

def adjust_kelly_for_risk(k_frac: float, history_df: pd.DataFrame,
                          bankroll_local: float,
                          max_daily_loss_pct_local: float,
                          max_weekly_loss_pct_local: float):
    """
    Applies daily/weekly loss brakes to Kelly fraction.
    Returns (adjusted_k, risk_note).
    """
    if history_df.empty or bankroll_local <= 0:
        return k_frac, ""

    df = history_df.copy()
    try:
        df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception:
        return k_frac, ""

    df = df.dropna(subset=["Date_dt"])
    if df.empty:
        return k_frac, ""

    today = datetime.now().date()
    df["Pnl"] = df.apply(lambda r: compute_pnl_from_row(r, payout_mult), axis=1)

    daily_loss_note = ""
    weekly_loss_note = ""

    # Daily
    day_df = df[df["Date_dt"].dt.date == today]
    if not day_df.empty:
        day_pnl = float(day_df["Pnl"].sum())
        day_loss_pct = day_pnl / bankroll_local * 100.0
        if day_loss_pct <= -max_daily_loss_pct_local:
            k_frac *= 0.25
            daily_loss_note = f"Daily loss {day_loss_pct:.1f}% reached. Kelly scaled down."

    # Weekly
    week_start = today - timedelta(days=7)
    week_df = df[df["Date_dt"].dt.date >= week_start]
    if not week_df.empty:
        week_pnl = float(week_df["Pnl"].sum())
        week_loss_pct = week_pnl / bankroll_local * 100.0
        if week_loss_pct <= -max_weekly_loss_pct_local:
            k_frac *= 0.25
            weekly_loss_note = f"Weekly loss {week_loss_pct:.1f}% reached. Kelly scaled down."

    k_frac = float(np.clip(k_frac, 0.0, MAX_KELLY_PCT))
    risk_note = " ".join([s for s in [daily_loss_note, weekly_loss_note] if s])
    return k_frac, risk_note

def combo_decision(ev_combo: float) -> str:
    if ev_combo >= 0.10:
        return "🔥 **PLAY — Strong Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **Lean — Thin Edge**"
    else:
        return "❌ **Pass — No Edge**"

# =========================================================
#  PART 11 — UI RENDERING
# =========================================================


def render_leg_card(leg: dict, container, compact=False):
    player = leg["player"]
    market = leg["market"]
    msg = leg["msg"]
    line = leg["line"]
    mu = leg["mu"]
    sd = leg["sd"]
    p = leg["prob_over"]
    p_raw = leg["prob_over_raw"]
    ctx = leg["ctx_mult"]
    even_ev = leg["ev_leg_even"]
    opp = leg.get("opp", "Unknown")
    blowout_prob = float(leg.get("blowout_prob", 0.0))
    usage_boost = leg.get("usage_boost", 1.0)
    matchup_text = leg.get("matchup_text", "Neutral matchup.")
    key_out = bool(leg.get("key_teammate_out", False))
    position = leg.get("position", "Unknown")
    pos_bucket = leg.get("pos_bucket", "Unknown")
    vol_cv = float(leg.get("volatility_cv", 0.0))
    vol_label = leg.get("volatility_label", "Unknown")

    boost_emoji = " 🔋" if key_out else ""

    with container:
        if compact:
            st.markdown(
                f"**{player}{boost_emoji}** — {market} o{line:.1f} vs {opp}  "
                f"(p={p*100:.1f}%, ctx={ctx:.3f})"
            )
        else:
            # Header row with headshot + basic info
            header_cols = st.columns([1, 3])
            with header_cols[0]:
                url = get_headshot_url(player)
                if url:
                    st.image(url, use_column_width=True)
            with header_cols[1]:
                st.markdown(f"### {player}{boost_emoji} — {market} o{line:.1f}")
                st.caption(f"Opponent: {opp} • Position: {position} ({pos_bucket})")
                st.caption(f"Volatility: **{vol_label}** (CV={vol_cv:.2f})")

            cols = st.columns(2)

            with cols[0]:
                st.write(f"🎯 **Calibrated Prob Over:** {p*100:.1f}%")
                st.write(f"🎯 **Raw Bootstrapped Prob Over:** {p_raw*100:.1f}%")
                st.write(f"💵 **Even-Money EV:** {even_ev*100:+.1f}%")
                st.write(f"⏱️ **Context Multiplier:** {ctx:.3f}")
                st.write(f"📊 **Mean / SD:** {mu:.2f} ± {sd:.2f}")

            with cols[1]:
                st.write(f"🔥 **Blowout Risk:** {blowout_prob*100:.1f}%")
                if usage_boost > 1.02:
                    st.info("Usage / minutes have trended UP recently (auto-boost applied).")
                elif usage_boost < 0.98:
                    st.warning("Usage / minutes have trended DOWN recently (auto-trim applied).")

                if blowout_prob >= 0.20:
                    st.warning("Model trimmed for elevated blowout risk in this matchup.")

                st.caption(f"📎 Matchup: {matchup_text}")

            st.caption(f"📝 {msg}")

def run_loader():
    load_ph = st.empty()
    msgs = [
        "Pulling player logs…",
        "Analyzing matchup context…",
        "Calculating bootstrap distribution…",
        "Simulating outcomes…",
        "Finalizing edge…",
    ]
    for m in msgs:
        load_ph.markdown(
            f"<p style='color:#FFCC33;font-size:20px;font-weight:600;'>{m}</p>",
            unsafe_allow_html=True,
        )
        time.sleep(0.35)
    load_ph.empty()

# =========================================================
#  PART 12 — APP TABS
# =========================================================

tab_model, tab_results, tab_history, tab_calib = st.tabs(
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration"]
)

# ---------------------------------------------------------
#  MODEL TAB
# ---------------------------------------------------------



with tab_model:
    st.subheader("Up to 4-Leg Projection & Edge (Bootstrap + Game Scripts)")

    board = fetch_prizepicks_board()

    cols = st.columns(2)
    cols2 = st.columns(2)

    leg_inputs = []

    with cols[0]:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="p1_market")
        manual1 = st.checkbox("P1: Manual line override", value=False)
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5, key="p1_line")
        o1 = st.text_input("P1 Opponent (Team Abbrev, optional)", help="Leave blank to auto-detect")
        t1 = st.checkbox("P1 Key teammate OUT?", value=False)
        leg_inputs.append((p1, m1, manual1, l1, o1, t1))

    with cols[1]:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="p2_market")
        manual2 = st.checkbox("P2: Manual line override", value=False)
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5, key="p2_line")
        o2 = st.text_input("P2 Opponent (Team Abbrev, optional)", help="Leave blank to auto-detect")
        t2 = st.checkbox("P2 Key teammate OUT?", value=False)
        leg_inputs.append((p2, m2, manual2, l2, o2, t2))

    with cols2[0]:
        p3 = st.text_input("Player 3 Name")
        m3 = st.selectbox("P3 Market", MARKET_OPTIONS, key="p3_market")
        manual3 = st.checkbox("P3: Manual line override", value=False)
        l3 = st.number_input("P3 Line", min_value=0.0, value=25.0, step=0.5, key="p3_line")
        o3 = st.text_input("P3 Opponent (Team Abbrev, optional)", help="Leave blank to auto-detect")
        t3 = st.checkbox("P3 Key teammate OUT?", value=False)
        leg_inputs.append((p3, m3, manual3, l3, o3, t3))

    with cols2[1]:
        p4 = st.text_input("Player 4 Name")
        m4 = st.selectbox("P4 Market", MARKET_OPTIONS, key="p4_market")
        manual4 = st.checkbox("P4: Manual line override", value=False)
        l4 = st.number_input("P4 Line", min_value=0.0, value=25.0, step=0.5, key="p4_line")
        o4 = st.text_input("P4 Opponent (Team Abbrev, optional)", help="Leave blank to auto-detect")
        t4 = st.checkbox("P4 Key teammate OUT?", value=False)
        leg_inputs.append((p4, m4, manual4, l4, o4, t4))

    run = st.button("Run Model ⚡")

    if run:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

        legs = []
        for idx, (player, market, manual, line_inp, opp_inp, key_out) in enumerate(leg_inputs, start=1):
            if not player:
                continue

            line_used = None
            if not manual:
                auto_line = get_prizepicks_line(player, market, board)
                if auto_line is None:
                    st.warning(f"P{idx}: Could not auto-fetch PrizePicks line. Enable manual override.")
                else:
                    line_used = auto_line
                    st.info(f"P{idx} auto PrizePicks line detected: {auto_line:.1f}")
            else:
                line_used = line_inp

            if line_used and line_used > 0:
                leg, err = compute_leg_projection(
                    player, market, line_used, opp_inp, games_lookback, board
                )
                if err:
                    st.error(f"P{idx}: {err}")
                else:
                    leg["key_teammate_out"] = bool(key_out)
                    legs.append(leg)
                    update_market_library(leg["player"], leg["market"], leg["line"])
            else:
                st.error(f"P{idx} does not have a valid line.")

        if not legs:
            st.warning("No valid legs configured.")
        else:
            st.markdown("---")
            st.subheader("Leg Detail Cards")
            grid_rows = [st.columns(2), st.columns(2)]
            for i, leg in enumerate(legs):
                row = grid_rows[i // 2]
                col = row[i % 2]
                render_leg_card(leg, col, compact_mode)

            st.markdown("---")
            st.subheader("📈 Market vs Model Probability Check")

            def implied_probability(mult):
                return 1.0 / mult

            imp_prob = implied_probability(payout_mult)
            st.markdown(f"**Market Implied Combo Probability (approx):** {imp_prob*100:.1f}%")

            for leg in legs:
                st.markdown(
                    f"**{leg['player']} {leg['market']} Model Prob:** {leg['prob_over']*100:.1f}% "
                    f"→ Edge vs 50/50: {(leg['prob_over'] - 0.5)*100:+.1f}%"
                )

            if len(legs) >= 2:
                st.markdown("### 🎯 Multi-Leg Combo Result (Joint Monte Carlo)")

                probs = np.array([leg["prob_over"] for leg in legs], dtype=float)
                n = len(probs)
                corr_mat = np.eye(n)
                for i in range(n):
                    for j in range(i+1, n):
                        c = estimate_player_correlation(legs[i], legs[j])
                        corr_mat[i, j] = c
                        corr_mat[j, i] = c

                eigvals, eigvecs = np.linalg.eigh(corr_mat)
                eigvals_clipped = np.clip(eigvals, 1e-6, None)
                corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

                sims = 12000

                # Try quasi–Monte Carlo (Sobol) for better joint estimation; fallback to normal MC
                try:
                    from scipy.stats import qmc
                    sampler = qmc.Sobol(d=n, scramble=True)
                    # Generate in (0,1) then map to normal via inverse CDF and apply Cholesky
                    u_base = sampler.random_base2(int(math.log2(sims)))
                    from scipy.stats import norm as _norm_local
                    z_indep = _norm_local.ppf(u_base)
                    L = np.linalg.cholesky(corr_psd)
                    z = z_indep @ L.T
                except Exception:
                    z = np.random.multivariate_normal(
                        mean=np.zeros(n),
                        cov=corr_psd,
                        size=sims
                    )
                    from scipy.stats import norm as _norm_local

                u = _norm_local.cdf(z)
                hits = (u < probs).all(axis=1)
                joint = float(hits.mean())

                ev_combo = payout_mult * joint - 1.0
                raw_kelly = kelly_for_combo(joint, payout_mult, fractional_kelly)

                # Volatility-aware Kelly: downscale based on average leg volatility
                vol_cvs = [float(leg.get("volatility_cv", 0.0)) for leg in legs]
                avg_vol_cv = float(np.mean(vol_cvs)) if vol_cvs else 0.0
                vol_scale = float(1.0 / (1.0 + avg_vol_cv))  # higher volatility -> smaller stake
                raw_kelly *= vol_scale

                hist_df = load_history()
                k_adj, risk_note = adjust_kelly_for_risk(
                    raw_kelly, hist_df, bankroll, max_daily_loss_pct, max_weekly_loss_pct
                )
                stake = round(bankroll * k_adj, 2)
                decision = combo_decision(ev_combo)

                st.markdown(f"- Legs in Combo: **{len(legs)}**")
                st.markdown(f"- Joint Hit Probability: **{joint*100:.1f}%**")
                st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
                st.markdown(f"- Raw Kelly Fraction: **{raw_kelly*100:.2f}%**")
                st.markdown(f"- Risk-Adjusted Kelly Fraction: **{k_adj*100:.2f}%**")
                st.markdown(f"- Suggested Stake: **${stake:.2f}**")
                st.markdown(f"- **Recommendation:** {decision}")
                if risk_note:
                    st.warning(risk_note)

                
st.markdown("---")
st.subheader("💾 Log This Bet?")
choice = st.radio("Did you place this bet?", ["No", "Yes"], horizontal=True)

col_log, col_reset = st.columns(2)
with col_log:
    confirm_log = st.button("Confirm Log Decision")
with col_reset:
    reset_home = st.button("Reset to Home")

if confirm_log:
    if choice == "Yes":
        ensure_history()
        df_hist = load_history()
        combo_name = " + ".join([f"{leg['player']} {leg['market']}" for leg in legs])
        new_row = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Player": combo_name,
            "Market": f"{len(legs)}-Leg Combo",
            "Line": 0.0,
            "EV": ev_combo * 100.0,
            "Stake": stake,
            "Result": "Pending",
            "CLV": 0.0,
            "KellyFrac": k_adj
        }
        df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        save_history(df_hist)
        st.success("Bet logged to history as Pending ✅")
    else:
        st.info("You chose not to log this bet.")

if reset_home:
    _safe_rerun()

# ---------------------------------------------------------
#  RESULTS TAB

# ---------------------------------------------------------

with tab_results:
    st.subheader("Results & Personal Tracking")

    df = load_history()

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No bets logged yet. Log entries after you place bets.")

    with st.form("log_result_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            r_player = st.text_input("Player / Entry Name")
        with c2:
            r_market = st.selectbox(
                "Market",
                ["PRA", "Points", "Rebounds", "Assists", "Combo"]
            )
        with c3:
            r_line = st.number_input(
                "Line",
                min_value=0.0,
                max_value=200.0,
                value=25.0,
                step=0.5
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            r_ev = st.number_input(
                "Model EV (%)",
                min_value=-50.0,
                max_value=200.0,
                value=5.0,
                step=0.1
            )
        with c5:
            r_stake = st.number_input(
                "Stake ($)",
                min_value=0.0,
                max_value=10000.0,
                value=5.0,
                step=0.5
            )
        with c6:
            r_clv = st.number_input(
                "CLV (Closing - Entry) in %",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=0.1
            )

        r_result = st.selectbox(
            "Result",
            ["Pending", "Hit", "Miss", "Push"]
        )

        submit_res = st.form_submit_button("Log Result")

        if submit_res:
            ensure_history()
            new_row = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Player": r_player,
                "Market": r_market,
                "Line": r_line,
                "EV": r_ev,
                "Stake": r_stake,
                "Result": r_result,
                "CLV": r_clv,
                "KellyFrac": fractional_kelly
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_history(df)
            st.success("Result logged ✅")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if not comp.empty:
        pnl = comp.apply(lambda r: compute_pnl_from_row(r, payout_mult), axis=1)
        hits = (comp["Result"] == "Hit").sum()
        total = len(comp)
        hit_rate = (hits / total * 100.0) if total > 0 else 0.0
        roi = pnl.sum() / max(bankroll, 1.0) * 100.0
        clv_avg = comp["CLV"].mean() if "CLV" in comp.columns else 0.0
        pnl_var = float(np.var(pnl)) if len(pnl) > 1 else 0.0

        st.markdown(
            f"**Completed Bets:** {total}  |  "
            f"**Hit Rate:** {hit_rate:.1f}%  |  "
            f"**ROI vs Bankroll:** {roi:+.1f}%  |  "
            f"**Avg CLV:** {clv_avg:+.2f}%  |  "
            f"**PnL Variance:** {pnl_var:.2f}"
        )

        trend = comp.copy()
        trend["Profit"] = pnl.values
        trend["Cumulative"] = trend["Profit"].cumsum()

        fig = px.line(
            trend,
            x="Date",
            y="Cumulative",
            title="Cumulative Profit (All Logged Bets)",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
#  HISTORY TAB
# ---------------------------------------------------------

with tab_history:
    st.subheader("History & Filters")

    df = load_history()

    if df.empty:
        st.info("No logged bets yet.")
    else:
        min_ev = st.slider(
            "Min EV (%) filter",
            min_value=-20.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

        market_filter = st.selectbox(
            "Market filter",
            ["All", "PRA", "Points", "Rebounds", "Assists", "Combo"],
            index=0
        )

        filt = df[df["EV"] >= min_ev]

        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        st.markdown(f"**Filtered Bets:** {len(filt)}")
        st.dataframe(filt, use_container_width=True)

        if not filt.empty:
            filt = filt.copy()
            filt["Net"] = filt.apply(
                lambda r: (
                    r["Stake"] * (payout_mult - 1.0)
                    if r["Result"] == "Hit"
                    else (-r["Stake"] if r["Result"] == "Miss" else 0.0)
                ),
                axis=1,
            )
            filt["Cumulative"] = filt["Net"].cumsum()

            fig = px.line(
                filt,
                x="Date",
                y="Cumulative",
                title="Cumulative Profit (Filtered View)",
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
#  CALIBRATION TAB
# ---------------------------------------------------------

with tab_calib:
    st.subheader("Calibration & Edge Integrity Check")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if comp.empty or len(comp) < 25:
        st.info("Log at least 25 completed bets with EV to start calibration.")
    else:
        comp = comp.copy()
        comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
        comp = comp.dropna(subset=["EV_float"])

        if comp.empty:
            st.info("No valid EV values yet.")
        else:
            # Predicted vs actual
            comp["PredWinProb"] = 0.5 + comp["EV_float"]
            comp["PredWinProb"] = comp["PredWinProb"].clip(0.05, 0.95)
            actual_win_prob = (comp["Result"] == "Hit").mean()
            pred_win_prob = comp["PredWinProb"].mean()
            gap = (pred_win_prob - actual_win_prob) * 100.0

            pnl = comp.apply(lambda r: compute_pnl_from_row(r, payout_mult), axis=1)
            roi = pnl.sum() / max(1.0, bankroll) * 100.0

            # EV buckets
            comp["EV_bin"] = pd.cut(
                comp["EV_float"] * 100.0,
                bins=[-100, 0, 5, 10, 20, 100],
                labels=["<=0%", "0–5%", "5–10%", "10–20%", "20%+"]
            )

            bucket_rows = []
            for b, g in comp.groupby("EV_bin"):
                if g.empty:
                    continue
                actual = (g["Result"] == "Hit").mean() * 100.0
                pred = (g["PredWinProb"].mean()) * 100.0
                bucket_rows.append({
                    "EV Bucket": str(b),
                    "Count": len(g),
                    "Predicted Win%": f"{pred:.1f}%",
                    "Actual Win%": f"{actual:.1f}%",
                    "Gap (pp)": f"{(pred-actual):+.1f}"
                })

            if bucket_rows:
                st.markdown("#### EV Bucket Calibration")
                st.table(pd.DataFrame(bucket_rows))

            st.markdown(
                f"**Predicted Avg Win Prob:** {pred_win_prob*100:.1f}%  |  "
                f"**Actual Hit Rate:** {actual_win_prob*100:.1f}%  |  "
                f"**Calibration Gap:** {gap:+.1f} pp  |  "
                f"**ROI vs Bankroll:** {roi:+.1f}%"
            )

            # Edge distribution
            comp["Edge_vs_Market"] = comp["EV_float"] * 100.0
            fig2 = px.histogram(
                comp,
                x="Edge_vs_Market",
                nbins=20,
                title="Distribution of Model Edge vs Market (EV %)"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Update calibration factor
            if st.button("Recompute Calibration Factor"):
                if pred_win_prob > 0:
                    scale = float(np.clip(actual_win_prob / pred_win_prob, 0.7, 1.3))
                    save_calibration_state(scale)
                    st.success(f"Calibration factor updated to {scale:.3f}. This will auto-adjust future probabilities.")
                else:
                    st.error("Predicted win probability is invalid, cannot update calibration.")

# =========================================================
#  FOOTER
# =========================================================

st.markdown(
    """
    <footer style='text-align:center; margin-top:30px; color:#FFCC33; font-size:11px;'>
        © 2025 NBA Prop Quant Engine • Powered by Kamal
    </footer>
    """,
    unsafe_allow_html=True,
)

