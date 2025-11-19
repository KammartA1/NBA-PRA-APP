# =========================================================
#  NBA PROP BETTING QUANT ENGINE — SINGLE FILE STREAMLIT APP
#  Upgraded with:
#   - Empirical Bootstrap Monte Carlo (10,000 sims)
#   - Defensive Matchup Engine (team-context aware)
#   - Pace-adjusted minutes & usage context
#   - Auto live book line pulling via public JSON mirror
#   - Auto opponent detection + blowout risk inference
#   - Expanded History + Calibration + Risk Controls
#   - EVERYTHING DEFENSE-ADJUSTED
# =========================================================

import os
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
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats

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
MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists"]

MARKET_METRICS = {
    "PRA": ["PTS", "REB", "AST"],
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
}

# Map our UI markets to live book stat types
ODDS_API_MARKET_MAP = {
    "PRA": ["Pts+Rebs+Asts", "PRA"],
    "Points": ["Points"],
    "Rebounds": ["Rebounds"],
    "Assists": ["Assists"],
}

MAX_KELLY_PCT = 0.03  # 3% hard cap

# Public live book mirror endpoint
ODDS_API_MIRROR_URL = "https://pp-public-mirror.vercel.app/api/board"

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

st.sidebar.caption("Model auto-pulls NBA stats & lines. Lines auto-fill from live book when possible.")

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

def get_context_multiplier(opp_abbrev: str | None, market: str) -> float:
    """
    Adjust projection using advanced opponent factors.
    Everything is defense-adjusted through this multiplier.
    """
    if not opp_abbrev or opp_abbrev not in TEAM_CTX or not LEAGUE_CTX:
        return 1.0

    opp = TEAM_CTX[opp_abbrev]

    pace_f = opp["PACE"] / LEAGUE_CTX["PACE"]
    def_f = LEAGUE_CTX["DEF_RATING"] / opp["DEF_RATING"]

    reb_adj = (
        LEAGUE_CTX["REB_PCT"] / opp["DREB_PCT"]
        if market == "Rebounds" else 1.0
    )
    ast_adj = (
        LEAGUE_CTX["AST_PCT"] / opp["AST_PCT"]
        if market == "Assists" else 1.0
    )

    # Simple blended multiplier
    if market == "Rebounds":
        mult = 0.4 * pace_f + 0.3 * def_f + 0.3 * reb_adj
    elif market == "Assists":
        mult = 0.4 * pace_f + 0.3 * def_f + 0.3 * ast_adj
    else:
        mult = 0.6 * pace_f + 0.4 * def_f

    return float(np.clip(mult, 0.80, 1.25))

# =========================================================
#  PART 4 — ODDS_API LIVE API (BEST-EFFORT) & MATCHUP HELPERS
# =========================================================
#  PART 4 — ODDS API PLAYER PROPS (DRAFTKINGS ETC.) & MATCHUP HELPERS
# =========================================================

# --- ODDS API KEY MANAGEMENT ---
# Priority:
# 1. Environment variable (local dev)
# 2. Streamlit Cloud Secrets (deployment)
# 3. Fallback: empty string (no crash)
ODDS_API_KEY = (
    os.environ.get("ODDS_API_KEY") or 
    st.secrets.get("ODDS_API_KEY", "")
).strip()

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT_KEY = "basketball_nba"

# Map our UI market names → The Odds API player prop market keys
ODDS_MARKET_MAP = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
    "PRA": "player_points_rebounds_assists",
}

ODDS_CACHE_DIR = os.path.join(TEMP_DIR, "odds_cache")
os.makedirs(ODDS_CACHE_DIR, exist_ok=True)


def _odds_cache_path(market_key: str) -> str:
    return os.path.join(ODDS_CACHE_DIR, f"{market_key}.json")


def _load_odds_cache(market_key: str):
    path = _odds_cache_path(market_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_odds_cache(market_key: str, data: dict):
    path = _odds_cache_path(market_key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _odds_request_log_path() -> str:
    return os.path.join(ODDS_CACHE_DIR, "request_log.json")


def _can_call_odds_api(max_calls_per_day: int = 12) -> bool:
    """Simple per-day call guard so we do not burn through the free tier."""
    path = _odds_request_log_path()
    today = datetime.now().strftime("%Y-%m-%d")
    data = {"date": today, "count": 0}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"date": today, "count": 0}
    if data.get("date") != today:
        data = {"date": today, "count": 0}
    if data["count"] >= max_calls_per_day:
        return False
    data["count"] += 1
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
    return True


def _call_odds_api_for_market(market_key: str):
    """Low-level caller. Returns list[dict] games or None."""
    if not ODDS_API_KEY:
        return None
    if not _can_call_odds_api():
        # Respect daily budget; rely on cache instead
        return None

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": market_key,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def fetch_oddsapi_board(market_key: str):
    """
    Fetch player props for a given market from The Odds API with caching.

    - Uses a JSON cache in /tmp so repeated page loads do not call the API.
    - Guards total calls per day via `_can_call_odds_api`.
    - Returns a list of game dicts (The Odds API native format).
    """
    # Try cache first
    cached = _load_odds_cache(market_key)
    if cached:
        return cached

    data = _call_odds_api_for_market(market_key)
    if data is None:
        return cached  # may be None as well
    _save_odds_cache(market_key, data)
    return data


def normalize_player_name_for_odds(name: str) -> str:
    return (
        str(name).lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )


def _build_odds_player_index(games: list, market_key: str) -> dict:
    """
    Build mapping: normalized player name -> average line for the specified market.
    We aggregate across all books that post that prop.
    """
    index: dict[str, float] = {}
    counts: dict[str, int] = {}

    if not games:
        return index

    for g in games:
        try:
            bookmakers = g.get("bookmakers", []) or []
            for bk in bookmakers:
                mkts = bk.get("markets", []) or []
                for m in mkts:
                    if m.get("key") != market_key:
                        continue
                    for o in m.get("outcomes", []) or []:
                        name = normalize_player_name_for_odds(o.get("name", ""))
                        point = o.get("point")
                        if point is None:
                            continue
                        point = float(point)
                        index[name] = index.get(name, 0.0) + point
                        counts[name] = counts.get(name, 0) + 1
        except Exception:
            continue

    # Convert sums to averages
    for nm, total in list(index.items()):
        c = counts.get(nm, 1)
        index[nm] = total / max(c, 1)
    return index


def get_oddsapi_line(player: str, market: str) -> float | None:
    """
    Main entry: given a player name + UI market, return the consensus line.

    - Maps UI market → Odds API market key.
    - Fetches / caches the board for that market.
    - Builds an index of player -> average line across books.
    """
    market_key = ODDS_MARKET_MAP.get(market)
    if not market_key:
        return None

    games = fetch_oddsapi_board(market_key)
    if not games:
        return None

    index = _build_odds_player_index(games, market_key)
    target = normalize_player_name_for_odds(player)
    return index.get(target)
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

# =========================================================
#  PART 8 — CORRELATION ENGINE
# =========================================================

def estimate_player_correlation(leg1: dict, leg2: dict) -> float:
    """
    Contextual correlation estimate between two legs.
    """
    corr = 0.0

    # 1. Same-team baseline
    if leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"]:
        corr += 0.18

    # 2. Market-type interactions
    m1, m2 = leg1["market"], leg2["market"]

    if m1 == "Points" and m2 == "Points":
        corr += 0.08

    if (m1 == "Points" and m2 == "Assists") or (m1 == "Assists" and m2 == "Points"):
        corr -= 0.10

    if (m1 == "Rebounds" and m2 == "Points") or (m1 == "Points" and m2 == "Rebounds"):
        corr -= 0.06

    if m1 == "PRA" or m2 == "PRA":
        corr += 0.03

    # 3. Context interaction
    ctx1, ctx2 = leg1.get("ctx_mult", 1.0), leg2.get("ctx_mult", 1.0)

    if ctx1 > 1.03 and ctx2 > 1.03:
        corr += 0.04
    if ctx1 < 0.97 and ctx2 < 0.97:
        corr += 0.03
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 1.03):
        corr -= 0.05

    # 4. Blowout risk — if same game & high risk, correlation increases
    if leg1.get("blowout") and leg2.get("blowout"):
        corr += 0.03

    corr = float(np.clip(corr, -0.25, 0.40))
    return corr

# =========================================================
#  PART 9 — MONTE CARLO (EMPIRICAL BOOTSTRAP)
# =========================================================

def apply_calibration_to_prob(p: float) -> float:
    """
    Applies self-learning calibration scaling around 50%.
    """
    scale = float(CAL_STATE.get("prob_scale", 1.0))
    # move p towards / away from 0.5 depending on scale
    centered = p - 0.5
    adj = 0.5 + centered * scale
    return float(np.clip(adj, 0.02, 0.98))

def compute_leg_projection(player: str, market: str, line: float | None,
                           user_opp: str | None, n_games: int,
                           board: dict):
    """
    Core projection engine for a single leg.
    Uses:
      - empirical bootstrap over last N games
      - opponent context
      - auto blowout & usage adjustments
    Returns (leg_dict, error_message).
    """
    samples, minutes, team, last_opp, msg = get_player_game_samples(player, n_games, market)
    if samples is None:
        return None, msg

    # If line is None or <= 0, caller should handle
    if line is None or line <= 0:
        return None, "No valid line for this player/market."

    # Opponent detection — preference: user input → live book → last game
    opp = None
    if user_opp:
        opp = user_opp.strip().upper()
    else:
        t_board, opp_board = auto_detect_matchup_from_board(player, board)
        if opp_board:
            opp = opp_board
        elif last_opp:
            opp = str(last_opp).upper()

    ctx_mult = get_context_multiplier(opp, market)

    # Usage / role expansion heuristic: compare last 3 games vs overall
    try:
        avg_min = float(minutes.mean())
        recent_min = float(np.mean(minutes[: min(3, len(minutes))]))
        usage_boost = 1.0
        if recent_min >= avg_min + 4:
            usage_boost = 1.06
        elif recent_min <= max(10.0, avg_min - 5):
            usage_boost = 0.94
    except Exception:
        avg_min = float(np.mean(minutes))
        recent_min = avg_min
        usage_boost = 1.0

    # Blowout risk from board
    blowout = estimate_blowout_risk(team, opp, board)

    # Build adjusted samples
    samples_adj = samples.astype(float) * ctx_mult * usage_boost
    if blowout:
        # Slight trim due to reduced minutes
        samples_adj *= 0.95

    mu = float(np.mean(samples_adj))
    sd = float(max(np.std(samples_adj, ddof=1), 0.75))

    # Empirical bootstrap Monte Carlo
    n_sims = 10000
    draws = np.random.choice(samples_adj, size=n_sims, replace=True)
    over_flags = draws > line
    p_over_raw = float(np.mean(over_flags))

    # Apply calibration layer
    p_over = apply_calibration_to_prob(p_over_raw)

    # EV vs even odds
    ev_leg_even = p_over - (1.0 - p_over)

    opp_display = opp if opp else "Unknown"

    # Simple positional / matchup text proxy
    matchup_text = "Neutral matchup."
    if opp and opp in TEAM_CTX and LEAGUE_CTX:
        opp_ctx = TEAM_CTX[opp]
        pace_rel = opp_ctx["PACE"] / LEAGUE_CTX["PACE"]
        def_rel = opp_ctx["DEF_RATING"] / LEAGUE_CTX["DEF_RATING"]
        if def_rel > 1.05 and pace_rel < 0.97:
            matchup_text = "Tough defensive matchup (slow, strong defense)."
        elif def_rel < 0.95 and pace_rel > 1.03:
            matchup_text = "Very favorable matchup (fast pace, weak defense)."
        elif def_rel < 0.97:
            matchup_text = "Soft defense vs this type of production."
        elif def_rel > 1.03:
            matchup_text = "Above-average defense; expectation tempered."

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
        "blowout": bool(blowout),
        "usage_boost": float(usage_boost),
        "matchup_text": matchup_text,
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
    blowout = leg.get("blowout", False)
    usage_boost = leg.get("usage_boost", 1.0)
    matchup_text = leg.get("matchup_text", "Neutral matchup.")

    headshot = get_headshot_url(player)

    with container:
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top:0;color:#FFCC33;">{player} — {market}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        cols = st.columns([1, 2]) if not compact else st.columns([1, 2])
        with cols[0]:
            if headshot:
                st.image(headshot, width=120)
        with cols[1]:
            st.write(f"🆚 **Opponent:** {opp}")
            st.write(f"📌 **Line:** {line}")
            st.write(f"📊 **Model Mean (def-adj):** {mu:.2f}")
            st.write(f"📉 **Model SD (volatility):** {sd:.2f}")
            st.write(f"⏱️ **Context Multiplier:** {ctx:.3f}")
            st.write(f"🎯 **Raw Bootstrapped Prob Over:** {p_raw*100:.1f}%")
            st.write(f"🎯 **Calibrated Prob Over:** {p*100:.1f}%")
            st.write(f"💵 **Even-Money EV:** {even_ev*100:+.1f}%")

            st.caption(f"📝 {msg}")
            st.caption(f"📎 Matchup: {matchup_text}")

            if usage_boost > 1.02:
                st.info("Usage / minutes have trended UP recently (auto-boost applied).")
            elif usage_boost < 0.98:
                st.warning("Usage / minutes have trended DOWN recently (auto-trim applied).")

            if blowout:
                st.warning("Blowout risk detected for this matchup (auto-trim applied).")

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
    st.subheader("2-Pick Projection & Edge (Bootstrap Monte Carlo)")

    board_pts = fetch_oddsapi_board("player_points")
    board_reb = fetch_oddsapi_board("player_rebounds")
    board_ast = fetch_oddsapi_board("player_assists")
    board_pra = fetch_oddsapi_board("player_points_rebounds_assists")


    c1, c2 = st.columns(2)

    # LEFT LEG
    with c1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="p1_market")
        manual1 = st.checkbox("P1: Manual line override", value=False)
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5, key="p1_line")
        o1 = st.text_input("P1 Opponent (Team Abbrev, optional)", help="Leave blank to auto-detect")

    # RIGHT LEG
    with c2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="p2_market")
        manual2 = st.checkbox("P2: Manual line override", value=False)
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5, key="p2_line")
        o2 = st.text_input("P2 Opponent (Team Abbrev, optional)", help="Leave blank to auto-detect")

    leg1 = None
    leg2 = None

    run = st.button("Run Model ⚡")

    if run:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

        # Auto fetch lines if manual override is off
        line1_used = None
        line2_used = None

        if p1 and not manual1:
            auto_line1 = get_oddsapi_line(p1, m1, board)
            if auto_line1 is None:
                st.warning("Could not auto-fetch live book line for Player 1. Please enable manual override and enter line.")
            else:
                line1_used = auto_line1
                st.info(f"P1 auto live book line detected: {auto_line1:.1f}")
        elif p1 and manual1:
            line1_used = l1

        if p2 and not manual2:
            auto_line2 = get_oddsapi_line(p2, m2, board)
            if auto_line2 is None:
                st.warning("Could not auto-fetch live book line for Player 2. Please enable manual override and enter line.")
            else:
                line2_used = auto_line2
                st.info(f"P2 auto live book line detected: {auto_line2:.1f}")
        elif p2 and manual2:
            line2_used = l2

        # Compute legs
        if p1 and line1_used and line1_used > 0:
            leg1, err1 = compute_leg_projection(
                p1, m1, line1_used, o1, games_lookback, board
            )
            if err1:
                st.error(f"P1: {err1}")
        elif p1:
            st.error("P1 does not have a valid line.")

        if p2 and line2_used and line2_used > 0:
            leg2, err2 = compute_leg_projection(
                p2, m2, line2_used, o2, games_lookback, board
            )
            if err2:
                st.error(f"P2: {err2}")
        elif p2:
            st.error("P2 does not have a valid line.")

        # Render legs
        colL, colR = st.columns(2)
        if leg1:
            render_leg_card(leg1, colL, compact_mode)
            update_market_library(leg1["player"], leg1["market"], leg1["line"])
        if leg2:
            render_leg_card(leg2, colR, compact_mode)
            update_market_library(leg2["player"], leg2["market"], leg2["line"])

        # Market implied probability from payout
        st.markdown("---")
        st.subheader("📈 Market vs Model Probability Check")

        def implied_probability(mult):
            return 1.0 / mult

        imp_prob = implied_probability(payout_mult)
        st.markdown(f"**Market Implied Combo Probability (approx):** {imp_prob*100:.1f}%")

        if leg1:
            st.markdown(
                f"**{leg1['player']} Model Prob:** {leg1['prob_over']*100:.1f}% "
                f"→ Edge vs 50/50: {(leg1['prob_over'] - 0.5)*100:+.1f}%"
            )
        if leg2:
            st.markdown(
                f"**{leg2['player']} Model Prob:** {leg2['prob_over']*100:.1f}% "
                f"→ Edge vs 50/50: {(leg2['prob_over'] - 0.5)*100:+.1f}%"
            )

        # 2-PICK COMBO
        if leg1 and leg2:
            corr_est = estimate_player_correlation(leg1, leg2)

            p1 = leg1["prob_over"]
            p2 = leg2["prob_over"]

            # Correlated Bernoulli via Gaussian copula
            rho = float(np.clip(corr_est, -0.99, 0.99))
            n_joint = 10000
            z1 = np.random.normal(size=n_joint)
            z2 = rho * z1 + np.sqrt(max(1.0 - rho**2, 1e-6)) * np.random.normal(size=n_joint)
            u1 = norm.cdf(z1)
            u2 = norm.cdf(z2)
            leg1_over_sim = u1 < p1
            leg2_over_sim = u2 < p2
            joint_sim = np.mean(leg1_over_sim & leg2_over_sim)

            joint = float(np.clip(joint_sim, 0.0, 1.0))

            ev_combo = payout_mult * joint - 1.0
            raw_kelly = kelly_for_combo(joint, payout_mult, fractional_kelly)

            hist_df = load_history()
            k_adj, risk_note = adjust_kelly_for_risk(
                raw_kelly, hist_df, bankroll, max_daily_loss_pct, max_weekly_loss_pct
            )
            stake = round(bankroll * k_adj, 2)
            decision = combo_decision(ev_combo)

            st.markdown("### 🎯 **2-Pick Combo Result (Joint Monte Carlo)**")
            st.markdown(f"- Estimated Correlation: **{corr_est:+.2f}**")
            st.markdown(f"- Joint Hit Probability: **{joint*100:.1f}%**")
            st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Raw Kelly Fraction: **{raw_kelly*100:.2f}%**")
            st.markdown(f"- Risk-Adjusted Kelly Fraction: **{k_adj*100:.2f}%**")
            st.markdown(f"- Suggested Stake: **${stake:.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")
            if risk_note:
                st.warning(risk_note)

            # Baseline library recap
            for leg in [leg1, leg2]:
                if leg:
                    mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
                    if mean_b:
                        st.caption(
                            f"📊 Market Baseline for {leg['player']} {leg['market']}: "
                            f"mean={mean_b:.1f}, median={med_b:.1f}"
                        )

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
