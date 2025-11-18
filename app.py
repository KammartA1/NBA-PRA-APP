# =========================================================
#  NBA PROP BETTING QUANT ENGINE — FINAL UPGRADED SINGLE-FILE APP
#  Streamlit App (app.py)
# =========================================================

import os
import time
import random
import difflib
import json
from datetime import datetime, date

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
    page_title="NBA Prop Quant Engine",
    page_icon="🏀",
    layout="wide"
)

TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
CARD_BG = "#17131C"
BG = "#0D0A12"

UND_DOG_ENDPOINT = "https://api.underdogfantasy.com/beta/v5/over_under_lines"

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

st.markdown('<p class="main-header">🏀 NBA Prop Quant Engine</p>', unsafe_allow_html=True)

# =========================================================
#  SIDEBAR (USER & BANKROLL SETTINGS)
# =========================================================

st.sidebar.header("User, Bankroll & Model Controls")

user_id = st.sidebar.text_input("Your ID (for personal history)", value="Me").strip() or "Me"
LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0)
payout_mult_2pick = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly (for 2-pick)", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Recent Games Sample (N)", 5, 20, 10)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)

daily_loss_cap_pct = st.sidebar.slider("Max Daily Loss (% of Bankroll)", 1, 50, 15)
data_source_mode = st.sidebar.selectbox("Line Source", ["Manual Input", "Underdog Autofill"])

st.sidebar.caption("Model auto-pulls NBA stats & Underdog lines. You only tweak context and risk.")

# =========================================================
#  MODEL CONSTANTS
# =========================================================

MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists"]

MARKET_METRICS = {
    "PRA": ["PTS", "REB", "AST"],
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
}

HEAVY_TAIL = {
    "PRA": 1.35,
    "Points": 1.25,
    "Rebounds": 1.25,
    "Assists": 1.25,
}

MAX_KELLY_PCT = 0.03  # 3% hard cap
N_SIMS = 10_000       # Monte Carlo samples

# =========================================================
#  HELPER — CURRENT SEASON (AUTO-ROLLING)
# =========================================================

def current_season():
    """
    Returns NBA season string like '2025-26' based on today's date.
    Rolls automatically every October.
    """
    today = datetime.now()
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"

# =========================================================
#  PLAYER RESOLUTION HELPERS
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
    """Resolves fuzzy player input → correct NBA API player ID & full_name."""
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

# =========================================================
#  TEAM CONTEXT (PACE, DEF, REB%, AST%)
# =========================================================

@st.cache_data(show_spinner=False, ttl=3600)
def get_team_context():
    """Pulls advanced opponent metrics for matchup adjustments."""
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
            "TEAM_ID","TEAM_ABBREVIATION","REB_PCT","OREB_PCT","DREB_PCT","AST_PCT","PACE"
        ]]

        defn = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"
        ]]

        df = base.merge(adv, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
        df = df.merge(defn, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")

        league_avg = {
            col: df[col].mean()
            for col in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]
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

def get_context_multiplier(opp_abbrev: str | None, market: str):
    """Adjust projection using advanced opponent factors. Also drives volatility."""
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

    mult = (0.4 * pace_f) + (0.3 * def_f) + (0.3 * (reb_adj if market == "Rebounds" else ast_adj))

    return float(np.clip(mult, 0.80, 1.20))

# =========================================================
#  MARKET BASELINE LIBRARY (LINES MEMORY)
# =========================================================

MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

def load_market_library():
    if not os.path.exists(MARKET_LIBRARY_FILE):
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])
    try:
        return pd.read_csv(MARKET_LIBRARY_FILE)
    except Exception:
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])

def save_market_library(df):
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
#  PLAYER GAME LOGS & BOOTSTRAP SAMPLES
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_game_log(name: str):
    """
    Returns (game_log_df, label, error_message).
    game_log_df is filtered to current season.
    """
    pid, label = resolve_player(name)
    if not pid:
        return None, None, f"No match for '{name}'."

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception as e:
        return None, label, f"Game log error: {e}"

    if gl.empty:
        return None, label, "No recent game logs found."

    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False)
    return gl, label, ""

def extract_market_values(gl: pd.DataFrame, market: str, n_games: int):
    """
    From a game log, build recent N-game sample for the desired market.
    Returns (values_array, minutes_array, team_abbrev).
    """
    gl_n = gl.head(n_games).copy()
    cols = MARKET_METRICS[market]

    values = []
    minutes = []

    for _, r in gl_n.iterrows():
        # Minutes
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

        total_val = sum(float(r.get(c, 0)) for c in cols)
        values.append(total_val)
        minutes.append(m)

    if not values:
        return None, None, None

    values = np.array(values, dtype=float)
    minutes = np.array(minutes, dtype=float)

    try:
        team_abbrev = gl_n["TEAM_ABBREVIATION"].mode().iloc[0]
    except Exception:
        team_abbrev = None

    return values, minutes, team_abbrev

def compute_usage_based_rates(values: np.ndarray, minutes: np.ndarray):
    """
    Computes per-minute mean and SD and avg minutes.
    """
    per_min = values / np.maximum(minutes, 1e-3)
    mu_per_min = float(np.mean(per_min))
    sd_per_min = max(
        float(np.std(per_min, ddof=1)),
        0.15 * max(mu_per_min, 0.5)
    )
    avg_min = float(np.mean(minutes))
    return mu_per_min, sd_per_min, avg_min

# =========================================================
#  GAME SCRIPT SIMULATION ENGINE
# =========================================================

def simulate_game_script(avg_min, sd_min, ctx_mult, teammate_out, blowout, opp_abbrev):
    """
    Simulates game script meta-variables that affect projections:
      - expected minute range
      - usage distribution scaling
      - game-level variance multiplier
      - quarter-level distribution
    """
    # Pace-based minute tweak
    pace_mult = 1.0
    if opp_abbrev and opp_abbrev in TEAM_CTX and LEAGUE_CTX:
        opp_pace = TEAM_CTX[opp_abbrev]["PACE"]
        league_pace = LEAGUE_CTX["PACE"]
        pace_mult = float(np.clip(opp_pace / league_pace, 0.9, 1.1))

    # Blowout risk → shave minutes
    blowout_mult = 0.92 if blowout else 1.0

    # Teammate out → slight minute & usage bump
    tm_mult = 1.06 if teammate_out else 1.0

    # Expected minutes range
    exp_min = avg_min * pace_mult * tm_mult * blowout_mult
    min_low = max(16.0, exp_min - 5.0)
    min_high = exp_min + 5.0

    # Usage scaling (simple)
    usage_center = 1.0 * tm_mult
    usage_spread = 0.10  # ±10% usage
    # Quarter-level distribution (heavier mid-game)
    q_dist = np.array([0.22, 0.26, 0.26, 0.26])

    # Game-level variance scaling
    base_var_scale = 1.0
    if blowout:
        base_var_scale *= 1.10
    if teammate_out:
        base_var_scale *= 1.08
    base_var_scale *= float(np.clip(ctx_mult, 0.9, 1.1))

    return {
        "exp_min": float(exp_min),
        "min_low": float(min_low),
        "min_high": float(min_high),
        "usage_center": float(usage_center),
        "usage_spread": float(usage_spread),
        "q_dist": q_dist,
        "var_scale": float(base_var_scale)
    }

# =========================================================
#  EMPIRICAL BOOTSTRAP MONTE CARLO ENGINE
# =========================================================

def run_bootstrap_mc(values, minutes, line, script, market):
    """
    Empirical bootstrap Monte Carlo with N_SIMS draws.
    Uses weighted sampling (recent games heavier), plus
    game script scaling for minutes/usage/variance.
    Returns (p_over, boot_mean, boot_dist).
    """
    if values is None or len(values) == 0:
        return 0.5, None, None

    v = np.array(values, dtype=float)
    m = np.array(minutes, dtype=float)
    n = len(v)

    # Weighted bootstrap: more recent games get higher weight
    idx = np.arange(n)
    raw_w = np.linspace(1.0, 2.0, n)  # last game highest weight
    w = raw_w / raw_w.sum()

    # Script multipliers
    usage_center = script["usage_center"]
    usage_spread = script["usage_spread"]
    var_scale = script["var_scale"]

    sims = []
    for _ in range(N_SIMS):
        k = np.random.choice(idx, p=w)
        base_val = v[k]
        base_min = m[k]

        # Usage randomization around script usage
        usage_shock = np.random.normal(loc=usage_center, scale=usage_spread / 2.0)
        usage_shock = float(np.clip(usage_shock, usage_center - usage_spread, usage_center + usage_spread))

        # Minute scaling: map to expected minutes vs sample minutes
        if base_min > 0:
            minute_scale = script["exp_min"] / base_min
        else:
            minute_scale = 1.0

        # Heavy-tail volatility
        heavy = HEAVY_TAIL.get(market, 1.15)
        noise_scale = (heavy - 1.0) * 0.10

        val = base_val * minute_scale * usage_shock
        val *= np.random.normal(loc=1.0, scale=noise_scale * var_scale)
        sims.append(val)

    sims = np.array(sims, dtype=float)
    p_over = float(np.mean(sims > line))
    boot_mean = float(np.mean(sims))
    return p_over, boot_mean, sims

# =========================================================
#  ENSEMBLE PROJECTION ENGINE
# =========================================================

def build_ensemble_projection(best_mu, hist_values, line, boot_mean, script_mean):
    """
    Combines:
      - best engine mean (usage x minutes x context)
      - historical mean
      - Underdog market mean (line)
      - usage-predicted mean (best_mu approximation)
      - game-script mean
      - bootstrap mean
    into an ensemble projection.
    """
    hist_mean = float(np.mean(hist_values)) if hist_values is not None and len(hist_values) > 0 else best_mu
    market_mean = float(line)  # simple approximation of market's implied mean
    usage_mean = best_mu       # current best engine is usage-driven

    pieces = {
        "best": best_mu,
        "bootstrap": boot_mean if boot_mean is not None else best_mu,
        "hist": hist_mean,
        "market": market_mean,
        "usage": usage_mean,
        "script": script_mean
    }

    weights = {
        "best": 0.35,
        "bootstrap": 0.25,
        "hist": 0.15,
        "market": 0.10,
        "usage": 0.05,
        "script": 0.10,
    }

    num = 0.0
    den = 0.0
    for k, v in pieces.items():
        w = weights[k]
        num += w * v
        den += w
    if den <= 0:
        return best_mu
    return float(num / den)

# =========================================================
#  CORRELATION & COVARIANCE-BASED JOINT MC
# =========================================================

def estimate_player_correlation(leg1, leg2):
    """
    Context-aware correlation estimate using:
      - same team
      - minutes expectation
      - market pair type (PTS/REB/AST/PRA)
      - defensive context
      - pace context
      - on/off style factors
    """
    corr = 0.0

    # Same team bump
    if leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"]:
        corr += 0.18

    # Minutes dependency
    m1 = leg1.get("exp_min", 30.0)
    m2 = leg2.get("exp_min", 30.0)
    if m1 > 32 and m2 > 32:
        corr += 0.05
    elif m1 < 22 or m2 < 22:
        corr -= 0.04

    # Market interaction
    mkt1, mkt2 = leg1["market"], leg2["market"]
    if mkt1 == "Points" and mkt2 == "Points":
        corr += 0.08
    if (mkt1 == "Points" and mkt2 == "Assists") or (mkt1 == "Assists" and mkt2 == "Points"):
        corr -= 0.10
    if (mkt1 == "Rebounds" and mkt2 == "Points") or (mkt1 == "Points" and mkt2 == "Rebounds"):
        corr -= 0.06
    if mkt1 == "PRA" or mkt2 == "PRA":
        corr += 0.03

    # Defensive & pace covariance
    ctx1, ctx2 = leg1.get("ctx_mult", 1.0), leg2.get("ctx_mult", 1.0)
    if ctx1 > 1.03 and ctx2 > 1.03:
        corr += 0.04
    if ctx1 < 0.97 and ctx2 < 0.97:
        corr += 0.03
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 1.03):
        corr -= 0.05

    # Clamp
    return float(np.clip(corr, -0.30, 0.45))

def joint_gaussian_mc(p1, p2, corr, n_sims=N_SIMS):
    """
    Covariance-based joint Monte Carlo for two binary events (over hits).
    Uses Gaussian copula.
    """
    if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
        return p1 * p2

    mean = [0.0, 0.0]
    cov = [[1.0, corr], [corr, 1.0]]
    z = np.random.multivariate_normal(mean, cov, size=n_sims)
    u1 = norm.cdf(z[:, 0])
    u2 = norm.cdf(z[:, 1])
    over1 = u1 < p1
    over2 = u2 < p2
    joint = float(np.mean(over1 & over2))
    return joint

# =========================================================
#  KELLY FORMULA FOR 2-PICK
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float):
    """
    Kelly criterion for 2-pick entries.
    """
    b = payout_mult - 1.0
    q = 1 - p_joint
    raw = (b * p_joint - q) / b
    k = raw * frac
    return float(np.clip(k, 0, MAX_KELLY_PCT))

# =========================================================
#  SELF-LEARNING CALIBRATION HOOK (SKELETON)
# =========================================================

def compute_model_drift(history_df):
    """
    Simple drift indicator using EV vs realized outcomes.
    Returns (ev_bias, clv_bias) multipliers.
    """
    if history_df is None or history_df.empty:
        return 1.0, 1.0

    comp = history_df[history_df["Result"].isin(["Hit","Miss"])].copy()
    if comp.empty:
        return 1.0, 1.0

    comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
    comp = comp.dropna(subset=["EV_float"])
    if comp.empty:
        return 1.0, 1.0

    pred = 0.5 + comp["EV_float"].mean()
    actual = (comp["Result"] == "Hit").mean()
    if actual <= 0 or pred <= 0:
        return 1.0, 1.0

    ev_bias = float(np.clip(actual / pred, 0.8, 1.2))

    # CLV bias ~ average CLV relative to absolute value
    clv = pd.to_numeric(comp["CLV"], errors="coerce").fillna(0.0)
    if not clv.empty:
        mean_clv = float(clv.mean())
        clv_bias = float(np.clip(1.0 + mean_clv / 20.0, 0.8, 1.2))
    else:
        clv_bias = 1.0

    return ev_bias, clv_bias

# =========================================================
#  UNDERDOG LINES SCRAPER (ROBUST / FAIL-SAFE)
# =========================================================

@st.cache_data(show_spinner=False, ttl=60)
def fetch_underdog_nba_lines():
    """
    Attempts to pull all NBA prop lines (PTS/REB/AST/PRA) from Underdog.
    This is written to be defensive: if the JSON shape changes or fails,
    it returns an empty DataFrame instead of crashing the app.
    """
    try:
        resp = requests.get(UND_DOG_ENDPOINT, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()

        # The exact Underdog JSON structure can change.
        # We attempt a flexible extraction pattern and guard with try/except.
        lines = []

        # Common structure pattern used by community scrapers:
        over_under_lines = data.get("over_under_lines") or data.get("over_under_lines", [])
        over_unders = {ou["id"]: ou for ou in data.get("over_unders", [])} if data.get("over_unders") else {}
        players = {p["id"]: p for p in data.get("players", [])} if data.get("players") else {}
        contests = {c["id"]: c for c in data.get("contests", [])} if data.get("contests") else {}

        for oul in over_under_lines:
            try:
                ou_id = oul.get("over_under_id")
                ou_meta = over_unders.get(ou_id, {})
                stat_type = ou_meta.get("stat_type")
                if stat_type not in ["points", "rebounds", "assists", "pra"]:
                    continue

                player_id = ou_meta.get("player_id")
                player_meta = players.get(player_id, {})
                player_name = player_meta.get("name") or player_meta.get("full_name") or "Unknown"

                contest_id = ou_meta.get("contest_id")
                contest = contests.get(contest_id, {})
                league = (contest.get("sport") or "").upper()
                if league != "NBA":
                    continue

                line_val = float(oul.get("stat_value") or ou_meta.get("line") or 0.0)
                if line_val <= 0:
                    continue

                team_abbrev = (player_meta.get("team") or "").upper()
                opp_abbrev = (contest.get("opponent") or "").upper()

                if stat_type == "points":
                    market = "Points"
                elif stat_type == "rebounds":
                    market = "Rebounds"
                elif stat_type == "assists":
                    market = "Assists"
                else:
                    market = "PRA"

                lines.append({
                    "player": player_name,
                    "market": market,
                    "line": line_val,
                    "team": team_abbrev,
                    "opp": opp_abbrev
                })
            except Exception:
                continue

        if not lines:
            return pd.DataFrame()

        df = pd.DataFrame(lines)
        return df

    except Exception:
        return pd.DataFrame()

# =========================================================
#  CORE PROJECTION ENGINE FOR A SINGLE LEG
# =========================================================

def compute_leg_projection(player, market, line, opp, teammate_out, blowout, n_games, history_df=None):
    """
    Full projection engine for a single leg.
    Uses:
      - rolling game logs (current season)
      - usage / per-minute rates
      - defensive matchup context
      - game script simulation
      - empirical bootstrap Monte Carlo
      - ensemble averaging

    Returns (leg_dict, error_message).
    """
    gl, label, err = get_player_game_log(player)
    if gl is None:
        return None, err

    values, minutes, team_abbrev = extract_market_values(gl, market, n_games)
    if values is None or len(values) == 0:
        return None, "Insufficient recent data."

    mu_per_min, sd_per_min, avg_min = compute_usage_based_rates(values, minutes)

    # Defensive and pace context
    opp_abbrev = opp.strip().upper() if opp else None
    ctx_mult = get_context_multiplier(opp_abbrev, market)

    # Game script
    script = simulate_game_script(avg_min, sd_per_min, ctx_mult, teammate_out, blowout, opp_abbrev)

    # Best-engine mean (usage x minutes x context)
    best_mu = mu_per_min * script["exp_min"] * ctx_mult

    # Script-specific mean (slightly adjusted)
    script_mean = best_mu * (1.0 + 0.03 * (script["var_scale"] - 1.0))

    # Bootstrap Monte Carlo (empirical)
    p_over_boot, boot_mean, sims = run_bootstrap_mc(values, minutes, line, script, market)

    # Ensemble projection
    ensemble_mean = build_ensemble_projection(best_mu, values, line, boot_mean, script_mean)

    # Convert ensemble mean and bootstrap spread to probability vs line
    # Guard: if sims exists, use its distribution directly
    if sims is not None:
        p_over = float(np.mean(sims > line))
    else:
        # fallback to normal approximation
        sd_est = float(np.std(values)) if len(values) > 1 else max(1.0, best_mu * 0.35)
        p_over = float(1.0 - norm.cdf(line, loc=ensemble_mean, scale=sd_est))

    p_over = float(np.clip(p_over, 0.01, 0.99))
    ev_leg_even = p_over - (1 - p_over)

    # Drift correction (very light – soft learning)
    if history_df is not None and not history_df.empty:
        ev_bias, _ = compute_model_drift(history_df)
        # Slightly nudge probability towards empirical bias
        p_over_adj = float(np.clip(0.5 + (p_over - 0.5) * ev_bias, 0.01, 0.99))
    else:
        p_over_adj = p_over

    leg = {
        "player": label or player,
        "market": market,
        "line": float(line),
        "mu": float(ensemble_mean),
        "boot_mean": float(boot_mean if boot_mean is not None else ensemble_mean),
        "sd_sample": float(np.std(values) if len(values) > 1 else 0.0),
        "prob_over": float(p_over_adj),
        "prob_over_raw": float(p_over),
        "ev_leg_even": float(ev_leg_even),
        "team": team_abbrev,
        "ctx_mult": float(ctx_mult),
        "msg": f"{label}: {len(values)} games • {script['exp_min']:.1f} exp. min",
        "teammate_out": bool(teammate_out),
        "blowout": bool(blowout),
        "exp_min": float(script["exp_min"]),
        "min_low": float(script["min_low"]),
        "min_high": float(script["min_high"]),
        "usage_center": float(script["usage_center"]),
        "var_scale": float(script["var_scale"]),
        "opp": opp_abbrev,
    }
    return leg, None

# =========================================================
#  HISTORY HELPERS
# =========================================================

def ensure_history():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac","Combo","Notes"
        ])
        df.to_csv(LOG_FILE, index=False)

def load_history():
    ensure_history()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac","Combo","Notes"
        ])

def save_history(df):
    df.to_csv(LOG_FILE, index=False)

# =========================================================
#  UI RENDERING HELPERS
# =========================================================

def render_leg_card(leg: dict, container, compact=False):
    """
    Displays player card with:
      - headshot
      - opponent defensive context
      - ensemble mean & probability
      - script expectations
    """
    player = leg["player"]
    market = leg["market"]
    msg = leg["msg"]
    line = leg["line"]
    mu = leg["mu"]
    p = leg["prob_over"]
    ctx = leg["ctx_mult"]
    even_ev = leg["ev_leg_even"]
    teammate_out = leg["teammate_out"]
    blowout = leg["blowout"]
    opp = leg.get("opp")

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

        col1, col2 = st.columns([1,2])
        with col1:
            if headshot:
                st.image(headshot, width=120)
        with col2:
            st.write(f"📌 **Line:** {line}")
            st.write(f"📊 **Ensemble Projection:** {mu:.2f}")
            st.write(f"🎯 **Model Probability Over:** {p*100:.1f}%")
            st.write(f"💵 **Even-Money EV:** {even_ev*100:+.1f}%")
            st.caption(f"📝 {msg}")

            if opp and opp in TEAM_CTX and LEAGUE_CTX:
                ctx_team = TEAM_CTX[opp]
                st.write(
                    f"🛡️ **Opponent ({opp}) Defense:** "
                    f"Def Rating {ctx_team['DEF_RATING']:.1f} | "
                    f"Pace {ctx_team['PACE']:.1f}"
                )
            st.write(
                f"⏱️ **Expected Minutes Range:** "
                f"{leg['min_low']:.1f} – {leg['min_high']:.1f} (center {leg['exp_min']:.1f})"
            )

        if teammate_out:
            st.info("⚠️ Key teammate out → usage & minutes boost applied.")
        if blowout:
            st.warning("⚠️ Blowout risk → minutes trimmed in downside scenarios.")

def run_loader():
    msgs = [
        "Pulling player logs…",
        "Analyzing matchup context…",
        "Simulating game scripts…",
        "Running bootstrap Monte Carlo…",
        "Finalizing edge & risk…",
    ]
    load_ph = st.empty()
    for m in msgs:
        load_ph.markdown(
            f"<p style='color:#FFCC33;font-size:20px;font-weight:600;'>{m}</p>",
            unsafe_allow_html=True,
        )
        time.sleep(0.35)
    load_ph.empty()

def combo_decision(ev_combo: float) -> str:
    if ev_combo >= 0.10:
        return "🔥 **PLAY — Strong Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **Lean — Thin Edge**"
    else:
        return "❌ **Pass — No Edge**"

# =========================================================
#  APP TABS
# =========================================================

tab_model, tab_results, tab_history, tab_calib, tab_scanner = st.tabs(
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration", "📡 Edge Scanner"]
)

# =========================================================
#  MODEL TAB
# =========================================================

with tab_model:
    st.subheader("2-Pick Projection & Edge (Empirical Bootstrap + Ensemble)")

    history_df = load_history()

    c1, c2 = st.columns(2)

    # LEFT LEG
    with c1:
        st.markdown("### Leg 1")
        if data_source_mode == "Underdog Autofill":
            st.caption("You can use the Edge Scanner tab to pick targets, then manually enter here.")
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="m1")
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5, key="l1")
        o1 = st.text_input("P1 Opponent (abbr)", help="Example: BOS, DEN", key="o1")
        p1_teammate_out = st.checkbox("P1: Auto/Manual Key Teammate Out?", value=False, key="p1_tm")
        p1_blowout = st.checkbox("P1: Blowout risk high?", value=False, key="p1_bo")

    # RIGHT LEG
    with c2:
        st.markdown("### Leg 2")
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="m2")
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5, key="l2")
        o2 = st.text_input("P2 Opponent (abbr)", help="Example: BOS, DEN", key="o2")
        p2_teammate_out = st.checkbox("P2: Auto/Manual Key Teammate Out?", value=False, key="p2_tm")
        p2_blowout = st.checkbox("P2: Blowout risk high?", value=False, key="p2_bo")

    leg1 = None
    leg2 = None

    run = st.button("Run Model ⚡")

    if run:
        if payout_mult_2pick <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

        # Compute legs
        leg1, err1 = (
            compute_leg_projection(
                p1, m1, l1, o1, p1_teammate_out, p1_blowout, games_lookback, history_df
            )
            if p1 and l1 > 0 else (None, None)
        )
        leg2, err2 = (
            compute_leg_projection(
                p2, m2, l2, o2, p2_teammate_out, p2_blowout, games_lookback, history_df
            )
            if p2 and l2 > 0 else (None, None)
        )

        if err1:
            st.error(f"P1: {err1}")
        if err2:
            st.error(f"P2: {err2}")

        colL, colR = st.columns(2)
        if leg1:
            render_leg_card(leg1, colL, compact_mode)
            update_market_library(leg1["player"], leg1["market"], leg1["line"])
        if leg2:
            render_leg_card(leg2, colR, compact_mode)
            update_market_library(leg2["player"], leg2["market"], leg2["line"])

        st.markdown("---")
        st.subheader("📈 Market vs Model Probability Check")

        def implied_probability_from_payout(mult):
            # Single leg in 2-pick style environments is roughly 0.5 implied,
            # but this scales so you can tune later.
            return 1.0 / mult

        imp_prob_single = implied_probability_from_payout(payout_mult_2pick)

        if leg1:
            st.markdown(
                f"**{leg1['player']} Model Prob:** {leg1['prob_over']*100:.1f}% "
                f"→ Edge vs ~{imp_prob_single*100:.1f}%: "
                f"{(leg1['prob_over'] - imp_prob_single)*100:+.1f}%"
            )
        if leg2:
            st.markdown(
                f"**{leg2['player']} Model Prob:** {leg2['prob_over']*100:.1f}% "
                f"→ Edge vs ~{imp_prob_single*100:.1f}%: "
                f"{(leg2['prob_over'] - imp_prob_single)*100:+.1f}%"
            )

        # 2-PICK COMBO
        if leg1 and leg2:
            corr = estimate_player_correlation(leg1, leg2)
            joint = joint_gaussian_mc(leg1["prob_over"], leg2["prob_over"], corr, n_sims=N_SIMS)
            ev_combo = payout_mult_2pick * joint - 1.0
            k_frac = kelly_for_combo(joint, payout_mult_2pick, fractional_kelly)

            # Daily loss cap logic (based on today's PnL)
            hist = load_history()
            today_str = datetime.now().strftime("%Y-%m-%d")
            todays = hist[hist["Date"].str.startswith(today_str)] if not hist.empty else pd.DataFrame()
            if not todays.empty:
                pnl_today = todays.apply(
                    lambda r:
                        r["Stake"] * (payout_mult_2pick - 1.0)
                        if r["Result"] == "Hit"
                        else (-r["Stake"] if r["Result"] == "Miss" else 0.0),
                    axis=1
                ).sum()
            else:
                pnl_today = 0.0

            loss_cap = -bankroll * (daily_loss_cap_pct / 100.0)
            if pnl_today <= loss_cap:
                st.error("🚫 Daily loss cap reached — suggested stake set to $0 until tomorrow.")
                k_frac_sanitized = 0.0
            else:
                k_frac_sanitized = k_frac

            stake = round(bankroll * k_frac_sanitized, 2)
            decision = combo_decision(ev_combo)

            st.markdown("### 🎯 **2-Pick Combo Result (Covariance-Based Joint MC)**")
            st.markdown(f"- Correlation: **{corr:+.2f}**")
            st.markdown(f"- Joint Probability (MC): **{joint*100:.1f}%**")
            st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Suggested Stake (Kelly-capped + Daily Loss Cap): **${stake:.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")

# =========================================================
#  RESULTS TAB
# =========================================================

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
                "CLV (Closing - Entry, in pts)",  # simple unit
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1
            )

        r_result = st.selectbox(
            "Result",
            ["Pending", "Hit", "Miss", "Push"]
        )

        r_combo = st.checkbox("This was a 2-pick combo entry?")
        r_notes = st.text_input("Notes (optional)")

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
                "KellyFrac": fractional_kelly,
                "Combo": r_combo,
                "Notes": r_notes,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_history(df)
            st.success("Result logged ✅")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if not comp.empty:
        pnl = comp.apply(
            lambda r:
                r["Stake"] * (payout_mult_2pick - 1.0)
                if r["Result"] == "Hit"
                else -r["Stake"],
            axis=1,
        )

        hits = (comp["Result"] == "Hit").sum()
        total = len(comp)

        hit_rate = (hits / total * 100) if total > 0 else 0.0
        roi = pnl.sum() / max(bankroll, 1.0) * 100

        mean_clv = pd.to_numeric(comp["CLV"], errors="coerce").fillna(0.0).mean()
        pnl_var = float(np.var(pnl)) if len(pnl) > 1 else 0.0

        st.markdown(
            f"**Completed Bets:** {total}  |  "
            f"**Hit Rate:** {hit_rate:.1f}%  |  "
            f"**ROI (vs current bankroll):** {roi:+.1f}%  |  "
            f"**Avg CLV:** {mean_clv:+.2f} pts  |  "
            f"**Outcome Variance:** {pnl_var:.2f}"
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
    else:
        st.info("No completed bets yet for summary metrics.")

# =========================================================
#  HISTORY TAB
# =========================================================

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

        combo_only = st.checkbox("Show only Combo (2-pick) entries", value=False)

        filt = df[df["EV"] >= min_ev]

        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        if combo_only:
            filt = filt[filt["Combo"] == True]

        st.markdown(f"**Filtered Bets:** {len(filt)}")
        st.dataframe(filt, use_container_width=True)

        if not filt.empty:
            filt = filt.copy()
            filt["Net"] = filt.apply(
                lambda r:
                    r["Stake"] * (payout_mult_2pick - 1.0)
                    if r["Result"] == "Hit"
                    else (
                        -r["Stake"] if r["Result"] == "Miss" else 0.0
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

# =========================================================
#  CALIBRATION TAB
# =========================================================

with tab_calib:
    st.subheader("Calibration & Edge Integrity Check")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if comp.empty or len(comp) < 15:
        st.info("Log at least 15 completed bets with EV to start calibration.")
    else:
        comp = comp.copy()
        comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
        comp = comp.dropna(subset=["EV_float"])

        if comp.empty:
            st.info("No valid EV values yet.")
        else:
            pred_win_prob = 0.5 + comp["EV_float"].mean()
            actual_win_prob = (comp["Result"] == "Hit").mean()
            gap = (pred_win_prob - actual_win_prob) * 100

            pnl = comp.apply(
                lambda r:
                    r["Stake"] * (payout_mult_2pick - 1.0)
                    if r["Result"] == "Hit"
                    else -r["Stake"],
                axis=1,
            )
            roi = pnl.sum() / max(1.0, bankroll) * 100

            st.markdown("---")
            st.subheader("Market vs Model Performance Trend")

            comp["Edge_vs_Market"] = comp["EV_float"] * 100

            fig2 = px.histogram(
                comp,
                x="Edge_vs_Market",
                nbins=20,
                title="Distribution of Model Edge vs Market (EV%)"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # EV buckets
            buckets = pd.cut(
                comp["EV_float"] * 100,
                bins=[-5, 0, 5, 10, 20, 100],
                labels=["<0%", "0–5%", "5–10%", "10–20%", "20%+"]
            )
            comp["EV_bucket"] = buckets
            bucket_stats = comp.groupby("EV_bucket")["Result"].apply(
                lambda x: (x == "Hit").mean() * 100
            ).reset_index(name="HitRate")

            st.markdown("### EV Buckets vs Realized Hit Rate")
            st.dataframe(bucket_stats, use_container_width=True)

            st.markdown(
                f"**Predicted Avg Win Prob (approx):** {pred_win_prob*100:.1f}%\n\n"
                f"**Actual Hit Rate:** {actual_win_prob*100:.1f}%\n\n"
                f"**Calibration Gap:** {gap:+.1f}% | **ROI:** {roi:+.1f}%"
            )

            if gap > 5:
                st.warning(
                    "Model appears overconfident → consider requiring higher EV before firing."
                )
            elif gap < -5:
                st.info(
                    "Model appears conservative → thin edges may be slightly under-trusted."
                )
            else:
                st.success("Model and results are reasonably aligned ✅")

# =========================================================
#  EDGE SCANNER TAB (UNDERDOG LIVE LINES)
# =========================================================

with tab_scanner:
    st.subheader("Live Edge Scanner — Underdog NBA (PTS / REB / AST / PRA)")

    st.caption(
        "This scanner attempts to pull all NBA prop lines from Underdog, "
        "run the same ensemble + bootstrap engine, and rank edges. "
        "If Underdog changes their API or structure, this will gracefully fail."
    )

    scan_button = st.button("Scan Underdog for Edges 🚀")

    if scan_button:
        lines_df = fetch_underdog_nba_lines()

        if lines_df.empty:
            st.error("No Underdog data available. The endpoint structure may have changed, or there is a network issue.")
        else:
            st.success(f"Pulled {len(lines_df)} Underdog NBA props. Running model…")

            records = []
            history_df = load_history()

            # Limit for performance; you can raise this if needed
            max_props = min(120, len(lines_df))
            subset = lines_df.head(max_props)

            for _, r in subset.iterrows():
                player_name = r["player"]
                market = r["market"]
                line_val = float(r["line"])
                team_abbrev = r.get("team") or ""
                opp_abbrev = r.get("opp") or ""

                leg, err = compute_leg_projection(
                    player_name,
                    market,
                    line_val,
                    opp_abbrev,
                    teammate_out=False,
                    blowout=False,
                    n_games=games_lookback,
                    history_df=history_df
                )
                if leg is None or err:
                    continue

                # Implied probability baseline (can refine later)
                imp_prob = 0.5
                model_prob = leg["prob_over"]
                edge = model_prob - imp_prob
                ev_diff_pct = edge * 100

                # Tier classification
                if ev_diff_pct >= 8.0:
                    tier = "Elite"
                elif ev_diff_pct >= 4.0:
                    tier = "Medium"
                elif ev_diff_pct >= 1.5:
                    tier = "Thin"
                else:
                    tier = "No Edge"

                records.append({
                    "Player": leg["player"],
                    "Team": leg.get("team"),
                    "Opponent": leg.get("opp"),
                    "Market": market,
                    "Line": line_val,
                    "Model_Line": leg["mu"],
                    "Model_Prob_%": model_prob * 100.0,
                    "Implied_Prob_%": imp_prob * 100.0,
                    "EV_Diff_%": ev_diff_pct,
                    "Tier": tier,
                })

            if not records:
                st.warning("Model could not resolve any players from the Underdog feed. Check names and try again later.")
            else:
                edges_df = pd.DataFrame(records)
                # Correlated edges annotation: same team or same opponent flagged
                edges_df["Correlated_Flag"] = edges_df.duplicated(subset=["Team"], keep=False) | \
                                              edges_df.duplicated(subset=["Opponent"], keep=False)

                edges_df = edges_df.sort_values("EV_Diff_%", ascending=False)

                st.markdown("### Top Edges")
                st.dataframe(edges_df, use_container_width=True)

                st.markdown("#### Elite / Medium Edges")
                top_tiers = edges_df[edges_df["Tier"].isin(["Elite","Medium"])]
                if not top_tiers.empty:
                    st.dataframe(top_tiers, use_container_width=True)
                else:
                    st.info("No Elite/Medium edges detected under current settings.")

# =========================================================
#  FOOTER
# =========================================================

st.markdown(
    """
    <footer style='text-align:center; margin-top:30px; color:#FFCC33; font-size:11px;'>
        © 2025 NBA Prop Quant Engine • Built for Kamal — Empirical Bootstrap • Ensemble • Covariance MC • Underdog Scanner
    </footer>
    """,
    unsafe_allow_html=True,
)
