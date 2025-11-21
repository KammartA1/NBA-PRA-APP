# =========================================================
#  NBA PROP BETTING QUANT ENGINE — UPGRADED SINGLE-FILE APP
#  Based on: nba_prop_model_option_b2 (1).py
#  Upgrades:
#    - Odds API live line support (Points / Rebounds / Assists / PRA)
#    - Monte Carlo simulation (10,000 sims single + joint)
#    - Covariance-based joint MC
#    - Volatility engine (defense, pace, role, blowout)
#    - Bankroll controls (fractional Kelly, daily/weekly loss caps)
#    - Expanded tracking (hit rate, ROI, CLV, variance)
#    - Self-learning calibration via EV buckets + drift scaling
#    - Live edge scanner (inside Model tab) with EV > 65% filter
#    - Auto-history logging: "Did you place this bet?"
# =========================================================

import os, time, random, difflib, json
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
    page_title="NBA Prop Quant Engine",
    page_icon="🏀",
    layout="wide"
)

# Use a temp dir for all persistent state
TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
CARD_BG = "#17131C"
BG = "#0D0A12"

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

# Bankroll risk caps
st.sidebar.subheader("Risk Controls")
max_daily_loss_pct = st.sidebar.slider("Max daily loss (% of bankroll)", 0.0, 100.0, 20.0, 1.0)
max_weekly_loss_pct = st.sidebar.slider("Max weekly loss (% of bankroll)", 0.0, 100.0, 40.0, 1.0)

st.sidebar.caption("Model auto-pulls NBA stats & Odds API lines. You mostly choose the players and markets.")

# =========================================================
#  PART 2.1 — MODEL CONSTANTS
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

# Odds API configuration
ODDS_API_KEY = "621ec92ab709da9f9ce59cf2e513af55"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_USAGE_FILE = os.path.join(TEMP_DIR, "odds_api_usage.json")
ODDS_MAX_REQUESTS_PER_DAY = 20

# =========================================================
#  UTILS — DATES & ODDS API USAGE GOVERNOR
# =========================================================

def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def load_odds_usage():
    if not os.path.exists(ODDS_USAGE_FILE):
        return {"date": today_str(), "count": 0}
    try:
        with open(ODDS_USAGE_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = {"date": today_str(), "count": 0}
    if data.get("date") != today_str():
        data = {"date": today_str(), "count": 0}
    return data


def save_odds_usage(data):
    try:
        with open(ODDS_USAGE_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def can_use_odds_api():
    """Cap daily Odds API requests at 20, and restrict to mornings / pre-slate."""
    data = load_odds_usage()
    if data["count"] >= ODDS_MAX_REQUESTS_PER_DAY:
        return False, "Daily Odds API request cap reached (20)."
    now = datetime.now()
    # Soft rule: only auto-hit Odds API before 5pm local
    if now.hour >= 17:
        return False, "Odds API restricted to pre-slate hours (before 5pm)."
    return True, ""


def register_odds_call():
    data = load_odds_usage()
    data["count"] = int(data.get("count", 0)) + 1
    save_odds_usage(data)

# =========================================================
#  PART 2.2 — PLAYER LOOKUP HELPERS
# =========================================================

def current_season():
    today = datetime.now()
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"


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
#  PART 2.3 — TEAM CONTEXT (PACE, DEF, REB%, AST%)
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

        # League averages
        league_avg = {
            col: df[col].mean()
            for col in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]
        }

        # Context per team
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
    """Adjust projection using advanced opponent factors."""
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
#  PART 2.4 — MARKET BASELINE LIBRARY
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
#  PART 3 — PLAYER GAME LOG ENGINE & PROJECTION MODEL
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_logs(name: str, n_games: int, market: str):
    """
    Pulls recent player logs, computes:
      - per-minute samples
      - minutes samples
      - team abbreviation
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
    per_min_vals = []
    minutes_vals = []

    for _, r in gl.iterrows():
        m = 0
        try:
            m_str = r.get("MIN", "0")
            if isinstance(m_str, str) and ":" in m_str:
                mm, ss = m_str.split(":")
                m = float(mm) + float(ss) / 60
            else:
                m = float(m_str)
        except Exception:
            m = 0

        if m <= 0:
            continue

        total_val = sum(float(r.get(c, 0)) for c in cols)
        per_min_vals.append(total_val / m)
        minutes_vals.append(m)

    if not per_min_vals:
        return None, None, None, None, "Insufficient data."

    per_min_vals = np.array(per_min_vals, dtype=float)
    minutes_vals = np.array(minutes_vals, dtype=float)

    team = None
    try:
        team = gl["TEAM_ABBREVIATION"].mode().iloc[0]
    except Exception:
        team = None

    msg = f"{label}: {len(per_min_vals)} games • {minutes_vals.mean():.1f} min"
    return per_min_vals, minutes_vals, team, gl, msg


def summarize_rates(per_min_vals: np.ndarray, minutes_vals: np.ndarray):
    """
    Weighted bootstrap-friendly rates.
    """
    if per_min_vals is None or len(per_min_vals) == 0:
        return None, None, None
    weights = np.linspace(0.5, 1.5, len(per_min_vals))
    weights /= weights.sum()

    mu_per_min = float(np.average(per_min_vals, weights=weights))
    avg_min = float(np.average(minutes_vals, weights=weights))
    sd_per_min = float(
        max(
            np.sqrt(np.average((per_min_vals - mu_per_min) ** 2, weights=weights)),
            0.15 * max(mu_per_min, 0.5),
        )
    )
    return mu_per_min, sd_per_min, avg_min

# =====================================================
# HYBRID ANALYTIC ENGINE
# =====================================================

def hybrid_prob_over(line, mu, sd, market):
    normal_p = 1 - norm.cdf(line, mu, sd)

    if mu <= 0 or sd <= 0 or np.isnan(mu) or np.isnan(sd):
        return float(np.clip(normal_p, 0.01, 0.99))

    try:
        variance = sd ** 2
        phi = np.sqrt(variance + mu ** 2)

        mu_log = np.log(mu ** 2 / phi)
        sd_log = np.sqrt(np.log(phi ** 2 / mu ** 2))

        if np.isnan(mu_log) or np.isnan(sd_log) or sd_log <= 0:
            lognorm_p = normal_p
        else:
            lognorm_p = 1 - norm.cdf(np.log(line + 1e-9), mu_log, sd_log)
    except Exception:
        lognorm_p = normal_p

    w = {
        "PRA": 0.70,
        "Points": 0.55,
        "Rebounds": 0.40,
        "Assists": 0.30
    }.get(market, 0.50)

    hybrid = w * lognorm_p + (1 - w) * normal_p
    return float(np.clip(hybrid, 0.02, 0.98))

# ======================================================
# ADVANCED PLAYER CORRELATION ENGINE
# ======================================================

def estimate_player_correlation(leg1, leg2):
    corr = 0.0

    if leg1["team"] == leg2["team"] and leg1["team"] is not None:
        corr += 0.18

    try:
        avg_min1 = leg1["minutes"]
        avg_min2 = leg2["minutes"]
    except Exception:
        avg_min1 = avg_min2 = 28.0

    if avg_min1 > 30 and avg_min2 > 30:
        corr += 0.05
    elif avg_min1 < 22 or avg_min2 < 22:
        corr -= 0.04

    m1, m2 = leg1["market"], leg2["market"]

    if m1 == "Points" and m2 == "Points":
        corr += 0.08
    if (m1 == "Points" and m2 == "Assists") or (m1 == "Assists" and m2 == "Points"):
        corr -= 0.10
    if (m1 == "Rebounds" and m2 == "Points") or (m1 == "Points" and m2 == "Rebounds"):
        corr -= 0.06
    if m1 == "PRA" or m2 == "PRA":
        corr += 0.03

    ctx1, ctx2 = leg1["ctx_mult"], leg2["ctx_mult"]

    if ctx1 > 1.03 and ctx2 > 1.03:
        corr += 0.04
    if ctx1 < 0.97 and ctx2 < 0.97:
        corr += 0.03
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 1.03):
        corr -= 0.05

    corr = float(np.clip(corr, -0.25, 0.40))
    return corr

# =====================================================
# VOLATILITY ENGINE + CORE PROJECTION
# =====================================================

def apply_volatility(mu_min, sd_min, avg_min, market, opp, teammate_out, blowout):
    minutes = avg_min
    if teammate_out:
        minutes *= 1.05
        mu_min *= 1.06
    if blowout:
        minutes *= 0.90

    ctx_mult = get_context_multiplier(opp.strip().upper() if opp else None, market)

    mu = mu_min * minutes * ctx_mult
    heavy = HEAVY_TAIL[market]
    sd_base = max(1.0, sd_min * np.sqrt(max(minutes, 1.0)) * heavy)
    sd_final = sd_base

    if opp:
        opp_abbrev = opp.strip().upper()
        if opp_abbrev in TEAM_CTX:
            opp_def = TEAM_CTX[opp_abbrev]["DEF_RATING"]
            league_def = LEAGUE_CTX["DEF_RATING"]
            def_vol = np.clip(opp_def / league_def, 0.85, 1.20)
            sd_final *= def_vol

            opp_pace = TEAM_CTX[opp_abbrev]["PACE"]
            league_pace = LEAGUE_CTX["PACE"]
            pace_vol = np.clip(opp_pace / league_pace, 0.90, 1.18)
            sd_final *= pace_vol

    if market == "Rebounds":
        sd_final *= 1.10
    elif market == "Assists":
        sd_final *= 1.06
    elif market == "Points":
        sd_final *= 1.12
    elif market == "PRA":
        sd_final *= 1.15

    if teammate_out:
        sd_final *= 1.07
    if blowout:
        sd_final *= 1.10

    sd_final *= (1 + 0.10 * (heavy - 1))
    sd_final = float(np.clip(sd_final, sd_base * 0.80, sd_base * 1.60))

    return mu, sd_final, minutes, ctx_mult

# =====================================================
# SELF-LEARNING CALIBRATION ENGINE
# =====================================================

def compute_model_drift(history_df: pd.DataFrame):
    """
    Look at EV buckets and realized hit rates to adjust:
      - probability scaling
      - variance scaling (soft)
    Returns (prob_scale, vol_scale).
    """
    if history_df is None or history_df.empty:
        return 1.0, 1.0

    df = history_df.copy()
    df = df[df["Result"].isin(["Hit", "Miss"])]
    if df.empty:
        return 1.0, 1.0

    try:
        df["EV_float"] = pd.to_numeric(df["EV"], errors="coerce") / 100.0
    except Exception:
        return 1.0, 1.0

    df = df.dropna(subset=["EV_float"])
    if df.empty:
        return 1.0, 1.0

    df["is_hit"] = (df["Result"] == "Hit").astype(int)
    pred_prob = 0.5 + df["EV_float"]
    pred_prob = pred_prob.clip(0.01, 0.99)

    actual_prob = df["is_hit"].mean()
    expected_prob = pred_prob.mean()

    if expected_prob <= 0:
        return 1.0, 1.0

    prob_scale = float(np.clip(actual_prob / expected_prob, 0.6, 1.4))

    pnl = df.apply(
        lambda r:
            r["Stake"] * (payout_mult - 1.0)
            if r["Result"] == "Hit"
            else -r["Stake"],
        axis=1,
    )
    vol_scale = float(np.clip((pnl.std() / max(abs(pnl.mean()), 1e-6)) if len(pnl) > 1 else 1.0, 0.7, 1.3))

    return prob_scale, vol_scale

# =====================================================
# MONTE CARLO ENGINES
# =====================================================

def run_monte_carlo_leg(leg: dict, n_sims: int = 10000, rng=None):
    """
    Simulate a single leg outcome distribution.
    Fully defense-adjusted via leg['mu'] and leg['sd'] (already context-adjusted).
    """
    if rng is None:
        rng = np.random.default_rng()

    mu = leg["mu"]
    sd = leg["sd"]
    line = leg["line"]

    sims = rng.normal(mu, sd, size=n_sims)
    prob_over_mc = float(np.mean(sims > line))

    leg["prob_over_mc"] = prob_over_mc
    return sims, prob_over_mc


def run_joint_monte_carlo(leg1: dict, leg2: dict, corr: float, n_sims: int = 10000, rng=None):
    """
    Covariance-based joint MC using correlated normal draws.
    """
    if rng is None:
        rng = np.random.default_rng()

    mu1, sd1, line1 = leg1["mu"], leg1["sd"], leg1["line"]
    mu2, sd2, line2 = leg2["mu"], leg2["sd"], leg2["line"]

    cov = corr * sd1 * sd2
    cov_matrix = np.array([[sd1 ** 2, cov], [cov, sd2 ** 2]])

    try:
        sims = rng.multivariate_normal([mu1, mu2], cov_matrix, size=n_sims)
    except np.linalg.LinAlgError:
        sims = rng.normal([mu1, mu2], [sd1, sd2], size=(n_sims, 2))

    over1 = sims[:, 0] > line1
    over2 = sims[:, 1] > line2
    joint_prob_mc = float(np.mean(over1 & over2))

    return joint_prob_mc

# =====================================================
# LEG PROJECTION WRAPPER
# =====================================================

def compute_leg_projection(player, market, line, opp, teammate_out, blowout, n_games, history_df):
    """
    Core projection engine for a single leg, including:
      - rolling weighted stats
      - volatility engine
      - hybrid analytic probability
      - Monte Carlo leg simulation
      - defense-adjusted at every step
    """
    per_min_vals, minutes_vals, team, gl, msg = get_player_logs(player, n_games, market)
    if per_min_vals is None:
        return None, msg

    mu_min, sd_min, avg_min = summarize_rates(per_min_vals, minutes_vals)
    if mu_min is None:
        return None, "Insufficient data."

    mu, sd_final, minutes, ctx_mult = apply_volatility(
        mu_min, sd_min, avg_min, market, opp, teammate_out, blowout
    )

    base_prob = hybrid_prob_over(line, mu, sd_final, market)

    hist_df = history_df if history_df is not None else pd.DataFrame()
    prob_scale, vol_scale = compute_model_drift(hist_df)

    mu_adj = mu
    sd_adj = sd_final * vol_scale
    prob_adj = float(np.clip(base_prob * prob_scale, 0.02, 0.98))

    leg = {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": float(mu_adj),
        "sd": float(sd_adj),
        "prob_over": float(prob_adj),
        "team": team,
        "ctx_mult": float(ctx_mult),
        "msg": msg,
        "teammate_out": bool(teammate_out),
        "blowout": bool(blowout),
        "minutes": float(minutes),
    }

    sims, prob_over_mc = run_monte_carlo_leg(leg, n_sims=10000)
    leg["prob_over_mc"] = float(prob_over_mc)
    leg["prob_over_final"] = float(np.clip(0.5 * leg["prob_over"] + 0.5 * prob_over_mc, 0.02, 0.98))

    ev_leg_even = leg["prob_over_final"] - (1 - leg["prob_over_final"])
    leg["ev_leg_even"] = float(ev_leg_even)

    return leg, None

# =========================================================
#  PART 3.2 — KELLY FORMULA FOR 2-PICK
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float):
    b = payout_mult - 1
    q = 1 - p_joint
    raw = (b * p_joint - q) / b
    k = raw * frac
    return float(np.clip(k, 0, MAX_KELLY_PCT))

# =====================================================
# HISTORY HELPERS
# =====================================================

def ensure_history():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac","JointProb","PayoutMult"
        ])
        df.to_csv(LOG_FILE, index=False)

def load_history():
    ensure_history()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac","JointProb","PayoutMult"
        ])

def save_history(df):
    df.to_csv(LOG_FILE, index=False)

# =====================================================
# BANKROLL RISK CHECKS
# =====================================================

def compute_pnl(df: pd.DataFrame):
    if df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["Net"] = df.apply(
        lambda r:
            r["Stake"] * (r.get("PayoutMult", payout_mult) - 1.0)
            if r["Result"] == "Hit"
            else (-r["Stake"] if r["Result"] == "Miss" else 0.0),
        axis=1,
    )
    return df["Net"]

def risk_lockout_flags(df: pd.DataFrame, bankroll: float, max_daily_pct: float, max_weekly_pct: float):
    if df.empty:
        return False, False, 0.0, 0.0

    df = df.copy()
    df["DateOnly"] = pd.to_datetime(df["Date"]).dt.date
    pnl = compute_pnl(df)

    df["Net"] = pnl.values

    today = date.today()
    week_ago = today - timedelta(days=7)

    daily_pnl = df[df["DateOnly"] == today]["Net"].sum()
    weekly_pnl = df[(df["DateOnly"] >= week_ago) & (df["DateOnly"] <= today)]["Net"].sum()

    daily_limit = -bankroll * max_daily_pct / 100.0
    weekly_limit = -bankroll * max_weekly_pct / 100.0

    daily_lock = daily_pnl <= daily_limit
    weekly_lock = weekly_pnl <= weekly_limit

    return daily_lock, weekly_lock, daily_pnl, weekly_pnl

# =========================================================
# PART 4 — UI RENDER ENGINE + LOADERS + DECISION LOGIC
# =========================================================

def render_leg_card(leg: dict, container, compact=False):
    player = leg["player"]
    market = leg["market"]
    msg = leg["msg"]
    line = leg["line"]
    mu = leg["mu"]
    sd = leg["sd"]
    p = leg["prob_over_final"]
    ctx = leg["ctx_mult"]
    even_ev = leg["ev_leg_even"]
    teammate_out = leg["teammate_out"]
    blowout = leg["blowout"]
    prob_mc = leg.get("prob_over_mc", None)

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

        if headshot and not compact:
            st.image(headshot, width=120)

        st.write(f"📌 **Line:** {line}")
        st.write(f"📊 **Model Mean:** {mu:.2f}")
        st.write(f"📉 **Model SD (volatility-adjusted):** {sd:.2f}")
        st.write(f"⏱️ **Context Multiplier (pace+def+reb/ast):** {ctx:.3f}")
        if prob_mc is not None:
            st.write(f"🎯 **Analytic Prob Over:** {leg['prob_over']*100:.1f}%")
            st.write(f"🎲 **Monte Carlo Prob Over (10k sims):** {prob_mc*100:.1f}%")
        st.write(f"🧠 **Final Blended Prob Over:** {p*100:.1f}%")
        st.write(f"💵 **Even-Money EV (defense-adjusted):** {even_ev*100:+.1f}%")
        st.caption(f"📝 {msg}")

        if teammate_out:
            st.info("⚠️ Key teammate out → usage & minutes boost applied.")
        if blowout:
            st.warning("⚠️ Blowout risk → minutes reduced, volatility increased.")

def run_loader():
    load_ph = st.empty()
    msgs = [
        "Pulling player logs…",
        "Analyzing matchup & pace…",
        "Building distributions…",
        "Running Monte Carlo (10,000 sims)…",
        "Optimizing stake & edge…",
    ]
    for m in msgs:
        load_ph.markdown(
            f"<p style='color:#FFCC33;font-size:20px;font-weight:600;'>{m}</p>",
            unsafe_allow_html=True,
        )
        time.sleep(0.25)
    load_ph.empty()

def combo_decision(ev_combo: float) -> str:
    if ev_combo >= 0.10:
        return "🔥 **PLAY — Strong Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **Lean — Thin Edge**"
    else:
        return "❌ **Pass — No Edge**"

# =====================================================
# ODDS API LINE FETCHER
# =====================================================

def fetch_live_line_from_odds_api(player_name: str, market: str):
    """
    Very lightweight Odds API wrapper.
    NOTE: The Odds API's exact player prop endpoints may vary; this is a best-effort
    implementation that fails gracefully and falls back to manual input.
    """
    allow, reason = can_use_odds_api()
    if not allow:
        return None, reason

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "player_props",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    url = f"{ODDS_API_BASE}/sports/basketball_nba/odds"
    try:
        resp = requests.get(url, params=params, timeout=8)
        register_odds_call()
        if resp.status_code != 200:
            return None, f"Odds API error: {resp.status_code}"
        data = resp.json()
    except Exception as e:
        return None, f"Odds API error: {e}"

    player_norm = _norm_name(player_name)
    target_markets = {
        "Points": ["points", "player_points"],
        "Rebounds": ["rebounds", "player_rebounds"],
        "Assists": ["assists", "player_assists"],
        "PRA": ["pra", "points_rebounds_assists", "player_pra"],
    }.get(market, [])

    best_line = None
    for game in data:
        for bookmaker in game.get("bookmakers", []):
            for market_obj in bookmaker.get("markets", []):
                key = market_obj.get("key", "").lower()
                if not any(k in key for k in target_markets):
                    continue
                for outcome in market_obj.get("outcomes", []):
                    name = _norm_name(outcome.get("description", ""))
                    if player_norm in name:
                        try:
                            line_val = float(outcome.get("point", None))
                        except Exception:
                            continue
                        if best_line is None or abs(line_val - (best_line or line_val)) < 0.25:
                            best_line = line_val
    if best_line is None:
        return None, "No matching live line found."
    return best_line, ""

# =====================================================
# APP TABS (KEEP EXISTING STRUCTURE)
# =====================================================

tab_model, tab_results, tab_history, tab_calib = st.tabs(
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration"]
)

# =====================================================
# PART 5 — MODEL TAB
# =====================================================

with tab_model:
    st.subheader("2-Pick Projection & Edge (Auto stats + Odds API lines)")

    history_df = load_history()

    daily_lock, weekly_lock, daily_pnl, weekly_pnl = risk_lockout_flags(
        history_df, bankroll, max_daily_loss_pct, max_weekly_loss_pct
    )

    if daily_lock:
        st.error(f"Daily loss cap reached (P&L today: ${daily_pnl:.2f}). New stakes are set to $0 by discipline rule.")
    if weekly_lock:
        st.warning(f"Weekly loss cap reached (7-day P&L: ${weekly_pnl:.2f}). Stakes heavily constrained.")

    c1, c2 = st.columns(2)

    with c1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="p1_market")
        auto_line_p1 = st.checkbox("Pull P1 line from Odds API (if available)", key="p1_auto_line")
        if auto_line_p1 and p1:
            if st.button("Fetch P1 Live Line", key="btn_p1_fetch"):
                line_p1, msg_p1 = fetch_live_line_from_odds_api(p1, m1)
                if line_p1 is None:
                    st.warning(msg_p1 or "Unable to fetch live line.")
                    l1 = st.number_input("P1 Line (manual)", min_value=0.0, value=25.0, step=0.5, key="p1_line_manual")
                else:
                    st.success(f"Fetched live line: {line_p1}")
                    l1 = st.number_input("P1 Line", min_value=0.0, value=float(line_p1), step=0.5, key="p1_line_live")
            else:
                l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5, key="p1_line_default")
        else:
            l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5, key="p1_line_plain")

        o1 = st.text_input("P1 Opponent (abbr)", help="Example: BOS, DEN", key="p1_opp")
        p1_teammate_out = st.checkbox("P1: Key teammate out?", key="p1_teammate_out")
        p1_blowout = st.checkbox("P1: Blowout risk high?", key="p1_blowout")

    with c2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="p2_market")
        auto_line_p2 = st.checkbox("Pull P2 line from Odds API (if available)", key="p2_auto_line")
        if auto_line_p2 and p2:
            if st.button("Fetch P2 Live Line", key="btn_p2_fetch"):
                line_p2, msg_p2 = fetch_live_line_from_odds_api(p2, m2)
                if line_p2 is None:
                    st.warning(msg_p2 or "Unable to fetch live line.")
                    l2 = st.number_input("P2 Line (manual)", min_value=0.0, value=25.0, step=0.5, key="p2_line_manual")
                else:
                    st.success(f"Fetched live line: {line_p2}")
                    l2 = st.number_input("P2 Line", min_value=0.0, value=float(line_p2), step=0.5, key="p2_line_live")
            else:
                l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5, key="p2_line_default")
        else:
            l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5, key="p2_line_plain")

        o2 = st.text_input("P2 Opponent (abbr)", help="Example: BOS, DEN", key="p2_opp")
        p2_teammate_out = st.checkbox("P2: Key teammate out?", key="p2_teammate_out")
        p2_blowout = st.checkbox("P2: Blowout risk high?", key="p2_blowout")

    leg1 = None
    leg2 = None
    joint_result = None

    run = st.button("Run Model ⚡")

    if run:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

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
        if leg2:
            render_leg_card(leg2, colR, compact_mode)

        st.markdown("---")
        st.subheader("📈 Market vs Model Probability Check")

        def implied_probability(mult):
            return 1.0 / mult

        imp_prob = implied_probability(payout_mult)
        st.markdown(f"**Market Implied Probability (2-pick fair):** {imp_prob*100:.1f}%")

        if leg1:
            st.markdown(
                f"**{leg1['player']} Model Prob (final):** {leg1['prob_over_final']*100:.1f}% "
                f"→ Edge vs market: {(leg1['prob_over_final'] - imp_prob)*100:+.1f}%"
            )
        if leg2:
            st.markdown(
                f"**{leg2['player']} Model Prob (final):** {leg2['prob_over_final']*100:.1f}% "
                f"→ Edge vs market: {(leg2['prob_over_final'] - imp_prob)*100:+.1f}%"
            )

        if leg1 and leg2:
            corr = estimate_player_correlation(leg1, leg2)

            base_joint = leg1["prob_over_final"] * leg2["prob_over_final"]
            joint_analytic = base_joint + corr * (
                min(leg1["prob_over_final"], leg2["prob_over_final"]) - base_joint
            )
            joint_analytic = float(np.clip(joint_analytic, 0.0, 1.0))

            joint_mc = run_joint_monte_carlo(leg1, leg2, corr, n_sims=10000)

            joint_final = float(np.clip(0.5 * joint_analytic + 0.5 * joint_mc, 0.0, 1.0))

            ev_combo = payout_mult * joint_final - 1.0
            k_frac = kelly_for_combo(joint_final, payout_mult, fractional_kelly)

            if daily_lock or weekly_lock:
                k_frac = 0.0

            stake = round(bankroll * k_frac, 2)
            decision = combo_decision(ev_combo)

            joint_result = {
                "corr": corr,
                "joint_analytic": joint_analytic,
                "joint_mc": joint_mc,
                "joint_final": joint_final,
                "ev_combo": ev_combo,
                "stake": stake,
                "k_frac": k_frac,
            }

            st.markdown("### 🎯 **2-Pick Combo Result (Defense-Adjusted + Monte Carlo)**")
            st.markdown(f"- Correlation (context-aware): **{corr:+.2f}**")
            st.markdown(f"- Joint Prob Analytic: **{joint_analytic*100:.1f}%**")
            st.markdown(f"- Joint Prob Monte Carlo (10k sims): **{joint_mc*100:.1f}%**")
            st.markdown(f"- **Blended Joint Probability:** {joint_final*100:.1f}%")
            st.markdown(f"- EV per $1: **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Suggested Stake (fractional Kelly, risk-capped): **${stake:.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")

        for leg in [leg1, leg2]:
            if leg:
                mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
                update_market_library(leg["player"], leg["market"], leg["line"])
                if mean_b:
                    st.caption(
                        f"📊 Market Baseline for {leg['player']} {leg['market']}: "
                        f"mean={mean_b:.1f}, median={med_b:.1f}"
                    )

        if joint_result and leg1 and leg2:
            st.markdown("---")
            st.subheader("📌 Post-Run Logging")

            placed = st.radio("Did you place this 2-pick bet?", ["No", "Yes"], index=0)
            stake_actual = st.number_input(
                "If yes, what stake did you use? ($)",
                min_value=0.0,
                max_value=10000.0,
                value=joint_result["stake"],
                step=1.0,
            )
            clv_input = st.number_input(
                "Optional: CLV (closing line - entry line)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1
            )

            if placed == "Yes" and st.button("Log This Bet to History", key="log_post_run"):
                ensure_history()
                df_hist = load_history()

                new_row = {
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Player": f"{leg1['player']} + {leg2['player']}",
                    "Market": "Combo",
                    "Line": f"{leg1['line']} / {leg2['line']}",
                    "EV": joint_result["ev_combo"] * 100.0,
                    "Stake": stake_actual,
                    "Result": "Pending",
                    "CLV": clv_input,
                    "KellyFrac": joint_result["k_frac"],
                    "JointProb": joint_result["joint_final"],
                    "PayoutMult": payout_mult,
                }

                df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
                save_history(df_hist)
                st.success("Bet logged to history ✅")

    st.markdown("---")
    st.subheader("📡 Live Edge Scanner (EV > 65% filter, inside existing tab)")

    st.caption(
        "This scanner uses the same projection engine + Monte Carlo to evaluate a small list of players. "
        "It flags only very strong edges (approx win prob ≥ 65% and positive EV vs market). "
        "To respect Odds API limits, the scanner is intentionally lightweight."
    )

    scanner_players = st.text_area(
        "Enter players (one per line) as: Player Name | Market | Opp (abbr)",
        value="""Nikola Jokic | PRA | LAL
    Anthony Davis | Rebounds | DEN""",
)


    run_scan = st.button("Run Edge Scanner 🔍")

    if run_scan:
        lines = [x.strip() for x in scanner_players.splitlines() if x.strip()]
        if not lines:
            st.info("Add at least one player line above.")
        else:
            records = []
            hist_df = load_history()
            for row in lines:
                try:
                    parts = [p.strip() for p in row.split("|")]
                    if len(parts) != 3:
                        continue
                    name, mkt, opp = parts
                    if mkt not in MARKET_OPTIONS:
                        continue
                    live_line, msg = fetch_live_line_from_odds_api(name, mkt)
                    if live_line is None:
                        continue
                    leg, err = compute_leg_projection(
                        name, mkt, live_line, opp, False, False, games_lookback, hist_df
                    )
                    if err or not leg:
                        continue
                    imp = 1.0 / payout_mult
                    p = leg["prob_over_final"]
                    ev = payout_mult * p - 1.0
                    if (p >= 0.65) and (ev > 0):
                        records.append({
                            "Player": name,
                            "Market": mkt,
                            "Opp": opp,
                            "Line": live_line,
                            "ModelProb%": p * 100,
                            "ImpliedProb%": imp * 100,
                            "EV%": ev * 100,
                        })
                except Exception:
                    continue

            if not records:
                st.info("No edges above 65% win probability with positive EV were found in this quick scan.")
            else:
                df_edges = pd.DataFrame(records)
                st.success(f"Found {len(df_edges)} strong edges (live scanner).")
                st.dataframe(df_edges, use_container_width=True)

# =========================================================
# PART 6 — RESULTS TAB
# =========================================================

with tab_results:
    st.subheader("Results & Performance Tracking")

    df = load_history()

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No bets logged yet. Log entries after you place bets or via the form.")

    with st.form("log_result_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            r_player = st.text_input("Player / Entry Name (or Combo)")
        with c2:
            r_market = st.selectbox(
                "Market",
                ["PRA", "Points", "Rebounds", "Assists", "Combo"]
            )
        with c3:
            r_line = st.text_input(
                "Line (or combo lines)",
                value="25.5"
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
                "CLV (Closing - Entry)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1
            )

        r_joint = st.number_input(
            "Joint Probability (if known, %)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )

        r_result = st.selectbox(
            "Result",
            ["Pending", "Hit", "Miss", "Push"]
        )

        submit_res = st.form_submit_button("Log / Update Result")

        if submit_res:
            ensure_history()
            df = load_history()

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
                "JointProb": r_joint / 100.0 if r_joint > 0 else None,
                "PayoutMult": payout_mult,
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_history(df)

            st.success("Result logged ✅")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if not comp.empty:
        pnl = compute_pnl(comp)

        hits = (comp["Result"] == "Hit").sum()
        total = len(comp)
        hit_rate = (hits / total * 100) if total > 0 else 0.0
        roi = pnl.sum() / max(bankroll, 1.0) * 100
        clv_avg = comp["CLV"].mean() if "CLV" in comp.columns else 0.0
        pnl_var = pnl.var() if len(pnl) > 1 else 0.0

        st.markdown(
            f"**Completed Bets:** {total}  |  "
            f"**Hit Rate:** {hit_rate:.1f}%  |  "
            f"**ROI (vs bankroll):** {roi:+.1f}%  |  "
            f"**Avg CLV:** {clv_avg:+.2f}  |  "
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

# =========================================================
# PART 7 — HISTORY TAB
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

        filt = df.copy()
        filt = filt[pd.to_numeric(filt["EV"], errors="coerce") >= min_ev]

        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        st.markdown(f"**Filtered Bets:** {len(filt)}")
        st.dataframe(filt, use_container_width=True)

        if not filt.empty:
            pnl_filt = compute_pnl(filt)
            filt = filt.copy()
            filt["Net"] = pnl_filt.values
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
# PART 8 — CALIBRATION TAB
# =========================================================

with tab_calib:
    st.subheader("Calibration, EV Buckets & Edge Sustainability")

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
            comp["is_hit"] = (comp["Result"] == "Hit").astype(int)
            comp["PredProb"] = (0.5 + comp["EV_float"]).clip(0.01, 0.99)

            ev_bins = pd.cut(
                comp["EV_float"] * 100,
                bins=[-100, 0, 10, 20, 1000],
                labels=["≤0%", "0–10%", "10–20%", ">20%"]
            )
            comp["EV_Bin"] = ev_bins

            bucket_stats = (
                comp
                .groupby("EV_Bin")
                .agg(
                    Count=("is_hit", "size"),
                    PredProb=("PredProb", "mean"),
                    ActualHit=("is_hit", "mean"),
                    AvgEV=("EV_float", "mean")
                )
                .reset_index()
            )

            bucket_stats["CalibrationGap%"] = (bucket_stats["PredProb"] - bucket_stats["ActualHit"]) * 100

            st.markdown("### 📊 EV Buckets — Predicted vs Actual Hit Rates")
            st.dataframe(bucket_stats, use_container_width=True)

            fig2 = px.histogram(
                comp,
                x=comp["EV_float"] * 100,
                nbins=20,
                title="Distribution of Model Edge vs Market (EV%)",
            )
            st.plotly_chart(fig2, use_container_width=True)

            pnl = compute_pnl(comp)
            roi = pnl.sum() / max(1.0, bankroll) * 100

            st.markdown(
                f"**Overall Predicted Avg Win Prob:** {comp['PredProb'].mean()*100:.1f}%  |  "
                f"**Actual Hit Rate:** {comp['is_hit'].mean()*100:.1f}%  |  "
                f"**ROI:** {roi:+.1f}%"
            )

            prob_scale, vol_scale = compute_model_drift(comp)
            st.markdown(
                f"**Current drift adjustments:** "
                f"ProbScale={prob_scale:.3f}, VolScale={vol_scale:.3f}"
            )

            st.markdown("---")
            st.subheader("Edge Sustainability & Playbook Notes")

            by_market = (
                comp
                .groupby("Market")
                .agg(
                    Count=("is_hit", "size"),
                    HitRate=("is_hit", "mean"),
                    AvgEV=("EV_float", "mean"),
                    AvgCLV=("CLV", "mean")
                )
                .reset_index()
            )
            st.markdown("**Performance by Market Type**")
            st.dataframe(by_market, use_container_width=True)

            st.caption(
                "Use this to spot where your edges come from (e.g., rebounds vs assists). "
                "If CLV and hit rate both slip over time, consider tightening EV thresholds "
                "or focusing on your strongest buckets."
            )

# =========================================================
# PART 9 — FOOTER
# =========================================================

st.markdown(
    """
    <footer style='text-align:center; margin-top:30px; color:#FFCC33; font-size:11px;'>
        © 2025 NBA Prop Quant Engine • Powered by Kamal
    </footer>
    """,
    unsafe_allow_html=True,
)
