# =========================================================
#  NBA PROP BETTING QUANT ENGINE — SINGLE FILE APP
#  Version 1.0 — Empirical Bootstrap + Defensive Context
# =========================================================

import os
import json
import time
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from scipy.stats import norm
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

# Paths
TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

LOG_FILE = os.path.join(TEMP_DIR, "bet_history.csv")
CALIBRATION_FILE = os.path.join(TEMP_DIR, "calibration_state.json")
MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

# Colors
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
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🏀 NBA Prop Quant Engine</p>', unsafe_allow_html=True)

# =========================================================
#  CONSTANTS & MARKET DEFINITIONS
# =========================================================

MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists"]

MARKET_METRICS = {
    "PRA": ["PTS", "REB", "AST"],
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
}

MAX_KELLY_PCT = 0.03  # 3% bankroll cap per play
N_MONTE_CARLO = 10_000

# =========================================================
#  SIDEBAR — USER SETTINGS & BANKROLL
# =========================================================

st.sidebar.header("User & Bankroll")

user_id = st.sidebar.text_input("Your ID (for personal history)", value="Me").strip() or "Me"
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0)
payout_mult = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Recent Games Sample (N)", 5, 20, 10)
max_daily_loss_pct = st.sidebar.slider("Max Daily Loss % (of bankroll)", 1.0, 50.0, 15.0, 1.0)
max_weekly_loss_pct = st.sidebar.slider("Max Weekly Loss % (of bankroll)", 5.0, 80.0, 30.0, 1.0)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)

st.sidebar.caption("Model auto-pulls NBA stats & PrizePicks lines when available. You only enter overrides.")

# =========================================================
#  HELPERS — SEASON, PLAYERS, TEAM CONTEXT
# =========================================================

def current_season() -> str:
    """
    Auto-detect current NBA season based on today's date.
    Example: for November 2025 → '2025-26'
    """
    today = datetime.now()
    if today.month >= 10:  # October or later → new season starts
        start_year = today.year
    else:
        start_year = today.year - 1
    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


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
    """Resolve fuzzy player input → (player_id, full_name)."""
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
    from difflib import get_close_matches
    best = get_close_matches(target, names, n=1, cutoff=0.7)
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
    Pull basic pace + defensive context from LeagueDashTeamStats.
    Used for:
      - defensive matchup multiplier
      - blowout risk approximation
      - pace-adjusted minutes.
    """
    try:
        season = current_season()
        base = LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        adv = LeagueDashTeamStats(
            season=season,
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID", "TEAM_ABBREVIATION", "PACE", "OFF_RATING", "DEF_RATING"
        ]]

        df = base.merge(adv, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="left")

        league_avg = {
            "PACE": df["PACE"].mean(),
            "OFF_RATING": df["OFF_RATING"].mean(),
            "DEF_RATING": df["DEF_RATING"].mean(),
        }

        ctx = {
            r["TEAM_ABBREVIATION"]: {
                "PACE": float(r["PACE"]),
                "OFF_RATING": float(r["OFF_RATING"]),
                "DEF_RATING": float(r["DEF_RATING"]),
                "NET_RATING": float(r["OFF_RATING"] - r["DEF_RATING"]),
            }
            for _, r in df.iterrows()
        }
        return ctx, league_avg
    except Exception:
        return {}, {}

TEAM_CTX, LEAGUE_CTX = get_team_context()

def defensive_multiplier(opp_abbrev: str | None, market: str) -> float:
    """Defense-aware multiplier applied to every projection & Monte Carlo sample."""
    if not opp_abbrev or opp_abbrev not in TEAM_CTX or not LEAGUE_CTX:
        return 1.0

    opp = TEAM_CTX[opp_abbrev]
    pace_rel = opp["PACE"] / LEAGUE_CTX["PACE"] if LEAGUE_CTX["PACE"] else 1.0
    def_rel = LEAGUE_CTX["DEF_RATING"] / opp["DEF_RATING"] if opp["DEF_RATING"] else 1.0

    # Market-specific defensive sensitivity
    if market == "Points":
        w_pace, w_def = 0.4, 0.6
    elif market == "Rebounds":
        w_pace, w_def = 0.6, 0.4
    elif market == "Assists":
        w_pace, w_def = 0.5, 0.5
    else:  # PRA
        w_pace, w_def = 0.5, 0.5

    mult = w_pace * pace_rel + w_def * def_rel
    return float(np.clip(mult, 0.80, 1.25))


def estimate_blowout_risk(team_abbrev: str | None, opp_abbrev: str | None) -> float:
    """
    Approximate blowout risk using net rating difference.
    Returns probability between 0 and 1.
    """
    if not team_abbrev or not opp_abbrev:
        return 0.10  # mild baseline
    if team_abbrev not in TEAM_CTX or opp_abbrev not in TEAM_CTX:
        return 0.10

    t = TEAM_CTX[team_abbrev]
    o = TEAM_CTX[opp_abbrev]
    diff = abs(t["NET_RATING"] - o["NET_RATING"])

    # 0 → 0.05, 10+ → 0.40
    risk = 0.05 + (diff / 10.0) * 0.35
    return float(np.clip(risk, 0.05, 0.40))


# =========================================================
#  PRIZEPICKS LINE FETCHER (Robust w/ Fallback)
# =========================================================

@st.cache_data(show_spinner=False, ttl=300)
def fetch_prizepicks_raw():
    """
    Fetch raw PrizePicks projection JSON.
    If anything fails, gracefully return empty list.
    """
    url = "https://api.prizepicks.com/projections"
    params = {"league_id": 7}  # 7 = NBA in many known integrations
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("data", [])
    except Exception:
        return []


def normalize_market_for_prizepicks(market: str):
    mapping = {
        "Points": "Points",
        "Rebounds": "Rebounds",
        "Assists": "Assists",
        "PRA": "Pts+Rebs+Asts",
    }
    return mapping.get(market)


@st.cache_data(show_spinner=False, ttl=300)
def get_prizepicks_line(player_name: str, market: str):
    """
    Attempt to auto-fetch PrizePicks line for the given player & market.
    Returns float or None.
    """
    target_market = normalize_market_for_prizepicks(market)
    if not target_market:
        return None

    raw = fetch_prizepicks_raw()
    if not raw:
        return None

    norm_target = _norm_name(player_name)

    for item in raw:
        try:
            attr = item.get("attributes", {})
            line_score = float(attr.get("line_score", 0.0))
            stat_type = attr.get("stat_type", "")
            proj_player = attr.get("description", "")
        except Exception:
            continue

        if not proj_player or not stat_type:
            continue

        if _norm_name(proj_player) == norm_target and stat_type == target_market:
            return line_score

    return None


# =========================================================
#  GAME LOGS, USAGE & EMPIRICAL BOOTSTRAP ENGINE
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_game_logs(player_name: str):
    """
    Return full game log DataFrame for the current season.
    """
    pid, label = resolve_player(player_name)
    if not pid:
        return None, f"No match for '{player_name}'."

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception as e:
        return None, f"Game log error: {e}"

    if gl.empty:
        return None, "No game logs found for current season."

    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False)
    return gl, f"{label}: {len(gl)} games this season"


def compute_usage_and_minutes(gl: pd.DataFrame, n_games: int, market: str):
    """
    Rolling weighted minutes & per-minute stat rate from last N games.
    """
    recent = gl.head(n_games).copy()
    cols = MARKET_METRICS[market]

    minutes = []
    totals = []
    for _, r in recent.iterrows():
        m_raw = r.get("MIN", "0")
        try:
            if isinstance(m_raw, str) and ":" in m_raw:
                mm, ss = m_raw.split(":")
                m_val = float(mm) + float(ss)/60.0
            else:
                m_val = float(m_raw)
        except Exception:
            m_val = 0.0
        if m_val <= 0:
            continue
        minutes.append(m_val)
        totals.append(sum(float(r.get(c, 0)) for c in cols))

    if not minutes:
        return None, None, None

    minutes = np.array(minutes)
    totals = np.array(totals)

    weights = np.linspace(1.0, 2.0, len(minutes))
    weights /= weights.sum()

    avg_min = float(np.average(minutes, weights=weights))
    per_min = totals / minutes
    mu_per_min = float(np.average(per_min, weights=weights))
    sd_per_min = float(max(np.std(per_min, ddof=1), 0.15 * max(mu_per_min, 0.5)))

    return mu_per_min, sd_per_min, avg_min


def build_empirical_samples(gl: pd.DataFrame, n_games: int, market: str, opp_abbrev: str | None):
    """
    Build empirical samples from last N games, defense + pace adjusted.
    Each sample is one game's stat outcome, adjusted by matchup.
    """
    recent = gl.head(n_games).copy()
    cols = MARKET_METRICS[market]

    try:
        team_abbrev = recent["TEAM_ABBREVIATION"].mode().iloc[0]
    except Exception:
        team_abbrev = None

    ctx_mult = defensive_multiplier(opp_abbrev, market)

    samples = []
    minutes = []
    for _, r in recent.iterrows():
        m_raw = r.get("MIN", "0")
        try:
            if isinstance(m_raw, str) and ":" in m_raw:
                mm, ss = m_raw.split(":")
                m_val = float(mm) + float(ss)/60.0
            else:
                m_val = float(m_raw)
        except Exception:
            m_val = 0.0

        if m_val <= 0:
            continue

        total_val = sum(float(r.get(c, 0)) for c in cols)

        adj_val = total_val * ctx_mult
        samples.append(adj_val)
        minutes.append(m_val)

    if not samples:
        return None, None, None, team_abbrev, ctx_mult

    samples = np.array(samples, dtype=float)
    minutes = np.array(minutes, dtype=float)

    return samples, minutes, team_abbrev, ctx_mult


def estimate_role_boost(minutes: np.ndarray):
    """
    Crude proxy for 'usage / on-off boost':
    if recent minutes are significantly above earlier season average,
    treat as increased role.
    """
    if len(minutes) < 3:
        return 1.0

    recent_avg = minutes[:min(5, len(minutes))].mean()
    long_avg = minutes.mean()

    if long_avg <= 0:
        return 1.0

    ratio = recent_avg / long_avg
    return float(np.clip(ratio, 0.90, 1.15))


def monte_carlo_single_leg(player: str, market: str, line: float, opp: str | None, n_games: int):
    """
    Empirical bootstrap Monte Carlo:
      - draws from last N game outcomes
      - each outcome is defense-adjusted
      - usage / role multiplier applied
      - blowout risk affects tail
    """
    gl, msg = get_player_game_logs(player)
    if gl is None:
        return None, msg

    opp_abbrev = opp.strip().upper() if opp else None
    samples, minutes, team_abbrev, ctx_mult = build_empirical_samples(gl, n_games, market, opp_abbrev)
    if samples is None:
        return None, msg

    role_mult = estimate_role_boost(minutes)
    blowout_prob = estimate_blowout_risk(team_abbrev, opp_abbrev)

    eff_samples = samples * role_mult

    if len(eff_samples) == 0:
        return None, "Not enough valid game samples."

    draws = np.random.choice(eff_samples, size=N_MONTE_CARLO, replace=True)

    if blowout_prob > 0:
        mask = np.random.rand(N_MONTE_CARLO) < blowout_prob
        draws[mask] *= 0.90  # 10% trim on blowout tail

    mu = float(np.mean(draws))
    sd = float(np.std(draws, ddof=1))
    p_over = float(np.mean(draws > line))
    p_over = float(np.clip(p_over, 0.01, 0.99))

    return {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": mu,
        "sd": sd,
        "prob_over": p_over,
        "team": team_abbrev,
        "ctx_mult": ctx_mult,
        "role_mult": role_mult,
        "blowout_prob": blowout_prob,
        "msg": msg,
        "samples": eff_samples.tolist(),
    }, None


# =========================================================
#  CORRELATION & JOINT MONTE CARLO ENGINE
# =========================================================

def estimate_player_correlation(leg1: dict, leg2: dict) -> float:
    """
    Context-aware synthetic correlation between two legs.
    Uses:
      - same team
      - similar markets
      - defensive context alignment
      - role multipliers
    """
    corr = 0.0

    if leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"]:
        corr += 0.18

    m1, m2 = leg1["market"], leg2["market"]
    if m1 == m2:
        corr += 0.06
    if (m1 == "Points" and m2 == "Assists") or (m1 == "Assists" and m2 == "Points"):
        corr -= 0.08
    if (m1 == "Points" and m2 == "Rebounds") or (m1 == "Rebounds" and m2 == "Points"):
        corr -= 0.05
    if m1 == "PRA" or m2 == "PRA":
        corr += 0.03

    ctx1, ctx2 = leg1.get("ctx_mult", 1.0), leg2.get("ctx_mult", 1.0)
    if ctx1 > 1.03 and ctx2 > 1.03:
        corr += 0.04
    if ctx1 < 0.97 and ctx2 < 0.97:
        corr += 0.03

    r1, r2 = leg1.get("role_mult", 1.0), leg2.get("role_mult", 1.0)
    if r1 > 1.05 and r2 > 1.05:
        corr += 0.03

    return float(np.clip(corr, -0.30, 0.45))


def joint_probability_via_copula(p1: float, p2: float, rho: float, n_sims: int = N_MONTE_CARLO):
    """
    Correlated Bernoulli via Gaussian copula.
    Returns Monte Carlo estimate of P(both hit).
    """
    p1 = float(np.clip(p1, 0.01, 0.99))
    p2 = float(np.clip(p2, 0.01, 0.99))
    rho = float(np.clip(rho, -0.95, 0.95))

    z1 = np.random.randn(n_sims)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn(n_sims)

    u1 = norm.cdf(z1)
    u2 = norm.cdf(z2)

    hits1 = u1 < p1
    hits2 = u2 < p2

    joint = float(np.mean(hits1 & hits2))
    return joint


# =========================================================
#  KELLY, RISK LIMITS, & MARKET BASELINES
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float) -> float:
    """
    Kelly criterion for 2-pick entry.
    """
    b = payout_mult - 1.0
    q = 1.0 - p_joint
    raw = (b * p_joint - q) / b
    k = raw * frac
    return float(np.clip(k, 0.0, MAX_KELLY_PCT))


def ensure_history():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Date", "Player", "Market", "Line", "EV",
            "Stake", "Result", "CLV", "KellyFrac", "EntryType"
        ])
        df.to_csv(LOG_FILE, index=False)


def load_history():
    ensure_history()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Date", "Player", "Market", "Line", "EV",
            "Stake", "Result", "CLV", "KellyFrac", "EntryType"
        ])


def save_history(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)


def compute_recent_pnl(df: pd.DataFrame):
    """
    Compute daily & weekly PnL to enforce loss limits.
    """
    if df.empty:
        return 0.0, 0.0

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    completed = df[df["Result"].isin(["Hit", "Miss"])]

    if completed.empty:
        return 0.0, 0.0

    now = datetime.now()
    today_mask = completed["Date"].dt.date == now.date()
    week_mask = completed["Date"] >= (now - timedelta(days=7))

    def pnl_sub(sub):
        if sub.empty:
            return 0.0
        def _p(r):
            if r["Result"] == "Hit":
                return r["Stake"] * (payout_mult - 1.0)
            elif r["Result"] == "Miss":
                return -r["Stake"]
            else:
                return 0.0
        return float(sub.apply(_p, axis=1).sum())

    daily = pnl_sub(completed[today_mask])
    weekly = pnl_sub(completed[week_mask])
    return daily, weekly


def risk_adjusted_kelly(k_frac: float, bankroll: float, df_history: pd.DataFrame):
    """
    Reduce Kelly fraction if recent losses exceed thresholds.
    """
    daily_pnl, weekly_pnl = compute_recent_pnl(df_history)

    daily_loss_pct = (-daily_pnl / bankroll * 100) if daily_pnl < 0 else 0.0
    weekly_loss_pct = (-weekly_pnl / bankroll * 100) if weekly_pnl < 0 else 0.0

    scale = 1.0
    if daily_loss_pct > max_daily_loss_pct:
        scale *= 0.3
    if weekly_loss_pct > max_weekly_loss_pct:
        scale *= 0.5

    return float(np.clip(k_frac * scale, 0.0, MAX_KELLY_PCT)), daily_pnl, weekly_pnl


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
#  SELF-LEARNING CALIBRATION ENGINE
# =========================================================

def load_calibration_state():
    """
    Calibration file stores small biases & multipliers learned from history.
    """
    if not os.path.exists(CALIBRATION_FILE):
        return {
            "p_bias": 0.0,
            "last_update_month": None,
        }
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "p_bias": 0.0,
            "last_update_month": None,
        }


def save_calibration_state(state: dict):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(state, f)


def apply_calibration(p_over: float, calib_state: dict):
    """
    Adjust probability using learned bias (if any).
    """
    bias = float(calib_state.get("p_bias", 0.0))
    return float(np.clip(p_over - bias, 0.01, 0.99))


def update_calibration_from_history(df: pd.DataFrame):
    """
    Once per month, compare predicted EV vs actual outcomes and learn bias.
    """
    state = load_calibration_state()
    now = datetime.now()
    month_key = now.strftime("%Y-%m")

    if state.get("last_update_month") == month_key:
        return state  # already updated this month

    comp = df[df["Result"].isin(["Hit", "Miss"])].copy()
    if comp.empty:
        return state

    comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
    comp = comp.dropna(subset=["EV_float"])
    if comp.empty:
        return state

    pred_win_prob = 0.5 + comp["EV_float"].mean() / 2.0
    actual_win_prob = (comp["Result"] == "Hit").mean()

    bias = pred_win_prob - actual_win_prob  # positive means overconfident
    state["p_bias"] = float(np.clip(bias * 0.5, -0.10, 0.10))
    state["last_update_month"] = month_key
    save_calibration_state(state)
    return state


# =========================================================
#  UI HELPERS
# =========================================================

def run_loader():
    msgs = [
        "Pulling player logs…",
        "Analyzing matchup context…",
        "Adjusting for pace & usage…",
        "Running Monte Carlo simulations…",
        "Finalizing edge & stake…",
    ]
    ph = st.empty()
    for m in msgs:
        ph.markdown(
            f"<p style='color:{GOLD};font-size:20px;font-weight:600;'>{m}</p>",
            unsafe_allow_html=True
        )
        time.sleep(0.35)
    ph.empty()


def combo_decision(ev_combo: float) -> str:
    if ev_combo >= 0.10:
        return "🔥 **PLAY — Strong Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **Lean — Thin Edge**"
    else:
        return "❌ **Pass — No Edge**"


def render_leg_card(leg: dict, container, compact=False):
    """
    Render a player card including opponent & matchup context.
    """
    player = leg["player"]
    market = leg["market"]
    line = leg["line"]
    mu = leg["mu"]
    sd = leg["sd"]
    p = leg["prob_over"]
    ctx = leg.get("ctx_mult", 1.0)
    role = leg.get("role_mult", 1.0)
    blow_risk = leg.get("blowout_prob", 0.0)
    opp = leg.get("opp")
    msg = leg.get("msg", "")

    headshot = get_headshot_url(player)

    with container:
        st.markdown(
            f"""
            <div class="card">
                <h3 style="margin-top:0;color:{GOLD};">{player} — {market}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not compact and headshot:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(headshot, width=120)
            main_col = cols[1]
        else:
            main_col = st

        with main_col:
            st.write(f"📌 **Line:** {line}")
            if opp:
                st.write(f"🆚 **Opponent:** {opp}")
            st.write(f"📊 **Model Mean:** {mu:.2f}")
            st.write(f"📉 **Model SD:** {sd:.2f}")
            st.write(f"🛡️ **Defense/Pace Multiplier:** {ctx:.3f}")
            st.write(f"📈 **Role / Usage Multiplier:** {role:.3f}")
            st.write(f"🎯 **Model Probability Over:** {p*100:.1f}%")
            st.write(f"⚠️ **Approx. Blowout Risk:** {blow_risk*100:.1f}%")
            st.caption(f"📝 {msg}")


# =========================================================
#  APP TABS
# =========================================================

tab_model, tab_results, tab_history, tab_calib = st.tabs(
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration"]
)

# ---------------------------------------------------------
#  MODEL TAB
# ---------------------------------------------------------
with tab_model:
    st.subheader("2-Pick Projection & Edge (Empirical Monte Carlo)")

    c1, c2 = st.columns(2)

    # LEFT LEG INPUT
    with c1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="p1_market")
        auto_line1 = get_prizepicks_line(p1, m1) if p1 else None
        override1 = st.checkbox("Override P1 Line Manually", value=False)
        if auto_line1 is not None and not override1:
            st.success(f"Auto PrizePicks Line: {auto_line1}")
            l1 = st.number_input("P1 Line", min_value=0.0, value=float(auto_line1), step=0.5)
        else:
            if auto_line1 is not None:
                st.info(f"Auto line found: {auto_line1}. You may override if desired.")
                default_line = float(auto_line1)
            else:
                st.warning("No PrizePicks line found — please enter manually.")
                default_line = 25.0
            l1 = st.number_input("P1 Line", min_value=0.0, value=default_line, step=0.5)
        o1 = st.text_input("P1 Opponent (abbr)", help="Example: BOS, DEN")

    # RIGHT LEG INPUT
    with c2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="p2_market")
        auto_line2 = get_prizepicks_line(p2, m2) if p2 else None
        override2 = st.checkbox("Override P2 Line Manually", value=False)
        if auto_line2 is not None and not override2:
            st.success(f"Auto PrizePicks Line: {auto_line2}")
            l2 = st.number_input("P2 Line", min_value=0.0, value=float(auto_line2), step=0.5)
        else:
            if auto_line2 is not None:
                st.info(f"Auto line found: {auto_line2}. You may override if desired.")
                default_line2 = float(auto_line2)
            else:
                st.warning("No PrizePicks line found — please enter manually.")
                default_line2 = 25.0
            l2 = st.number_input("P2 Line", min_value=0.0, value=default_line2, step=0.5)
        o2 = st.text_input("P2 Opponent (abbr)", help="Example: BOS, DEN")

    run = st.button("Run Model ⚡")

    leg1 = None
    leg2 = None

    if run:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

        leg1, err1 = (
            monte_carlo_single_leg(p1, m1, l1, o1, games_lookback)
            if p1 and l1 > 0 else (None, None)
        )
        leg2, err2 = (
            monte_carlo_single_leg(p2, m2, l2, o2, games_lookback)
            if p2 and l2 > 0 else (None, None)
        )

        if err1:
            st.error(f"P1: {err1}")
        if err2:
            st.error(f"P2: {err2}")

        if leg1:
            leg1["opp"] = o1.strip().upper() if o1 else None
        if leg2:
            leg2["opp"] = o2.strip().upper() if o2 else None

        colL, colR = st.columns(2)
        if leg1:
            render_leg_card(leg1, colL, compact_mode)
        if leg2:
            render_leg_card(leg2, colR, compact_mode)

        st.markdown("---")
        st.subheader("📈 Market vs Model Probability Check")

        def implied_prob_from_payout(mult):
            return 1.0 / mult

        imp_prob = implied_prob_from_payout(payout_mult)
        st.markdown(f"**Market Implied Probability (per leg, even payout):** {imp_prob*100:.1f}%")

        for leg in [leg1, leg2]:
            if leg:
                st.markdown(
                    f"**{leg['player']} {leg['market']}** — "
                    f"Model: {leg['prob_over']*100:.1f}% | "
                    f"Edge vs {imp_prob*100:.1f}%: {(leg['prob_over']-imp_prob)*100:+.1f}%"
                )

        if leg1 and leg2:
            rho = estimate_player_correlation(leg1, leg2)
            p_joint_raw = joint_probability_via_copula(leg1["prob_over"], leg2["prob_over"], rho, N_MONTE_CARLO)

            # Apply calibration
            hist_df_for_cal = load_history()
            calib_state = update_calibration_from_history(hist_df_for_cal)
            p1_cal = apply_calibration(leg1["prob_over"], calib_state)
            p2_cal = apply_calibration(leg2["prob_over"], calib_state)
            p_joint = joint_probability_via_copula(p1_cal, p2_cal, rho, N_MONTE_CARLO)

            ev_combo = payout_mult * p_joint - 1.0
            base_kelly = kelly_for_combo(p_joint, payout_mult, fractional_kelly)

            hist_df = load_history()
            adj_kelly, daily_pnl, weekly_pnl = risk_adjusted_kelly(base_kelly, bankroll, hist_df)
            stake = round(bankroll * adj_kelly, 2)
            decision = combo_decision(ev_combo)

            st.markdown("### 🎯 2-Pick Combo Result")
            st.markdown(f"- Estimated Correlation: **{rho:+.2f}**")
            st.markdown(f"- Joint Probability (uncalibrated): **{p_joint_raw*100:.1f}%**")
            st.markdown(f"- Joint Probability (calibrated): **{p_joint*100:.1f}%**")
            st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Raw Kelly Fraction: **{base_kelly*100:.2f}%**")
            st.markdown(f"- Risk-Adjusted Kelly Fraction: **{adj_kelly*100:.2f}%**")
            st.markdown(f"- Suggested Stake: **${stake:.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")

            if daily_pnl < 0:
                st.caption(f"Today PnL: {daily_pnl:+.2f}")
            if weekly_pnl < 0:
                st.caption(f"Last 7 days PnL: {weekly_pnl:+.2f}")

            # Store market baselines silently
            update_market_library(leg1["player"], leg1["market"], leg1["line"])
            update_market_library(leg2["player"], leg2["market"], leg2["line"])

            for leg in [leg1, leg2]:
                mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
                if mean_b is not None:
                    st.caption(
                        f"📊 Baseline for {leg['player']} {leg['market']}: "
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
            r_line = st.number_input("Line", min_value=0.0, max_value=200.0, value=25.0, step=0.5)

        c4, c5, c6 = st.columns(3)
        with c4:
            r_ev = st.number_input("Model EV (%)", min_value=-50.0, max_value=200.0, value=5.0, step=0.1)
        with c5:
            r_stake = st.number_input("Stake ($)", min_value=0.0, max_value=10000.0, value=5.0, step=0.5)
        with c6:
            r_clv = st.number_input("CLV (Closing - Entry)", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)

        r_result = st.selectbox("Result", ["Pending", "Hit", "Miss", "Push"])
        r_type = st.selectbox("Entry Type", ["2-Pick", "Single", "Other"], index=0)

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
                "EntryType": r_type,
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_history(df)
            st.success("Result logged ✅")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if not comp.empty:
        def pnl_row(r):
            if r["Result"] == "Hit":
                return r["Stake"] * (payout_mult - 1.0)
            elif r["Result"] == "Miss":
                return -r["Stake"]
            else:
                return 0.0

        pnl = comp.apply(pnl_row, axis=1)
        hits = (comp["Result"] == "Hit").sum()
        total = len(comp)

        hit_rate = (hits / total * 100.0) if total > 0 else 0.0
        roi = pnl.sum() / max(bankroll, 1.0) * 100.0
        avg_clv = comp["CLV"].mean() if "CLV" in comp.columns else 0.0
        var_out = float(np.var(pnl)) if len(pnl) > 1 else 0.0

        st.markdown(
            f"**Completed Bets:** {total} | "
            f"**Hit Rate:** {hit_rate:.1f}% | "
            f"**ROI vs Bankroll:** {roi:+.1f}% | "
            f"**Avg CLV:** {avg_clv:+.2f} | "
            f"**Outcome Variance:** {var_out:.2f}"
        )

        trend = comp.copy()
        trend["Profit"] = pnl.values
        trend["Cumulative"] = trend["Profit"].cumsum()

        fig = px.line(trend, x="Date", y="Cumulative", title="Cumulative Profit (All Logged Bets)", markers=True)
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
        min_ev = st.slider("Min EV (%) filter", min_value=-20.0, max_value=100.0, value=0.0, step=1.0)
        market_filter = st.selectbox(
            "Market filter",
            ["All", "PRA", "Points", "Rebounds", "Assists", "Combo"],
            index=0,
        )

        filt = df[df["EV"] >= min_ev]
        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        st.markdown(f"**Filtered Bets:** {len(filt)}")
        st.dataframe(filt, use_container_width=True)

        if not filt.empty:
            filt = filt.copy()

            def pnl_row2(r):
                if r["Result"] == "Hit":
                    return r["Stake"] * (payout_mult - 1.0)
                elif r["Result"] == "Miss":
                    return -r["Stake"]
                else:
                    return 0.0

            filt["Net"] = filt.apply(pnl_row2, axis=1)
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
    st.subheader("Calibration, EV Buckets & Edge Integrity")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if comp.empty or len(comp) < 25:
        st.info("Log at least 25 completed bets to start calibration.")
    else:
        comp = comp.copy()
        comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
        comp = comp.dropna(subset=["EV_float"])

        if comp.empty:
            st.info("No valid EV values yet.")
        else:
            bins = [-1.0, 0.0, 0.05, 0.10, 0.20, 1.0]
            labels = ["<=0%", "0–5%", "5–10%", "10–20%", ">20%"]
            comp["EV_bucket"] = pd.cut(comp["EV_float"], bins=bins, labels=labels)

            bucket_stats = (
                comp.groupby("EV_bucket")
                .apply(lambda g: pd.Series({
                    "Count": len(g),
                    "HitRate": (g["Result"] == "Hit").mean() * 100.0,
                    "AvgEV": g["EV_float"].mean() * 100.0,
                    "AvgCLV": g["CLV"].mean() if "CLV" in g.columns else 0.0,
                }))
                .reset_index()
            )

            st.markdown("### EV Buckets vs Real Hit Rate")
            st.dataframe(bucket_stats, use_container_width=True)

            fig2 = px.bar(
                bucket_stats,
                x="EV_bucket",
                y="HitRate",
                title="Actual Hit Rate by EV Bucket",
            )
            st.plotly_chart(fig2, use_container_width=True)

            def pnl_row3(r):
                if r["Result"] == "Hit":
                    return r["Stake"] * (payout_mult - 1.0)
                elif r["Result"] == "Miss":
                    return -r["Stake"]
                else:
                    return 0.0

            pnl = comp.apply(pnl_row3, axis=1)
            roi = pnl.sum() / max(1.0, bankroll) * 100.0

            st.markdown(f"**Overall ROI (logged):** {roi:+.1f}%")

            calib_state = update_calibration_from_history(comp)
            st.markdown(
                f"**Current Probability Bias Correction:** {calib_state.get('p_bias', 0.0)*100:+.2f}% "
                f"(positive = model has been overconfident historically)"
            )

            st.caption("Calibration updates at most once per month based on all completed bets.")

st.markdown(
    """
    <footer style='text-align:center; margin-top:30px; color:#FFCC33; font-size:11px;'>
        © 2025 NBA Prop Quant Engine • Powered by Kamal
    </footer>
    """,
    unsafe_allow_html=True,
)
