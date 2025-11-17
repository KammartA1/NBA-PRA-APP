# Let's build the full upgraded app.py, requirements.txt, and README.md,
# write them into /mnt/data/nba_quant_app, and zip them for the user.

import os, textwrap, zipfile, pathlib, math, json, sys

base_dir = "/mnt/data/nba_quant_app"
os.makedirs(base_dir, exist_ok=True)

app_py = r'''# =========================================================
#  NBA PROP BETTING QUANT ENGINE — SINGLE FILE STREAMLIT APP
#  Upgraded Version (Empirical Bootstrap + Defensive Context)
# =========================================================

import os
import time
import random
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm

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

MC_SIMS = 10_000  # Monte Carlo simulations per leg / combo

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

st.markdown('<p class="main-header">🏀 NBA Prop Model</p>', unsafe_allow_html=True)

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
max_daily_loss_pct = st.sidebar.slider("Max Daily Loss (% of bankroll)", 5, 50, 20, 1)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)

st.sidebar.caption("Model auto-pulls NBA stats. You only enter the lines.")

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

# =========================================================
#  PART 2.2 — PLAYER LOOKUP HELPERS
# =========================================================

def current_season() -> str:
    """
    Returns the current NBA season string, e.g. '2025-26'.

    Logic:
    - If today's month is October or later, season = current_year–next_year
    - Else, season = previous_year–current_year

    This keeps the app automatically in sync with each new NBA season.
    """
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
    """Adjust projection using advanced opponent factors (defense + pace + stat-specific)."""
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

    # Blend: pace + defense + stat-specific factor
    mult_stat = reb_adj if market == "Rebounds" else ast_adj
    mult = (0.4 * pace_f) + (0.3 * def_f) + (0.3 * mult_stat)

    return float(np.clip(mult, 0.80, 1.20))

# =========================================================
#  PART 2.4 — MARKET BASELINE LIBRARY
# =========================================================

MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

def load_market_library():
    """Loads market baselines; safe fallback on first run."""
    if not os.path.exists(MARKET_LIBRARY_FILE):
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])
    try:
        return pd.read_csv(MARKET_LIBRARY_FILE)
    except Exception:
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])

def save_market_library(df: pd.DataFrame):
    df.to_csv(MARKET_LIBRARY_FILE, index=False)

def update_market_library(player: str, market: str, line: float):
    """Stores every entered line to build mean/median reference ranges."""
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
    """Returns (mean, median) of historical market lines."""
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
def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    """
    Pulls recent player logs for the current season, computes:
      - per-minute production (mu_per_min)
      - per-minute standard deviation (sd_per_min)
      - average minutes
      - team abbreviation
      - per-game per-minute samples (for empirical bootstrap)
      - per-game minutes samples
    """
    pid, label = resolve_player(name)
    if not pid:
        return None, None, None, None, None, None, f"No match for '{name}'."

    # Try requesting game logs for current season only
    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception as e:
        return None, None, None, None, None, None, f"Game log error: {e}"

    if gl.empty:
        return None, None, None, None, None, None, "No recent game logs found."

    # Sort newest → oldest, take N games
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False).head(n_games)

    cols = MARKET_METRICS[market]
    per_min_vals = []
    minutes_vals = []

    # -----------------------------
    # Compute per-minute values
    # -----------------------------
    for _, r in gl.iterrows():
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
        per_min_vals.append(total_val / m)
        minutes_vals.append(m)

    if not per_min_vals:
        return None, None, None, None, None, None, "Insufficient data."

    per_min_arr = np.array(per_min_vals, dtype=float)
    minutes_arr = np.array(minutes_vals, dtype=float)

    mu_per_min = float(np.mean(per_min_arr))
    avg_min = float(np.mean(minutes_arr))
    sd_per_min = max(
        float(np.std(per_min_arr, ddof=1)),
        0.15 * max(mu_per_min, 0.5)
    )

    # Team abbreviation
    try:
        team = gl["TEAM_ABBREVIATION"].mode().iloc[0]
    except Exception:
        team = None

    msg = f"{label}: {len(per_min_arr)} games • {avg_min:.1f} min"

    # Return lists so cache hashing is stable
    return (
        mu_per_min,
        sd_per_min,
        avg_min,
        team,
        per_min_arr.tolist(),
        minutes_arr.tolist(),
        msg,
    )

# =========================================================
#  PART 3.1 — EMPIRICAL BOOTSTRAP MONTE CARLO ENGINE
# =========================================================

def run_empirical_monte_carlo(
    line: float,
    market: str,
    per_min_samples: list,
    minute_samples: list,
    opp_abbrev: str | None,
    ctx_mult: float,
    teammate_out: bool,
    blowout: bool,
    heavy_tail_factor: float,
    n_sims: int = MC_SIMS,
):
    """
    Empirical bootstrap Monte Carlo:
    - Sample from last N games (per-minute + minutes).
    - Apply defensive context multiplier, pace, usage, and on/off adjustments.
    - Generates full distribution and returns (mean, sd, prob_over).
    """

    if not per_min_samples or not minute_samples:
        return None, None, None

    per_min_arr = np.array(per_min_samples, dtype=float)
    minutes_arr = np.array(minute_samples, dtype=float)

    n_obs = len(per_min_arr)
    if n_obs == 0:
        return None, None, None

    # ---------------------------------
    # Usage & on/off adjustment
    # ---------------------------------
    usage_mult = 1.0
    if teammate_out:
        usage_mult *= 1.08  # on-ball boost when key teammate is out

    # ---------------------------------
    # Pace-adjusted minutes expectation
    # ---------------------------------
    minutes_mult = 1.0
    if opp_abbrev and opp_abbrev in TEAM_CTX and LEAGUE_CTX:
        opp_pace = TEAM_CTX[opp_abbrev]["PACE"]
        league_pace = LEAGUE_CTX["PACE"]
        pace_ratio = opp_pace / league_pace
        # Faster games → slightly more minutes; slower games → slightly fewer
        minutes_mult *= float(np.clip(0.95 + 0.10 * (pace_ratio - 1.0), 0.90, 1.10))

    if blowout:
        minutes_mult *= 0.93  # trim expectation due to blowout risk

    # Final adjusted per-minute + minutes samples
    adj_per_min = per_min_arr * ctx_mult * usage_mult
    adj_minutes = minutes_arr * minutes_mult

    # Extra volatility for heavy markets
    vol_sigma = 0.10 + 0.05 * (heavy_tail_factor - 1.0)

    # ---------------------------------
    # Recency weighting for sampling
    # Newest game gets highest weight.
    # ---------------------------------
    weights = np.linspace(1.0, 0.4, n_obs)
    weights = weights / weights.sum()

    idx = np.random.choice(np.arange(n_obs), size=n_sims, p=weights)
    sampled_rates = adj_per_min[idx]
    sampled_minutes = adj_minutes[idx]

    # Sample lognormal noise for extra volatility
    noise = np.random.lognormal(mean=0.0, sigma=vol_sigma, size=n_sims)

    simulated_vals = sampled_rates * sampled_minutes * noise

    mean_val = float(simulated_vals.mean())
    sd_val = float(simulated_vals.std(ddof=1))
    prob_over = float(np.mean(simulated_vals > float(line)))

    # Clip probability to avoid extremes
    prob_over = float(np.clip(prob_over, 0.01, 0.99))

    return mean_val, sd_val, prob_over

# =========================================================
#  PART 3.2 — ADVANCED PLAYER CORRELATION ENGINE
# =========================================================

def estimate_player_correlation(leg1: dict, leg2: dict) -> float:
    """
    Produces a dynamic, data-driven correlation estimate.

    Factors used:
    - Shared team → strongly increases correlation
    - Minutes expectations
    - Market interactions (Points vs Assists vs Rebounds vs PRA)
    - Opponent context → shared boost / suppression
    """
    corr = 0.0

    # 1. Same-team baseline
    if leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"]:
        corr += 0.18

    # 2. Minutes dependency (rough)
    m1 = leg1.get("avg_min", 30.0)
    m2 = leg2.get("avg_min", 30.0)
    if m1 > 32 and m2 > 32:
        corr += 0.05
    elif m1 < 22 or m2 < 22:
        corr -= 0.04

    # 3. Market-type interactions
    mkt1, mkt2 = leg1["market"], leg2["market"]

    if mkt1 == "Points" and mkt2 == "Points":
        corr += 0.08

    if (mkt1 == "Points" and mkt2 == "Assists") or (mkt1 == "Assists" and mkt2 == "Points"):
        corr -= 0.10

    if (mkt1 == "Rebounds" and mkt2 == "Points") or (mkt1 == "Points" and mkt2 == "Rebounds"):
        corr -= 0.06

    if mkt1 == "PRA" or mkt2 == "PRA":
        corr += 0.03

    # 4. Opponent-defense adjustment
    ctx1, ctx2 = leg1.get("ctx_mult", 1.0), leg2.get("ctx_mult", 1.0)

    if ctx1 > 1.03 and ctx2 > 1.03:
        corr += 0.04
    if ctx1 < 0.97 and ctx2 < 0.97:
        corr += 0.03
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 1.03):
        corr -= 0.05

    # Final clamp for stability
    corr = float(np.clip(corr, -0.25, 0.40))
    return corr

# =========================================================
#  PART 3.3 — SINGLE-LEG PROJECTION ENGINE
# =========================================================

def compute_leg_projection(player, market, line, opp, teammate_out, blowout, n_games):
    """
    Core projection engine for a single leg.

    Uses:
      - recent per-minute production (last N games)
      - opponent pace/defense context (defensive matchup engine)
      - usage rate & on/off boost
      - pace-adjusted minutes expectation
      - empirical bootstrap Monte Carlo distribution

    Returns (leg_dict, error_message).
    """
    # --------------------------------------------------------
    # 1. Player rates from game logs
    # --------------------------------------------------------
    (
        mu_min,
        sd_min,
        avg_min,
        team,
        per_min_samples,
        minute_samples,
        msg,
    ) = get_player_rate_and_minutes(player, n_games, market)

    if mu_min is None:
        return None, msg

    opp_abbrev = opp.strip().upper() if opp else None

    # --------------------------------------------------------
    # 2. Context multipliers (defense + pace)
    # --------------------------------------------------------
    ctx_mult = get_context_multiplier(opp_abbrev, market)
    heavy = HEAVY_TAIL[market]

    # Baseline minutes expectation
    minutes = avg_min
    if teammate_out:
        minutes *= 1.05
        mu_min *= 1.06
    if blowout:
        minutes *= 0.90

    # --------------------------------------------------------
    # 3. Empirical Monte Carlo to get mean, sd, and probability
    # --------------------------------------------------------
    mc_mean, mc_sd, mc_prob = run_empirical_monte_carlo(
        line=line,
        market=market,
        per_min_samples=per_min_samples,
        minute_samples=minute_samples,
        opp_abbrev=opp_abbrev,
        ctx_mult=ctx_mult,
        teammate_out=teammate_out,
        blowout=blowout,
        heavy_tail_factor=heavy,
    )

    if mc_mean is None or mc_sd is None or mc_prob is None:
        # Fallback: simple normal model from adjusted per-minute rates
        # Mean outcome
        mu = mu_min * minutes * ctx_mult
        sd_base = max(1.0, sd_min * np.sqrt(max(minutes, 1.0)) * heavy)
        p_over = float(1.0 - norm.cdf(line, mu, sd_base))
        p_over = float(np.clip(p_over, 0.02, 0.98))
        mc_mean, mc_sd, mc_prob = float(mu), float(sd_base), float(p_over)

    ev_leg_even = mc_prob - (1.0 - mc_prob)

    return {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": float(mc_mean),
        "sd": float(mc_sd),
        "prob_over": float(mc_prob),
        "ev_leg_even": float(ev_leg_even),
        "team": team,
        "ctx_mult": float(ctx_mult),
        "msg": msg,
        "teammate_out": bool(teammate_out),
        "blowout": bool(blowout),
        "avg_min": float(avg_min),
    }, None

# =========================================================
#  PART 3.4 — KELLY FORMULA FOR 2-PICK
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float):
    """
    Kelly criterion for 2-pick entries.
    """
    b = payout_mult - 1.0
    q = 1.0 - p_joint
    raw = (b * p_joint - q) / b
    k = raw * frac
    return float(np.clip(k, 0.0, MAX_KELLY_PCT))  # cap at MAX_KELLY_PCT

# =========================================================
#  PART 3.5 — SELF-LEARNING CALIBRATION ENGINE
# =========================================================

def compute_model_drift(history_df: pd.DataFrame):
    """
    Self-learning calibration hook.

    Looks at completed bets and compares:
      - predicted edge (EV)
      - actual hit rate
      - CLV trends

    Returns:
      prob_mult: scales distance from 50% (over/under confidence)
      ev_shift:  small additive shift to EV (in probability space)
    """
    if history_df is None or history_df.empty:
        return 1.0, 0.0

    comp = history_df[history_df["Result"].isin(["Hit", "Miss"])].copy()
    if comp.empty or len(comp) < 30:
        # Not enough data to trust calibration
        return 1.0, 0.0

    comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
    comp = comp.dropna(subset=["EV_float"])
    if comp.empty:
        return 1.0, 0.0

    # Predicted average win probability from EV
    pred_win_prob = 0.5 + comp["EV_float"].mean()
    actual_win_prob = (comp["Result"] == "Hit").mean()

    gap = actual_win_prob - pred_win_prob  # positive → model underconfident

    # Probability multiplier (shrink or stretch away from 50%)
    if gap < -0.05:
        prob_mult = 0.93  # too optimistic
    elif gap < -0.02:
        prob_mult = 0.97
    elif gap > 0.05:
        prob_mult = 1.07  # too conservative
    elif gap > 0.02:
        prob_mult = 1.03
    else:
        prob_mult = 1.0

    ev_shift = float(np.clip(gap, -0.05, 0.05))

    return float(prob_mult), ev_shift

def apply_calibration_to_leg(leg: dict | None, prob_mult: float, ev_shift: float):
    """Adjusts leg probability & EV using self-learning calibration outputs."""
    if not leg:
        return

    base_p = float(leg["prob_over"])
    # Transform around 0.5 so extremes are dampened
    adj_p = 0.5 + (base_p - 0.5) * prob_mult + ev_shift
    adj_p = float(np.clip(adj_p, 0.01, 0.99))

    leg["prob_over_raw"] = base_p
    leg["prob_over"] = adj_p
    leg["ev_leg_even"] = float(adj_p - (1.0 - adj_p))

# =========================================================
#  PART 4 — UI RENDER ENGINE + LOADERS + DECISION LOGIC
# =========================================================

def render_leg_card(leg: dict, container, compact=False):
    """
    Displays a stylized card showing:
      - headshot
      - player + market info
      - mean, sd, ctx multiplier
      - model probability
      - EV at even money
    """
    player = leg["player"]
    market = leg["market"]
    msg = leg["msg"]
    line = leg["line"]
    mu = leg["mu"]
    sd = leg["sd"]
    p = leg["prob_over"]
    ctx = leg["ctx_mult"]
    even_ev = leg["ev_leg_even"]
    teammate_out = leg["teammate_out"]
    blowout = leg["blowout"]

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

        if headshot:
            st.image(headshot, width=120)

        st.write(f"📌 **Line:** {line}")
        st.write(f"📊 **Model Mean (MC):** {mu:.2f}")
        st.write(f"📉 **Model SD (MC):** {sd:.2f}")
        st.write(f"⏱️ **Context Multiplier (Defense/Pace):** {ctx:.3f}")
        st.write(f"🎯 **Model Probability Over (Calibrated):** {p*100:.1f}%")
        st.write(f"💵 **Even-Money EV (Calibrated):** {even_ev*100:+.1f}%")
        st.caption(f"📝 {msg}")

        # Risk flags
        if teammate_out:
            st.info("⚠️ Teammate out → usage & minutes boost applied.")
        if blowout:
            st.warning("⚠️ Blowout risk → minutes trimmed.")

def run_loader():
    """Friendly loading animation for model runs."""
    load_ph = st.empty()
    msgs = [
        "Pulling player logs…",
        "Analyzing matchup context…",
        "Adjusting pace & usage…",
        "Running Monte Carlo simulations…",
        "Finalizing edge…",
    ]
    for m in msgs:
        load_ph.markdown(
            f"<p style='color:#FFCC33;font-size:20px;font-weight:600;'>{m}</p>",
            unsafe_allow_html=True,
        )
        time.sleep(0.35)
    load_ph.empty()

def combo_decision(ev_combo: float) -> str:
    """Converts EV into a recommendation string."""
    if ev_combo >= 0.10:
        return "🔥 **PLAY — Strong Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **Lean — Thin Edge**"
    else:
        return "❌ **Pass — No Edge**"

# =========================================================
#  PART 5 — HISTORY HELPERS
# =========================================================

def ensure_history():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac"
        ])
        df.to_csv(LOG_FILE, index=False)

def load_history() -> pd.DataFrame:
    ensure_history()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac"
        ])

def save_history(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)

def compute_today_pnl(df: pd.DataFrame, payout_mult: float) -> float:
    """Computes today's realized PnL from logged history."""
    if df is None or df.empty:
        return 0.0
    df = df.copy()
    df["DateOnly"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    today = date.today()
    today_df = df[df["DateOnly"] == today]
    if today_df.empty:
        return 0.0

    def _net(row):
        if row["Result"] == "Hit":
            return row["Stake"] * (payout_mult - 1.0)
        elif row["Result"] == "Miss":
            return -row["Stake"]
        else:
            return 0.0

    pnl_vals = today_df.apply(_net, axis=1)
    return float(pnl_vals.sum())

# =========================================================
#  PART 6 — APP TABS
# =========================================================

tab_model, tab_results, tab_history, tab_calib = st.tabs(
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration"]
)

# =========================================================
#  PART 6.1 — MODEL TAB
# =========================================================

with tab_model:

    st.subheader("2-Pick Projection & Edge (Auto stats + manual lines)")

    c1, c2 = st.columns(2)

    # LEFT LEG — PLAYER 1 INPUTS
    with c1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS)
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5)
        o1 = st.text_input("P1 Opponent (abbr)", help="Example: BOS, DEN")
        p1_teammate_out = st.checkbox("P1: Key teammate out?")
        p1_blowout = st.checkbox("P1: Blowout risk high?")

    # RIGHT LEG — PLAYER 2 INPUTS
    with c2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS)
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5)
        o2 = st.text_input("P2 Opponent (abbr)", help="Example: BOS, DEN")
        p2_teammate_out = st.checkbox("P2: Key teammate out?")
        p2_blowout = st.checkbox("P2: Blowout risk high?")

    leg1 = None
    leg2 = None

    run = st.button("Run Model ⚡")

    if run:

        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

        # Compute legs
        leg1, err1 = (
            compute_leg_projection(
                p1, m1, l1, o1, p1_teammate_out, p1_blowout, games_lookback
            )
            if p1 and l1 > 0 else (None, None)
        )

        leg2, err2 = (
            compute_leg_projection(
                p2, m2, l2, o2, p2_teammate_out, p2_blowout, games_lookback
            )
            if p2 and l2 > 0 else (None, None)
        )

        if err1:
            st.error(f"P1: {err1}")
        if err2:
            st.error(f"P2: {err2}")

        # Load history for calibration / bankroll checks
        history_df = load_history()
        prob_mult, ev_shift = compute_model_drift(history_df)

        # Apply calibration
        apply_calibration_to_leg(leg1, prob_mult, ev_shift)
        apply_calibration_to_leg(leg2, prob_mult, ev_shift)

        # Render Leg Cards
        colL, colR = st.columns(2)
        if leg1:
            render_leg_card(leg1, colL, compact_mode)
        if leg2:
            render_leg_card(leg2, colR, compact_mode)

        # -----------------------------------------------------
        # Market vs Model Probability Check (single legs)
        # -----------------------------------------------------
        st.markdown("---")
        st.subheader("📈 Market vs Model Probability Check")

        def implied_probability(mult):
            return 1.0 / float(mult)

        imp_prob = implied_probability(payout_mult)
        st.markdown(f"**Market Implied Probability (per leg at {payout_mult:.2f}x):** {imp_prob*100:.1f}%")

        if leg1:
            st.markdown(
                f"**{leg1['player']} Model Prob:** {leg1['prob_over']*100:.1f}% "
                f"→ Edge: {(leg1['prob_over'] - imp_prob)*100:+.1f}%"
            )
        if leg2:
            st.markdown(
                f"**{leg2['player']} Model Prob:** {leg2['prob_over']*100:.1f}% "
                f"→ Edge: {(leg2['prob_over'] - imp_prob)*100:+.1f}%"
            )

        # -----------------------------------------------------
        # 2-PICK COMBO — JOINT MONTE CARLO RESULT
        # -----------------------------------------------------
        if leg1 and leg2:

            # Correlation estimate from engine
            corr = estimate_player_correlation(leg1, leg2)

            # Joint Monte Carlo on correlated normals using MC marginals
            mu1, sd1, line1 = leg1["mu"], leg1["sd"], leg1["line"]
            mu2, sd2, line2 = leg2["mu"], leg2["sd"], leg2["line"]

            z1 = np.random.normal(size=MC_SIMS)
            eps = np.random.normal(size=MC_SIMS)
            z2 = corr * z1 + np.sqrt(max(1e-6, 1.0 - corr**2)) * eps

            sim1 = mu1 + sd1 * z1
            sim2 = mu2 + sd2 * z2

            leg1_mc_prob = float(np.mean(sim1 > line1))
            leg2_mc_prob = float(np.mean(sim2 > line2))
            joint_prob = float(np.mean((sim1 > line1) & (sim2 > line2)))

            # Use calibrated single-leg probabilities when close; otherwise blend
            p1 = leg1["prob_over"]
            p2 = leg2["prob_over"]
            single_joint = p1 * p2
            joint_prob = 0.6 * joint_prob + 0.4 * single_joint
            joint_prob = float(np.clip(joint_prob, 0.0, 1.0))

            ev_combo = payout_mult * joint_prob - 1.0
            k_frac = kelly_for_combo(joint_prob, payout_mult, fractional_kelly)
            raw_stake = bankroll * k_frac

            # Daily loss cap enforcement
            today_pnl = compute_today_pnl(history_df, payout_mult)
            max_daily_loss = -abs(bankroll * max_daily_loss_pct / 100.0)

            if today_pnl <= max_daily_loss:
                stake = 0.0
                decision = "❌ Daily loss limit reached — reduce volume or stop for today."
                st.warning(
                    f"Daily loss cap hit. Today's PnL: {today_pnl:.2f} vs cap {max_daily_loss:.2f}. "
                    "Stake set to $0."
                )
            else:
                stake = round(raw_stake, 2)
                decision = combo_decision(ev_combo)

            st.markdown("### 🎯 **2-Pick Combo Result (Joint Monte Carlo)**")
            st.markdown(f"- Correlation Estimate: **{corr:+.2f}**")
            st.markdown(f"- Joint Probability (MC): **{joint_prob*100:.1f}%**")
            st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Suggested Stake (Kelly-capped): **${stake:.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")

        # -----------------------------------------------------
        # Market Baseline Library Info
        # -----------------------------------------------------
        for leg in [leg1, leg2]:
            if leg:
                mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
                if mean_b is not None:
                    st.caption(
                        f"📊 Market Baseline for {leg['player']} {leg['market']}: "
                        f"mean={mean_b:.1f}, median={med_b:.1f}"
                    )

# =========================================================
#  PART 6.2 — RESULTS TAB
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
                "CLV (Closing - Entry) in %",
                min_value=-20.0,
                max_value=20.0,
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

            df = pd.concat(
                [df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            save_history(df)

            st.success("Result logged ✅")

    # Summary metrics
    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    if not comp.empty:

        pnl = comp.apply(
            lambda r:
                r["Stake"] * (payout_mult - 1.0)
                if r["Result"] == "Hit"
                else -r["Stake"],
            axis=1,
        )

        hits = (comp["Result"] == "Hit").sum()
        total = len(comp)

        hit_rate = (hits / total * 100.0) if total > 0 else 0.0
        roi = pnl.sum() / max(1.0, bankroll) * 100.0

        st.markdown(
            f"**Completed Bets:** {total}  |  "
            f"**Hit Rate:** {hit_rate:.1f}%  |  "
            f"**ROI vs Bankroll:** {roi:+.1f}%"
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
#  PART 6.3 — HISTORY TAB
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

        filt = df[df["EV"] >= min_ev]

        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        st.markdown(f"**Filtered Bets:** {len(filt)}")
        st.dataframe(filt, use_container_width=True)

        if not filt.empty:

            def _net(row):
                if row["Result"] == "Hit":
                    return row["Stake"] * (payout_mult - 1.0)
                if row["Result"] == "Miss":
                    return -row["Stake"]
                return 0.0

            filt = filt.copy()
            filt["Net"] = filt.apply(_net, axis=1)
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
#  PART 6.4 — CALIBRATION TAB
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
            pnl = comp.apply(
                lambda r:
                    r["Stake"] * (payout_mult - 1.0)
                    if r["Result"] == "Hit"
                    else -r["Stake"],
                axis=1,
            )
            roi = pnl.sum() / max(1.0, bankroll) * 100.0

            comp["Win"] = (comp["Result"] == "Hit").astype(int)
            comp["Edge_vs_Market"] = comp["EV_float"] * 100.0

            # EV buckets
            buckets = pd.cut(
                comp["EV_float"],
                bins=[-1.0, 0.0, 0.05, 0.10, 0.20, 1.0],
                labels=["EV < 0", "0–5%", "5–10%", "10–20%", "20%+"]
            )

            bucket_summary = comp.groupby(buckets).agg(
                Count=("Win", "size"),
                HitRate=("Win", "mean"),
                AvgEV=("EV_float", "mean"),
                AvgCLV=("CLV", "mean"),
            ).reset_index().rename(columns={"EV_float": "AvgEV"})

            bucket_summary["HitRate"] = bucket_summary["HitRate"] * 100.0
            bucket_summary["AvgEV"] = bucket_summary["AvgEV"] * 100.0

            st.markdown("### 📊 EV Bucket Calibration")
            st.dataframe(bucket_summary, use_container_width=True)

            # Overall calibration
            pred_win_prob = 0.5 + comp["EV_float"].mean()
            actual_win_prob = comp["Win"].mean()
            gap = (pred_win_prob - actual_win_prob) * 100.0

            st.markdown("---")
            st.subheader("Market vs Model Performance Trend")

            fig2 = px.histogram(
                comp,
                x="Edge_vs_Market",
                nbins=20,
                title="Distribution of Model Edge vs Market (EV%)",
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                f"**Predicted Avg Win Prob (approx):** {pred_win_prob*100:.1f}%"
            )
            st.markdown(
                f"**Actual Hit Rate:** {actual_win_prob*100:.1f}%"
            )
            st.markdown(
                f"**Calibration Gap (Predicted − Actual):** {gap:+.1f}% | **ROI:** {roi:+.1f}%"
            )

            avg_clv = pd.to_numeric(comp["CLV"], errors="coerce").mean()
            st.markdown(f"**Average CLV:** {avg_clv:+.2f}%")

            # Model integrity checks
            if gap > 5:
                st.warning(
                    "Model appears **overconfident** → consider requiring higher EV before firing or shrinking Kelly fraction."
                )
            elif gap < -5:
                st.info(
                    "Model appears **conservative** → thin edges may be slightly under-trusted."
                )
            else:
                st.success("Model and results are reasonably aligned ✅")

            if avg_clv < 0:
                st.warning(
                    "Average CLV is negative → the market is moving against you. "
                    "Edges may be shrinking; consider reducing volume or revisiting assumptions."
                )

# =========================================================
#  PART 7 — FOOTER
# =========================================================

st.markdown(
    """
    <footer style='text-align:center; margin-top:30px; color:#FFCC33; font-size:11px;'>
        © 2025 NBA Prop Model • Powered by Kamal • Empirical MC, Defense-Adjusted
    </footer>
    """,
    unsafe_allow_html=True,
)
'''

# fix missing import difflib
app_py = app_py.replace("from nba_api.stats.static import players as nba_players", "from nba_api.stats.static import players as nba_players\nimport difflib")

req_txt = """streamlit
pandas
numpy
plotly
scipy
nba_api
requests
"""

readme_md = """# NBA Prop Betting Quant Engine (Streamlit)

This is a single-file Streamlit application that implements a **defense-adjusted NBA prop betting model** with:

- Automated NBA data ingestion via `nba_api`
- **Empirical bootstrap Monte Carlo** engine (10,000 simulations per leg)
- Defensive matchup engine (team + stat-specific)
- Pace-adjusted minute expectation model
- Usage + on/off boost system for injuries
- Correlated **2-pick joint Monte Carlo**
- Bankroll management with fractional Kelly + daily loss cap
- Self-learning calibration engine (EV buckets, CLV, hit-rate tracking)
- History logging, results tracking, and performance visualization

The app keeps the original layout:
- `📊 Model` tab
- `📓 Results` tab
- `📜 History` tab
- `🧠 Calibration` tab

All upgrades are **under the hood** — the UI remains simple and fast.

---

## 1. Installation

### Prerequisites

- Python 3.10+
- pip

### Clone / Download

Unzip the project, then from the project directory run:

```bash
pip install -r requirements.txt
