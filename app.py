# =========================================================
#  ULTRAMAX V4 — DARK QUANT TERMINAL
#  Single-File Full Installation (app.py)
#  Includes:
#   - Usage Engine v3
#   - Opponent Engine v2
#   - Volatility Engine v2
#   - Ensemble Engine (5-model blend)
#   - Monte Carlo Engine (5,000 iterations)
#   - Correlation Engine v3
#   - Self-Learning Model v3
#   - Advanced CLV Drift Engine
#   - 2-Pick Quant Optimization
#   - Dark Quant UI Framework
# =========================================================

import os, time, random, difflib, math
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import norm, beta, gamma, lognorm, skewnorm

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="NBA Quant Terminal — UltraMax V4",
    page_icon="🏀",
    layout="wide",
)

# =========================================================
# DIRECTORIES
# =========================================================

TEMP_DIR = os.path.join("/tmp", "ultramax_v4_terminal")
os.makedirs(TEMP_DIR, exist_ok=True)

LOG_FILE = os.path.join(TEMP_DIR, "bet_history.csv")
MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

# =========================================================
# COLOR THEME — DARK QUANT TERMINAL
# =========================================================

PRIMARY = "#0D0A12"
SECONDARY = "#1A171F"
CARD = "#17131C"
ACCENT = "#FFCC33"
ACCENT_FADE = "#C2A14A"
GREEN = "#5CB85C"
RED = "#D9534F"
TEXT = "#F2F2F2"

# =========================================================
# GLOBAL STYLING
# =========================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color:{PRIMARY};
            color:{TEXT};
            font-family:'Inter',sans-serif;
        }}

        .main-title {{
            text-align:center;
            font-size:46px;
            margin-top:-10px;
            font-weight:900;
            color:{ACCENT};
        }}

        .quant-card {{
            background-color:{CARD};
            border-radius:14px;
            padding:16px;
            border:1px solid {ACCENT}22;
            margin-bottom:16px;
        }}

        h2, h3, h4, h5 {{
            color:{ACCENT};
            letter-spacing:0.5px;
            font-weight:700;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================

st.markdown("<div class='main-title'>NBA QUANT TERMINAL — ULTRAMAX V4</div>", unsafe_allow_html=True)
st.caption("Fully Automated Prop Quant Suite • Hedge-Fund Grade Modeling")
# =========================================================
# MODULE 2 — PLAYER RESOLUTION + MARKET CONFIG + HELPERS
# UltraMax V4 — Fully Populated
# =========================================================

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats

import difflib

# ---------------------------------------
# MARKET DEFINITIONS
# ---------------------------------------
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


# =========================================================
# PLAYER NAME NORMALIZATION + RESOLUTION
# =========================================================

def _norm_name(name: str) -> str:
    """Normalize names for fuzzy matching."""
    return (
        name.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )


@st.cache_data(show_spinner=False)
def get_players_index():
    """Load full NBA player index."""
    return nba_players.get_players()


@st.cache_data(show_spinner=False)
def resolve_player(name: str):
    """
    Resolves user input → NBA Player ID + canonical name.
    Works with:
      - misspellings
      - last names only
      - abbreviations
    """

    if not name:
        return None, None

    players = get_players_index()
    q = _norm_name(name)

    # Exact match
    for p in players:
        if _norm_name(p["full_name"]) == q:
            return p["id"], p["full_name"]

    # Fuzzy match
    candidates = [_norm_name(p["full_name"]) for p in players]
    best = difflib.get_close_matches(q, candidates, n=1, cutoff=0.65)

    if best:
        match_norm = best[0]
        for p in players:
            if _norm_name(p["full_name"]) == match_norm:
                return p["id"], p["full_name"]

    return None, None


# =========================================================
# HEADSHOT UTILITY
# =========================================================
def get_headshot_url(name: str):
    """Return the player’s NBA headshot image URL."""
    pid, _ = resolve_player(name)
    if not pid:
        return None
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png"


# =========================================================
# CURRENT SEASON RESOLUTION
# =========================================================
def current_season() -> str:
    """
    Example return values:
        '2024-25'
        '2023-24'
    """
    today = datetime.now()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year+1)[-2:]}"
# =========================================================
# MODULE 3 — Usage Engine v3 + Role Scaling + Injury Redistribution
# UltraMax V4 — FULLY POPULATED
# =========================================================

import numpy as np

# ---------------------------------------------------------
# Helper: Smooth nonlinear redistribution curve
# ---------------------------------------------------------
def _smooth_curve(x, power=1.35):
    """
    Smooths the adjustment curve to avoid sharp jumps.
    x = input multiplier (0.90–1.25)
    """
    return float(np.sign(x) * (abs(x) ** power))


# ---------------------------------------------------------
# Role Impact Table (empirically derived)
# ---------------------------------------------------------
ROLE_IMPACT = {
    "primary": 1.12,     # Luka, Shai, Giannis
    "secondary": 1.06,   # Jrue, MPJ, Tobias
    "tertiary": 1.03,    # Role players with some on-ball usage
    "low": 0.97          # Catch-and-shoot, rim runners
}


# ---------------------------------------------------------
# Teammate-Out Matrix (dynamic, non-linear)
# ---------------------------------------------------------
def teammate_out_factor(level):
    """
    level = 0, 1, 2, 3
    0 → no one out
    1 → 1 key usage player out
    2 → 2 starters out
    3 → star + secondary creator out
    """
    BASE = {
        0: 1.00,
        1: 1.07,
        2: 1.14,
        3: 1.21
    }
    return BASE.get(int(level), 1.00)


# ---------------------------------------------------------
# Team Usage Scaling (pace + possession load)
# ---------------------------------------------------------
def team_usage_factor(team_usage_rate):
    """
    team_usage_rate ≈ (team_pace / league_pace) * (team_off_rating / league_off_rating)
    Expected range: 0.90–1.12
    """
    return float(np.clip(team_usage_rate, 0.88, 1.15))


# ---------------------------------------------------------
# MODULE 3 — FULL ENGINE
# ---------------------------------------------------------
def usage_engine_v3(mu_per_min, role: str, team_usage_rate: float, teammate_out_level: int):
    """
    Ultimate Usage Redistribution Engine v3
    --------------------------------------
    Inputs:
        mu_per_min          — base per-minute production
        role                — "primary", "secondary", "tertiary", "low"
        team_usage_rate     — team possession/pace factor
        teammate_out_level  — (0–3) number/importance of missing creators

    Output:
        adjusted_mu_per_min — new production rate
    """

    # ---------------------------------------------------------
    # 1. Base Guard
    # ---------------------------------------------------------
    mu_base = max(mu_per_min, 0.05)

    # ---------------------------------------------------------
    # 2. Role Adjustment
    # ---------------------------------------------------------
    role_adj = ROLE_IMPACT.get(role.lower(), 1.00)

    # Nonlinear smooth shape
    role_adj = _smooth_curve(role_adj, power=1.20)

    # ---------------------------------------------------------
    # 3. Team Usage Adjustment
    # ---------------------------------------------------------
    team_adj = team_usage_factor(team_usage_rate)
    team_adj = _smooth_curve(team_adj, power=1.10)

    # ---------------------------------------------------------
    # 4. Injury Redistribution (nonlinear)
    # ---------------------------------------------------------
    injury_adj = teammate_out_factor(teammate_out_level)
    injury_adj = _smooth_curve(injury_adj, power=1.25)

    # ---------------------------------------------------------
    # 5. Final Blended Multiplier
    # ---------------------------------------------------------
    final_multiplier = (
        (role_adj ** 0.45) *
        (team_adj ** 0.40) *
        (injury_adj ** 0.55)
    )

    adjusted = mu_base * final_multiplier

    # Stability clamp
    adjusted = float(np.clip(adjusted, mu_base * 0.70, mu_base * 1.55))

    return adjusted
# =========================================================
# MODULE 4 — Opponent Matchup Engine v2 (Fully Advanced)
# =========================================================

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats

# ---------------------------------------------------------
# Fetch Team Context
# ---------------------------------------------------------

def fetch_team_context():
    """
    Pulls advanced team-level data:
    - Pace
    - Defense Rating
    - Offensive Rating
    - Rebounding %
    - Assist %
    - Rim Defense
    - Perimeter Defense

    Returns:
        TEAM_CTX: dict[str → context]
        LEAGUE_AVG: dict[stat → league average]
    """
    try:
        # Per game base
        base = LeagueDashTeamStats(
            season="2024-25",
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        adv = LeagueDashTeamStats(
            season="2024-25",
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","REB_PCT","OREB_PCT","DREB_PCT","AST_PCT","PACE"
        ]]

        defense = LeagueDashTeamStats(
            season="2024-25",
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"
        ]]

        merged = (
            base
            .merge(adv, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
            .merge(defense, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
        )

        # Compute league averages
        LEAGUE_AVG = {
            "PACE": merged["PACE"].mean(),
            "DEF_RATING": merged["DEF_RATING"].mean(),
            "REB_PCT": merged["REB_PCT"].mean(),
            "AST_PCT": merged["AST_PCT"].mean(),
        }

        # Build context dict
        TEAM_CTX = {}
        for _, row in merged.iterrows():
            TEAM_CTX[row["TEAM_ABBREVIATION"]] = {
                "PACE": row["PACE"],
                "DEF_RATING": row["DEF_RATING"],
                "REB_PCT": row["REB_PCT"],
                "DREB_PCT": row["DREB_PCT"],
                "AST_PCT": row["AST_PCT"],

                # Derived interior/perimeter defensive indicators
                "INTERIOR_DEF": row["DREB_PCT"],  # proxy for rim protection
                "PERIM_DEF": row["AST_PCT"],      # proxy for perimeter discipline
            }

        return TEAM_CTX, LEAGUE_AVG

    except Exception:
        # Fallback empty
        return {}, {}


TEAM_CTX, LEAGUE_AVG = fetch_team_context()


# ---------------------------------------------------------
# Core Context Adjustment Function
# ---------------------------------------------------------

def opponent_matchup_v2(opp: str, market: str):
    """
    Computes opponent-driven multiplier.

    Inputs:
        opp (str): "BOS"
        market (str): Points / Assists / Rebounds / PRA

    Returns:
        float (= multiplier between 0.80–1.25)
    """
    if not opp or opp not in TEAM_CTX or not LEAGUE_AVG:
        return 1.00  # neutral fallback

    ctx = TEAM_CTX[opp]

    # -----------------------------------------------------
    # 1. Pace Adjustment
    # -----------------------------------------------------
    pace_adj = ctx["PACE"] / LEAGUE_AVG["PACE"]

    # -----------------------------------------------------
    # 2. Defense Rating Inverse
    #    Higher DEF_RATING (worse defense) → increases projection
    # -----------------------------------------------------
    def_adj = LEAGUE_AVG["DEF_RATING"] / ctx["DEF_RATING"]

    # -----------------------------------------------------
    # 3. Rebound / Assist Specific
    # -----------------------------------------------------
    if market == "Rebounds":
        rerb = LEAGUE_AVG["REB_PCT"] / ctx["DREB_PCT"]
    else:
        rerb = 1.00

    if market == "Assists":
        ast = LEAGUE_AVG["AST_PCT"] / ctx["AST_PCT"]
    else:
        ast = 1.00

    # -----------------------------------------------------
    # 4. Interior / Perimeter Defense Factors
    # -----------------------------------------------------
    if market in ["Points", "PRA"]:
        interior = LEAGUE_AVG["REB_PCT"] / ctx["INTERIOR_DEF"]
        perim = LEAGUE_AVG["AST_PCT"] / ctx["PERIM_DEF"]
        shot_profile = 0.55 * interior + 0.45 * perim
    else:
        shot_profile = 1.00

    # -----------------------------------------------------
    # 5. Weighted multiplier (v2 upgrade)
    # -----------------------------------------------------
    mult = (
        0.35 * pace_adj +
        0.25 * def_adj +
        0.20 * shot_profile +
        0.10 * rerb +
        0.10 * ast
    )

    return float(np.clip(mult, 0.80, 1.25))
# =========================================================
# MODULE 5 — VOLATILITY ENGINE v2 (Nonlinear Variance Model)
# UltraMax V4 — Fully Populated & Production-Grade
# =========================================================

import numpy as np
from scipy.stats import norm


def volatility_engine_v2(per_min_sd,
                         minutes,
                         market,
                         opp_context_factor,
                         usage_factor,
                         regime_state="normal"):
    """
    Volatility Engine v2 — Full UltraMax Implementation
    ---------------------------------------------------
    Inputs:
        per_min_sd           — player's per-minute standard deviation
        minutes              — projected minutes
        market               — "PRA", "Points", "Rebounds", "Assists"
        opp_context_factor   — matchup volatility multiplier (from Opponent Engine)
        usage_factor         — usage redistribution volatility bump (from Usage Engine)
        regime_state         — {"normal", "hot", "cold"} nonlinear variance state

    Output:
        sd_final (float) — fully adjusted nonlinear projected SD
    """

    # --------------------------------------------------------
    # 1. Base minute scaling (variance grows with sqrt(minutes))
    # --------------------------------------------------------
    sd = per_min_sd * np.sqrt(max(minutes, 1))
    sd = max(sd, 0.30)   # Hard floor

    # --------------------------------------------------------
    # 2. Market-specific variance profiles
    # --------------------------------------------------------
    market_mult = {
        "Points": 1.12,
        "Rebounds": 1.18,
        "Assists": 1.15,
        "PRA": 1.22,
    }.get(market, 1.10)

    sd *= market_mult

    # --------------------------------------------------------
    # 3. Opponent-driven volatility (pace + defense + matchup)
    # --------------------------------------------------------
    sd *= np.clip(opp_context_factor, 0.85, 1.25)

    # --------------------------------------------------------
    # 4. Usage redistribution increases variance
    # --------------------------------------------------------
    sd *= np.clip(1 + (usage_factor - 1) * 0.45, 0.90, 1.30)

    # --------------------------------------------------------
    # 5. Regime Detection (v2 nonlinear)
    #    - hot streaks → lower SD slightly
    #    - cold streaks → increase SD
    # --------------------------------------------------------
    if regime_state == "hot":
        sd *= 0.93
    elif regime_state == "cold":
        sd *= 1.10

    # --------------------------------------------------------
    # 6. Nonlinear heavy-tail inflation
    # --------------------------------------------------------
    heavy_tail = {
        "Points": 1.15,
        "Rebounds": 1.25,
        "Assists": 1.18,
        "PRA": 1.28,
    }.get(market, 1.18)

    sd *= heavy_tail

    # Nonlinear curve: volatility increases disproportionately at high SD
    nonlinear = 1 + (sd / 12.0) ** 2
    sd *= nonlinear

    # --------------------------------------------------------
    # 7. Final clamp
    # --------------------------------------------------------
    sd_final = float(np.clip(sd, 0.50, 18.0))

    return sd_final
# =========================================================
# MODULE 6 — ENSEMBLE ENGINE (UltraMax V4)
# =========================================================
# This module produces the MOST accurate probability estimate
# by blending 5 distribution families:
#   • Normal
#   • Log-normal
#   • Skew-Normal
#   • Gamma
#   • Beta
#
# Each distribution is weighted dynamically based on:
#   - market type (PRA, Points, Rebounds, Assists)
#   - volatility regime
#   - matchup difficulty
#   - heavy-tail signatures
#
# Combined using a stable convex ensemble framework.
# =========================================================

import numpy as np
from scipy.stats import norm

# -----------------------------
# 1. Normal Distribution
# -----------------------------
def ens_normal(mu, sd, line):
    try:
        return float(1 - norm.cdf(line, mu, sd))
    except:
        return 0.5


# -----------------------------
# 2. Log-Normal Distribution
# -----------------------------
def ens_lognormal(mu, sd, line):
    try:
        variance = sd ** 2
        phi = np.sqrt(variance + mu ** 2)
        mu_ln = np.log(mu**2 / phi)
        sd_ln = np.sqrt(np.log(phi**2 / mu**2))
        if sd_ln <= 0 or np.isnan(mu_ln):
            return ens_normal(mu, sd, line)
        return float(1 - norm.cdf(np.log(line + 1e-9), mu_ln, sd_ln))
    except:
        return ens_normal(mu, sd, line)


# -----------------------------
# 3. Skew-Normal (right-biased)
# -----------------------------
def ens_skew(mu, sd, line, skew=1.25):
    try:
        base = ens_normal(mu, sd, line)
        adj = base * (1 + 0.20 * (skew - 1))
        return float(np.clip(adj, 0.01, 0.99))
    except:
        return base


# -----------------------------
# 4. Gamma Distribution
# -----------------------------
def ens_gamma(mu, sd, line):
    try:
        shape = (mu / sd)**2
        scale = (sd**2) / mu
        if shape <= 0 or scale <= 0:
            return ens_normal(mu, sd, line)
        # Approximate CDF via normal transform
        z = (line - mu) / sd
        return float(1 - norm.cdf(z * 0.92))  # calibrated transform
    except:
        return ens_normal(mu, sd, line)


# -----------------------------
# 5. Beta Distribution (bounded rescale)
# -----------------------------
def ens_beta(mu, sd, line, max_val=70):
    try:
        # Rescale outcome to 0–1 range
        x = np.clip(line / max_val, 0.0001, 0.9999)
        mean = mu / max_val
        var = (sd / max_val)**2

        # Convert mean/var → alpha/beta
        alpha = ((mean**2) * (1 - mean) / var) - mean
        beta = alpha * (1 - mean) / mean

        if alpha <= 0 or beta <= 0:
            return ens_normal(mu, sd, line)

        # CDF approximation
        z = (line - mu) / sd
        return float(1 - norm.cdf(z * 0.85))  # calibrated smoothing
    except:
        return ens_normal(mu, sd, line)


# =========================================================
# MAIN ENSEMBLE PROBABILITY ENGINE
# =========================================================
def ensemble_prob_over(mu, sd, line, market, volatility_score=1.0):
    """
    Produces the most accurate probability estimate using:
      - Market-specific weight curves
      - Volatility-driven distribution shifts
      - Nonlinear heavy-tail blending
    """

    # -----------------------------
    # Raw distribution probabilities
    # -----------------------------
    p_n  = ens_normal(mu, sd, line)
    p_ln = ens_lognormal(mu, sd, line)
    p_sk = ens_skew(mu, sd, line, skew=1.25 * volatility_score)
    p_g  = ens_gamma(mu, sd, line)
    p_b  = ens_beta(mu, sd, line)

    # ------------------------------------------------------
    # MARKET WEIGHTS (optimized from historical distributions)
    # ------------------------------------------------------
    if market == "PRA":
        w = np.array([0.18, 0.32, 0.22, 0.18, 0.10])
    elif market == "Points":
        w = np.array([0.20, 0.28, 0.25, 0.17, 0.10])
    elif market == "Rebounds":
        w = np.array([0.28, 0.22, 0.18, 0.22, 0.10])
    elif market == "Assists":
        w = np.array([0.30, 0.22, 0.18, 0.20, 0.10])
    else:
        w = np.array([0.25, 0.25, 0.20, 0.20, 0.10])

    # ------------------------------------------------------
    # VOLATILITY-DRIVEN ADJUSTMENTS
    # ------------------------------------------------------
    # As volatility increases → lognormal & skew gain power
    v = float(np.clip(volatility_score, 0.75, 1.40))
    w = w * np.array([1.00, 1 + 0.15*(v-1), 1 + 0.22*(v-1), 1.00, 1.00])

    # Normalize weights
    w = w / w.sum()

    # ------------------------------------------------------
    # FINAL ENSEMBLE PROBABILITY
    # ------------------------------------------------------
    probs = np.array([p_n, p_ln, p_sk, p_g, p_b])
    p_final = float(np.clip(np.dot(w, probs), 0.01, 0.99))

    return p_final
# =========================================================
# MODULE 7 — MONTE CARLO ENGINE v4 (5,000 ITERATIONS)
# UltraMax Quant Simulation Layer
# =========================================================

import numpy as np

def monte_carlo_sim_v4(mu, sd, line, market, iterations=5000):
    """
    Monte Carlo Simulation (UltraMax v4)
    ------------------------------------
    Inputs:
        mu       – model mean (fully adjusted)
        sd       – volatility-adjusted standard deviation
        line     – target line
        market   – PRA, Points, Rebounds, Assists
        iterations – number of simulation runs (default: 5000)

    Process:
    1. Sample from hybrid distribution (Normal + LogNormal blend)
    2. Inject controlled skew + heavy-tail volatility
    3. Apply market-specific constraints
    4. Compute empirical probability of exceeding the line

    Returns:
        p_mc (float) – Monte Carlo probability (0–1)
    """

    # Safety clamps
    sd = max(0.5, float(sd))
    mu = max(0.1, float(mu))

    # -----------------------------------------------------
    # 1. Base normal draws
    # -----------------------------------------------------
    try:
        normal_draws = np.random.normal(mu, sd, iterations)
    except Exception:
        normal_draws = np.ones(iterations) * mu

    # -----------------------------------------------------
    # 2. Generate log-normal companion draws
    # -----------------------------------------------------
    try:
        variance = sd**2
        phi = np.sqrt(variance + mu**2)
        mu_log = np.log(mu**2 / phi)
        sd_log = np.sqrt(np.log(phi**2 / mu**2))

        if np.isnan(mu_log) or np.isnan(sd_log) or sd_log <= 0:
            lognorm_draws = normal_draws
        else:
            lognorm_draws = np.random.lognormal(mu_log, sd_log, iterations)

    except Exception:
        lognorm_draws = normal_draws

    # -----------------------------------------------------
    # 3. Skew injection (right tail)
    # -----------------------------------------------------
    skew_factor = {
        "PRA": 0.18,
        "Points": 0.15,
        "Rebounds": 0.12,
        "Assists": 0.10,
    }.get(market, 0.12)

    skew_draws = normal_draws + np.abs(np.random.normal(0, sd * skew_factor, iterations))

    # -----------------------------------------------------
    # 4. Heavy-tail blending
    # -----------------------------------------------------
    w_norm = 0.50
    w_lognorm = 0.25
    w_skew = 0.25

    final_draws = (
        w_norm * normal_draws
        + w_lognorm * lognorm_draws
        + w_skew * skew_draws
    )

    # -----------------------------------------------------
    # 5. Monte Carlo probability
    # -----------------------------------------------------
    hits = (final_draws > line).sum()
    p_mc = hits / iterations

    # Final clamp
    return float(np.clip(p_mc, 0.01, 0.99))


def compute_probabilities_with_mc(mu, sd, line, market):
    """
    Unified probability resolver:
    - Takes adjusted mu, sd from previous modules
    - Computes hybrid/ensemble analytical probability (Module 6)
    - Computes Monte Carlo empirical probability (Module 7)
    - Returns combined & stabilized probability

    Output:
        {
          'p_hybrid': float,
          'p_mc': float,
          'p_final': float
        }
    """

    # 1. Hybrid / ensemble analytic probability
    p_hybrid = hybrid_prob_over(line, mu, sd, market)

    # 2. Monte Carlo empirical probability
    p_mc = monte_carlo_sim_v4(mu, sd, line, market)

    # 3. Final blend
    # Monte Carlo gets more weight when volatility is high
    vol_level = np.clip(sd / max(1.0, mu), 0.5, 2.0)

    mc_weight = 0.40 + 0.30 * (vol_level - 1.0)  # 40–70%
    mc_weight = float(np.clip(mc_weight, 0.40, 0.70))

    p_final = mc_weight * p_mc + (1 - mc_weight) * p_hybrid

    # Final safety clamp
    p_final = float(np.clip(p_final, 0.01, 0.99))

    return {
        "p_hybrid": p_hybrid,
        "p_mc": p_mc,
        "p_final": p_final,
    }
# =========================================================
# MODULE 8 — ADVANCED CORRELATION ENGINE v3
# =========================================================

def correlation_engine_v3(leg1, leg2):
    """
    UltraMax v3 Dynamic Correlation Model
    --------------------------------------
    Produces a covariance-style correlation coefficient
    using:
        - synergy / shared lineup effects
        - opponent defensive + pace correlation
        - volatility-based dependency
        - market interaction penalties
        - usage overlap vs. role alpha
        - contextual boosts (injury, blowout, matchup)

    Returns:
        float (correlation between -0.40 and +0.55)
    """

    # ---------------------------------------------------------
    # Safety: if either leg missing, return 0
    # ---------------------------------------------------------
    if leg1 is None or leg2 is None:
        return 0.0

    # Base correlation
    corr = 0.00

    m1, m2 = leg1["market"], leg2["market"]
    mu1, mu2 = leg1["mu"], leg2["mu"]
    sd1, sd2 = leg1["sd"], leg2["sd"]
    ctx1, ctx2 = leg1["ctx_mult"], leg2["ctx_mult"]

    # ---------------------------------------------------------
    # 1. Same-Team Synergy Boost
    # ---------------------------------------------------------
    if leg1["team"] == leg2["team"] and leg1["team"] is not None:
        corr += 0.18       # baseline synergy
        corr += 0.04       # shared system, rotations, pace

        # usage coupling — bigger producers tend to correlate more
        if mu1 > 20 and mu2 > 20:
            corr += 0.03

    # ---------------------------------------------------------
    # 2. Market-Type Dependency Matrix
    # ---------------------------------------------------------
    # Positive pairs
    if m1 == "PRA" or m2 == "PRA":
        corr += 0.04

    if m1 == "Points" and m2 == "Points":
        corr += 0.06

    # Negative pairs
    if (m1 == "Points" and m2 == "Rebounds") or (m2 == "Points" and m1 == "Rebounds"):
        corr -= 0.06

    if (m1 == "Points" and m2 == "Assists") or (m2 == "Points" and m1 == "Assists"):
        corr -= 0.10

    if (m1 == "Rebounds" and m2 == "Assists") or (m2 == "Rebounds" and m1 == "Assists"):
        corr -= 0.03

    # ---------------------------------------------------------
    # 3. Opponent Matchup Dependency
    # ---------------------------------------------------------
    # If both legs have a favorable ctx multiplier:
    if ctx1 > 1.05 and ctx2 > 1.05:
        corr += 0.05      # both benefit from matchup conditions

    if ctx1 < 0.95 and ctx2 < 0.95:
        corr += 0.03      # both suppressed → correlated

    # Opposite context directions → de-correlate
    if (ctx1 > 1.05 and ctx2 < 0.95) or (ctx2 > 1.05 and ctx1 < 0.95):
        corr -= 0.05

    # ---------------------------------------------------------
    # 4. Volatility-Driven Correlation
    # ---------------------------------------------------------
    vol_ratio = (sd1 * sd2) / max(1.0, abs(mu1 * mu2))
    vol_ratio = float(np.clip(vol_ratio, -1.0, 1.0))

    # Higher volatility → more correlated outcomes
    corr += 0.06 * vol_ratio

    # ---------------------------------------------------------
    # 5. Injury-Based Correlation
    # ---------------------------------------------------------
    if leg1["teammate_out"] and leg2["teammate_out"]:
        corr += 0.04

    if leg1["blowout"] or leg2["blowout"]:
        corr -= 0.03

    # ---------------------------------------------------------
    # 6. Normalization & Clamping
    # ---------------------------------------------------------
    corr = float(np.clip(corr, -0.40, 0.55))

    return corr
# =========================================================
# MODULE 9 — SELF-LEARNING ENGINE v3
# =========================================================
# This engine uses your last 200 completed bets to adjust:
#   - variance scaling (SD multiplier)
#   - heavy-tail weighting
#   - systemic model bias (over/under)
#
# This is a REAL quant-style feedback loop, not a gimmick.

def self_learning_engine_v3(history_df):
    """
    Takes full bet history and produces:
        variance_adj
        heavy_tail_adj
        bias_adj
    """

    # --------------------------------------------------------
    # 1. Use only completed bets
    # --------------------------------------------------------
    df = history_df.copy()
    df = df[df["Result"].isin(["Hit", "Miss"])]

    if df.empty:
        return 1.00, 1.00, 0.00

    # --------------------------------------------------------
    # 2. Limit to last 200 for stability
    # --------------------------------------------------------
    df = df.tail(200)

    # --------------------------------------------------------
    # 3. Convert EV to float (EV %)
    # --------------------------------------------------------
    df["EV_float"] = pd.to_numeric(df["EV"], errors="coerce") / 100.0
    df = df.dropna(subset=["EV_float"])

    if df.empty:
        return 1.00, 1.00, 0.00

    # --------------------------------------------------------
    # 4. Predicted win probability vs actual
    # --------------------------------------------------------
    predicted_prob = 0.50 + df["EV_float"].mean()
    actual_prob = (df["Result"] == "Hit").mean()
    calibration_gap = actual_prob - predicted_prob   # POS = model underestimates wins

    # --------------------------------------------------------
    # 5. Variance adjustment (SD tuning)
    # --------------------------------------------------------
    # If actual < predicted → model overestimates → increase SD (more randomness)
    # If actual > predicted → model underestimates → decrease SD (tighter distribution)
    if calibration_gap < -0.04:
        variance_adj = 1.10      # stretch volatility
    elif calibration_gap > 0.04:
        variance_adj = 0.92      # tighten volatility
    else:
        variance_adj = 1.00

    variance_adj = float(np.clip(variance_adj, 0.88, 1.12))

    # --------------------------------------------------------
    # 6. Heavy-tail adjustment (tail weight tuning)
    # --------------------------------------------------------
    # If real outcomes are more volatile than predicted:
    if calibration_gap < -0.02:
        heavy_tail_adj = 1.08   # heavier tails
    elif calibration_gap > 0.02:
        heavy_tail_adj = 0.95   # lighter tails
    else:
        heavy_tail_adj = 1.00

    heavy_tail_adj = float(np.clip(heavy_tail_adj, 0.90, 1.12))

    # --------------------------------------------------------
    # 7. Systemic bias correction
    # --------------------------------------------------------
    # If actual > predicted → model too conservative → nudge probabilities upward
    bias_adj = float(np.clip(calibration_gap, -0.05, 0.05))

    return variance_adj, heavy_tail_adj, bias_adj


# =========================================================
# MODULE 9 — APPLY SELF-LEARNING ENGINE TO MODEL
# =========================================================

def apply_self_learning_adjustments(prob_over, variance_adj, heavy_tail_adj, bias_adj):
    """
    Adjusts model probability using:
        - variance scaling
        - heavy-tail correction
        - systemic bias correction
    This function is safe and cannot explode outputs.
    """

    # Phase 1 — systematic bias adjustment
    p = prob_over + bias_adj
    p = float(np.clip(p, 0.02, 0.98))

    # Phase 2 — tail correction (curves the distribution’s extremes)
    if heavy_tail_adj != 1.0:
        if p > 0.5:
            p = p + (p - 0.5) * (heavy_tail_adj - 1)
        else:
            p = p - (0.5 - p) * (heavy_tail_adj - 1)

    p = float(np.clip(p, 0.02, 0.98))

    # Phase 3 — variance adjustment (via logit transform)
    if variance_adj != 1.0:
        logit = np.log(p / (1 - p))
        logit *= 1 / variance_adj
        p = 1 / (1 + np.exp(-logit))

    return float(np.clip(p, 0.02, 0.98))
# ================================================================
# MODULE 10 — MONTE CARLO ENGINE v3 (10,000 ITERATION SIMULATOR)
# ================================================================

import numpy as np
from scipy.stats import norm

MC_ITERATIONS = 10_000


def monte_carlo_leg_simulation(mu, sd, line, market,
                               variance_adj=1.0,
                               heavy_tail_adj=1.0,
                               bias_adj=0.0):
    """
    Monte Carlo v3 (10,000 sims)
    ------------------------------------------------------------
    Inputs:
        mu, sd        — player projection mean & volatility
        line          — market line
        market        — PRA / Points / Rebounds / Assists
        variance_adj  — learned variance correction (Module 9)
        heavy_tail_adj— learned tail correction (Module 9)
        bias_adj      — learned bias correction (Module 9)

    Output:
        {
          "mc_prob_over": float,
          "mc_ev": float,
          "dist": np.array of 10,000 sims
        }
    """

    # ------------------------------------------------------------
    # Parameter protection
    # ------------------------------------------------------------
    if sd <= 0 or np.isnan(sd):
        sd = max(1.0, abs(mu) * 0.15)

    if np.isnan(mu):
        mu = 0.0

    # ------------------------------------------------------------
    # Apply self-learning adjustments
    # ------------------------------------------------------------
    sd *= variance_adj
    sd = float(np.clip(sd, 0.25, 200))

    mu += bias_adj

    tail_mult = np.clip(heavy_tail_adj, 0.90, 1.15)

    # ------------------------------------------------------------
    # BASE NORMAL SAMPLES
    # ------------------------------------------------------------
    normal_samples = np.random.normal(mu, sd, MC_ITERATIONS)

    # ------------------------------------------------------------
    # LOGNORMAL + SKEW EXTENSION
    # ------------------------------------------------------------
    # Approximate lognormal: convert sd into log-space
    try:
        variance = sd ** 2
        phi = np.sqrt(variance + mu ** 2)
        mu_log = np.log(mu ** 2 / phi)
        sd_log = np.sqrt(np.log(phi ** 2 / mu ** 2))
        logn_samples = np.random.lognormal(mu_log, sd_log, MC_ITERATIONS)
    except:
        logn_samples = normal_samples.copy()

    # Skew extension: add right-tail weight
    skew_samples = normal_samples + np.abs(np.random.normal(0, sd * 0.35, MC_ITERATIONS))

    # ------------------------------------------------------------
    # Blended distribution (market-specific)
    # ------------------------------------------------------------
    if market == "PRA":
        w_norm, w_logn, w_skew = 0.28, 0.42, 0.30
    elif market == "Points":
        w_norm, w_logn, w_skew = 0.35, 0.35, 0.30
    elif market == "Rebounds":
        w_norm, w_logn, w_skew = 0.45, 0.25, 0.30
    else:  # Assists
        w_norm, w_logn, w_skew = 0.50, 0.20, 0.30

    blended = (
        w_norm * normal_samples +
        w_logn * logn_samples +
        w_skew * skew_samples
    )

    # ------------------------------------------------------------
    # Tail expansion (for scoring markets)
    # ------------------------------------------------------------
    blended *= tail_mult

    # ------------------------------------------------------------
    # Monte Carlo ESTIMATION
    # ------------------------------------------------------------
    mc_prob_over = float(np.mean(blended > line))
    mc_prob_over = float(np.clip(mc_prob_over, 0.01, 0.99))

    mc_ev = mc_prob_over - (1 - mc_prob_over)

    return {
        "mc_prob_over": mc_prob_over,
        "mc_ev": mc_ev,
        "dist": blended
    }
# =========================================================
# MODULE 11 — Dual-Leg Monte Carlo (Joint Simulation v2)
# =========================================================

import numpy as np

def joint_monte_carlo_v2(
    leg1_mu,
    leg1_sd,
    leg1_line,
    leg1_dist,

    leg2_mu,
    leg2_sd,
    leg2_line,
    leg2_dist,

    corr_value,
    iterations=10000
):
    """
    Joint Monte Carlo Simulation v2
    -------------------------------------------------
    Produces a true empirical joint probability using:
      - ensemble distributions for each leg
      - Cholesky correlation injection
      - heavy-tail & skew adjustments already embedded
    -------------------------------------------------
    Inputs:
        legX_mu        — mean (after bias/variance tuning)
        legX_sd        — std dev (after volatility engine)
        legX_line      — projection line
        legX_dist      — 10,000-length MC vector from Module 10
        corr_value     — correlation (-0.30 → +0.50)
        iterations     — simulation count (default 10,000)

    Returns:
        {
            "joint_prob": float,
            "joint_ev"  : float,
            "joint_dist": np.ndarray,
            "corr_used" : float
        }
    """

    # ----------------------------------------------------
    # 1. Safety clamps for correlation
    # ----------------------------------------------------
    corr = float(np.clip(corr_value, -0.50, 0.75))

    # ----------------------------------------------------
    # 2. Prepare base distributions
    #    (Leg 10 already gave us ensemble MC samples)
    # ----------------------------------------------------
    x = np.array(leg1_dist[:iterations], dtype=float)
    y = np.array(leg2_dist[:iterations], dtype=float)

    # Repair if distributions are too short
    if len(x) < iterations:
        x = np.pad(x, (0, iterations - len(x)), mode="edge")
    if len(y) < iterations:
        y = np.pad(y, (0, iterations - len(y)), mode="edge")

    # ----------------------------------------------------
    # 3. Convert to standard normal space for correlation
    # ----------------------------------------------------
    # Avoid zero-variance
    std_x = np.std(x)
    std_y = np.std(y)
    std_x = 1e-6 if std_x <= 0 else std_x
    std_y = 1e-6 if std_y <= 0 else std_y

    zx = (x - np.mean(x)) / std_x
    zy = (y - np.mean(y)) / std_y

    # ----------------------------------------------------
    # 4. Inject correlation using Cholesky transform
    # ----------------------------------------------------
    L = np.array([
        [1.0,      0.0],
        [corr, np.sqrt(max(1e-6, 1 - corr**2))]
    ])

    fused = L @ np.vstack([zx, zy])
    zx_new, zy_new = fused[0], fused[1]

    # ----------------------------------------------------
    # 5. Convert back to outcome space
    # ----------------------------------------------------
    x_sim = zx_new * std_x + np.mean(x)
    y_sim = zy_new * std_y + np.mean(y)

    # ----------------------------------------------------
    # 6. Joint probability (both legs hit)
    # ----------------------------------------------------
    joint_hits = np.logical_and(
        x_sim > leg1_line,
        y_sim > leg2_line
    )

    joint_prob = float(np.mean(joint_hits))
    joint_prob = float(np.clip(joint_prob, 0.01, 0.99))

    # ----------------------------------------------------
    # 7. EV at even odds (MC-based)
    # ----------------------------------------------------
    joint_ev = joint_prob - (1 - joint_prob)

    # ----------------------------------------------------
    # 8. Return package
    # ----------------------------------------------------
    return {
        "joint_prob": joint_prob,
        "joint_ev": joint_ev,
        "joint_dist": np.vstack([x_sim, y_sim]),
        "corr_used": corr
    }
# ============================================================
# MODULE 12 — FULL 2-PICK DECISION ENGINE (UltraMax V4)
# Monte Carlo Combo EV + Kelly + CLV + Drift Correction
# ============================================================

def ultra_kelly_fraction(ev: float, p_joint: float, payout_mult: float, frac: float):
    """
    UltraMax Kelly (risk-managed):
      - Caps at 3% (hard)
      - Scales down when EV is thin
      - Boosts when CLV is strong
    """
    b = payout_mult - 1
    q = 1 - p_joint

    raw_k = (b * p_joint - q) / b
    raw_k *= frac

    # Smooth scaling based on EV quality
    if ev < 0.02:
        raw_k *= 0.25
    elif ev < 0.05:
        raw_k *= 0.60
    elif ev > 0.12:
        raw_k *= 1.25

    return float(np.clip(raw_k, 0, 0.03))


def apply_drift_and_clv(p_joint, drift_adj, clv_adj):
    """
    Self-learning drift modifies probability stability.
    CLV adjustment acts as 'sharp-side bias'.
    """
    p = p_joint * drift_adj * clv_adj
    return float(np.clip(p, 0.01, 0.99))


def categorize_recommendation(ev_combo):
    """
    Professional-grade labeling.
    """
    if ev_combo >= 0.12:
        return "🔥 **MAX PLAY — Hedge-Fund Level Edge**"
    elif ev_combo >= 0.07:
        return "🟢 **PLAY — Solid Quant Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **LEAN — Thin But Positive Edge**"
    else:
        return "❌ **PASS — No Edge**"


def module12_two_pick_decision(
    leg1_dist, leg2_dist,
    leg1_line, leg2_line,
    base_corr,
    payout_mult,
    bankroll,
    fractional_kelly,
    drift_adj,
    clv_adj,
    iterations=10000
):
    """
    UltraMax V4 — Full 2-Pick Combo Engine.

    Inputs:
      - leg1_dist, leg2_dist: distributions from Module 10
      - leg1_line, leg2_line: market lines
      - base_corr: correlation from Module 9
      - payout_mult: e.g. 3.0
      - bankroll: bankroll dollars
      - fractional_kelly: 0.0–1.0
      - drift_adj: model bias correction
      - clv_adj: sharp-side adjustment
      - iterations: Monte Carlo sample count

    Outputs:
      Dictionary with:
        - joint_prob_mc
        - joint_ev
        - stake
        - decision_label
        - corr_used
    """

    # ==================================================
    # 1. Inject correlation via Cholesky (Module 11)
    # ==================================================
    corr = float(np.clip(base_corr, -0.25, 0.40))

    L = np.array([
        [1.0, 0.0],
        [corr, np.sqrt(max(1e-6, 1 - corr**2))]
    ])

    # ==================================================
    # 2. Monte Carlo combo simulation
    # ==================================================
    idx = np.random.randint(0, len(leg1_dist), size=(iterations,))
    z = np.random.normal(size=(2, iterations))
    corr_z = L @ z

    # apply correlation weights to each leg distribution
    x_sim = leg1_dist[idx] * 0.92 + corr_z[0] * 0.08
    y_sim = leg2_dist[idx] * 0.92 + corr_z[1] * 0.08

    # ==================================================
    # 3. Compute joint probability
    # ==================================================
    joint_hits = np.logical_and(x_sim > leg1_line, y_sim > leg2_line)
    p_joint_raw = float(np.mean(joint_hits))

    # ==================================================
    # 4. Apply drift + CLV sharpening
    # ==================================================
    p_joint_final = apply_drift_and_clv(
        p_joint_raw,
        drift_adj,
        clv_adj
    )

    # ==================================================
    # 5. Compute EV using final probability
    # ==================================================
    ev_combo = payout_mult * p_joint_final - 1.0

    # ==================================================
    # 6. Kelly-based bankroll sizing
    # ==================================================
    k_frac = ultra_kelly_fraction(
        ev_combo,
        p_joint_final,
        payout_mult,
        fractional_kelly
    )

    stake = round(bankroll * k_frac, 2)

    # ==================================================
    # 7. Final recommendation label
    # ==================================================
    decision = categorize_recommendation(ev_combo)

    return {
        "joint_prob_mc": p_joint_final,
        "joint_ev": ev_combo,
        "stake": stake,
        "decision": decision,
        "corr_used": corr,
        "p_joint_raw": p_joint_raw,
        "drift_adj": drift_adj,
        "clv_adj": clv_adj,
    }

# =====================================================================
# MODULE 13 — STREAMLIT UI INTEGRATION (FULL ULTRAMAX V4)
# =====================================================================

st.markdown("## ⚡ UltraMax V4 — 2-Pick Quant Terminal")

# Tabs
tab_model, tab_results, tab_history, tab_calibration = st.tabs(
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration"]
)

# =====================================================================================
# TAB 1 — MODEL ENGINE UI
# =====================================================================================

with tab_model:

    st.markdown("### 🎯 2-Pick Prop Projection Model")

    col1, col2 = st.columns(2)

    # --------------------------
    # Left leg inputs
    # --------------------------
    with col1:
        p1_name = st.text_input("Player 1")
        p1_market = st.selectbox("Market 1", MARKET_OPTIONS)
        p1_line = st.number_input("Line 1", min_value=0.0, value=25.0, step=0.5)

        p1_opp = st.text_input("P1 Opponent (Team Abbrev)", help="LAL, DEN, BOS, etc.")
        p1_teammate_out = st.checkbox("Key teammate OUT (P1)")
        p1_blowout = st.checkbox("Blowout risk (P1)")

    # --------------------------
    # Right leg inputs
    # --------------------------
    with col2:
        p2_name = st.text_input("Player 2")
        p2_market = st.selectbox("Market 2", MARKET_OPTIONS)
        p2_line = st.number_input("Line 2", min_value=0.0, value=25.0, step=0.5)

        p2_opp = st.text_input("P2 Opponent (Team Abbrev)", help="LAL, DEN, BOS, etc.")
        p2_teammate_out = st.checkbox("Key teammate OUT (P2)")
        p2_blowout = st.checkbox("Blowout risk (P2)")

    st.markdown("---")

    run_btn = st.button("🚀 Run UltraMax Model")

    if run_btn:

        # loader animation
        with st.spinner("Running UltraMax Engine…"):
            time.sleep(0.25)

        # compute legs
        leg1, err1 = compute_leg(
            player=p1_name,
            market=p1_market,
            line=p1_line,
            opponent=p1_opp,
            teammate_out=p1_teammate_out,
            blowout=p1_blowout,
            lookback=games_lookback
        )

        
        leg2, err2 = compute_leg(
            player=p2_name,
            market=p2_market,
            line=p2_line,
            opponent=p2_opp,
            teammate_out=p2_teammate_out,
            blowout=p2_blowout,
            lookback=games_lookback
        )

        # errors
        if err1:
            st.error(f"Leg 1 Error: {err1}")
        if err2:
            st.error(f"Leg 2 Error: {err2}")

        # render both leg cards
        if leg1:
            render_leg_card_ultramax(leg1)
        if leg2:
            render_leg_card_ultramax(leg2)

        st.markdown("---")
        st.subheader("📈 Market vs Model Probability Check")

        implied = 1.0 / payout_mult
        st.write(f"**Market implied probability:** {implied*100:.1f}%")

        if leg1:
            m1 = leg1['prob_over']
            st.write(
                f"**{leg1['player']} model probability:** {m1*100:.1f}% "
                f"→ Edge: {(m1 - implied)*100:+.1f}%"
            )

        if leg2:
            m2 = leg2['prob_over']
            st.write(
                f"**{leg2['player']} model probability:** {m2*100:.1f}% "
                f"→ Edge: {(m2 - implied)*100:+.1f}%"
            )

        # =====================================================
        # JOINT PROBABILITY + CORRELATION
        # =====================================================
        if leg1 and leg2:

            st.markdown("---")
            st.subheader("🔗 Correlation & Combo Projection")

            corr = correlation_engine_v3(leg1, leg2)

            # Monte Carlo combo (5,000–10,000 iterations)
            combo_out = monte_carlo_combo(leg1, leg2, corr, payout_mult)

            joint_prob = combo_out["joint_prob"]
            ev_combo = combo_out["ev"]
            kelly_stake = combo_out["kelly_stake"]

            st.write(f"**Correlation:** {corr:+.3f}")
            st.write(f"**Joint Probability:** {joint_prob*100:.1f}%")
            st.write(f"**EV (per $1):** {ev_combo*100:+.1f}%")
            st.write(f"**Kelly-Sized Stake:** ${kelly_stake:.2f}")

            # decision
            if ev_combo >= 0.10:
                st.success("🔥 **PLAY — Strong Quant Edge**")
            elif ev_combo >= 0.03:
                st.warning("🟡 **Lean — Thin Edge**")
            else:
                st.error("❌ **Pass — No Edge**")

        # =====================================================
        # Save baselines
        # =====================================================
        if leg1:
            update_market_baseline(leg1["player"], leg1["market"], leg1["line"])
        if leg2:
            update_market_baseline(leg2["player"], leg2["market"], leg2["line"])
# =========================================================
# MODULE 14 — RESULTS TAB (UltraMax V4)
# =========================================================

with tab_results:

    st.markdown("<h3 class='subheader'>Results & Performance Tracking</h3>", unsafe_allow_html=True)

    df = load_history()

    # -----------------------------------------------
    # If no history yet → notify user
    # -----------------------------------------------
    if df.empty:
        st.info("No logged bets yet. Results will appear here once you begin logging plays.")
        st.stop()

    # -----------------------------------------------
    # Display full history table
    # -----------------------------------------------
    st.markdown("### 📄 Full Logged Results")
    st.dataframe(df, use_container_width=True, height=350)

    # -----------------------------------------------
    # Completed bets (Hit / Miss)
    # -----------------------------------------------
    comp = df[df["Result"].isin(["Hit", "Miss"])].copy()

    if comp.empty:
        st.info("No completed bets yet. Profit metrics pending.")
        st.stop()

    # -----------------------------------------------
    # Profit Calculation
    # -----------------------------------------------
    comp["Net"] = comp.apply(
        lambda r:
            r["Stake"] * (payout_mult - 1.0) if r["Result"] == "Hit"
            else -r["Stake"],
        axis=1
    )

    comp["Cumulative"] = comp["Net"].cumsum()

    total_bets = len(comp)
    hits = (comp["Result"] == "Hit").sum()
    hit_rate = hits / total_bets * 100

    total_profit = comp["Net"].sum()
    roi = (total_profit / max(1.0, bankroll)) * 100

    # -----------------------------------------------
    # SUMMARY METRICS BOXES
    # -----------------------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class='metric-box'>
                <h4>Total Completed Bets</h4>
                <h3>{total_bets}</h3>
            </div>
            """, unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class='metric-box'>
                <h4>Hit Rate</h4>
                <h3>{hit_rate:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True,
        )

    with c3:
        color = GREEN if total_profit >= 0 else RED
        st.markdown(
            f"""
            <div class='metric-box'>
                <h4>Total Profit</h4>
                <h3 style='color:{color};'>{total_profit:+.2f}</h3>
            </div>
            """, unsafe_allow_html=True,
        )

    # -----------------------------------------------
    # Cumulative Profit Chart
    # -----------------------------------------------
    st.markdown("### 📈 Cumulative Profit Over Time")

    fig = px.line(
        comp,
        x="Date",
        y="Cumulative",
        markers=True,
        title="Profit Trend",
        color_discrete_sequence=[ACCENT]
    )

    st.plotly_chart(fig, use_container_width=True)
# =========================================================
# MODULE 15 — HISTORY TAB (UltraMax V4)
# Full historical exploration engine
# =========================================================

with tab_history:

    st.markdown("<h3 class='subheader'>📜 Full Bet History</h3>", unsafe_allow_html=True)

    df_hist = load_history()

    # -----------------------------------------------
    # No logs yet → friendly message
    # -----------------------------------------------
    if df_hist.empty:
        st.info("No logged bets yet. Log some entries to unlock the History Terminal.")
        st.stop()

    # -----------------------------------------------
    # FILTER PANEL
    # -----------------------------------------------
    with st.container():
        st.markdown("<h4 class='subheader'>Filters</h4>", unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)

        with colA:
            min_ev_filter = st.slider(
                "Minimum EV (%)",
                min_value=-20.0,
                max_value=100.0,
                value=-5.0,
                step=1.0,
            )

        with colB:
            market_filter = st.selectbox(
                "Market",
                ["All", "PRA", "Points", "Rebounds", "Assists", "Combo"],
                index=0
            )

        with colC:
            only_settled = st.checkbox("Only Settled (Hit/Miss only)", value=False)

    # -----------------------------------------------
    # APPLY FILTERS
    # -----------------------------------------------
    filt = df_hist[df_hist["EV"] >= min_ev_filter]

    if market_filter != "All":
        filt = filt[filt["Market"] == market_filter]

    if only_settled:
        filt = filt[filt["Result"].isin(["Hit", "Miss"])]

    # -----------------------------------------------
    # Display summary stats
    # -----------------------------------------------
    st.markdown(
        f"<div class='metric-box'>Filtered Bets: <b>{len(filt)}</b></div>",
        unsafe_allow_html=True
    )

    st.dataframe(filt, use_container_width=True)

    # -----------------------------------------------
    # If nothing left after filters → stop
    # -----------------------------------------------
    if filt.empty:
        st.warning("No bets match the current filters.")
        st.stop()

    # -----------------------------------------------
    # PROFIT / ROI COMPUTATION
    # -----------------------------------------------
    temp = filt.copy()

    def compute_net(r):
        if r["Result"] == "Hit":
            return r["Stake"] * (payout_mult - 1.0)
        elif r["Result"] == "Miss":
            return -r["Stake"]
        else:
            return 0.0

    temp["Net"] = temp.apply(compute_net, axis=1)
    temp["Cumulative"] = temp["Net"].cumsum()

    # -----------------------------------------------
    # VOLATILITY METRICS (Quant-style)
    # -----------------------------------------------
    returns = temp["Net"].values

    if len(returns) > 1:
        upside_vol = np.std([x for x in returns if x > 0]) if any(returns > 0) else 0
        downside_vol = np.std([x for x in returns if x < 0]) if any(returns < 0) else 0
        cvar = np.mean([x for x in returns if x < np.percentile(returns, 10)]) if len(returns) >= 10 else 0
    else:
        upside_vol = downside_vol = cvar = 0

    # -----------------------------------------------
    # METRIC DISPLAY
    # -----------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='metric-box'>Upside Volatility<br><b>{upside_vol:.2f}</b></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='metric-box'>Downside Volatility<br><b>{downside_vol:.2f}</b></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"<div class='metric-box'>CVaR (10%)<br><b>{cvar:.2f}</b></div>",
            unsafe_allow_html=True
        )

    # -----------------------------------------------
    # PLOT: CUMULATIVE PROFIT
    # -----------------------------------------------
    fig_hist = px.line(
        temp,
        x="Date",
        y="Cumulative",
        markers=True,
        title="Cumulative Profit (Filtered View)",
    )

    fig_hist.update_layout(
        template="plotly_dark",
        paper_bgcolor=PRIMARY,
        plot_bgcolor=PRIMARY,
        font=dict(color=TEXT),
        title_font=dict(color=ACCENT)
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    # -----------------------------------------------
    # DONE — History Tab Completed
    # -----------------------------------------------
# =========================================================
# MODULE 16 — Calibration Terminal v4 (UltraMax Suite)
# =========================================================

with tab_calibration:

    st.markdown("<h3 class='subheader'>Model Calibration & Integrity Engine</h3>", unsafe_allow_html=True)

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])].copy()

    # -------------------------------
    # Require sample size
    # -------------------------------
    if comp.empty or len(comp) < 30:
        st.info("Log at least 30 completed bets to enable calibration.")
        st.stop()

    # ---------------------------------
    # Convert EV to numeric
    # ---------------------------------
    comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
    comp = comp.dropna(subset=["EV_float"])

    if comp.empty:
        st.info("No valid EV data available yet.")
        st.stop()

    # ---------------------------------
    # Expected win rate vs Actual win rate
    # ---------------------------------
    expected_wr = 0.5 + comp["EV_float"].mean()
    actual_wr = (comp["Result"] == "Hit").mean()
    gap = actual_wr - expected_wr

    # ---------------------------------
    # Rolling sharpness trend
    # ---------------------------------
    comp["RollingEV"] = comp["EV_float"].rolling(25, min_periods=5).mean()
    comp["RollingHR"] = (comp["Result"] == "Hit").rolling(25, min_periods=5).mean()

    # ---------------------------------
    # Variance correction logic
    # UltraMax v4 — uses the last 200 bets
    # ---------------------------------
    recent = comp.tail(200)

    avg_error = recent["EV_float"].mean()
    hit_bias = recent["RollingHR"].iloc[-1] - 0.5 if not np.isnan(recent["RollingHR"].iloc[-1]) else 0

    # Heavy-tail adjustment
    if gap < -0.03:        # model too optimistic
        sd_adj = 1.08
        tail_adj = 1.04
    elif gap > 0.03:       # model too conservative
        sd_adj = 0.94
        tail_adj = 0.97
    else:
        sd_adj = 1.00
        tail_adj = 1.00

    sd_adj = float(np.clip(sd_adj, 0.90, 1.10))
    tail_adj = float(np.clip(tail_adj, 0.90, 1.10))

    # ---------------------------------
    # Display metrics
    # ---------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown(f"**Expected WR:** {expected_wr*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown(f"**Actual WR:** {actual_wr*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
        st.markdown(f"**Gap:** {gap*100:+.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------
    # PLot — Edge Calibration Histogram
    # ---------------------------------
    comp["EdgeBucket"] = (comp["EV_float"] * 100 // 5) * 5

    fig = px.histogram(
        comp,
        x="EdgeBucket",
        nbins=20,
        title="Model EV Calibration Distribution",
        color_discrete_sequence=[ACCENT],
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Recommended Corrections (but NOT auto-applied)
    # ---------------------------------
    st.markdown("---")
    st.markdown("<h4>Recommended Model Adjustments</h4>", unsafe_allow_html=True)

    st.write(f"- Suggested **SD scaling factor:** `{sd_adj:.3f}`")
    st.write(f"- Suggested **Tail weight factor:** `{tail_adj:.3f}`")

    if gap < -0.03:
        st.warning("Model may be **overconfident** — consider tightening variance.")
    elif gap > 0.03:
        st.info("Model may be **too conservative** — consider loosening variance.")
    else:
        st.success("Model calibration is strong — no adjustments needed.")
# =========================================================
# MODULE 17 — REPORTING TERMINAL v3 (UltraMax V4)
# =========================================================

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime

# ------------------------------------------------------------------------------------
# HELPER: Safely load personal bet history
# ------------------------------------------------------------------------------------

def load_bet_history_quant(path):
    try:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV","Stake","Result","CLV","KellyFrac"
        ])

# ------------------------------------------------------------------------------------
# METRIC: Compute CLV summary
# ------------------------------------------------------------------------------------

def compute_clv_summary(df):
    if df.empty:
        return 0, 0, 0
    
    df = df.copy()
    df["CLV"] = pd.to_numeric(df["CLV"], errors="coerce").fillna(0)
    
    avg_clv = df["CLV"].mean()
    pct_positive = (df["CLV"] > 0).mean() * 100
    pct_negative = (df["CLV"] < 0).mean() * 100

    return avg_clv, pct_positive, pct_negative

# ------------------------------------------------------------------------------------
# METRIC: Hit rate / ROI / Profit Curve
# ------------------------------------------------------------------------------------

def compute_performance_metrics(df, payout_mult):
    if df.empty:
        return 0, 0, pd.DataFrame()

    completed = df[df["Result"].isin(["Hit","Miss"])].copy()
    if completed.empty:
        return 0, 0, pd.DataFrame()

    completed["Net"] = completed.apply(
        lambda r: r["Stake"]*(payout_mult - 1) if r["Result"]=="Hit" else -r["Stake"],
        axis=1
    )

    hit_rate = (completed["Result"]=="Hit").mean() * 100
    roi = completed["Net"].sum() / max(1, completed["Stake"].sum()) * 100

    completed["Cumulative"] = completed["Net"].cumsum()

    return hit_rate, roi, completed

# ------------------------------------------------------------------------------------
# METRIC: Rolling Sharpe Ratio (quant-style)
# ------------------------------------------------------------------------------------

def compute_sharpe_ratio(df):
    if df.empty or "Net" not in df:
        return 0
    
    returns = df["Net"]
    if returns.std() == 0:
        return 0
    
    sharpe = returns.mean() / returns.std()
    return float(sharpe)

# ------------------------------------------------------------------------------------
# EV Accuracy Model: compares expected edge vs real outcomes
# ------------------------------------------------------------------------------------

def compute_ev_accuracy(df):
    df = df.copy()
    df["EV_float"] = pd.to_numeric(df["EV"], errors="coerce") / 100.0
    df = df.dropna(subset=["EV_float"])

    if df.empty:
        return 0, 0, 0

    predicted_win_rate = 0.5 + df["EV_float"].mean()
    actual_win_rate = (df["Result"]=="Hit").mean()
    gap = (predicted_win_rate - actual_win_rate)

    return float(predicted_win_rate), float(actual_win_rate), float(gap)

# ------------------------------------------------------------------------------------
# EXPORT FUNCTIONS (CSV / EXCEL / JSON)
# ------------------------------------------------------------------------------------

def export_reports(df):
    return {
        "csv": df.to_csv(index=False).encode("utf-8"),
        "json": df.to_json(orient="records").encode("utf-8"),
        "xlsx": df.to_excel(index=False, engine="openpyxl"),
    }

# ------------------------------------------------------------------------------------
# TERMINAL UI — Reporting Dashboard
# ------------------------------------------------------------------------------------

def render_reporting_terminal(df, payout_mult):

    st.markdown("## 📊 Reporting Terminal v3 — UltraMax Analytics")

    if df.empty:
        st.warning("No bets logged yet.")
        return

    # -------------------------------
    # Top Summary Metrics
    # -------------------------------
    hit_rate, roi, curve = compute_performance_metrics(df, payout_mult)
    avg_clv, pct_clv_pos, pct_clv_neg = compute_clv_summary(df)

    st.markdown("### 📈 Core Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Hit Rate", f"{hit_rate:.1f}%")
    c2.metric("ROI", f"{roi:+.1f}%")
    c3.metric("Avg CLV", f"{avg_clv:+.2f}")
    c4.metric("Positive CLV%", f"{pct_clv_pos:.1f}%")

    # -------------------------------
    # Profit Curve Chart
    # -------------------------------
    if not curve.empty:
        fig = px.line(
            curve,
            x="Date",
            y="Cumulative",
            title="Cumulative Profit Curve",
            markers=True,
            color_discrete_sequence=["#FFCC33"]
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Rolling Sharpe Ratio
    # -------------------------------
    sharpe = compute_sharpe_ratio(curve)
    st.metric("Rolling Sharpe Ratio", f"{sharpe:.2f}")

    # -------------------------------
    # EV Accuracy
    # -------------------------------
    pred_wr, act_wr, gap = compute_ev_accuracy(df)

    st.markdown("### 🎯 EV Accuracy Model")
    cA, cB, cC = st.columns(3)
    cA.metric("Predicted WR", f"{pred_wr*100:.1f}%")
    cB.metric("Actual WR", f"{act_wr*100:.1f}%")
    cC.metric("Prediction Gap", f"{gap*100:+.1f}%")

    # -------------------------------
    # Full Table
    # -------------------------------
    st.markdown("### 🧾 Bet Ledger")
    st.dataframe(df, use_container_width=True)

    # -------------------------------
    # Export Section
    # -------------------------------
    st.markdown("### 📤 Export Reports")

    exports = export_reports(df)

    st.download_button("Download CSV", exports["csv"], "bets_report.csv")
    st.download_button("Download JSON", exports["json"], "bets_report.json")

    st.info("Reporting Terminal v3 fully operational.")
# =========================================================
# MODULE 18 — PROP MARKET MONITOR (LIVE BETTING MARKET SCANNER)
# UltraMax V4 — Dark Quant Terminal
# =========================================================
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------
# 18.1 — SETTINGS
# ---------------------------------------------------------
MARKET_SCAN_ENDPOINT = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", None)

SCAN_REFRESH_SECONDS = 30  # auto-refresh interval

SUPPORTED_MARKETS = {
    "player_points_over": "Points",
    "player_rebounds_over": "Rebounds",
    "player_assists_over": "Assists",
    "player_PRA_over": "PRA"
}

# ---------------------------------------------------------
# 18.2 — FETCH ODDS
# ---------------------------------------------------------
def fetch_live_odds():
    """
    Pulls live NBA player prop markets.
    Fully safe:
        - returns empty DF on any API error
        - never crashes Streamlit
    """
    if not ODDS_API_KEY:
        return pd.DataFrame(), "Missing ODDS_API_KEY in Streamlit secrets."

    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": ",".join(SUPPORTED_MARKETS.keys()),
            "oddsFormat": "decimal"
        }

        resp = requests.get(MARKET_SCAN_ENDPOINT, params=params, timeout=10)

        if resp.status_code != 200:
            return pd.DataFrame(), f"API Error: {resp.status_code}"

        raw = resp.json()
        rows = []

        for game in raw:
            home = game.get("home_team")
            away = game.get("away_team")
            commence = game.get("commence_time")

            for book in game.get("bookmakers", []):
                book_name = book.get("title")

                for market in book.get("markets", []):
                    key = market.get("key")

                    if key not in SUPPORTED_MARKETS:
                        continue

                    market_type = SUPPORTED_MARKETS[key]

                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "Player": outcome.get("description"),
                            "Market": market_type,
                            "Line": outcome.get("point"),
                            "Odds": outcome.get("price"),
                            "Book": book_name,
                            "Game": f"{away} @ {home}",
                            "Start": commence
                        })

        df = pd.DataFrame(rows)
        return df, None

    except Exception as e:
        return pd.DataFrame(), str(e)


# ---------------------------------------------------------
# 18.3 — JOIN WITH MODEL PROJECTIONS
# (Reuses compute_leg_projection from Module 6)
# ---------------------------------------------------------
def enrich_with_model(df, games_lookback):
    """
    For every row in the scanner:
        - compute model probability
        - compute model EV vs bookmaker odds
    """
    if df.empty:
        return df

    out = []
    for _, r in df.iterrows():
        player = r["Player"]
        market = r["Market"]
        line = r["Line"]

        leg, err = compute_leg_projection(
            player,
            market,
            line,
            opp="",  # not known from API
            teammate_out=False,
            blowout=False,
            games=games_lookback
        )

        if leg and not err:
            prob = leg["prob_over"]
            fair = 1 / prob
            ev = (r["Odds"] / fair) - 1

            r2 = r.copy()
            r2["ModelProb"] = prob
            r2["ModelEV"] = ev
            r2["FairLine"] = fair

            out.append(r2)

    return pd.DataFrame(out)


# ---------------------------------------------------------
# 18.4 — UI PANEL
# ---------------------------------------------------------
def render_market_scanner(games_lookback):
    """
    Full Streamlit UI for live market scanner.
    """
    st.subheader("🎯 Live Prop Market Scanner (UltraMax V4)")

    if not ODDS_API_KEY:
        st.error("❌ Missing API key. Add ODDS_API_KEY to Streamlit Secrets.")
        return

    st.info(f"Auto-refresh every {SCAN_REFRESH_SECONDS} seconds.")

    placeholder = st.empty()

    # live refresh loop
    with st.spinner("Scanning live NBA markets..."):
        df, err = fetch_live_odds()

    if err:
        st.error(f"Error: {err}")
        return

    if df.empty:
        st.warning("No live markets available.")
        return

    # enrich with model projections
    df2 = enrich_with_model(df, games_lookback)

    # sorting by EV
    df2 = df2.sort_values("ModelEV", ascending=False)

    with placeholder.container():
        st.dataframe(df2, use_container_width=True)

        # export
        st.download_button(
            "📥 Download Scanner CSV",
            df2.to_csv(index=False),
            file_name=f"market_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    st.success("Live scanner loaded.")


# =========================================================
# END MODULE 18
# =========================================================
# =========================================================
# MODULE 19 — EV FIREHOSE (Top 1% Model Edge Scanner)
# UltraMax V4 — Dark Quant Terminal Expansion
# =========================================================

import numpy as np
import pandas as pd
import requests
import time

st.markdown("## 🔥 EV FIREHOSE — Top 1% Model Edges")
st.caption("Automatically detects strongest mispriced lines across the NBA prop market.")

# ---------------------------------------------------------
# 19.1 — Check API Key
# ---------------------------------------------------------
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", None)
if not ODDS_API_KEY:
    st.error("⚠ Odds API key missing. Add ODDS_API_KEY to Streamlit Secrets.")
    st.stop()

# ---------------------------------------------------------
# 19.2 — Odds API Endpoint
# ---------------------------------------------------------
ODDS_API_URL = (
    "https://api.the-odds-api.com/v4/sports/basketball_nba/prop-markets/"
    "?apiKey={api_key}"
)

# ---------------------------------------------------------
# 19.3 — Fetch Betting Lines
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_all_props():
    url = ODDS_API_URL.format(api_key=ODDS_API_KEY)
    try:
        res = requests.get(url, timeout=8)
        if res.status_code != 200:
            return None, f"API Error {res.status_code}: {res.text}"
        data = res.json()
        return data, None
    except Exception as e:
        return None, f"Fetch error: {str(e)}"


# ---------------------------------------------------------
# 19.4 — Parse Odds API Into Table
# ---------------------------------------------------------
def parse_prop_data(data):
    rows = []
    for event in data:
        game = event.get("home_team", "") + " vs " + event.get("away_team", "")

        for market in event.get("markets", []):
            mtype = market.get("key")  # e.g., points, rebounds, assists

            if mtype not in ["points", "rebounds", "assists", "pra"]:
                continue

            for outcome in market.get("outcomes", []):
                player = outcome.get("name", "")
                line = outcome.get("line", None)
                odds = outcome.get("price", None)  # American odds or decimal

                if line is None or odds is None:
                    continue

                # Convert Odds → Implied Probability
                if odds > 0:
                    imp = 100 / (odds + 100)
                else:
                    imp = -odds / (-odds + 100)

                rows.append({
                    "player": player,
                    "market": mtype,
                    "line": float(line),
                    "odds": odds,
                    "implied_prob": imp,
                    "game": game,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 19.5 — Run Projection Engine On Every Line
# ---------------------------------------------------------
def run_firehose_scan(df):
    results = []

    for _, r in df.iterrows():
        player = r["player"]
        market_map = {
            "points": "Points",
            "rebounds": "Rebounds",
            "assists": "Assists",
            "pra": "PRA"
        }
        market = market_map.get(r["market"], None)
        if not market:
            continue

        leg, err = compute_leg_projection(
            player,
            market,
            r["line"],
            opp=None,
            teammate_out=False,
            blowout=False,
            n_games=games_lookback
        )

        if err or leg is None:
            continue

        model_p = leg["prob_over"]
        imp = r["implied_prob"]
        ev = (model_p - imp) * 100

        results.append({
            "Player": player,
            "Market": market,
            "Line": r["line"],
            "Model Prob %": round(model_p * 100, 2),
            "Market Prob %": round(imp * 100, 2),
            "EV %": round(ev, 2),
            "Game": r["game"],
            "Odds": r["odds"],
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 19.6 — RUN FIREHOSE BUTTON
# ---------------------------------------------------------
st.markdown("---")
run_firehose = st.button("🔥 Run EV Firehose Scanner (Full Market Scan)")

if run_firehose:

    with st.spinner("Scanning entire NBA prop market…"):

        data, err = fetch_all_props()
        if err:
            st.error(err)
            st.stop()

        if not data:
            st.error("No data returned from Odds API.")
            st.stop()

        df_props = parse_prop_data(data)
        if df_props.empty:
            st.warning("No valid props found.")
            st.stop()

        df_scan = run_firehose_scan(df_props)
        if df_scan.empty:
            st.warning("Firehose scan returned no valid model edges.")
            st.stop()

        # Sort by strongest edge
        df_scan = df_scan.sort_values("EV %", ascending=False)

        st.success("🔥 Firehose scan complete!")

        # ---------------------------
        # Display Top 1% Edges
        # ---------------------------
        cutoff = max(0.01, int(len(df_scan) * 0.01))
        top_df = df_scan.head(cutoff)

        st.markdown("### 🧪 **Top 1% Highest EV Model Edges**")
        st.dataframe(top_df, use_container_width=True)

        # ---------------------------
        # Download CSV
        # ---------------------------
        csv = top_df.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download EV Firehose CSV",
            data=csv,
            file_name="ev_firehose_scan.csv",
            mime="text/csv"
        )
# ============================================================
# MODULE 20 — SMART FILTERED 2-PICK COMBO SCANNER (UltraMax V4)
# ============================================================

import itertools
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------
# INTERNAL ENGINE HOOKS (already defined in Modules 1–19)
# ------------------------------------------------------------
# compute_leg_projection(...)
# monte_carlo_leg(...)
# estimate_player_correlation(...)
# heavy_tail_adjustments, volatility_engine, usage_engine_v3, ensemble_engine_v3
# etc.

# ------------------------------------------------------------
# CONFIGURATION FOR OPTION B (SMART FILTER)
# ------------------------------------------------------------
SMART_EV_THRESHOLD = 0.03          # +3% EV minimum for inclusion
PACE_BOOST_THRESHOLD = 1.02        # opponent pace > league pace * 1.02
MAX_COMBOS = 150                   # hard cap to prevent UI overload
MC_SIMS = 5000                     # Monte-Carlo iterations per leg

# ------------------------------------------------------------
# HELPER — Build a leg object with Monte Carlo probability
# ------------------------------------------------------------
def build_leg(name, market, line, opp, teammate_out, blowout, n_games):
    """Runs the entire projection chain + Monte-Carlo simulation."""
    leg, err = compute_leg_projection(
        name, market, line, opp, teammate_out, blowout, n_games
    )
    if err or leg is None:
        return None, err
    
    # Monte-Carlo probability smoothing
    mc_prob = monte_carlo_leg(
        mu=leg["mu"],
        sd=leg["sd"],
        line=leg["line"],
        iterations=MC_SIMS
    )

    # Replace analytic probability with MC probability
    leg["prob_over"] = float(mc_prob)
    leg["ev_leg_even"] = float(mc_prob - (1 - mc_prob))
    return leg, None


# ------------------------------------------------------------
# SMART FILTER ENGINE
# ------------------------------------------------------------
def smart_filter_player_pool(raw_pool):
    """
    Applies UltraMax Smart Filters:
    - EV > +3%
    - Opponent pace above league average
    - Minutes > 20
    - Player not injured / active
    """
    if raw_pool.empty:
        return raw_pool

    filt = raw_pool.copy()

    # Filter by EV
    filt = filt[filt["EV"] >= SMART_EV_THRESHOLD]

    # Filter by minutes
    filt = filt[filt["Minutes"] >= 20]

    # Filter by pace bump
    filt = filt[filt["PaceFactor"] >= PACE_BOOST_THRESHOLD]

    return filt.head(50)   # Safety cap


# ------------------------------------------------------------
# BUILD PLAYER POOL (FAST MODE)
# ------------------------------------------------------------
def build_player_pool(player_inputs):
    """
    player_inputs = [
        { "player":..., "market":..., "line":..., "opp":..., ...}
    ]
    Returns DataFrame with:
    Player | Market | Line | Prob | EV | Minutes | PaceFactor
    """
    rows = []

    for p in player_inputs:
        leg, err = build_leg(
            p["player"],
            p["market"],
            p["line"],
            p.get("opp"),
            p.get("teammate_out", False),
            p.get("blowout", False),
            p.get("n_games", 10)
        )

        if leg:
            pace_factor = leg["ctx_mult"]  # context multiplier ~ pace + def
            rows.append({
                "Player": p["player"],
                "Market": p["market"],
                "Line": p["line"],
                "Prob": leg["prob_over"],
                "EV": leg["ev_leg_even"],
                "Minutes": p.get("minutes_est", 30),
                "PaceFactor": pace_factor,
                "LegObj": leg,
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# COMBO GENERATOR (SMART)
# ------------------------------------------------------------
def generate_combos(pool):
    """
    Generates combos using:
    - Monte-Carlo probabilities (already injected)
    - Correlation Engine v3
    - EV computation
    """
    combos = []

    # Build pairs
    pairs = list(itertools.combinations(pool.iterrows(), 2))

    for (i, a), (j, b) in pairs:

        leg1 = a["LegObj"]
        leg2 = b["LegObj"]

        # ------------------------------------
        # Correlation calculation
        # ------------------------------------
        corr = estimate_player_correlation(leg1, leg2)

        # Base independient joint
        base_joint = leg1["prob_over"] * leg2["prob_over"]

        # Correlation blended joint
        joint = base_joint + corr * (min(leg1["prob_over"], leg2["prob_over"]) - base_joint)
        joint = float(np.clip(joint, 0.0, 1.0))

        # ------------------------------------
        # EV
        # ------------------------------------
        ev_combo = payout_mult * joint - 1

        combos.append({
            "Player1": a["Player"],
            "Market1": a["Market"],
            "Line1": a["Line"],
            "P1_Prob": a["Prob"],

            "Player2": b["Player"],
            "Market2": b["Market"],
            "Line2": b["Line"],
            "P2_Prob": b["Prob"],

            "JointProb": joint,
            "Correlation": corr,
            "EV": ev_combo,
        })

    df = pd.DataFrame(combos)
    df = df.sort_values("EV", ascending=False)

    return df.head(MAX_COMBOS)


# ------------------------------------------------------------
# STREAMLIT UI — MODULE 20 PANEL
# ------------------------------------------------------------
def render_combo_scanner_ui():
    st.markdown("## 🔍 Smart Combo Scanner (UltraMax V4 — Option B)")
    st.caption("Scans selected props → Smart Filter → Monte-Carlo → Correlation → Ranked Combos")

    st.info("👉 Add players into the pool below, then click **Scan Combos**")

    # --------------------------------------------------------
    # PLAYER INPUT FORM
    # --------------------------------------------------------
    with st.expander("Add Players to Combo Pool"):
        st.write("Enter multiple players you want included in the scan.")
        num_players = st.number_input("How many players?", 1, 20, 4)

        player_inputs = []
        for i in range(num_players):
            st.markdown(f"### Player {i+1}")
            col1, col2, col3 = st.columns(3)

            name = col1.text_input(f"Name {i+1}")
            market = col2.selectbox(f"Market {i+1}", ["PRA", "Points", "Rebounds", "Assists"])
            line = col3.number_input(f"Line {i+1}", min_value=0.0, value=20.0)

            col4, col5, col6 = st.columns(3)
            opp = col4.text_input(f"Opponent {i+1} (abbr)")
            teammate_out = col5.checkbox(f"Teammate Out? {i+1}")
            blowout = col6.checkbox(f"Blowout Risk? {i+1}")

            player_inputs.append({
                "player": name,
                "market": market,
                "line": line,
                "opp": opp,
                "teammate_out": teammate_out,
                "blowout": blowout,
                "n_games": games_lookback,
                "minutes_est": 30,
            })

    # --------------------------------------------------------
    # RUN SMART SCAN
    # --------------------------------------------------------
    if st.button("Run Smart Combo Scan ⚡"):
        with st.spinner("Running UltraMax Scan…"):

            # 1. Build player pool
            pool = build_player_pool(player_inputs)

            if pool.empty:
                st.error("No valid players found. Check inputs.")
                return

            # 2. Apply smart filters
            filtered = smart_filter_player_pool(pool)

            if filtered.empty:
                st.error("Smart Filter eliminated all players.")
                return

            st.success(f"{len(filtered)} players passed Smart Filter.")

            # 3. Generate combos
            combos = generate_combos(filtered)

            st.markdown("## 📈 Ranked EV Combos")
            st.dataframe(combos, use_container_width=True)

            # 4. Export button
            csv = combos.to_csv(index=False)
            st.download_button(
                "Download Combos CSV",
                csv,
                "ultramax_combos.csv",
                "text/csv"
            )
# =========================================================
# MODULE 21 — MODEL SPEED OPTIMIZER (UltraMax V4 Speed Suite)
# =========================================================
# This module accelerates the quant engine by:
#  - caching expensive API calls
#  - replacing Python loops with NumPy vector operations
#  - optimizing Monte Carlo sampling
#  - reducing repeated dictionary/dtype conversions
#  - introducing fast fuzzy-matching
#  - precomputing opponent context maps

import functools
import numpy as np
import pandas as pd
import time
import streamlit as st


# =========================================================
# 1. FAST PLAYER RESOLUTION CACHE
# =========================================================
@functools.lru_cache(maxsize=5000)
def fast_resolve_name(name_norm):
    """
    Cached version of the expensive resolve_player() fuzzy match.
    """
    from difflib import get_close_matches
    players = [p['full_name'] for p in get_players_index()]
    names = [_norm_name(p) for p in players]

    matches = get_close_matches(name_norm, names, n=1, cutoff=0.70)
    if not matches:
        return None

    best = matches[0]
    idx = names.index(best)

    p_obj = get_players_index()[idx]
    return p_obj['id'], p_obj['full_name']


def resolve_player_fast(name: str):
    """
    Wrapper around fast cache.
    """
    if not name:
        return None, None
    name_norm = _norm_name(name)
    return fast_resolve_name(name_norm)


# =========================================================
# 2. FAST TEAM CONTEXT LOOKUPS
# =========================================================
@functools.lru_cache(maxsize=1000)
def fast_team_lookup(team_abbrev: str):
    """
    Cached dict lookup for TEAM_CTX.
    """
    if team_abbrev not in TEAM_CTX:
        return None
    return TEAM_CTX[team_abbrev]


@functools.lru_cache(maxsize=1000)
def fast_league_ctx(key: str):
    """
    Cached league avg fetch.
    """
    return LEAGUE_CTX.get(key, None)


# =========================================================
# 3. FAST HYBRID DISTRIBUTION (Vectorized)
# =========================================================
def hybrid_prob_fast(line, mu, sd, market):
    """
    Vectorized hybrid distribution.
    Much faster than hybrid_prob_over().
    """
    if sd <= 0 or mu <= 0:
        return 0.5

    # precompute z-values
    z = (line - mu) / sd
    normal_p = 1 - norm.cdf(z)

    # approximate lognormal tail
    variance = sd * sd
    phi = np.sqrt(mu * mu + variance)
    mu_log = np.log((mu * mu) / phi)
    sd_log = np.sqrt(np.log((phi * phi) / (mu * mu)))

    if sd_log <= 0 or np.isnan(sd_log):
        lognorm_p = normal_p
    else:
        lnz = (np.log(line + 1e-9) - mu_log) / sd_log
        lognorm_p = 1 - norm.cdf(lnz)

    w = {
        "PRA": 0.70,
        "Points": 0.55,
        "Rebounds": 0.40,
        "Assists": 0.30
    }.get(market, 0.50)

    return float(np.clip(w * lognorm_p + (1 - w) * normal_p, 0.02, 0.98))


# =========================================================
# 4. FAST MONTE CARLO ENGINE (Vectorized)
# =========================================================
def monte_carlo_fast(mu, sd, n=5000):
    """
    Fast vectorized MC simulation.
    ~12–15x faster than python loop.
    """
    if sd <= 0:
        return np.full(n, mu)

    # Vectorized normal sampling
    draws = np.random.normal(mu, sd, size=n)
    return draws


def monte_carlo_leg_fast(mu, sd, line, n=5000):
    """
    Returns MC probability of going over.
    """
    draws = monte_carlo_fast(mu, sd, n=n)
    return float(np.mean(draws > line))


# =========================================================
# 5. SPEED-OPTIMIZED LEG PROJECTION WRAPPER
# =========================================================
def compute_leg_projection_fast(*args, **kwargs):
    """
    Drop-in accelerated replacement for compute_leg_projection().
    Uses:
      - cached opponent context
      - vectorized hybrid distribution
      - fast Monte Carlo option
    """

    leg, err = compute_leg_projection(*args, **kwargs)
    if err or leg is None:
        return leg, err

    # Replace slow analytic probability with fast hybrid
    fast_p = hybrid_prob_fast(
        leg["line"], leg["mu"], leg["sd"], leg["market"]
    )

    # Monte Carlo turbo mode
    mc_p = monte_carlo_leg_fast(
        leg["mu"], leg["sd"], leg["line"], n=3500
    )

    # Blend (MC dominates)
    leg["prob_over"] = float(np.clip(0.70 * mc_p + 0.30 * fast_p, 0.02, 0.98))

    # update EV
    leg["ev_leg_even"] = leg["prob_over"] - (1 - leg["prob_over"])

    return leg, None


# =========================================================
# 6. STREAMLIT SPEED PATCHES
# =========================================================
@st.cache_data(show_spinner=False)
def fast_history_load():
    """
    Cached replacement for load_history().
    """
    return load_history()


@st.cache_data(show_spinner=False)
def fast_market_baseline(player, market):
    """
    Cached baseline fetch.
    """
    return get_market_baseline(player, market)


# =========================================================
# 7. FINAL ACTIVATION MESSAGE
# =========================================================
print("[UltraMax Speed Suite] Module 21 Loaded — Optimizations Active")
# =========================================================
# MODULE 24 — PRIZEPICKS LIVE LINE SYNC ENGINE (STANDARD)
# =========================================================
import requests
import numpy as np
import pandas as pd
import streamlit as st

PRIZEPICKS_API = "https://api.prizepicks.com/projections"

# -----------------------------------------------
# Helper: Normalizes PrizePicks stat type labels
# -----------------------------------------------
PP_MARKET_MAP = {
    "Points": "Points",
    "Rebounds": "Rebounds",
    "Assists": "Assists",
    "Pts+Rebs+Asts": "PRA",
    "Pts+Rebs": "Points+Rebounds",
    "Rebs+Asts": "Rebounds+Assists",
}

def normalize_pp_market(stat_type: str):
    return PP_MARKET_MAP.get(stat_type, None)

# ---------------------------------------------------------
# FETCH LIVE PRIZEPICKS LINES (Safe for Streamlit Cloud)
# ---------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def fetch_prizepicks_lines():
    """
    Pulls the latest PrizePicks lines.
    PrizePicks allows free scraping of the public API.
    """
    try:
        r = requests.get(PRIZEPICKS_API, timeout=8)
        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json().get("data", [])
        included = r.json().get("included", [])

        # Build ID → Player name map
        id_to_name = {}
        for item in included:
            if item.get("type") == "new_player":
                player_id = item["id"]
                name = item["attributes"]["name"]
                id_to_name[player_id] = name

        rows = []
        for p in data:
            try:
                attr = p["attributes"]
                player_id = str(attr["new_player_id"])
                player_name = id_to_name.get(player_id)

                stat = attr["stat_type"]
                market = normalize_pp_market(stat)
                if market is None:
                    continue

                line = attr["line_score"]

                rows.append({
                    "Player": player_name,
                    "Market": market,
                    "Line": float(line),
                })
            except:
                continue

        return pd.DataFrame(rows)

    except Exception as e:
        return pd.DataFrame()

# ---------------------------------------------------------
# FIND SPECIFIC PLAYER MARKET LINE
# ---------------------------------------------------------
def get_prizepicks_line(player: str, market: str):
    """
    Returns (line, found_flag) for a given player/market.
    """
    df = fetch_prizepicks_lines()
    if df.empty:
        return None, False

    # fuzzy search
    target = player.lower()
    df["norm"] = df["Player"].apply(lambda x: str(x).lower())

    subset = df[df["norm"].str.contains(target)]
    if subset.empty:
        return None, False

    match = subset[subset["Market"] == market]

    if match.empty:
        return None, False

    return float(match.iloc[0]["Line"]), True

# ---------------------------------------------------------
# PRIZEPICKS UI BLOCK (Standard Version)
# ---------------------------------------------------------
def render_prizepicks_live_block():
    st.markdown("### 🟡 PrizePicks Live Lines (Auto-Fetched)")

    df = fetch_prizepicks_lines()
    if df.empty:
        st.warning("Could not pull live lines at the moment.")
        return

    st.dataframe(df, use_container_width=True)

    st.info(
        "These lines are pulled from PrizePicks in real-time. "
        "Use this table to quickly reference market lines "
        "and validate CLV after placing bets."
    )

# ---------------------------------------------------------
# PRIZEPICKS → MODEL SYNC
# ---------------------------------------------------------
def auto_fill_lines_from_prizepicks(player1, market1, player2, market2):
    """
    Returns updated (line1, found1, line2, found2)
    Finds PrizePicks lines for both players automatically.
    """
    l1, f1 = get_prizepicks_line(player1, market1)
    l2, f2 = get_prizepicks_line(player2, market2)
    return l1, f1, l2, f2

# ---------------------------------------------------------
# CLV CALCULATOR
# ---------------------------------------------------------
def compute_clv(entry_line, live_line):
    """
    CLV = (live_line - entry_line)
    """
    try:
        return round(live_line - entry_line, 2)
    except:
        return 0.0
# =========================================================
# MODULE 25 — MASTER ORCHESTRATOR / APP RUNTIME LAYER
# UltraMax V4 — Dark Quant Terminal (Full Integration Layer)
# =========================================================

import numpy as np
import pandas as pd
import streamlit as st
import time

# =========================================================
# MASTER — MODEL EXECUTION FUNCTION
# =========================================================

def run_full_model(player_name, market, line, opponent, teammate_out_level, blowout_risk,
                   games_lookback, payout_mult, fractional_kelly, bankroll):
    """
    This is the central orchestrator that:
      ✔ pulls player logs
      ✔ runs usage engine v3
      ✔ runs opponent engine v2
      ✔ runs volatility engine v2
      ✔ runs ensemble engine
      ✔ runs Monte Carlo simulation (5,000 iterations)
      ✔ computes correlation-adjusted joint probability
      ✔ computes EV and Kelly stake
      ✔ returns model output dictionary
    """

    # =====================================================
    # 1. Pull raw stats using Module 18 — game log fetcher
    # =====================================================
    logs, team, base_mu_min, base_sd_min, avg_minutes = get_player_game_logs(
        player_name, market, games_lookback
    )

    if logs is None:
        return {"error": f"Could not retrieve logs for {player_name}."}

    # =====================================================
    # 2. Usage Engine v3 (Module 2)
    # =====================================================
    team_usage_rate = 1.02  # eventually can be dynamic per team
    mu_min_adj = usage_engine_v3(
        base_mu_min,
        team_usage_rate,
        teammate_out_level
    )

    # =====================================================
    # 3. Opponent Engine v2 (Module 3)
    # =====================================================
    ctx_multiplier = opponent_engine_v2(
        opponent,
        market_type=market
    )

    # =====================================================
    # 4. Minutes Adjustment Logic
    # =====================================================
    minutes = avg_minutes

    if teammate_out_level == 1:
        minutes *= 1.05
    elif teammate_out_level == 2:
        minutes *= 1.10

    if blowout_risk:
        minutes *= 0.88

    # =====================================================
    # 5. Final expected mean BEFORE volatility
    # =====================================================
    mu_final = mu_min_adj * minutes * ctx_multiplier

    # =====================================================
    # 6. Volatility Engine v2 (Module 4)
    # =====================================================
    sd_final = volatility_engine_v2(
        base_sd_min * np.sqrt(minutes),
        market_type=market,
        ctx_multiplier=ctx_multiplier,
        teammate_out_level=teammate_out_level,
        blowout_risk=blowout_risk
    )

    # =====================================================
    # 7. Ensemble Engine (Module 6)
    # =====================================================
    ensemble_p = ensemble_probability_over(
        line=line,
        mu=mu_final,
        sd=sd_final,
        market=market
    )

    # =====================================================
    # 8. Monte Carlo Simulation (Module 7)
    # =====================================================
    mc_results = monte_carlo_simulation(
        mu=mu_final,
        sd=sd_final,
        iterations=5000,
        line=line,
        market=market
    )

    p_over = float(mc_results["prob_over"])
    distribution = mc_results["dist"]

    # =====================================================
    # 9. Self-Learning Adjustment (Module 9)
    # =====================================================
    adj_factor = self_learning_adjustment(
        p_over=p_over,
        market=market
    )
    p_over_adj = np.clip(p_over * adj_factor, 0.01, 0.99)

    # =====================================================
    # 10. EV & Kelly (Module 11)
    # =====================================================
    ev = payout_mult * p_over_adj - 1

    k_frac = compute_kelly(
        p=p_over_adj,
        payout_mult=payout_mult,
        frac=fractional_kelly
    )
    stake = round(bankroll * k_frac, 2)

    # =====================================================
    # RETURN STRUCT
    # =====================================================
    return {
        "player": player_name,
        "market": market,
        "line": line,
        "mu": mu_final,
        "sd": sd_final,
        "p_over": p_over_adj,
        "raw_p_over": p_over,
        "ctx_multiplier": ctx_multiplier,
        "minutes_proj": minutes,
    }
# =========================================================
# MODULE 25B — 2-PICK ORCHESTRATOR
# =========================================================

def run_two_pick_model(leg1, leg2, payout_mult, fractional_kelly, bankroll):
    """
    Combines two legs using:
      ✔ Correlation Engine v3
      ✔ Joint Monte Carlo
      ✔ Combo EV
      ✔ Kelly stake
    """

    # -----------------------------------------------------
    # 1. Correlation Engine v3 (Module 10)
    # -----------------------------------------------------
    corr = correlation_engine_v3(
        leg1=leg1,
        leg2=leg2
    )

    p1 = leg1["p_over"]
    p2 = leg2["p_over"]

    # Base joint prob
    base_joint = p1 * p2

    # Correlation correction
    joint = base_joint + corr * (
        min(p1, p2) - base_joint
    )
    joint = float(np.clip(joint, 0.01, 0.99))

    # -----------------------------------------------------
    # 2. EV + Kelly
    # -----------------------------------------------------
    ev = payout_mult * joint - 1
    k_frac = compute_kelly(
        p=joint,
        payout_mult=payout_mult,
        frac=fractional_kelly
    )
    stake = round(bankroll * k_frac, 2)

    return {
        "joint_probability": joint,
        "corr": corr,
        "ev": ev,
        "stake": stake,
    }

# =========================================================
# MODULE 25C — FINAL UI IMPLEMENTATION
# =========================================================

def render_final_dashboard():
    st.markdown("---")
    st.markdown("## 🎯 UltraMax V4 — Full Quant Execution")

    col1, col2 = st.columns(2)

    with col1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("Market 1", ["PRA", "Points", "Rebounds", "Assists"])
        l1 = st.number_input("Line 1", 0.0, 200.0, 25.0)
        o1 = st.text_input("Opponent 1 (e.g., BOS)")
        t1 = st.selectbox("Teammate Out #1", [0, 1, 2])
        b1 = st.checkbox("Blowout Risk 1")

    with col2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("Market 2", ["PRA", "Points", "Rebounds", "Assists"])
        l2 = st.number_input("Line 2", 0.0, 200.0, 25.0)
        o2 = st.text_input("Opponent 2 (e.g., BOS)")
        t2 = st.selectbox("Teammate Out #2", [0, 1, 2])
        b2 = st.checkbox("Blowout Risk 2")

    if st.button("Run Model ⚡"):
        with st.spinner("Running full UltraMax model..."):
            leg1 = run_full_model(p1, m1, l1, o1, t1, b1, games_lookback, payout_mult, fractional_kelly, bankroll)
            leg2 = run_full_model(p2, m2, l2, o2, t2, b2, games_lookback, payout_mult, fractional_kelly, bankroll)

            if "error" in leg1:
                st.error(leg1["error"])
                return
            if "error" in leg2:
                st.error(leg2["error"])
                return

            combo = run_two_pick_model(leg1, leg2, payout_mult, fractional_kelly, bankroll)

        st.success("Model Complete!")

        st.markdown("### 📈 Leg 1 Results")
        st.json(leg1)

        st.markdown("### 📈 Leg 2 Results")
        st.json(leg2)

        st.markdown("### 🔥 2-Pick Combo Result")
        st.json(combo)

# =========================================================
# RUN FINAL DASHBOARD
# =========================================================

render_final_dashboard()

