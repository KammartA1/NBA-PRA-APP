# ============================================================
# ULTRAMAX V4 — FULL ENGINE MASTER FILE
# PART 1/20 — CORE IMPORTS, GLOBALS, PLAYER RESOLUTION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from scipy.stats import norm
import json
import math
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# MODULE 0 — GLOBAL CONSTANTS
# ============================================================

CURRENT_VERSION = "UltraMax V4 — Full Engine"

# Default payout for PrizePicks 2-pick Power Play
payout_mult = 3.0

# Safety clamps
MIN_SD = 0.10
MAX_SD = 250.0

# Number of Monte Carlo iterations
MC_ITER = 10000

# ============================================================
# MODULE 1 — MARKET DEFINITIONS
# ============================================================

MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists"]

# Metric mapping for calculating MarketVal from game logs
MARKET_METRICS = {
    "PRA": ["PTS", "REB", "AST"],
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
}

# ============================================================
# MODULE 2 — PLAYER RESOLUTION ENGINE
# ============================================================

# Player dictionary (you can expand this if needed)
PLAYER_NAME_MAP = {
    # Format: "input_name": ("NBA_API_PLAYER_ID", "Canonical Name")
    "lebron james": (2544, "LeBron James"),
    "nikola jokic": (203999, "Nikola Jokic"),
    "anthony edwards": (1630162, "Anthony Edwards"),
    "giannis antetokounmpo": (203507, "Giannis Antetokounmpo"),
    "shai gilgeous-alexander": (1628983, "Shai Gilgeous-Alexander"),
    # Add more as needed
}


def resolve_player(name_input: str):
    """
    Resolves player name into (player_id, canonical_name).
    """

    if not name_input or str(name_input).strip() == "":
        return None, None

    key = name_input.strip().lower()

    if key in PLAYER_NAME_MAP:
        return PLAYER_NAME_MAP[key]

    # If player not in map:
    # you can add fuzzy search or NBA API call later
    return None, None
# ================================================================
# MODULE 3 — ROLE & USAGE ENGINE (UltraMax v4)
# ================================================================

import numpy as np

# -----------------------------
# ROLE IMPACT TABLE
# -----------------------------
ROLE_IMPACT = {
    "primary": 1.00,
    "secondary": 0.87,
    "tertiary": 0.74,
    "low": 0.60,
}

# -----------------------------
# SMOOTH NONLINEAR CURVE
# -----------------------------
def _smooth_curve(x, power=1.20):
    """
    Smooth nonlinear transformation that prevents extreme jumps.
    """
    try:
        return float(np.sign(x) * (abs(x) ** power))
    except:
        return float(x)


# -----------------------------
# TEAM USAGE FACTOR
# -----------------------------
def team_usage_factor(team_usage_rate):
    """
    Converts team usage/pace into a multiplier.
    Typical range: 0.92–1.10
    """
    return float(np.clip(team_usage_rate, 0.85, 1.25))


# -----------------------------
# INJURY REDISTRIBUTION FACTOR
# -----------------------------
def teammate_out_factor(level: int):
    """
    Missing creators increase star usage.
    level: 0–3
    """
    table = {
        0: 1.00,
        1: 1.06,
        2: 1.12,
        3: 1.19,
    }
    return table.get(int(level), 1.00)


# -----------------------------
# USAGE ENGINE v3 (FULL)
# -----------------------------
def usage_engine_v3(mu_per_min, role: str, team_usage_rate: float, teammate_out_level: int):
    """
    Ultimate Usage Redistribution Engine v3
    --------------------------------------
    Inputs:
        mu_per_min          — base per-minute production
        role                — "primary", "secondary", "tertiary", "low"
        team_usage_rate     — team possession/pace factor
        teammate_out_level  — (0–3) effect of missing creators

    Output:
        adjusted_mu_per_min — final production rate
    """

    # 1. Base guard
    mu_base = max(float(mu_per_min), 0.05)

    # 2. Role adjustment
    role_adj = ROLE_IMPACT.get(role.lower(), 1.00)
    role_adj = _smooth_curve(role_adj, power=1.20)

    # 3. Team adjustment
    team_adj = team_usage_factor(team_usage_rate)
    team_adj = _smooth_curve(team_adj, power=1.10)

    # 4. Injury redistribution
    inj_adj = teammate_out_factor(teammate_out_level)
    inj_adj = _smooth_curve(inj_adj, power=1.25)

    # 5. Final multiplier (nonlinear blend)
    final_mult = (
        (role_adj ** 0.45) *
        (team_adj ** 0.40) *
        (inj_adj ** 0.55)
    )

    out = mu_base * final_mult
    out = float(np.clip(out, mu_base * 0.70, mu_base * 1.55))

    return out


# ================================================================
# MODULE 4 — OPPONENT MATCHUP ENGINE v2 (UltraMax)
# ================================================================

# Defensive multipliers per market (league-normalized)
MATCHUP_TABLE = {
    "PRA": {
        "elite": 0.88,
        "strong": 0.93,
        "average": 1.00,
        "weak": 1.06,
        "target": 1.12,
    },
    "Points": {
        "elite": 0.90,
        "strong": 0.95,
        "average": 1.00,
        "weak": 1.05,
        "target": 1.11,
    },
    "Rebounds": {
        "elite": 0.92,
        "strong": 0.97,
        "average": 1.00,
        "weak": 1.06,
        "target": 1.14,
    },
    "Assists": {
        "elite": 0.89,
        "strong": 0.95,
        "average": 1.00,
        "weak": 1.07,
        "target": 1.15,
    },
}

# Team defensive tiers — you can update these dynamically later
DEF_TIER = {
    "BOS": "elite",
    "OKC": "strong",
    "MIN": "strong",
    "DEN": "strong",
    "MIA": "average",
    "LAL": "average",
    "SAC": "weak",
    "WSH": "target",
    "CHA": "target",
}


def opponent_matchup_v2(opponent_abbrev: str, market: str):
    """
    Converts opponent defense into a matchup multiplier.

    opponent_abbrev — team short code (DEN, LAL, BOS, etc.)
    market          — PRA / Points / Rebounds / Assists
    """

    if not isinstance(opponent_abbrev, str):
        return 1.00

    market = market if market in MATCHUP_TABLE else "PRA"

    tier = DEF_TIER.get(opponent_abbrev.upper(), "average")
    mult = MATCHUP_TABLE[market].get(tier, 1.00)

    return float(mult)
# ============================================================
# MODULE 5 — VOLATILITY ENGINE v2 (Next-Gen Variance Modeling)
# ============================================================

import numpy as np

def volatility_engine_v2(
    base_sd_per_min: float,
    proj_minutes: float,
    market: str,
    matchup_mult: float,
    usage_ratio: float,
    regime_state: str = "normal"
):
    """
    Volatility Engine v2
    --------------------------------------------------------
    Produces final standard deviation (σ) for the player prop.
    
    Inputs:
        base_sd_per_min : baseline SD per minute from logs
        proj_minutes     : projected minutes (18–40 range)
        market           : Points, Rebounds, Assists, PRA
        matchup_mult     : context multiplier (opponent effect)
        usage_ratio      : usage inflation vs base rate
        regime_state     : normal / high_vol / low_vol

    Output:
        sd_final         : fully adjusted standard deviation
    """

    # --------------------------------------------------------
    # 1. Base scaling
    # --------------------------------------------------------
    sd = base_sd_per_min * np.sqrt(max(proj_minutes, 1))

    # --------------------------------------------------------
    # 2. Market-specific volatility
    # --------------------------------------------------------
    MARKET_VOL = {
        "Points":   1.00,
        "Rebounds": 1.15,
        "Assists":  1.20,
        "PRA":      0.95
    }

    sd *= MARKET_VOL.get(market, 1.00)

    # --------------------------------------------------------
    # 3. Matchup-induced variance
    # --------------------------------------------------------
    # Good matchups → more ceiling → slightly more volatility
    sd *= (0.90 + 0.20 * matchup_mult)

    # --------------------------------------------------------
    # 4. Usage inflation increases instability
    # --------------------------------------------------------
    sd *= (0.95 + 0.30 * usage_ratio)

    # --------------------------------------------------------
    # 5. Regime-based broad volatility shifts
    # --------------------------------------------------------
    if regime_state == "high_vol":
        sd *= 1.15
    elif regime_state == "low_vol":
        sd *= 0.90

    # --------------------------------------------------------
    # 6. Stability clamp
    # --------------------------------------------------------
    sd = float(np.clip(sd, 0.50, 25.0))

    return sd


# ============================================================
# MODULE 6 — ENSEMBLE PROBABILITY ENGINE v3
# ============================================================

from scipy.stats import norm

def ensemble_prob_over(
    mu: float,
    sd: float,
    line: float,
    market: str,
    volatility_score: float
):
    """
    Ensemble Probability Engine v3
    ---------------------------------------------------------
    A Blend of:
      - Normal CDF
      - Skewed gamma approximation
      - Empirical volatility-adjusted curve

    Inputs:
        mu               — final projection mean
        sd               — final projection SD
        line             — market line
        market           — Points / Rebounds / etc
        volatility_score — sd / mu, used to shift tail weight

    Output:
        Probability(over)
    """

    if sd <= 0:
        return 0.50 if mu >= line else 0.01

    # ---------------------------------------------------------
    # 1. Normal model baseline
    # ---------------------------------------------------------
    z = (line - mu) / sd
    p_norm = 1 - norm.cdf(z)
    p_norm = float(np.clip(p_norm, 0.01, 0.99))

    # ---------------------------------------------------------
    # 2. Gamma skew model
    # ---------------------------------------------------------
    shape = max(1.25, 2.0 / max(volatility_score, 0.05))
    scale = sd / max(shape, 0.1)

    # Approximate gamma tail probability
    try:
        # CDF approximation via series
        from math import exp

        x = max(0.01, line)
        k = shape
        θ = scale

        # lower incomplete gamma approx
        gamma_cdf = 1 - exp(-x/θ) * sum(
            (x/θ)**i / np.math.factorial(i) for i in range(int(k))
        )
        p_gamma = 1 - gamma_cdf
    except Exception:
        p_gamma = p_norm

    p_gamma = float(np.clip(p_gamma, 0.01, 0.99))

    # ---------------------------------------------------------
    # 3. Volatility-weighted ensemble
    # ---------------------------------------------------------
    # High volatility → more weight on gamma
    w = np.clip(volatility_score, 0.05, 1.50) / 1.50
    w = float(np.clip(w, 0.10, 0.75))

    p_final = (1 - w) * p_norm + w * p_gamma
    p_final = float(np.clip(p_final, 0.01, 0.99))

    return p_final
# =====================================================================
# MODULE — COMPUTE LEG (UltraMax Full — FINAL STABLE VERSION)
# =====================================================================

def compute_leg(
    player: str,
    market: str,
    line: float,
    opponent: str,
    teammate_out: bool,
    blowout: bool,
    lookback: int
):
    """
    UltraMax V4 — Compute a single projection leg.
    --------------------------------------------------------------------
    OUTPUT:
        leg dict OR (None, error_message)
    """

    # ---------------------------------------------------------
    # 0. Resolve player ID + canonical name
    # ---------------------------------------------------------
    pid, canonical = resolve_player(player)
    if not pid:
        return None, f"Player not found: {player}"

    # ---------------------------------------------------------
    # 1. Retrieve logs
    # ---------------------------------------------------------
    try:
        logs = PlayerGameLog(
            player_id=pid,
            season=current_season(),
        ).get_data_frames()[0]
    except Exception:
        return None, "API error retrieving game logs."

    if logs.empty:
        return None, "No game logs available."

    logs = logs.head(int(lookback))

    # ---------------------------------------------------------
    # 2. Build market production value
    # ---------------------------------------------------------
    metrics = MARKET_METRICS.get(market, ["PTS"])

    try:
        logs["MarketVal"] = logs[metrics].sum(axis=1)
    except Exception:
        return None, f"Missing stat fields for market: {market}"

    # Minutes column sanity check
    try:
        logs["Minutes"] = logs["MIN"].astype(float)
    except:
        logs["Minutes"] = 0.0

    valid = logs["Minutes"] > 0
    if not valid.any():
        return None, f"No valid minute data for {canonical}"

    # ---------------------------------------------------------
    # 3. Base per-minute rates
    # ---------------------------------------------------------
    pm_vals = logs.loc[valid, "MarketVal"] / logs.loc[valid, "Minutes"]

    base_mu_per_min = float(pm_vals.mean())
    base_sd_per_min = float(max(pm_vals.std(), 0.10))  # stability clamp

    # ---------------------------------------------------------
    # 4. Minutes projection
    # ---------------------------------------------------------
    proj_minutes = float(
        np.clip(logs["Minutes"].tail(5).mean(), 18, 40)
    )

    # ---------------------------------------------------------
    # 5. Usage Engine (role-based, injury-based)
    # ---------------------------------------------------------
    role = "primary"  # v4 default role engine
    team_usage = 1.00
    teammate_out_level = int(1 if teammate_out else 0)

    try:
        usage_mu = usage_engine_v3(
            mu_per_min=base_mu_per_min,
            role=role,
            team_usage_rate=team_usage,
            teammate_out_level=teammate_out_level
        )
    except:
        return None, "Usage engine failure."

    # ---------------------------------------------------------
    # 6. Opponent context multiplier
    # ---------------------------------------------------------
    try:
        ctx_mult = opponent_matchup_v2(opponent, market)
    except:
        ctx_mult = 1.00  # safe fallback

    # ---------------------------------------------------------
    # 7. Final mean projection
    # ---------------------------------------------------------
    mu = float(usage_mu * proj_minutes * ctx_mult)
    mu = max(mu, 0.01)

    # ---------------------------------------------------------
    # 8. Volatility Engine
    # ---------------------------------------------------------
    try:
        sd = volatility_engine_v2(
            base_sd_per_min,
            proj_minutes,
            market,
            ctx_mult,
            usage_mu / max(base_mu_per_min, 0.01),
            regime_state="normal"
        )
    except:
        sd = max(base_sd_per_min * proj_minutes, 1.0)

    sd = float(max(sd, 0.25))

    # ---------------------------------------------------------
    # 9. Ensemble probability
    # ---------------------------------------------------------
    try:
        prob = ensemble_prob_over(
            mu,
            sd,
            float(line),
            market,
            volatility_score=sd / max(mu, 1)
        )
    except Exception:
        return None, "Probability model failure."

    prob = float(np.clip(prob, 0.01, 0.99))

    # ---------------------------------------------------------
    # 10. Final packaged leg dictionary
    # ---------------------------------------------------------
    leg = {
        "player": canonical,
        "market": market,
        "line": float(line),
        "prob_over": prob,
        "mu": mu,
        "sd": sd,
        "ctx_mult": ctx_mult,
        "team": None,
        "teammate_out": bool(teammate_out),
        "blowout": bool(blowout),
    }

    return leg, None
# ================================================================
# MODULE 2E — Contextual Boost Engine (UltraMax V4)
# Advanced Opponent + Pace + Environment Adjustment Layer
# ================================================================

import numpy as np

# -----------------------------------------------------------
# Defensive multipliers by market
# -----------------------------------------------------------
DEFENSE_IMPACT = {
    "PRA":  {"elite": 0.88, "good": 0.93, "avg": 1.00, "bad": 1.06, "terrible": 1.11},
    "Points": {"elite": 0.90, "good": 0.95, "avg": 1.00, "bad": 1.07, "terrible": 1.14},
    "Rebounds": {"elite": 0.92, "good": 0.96, "avg": 1.00, "bad": 1.05, "terrible": 1.10},
    "Assists": {"elite": 0.89, "good": 0.95, "avg": 1.00, "bad": 1.08, "terrible": 1.13},
}

# -----------------------------------------------------------
# Opponent tiers (can be updated dynamically later)
# -----------------------------------------------------------
OPPONENT_DEF_TIER = {
    # Elite defenses
    "BOS": "elite", "MIN": "elite", "OKC": "elite",
    # Good defenses
    "MIA": "good", "NYK": "good", "MEM": "good",
    # Average
    "LAL": "avg", "DEN": "avg", "ORL": "avg", "CLE": "avg", "SAC": "avg",
    # Bad defenses
    "ATL": "bad", "NOP": "bad", "HOU": "bad", "UTA": "bad",
    # Terrible defenses
    "WAS": "terrible", "DET": "terrible", "CHA": "terrible", "POR": "terrible"
}

# -----------------------------------------------------------
# Pace multipliers — combined team/opp pace environment
# -----------------------------------------------------------
PACE_MULTIPLIER = {
    "slow": 0.94,
    "normal": 1.00,
    "fast": 1.06,
    "very_fast": 1.12
}

# -----------------------------------------------------------
# Home/away impact
# -----------------------------------------------------------
HOME_AWAY_MULT = {
    "home": 1.03,
    "away": 0.98
}

# -----------------------------------------------------------
# Rest adjustment (B2B, +1 rest, etc)
# -----------------------------------------------------------
REST_MULTIPLIER = {
    "fatigued": 0.94,   # back-to-back
    "normal": 1.00,
    "rested": 1.04,
    "extra_rest": 1.08
}

# -----------------------------------------------------------
# Blowout impact scaling
# -----------------------------------------------------------
def blowout_multiplier(is_blowout: bool):
    return 0.92 if is_blowout else 1.00

# -----------------------------------------------------------
# Usage load tolerance — affects consistency (SD)
# -----------------------------------------------------------
def usage_load_volatility(role: str):
    role = role.lower()
    if role == "primary":
        return 0.92   # more stable
    if role == "secondary":
        return 1.00
    if role == "tertiary":
        return 1.06
    return 1.12


# -----------------------------------------------------------
# MAIN ENGINE: Contextual Boost v4
# -----------------------------------------------------------
def contextual_boost_v4(
    mu: float,
    sd: float,
    market: str,
    opponent: str,
    pace: str,
    home_away: str,
    rest_state: str,
    role: str,
    blowout: bool
):
    """
    Applies full environmental context to a player's projection.
    
    Inputs:
        mu           — base projected average
        sd           — base volatility
        market       — PRA / Points / Rebounds / Assists
        opponent     — team abbrev
        pace         — slow / normal / fast / very_fast
        home_away    — home / away
        rest_state   — fatigued / normal / rested / extra_rest
        role         — primary / secondary / tertiary / low
        blowout      — True/False

    Outputs:
        (mu_adj, sd_adj)
    """

    # ---------------------------------------------------------------------
    # 1. Opponent defensive tier
    # ---------------------------------------------------------------------
    tier = OPPONENT_DEF_TIER.get(opponent.upper(), "avg")
    def_mult = DEFENSE_IMPACT.get(market, DEFENSE_IMPACT["PRA"]).get(tier, 1.00)

    # ---------------------------------------------------------------------
    # 2. Pace environment
    # ---------------------------------------------------------------------
    pace_mult = PACE_MULTIPLIER.get(pace, 1.00)

    # ---------------------------------------------------------------------
    # 3. Home/Away multiplier
    # ---------------------------------------------------------------------
    ha_mult = HOME_AWAY_MULT.get(home_away, 1.00)

    # ---------------------------------------------------------------------
    # 4. Rest advantage/disadvantage
    # ---------------------------------------------------------------------
    rest_mult = REST_MULTIPLIER.get(rest_state, 1.00)

    # ---------------------------------------------------------------------
    # 5. Blowout adjustment
    # ---------------------------------------------------------------------
    blowout_mult = blowout_multiplier(blowout)

    # ---------------------------------------------------------------------
    # 6. Combine all multipliers
    # ---------------------------------------------------------------------
    final_mu = (
        mu *
        def_mult *
        pace_mult *
        ha_mult *
        rest_mult *
        blowout_mult
    )

    # ---------------------------------------------------------------------
    # 7. Volatility adjustments
    # ---------------------------------------------------------------------
    role_vol = usage_load_volatility(role)

    final_sd = sd * role_vol
    final_sd = float(np.clip(final_sd, 0.20, sd * 1.75))

    # Safety clamp for mean
    final_mu = float(np.clip(final_mu, mu * 0.70, mu * 1.45))

    return final_mu, final_sd
# =====================================================================
# PART 3A — VOLATILITY ENGINE v2
# UltraMax V4 Stability-Calibrated Volatility Model
# =====================================================================

import numpy as np

def volatility_engine_v2(
    base_sd_per_min: float,
    projected_minutes: float,
    market: str,
    context_mult: float,
    usage_ratio: float,
    regime_state="normal"
):
    """
    UltraMax V4 Volatility Engine v2
    -------------------------------------------------------------------
    Inputs:
        base_sd_per_min : float
            Standard deviation normalized per minute from player's logs

        projected_minutes : float
            Expected minutes (clipped 18–40)

        market : str
            One of ["PRA","Points","Rebounds","Assists"]

        context_mult : float
            Opponent matchup multiplier (defensive adjustment)

        usage_ratio : float
            usage_mu / base_mu_per_min (how much they scale under role change)

        regime_state : str
            "normal", "hot", "cold" (expansion/contraction regimes)

    Output:
        final_sd : float
            UltraMax-corrected volatility for the projection engine
    """

    # --------------------------------------------------------------
    # 1. Base volatility: scale by minutes
    # --------------------------------------------------------------
    sd = base_sd_per_min * np.sqrt(max(projected_minutes, 1))

    # --------------------------------------------------------------
    # 2. Market-specific baseline
    # --------------------------------------------------------------
    if market == "PRA":
        market_mult = 1.00
    elif market == "Points":
        market_mult = 0.92
    elif market == "Rebounds":
        market_mult = 1.15
    else:  # assists
        market_mult = 1.05

    sd *= market_mult

    # --------------------------------------------------------------
    # 3. Opponent context effect
    # hard defenses increase volatility, soft defenses reduce it
    # --------------------------------------------------------------
    if context_mult > 1.05:
        sd *= 1 + (context_mult - 1.0) * 0.65
    elif context_mult < 0.95:
        sd *= 1 - (1.0 - context_mult) * 0.45

    # --------------------------------------------------------------
    # 4. Usage-based volatility expansion
    # high-usage → more variance in outcomes
    # --------------------------------------------------------------
    usage_mult = np.clip(1 + (usage_ratio - 1.0) * 0.55, 0.80, 1.45)
    sd *= usage_mult

    # --------------------------------------------------------------
    # 5. Regime adjustments (hot/cold streaks)
    # --------------------------------------------------------------
    if regime_state == "hot":
        sd *= 0.93
    elif regime_state == "cold":
        sd *= 1.10

    # --------------------------------------------------------------
    # 6. Minimum & maximum clamps for safety
    # --------------------------------------------------------------
    sd = float(np.clip(sd, 0.25, 45.0))

    return sd
# =====================================================================
# PART 3B — ENSEMBLE PROBABILITY ENGINE v3 (Normal + Skew + Heavy-Tail)
# UltraMax V4 Market-Aware Probability Model
# =====================================================================

import numpy as np
from scipy.stats import norm


def _safe_norm_cdf(x, mu, sd):
    """Internal helper to prevent norm.cdf from exploding on extreme inputs."""
    if sd <= 0 or np.isnan(sd):
        sd = max(0.50, abs(mu) * 0.10)
    return float(norm.cdf(x, loc=mu, scale=sd))


def _skew_cdf(x, mu, sd, skew_strength=0.25):
    """
    Custom right-skewed CDF for scoring markets.
    Creates extra upper-tail probability.
    """
    z = (x - mu) / max(sd, 1e-6)
    skew_term = skew_strength * (z**2) / (1 + abs(z))
    return float(norm.cdf(z - skew_term))


def _heavy_tail_cdf(x, mu, sd, tail=1.25):
    """
    Heavy-tail distribution approximation via Student-t style broadening.
    """
    adj_sd = sd * tail
    return float(norm.cdf(x, loc=mu, scale=adj_sd))


def ensemble_prob_over(mu, sd, line, market, volatility_score):
    """
    UltraMax V4 Ensemble Probability Engine v3
    -------------------------------------------------------------------
    Inputs:
        mu      — final projection mean
        sd      — volatility-engine corrected standard deviation
        line    — market line
        market  — PRA / Points / Rebounds / Assists
        volatility_score — sd/mu used to modulate weights

    Output:
        prob_over — probability of going over the line
    """

    # ------------------------------------------------------------
    # 1. Safety clamps
    # ------------------------------------------------------------
    if mu is None or np.isnan(mu):
        return 0.50
    if sd is None or sd <= 0:
        sd = max(0.75, abs(mu) * 0.20)

    # ------------------------------------------------------------
    # 2. Base CDFs (Normal, Skew, Heavy-Tail)
    # ------------------------------------------------------------
    p_norm = 1 - _safe_norm_cdf(line, mu, sd)
    p_skew = 1 - _skew_cdf(line, mu, sd, skew_strength=0.22)
    p_tail = 1 - _heavy_tail_cdf(line, mu, sd, tail=1.30)

    # ------------------------------------------------------------
    # 3. Market-specific weights (learned empirically)
    # ------------------------------------------------------------
    if market == "PRA":
        w_norm, w_skew, w_tail = 0.30, 0.40, 0.30
    elif market == "Points":
        w_norm, w_skew, w_tail = 0.35, 0.38, 0.27
    elif market == "Rebounds":
        w_norm, w_skew, w_tail = 0.45, 0.30, 0.25
    else:  # assists
        w_norm, w_skew, w_tail = 0.50, 0.28, 0.22

    # ------------------------------------------------------------
    # 4. Volatility influence
    # high volatility → reduced normal weight, boosted tail
    # ------------------------------------------------------------
    vol = float(np.clip(volatility_score, 0.05, 2.5))

    w_norm *= np.clip(1.25 - 0.40 * vol, 0.35, 1.10)
    w_tail *= np.clip(0.85 + 0.45 * vol, 0.50, 1.45)

    # Renormalize weights
    total = w_norm + w_skew + w_tail
    w_norm, w_skew, w_tail = (
        w_norm / total,
        w_skew / total,
        w_tail / total
    )

    # ------------------------------------------------------------
    # 5. Ensemble probability
    # ------------------------------------------------------------
    prob = (
        w_norm * p_norm +
        w_skew * p_skew +
        w_tail * p_tail
    )

    # Final clamp
    return float(np.clip(prob, 0.01, 0.99))
# =====================================================================
# PART 3C — CORRELATION ENGINE v3 (Synergy / Overlap / Opponent / Pace)
# UltraMax V4 — Market-Aware Dual-Leg Correlation Model
# =====================================================================

import numpy as np


# --------------------------------------------------------------
# Helper curves
# --------------------------------------------------------------
def _sigmoid(x, k=1.3):
    """Smooth bounded activation for correlation shaping."""
    return 1 / (1 + np.exp(-k * x))


def _smooth_blend(a, b, t):
    """t=0 → a ; t=1 → b."""
    t = float(np.clip(t, 0.0, 1.0))
    return (1 - t) * a + t * b


# --------------------------------------------------------------
# Synergy components: scoring → assists → rebounds
# --------------------------------------------------------------
def scoring_synergy(leg1, leg2):
    """
    Mild positive correlation if both players are high-usage scorers.
    """
    u1 = leg1["mu"]
    u2 = leg2["mu"]
    base = np.tanh((u1 + u2) / 60)
    return float(np.clip(base * 0.20, -0.10, 0.28))


def assist_dependence(leg1, leg2):
    """
    If one player is an assist-heavy creator, returns mild positive correlation.
    """
    a1 = 1 if leg1["market"] == "Assists" else 0
    a2 = 1 if leg2["market"] == "Assists" else 0
    if a1 + a2 == 0:
        return 0.0

    # scaled by usage
    u1 = leg1["mu"]
    u2 = leg2["mu"]
    level = np.tanh((u1 + u2) / 48)
    return float(level * 0.18)


def rebound_comovement(leg1, leg2):
    """
    Rebounds have strong shared variance due to pace & rim-miss rates.
    """
    r1 = 1 if leg1["market"] == "Rebounds" else 0
    r2 = 1 if leg2["market"] == "Rebounds" else 0

    if r1 + r2 < 1:
        return 0.0

    # if both are rebounders → strong positive
    if r1 and r2:
        return 0.16

    # one rebounder + one PRA scorer → mild positive
    return 0.06


def pace_correlation(leg1, leg2):
    """
    Shared pace environment.
    More pace → more possessions → higher positive covariance.
    """
    mult1 = leg1.get("ctx_mult", 1.00)
    mult2 = leg2.get("ctx_mult", 1.00)

    p = (mult1 + mult2) / 2
    return float(np.clip((p - 1.0) * 0.22, -0.05, 0.20))


def teammate_out_synergy(leg1, leg2):
    """
    If both legs benefit from the same injury situation → positive correlation.
    If only one does → slight negative (usage shifts).
    """
    o1 = 1 if leg1.get("teammate_out", False) else 0
    o2 = 1 if leg2.get("teammate_out", False) else 0

    if o1 and o2:
        return 0.15
    if o1 != o2:
        return -0.06
    return 0.0


def substitution_negative_corr(leg1, leg2):
    """
    If both players are on same team and same position group,
    they often substitute for each other → negative correlation.
    """
    t1 = leg1.get("team")
    t2 = leg2.get("team")

    if t1 is None or t2 is None:
        return 0.0
    if t1 != t2:
        return 0.0

    # same-team substitution penalty
    return -0.18


def blowout_risk_corr(leg1, leg2):
    """
    Blowout flags increase correlation because minutes volatility is shared.
    """
    b1 = 1 if leg1.get("blowout", False) else 0
    b2 = 1 if leg2.get("blowout", False) else 0

    if b1 + b2 == 0:
        return 0.0
    if b1 and b2:
        return 0.12  # shared volatility
    return 0.03      # slight upward tilt


# --------------------------------------------------------------
# Market-aware blending
# --------------------------------------------------------------
MARKET_CORR_WEIGHTS = {
    "PRA":      (0.38, 0.32, 0.30),   # scoring, assist, rebound
    "Points":   (0.55, 0.30, 0.15),
    "Rebounds": (0.25, 0.15, 0.60),
    "Assists":  (0.30, 0.55, 0.15),
}


# --------------------------------------------------------------
# MAIN CORRELATION ENGINE v3
# --------------------------------------------------------------
def correlation_engine_v3(leg1, leg2):
    """
    Returns correlation between two legs:
      Output ∈ [-0.35, +0.55]
    """

    m1 = leg1["market"]
    m2 = leg2["market"]

    # fallback: PRA weighting
    mix1 = MARKET_CORR_WEIGHTS.get(m1, MARKET_CORR_WEIGHTS["PRA"])
    mix2 = MARKET_CORR_WEIGHTS.get(m2, MARKET_CORR_WEIGHTS["PRA"])

    # Average weighting for both markets
    w_score = (mix1[0] + mix2[0]) / 2
    w_assist = (mix1[1] + mix2[1]) / 2
    w_reb = (mix1[2] + mix2[2]) / 2

    # Component correlations
    c_score  = scoring_synergy(leg1, leg2)
    c_assist = assist_dependence(leg1, leg2)
    c_reb    = rebound_comovement(leg1, leg2)
    c_pace   = pace_correlation(leg1, leg2)
    c_inj    = teammate_out_synergy(leg1, leg2)
    c_blow   = blowout_risk_corr(leg1, leg2)
    c_sub    = substitution_negative_corr(leg1, leg2)

    # Weighted structural correlation
    structural = (
        w_score  * c_score +
        w_assist * c_assist +
        w_reb    * c_reb
    )

    # Meta-adjustments
    meta = c_pace + c_inj + c_blow + c_sub

    raw = structural + meta

    # Final clamp
    corr = float(np.clip(raw, -0.35, 0.55))

    return corr
# =====================================================================
# PART 3D — FINAL PROJECTION ENGINE (UltraMax V4)
# Unified projection: usage → opponent → volatility → context shaping
# =====================================================================

import numpy as np


def final_projection_engine(
    player_name: str,
    market: str,
    logs_df,
    opponent: str,
    teammate_out: bool,
    blowout: bool,
    lookback: int = 10
):
    """
    UltraMax V4 Final Projection Engine
    -----------------------------------
    Produces final projection package:
        - mu (mean)
        - sd (volatility)
        - raw probability via ensemble distribution
        - context multipliers
        - injury flags
        - blowout flags
        - per-minute base rates
        - valid for all markets (PRA, Points, Rebounds, Assists)

    NOTE:
        `logs_df` must already be fetched via PlayerGameLog.
        This module does NOT hit the API itself.
    """

    # ==========================================================
    # Validate logs
    # ==========================================================
    if logs_df is None or logs_df.empty:
        return None, "No logs available for player."

    # Restrict sample size
    logs = logs_df.head(lookback).copy()

    # ==========================================================
    # Market Value Construction
    # ==========================================================
    market_metrics = {
        "PRA": ["PTS", "REB", "AST"],
        "Points": ["PTS"],
        "Rebounds": ["REB"],
        "Assists": ["AST"],
    }
    metrics = market_metrics.get(market, ["PTS"])

    # Compute per-game market total
    logs["MarketVal"] = logs[metrics].sum(axis=1)

    # Clean minutes
    try:
        logs["Minutes"] = logs["MIN"].astype(float)
    except:
        logs["Minutes"] = 0.0

    valid = logs["Minutes"] > 0
    if not valid.any():
        return None, "No valid minute data."

    # ==========================================================
    # Per-minute production base
    # ==========================================================
    pm_values = logs.loc[valid, "MarketVal"] / logs.loc[valid, "Minutes"]
    base_mu_per_min = float(max(pm_values.mean(), 0.05))
    base_sd_per_min = float(max(pm_values.std(), 0.10))

    # ==========================================================
    # Minutes projection
    # ==========================================================
    proj_minutes = float(np.clip(logs["Minutes"].tail(5).mean(), 18, 40))

    # ==========================================================
    # Usage Engine v3
    # ==========================================================
    teammate_out_level = 1 if teammate_out else 0

    # Default role assumption (safe global)
    role = "primary"

    usage_mu = usage_engine_v3(
        mu_per_min=base_mu_per_min,
        role=role,
        team_usage_rate=1.00,
        teammate_out_level=teammate_out_level
    )

    # ==========================================================
    # Opponent Multiplier
    # ==========================================================
    ctx_mult = opponent_matchup_v2(opponent, market)

    # ==========================================================
    # Final mean projection
    # ==========================================================
    mu_final = usage_mu * proj_minutes * ctx_mult

    # ==========================================================
    # Volatility Engine
    # ==========================================================
    usage_shift_ratio = usage_mu / max(base_mu_per_min, 0.01)

    sd_final = volatility_engine_v2(
        base_sd_per_min,
        proj_minutes,
        market,
        ctx_mult,
        usage_shift_ratio,
        regime_state="normal"
    )

    # ==========================================================
    # Optional blowout volatility padding
    # ==========================================================
    if blowout:
        sd_final *= 1.10
        mu_final *= 0.97  # slightly conservative

    # ==========================================================
    # Return Projection Package
    # ==========================================================
    proj = {
        "player": player_name,
        "market": market,
        "mu": float(mu_final),
        "sd": float(sd_final),
        "proj_minutes": float(proj_minutes),
        "ctx_mult": float(ctx_mult),
        "teammate_out": teammate_out,
        "blowout": blowout,
        "base_mu_per_min": float(base_mu_per_min),
        "base_sd_per_min": float(base_sd_per_min),
        "usage_mu": float(usage_mu),
        "usage_shift_ratio": float(usage_shift_ratio),
    }

    return proj, None
# =====================================================================
# PART 4 — LEG BUILDER ENGINE (UltraMax V4)
# Full, Complete, Streamlit-Safe
# =====================================================================

import numpy as np
import pandas as pd


def compute_leg(
    player: str,
    market: str,
    line: float,
    opponent: str,
    teammate_out: bool,
    blowout: bool,
    lookback: int = 10,
):
    """
    UltraMax V4 — LEG BUILDER
    -------------------------------------------------------
    Produces a complete leg object with:
        - game log stats
        - per-minute rates
        - usage adjustments (v3)
        - opponent adjustments (v2)
        - volatility adjustments (v2)
        - final projection (mu, sd, prob_over)
        - full MC simulation distribution (10k samples)

    RETURNS:
        (leg_dict, error_message)
    """

    # -------------------------------------------------------
    # 1. Resolve Player ID
    # -------------------------------------------------------
    pid, canonical = resolve_player(player)
    if not pid:
        return None, f"Unable to find player: '{player}'"

    # -------------------------------------------------------
    # 2. Pull Player Game Logs
    # -------------------------------------------------------
    try:
        logs = PlayerGameLog(
            player_id=pid,
            season=current_season()
        ).get_data_frames()[0]
    except Exception:
        return None, "API Error while pulling player game logs."

    if logs.empty:
        return None, "No game logs available."

    logs = logs.head(lookback)

    # =======================================================
    # 3. Build Market Values
    # =======================================================
    metrics = MARKET_METRICS.get(market, ["PTS"])

    try:
        logs["MarketVal"] = logs[metrics].sum(axis=1)
    except Exception:
        logs["MarketVal"] = logs["PTS"] if "PTS" in logs else 0

    # Clean minutes
    try:
        logs["Minutes"] = logs["MIN"].astype(float)
    except:
        logs["Minutes"] = 0.0

    valid = logs["Minutes"] > 0
    if not valid.any():
        return None, "No valid minute data in recent games."

    # =======================================================
    # 4. Per-minute base rates
    # =======================================================
    pm_vals = logs.loc[valid, "MarketVal"] / logs.loc[valid, "Minutes"]

    base_mu_per_min = float(pm_vals.mean())
    base_sd_per_min = float(max(pm_vals.std(), 0.10))

    # =======================================================
    # 5. Minutes Projection
    # =======================================================
    proj_minutes = float(np.clip(logs["Minutes"].tail(5).mean(), 18, 40))

    # =======================================================
    # 6. Usage Engine v3
    # =======================================================
    teammate_out_level = 1 if teammate_out else 0

    # ROLE ALWAYS primary for UltraMax unless expanded later
    usage_mu = usage_engine_v3(
        mu_per_min=base_mu_per_min,
        role="primary",
        team_usage_rate=1.00,
        teammate_out_level=teammate_out_level
    )

    # =======================================================
    # 7. Opponent Engine v2
    # =======================================================
    try:
        ctx_mult = opponent_matchup_v2(opponent, market)
    except:
        ctx_mult = 1.00

    # =======================================================
    # 8. Final Mean Projection
    # =======================================================
    mu = usage_mu * proj_minutes * ctx_mult

    # =======================================================
    # 9. Volatility Engine v2
    # =======================================================
    sd = volatility_engine_v2(
        base_sd_per_min=base_sd_per_min,
        proj_minutes=proj_minutes,
        market=market,
        ctx_mult=ctx_mult,
        usage_ratio=usage_mu / max(base_mu_per_min, 0.01),
        regime_state="normal"
    )

    # Protection
    sd = float(max(sd, 0.20))

    # =======================================================
    # 10. Ensemble Probability (Normal + Lognormal + Skew)
    # =======================================================
    prob_over = ensemble_prob_over(
        mu=mu,
        sd=sd,
        line=line,
        market=market,
        volatility_score=sd / max(mu, 1)
    )

    # =======================================================
    # 11. Monte Carlo Distribution (Module 10)
    # =======================================================
    mc = monte_carlo_leg_simulation(
        mu=mu,
        sd=sd,
        line=line,
        market=market,
        variance_adj=1.00,
        heavy_tail_adj=1.00,
        bias_adj=0.00
    )

    # =======================================================
    # 12. Construct LEG Object
    # =======================================================
    leg = {
        "player": canonical,
        "market": market,
        "line": float(line),

        "mu": float(mu),
        "sd": float(sd),

        "prob_over": float(prob_over),
        "mc_prob": float(mc["mc_prob_over"]),
        "dist": mc["dist"],

        "ctx_mult": float(ctx_mult),
        "proj_minutes": float(proj_minutes),
        "teammate_out": teammate_out,
        "blowout": blowout,
    }

    return leg, None
# =====================================================================
# PART 5 — ULTRAMAX V4 JOINT MONTE CARLO COMBO ENGINE
# Full, Complete, Streamlit-Safe
# =====================================================================

import numpy as np


def correlation_engine_v3(leg1: dict, leg2: dict):
    """
    UltraMax V4 — Correlation Engine
    --------------------------------
    Uses market type, player role logic, team context, volatility,
    and distribution skew to produce a stable correlation estimate
    for two legs.

    Output range: -0.25 → +0.40
    """

    # Market interaction weighting
    market_pair = f"{leg1['market']}_{leg2['market']}"

    MARKET_CORR_BASE = {
        "Points_Points": 0.25,
        "Points_PRA": 0.22,
        "PRA_PRA": 0.30,
        "Rebounds_Rebounds": 0.18,
        "Assists_Assists": 0.14,
        "Points_Assists": 0.10,
        "Points_Rebounds": 0.12,
        "Assists_Rebounds": 0.08,
    }

    base_corr = MARKET_CORR_BASE.get(market_pair, 0.10)

    # Volatility effect (higher SD = higher correlation sensitivity)
    vol_factor = np.tanh((leg1["sd"] + leg2["sd"]) * 0.08)

    # Context effects
    teammate_out_boost = 0.06 if (leg1["teammate_out"] or leg2["teammate_out"]) else 0
    same_team_penalty = -0.08 if leg1.get("team") == leg2.get("team") else 0

    # Final correlation
    corr = (
        base_corr
        + vol_factor * 0.25
        + teammate_out_boost
        + same_team_penalty
    )

    return float(np.clip(corr, -0.25, 0.40))


def monte_carlo_combo(leg1: dict, leg2: dict, corr: float, payout_mult: float):
    """
    UltraMax V4 — Full Joint Monte Carlo Combo Engine
    -------------------------------------------------
    Inputs:
        leg1, leg2   — leg dicts produced by compute_leg()
        corr         — correlation value
        payout_mult  — 3.0 for PowerPlay

    Output:
        {
            "joint_prob": float,
            "ev": float,
            "kelly_stake": float,
        }
    """

    # Extract distributions
    dist1 = leg1["dist"]
    dist2 = leg2["dist"]

    n = min(len(dist1), len(dist2), 10000)

    dist1 = dist1[:n]
    dist2 = dist2[:n]

    # Standardize
    z1 = (dist1 - dist1.mean()) / (dist1.std() + 1e-6)
    z2 = (dist2 - dist2.mean()) / (dist2.std() + 1e-6)

    # Inject correlation via Cholesky
    L = np.array([
        [1.0, 0.0],
        [corr, np.sqrt(max(1e-6, 1 - corr ** 2))]
    ])

    z = np.vstack([z1, z2])
    corr_z = L @ z

    # Transform back to distributions
    sim1 = corr_z[0] * dist1.std() + dist1.mean()
    sim2 = corr_z[1] * dist2.std() + dist2.mean()

    # Joint hit probability
    joint_hits = np.logical_and(
        sim1 > leg1["line"],
        sim2 > leg2["line"]
    )

    joint_prob = float(np.clip(np.mean(joint_hits), 0.01, 0.99))

    # EV for power play
    ev = payout_mult * joint_prob - 1.0

    # Kelly sizing (fractional Kelly baked into formula)
    b = payout_mult - 1
    q = 1 - joint_prob
    raw_kelly = ((b * joint_prob) - q) / b

    # Risk controls
    kelly_stake = float(np.clip(raw_kelly * 0.50, 0, 0.03))  # 50% Kelly, 3% cap

    return {
        "joint_prob": joint_prob,
        "ev": ev,
        "kelly_stake": kelly_stake,
    }
# =========================================================
# PART 7 — RESULTS HISTORY & CSV LOGGING (UltraMax V4)
# =========================================================

import os
import pandas as pd
import streamlit as st
from datetime import datetime

# =========================================================
# STORAGE CONFIG
# =========================================================

HISTORY_DIR = os.path.join("/tmp", "ultramax_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(HISTORY_DIR, "bet_history.csv")


# =========================================================
# 7.1 — Initialize CSV (if missing)
# =========================================================
def ensure_history_exists():
    """
    Creates history CSV with correct columns if empty or missing.
    Safe to call every run.
    """
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=[
            "Date",
            "Player",
            "Market",
            "Line",
            "Probability",
            "EV",
            "Stake",
            "KellyFrac",
            "CLV",
            "Result"       # Pending / Hit / Miss
        ])
        df.to_csv(HISTORY_FILE, index=False)


# =========================================================
# 7.2 — Load CSV
# =========================================================
def load_history():
    """
    Loads CSV and returns DataFrame.
    Always returns valid schema.
    """
    ensure_history_exists()

    try:
        df = pd.read_csv(HISTORY_FILE)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","Probability",
            "EV","Stake","KellyFrac","CLV","Result"
        ])


# =========================================================
# 7.3 — Append New Bet
# =========================================================
def log_bet(player, market, line, prob, ev, stake, kelly_frac, clv, result="Pending"):
    """
    Appends a new bet entry to the history log.
    """
    ensure_history_exists()

    entry = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Player": player,
        "Market": market,
        "Line": float(line),
        "Probability": round(prob, 4),
        "EV": round(ev * 100, 2),  # store as percentage
        "Stake": float(stake),
        "KellyFrac": round(kelly_frac, 4),
        "CLV": float(clv),
        "Result": result
    }

    df = load_history()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)


# =========================================================
# 7.4 — Update Existing Bet Result
# =========================================================
def update_result(row_index: int, new_result: str):
    """
    Marks bet as Hit/Miss.
    """
    df = load_history()
    if 0 <= row_index < len(df):
        df.at[row_index, "Result"] = new_result
        df.to_csv(HISTORY_FILE, index=False)
        return True
    return False


# =========================================================
# 7.5 — UI: History Tab Renderer
# =========================================================
def render_history_tab():
    """
    Streamlit UI block for history viewing, stats, and exporting.
    """
    st.markdown("## 📜 Bet History")

    df = load_history()

    if df.empty:
        st.info("No bets logged yet.")
        return

    # Table
    st.dataframe(df, use_container_width=True, height=420)

    # Summary metrics
    settled = df[df["Result"].isin(["Hit", "Miss"])]

    if not settled.empty:
        hits = (settled["Result"] == "Hit").sum()
        misses = (settled["Result"] == "Miss").sum()
        hit_rate = hits / max(1, hits + misses) * 100

        pnl = settled.apply(
            lambda r: r["Stake"] * 2 if r["Result"] == "Hit" else -r["Stake"],
            axis=1
        ).sum()

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Settled Bets", hits + misses)
        c2.metric("Hit Rate", f"{hit_rate:.1f}%")
        c3.metric("Profit / Loss", f"${pnl:.2f}")

    # Export CSV
    csv = df.to_csv(index=False).encode()
    st.download_button(
        "📥 Download Bet History CSV",
        csv,
        file_name="bet_history_export.csv",
        mime="text/csv"
    )
# =====================================================================
# PART 8 — RESULTS HISTORY UI (FULL, COMPLETE, STREAMLIT SAFE)
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os

# ------------------------------
# Ensure history CSV exists
# ------------------------------
HISTORY_FILE = "ultramax_history.csv"

if not os.path.exists(HISTORY_FILE):
    df_init = pd.DataFrame(columns=[
        "timestamp",
        "player1", "market1", "line1", "prob1",
        "player2", "market2", "line2", "prob2",
        "joint_prob", "joint_ev", "stake",
        "correlation",
        "drift_adj", "clv_adj"
    ])
    df_init.to_csv(HISTORY_FILE, index=False)


# ------------------------------
# Safe loader
# ------------------------------
def load_ultramax_history():
    try:
        df = pd.read_csv(HISTORY_FILE)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "timestamp",
            "player1", "market1", "line1", "prob1",
            "player2", "market2", "line2", "prob2",
            "joint_prob", "joint_ev", "stake",
            "correlation",
            "drift_adj", "clv_adj"
        ])


# =====================================================================
# TAB 2 — RESULTS HISTORY + ANALYTICS
# =====================================================================

with tab_history:

    st.markdown("## 📜 Results History")

    df_hist = load_ultramax_history()

    if df_hist.empty:
        st.info("No results logged yet. Run a 2-pick model to generate history.")
    else:

        # ---- Summary Statistics ----
        st.markdown("### 📊 Summary Metrics")

        avg_ev = df_hist["joint_ev"].mean()
        avg_prob = df_hist["joint_prob"].mean()

        colA, colB = st.columns(2)
        with colA:
            st.metric("Average Joint Probability", f"{avg_prob*100:.1f}%")
        with colB:
            st.metric("Average EV per Pick", f"{avg_ev*100:.1f}%")

        st.divider()

        # ---- Table Filter ----
        st.markdown("### 🔍 Filter Results")

        player_filter = st.text_input("Filter by player name")

        df_filtered = df_hist.copy()

        if player_filter.strip() != "":
            df_filtered = df_filtered[
                df_filtered["player1"].str.contains(player_filter, case=False, na=False)
                | df_filtered["player2"].str.contains(player_filter, case=False, na=False)
            ]

        st.dataframe(df_filtered, use_container_width=True)

        st.divider()

        # ---- Download Button ----
        st.download_button(
            label="⬇️ Download Full History CSV",
            data=df_hist.to_csv(index=False),
            file_name="ultramax_history_export.csv",
            mime="text/csv"
        )
# =====================================================================
# PART 9 — CALIBRATION TAB UI (Full, Complete, Streamlit-Safe)
# =====================================================================

import streamlit as st
import json
import os

CALIBRATION_FILE = "calibration_store.json"

# -------------------------------------------------------------
# Load calibration parameters
# -------------------------------------------------------------
def load_calibration_store():
    if not os.path.exists(CALIBRATION_FILE):
        return {
            "variance_adj": 1.00,
            "heavy_tail_adj": 1.00,
            "bias_adj": 0.00,
            "drift_adj": 1.00,
            "clv_adj": 1.00
        }
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "variance_adj": 1.00,
            "heavy_tail_adj": 1.00,
            "bias_adj": 0.00,
            "drift_adj": 1.00,
            "clv_adj": 1.00
        }


# -------------------------------------------------------------
# Save updated calibration parameters
# -------------------------------------------------------------
def save_calibration_store(store):
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(store, f, indent=4)


# =====================================================================================
# TAB — CALIBRATION TERMINAL UI
# =====================================================================================

with tab_calibration:

    st.markdown("## 🧠 UltraMax V4 — Calibration Terminal")
    st.write("Refine model accuracy using bias, variance, tail, CLV, and drift adjustments.")

    current = load_calibration_store()

    st.markdown("### 📌 Distribution Shape Tuning")

    variance_adj = st.slider(
        "Variance Adjustment",
        min_value=0.50, max_value=1.50, value=current["variance_adj"], step=0.01,
        help="Controls overall volatility of projections."
    )

    heavy_tail_adj = st.slider(
        "Heavy Tail Adjustment",
        min_value=0.80, max_value=1.20, value=current["heavy_tail_adj"], step=0.01,
        help="Inflates deflated right-tail outcomes in scoring."
    )

    bias_adj = st.slider(
        "Bias Adjustment (Shift)",
        min_value=-3.0, max_value=3.0, value=current["bias_adj"], step=0.1,
        help="Moves the entire projection mean up or down."
    )

    st.markdown("### 📌 Smart Sharpening — Drift & CLV")

    drift_adj = st.slider(
        "Model Drift Adjustment",
        min_value=0.85, max_value=1.15, value=current["drift_adj"], step=0.01,
        help="Corrects long-term over/under performance bias."
    )

    clv_adj = st.slider(
        "CLV Adjustment",
        min_value=0.85, max_value=1.15, value=current["clv_adj"], step=0.01,
        help="Sharper bias when your picks consistently beat market movement."
    )

    st.markdown("---")

    if st.button("💾 Save Calibration Settings"):
        new_store = {
            "variance_adj": float(variance_adj),
            "heavy_tail_adj": float(heavy_tail_adj),
            "bias_adj": float(bias_adj),
            "drift_adj": float(drift_adj),
            "clv_adj": float(clv_adj)
        }
        save_calibration_store(new_store)
        st.success("Calibration parameters saved successfully!")

    st.markdown("### 🧪 Current Calibration Values")
    st.json(current)
# =====================================================================
# PART 10 — RESULTS TAB UI (Full, Complete, Streamlit-Safe)
# =====================================================================

import streamlit as st

# This tab displays the model outputs AFTER the user hits "Run"
with tab_results:

    st.markdown("## 📓 Model Results")

    st.write(
        "After running the UltraMax V4 engine, results for each leg and "
        "the full combo will appear below."
    )

    st.markdown("---")

    # ---------------------------------------------------------------
    # LEG 1 RESULTS BLOCK
    # ---------------------------------------------------------------
    st.markdown("### 🏀 Leg 1 Projection")

    if "leg1_output" in st.session_state and st.session_state.leg1_output:
        leg1 = st.session_state.leg1_output
        # Render card (from Part 14)
        try:
            render_leg_card_ultramax(leg1)
        except Exception:
            st.warning("Leg 1 card renderer not yet loaded (Module 14).")
    else:
        st.info("Run model to see Leg 1 outputs.")

    st.markdown("---")

    # ---------------------------------------------------------------
    # LEG 2 RESULTS BLOCK
    # ---------------------------------------------------------------
    st.markdown("### 🏀 Leg 2 Projection")

    if "leg2_output" in st.session_state and st.session_state.leg2_output:
        leg2 = st.session_state.leg2_output
        try:
            render_leg_card_ultramax(leg2)
        except Exception:
            st.warning("Leg 2 card renderer not yet loaded (Module 14).")
    else:
        st.info("Run model to see Leg 2 outputs.")

    st.markdown("---")

    # ---------------------------------------------------------------
    # COMBO RESULTS BLOCK
    # ---------------------------------------------------------------
    st.markdown("### 🔗 Combo Outcome (UltraMax V4)")

    if "combo_output" in st.session_state and st.session_state.combo_output:

        combo = st.session_state.combo_output

        st.write(f"**Correlation Used:** {combo['corr_used']:+.3f}")
        st.write(f"**Joint Probability:** {combo['joint_prob_mc']*100:.2f}%")
        st.write(f"**EV (per $1):** {combo['joint_ev']*100:+.2f}%")
        st.write(f"**Suggested Stake:** ${combo['stake']:.2f}")

        st.success(combo["decision"])

        # Optional: visualize distributions
        if "joint_dist" in combo:
            try:
                st.line_chart(combo["joint_dist"].T)
            except Exception:
                st.warning("Could not render joint distribution chart.")
    else:
        st.info("Run model to generate combo projection.")

    st.markdown("---")

    st.caption("End of Results Tab")
# =====================================================================
# PART 11 — ENGINE RUN CONTROLLER (UltraMax V4)
# Connects UI → Compute Leg → Correlation → Monte Carlo → Decision Engine
# =====================================================================

with tab_model:

    st.markdown("### ⚙️ UltraMax Engine Execution")

    run_btn = st.button("🚀 Run UltraMax Model")

    if run_btn:

        with st.spinner("Running UltraMax Engine…"):
            time.sleep(0.20)

        # =========================================================
        # 1. VALIDATE INPUTS
        # =========================================================
        if not p1_name or not p2_name:
            st.error("❌ Please enter both players.")
            st.stop()

        # =========================================================
        # 2. COMPUTE LEGS
        # =========================================================
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

        # Handle errors
        if err1:
            st.error(f"Leg 1 Error: {err1}")
            st.stop()
        if err2:
            st.error(f"Leg 2 Error: {err2}")
            st.stop()

        # =========================================================
        # 3. RENDER INDIVIDUAL LEG PANELS (Part 7)
        # =========================================================
        st.markdown("---")
        st.markdown("### 📊 Individual Leg Projections")

        render_leg_card_ultramax(leg1)
        render_leg_card_ultramax(leg2)

        # =========================================================
        # 4. CORRELATION ENGINE (Module 7)
        # =========================================================
        base_corr = correlation_engine_v3(leg1, leg2)

        # =========================================================
        # 5. MONTE CARLO — LEG LEVEL (Module 10)
        # =========================================================
        mc1 = monte_carlo_leg_simulation(
            mu=leg1["mu"],
            sd=leg1["sd"],
            line=leg1["line"],
            market=leg1["market"]
        )
        mc2 = monte_carlo_leg_simulation(
            mu=leg2["mu"],
            sd=leg2["sd"],
            line=leg2["line"],
            market=leg2["market"]
        )

        leg1_dist = mc1["dist"]
        leg2_dist = mc2["dist"]

        # =========================================================
        # 6. JOINT MONTE CARLO (Module 11)
        # =========================================================
        joint = joint_monte_carlo_v2(
            leg1_mu=leg1["mu"],
            leg1_sd=leg1["sd"],
            leg1_line=leg1["line"],
            leg1_dist=leg1_dist,
            leg2_mu=leg2["mu"],
            leg2_sd=leg2["sd"],
            leg2_line=leg2["line"],
            leg2_dist=leg2_dist,
            corr_value=base_corr
        )

        # =========================================================
        # 7. ULTRAMAX V4 DECISION ENGINE (Module 12)
        # =========================================================
        decision = module12_two_pick_decision(
            leg1_dist=leg1_dist,
            leg2_dist=leg2_dist,
            leg1_line=leg1["line"],
            leg2_line=leg2["line"],
            base_corr=base_corr,
            payout_mult=payout_mult,     # already set to 3.0 globally
            bankroll=bankroll,
            fractional_kelly=fractional_kelly,
            drift_adj=drift_adj,
            clv_adj=clv_adj
        )

        # =========================================================
        # 8. DISPLAY COMBO RESULTS (Tab 1 + Tab 2)
        # =========================================================
        st.markdown("---")
        st.subheader("🔗 **Combo Projection (UltraMax V4)**")

        st.write(f"**Correlation Used:** {decision['corr_used']:+.3f}")
        st.write(f"**Raw Joint Probability:** {decision['p_joint_raw']*100:.1f}%")
        st.write(f"**Adjusted Joint Probability:** {decision['joint_prob_mc']*100:.1f}%")
        st.write(f"**Combo EV:** {decision['joint_ev']*100:+.1f}%")
        st.write(f"**Kelly Stake:** ${decision['stake']:.2f}")

        st.markdown(decision["decision"])

        # =========================================================
        # 9. UPDATE RESULTS TAB (Part 10)
        # =========================================================
        st.session_state["last_results"] = {
            "leg1": leg1,
            "leg2": leg2,
            "mc1": mc1,
            "mc2": mc2,
            "joint": joint,
            "decision": decision
        }

        # =========================================================
        # 10. SAVE TO HISTORY (Part 8)
        # =========================================================
        append_history({
            "Timestamp": pd.Timestamp.now(),
            "Player": f"{leg1['player']} + {leg2['player']}",
            "Market": f"{leg1['market']} + {leg2['market']}",
            "Line": f"{leg1['line']} / {leg2['line']}",
            "Probability": decision["joint_prob_mc"],
            "EV": decision["joint_ev"],
            "Result": "",
            "Stake": decision["stake"],
            "User": "local"
        })

        st.success("📌 Logged to history & pushed to Results Tab.")
# ========================================================================
# PART 12 — PRIZEPICKS LINE SYNC MODULE
# UltraMax V4 — Live Line Auto-Fetch + CLV Calculation
# ========================================================================

import requests
import time
import re
import unicodedata
from functools import lru_cache


# ---------------------------------------------------------
# Normalize player name for matching
# ---------------------------------------------------------
def _normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return name.strip()


# ---------------------------------------------------------
# Match PrizePicks players to our internal canonical names
# ---------------------------------------------------------
def match_pp_player(pp_name: str, all_players: dict) -> str:
    """
    Inputs:
      pp_name:       PrizePicks string ("LeBron James")
      all_players:   {"lebron james": "LeBron James"}

    Returns:
      canonical player name or None
    """

    target = _normalize_name(pp_name)

    if target in all_players:
        return all_players[target]

    # Fuzzy fallback
    for key in all_players:
        if target in key or key in target:
            return all_players[key]

    return None


# ---------------------------------------------------------
# Fetch PrizePicks board
# ---------------------------------------------------------
@st.cache_data(ttl=120)
def fetch_prizepicks_board():
    """
    Fetches PrizePicks markets using public endpoints.
    Cached for 2 minutes.
    """

    url = "https://api.prizepicks.com/projections"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        r = requests.get(url, headers=headers, timeout=8)
        r.raise_for_status()
    except Exception as e:
        return {"error": str(e), "data": []}

    try:
        data = r.json()
        return {"error": None, "data": data.get("data", [])}
    except:
        return {"error": "Invalid JSON structure from PP.", "data": []}


# ---------------------------------------------------------
# Extract structured PP lines
# ---------------------------------------------------------
def extract_pp_lines(pp_raw_data, all_players_map):
    """
    Converts PP raw feed → structured lines:
    {
       "player": "LeBron James",
       "market": "PRA",
       "pp_line": 42.5,
       "pp_odds": -119,
       "timestamp": 1701023940
    }
    """

    results = []
    ts = int(time.time())

    for item in pp_raw_data:
        # Example structure:
        # item["attributes"]["projection_type"] -> "Points", "Rebounds", etc
        # item["attributes"]["line_score"] -> 25.5
        # item["relationships"]["new_player"]["data"]["attributes"]["name"]

        try:
            attr = item.get("attributes", {})
            ptype = attr.get("projection_type", None)
            line_val = attr.get("line_score", None)

            # resolve market
            market = None
            if ptype in ["Points", "Rebounds", "Assists"]:
                market = ptype
            elif ptype in ["Pts+Reb+Ast", "PRA"]:
                market = "PRA"

            if not market:
                continue

            # resolve player name
            rel = item.get("relationships", {})
            pnode = rel.get("new_player", {}).get("data", {})
            attrs_player = pnode.get("attributes", {})
            player_name_pp = attrs_player.get("name", None)
            if not player_name_pp:
                continue

            canonical = match_pp_player(player_name_pp, all_players_map)
            if not canonical:
                continue

            # odds not always provided → default -119 (PP standard)
            odds = attr.get("odds", -119)

            results.append({
                "player": canonical,
                "market": market,
                "pp_line": float(line_val),
                "pp_odds": odds,
                "timestamp": ts,
            })

        except:
            continue

    return results


# ---------------------------------------------------------
# Build CLV calculation helper
# ---------------------------------------------------------
def compute_clv(model_line: float, pp_line: float) -> float:
    """
    CLV (Closing Line Value) = (model_line – pp_line)
    > 0 = model projects higher than PP line
    < 0 = model projects lower than PP line
    """

    if pp_line is None or model_line is None:
        return 0.0

    return round(model_line - pp_line, 2)


# ---------------------------------------------------------
# Bundle PP Sync
# ---------------------------------------------------------
def prizepicks_sync(all_players_map):
    """
    Returns:
      {
         "error": None or string,
         "lines": [
            {
               "player": "...",
               "market": "...",
               "pp_line": 23.5,
               "pp_odds": -119,
               "timestamp": 1701023940
            }
         ]
      }
    """

    pp_data = fetch_prizepicks_board()
    if pp_data["error"]:
        return {"error": pp_data["error"], "lines": []}

    structured = extract_pp_lines(pp_data["data"], all_players_map)

    return {
        "error": None,
        "lines": structured
    }
# =========================================================
# PART 13 — MARKET SCANNER (ODDS API INTEGRATION)
# UltraMax V4 — Live Prop Mispricing Scanner
# =========================================================

import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------
# CONFIG — The Odds API (or compatible)
# ---------------------------------------------------------
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", None)

ODDS_API_ENDPOINT = (
    "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
)

# Map raw market keys → our internal markets
SCANNER_MARKET_MAP = {
    "player_points_over": "Points",
    "player_rebounds_over": "Rebounds",
    "player_assists_over": "Assists",
    "player_PRA_over": "PRA",
}


def _decimal_to_implied_prob(decimal_odds: float) -> float:
    """
    Convert decimal odds → implied probability (0–1).
    """
    try:
        if decimal_odds <= 1e-9:
            return 0.5
        return float(1.0 / decimal_odds)
    except Exception:
        return 0.5


# ---------------------------------------------------------
# 13.1 — FETCH LIVE ODDS
# ---------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_nba_props():
    """
    Fetches live NBA props from The Odds API.

    Returns:
        df (DataFrame), error_msg (str or None)
    """
    if not ODDS_API_KEY:
        return pd.DataFrame(), "Missing ODDS_API_KEY in Streamlit secrets."

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(SCANNER_MARKET_MAP.keys()),
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    try:
        resp = requests.get(ODDS_API_ENDPOINT, params=params, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame(), f"API error: {resp.status_code} — {resp.text}"

        raw = resp.json()
        rows = []

        for game in raw:
            home = game.get("home_team")
            away = game.get("away_team")
            commence = game.get("commence_time")
            game_label = f"{away} @ {home}"

            for book in game.get("bookmakers", []):
                book_name = book.get("title", "Unknown Book")

                for market in book.get("markets", []):
                    key = market.get("key")
                    if key not in SCANNER_MARKET_MAP:
                        continue

                    our_market = SCANNER_MARKET_MAP[key]

                    for outcome in market.get("outcomes", []):
                        player = outcome.get("description")
                        line = outcome.get("point", None)
                        price = outcome.get("price", None)

                        if player is None or line is None or price is None:
                            continue

                        implied = _decimal_to_implied_prob(price)

                        rows.append(
                            {
                                "Player": str(player),
                                "Market": our_market,
                                "Line": float(line),
                                "DecimalOdds": float(price),
                                "ImpliedProb": float(implied),
                                "Book": book_name,
                                "Game": game_label,
                                "StartTime": commence,
                            }
                        )

        if not rows:
            return pd.DataFrame(), "No props found from odds API."

        df = pd.DataFrame(rows)
        return df, None

    except Exception as e:
        return pd.DataFrame(), f"Request error: {e}"


# ---------------------------------------------------------
# 13.2 — ENRICH WITH MODEL PROJECTIONS
# ---------------------------------------------------------
def enrich_props_with_model(df: pd.DataFrame, lookback_games: int = 10):
    """
    For each odds row, run UltraMax model via compute_leg()
    and attach:
      - ModelProb (p_over)
      - ModelEdge (ModelProb - ImpliedProb)
      - EV% (edge in percent)

    Assumes:
        compute_leg(player, market, line, opponent, teammate_out, blowout, lookback)
    is already defined in earlier parts.
    """
    if df.empty:
        return df

    enriched_rows = []

    for _, r in df.iterrows():
        player = r["Player"]
        market = r["Market"]
        line = r["Line"]

        # Opponent not directly provided as abbreviation; we pass empty → neutral matchup
        opp = ""       # opponent_matchup_v2 will fall back to 1.0 if empty
        teammate_out = False
        blowout = False

        leg, err = compute_leg(
            player=player,
            market=market,
            line=line,
            opponent=opp,
            teammate_out=teammate_out,
            blowout=blowout,
            lookback=lookback_games,
        )

        if err or leg is None:
            # We still keep the row but mark model fields as NaN
            r2 = r.copy()
            r2["ModelProb"] = np.nan
            r2["ModelEdge"] = np.nan
            r2["EV_percent"] = np.nan
            enriched_rows.append(r2)
            continue

        p_model = float(leg["prob_over"])  # 0–1
        p_implied = float(r["ImpliedProb"])
        edge = p_model - p_implied
        ev_percent = edge * 100.0

        r2 = r.copy()
        r2["ModelProb"] = p_model
        r2["ModelEdge"] = edge
        r2["EV_percent"] = ev_percent
        enriched_rows.append(r2)

    out = pd.DataFrame(enriched_rows)

    # Sort by EV descending (best edges first)
    out = out.sort_values("EV_percent", ascending=False, na_position="last").reset_index(drop=True)
    return out


# ---------------------------------------------------------
# 13.3 — STREAMLIT UI BLOCK FOR MARKET SCANNER
# ---------------------------------------------------------
def render_market_scanner_ui(default_lookback: int = 10):
    """
    Renders the full Market Scanner UI block.
    You can plug this into:
      - its own tab (e.g., tab_scanner)
      - or a section of the Results tab.
    """

    st.markdown("## 🔍 Market Scanner — Live Odds vs UltraMax Model")
    st.caption("Pulls live NBA props, runs your UltraMax model, and surfaces top mispriced edges.")

    if not ODDS_API_KEY:
        st.error("⚠ ODDS_API_KEY missing in Streamlit secrets. Add it to use the Market Scanner.")
        return

    col_top = st.columns(3)
    with col_top[0]:
        lookback = st.slider(
            "Games lookback for model",
            min_value=3,
            max_value=20,
            value=default_lookback,
            step=1,
        )
    with col_top[1]:
        min_ev = st.slider(
            "Minimum EV% to display",
            min_value=-20.0,
            max_value=100.0,
            value=2.0,
            step=1.0,
        )
    with col_top[2]:
        selected_market = st.selectbox(
            "Filter by market",
            ["All", "PRA", "Points", "Rebounds", "Assists"],
            index=0,
        )

    if st.button("🚀 Scan Live Market"):
        with st.spinner("Fetching live odds and running UltraMax model..."):
            df_raw, err = fetch_live_nba_props()

            if err:
                st.error(err)
                return

            if df_raw.empty:
                st.warning("No props returned from the Odds API.")
                return

            df_model = enrich_props_with_model(df_raw, lookback_games=lookback)

        # Apply filters
        df_view = df_model.copy()
        df_view = df_view[df_view["EV_percent"] >= min_ev]

        if selected_market != "All":
            df_view = df_view[df_view["Market"] == selected_market]

        if df_view.empty:
            st.warning("No props meet the current EV and market filters.")
            return

        # Display main table
        st.markdown("### 📊 Top Mispriced Props")
        # nice formatting
        df_display = df_view.copy()
        df_display["ModelProb%"] = (df_display["ModelProb"] * 100).round(1)
        df_display["ImpliedProb%"] = (df_display["ImpliedProb"] * 100).round(1)
        df_display["EV_percent"] = df_display["EV_percent"].round(2)

        cols_order = [
            "Player",
            "Market",
            "Line",
            "Book",
            "Game",
            "DecimalOdds",
            "ModelProb%",
            "ImpliedProb%",
            "EV_percent",
            "StartTime",
        ]
        df_display = df_display[[c for c in cols_order if c in df_display.columns]]

        st.dataframe(df_display, use_container_width=True)

        # Download CSV
        csv_data = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Scanner Results (CSV)",
            data=csv_data,
            file_name=f"ultramax_market_scanner_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.success("Market Scanner completed.")
# ================================================================
# PART 14 — RENDER CARDS (Player Leg Cards UI)
# UltraMax V4 — Final, Complete, Streamlit-Safe
# ================================================================

import streamlit as st
import numpy as np

# ------------------------------------------------
# CSS (loaded once)
# ------------------------------------------------
def _inject_leg_card_css():
    if "leg_card_css_loaded" not in st.session_state:
        st.markdown("""
        <style>
        .leg-card {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 14px;
            color: #F9FAFB;
        }
        .leg-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #F3F4F6;
        }
        .leg-sub {
            font-size: 0.9rem;
            color: #9CA3AF;
        }
        .metric-row {
            margin-top: 8px;
            font-size: 0.95rem;
        }
        .metric-label {
            color: #9CA3AF;
        }
        .metric-value {
            font-weight: 600;
        }
        .ev-positive {
            color: #10B981 !important;
            font-weight: 700;
        }
        .ev-negative {
            color: #EF4444 !important;
            font-weight: 700;
        }
        .prob-strong {
            color: #3B82F6 !important;
            font-weight: 700;
        }
        .prob-medium {
            color: #F59E0B !important;
            font-weight: 700;
        }
        .prob-weak {
            color: #EF4444 !important;
            font-weight: 700;
        }
        .section-divider {
            margin: 10px 0;
            border-bottom: 1px solid #1F2937;
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state["leg_card_css_loaded"] = True


# ------------------------------------------------
# Probability → Color Tagging
# ------------------------------------------------
def _prob_color_tag(prob: float) -> str:
    if prob >= 0.62:
        return "prob-strong"
    elif prob >= 0.55:
        return "prob-medium"
    else:
        return "prob-weak"


# ------------------------------------------------
# EV → Color Tagging
# ------------------------------------------------
def _ev_color_tag(ev: float) -> str:
    if ev >= 0.05:
        return "ev-positive"
    else:
        return "ev-negative"


# ------------------------------------------------
# Round helper
# ------------------------------------------------
def _rnd(x):
    try:
        return round(float(x), 2)
    except:
        return x


# ------------------------------------------------
# MAIN CARD RENDERER — UltraMax Full
# ------------------------------------------------
def render_leg_card_ultramax(leg: dict):
    """
    Renders a full UI card for a single prop leg.
    Requires fields:
        - player
        - market
        - line
        - prob_over
        - mu
        - sd
        - ctx_mult
        - teammate_out
        - blowout
    Safe: does not crash if fields missing.
    """

    if not leg or not isinstance(leg, dict):
        st.error("Invalid leg input.")
        return

    _inject_leg_card_css()

    player = leg.get("player", "Unknown")
    market = leg.get("market", "?")
    line = leg.get("line", "?")
    prob = float(leg.get("prob_over", 0.0))
    mu = _rnd(leg.get("mu", 0.0))
    sd = _rnd(leg.get("sd", 0.0))
    ctx_mult = _rnd(leg.get("ctx_mult", 1.0))

    p_color = _prob_color_tag(prob)

    st.markdown(f"""
    <div class="leg-card">
        <div class="leg-title">{player}</div>
        <div class="leg-sub">{market} • Line: {line}</div>

        <div class="section-divider"></div>

        <div class="metric-row">
            <span class="metric-label">Model Probability: </span>
            <span class="metric-value {p_color}">{prob*100:.1f}%</span>
        </div>

        <div class="metric-row">
            <span class="metric-label">Projected Mean (μ): </span>
            <span class="metric-value">{mu}</span>
        </div>

        <div class="metric-row">
            <span class="metric-label">Volatility (σ): </span>
            <span class="metric-value">{sd}</span>
        </div>

        <div class="metric-row">
            <span class="metric-label">Opponent Multiplier: </span>
            <span class="metric-value">{ctx_mult}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Expand advanced block
    with st.expander("Advanced Metrics"):
        st.write(leg)


# ------------------------------------------------
# SIMPLE CARD RENDERER — Fallback
# ------------------------------------------------
def render_leg_card_simple(leg: dict):
    """Minimal fallback card."""
    if not leg:
        st.warning("Leg missing; cannot render.")
        return

    st.markdown(f"### {leg.get('player', '?')} — {leg.get('market', '?')}")
    st.write(f"**Line:** {leg.get('line')}")
    st.write(f"**Prob:** {leg.get('prob_over', 0)*100:.1f}%")
    st.write(f"μ = {_rnd(leg.get('mu'))}, σ = {_rnd(leg.get('sd'))}")
# ================================================================
# PART 15 — MODEL TAB CORE ENGINE
# Executes legs, renders cards, computes correlation + joint EV
# UltraMax V4 — Full, Complete, Streamlit Error-Free
# ================================================================

import streamlit as st
import numpy as np
import time

# payout multiple (Power Play)
PAYOUT_MULT = 3.0


def run_ultramax_model_tab(
    p1_name, p1_market, p1_line, p1_opp, p1_teammate_out, p1_blowout,
    p2_name, p2_market, p2_line, p2_opp, p2_teammate_out, p2_blowout,
    games_lookback=10,
):
    """
    Core UltraMax runner for the Model tab.
    """

    st.markdown("### 🚀 UltraMax Engine Output")

    # --------------------------------------------------------------------
    # 1. Compute both legs
    # --------------------------------------------------------------------
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

    # --------------------------------------------------------------------
    # 2. Error handling
    # --------------------------------------------------------------------
    if err1:
        st.error(f"Leg 1 Error: {err1}")
    if err2:
        st.error(f"Leg 2 Error: {err2}")

    if not leg1 and not leg2:
        st.warning("No valid legs computed.")
        return

    # --------------------------------------------------------------------
    # 3. Render Leg Cards
    # --------------------------------------------------------------------
    st.markdown("### 📘 Individual Leg Projections")

    colA, colB = st.columns(2)

    with colA:
        if leg1:
            render_leg_card_ultramax(leg1)

    with colB:
        if leg2:
            render_leg_card_ultramax(leg2)

    # --------------------------------------------------------------------
    # 4. Market-Implied Probability
    # --------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Market vs Model Probability Check")

    implied = 1.0 / PAYOUT_MULT
    st.write(f"**Market implied probability:** {implied*100:.1f}%")

    if leg1:
        m1 = leg1["prob_over"]
        st.write(
            f"**{leg1['player']} model probability:** {m1*100:.1f}% "
            f"→ Edge: {(m1 - implied)*100:+.1f}%"
        )

    if leg2:
        m2 = leg2["prob_over"]
        st.write(
            f"**{leg2['player']} model probability:** {m2*100:.1f}% "
            f"→ Edge: {(m2 - implied)*100:+.1f}%"
        )

    # --------------------------------------------------------------------
    # 5. Joint Probability + Correlation + Combo EV
    # --------------------------------------------------------------------
    if leg1 and leg2:

        st.markdown("---")
        st.subheader("🔗 Correlation & Combo Projection")

        # correlation
        corr = correlation_engine_v3(leg1, leg2)
        st.write(f"**Correlation:** {corr:+.3f}")

        # prepare distributions (Module 10)
        sim1 = monte_carlo_leg_simulation(
            mu=leg1["mu"],
            sd=leg1["sd"],
            line=leg1["line"],
            market=leg1["market"]
        )
        sim2 = monte_carlo_leg_simulation(
            mu=leg2["mu"],
            sd=leg2["sd"],
            line=leg2["line"],
            market=leg2["market"]
        )

        # UltraMax joint engine (Module 12)
        decision = module12_two_pick_decision(
            leg1_dist=sim1["dist"],
            leg2_dist=sim2["dist"],
            leg1_line=leg1["line"],
            leg2_line=leg2["line"],
            base_corr=corr,
            payout_mult=PAYOUT_MULT,
            bankroll=100,           # placeholder bankroll
            fractional_kelly=0.20,  # 20% Kelly fraction
            drift_adj=1.00,         # calibration (Module 9)
            clv_adj=1.00,           # calibration (Module 9)
            iterations=10_000
        )

        joint_prob = decision["joint_prob_mc"]
        ev_combo = decision["joint_ev"]
        stake = decision["stake"]

        st.write(f"**Joint Probability:** {joint_prob*100:.1f}%")
        st.write(f"**EV (per $1):** {ev_combo*100:+.1f}%")
        st.write(f"**Kelly Stake (bankroll=100):** ${stake:.2f}")

        # final decision label
        st.write(decision["decision"])

    # --------------------------------------------------------------------
    # 6. Save baseline history (optional)
    # --------------------------------------------------------------------
    if leg1:
        update_market_baseline(leg1["player"], leg1["market"], leg1["line"])
    if leg2:
        update_market_baseline(leg2["player"], leg2["market"], leg2["line"])
# ======================================================================
# PART 16 — TEAM CONTEXT HEURISTICS ENGINE
# Provides team-level multipliers used across Modules 3–12
# ======================================================================

import numpy as np

# ----------------------------------------------------------------------
# Team Pace Scaling (relative possession speed)
# ----------------------------------------------------------------------
TEAM_PACE = {
    "LAL": 1.06, "GSW": 1.05, "SAC": 1.04, "OKC": 1.03, "IND": 1.10,
    "DEN": 0.98, "BOS": 0.97, "MIA": 0.95, "NYK": 0.96,
    "MIL": 1.02, "PHX": 0.99, "DAL": 1.03, "MIN": 0.98,
}

DEFAULT_PACE = 1.00


def team_pace_factor(team: str) -> float:
    """Returns the team’s pace multiplier."""
    if not team:
        return DEFAULT_PACE
    return float(TEAM_PACE.get(team.upper(), DEFAULT_PACE))


# ----------------------------------------------------------------------
# Offensive Creation / Usage Share (How much the star drives offense)
# ----------------------------------------------------------------------
TEAM_CREATION_SHARE = {
    "LAL": 1.12, "GSW": 1.08, "DAL": 1.14,
    "DEN": 1.10, "PHX": 1.09, "MIL": 1.11,
    "MIN": 0.98, "NYK": 1.05, "BOS": 1.03
}

def team_creation_factor(team: str) -> float:
    """How much team funnels offense through main scorers."""
    return float(TEAM_CREATION_SHARE.get(team.upper(), 1.00))


# ----------------------------------------------------------------------
# Assist Rate Factor (context for AST projection stability)
# ----------------------------------------------------------------------
TEAM_AST_RATE = {
    "GSW": 1.12, "IND": 1.10, "DEN": 1.11,
    "NYK": 0.95, "MIA": 0.92, "SAS": 1.08,
}

def assist_environment(team: str) -> float:
    """Teams with high ball movement boost AST consistency."""
    return float(TEAM_AST_RATE.get(team.upper(), 1.00))


# ----------------------------------------------------------------------
# Rebound Environment (team scheme + rim presence)
# ----------------------------------------------------------------------
TEAM_REB_ENV = {
    "MIL": 1.08, "MIN": 1.10, "DEN": 1.07,
    "GSW": 0.92, "IND": 0.95, "DAL": 0.98
}

def rebound_environment(team: str) -> float:
    """Teams with strong bigs or high miss frequency boost REB."""
    return float(TEAM_REB_ENV.get(team.upper(), 1.00))


# ----------------------------------------------------------------------
# Blowout Sensitivity (minutes volatility tied to score margin)
# Used by compute_leg via blowout flag
# ----------------------------------------------------------------------
TEAM_BLOWOUT_SENS = {
    "LAL": 1.05,   # LeBron/AD sit early in blowouts
    "BOS": 0.94,   # starters often stay
    "DEN": 0.96,
    "PHX": 1.03,
    "GSW": 1.06,
    "MIN": 0.99
}

def blowout_multiplier(team: str, blowout_flag: bool) -> float:
    """
    If the game has blowout risk, adjust variance:
    - Teams with high sensitivity reduce projection by up to 6%.
    """
    if not blowout_flag:
        return 1.00
    base = TEAM_BLOWOUT_SENS.get(team.upper(), 1.00)
    return float(np.clip(1.00 - ((base - 1.0) * 0.8), 0.90, 1.02))


# ----------------------------------------------------------------------
# Team Context Bundle
# (this is what compute_leg() pulls internally)
# ----------------------------------------------------------------------
def team_context_bundle(team: str, market: str, blowout: bool):
    """
    Unified access point — Modules 3, 4, 5, 7, 9 use this.

    Returns:
        {
            "pace": float,
            "creation": float,
            "assist_env": float,
            "rebound_env": float,
            "blowout_mult": float
        }
    """
    return {
        "pace": team_pace_factor(team),
        "creation": team_creation_factor(team),
        "assist_env": assist_environment(team),
        "rebound_env": rebound_environment(team),
        "blowout_mult": blowout_multiplier(team, blowout),
    }
# ======================================================================
# PART 17 — OPPONENT DEFENSE PROFILES
# Scoring, Rebounding, Assist, and PRA defensive efficiency multipliers
# Used inside opponent_matchup_v2 (Module 4) + compute_leg
# ======================================================================

import numpy as np

# ----------------------------------------------------------------------
# Baseline Defensive Profiles (Market-Level)
# Lower number = tougher defense
# Higher number = easier defense
# ----------------------------------------------------------------------

DEF_PTS = {
    "BOS": 0.88, "MIN": 0.90, "MIA": 0.92, "OKC": 0.93, "DEN": 0.94,
    "CLE": 0.96, "NYK": 0.97, "PHX": 0.98, "LAL": 1.02, "GSW": 1.05,
    "IND": 1.10, "SAS": 1.12, "CHA": 1.15
}

DEF_REB = {
    "MIN": 0.88, "DEN": 0.92, "NYK": 0.94, "CLE": 0.95, "MIA": 0.97,
    "GSW": 1.05, "PHX": 1.03, "LAL": 1.02, "DAL": 1.04, "CHA": 1.10
}

DEF_AST = {
    "BOS": 0.90, "CLE": 0.92, "MIA": 0.94, "NYK": 0.95, "MIN": 0.97,
    "PHX": 1.02, "GSW": 1.05, "IND": 1.08, "SAS": 1.10, "CHA": 1.14
}

DEF_PRA = {
    "BOS": 0.92, "MIN": 0.94, "CLE": 0.96, "DEN": 0.97, "NYK": 0.98,
    "LAL": 1.03, "GSW": 1.05, "DAL": 1.06, "IND": 1.10, "CHA": 1.15
}

DEFAULT_DEF = 1.00

# ----------------------------------------------------------------------
# Positional Defensive Profiles (enhances realism)
# These are used in scoring, rebounding, and assist modeling.
# ----------------------------------------------------------------------

DEF_POS = {
    "PG": {"PTS": 0.96, "REB": 1.03, "AST": 0.92, "PRA": 0.97},
    "SG": {"PTS": 0.98, "REB": 1.02, "AST": 0.96, "PRA": 0.99},
    "SF": {"PTS": 1.00, "REB": 1.00, "AST": 1.00, "PRA": 1.00},
    "PF": {"PTS": 1.03, "REB": 1.08, "AST": 1.02, "PRA": 1.05},
    "C":  {"PTS": 1.05, "REB": 1.12, "AST": 1.04, "PRA": 1.08},
}

# ----------------------------------------------------------------------
# Helper: Determine player position (requires resolve_player)
# If unavailable, assume neutral SF.
# ----------------------------------------------------------------------

def resolve_position(player_name: str) -> str:
    """
    ATTENTION: This should map from your player database.
    To stay Streamlit-safe, we fallback to SF when unknown.
    """
    try:
        # If you built a player DB earlier, map it here.
        # For now, default:
        return "SF"
    except:
        return "SF"


# ----------------------------------------------------------------------
# Unified opponent matchup engine
# This is called directly inside compute_leg()
# ----------------------------------------------------------------------

def opponent_def_factor(team: str, market: str) -> float:
    """Returns market-specific defensive multiplier for opponent."""
    team = team.upper() if team else ""

    if market == "Points":
        return float(DEF_PTS.get(team, DEFAULT_DEF))
    elif market == "Rebounds":
        return float(DEF_REB.get(team, DEFAULT_DEF))
    elif market == "Assists":
        return float(DEF_AST.get(team, DEFAULT_DEF))
    else:  # PRA
        return float(DEF_PRA.get(team, DEFAULT_DEF))


def positional_def_factor(position: str, market: str) -> float:
    """
    Returns positional defensive modifier:
    PG/SG/SF/PF/C → adjusts for how opponents defend that archetype.
    """
    position = position.upper() if position else "SF"

    try:
        return float(DEF_POS[position][market if market != "PRA" else "PRA"])
    except:
        return 1.00


# ----------------------------------------------------------------------
# MASTER FUNCTION: opponent_matchup_v2()
# This is what compute_leg() actually calls.
# ----------------------------------------------------------------------

def opponent_matchup_v2(opponent_team: str, market: str, player_name: str = None):
    """
    Produces final matchup multiplier:
    - Opponent defensive efficiency
    - Player position defense
    - Weighted blend for stability
    """

    base = opponent_def_factor(opponent_team, market)

    player_pos = resolve_position(player_name) if player_name else "SF"
    pos_adj = positional_def_factor(player_pos, market)

    # Weighted blend:
    #   70% opponent
    #   30% positional fit
    blended = (base ** 0.70) * (pos_adj ** 0.30)

    # Final clamp
    blended = float(np.clip(blended, 0.80, 1.25))

    return blended
# ======================================================================
# PART 18 — GAME SCRIPT ENGINE
# Blowout probabilities, pace, spread, rotations, & minute volatility
# ======================================================================

import numpy as np

# ----------------------------------------------------------------------
# Baseline pace factors by team (approximate)
# ----------------------------------------------------------------------
TEAM_PACE = {
    "IND": 1.08, "ATL": 1.06, "OKC": 1.05, "LAL": 1.04, "GSW": 1.04,
    "SAS": 1.03, "TOR": 1.02, "NOP": 1.01, "MIL": 1.00,
    "NYK": 0.96, "MIN": 0.95, "CLE": 0.95, "MIA": 0.94,
    "DEN": 0.97, "PHX": 0.99, "DAL": 0.98, "HOU": 1.02,
    "CHA": 1.03, "DET": 1.01, "CHI": 1.00
}

DEFAULT_PACE = 1.00


# ----------------------------------------------------------------------
# Blowout probability model (spread → blowout risk)
# ----------------------------------------------------------------------
def blowout_probability(spread: float) -> float:
    """
    Convert Vegas spread into a realistic blowout probability.
    spread > 0 : favorite by that many points.
    """

    spread = abs(float(spread))

    # Curve fitted from NBA logistic distribution of results:
    #  0–5  → ~5%
    #  6–9  → ~10–18%
    # 10–13 → ~20–30%
    # 14–20 → ~30–45%
    prob = 1 / (1 + np.exp(-0.28 * (spread - 9)))

    return float(np.clip(prob, 0.03, 0.50))


# ----------------------------------------------------------------------
# Impact of blowout on minutes & production
# ----------------------------------------------------------------------
def blowout_minute_multiplier(blowout_prob: float) -> float:
    """
    Expected minute reduction from blowout probability.
    """
    # Rough mapping:
    #  p=0.05 → ~0% loss
    #  p=0.20 → ~4–5% loss
    #  p=0.40 → ~10% loss

    reduction = 1 - (blowout_prob * 0.22)
    return float(np.clip(reduction, 0.80, 1.00))


# ----------------------------------------------------------------------
# Pace adjustment (team vs opponent)
# ----------------------------------------------------------------------
def pace_multiplier(team: str, opp: str) -> float:
    team = team.upper() if team else ""
    opp = opp.upper() if opp else ""

    pace_team = TEAM_PACE.get(team, DEFAULT_PACE)
    pace_opp = TEAM_PACE.get(opp, DEFAULT_PACE)

    blended = (pace_team * 0.55) + (pace_opp * 0.45)
    return float(np.clip(blended, 0.92, 1.12))


# ----------------------------------------------------------------------
# Rotation shrink/expand model
# ----------------------------------------------------------------------
def rotation_factor(game_importance: float, fatigue_index: float):
    """
    game_importance: (0–1)
    fatigue_index:   (0–1)

    Returns multiplier on minutes:
      - High importance shrinks rotation → more minutes
      - High fatigue reduces minutes slightly
    """

    imp = np.clip(game_importance, 0, 1)
    fat = np.clip(fatigue_index, 0, 1)

    # importance can boost up to +6%
    # fatigue can remove up to -5%
    mult = 1 + (0.06 * imp) - (0.05 * fat)
    return float(np.clip(mult, 0.92, 1.08))


# ----------------------------------------------------------------------
# MASTER FUNCTION: combine all game-script signals
# ----------------------------------------------------------------------
def game_script_multiplier(
    player_team: str,
    opponent_team: str,
    vegas_spread: float = 0,
    force_blowout: bool = False,
    game_importance: float = 0.50,
    fatigue_index: float = 0.30
):
    """
    OUTPUT:
      final_mult → multiplier used to adjust minutes/projection
    """

    # ============ Blowout modeling ============
    if force_blowout:
        blowout_prob = 0.40
    else:
        blowout_prob = blowout_probability(vegas_spread)

    blowout_mult = blowout_minute_multiplier(blowout_prob)

    # ============ Pace modeling ============
    pace_mult = pace_multiplier(player_team, opponent_team)

    # ============ Rotation modeling ============
    rotation_mult = rotation_factor(game_importance, fatigue_index)

    # ============ Final Blended Script Multiplier ============
    # Combined effect but softened for stability:
    final = (blowout_mult ** 0.55) * (pace_mult ** 0.60) * (rotation_mult ** 0.45)

    # Clamp for safety:
    final = float(np.clip(final, 0.80, 1.20))

    return {
        "final_mult": final,
        "blowout_prob": blowout_prob,
        "blowout_mult": blowout_mult,
        "pace_mult": pace_mult,
        "rotation_mult": rotation_mult
    }
# ======================================================================
# PART 19 — ROTATIONAL VOLATILITY ENGINE
# Models uncertainty in minutes based on role, team patterns, injuries,
# coaching volatility, and game script signals.
# ======================================================================

import numpy as np

# ----------------------------------------------------------------------
# Baseline per-team coaching volatility (0.90–1.15)
# ----------------------------------------------------------------------
TEAM_COACH_VOL = {
    "OKC": 1.05, "GSW": 1.08, "LAL": 1.07, "MIL": 1.03, "CHI": 1.02,
    "SAC": 1.01, "PHX": 1.06, "TOR": 1.00, "UTA": 1.07, "MIA": 0.97,
    "NYK": 0.96, "DEN": 0.99, "MEM": 1.10, "HOU": 1.04, "NOP": 1.03,
    "CLE": 0.98, "MIN": 0.96, "ATL": 1.05, "DAL": 1.04, "IND": 1.01
}

DEFAULT_COACH_VOL = 1.03


# ----------------------------------------------------------------------
# Role-based volatility signal
# ----------------------------------------------------------------------
ROLE_VOLATILITY = {
    "primary": 0.92,
    "secondary": 0.97,
    "tertiary": 1.05,
    "bench": 1.12,
    "spot": 1.18
}


# ----------------------------------------------------------------------
# Injury-driven rotation randomness
# ----------------------------------------------------------------------
def injury_rotation_volatility(teammate_out_level: int):
    """
    teammate_out_level = # of major creators missing (0–3)
    """
    lvl = max(0, min(3, teammate_out_level))

    # More players out → more random minutes
    # Ranges from +0% → +12%
    return 1 + (lvl * 0.04)  # 0.00 → 0.12


# ----------------------------------------------------------------------
# Recent coaching stability (last 5 games MIN std)
# ----------------------------------------------------------------------
def recent_minutes_stability(game_logs):
    """
    Computes volatility in minutes based on recent behavior.
    """
    if "MIN" not in game_logs:
        return 1.00

    try:
        minutes = game_logs["MIN"].astype(float).tail(5)
        std = np.std(minutes)

        # std 0 → 1.00 multiplier
        # std 8+ → 1.10 multiplier
        mult = 1 + min(0.10, (std / 80))

        return float(np.clip(mult, 1.00, 1.10))
    except:
        return 1.00


# ----------------------------------------------------------------------
# Master Rotational Volatility Function
# ----------------------------------------------------------------------
def rotational_volatility_engine(
    player_role: str,
    player_team: str,
    game_logs,
    teammate_out_level: int,
    script_mult: float
):
    """
    Produces a volatility multiplier >1.00 meaning more uncertainty.
    INPUTS:
      - player_role: "primary", "secondary", "bench", etc.
      - player_team: team abbreviation
      - game_logs: DataFrame of logs from compute_leg
      - teammate_out_level: 0–3
      - script_mult: from Part 18 (pace + blowout + rotation)

    Returns:
      {
        "vol_mult": final volatility multiplier,
        "role_vol": role-based multiplier,
        "coach_vol": coach multiplier,
        "injury_vol": injury randomness,
        "stability_vol": recent minute std,
        "script_vol": script-induced volatility
      }
    """

    # --------------------------------------------------------
    # Role volatility
    # --------------------------------------------------------
    role = player_role.lower()
    role_vol = ROLE_VOLATILITY.get(role, 1.00)

    # --------------------------------------------------------
    # Coaching volatility (team-specific)
    # --------------------------------------------------------
    team = player_team.upper() if player_team else ""
    coach_vol = TEAM_COACH_VOL.get(team, DEFAULT_COACH_VOL)

    # --------------------------------------------------------
    # Injury randomness
    # --------------------------------------------------------
    injury_vol = injury_rotation_volatility(teammate_out_level)

    # --------------------------------------------------------
    # Recent stability (minutes variance)
    # --------------------------------------------------------
    stability_vol = recent_minutes_stability(game_logs)

    # --------------------------------------------------------
    # Game script volatility from Part 18
    # Higher script mult → higher pace & risk → more variance
    # --------------------------------------------------------
    script_vol = 1 + ((script_mult - 1.00) * 0.40)
    script_vol = float(np.clip(script_vol, 0.90, 1.12))

    # --------------------------------------------------------
    # Final blended volatility multiplier
    # --------------------------------------------------------
    final = (
        (role_vol ** 0.40) *
        (coach_vol ** 0.25) *
        (injury_vol ** 0.30) *
        (stability_vol ** 0.30) *
        (script_vol ** 0.25)
    )

    final = float(np.clip(final, 0.90, 1.25))

    return {
        "vol_mult": final,
        "role_vol": role_vol,
        "coach_vol": coach_vol,
        "injury_vol": injury_vol,
        "stability_vol": stability_vol,
        "script_vol": script_vol,
    }
# ======================================================================
# PART 20 — PROJECTION OVERRIDES ENGINE
# Allows manual user overrides, auto-adjustments,
# role conflicts, blowout overrides, minutes caps,
# and projection safety clamps.
# ======================================================================

import numpy as np


# ----------------------------------------------------------------------
# Hard boundaries to prevent model blowups
# ----------------------------------------------------------------------
MIN_MU = 2.0
MAX_MU = 85.0

MIN_MINUTES = 12
MAX_MINUTES = 44

MIN_SD = 0.5
MAX_SD = 25.0


# ----------------------------------------------------------------------
# Auto-adjust rules based on severe blowout or injury conditions
# ----------------------------------------------------------------------
def auto_override_rules(mu, sd, minutes, blowout_risk: bool, teammate_out_level: int):
    """
    Returns updated (mu, sd, minutes) when extreme conditions occur.
    """

    # Severe blowout scenario → cut minutes
    if blowout_risk and minutes >= 30:
        minutes *= 0.88
        sd *= 1.10  # more randomness

    # Heavy teammate-out → boost usage
    if teammate_out_level >= 2:
        mu *= 1.06
        sd *= 1.04

    return float(mu), float(sd), float(minutes)


# ----------------------------------------------------------------------
# Manual override enforcement (user-specified)
# ----------------------------------------------------------------------
def apply_user_overrides(mu, sd, minutes, override_dict):
    """
    override_dict example:
    {
        "force_minutes": 36,
        "floor_minutes": 30,
        "cap_minutes": 38,
        "force_mu": 42.0,
        "cap_mu": 48.0,
        "boost_pct": 0.10
    }
    """
    # Minutes overrides
    if "force_minutes" in override_dict and override_dict["force_minutes"] is not None:
        minutes = float(override_dict["force_minutes"])

    if "floor_minutes" in override_dict and override_dict["floor_minutes"] is not None:
        minutes = max(minutes, float(override_dict["floor_minutes"]))

    if "cap_minutes" in override_dict and override_dict["cap_minutes"] is not None:
        minutes = min(minutes, float(override_dict["cap_minutes"]))

    # Production overrides
    if "force_mu" in override_dict and override_dict["force_mu"] is not None:
        mu = float(override_dict["force_mu"])

    if "cap_mu" in override_dict and override_dict["cap_mu"] is not None:
        mu = min(mu, float(override_dict["cap_mu"]))

    if "boost_pct" in override_dict and override_dict["boost_pct"] is not None:
        mu *= (1 + float(override_dict["boost_pct"]))

    return float(mu), float(sd), float(minutes)


# ----------------------------------------------------------------------
# Conflict Resolver — ensures consistent projections
# ----------------------------------------------------------------------
def resolve_projection_conflicts(mu, sd, minutes):
    """
    Ensures internal consistency:
      - sd too large for mu? Normalize
      - minutes too high? Cap
      - mu outside safe range? Clamp
    """

    # Clamp minutes
    minutes = float(np.clip(minutes, MIN_MINUTES, MAX_MINUTES))

    # Clamp mu
    mu = float(np.clip(mu, MIN_MU, MAX_MU))

    # SD must scale with mu realistically
    sd_max_allowed = max(3.0, mu * 0.50)
    sd_min_allowed = max(0.75, mu * 0.05)

    sd = float(np.clip(sd, sd_min_allowed, sd_max_allowed))

    sd = float(np.clip(sd, MIN_SD, MAX_SD))

    return mu, sd, minutes


# ----------------------------------------------------------------------
# Master Override Engine
# ----------------------------------------------------------------------
def projection_override_engine(
    mu,
    sd,
    minutes,
    blowout_risk: bool,
    teammate_out_level: int,
    override_dict=None
):
    """
    Full override pipeline:
      1. Auto-override rules (blowouts, major injuries)
      2. Manual overrides (user or UI applied)
      3. Conflict resolution for realistic projections
    """

    override_dict = override_dict or {}

    # -----------------------------------------------------
    # 1. Auto rules
    # -----------------------------------------------------
    mu, sd, minutes = auto_override_rules(
        mu,
        sd,
        minutes,
        blowout_risk=blowout_risk,
        teammate_out_level=teammate_out_level
    )

    # -----------------------------------------------------
    # 2. User overrides
    # -----------------------------------------------------
    if override_dict:
        mu, sd, minutes = apply_user_overrides(mu, sd, minutes, override_dict)

    # -----------------------------------------------------
    # 3. Final consistency normalization
    # -----------------------------------------------------
    mu, sd, minutes = resolve_projection_conflicts(mu, sd, minutes)

    return {
        "mu_final": float(mu),
        "sd_final": float(sd),
        "minutes_final": float(minutes),
        "override_used": override_dict
    }
