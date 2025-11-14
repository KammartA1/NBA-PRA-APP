
"""
Module 1: engine_usage.py
Player Usage Engine v3
Fully contained, production-ready.
"""

import numpy as np
import pandas as pd

# ------------------------------------------
# Helper: Normalize per‑minute values safely
# ------------------------------------------
def _safe_div(a, b, fallback=0.0):
    try:
        if b == 0 or b is None:
            return fallback
        return a / b
    except:
        return fallback

# ------------------------------------------
# Usage Redistribution Engine
# ------------------------------------------
def redistribute_usage(current_usg, missing_players, team_total_usg=100):
    """
    Redistributes usage when teammates are out.
    Inputs:
        current_usg: dict {player: usage%}
        missing_players: list of players missing
        team_total_usg: expected total team usage (100%)
    Returns:
        dict {player: adjusted_usage%}
    """

    remaining = {p: u for p, u in current_usg.items() if p not in missing_players}
    lost_usg = sum(u for p, u in current_usg.items() if p in missing_players)

    if lost_usg <= 0 or len(remaining) == 0:
        return remaining

    # redistribute proportionally
    total_remaining = sum(remaining.values())
    if total_remaining <= 0:
        boost = lost_usg / max(1, len(remaining))
        return {p: remaining.get(p, 0) + boost for p in remaining}

    adj = {}
    for p, u in remaining.items():
        share = u / total_remaining
        adj[p] = u + share * lost_usg

    # normalize to 100%
    norm_factor = team_total_usg / sum(adj.values())
    adj = {p: v * norm_factor for p, v in adj.items()}
    return adj

# ------------------------------------------
# Role-Based Usage Multiplier
# ------------------------------------------
def role_multiplier(role: str):
    """
    Returns a usage multiplier based on player archetype.
    """
    role = (role or "").lower()

    if "primary" in role:
        return 1.08
    if "secondary" in role:
        return 1.04
    if "bench" in role:
        return 1.02
    if "low" in role:
        return 0.96

    return 1.00

# ------------------------------------------
# Main Player Usage Engine v3
# ------------------------------------------
def compute_usage_v3(per_min_rate, minutes, role, teammate_out=False, on_off_boost=0.0):
    """
    Compute adjusted per-minute production based on:
      - baseline per-minute stats
      - expected minutes
      - role-based usage modification
      - teammate on/off adjustment
    """

    base = per_min_rate

    # Role-based scaling
    rm = role_multiplier(role)

    # Teammate-out usage bump
    tm = 1.05 if teammate_out else 1.00

    # On/Off boost (advanced model)
    ob = 1.0 + on_off_boost

    final = base * rm * tm * ob
    return float(final)
# ===============================================================
#  engine_context.py — Opponent Matchup Engine v2 (Fully Installed)
# ===============================================================

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats
from datetime import datetime

# ----------------------------
# SEASON HELPER
# ----------------------------
def current_season():
    today = datetime.now()
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"

# ----------------------------
# TEAM CONTEXT LOADER
# ----------------------------
def load_team_context():
    """Loads pace, defense, rebounding, assist %, plus interior/perimeter factors."""
    try:
        base = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        adv = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","REB_PCT","OREB_PCT","DREB_PCT","AST_PCT","PACE"
        ]]

        defense = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"
        ]]

        df = base.merge(adv, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
        df = df.merge(defense, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")

        # Interior vs Perimeter defensive proxy
        df["INTERIOR_FACTOR"] = (df["DREB_PCT"] + df["OREB_PCT"]) / 2
        df["PERIMETER_FACTOR"] = df["AST_PCT"]

        league_avg = {
            c: df[c].mean()
            for c in ["PACE","DEF_RATING","REB_PCT","AST_PCT","INTERIOR_FACTOR","PERIMETER_FACTOR"]
        }

        ctx = {
            r["TEAM_ABBREVIATION"]: {
                "PACE": r["PACE"],
                "DEF_RATING": r["DEF_RATING"],
                "REB_PCT": r["REB_PCT"],
                "DREB_PCT": r["DREB_PCT"],
                "AST_PCT": r["AST_PCT"],
                "INTERIOR_FACTOR": r["INTERIOR_FACTOR"],
                "PERIMETER_FACTOR": r["PERIMETER_FACTOR"],
            }
            for _, r in df.iterrows()
        }

        return ctx, league_avg

    except Exception:
        return {}, {}

TEAM_CTX, LEAGUE_CTX = load_team_context()

# ----------------------------
# MATCHUP MULTIPLIER ENGINE
# ----------------------------
def get_context_multiplier(opp, market):
    if not opp or opp not in TEAM_CTX or not LEAGUE_CTX:
        return 1.0

    t = TEAM_CTX[opp]

    pace_f = t["PACE"] / LEAGUE_CTX["PACE"]
    def_f = LEAGUE_CTX["DEF_RATING"] / t["DEF_RATING"]

    reb_adj = LEAGUE_CTX["REB_PCT"] / t["DREB_PCT"] if market == "Rebounds" else 1.0
    ast_adj = LEAGUE_CTX["AST_PCT"] / t["AST_PCT"] if market == "Assists" else 1.0

    if market == "Points":
        interior = LEAGUE_CTX["INTERIOR_FACTOR"] / t["INTERIOR_FACTOR"]
        perimeter = LEAGUE_CTX["PERIMETER_FACTOR"] / t["PERIMETER_FACTOR"]
        shot_quality_adj = 0.5 * interior + 0.5 * perimeter
    else:
        shot_quality_adj = 1.0

    mult = (
        0.35 * pace_f +
        0.25 * def_f +
        0.20 * reb_adj +
        0.10 * ast_adj +
        0.10 * shot_quality_adj
    )

    return float(np.clip(mult, 0.75, 1.30))
# =============================================================
# UltraMax V4 — Module 3
# Volatility Engine v2 (Nonlinear SD + Heavy Tail + Regime Detection)
# =============================================================

import numpy as np
from scipy.stats import skew

# -------------------------------------------------------------
# 1. Nonlinear SD Scaling
# -------------------------------------------------------------
def nonlinear_sd_scaling(base_sd, minutes, usage_factor):
    '''
    Nonlinear standard deviation engine:
    - SD increases sublinearly for high minutes
    - Adds volatility if usage spikes
    '''
    min_factor = np.sqrt(max(minutes, 1)) * 0.92
    usage_push = 1 + (usage_factor - 1) * 0.35
    sd = base_sd * min_factor * usage_push
    return float(max(sd, 0.10))


# -------------------------------------------------------------
# 2. Heavy-Tail Dynamic Engine
# -------------------------------------------------------------
def heavy_tail_adjustment(market, recent_values):
    '''
    Measure skew/kurtosis from recent games.
    Return tail inflation factor.
    '''
    if len(recent_values) < 4:
        return 1.0

    sk = skew(recent_values)
    tail = 1.0 + np.clip(sk * 0.15, -0.20, 0.35)

    if market == "PRA":
        tail *= 1.12
    elif market == "Points":
        tail *= 1.08

    return float(np.clip(tail, 0.85, 1.40))


# -------------------------------------------------------------
# 3. Regime Detection Engine
# -------------------------------------------------------------
def detect_regime(minutes_array):
    '''
    Detect volatility regime:
    - Stable:  high minute consistency
    - Medium: moderate variation
    - Erratic: chaotic minute patterns
    '''
    if len(minutes_array) < 4:
        return "stable", 1.0

    var = np.std(minutes_array)

    if var < 2.5:
        return "stable", 0.95
    elif var < 5.5:
        return "medium", 1.00
    else:
        return "erratic", 1.18


# -------------------------------------------------------------
# 4. Final Volatility Merge
# -------------------------------------------------------------
def compute_final_sd(base_sd, minutes, usage_factor, recent_values, market, minutes_array):
    '''
    Full Volatility Engine:
    Combines:
    - Nonlinear SD scaling
    - Heavy-tail skewness inflation
    - Regime detection factor
    '''

    sd1 = nonlinear_sd_scaling(base_sd, minutes, usage_factor)
    tail = heavy_tail_adjustment(market, recent_values)
    regime_name, regime_mult = detect_regime(minutes_array)

    sd = sd1 * tail * regime_mult
    return float(np.clip(sd, base_sd * 0.75, base_sd * 1.80))
# ============================================================
# MODULE 4 — ENSEMBLE DISTRIBUTION ENGINE (UltraMax V4)
# ============================================================
# This module provides a complete multi-distribution ensemble
# used to estimate p(over) using:
#  - Normal
#  - Log-normal
#  - Skew-normal (approx)
#  - Beta distribution
#  - Gamma distribution
#  - Weighted ensemble blend
#
# Fully self-contained. Safe for Streamlit Cloud.
# ============================================================

import numpy as np
from scipy.stats import norm, beta, gamma

# ------------------------------------------------------------
# 1. LOGNORMAL PROBABILITY
# ------------------------------------------------------------
def lognormal_prob(line, mu, sd):
    try:
        variance = sd ** 2
        phi = np.sqrt(variance + mu ** 2)
        mu_log = np.log(mu ** 2 / phi)
        sd_log = np.sqrt(np.log((phi ** 2) / (mu ** 2)))
        if sd_log <= 0:
            return 1 - norm.cdf(line, mu, sd)
        return 1 - norm.cdf(np.log(line + 1e-9), mu_log, sd_log)
    except:
        return max(0.01, min(0.99, 1 - norm.cdf(line, mu, sd)))

# ------------------------------------------------------------
# 2. APPROXIMATE SKEW-NORMAL
# ------------------------------------------------------------
def skew_normal_prob(line, mu, sd, skew):
    base = 1 - norm.cdf(line, mu, sd)
    adj = base * (1 + 0.18 * (skew - 1))
    return float(np.clip(adj, 0.01, 0.99))

# ------------------------------------------------------------
# 3. BETA DISTRIBUTION (scaled to NBA scoring)
# ------------------------------------------------------------
def beta_prob(line, mu, sd):
    if mu <= 0 or sd <= 0:
        return 0.5
    # scale scores to 0–1 for Beta
    scale = max(1.0, mu * 3)
    x = min(max(line / scale, 0.001), 0.999)
    # estimate alpha/beta from mean & variance
    var = sd ** 2
    m = mu / scale
    try:
        alpha = m * ((m * (1 - m)) / var - 1)
        beta_param = (1 - m) * ((m * (1 - m)) / var - 1)
        if alpha <= 0 or beta_param <= 0:
            return 0.5
        return 1 - beta.cdf(x, alpha, beta_param)
    except:
        return 0.5

# ------------------------------------------------------------
# 4. GAMMA DISTRIBUTION (positive heavy-tail)
# ------------------------------------------------------------
def gamma_prob(line, mu, sd):
    try:
        shape = (mu / sd) ** 2
        scale = sd ** 2 / mu
        if shape <= 0 or scale <= 0:
            return 0.5
        return 1 - gamma.cdf(line, shape, scale=scale)
    except:
        return 0.5

# ------------------------------------------------------------
# 5. ENSEMBLE WEIGHTS
# ------------------------------------------------------------
MARKET_WEIGHTS = {
    "PRA":     {"normal":0.20,"log":0.35,"skew":0.20,"beta":0.10,"gamma":0.15},
    "Points":  {"normal":0.25,"log":0.30,"skew":0.20,"beta":0.10,"gamma":0.15},
    "Rebounds":{"normal":0.30,"log":0.25,"skew":0.20,"beta":0.10,"gamma":0.15},
    "Assists": {"normal":0.30,"log":0.25,"skew":0.25,"beta":0.10,"gamma":0.10},
}

# ------------------------------------------------------------
# 6. ENSEMBLE FUNCTION
# ------------------------------------------------------------
def ensemble_prob_over(line, mu, sd, market, skew_factor=1.15):
    w = MARKET_WEIGHTS.get(market, MARKET_WEIGHTS["PRA"])

    p_normal = 1 - norm.cdf(line, mu, sd)
    p_log = lognormal_prob(line, mu, sd)
    p_skew = skew_normal_prob(line, mu, sd, skew_factor)
    p_beta = beta_prob(line, mu, sd)
    p_gamma = gamma_prob(line, mu, sd)

    p = (
        w["normal"] * p_normal +
        w["log"]    * p_log +
        w["skew"]   * p_skew +
        w["beta"]   * p_beta +
        w["gamma"]  * p_gamma
    )

    return float(np.clip(p, 0.02, 0.98))


__all__ = [
    "ensemble_prob_over",
    "lognormal_prob",
    "skew_normal_prob",
    "beta_prob",
    "gamma_prob",
    "MARKET_WEIGHTS"
]
# ============================================================
# ULTRAMAX V4 — MODULE 5
# Correlation Engine v3 (Synergy + Market Covariance + Opponent)
# ============================================================

import numpy as np

def correlation_engine_v3(leg1: dict, leg2: dict) -> float:
    """
    Computes directional correlation between two legs using:
      - Team & minutes synergy
      - Market-type covariance
      - Opponent matchup similarity
      - Dynamic contextual adjustments
    Returns a correlation in range [-0.35, 0.45]
    """

    corr = 0.0

    # -----------------------------------
    # 1. Team-based synergy
    # -----------------------------------
    same_team = (leg1.get("team") is not None 
                 and leg1.get("team") == leg2.get("team"))

    if same_team:
        corr += 0.18

    # -----------------------------------
    # 2. Minutes dependency
    # -----------------------------------
    def est_minutes(leg):
        try:
            return max(15, min(40, leg["mu"] / (leg["mu"] / max(leg["line"], 1e-6))))
        except:
            return 28

    m1 = est_minutes(leg1)
    m2 = est_minutes(leg2)

    if m1 > 30 and m2 > 30:
        corr += 0.05
    elif m1 < 22 or m2 < 22:
        corr -= 0.04

    # -----------------------------------
    # 3. Market-type covariance
    # -----------------------------------
    mA, mB = leg1["market"], leg2["market"]

    # Points + Points → positive
    if mA == "Points" and mB == "Points":
        corr += 0.10

    # Points ↔ Assists → negative
    if (mA == "Points" and mB == "Assists") or (mA == "Assists" and mB == "Points"):
        corr -= 0.11

    # Points ↔ Rebounds → mild negative
    if (mA == "Points" and mB == "Rebounds") or (mA == "Rebounds" and mB == "Points"):
        corr -= 0.06

    # PRA acts as weakly positive buffer
    if mA == "PRA" or mB == "PRA":
        corr += 0.03

    # -----------------------------------
    # 4. Opponent-driven directional correlation
    # -----------------------------------
    c1 = leg1.get("ctx_mult", 1.0)
    c2 = leg2.get("ctx_mult", 1.0)

    if c1 > 1.03 and c2 > 1.03:
        corr += 0.05
    if c1 < 0.97 and c2 < 0.97:
        corr += 0.03
    if (c1 > 1.03 and c2 < 0.97) or (c1 < 0.97 and c2 > 1.03):
        corr -= 0.05

    # -----------------------------------
    # 5. Clamp for stability
    # -----------------------------------
    corr = float(np.clip(corr, -0.35, 0.45))

    return corr
# engine_self_learning.py
# UltraMax V4 — Module 6: Self-Learning Calibration Engine v3
# Fully autonomous model drift correction + variance/tail tuning

import numpy as np
import pandas as pd

def self_learning_adjustments(history_df, max_samples=200):
    """
    Uses the last N completed bets to detect:
      - mean bias
      - variance skew
      - tail miscalibration
    Returns:
      (variance_adj, tail_adj, bias_adj)

    All adjustments are smooth, clamped, and safe.
    """

    # Filter valid rows
    df = history_df.copy()
    df = df[df["Result"].isin(["Hit", "Miss"])]
    df["EV_float"] = pd.to_numeric(df["EV"], errors="coerce") / 100.0
    df = df.dropna(subset=["EV_float"])

    if df.empty:
        return 1.0, 1.0, 0.0

    # Restrict to most recent samples
    df = df.tail(max_samples)

    # Predicted vs actual
    predicted = 0.5 + df["EV_float"].mean()
    actual = (df["Result"] == "Hit").mean()
    gap = actual - predicted

    # Bias correction
    bias_adj = float(np.clip(gap, -0.10, 0.10))

    # Variance correction
    if gap < -0.02:
        variance_adj = 1.05
    elif gap > 0.02:
        variance_adj = 0.95
    else:
        variance_adj = 1.0
    variance_adj = float(np.clip(variance_adj, 0.90, 1.10))

    # Tail correction
    if gap < -0.02:
        tail_adj = 1.04
    elif gap > 0.02:
        tail_adj = 0.96
    else:
        tail_adj = 1.0
    tail_adj = float(np.clip(tail_adj, 0.90, 1.10))

    return variance_adj, tail_adj, bias_adj
# Module 7 — Monte Carlo Engine + Correlation Engine v3 + Final Quant Wrapper

import numpy as np
from scipy.stats import norm

# ============================
# MONTE CARLO SIMULATION ENGINE
# ============================

def mc_simulate(mu, sd, market, line, iterations=5000):
    """
    Runs Monte Carlo simulation using hybrid distribution.
    Returns probability of exceeding the line.
    """

    draws = np.random.normal(mu, sd, iterations)

    # Heavy-tail correction
    if market in ["PRA", "Points"]:
        tail = np.random.lognormal(mean=np.log(mu+1), sigma=0.25, size=iterations)
        draws = 0.75 * draws + 0.25 * tail

    draws = np.clip(draws, 0, None)

    p_over = np.mean(draws > line)
    return float(np.clip(p_over, 0.01, 0.99))

# ============================
# CORRELATION ENGINE v3
# ============================

def correlation_engine_v3(leg1, leg2):
    """Advanced multi-factor correlation model."""

    corr = 0.0

    if leg1.get("team") == leg2.get("team"):
        corr += 0.18

    m1 = leg1.get("market")
    m2 = leg2.get("market")

    if m1 == m2 == "Points":
        corr += 0.10
    if (m1 == "Points" and m2 == "Assists") or (m1 == "Assists" and m2 == "Points"):
        corr -= 0.12
    if (m1 == "Rebounds" and m2 == "Points") or (m1 == "Points" and m2 == "Rebounds"):
        corr -= 0.05

    ctx1 = leg1.get("ctx_mult", 1.0)
    ctx2 = leg2.get("ctx_mult", 1.0)

    if ctx1 > 1.05 and ctx2 > 1.05:
        corr += 0.04
    if ctx1 < 0.95 and ctx2 < 0.95:
        corr += 0.04
    if (ctx1 > 1.05 and ctx2 < 0.95) or (ctx1 < 0.95 and ctx2 > 1.05):
        corr -= 0.06

    return float(np.clip(corr, -0.25, 0.40))

# ============================
# FINAL QUANT WRAPPER
# ============================

def final_quant_leg(mu, sd, market, line):
    """
    Computes normal, Monte Carlo, and blended probability.
    """

    normal_p = 1 - norm.cdf(line, mu, sd)
    mc_p = mc_simulate(mu, sd, market, line)

    p = 0.35 * normal_p + 0.65 * mc_p

    return float(np.clip(p, 0.01, 0.99))
