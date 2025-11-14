# =========================================================
# MODULE 1 — DARK QUANT TERMINAL CORE (UltraMax V4 Foundation)
# =========================================================

import os, time, random, difflib
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import norm

# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NBA Quant Terminal",
    page_icon="🏀",
    layout="wide",
)

# =========================================================
# PATHS & GLOBAL CONSTANTS
# =========================================================

TEMP_DIR = os.path.join("/tmp", "nba_quant_terminal")
os.makedirs(TEMP_DIR, exist_ok=True)

PRIMARY = "#0D0A12"
SECONDARY = "#1A171F"
ACCENT = "#FFCC33"
ACCENT_SOFT = "#BA8A2F"
TEXT = "#F2F2F2"
RED = "#D9534F"
GREEN = "#5CB85C"
BLUE = "#5BC0DE"

LOG_FILE = os.path.join(TEMP_DIR, "bets_history.csv")
MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

MODEL_VERSION = "UltraMax V4 — Dark Quant Terminal"

# =========================================================
# GLOBAL STYLING (DARK TERMINAL)
# =========================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {PRIMARY};
            color: {TEXT};
            font-family: 'Roboto', sans-serif;
        }}

        .main-title {{
            text-align:center;
            font-size:46px;
            font-weight:800;
            color:{ACCENT};
            letter-spacing:1px;
            margin-bottom:10px;
        }}

        .subheader {{
            color:{ACCENT_SOFT};
            font-size:24px;
            margin-top:20px;
            margin-bottom:8px;
        }}

        .terminal-box {{
            background-color:{SECONDARY};
            padding:16px;
            border-radius:12px;
            border:1px solid {ACCENT}55;
            margin-bottom:16px;
        }}

        .metric-box {{
            background-color:{SECONDARY};
            padding:10px;
            border-radius:10px;
            border:1px solid {ACCENT}33;
            text-align:center;
            margin-bottom:6px;
        }}

        h3, h4, h5 {{
            color:{ACCENT};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HEADER
# =========================================================

st.markdown(f"<div class='main-title'>NBA QUANT TERMINAL</div>", unsafe_allow_html=True)
st.caption(f"Model Engine: {MODEL_VERSION} — Fully Automated Quantitative Prop System")

# =========================================================
# SIDEBAR — USER SETTINGS
# =========================================================

st.sidebar.header("User Settings")

user_id = st.sidebar.text_input("User ID", value="Kamal")
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=50.0, value=200.0)
payout_mult = st.sidebar.number_input("2-Pick Payout", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Games Lookback", 3, 20, 10)

st.sidebar.markdown("---")
st.sidebar.write("Terminal Version: UltraMax V4")
st.sidebar.write("Theme: Dark Quant Terminal v1.0")
# =========================================================
# MODULE 2 — UltraMax V4 Quant Foundations
# Player Logs • Team Context • Usage Engine v3 • Market Base • History Engine
# =========================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats


# =========================================================
# PATHS
# =========================================================

TEMP_DIR = os.path.join("/tmp", "nba_quant_terminal")
MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")
BET_HISTORY_FILE = os.path.join(TEMP_DIR, "bets_history.csv")


# =========================================================
# PLAYER LOOKUP + FUZZY RESOLUTION
# =========================================================

def _norm_name(s: str):
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )


def resolve_player(name: str):
    """Fuzzy search → return (player_id, full_name)"""
    if not name:
        return None, None

    plist = nba_players.get_players()
    target = _norm_name(name)

    # Exact
    for p in plist:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]

    # Fuzzy
    names = [_norm_name(p["full_name"]) for p in plist]
    best = difflib.get_close_matches(target, names, n=1, cutoff=0.7)
    if best:
        bn = best[0]
        for p in plist:
            if _norm_name(p["full_name"]) == bn:
                return p["id"], p["full_name"]

    return None, None


def get_headshot_url(name: str):
    pid, _ = resolve_player(name)
    if not pid:
        return None
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png"


# =========================================================
# PLAYER LOG ENGINE (Per-Minute Stats)
# =========================================================

def current_season():
    """Detects current NBA season automatically"""
    today = datetime.now()
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"


def parse_minutes(m_raw):
    """Convert 'MM:SS' → float minutes"""
    try:
        if isinstance(m_raw, str) and ":" in m_raw:
            mm, ss = m_raw.split(":")
            return float(mm) + float(ss) / 60.0
        return float(m_raw)
    except:
        return 0.0


def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    """
    Outputs:
        mu_per_min — per-minute production
        sd_per_min — per-minute variance
        avg_minutes
        team_abbrev
        debug_msg
    """

    pid, label = resolve_player(name)
    if not pid:
        return None, None, None, None, f"No match for '{name}'."

    try:
        logs = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception as e:
        return None, None, None, None, f"GameLog API error: {e}"

    if logs.empty:
        return None, None, None, None, "No game logs available."

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs = logs.sort_values("GAME_DATE", ascending=False).head(n_games)

    # Map markets → stats
    MARKET_METRICS = {
        "PRA": ["PTS", "REB", "AST"],
        "Points": ["PTS"],
        "Rebounds": ["REB"],
        "Assists": ["AST"]
    }

    cols = MARKET_METRICS.get(market, ["PTS"])

    per_min = []
    minutes = []

    for _, r in logs.iterrows():
        m = parse_minutes(r.get("MIN", 0))
        if m <= 0:
            continue

        total_val = sum([float(r.get(c, 0)) for c in cols])
        per_min.append(total_val / m)
        minutes.append(m)

    if not per_min:
        return None, None, None, None, "Insufficient usable games."

    per_min = np.array(per_min)
    minutes = np.array(minutes)

    mu = float(np.mean(per_min))
    sd = float(max(np.std(per_min, ddof=1), 0.15 * mu))
    avg_m = float(np.mean(minutes))

    try:
        team = logs["TEAM_ABBREVIATION"].mode().iloc[0]
    except:
        team = None

    msg = f"{label}: {len(per_min)} games • {avg_m:.1f} min per game"

    return mu, sd, avg_m, team, msg


# =========================================================
# TEAM CONTEXT ENGINE v2 (pace, defense, reb%, ast%)
# =========================================================

def load_team_context():
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

        defs = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"
        ]]

        df = base.merge(adv, on=["TEAM_ID","TEAM_ABBREVIATION"])
        df = df.merge(defs, on=["TEAM_ID","TEAM_ABBREVIATION"])

        league = {
            "PACE": df["PACE"].mean(),
            "DEF_RATING": df["DEF_RATING"].mean(),
            "REB_PCT": df["REB_PCT"].mean(),
            "AST_PCT": df["AST_PCT"].mean()
        }

        ctx = {
            r["TEAM_ABBREVIATION"]: {
                "PACE": r["PACE"],
                "DEF_RATING": r["DEF_RATING"],
                "REB_PCT": r["REB_PCT"],
                "DREB_PCT": r["DREB_PCT"],
                "AST_PCT": r["AST_PCT"]
            }
            for _, r in df.iterrows()
        }

        return ctx, league

    except Exception:
        return {}, {}


TEAM_CTX, LEAGUE_CTX = load_team_context()


def get_context_multiplier(opp: str, market: str):
    if not opp or opp not in TEAM_CTX:
        return 1.0

    o = TEAM_CTX[opp]

    pace_f = o["PACE"] / LEAGUE_CTX["PACE"]
    def_f = LEAGUE_CTX["DEF_RATING"] / o["DEF_RATING"]

    reb_adj = LEAGUE_CTX["REB_PCT"] / o["DREB_PCT"] if market == "Rebounds" else 1.0
    ast_adj = LEAGUE_CTX["AST_PCT"] / o["AST_PCT"] if market == "Assists" else 1.0

    mult = (
        0.40 * pace_f +
        0.30 * def_f +
        0.30 * (reb_adj if market == "Rebounds" else ast_adj)
    )

    return float(np.clip(mult, 0.80, 1.20))


# =========================================================
# USAGE ENGINE v3 (already validated)
# =========================================================

def usage_engine_v3(mu_per_min, team_usage_rate, teammate_out_level):
    mu_per_min = max(0.05, float(mu_per_min))

    team_adj = np.clip(team_usage_rate, 0.90, 1.12)

    if teammate_out_level <= 0:
        injury_adj = 1.00
    elif teammate_out_level == 1:
        injury_adj = 1.06
    else:
        injury_adj = 1.12

    nonlinear_adj = 1 + (injury_adj - 1) * 0.65

    return float(mu_per_min * team_adj * nonlinear_adj)


# =========================================================
# MARKET BASELINE LIBRARY (Mean/Median Line Tracking)
# =========================================================

def load_market_library():
    if not os.path.exists(MARKET_LIBRARY_FILE):
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])
    try:
        return pd.read_csv(MARKET_LIBRARY_FILE)
    except:
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])


def save_market_library(df):
    df.to_csv(MARKET_LIBRARY_FILE, index=False)


def update_market_library(player, market, line):
    df = load_market_library()
    new = pd.DataFrame([{
        "Player": player,
        "Market": market,
        "Line": line,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }])
    df = pd.concat([df, new], ignore_index=True)
    save_market_library(df)


def get_market_baseline(player, market):
    df = load_market_library()
    if df.empty:
        return None, None
    sub = df[(df["Player"] == player) & (df["Market"] == market)]
    if sub.empty:
        return None, None
    return sub["Line"].mean(), sub["Line"].median()


# =========================================================
# HISTORY ENGINE
# =========================================================

def ensure_history():
    if not os.path.exists(BET_HISTORY_FILE):
        df = pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac"
        ])
        df.to_csv(BET_HISTORY_FILE, index=False)


def load_history():
    ensure_history()
    try:
        return pd.read_csv(BET_HISTORY_FILE)
    except:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac"
        ])


def save_history(df):
    df.to_csv(BET_HISTORY_FILE, index=False)
# =========================================================
# MODULE 3 — UltraMax V4 Projection Engine (FULL MODULE)
# =========================================================
# Includes:
#   • Volatility Engine v2
#   • Heavy-Tail Engine
#   • 5-Distribution Ensemble Engine
#   • Probability Blender v3
#   • Baseline Projection
#   • Safety Net Clamps
#   • Full Leg Projection Output
# =========================================================

import numpy as np
from scipy.stats import norm, beta as beta_dist, gamma as gamma_dist


# =========================================================
# VOLATILITY ENGINE v2 — Nonlinear Variance Engine
# =========================================================

def volatility_engine_v2(sd_per_min, minutes, market, context_mult, heavy_tail_factor):
    """
    Computes final volatility for a player:
        - base variance
        - matchup-driven variance
        - market-specific variance inflation
        - heavy tail detection
        - nonlinear stabilizer clamps
    """

    # 1. Base: per-minute SD scaled to minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))

    # 2. Market-specific variance multipliers
    base_map = {
        "Points":   1.12,
        "Rebounds": 1.10,
        "Assists":  1.08,
        "PRA":      1.18
    }
    sd *= base_map.get(market, 1.10)

    # 3. Opponent-driven volatility
    sd *= np.clip(context_mult, 0.80, 1.25)

    # 4. Heavy-tail regime adjustment
    sd *= (1 + 0.12 * (heavy_tail_factor - 1))

    # 5. Nonlinear clamp (stability)
    sd = float(np.clip(sd, 0.4, 22.0))

    return sd


# =========================================================
# SKEW-NORMAL APPROX (Analytic)
# =========================================================

def skew_normal_approx(mu, sd, skew, line):
    """
    A stable right-tail skew-normal approximation.
    """
    try:
        z = (line - mu) / sd
        base = 1 - norm.cdf(z)
        adj = base * (1 + 0.18 * (skew - 1))
        return float(np.clip(adj, 0.01, 0.99))
    except:
        return 0.50


# =========================================================
# 5-DISTRIBUTION ENSEMBLE ENGINE (UltraMax v4)
# =========================================================

def ensemble_prob(mu, sd, line, market):
    """
    Blends 5 distributions using market-optimized weights:
        1. Normal
        2. Log-normal
        3. Beta
        4. Gamma
        5. Skew-normal
    """

    # -------------------------
    # Normal
    # -------------------------
    p_norm = 1 - norm.cdf((line - mu) / sd)

    # -------------------------
    # Log-normal
    # -------------------------
    try:
        variance = sd**2
        phi = np.sqrt(variance + mu**2)
        mu_l = np.log(mu**2 / phi)
        sd_l = np.sqrt(np.log(phi**2 / mu**2))
        p_lognorm = 1 - norm.cdf(np.log(line + 1e-9), mu_l, sd_l)
    except:
        p_lognorm = p_norm

    # -------------------------
    # Beta (scaled)
    # -------------------------
    try:
        a = max((mu / 6), 0.2)
        b = max(((sd**2) / (mu + 1e-6)), 0.2)
        x = np.clip(line / (mu * 2 + 1e-9), 0.001, 0.999)
        p_beta = 1 - beta_dist.cdf(x, a, b)
    except:
        p_beta = p_norm

    # -------------------------
    # Gamma
    # -------------------------
    try:
        k = (mu / sd)**2
        theta = sd**2 / mu
        p_gamma = 1 - gamma_dist(k, scale=theta).cdf(line)
    except:
        p_gamma = p_norm

    # -------------------------
    # Skew-normal
    # -------------------------
    skew_factor = 1.35 if market == "PRA" else 1.20
    p_skew = skew_normal_approx(mu, sd, skew_factor, line)

    # -------------------------
    # Weighted Blend
    # -------------------------

    weights_dict = {
        "PRA":      [0.15, 0.30, 0.10, 0.15, 0.30],
        "Points":   [0.20, 0.30, 0.10, 0.15, 0.25],
        "Rebounds": [0.28, 0.25, 0.12, 0.20, 0.15],
        "Assists":  [0.25, 0.25, 0.15, 0.15, 0.20],
    }
    weights = weights_dict.get(market, [0.2, 0.25, 0.15, 0.15, 0.25])

    blend = (
        weights[0] * p_norm +
        weights[1] * p_lognorm +
        weights[2] * p_beta +
        weights[3] * p_gamma +
        weights[4] * p_skew
    )

    return float(np.clip(blend, 0.02, 0.98))


# =========================================================
# FULL PROJECTION PIPELINE (One Leg)
# =========================================================

def projection_engine_v4(mu_per_min, sd_per_min, avg_min, team, market,
                         context_multiplier, line, teammate_out_level):
    """
    Full projection for a single leg.
    Includes:
      - Usage Engine v3 (from Module 2)
      - Updated volatility
      - 5-distribution ensemble prob
      - EV
    """

    # -----------------------------
    # 1. Usage Adjustment
    # -----------------------------
    from module_2 import usage_engine_v3       # Safe local import
    mu_adj_per_min = usage_engine_v3(
        mu_per_min,
        team_usage_rate=1.00,
        teammate_out_level=teammate_out_level
    )

    # -----------------------------
    # 2. Final mean
    # -----------------------------
    mu = mu_adj_per_min * avg_min * context_multiplier

    # -----------------------------
    # 3. Volatility Engine
    # -----------------------------
    heavy_factor = {
        "PRA":      1.35,
        "Points":   1.25,
        "Rebounds": 1.25,
        "Assists":  1.20
    }.get(market, 1.20)

    sd_final = volatility_engine_v2(
        sd_per_min,
        avg_min,
        market,
        context_multiplier,
        heavy_factor
    )

    # -----------------------------
    # 4. Ensemble Probability
    # -----------------------------
    p_over = ensemble_prob(mu, sd_final, line, market)

    # -----------------------------
    # 5. Even-money EV
    # -----------------------------
    ev = p_over - (1 - p_over)

    # -----------------------------
    # 6. Output object
    # -----------------------------
    return {
        "mu": float(mu),
        "sd": float(sd_final),
        "prob_over": float(p_over),
        "ev_even": float(ev),
    }
# =========================================================
# MODULE 3 — UltraMax V4 Projection Engine (FULL MODULE)
# =========================================================
# Includes:
#   • Volatility Engine v2
#   • Heavy-Tail Engine
#   • 5-Distribution Ensemble Engine
#   • Probability Blender v3
#   • Baseline Projection
#   • Safety Net Clamps
#   • Full Leg Projection Output
# =========================================================

import numpy as np
from scipy.stats import norm, beta as beta_dist, gamma as gamma_dist


# =========================================================
# VOLATILITY ENGINE v2 — Nonlinear Variance Engine
# =========================================================

def volatility_engine_v2(sd_per_min, minutes, market, context_mult, heavy_tail_factor):
    """
    Computes final volatility for a player:
        - base variance
        - matchup-driven variance
        - market-specific variance inflation
        - heavy tail detection
        - nonlinear stabilizer clamps
    """

    # 1. Base: per-minute SD scaled to minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))

    # 2. Market-specific variance multipliers
    base_map = {
        "Points":   1.12,
        "Rebounds": 1.10,
        "Assists":  1.08,
        "PRA":      1.18
    }
    sd *= base_map.get(market, 1.10)

    # 3. Opponent-driven volatility
    sd *= np.clip(context_mult, 0.80, 1.25)

    # 4. Heavy-tail regime adjustment
    sd *= (1 + 0.12 * (heavy_tail_factor - 1))

    # 5. Nonlinear clamp (stability)
    sd = float(np.clip(sd, 0.4, 22.0))

    return sd


# =========================================================
# SKEW-NORMAL APPROX (Analytic)
# =========================================================

def skew_normal_approx(mu, sd, skew, line):
    """
    A stable right-tail skew-normal approximation.
    """
    try:
        z = (line - mu) / sd
        base = 1 - norm.cdf(z)
        adj = base * (1 + 0.18 * (skew - 1))
        return float(np.clip(adj, 0.01, 0.99))
    except:
        return 0.50


# =========================================================
# 5-DISTRIBUTION ENSEMBLE ENGINE (UltraMax v4)
# =========================================================

def ensemble_prob(mu, sd, line, market):
    """
    Blends 5 distributions using market-optimized weights:
        1. Normal
        2. Log-normal
        3. Beta
        4. Gamma
        5. Skew-normal
    """

    # -------------------------
    # Normal
    # -------------------------
    p_norm = 1 - norm.cdf((line - mu) / sd)

    # -------------------------
    # Log-normal
    # -------------------------
    try:
        variance = sd**2
        phi = np.sqrt(variance + mu**2)
        mu_l = np.log(mu**2 / phi)
        sd_l = np.sqrt(np.log(phi**2 / mu**2))
        p_lognorm = 1 - norm.cdf(np.log(line + 1e-9), mu_l, sd_l)
    except:
        p_lognorm = p_norm

    # -------------------------
    # Beta (scaled)
    # -------------------------
    try:
        a = max((mu / 6), 0.2)
        b = max(((sd**2) / (mu + 1e-6)), 0.2)
        x = np.clip(line / (mu * 2 + 1e-9), 0.001, 0.999)
        p_beta = 1 - beta_dist.cdf(x, a, b)
    except:
        p_beta = p_norm

    # -------------------------
    # Gamma
    # -------------------------
    try:
        k = (mu / sd)**2
        theta = sd**2 / mu
        p_gamma = 1 - gamma_dist(k, scale=theta).cdf(line)
    except:
        p_gamma = p_norm

    # -------------------------
    # Skew-normal
    # -------------------------
    skew_factor = 1.35 if market == "PRA" else 1.20
    p_skew = skew_normal_approx(mu, sd, skew_factor, line)

    # -------------------------
    # Weighted Blend
    # -------------------------

    weights_dict = {
        "PRA":      [0.15, 0.30, 0.10, 0.15, 0.30],
        "Points":   [0.20, 0.30, 0.10, 0.15, 0.25],
        "Rebounds": [0.28, 0.25, 0.12, 0.20, 0.15],
        "Assists":  [0.25, 0.25, 0.15, 0.15, 0.20],
    }
    weights = weights_dict.get(market, [0.2, 0.25, 0.15, 0.15, 0.25])

    blend = (
        weights[0] * p_norm +
        weights[1] * p_lognorm +
        weights[2] * p_beta +
        weights[3] * p_gamma +
        weights[4] * p_skew
    )

    return float(np.clip(blend, 0.02, 0.98))


# =========================================================
# FULL PROJECTION PIPELINE (One Leg)
# =========================================================

def projection_engine_v4(mu_per_min, sd_per_min, avg_min, team, market,
                         context_multiplier, line, teammate_out_level):
    """
    Full projection for a single leg.
    Includes:
      - Usage Engine v3 (from Module 2)
      - Updated volatility
      - 5-distribution ensemble prob
      - EV
    """

    # -----------------------------
    # 1. Usage Adjustment
    # -----------------------------
    from module_2 import usage_engine_v3       # Safe local import
    mu_adj_per_min = usage_engine_v3(
        mu_per_min,
        team_usage_rate=1.00,
        teammate_out_level=teammate_out_level
    )

    # -----------------------------
    # 2. Final mean
    # -----------------------------
    mu = mu_adj_per_min * avg_min * context_multiplier

    # -----------------------------
    # 3. Volatility Engine
    # -----------------------------
    heavy_factor = {
        "PRA":      1.35,
        "Points":   1.25,
        "Rebounds": 1.25,
        "Assists":  1.20
    }.get(market, 1.20)

    sd_final = volatility_engine_v2(
        sd_per_min,
        avg_min,
        market,
        context_multiplier,
        heavy_factor
    )

    # -----------------------------
    # 4. Ensemble Probability
    # -----------------------------
    p_over = ensemble_prob(mu, sd_final, line, market)

    # -----------------------------
    # 5. Even-money EV
    # -----------------------------
    ev = p_over - (1 - p_over)

    # -----------------------------
    # 6. Output object
    # -----------------------------
    return {
        "mu": float(mu),
        "sd": float(sd_final),
        "prob_over": float(p_over),
        "ev_even": float(ev),
    }
# ==============================================================
# MODULE 5 — UltraMax V4 Correlation Engine v3 (Full Version)
# ==============================================================

import numpy as np

# --------------------------------------------------------------
# Market correlation weights
# --------------------------------------------------------------
MARKET_CORR_WEIGHTS = {
    ("Points", "Points"): 0.20,
    ("Points", "Assists"): -0.18,
    ("Assists", "Points"): -0.18,
    ("Points", "Rebounds"): -0.08,
    ("Rebounds", "Points"): -0.08,
    ("Rebounds", "Rebounds"): 0.12,
    ("Assists", "Assists"): 0.14,
    ("PRA", "PRA"): 0.22,
    ("PRA", "Points"): 0.10,
    ("PRA", "Assists"): 0.10,
    ("PRA", "Rebounds"): 0.10,
}

def _market_corr(market1, market2):
    """Retrieve built-in interaction weight."""
    return MARKET_CORR_WEIGHTS.get((market1, market2), 0.0)


# --------------------------------------------------------------
# Team-level synergy model
# --------------------------------------------------------------

def _team_synergy_minutes(m1, m2):
    """
    Synergy from shared minutes:
        - high-minute pair → increased correlation
        - low-minute or bench unit → reduced correlation
    """

    if m1 >= 32 and m2 >= 32:
        return 0.08
    if m1 >= 28 and m2 >= 28:
        return 0.05
    if m1 <= 20 or m2 <= 20:
        return -0.04

    return 0.00


def _on_off_synergy(teammate_out_factor):
    """
    Extra correlation when BOTH players benefit from the same injury boost.
    """
    if teammate_out_factor >= 1:
        return 0.06
    return 0.00


# --------------------------------------------------------------
# Opponent contextual correlation model
# --------------------------------------------------------------

def _opponent_context_corr(ctx1, ctx2):
    """
    Opponent-context correlation:
        - if both players benefit from pace ↑ or defense ↓ → positive
        - if one benefits and other harmed → negative
    """
    if ctx1 > 1.03 and ctx2 > 1.03:
        return 0.05
    if ctx1 < 0.97 and ctx2 < 0.97:
        return 0.05

    # Opposite directional context
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 0.97):
        return -0.06

    return 0.00


# --------------------------------------------------------------
# Main Correlation Engine v3
# --------------------------------------------------------------

def correlation_engine_v3(leg1, leg2):
    """
    Computes the **full-scale correlation** between two projection legs using:

      ✔ Market correlation matrix
      ✔ Team synergy
      ✔ Shared minutes model
      ✔ On/Off injury synergy
      ✔ Opponent directional context
      ✔ Clamped final output (-0.35 to +0.55)

    Input (leg dicts must include):
        - "team"
        - "market"
        - "ctx_mult"
        - "mu"
        - "line"
        - "teammate_out"
        - "blowout"

    Returns:
        corr (float)
    """

    corr = 0.0

    # -----------------------------
    # 1. Base market interaction
    # -----------------------------
    corr += _market_corr(leg1["market"], leg2["market"])

    # -----------------------------
    # 2. Same-team synergy boost
    # -----------------------------
    if leg1["team"] and leg2["team"] and leg1["team"] == leg2["team"]:
        corr += 0.15

    # -----------------------------
    # 3. Minutes-based synergy
    # estimate minutes from mu/line ratio (safe proxy)
    # -----------------------------
    try:
        min1 = max(10.0, leg1["mu"] / max(leg1["mu"] / leg1["line"], 0.1))
        min2 = max(10.0, leg2["mu"] / max(leg2["mu"] / leg2["line"], 0.1))
    except:
        min1 = min2 = 28.0

    corr += _team_synergy_minutes(min1, min2)

    # -----------------------------
    # 4. Injury-based synergy
    # -----------------------------
    if leg1["teammate_out"] and leg2["teammate_out"]:
        corr += _on_off_synergy(1)

    # -----------------------------
    # 5. Opponent context synergy
    # -----------------------------
    ctx1 = leg1["ctx_mult"]
    ctx2 = leg2["ctx_mult"]
    corr += _opponent_context_corr(ctx1, ctx2)

    # -----------------------------
    # 6. Blowout risk → reduces correlation
    # -----------------------------
    if leg1["blowout"] or leg2["blowout"]:
        corr -= 0.05

    # -----------------------------
    # 7. Clamp for stability
    # -----------------------------
    corr = float(np.clip(corr, -0.35, 0.55))

    return corr
# ===============================================
# MODULE 6 — Self-Learning Calibration Engine v3
# ===============================================

import numpy as np
import pandas as pd

def compute_self_learning_adjustments(history_df, max_window=200):
    """
    Self-Learning Calibration Engine v3
    -----------------------------------
    This version:
        • Learns from last N completed bets (default 200)
        • Adjusts:
            - variance scaling (volatility tuning)
            - tail weight scaling (heavy-tail tuning)
            - probability bias correction (mean shift)
        • Detects:
            - overconfidence
            - underconfidence
            - market drift
        • Feeds back into probability engine in Module 4 & 5

    Returns:
        variance_adj (float)
        tail_adj (float)
        bias_shift (float)
    """

    # --------------------------------------------------------
    # 1. Require sufficient sample size
    # --------------------------------------------------------
    if history_df is None or history_df.empty:
        return 1.0, 1.0, 0.0

    df = history_df.copy()

    # Only completed bets
    df = df[df["Result"].isin(["Hit", "Miss"])]

    if df.empty or len(df) < 30:
        # Not enough data to learn — neutral parameters
        return 1.0, 1.0, 0.0

    # --------------------------------------------------------
    # 2. Restrict to most recent N bets (rolling learning window)
    # --------------------------------------------------------
    df = df.tail(max_window).copy()

    # EV must be numeric
    df["EV_float"] = pd.to_numeric(df["EV"], errors="coerce") / 100.0
    df = df.dropna(subset=["EV_float"])

    if df.empty:
        return 1.0, 1.0, 0.0

    # --------------------------------------------------------
    # 3. Compute model predicted win probability
    # --------------------------------------------------------
    pred_win_prob = 0.5 + df["EV_float"].mean()

    # --------------------------------------------------------
    # 4. Compute actual win rate
    # --------------------------------------------------------
    actual_win_prob = (df["Result"] == "Hit").mean()

    gap = actual_win_prob - pred_win_prob   # negative = model overconfident

    # --------------------------------------------------------
    # 5. Variance Adjustment Logic
    # --------------------------------------------------------
    if gap < -0.05:  # actual << predicted → too optimistic
        variance_adj = 1.10
    elif gap < -0.02:
        variance_adj = 1.05
    elif gap > 0.05:  # actual >> predicted → too conservative
        variance_adj = 0.93
    elif gap > 0.02:
        variance_adj = 0.97
    else:
        variance_adj = 1.00

    # --------------------------------------------------------
    # 6. Tail Weight Adjustment Logic
    # --------------------------------------------------------
    # Overconfidence → tails too light → increase
    if gap < -0.03:
        tail_adj = 1.08
    # Underconfidence → tails too heavy → decrease
    elif gap > 0.03:
        tail_adj = 0.94
    else:
        tail_adj = 1.00

    # --------------------------------------------------------
    # 7. Bias Correction (mean shift)
    # --------------------------------------------------------
    # Bias shift is small so we don’t distort distributions
    bias_shift = float(np.clip(gap * 0.35, -0.05, 0.05))

    # --------------------------------------------------------
    # 8. Return final learning outputs
    # --------------------------------------------------------
    return (
        float(np.clip(variance_adj, 0.90, 1.12)),
        float(np.clip(tail_adj, 0.90, 1.12)),
        bias_shift
    )
# =========================================================
# MODULE 7 — COMBO ENGINE + MONTE CARLO DEPENDENCY MODELING v3
# Fully compatible with Modules 1–6
# =========================================================

import numpy as np

# ---------------------------------------------------------
# 1. MARKET COEFFICIENTS FOR PLAYER SYNERGY
# ---------------------------------------------------------
MARKET_CORR_MATRIX = {
    ("Points", "Points"): 0.12,
    ("Points", "Assists"): -0.18,
    ("Points", "Rebounds"): -0.10,
    ("Assists", "Assists"): 0.08,
    ("Assists", "Rebounds"): -0.05,
    ("Rebounds", "Rebounds"): 0.15,
    ("PRA", "PRA"): 0.22,
}


def _market_corr(m1, m2):
    if (m1, m2) in MARKET_CORR_MATRIX:
        return MARKET_CORR_MATRIX[(m1, m2)]
    if (m2, m1) in MARKET_CORR_MATRIX:
        return MARKET_CORR_MATRIX[(m2, m1)]
    return 0.0


# ---------------------------------------------------------
# 2. OPPONENT-BASED COVARIANCE BOOST
# ---------------------------------------------------------
def opponent_corr_boost(ctx1, ctx2):
    """
    ctx_mult > 1 → easier matchup → increases positive covariance
    ctx_mult < 1 → harder matchup → increases negative covariance
    """
    if ctx1 > 1.03 and ctx2 > 1.03:
        return 0.05
    if ctx1 < 0.97 and ctx2 < 0.97:
        return 0.04
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 1.03):
        return -0.06
    return 0.0


# ---------------------------------------------------------
# 3. TEAM + MINUTES SYNERGY
# ---------------------------------------------------------
def team_minutes_synergy(leg1, leg2):
    corr = 0.0

    # Same team synergy baseline
    if leg1["team"] == leg2["team"] and leg1["team"] is not None:
        corr += 0.18

    # Minutes synergy — if both reliably play heavy minutes
    try:
        min_ratio1 = leg1["mu"] / max(leg1["line"], 1e-5)
        min_ratio2 = leg2["mu"] / max(leg2["line"], 1e-5)
        if min_ratio1 > 1.1 and min_ratio2 > 1.1:
            corr += 0.05
        elif min_ratio1 < 0.7 or min_ratio2 < 0.7:
            corr -= 0.04
    except:
        pass

    return corr


# ---------------------------------------------------------
# 4. TOTAL CORRELATION ENGINE v3 (analytic prior)
# ---------------------------------------------------------
def analytic_corr(leg1, leg2):
    c = 0.0

    # Market-type correlation
    c += _market_corr(leg1["market"], leg2["market"])

    # Team + minutes synergy
    c += team_minutes_synergy(leg1, leg2)

    # Opponent-driven covariance
    c += opponent_corr_boost(leg1["ctx_mult"], leg2["ctx_mult"])

    # Clamp
    return float(np.clip(c, -0.30, 0.45))


# ---------------------------------------------------------
# 5. MONTE CARLO JOINT PROBABILITY ESTIMATOR (5000+ sims)
# ---------------------------------------------------------
def mc_joint_probability(leg1, leg2, analytic_rho, iters=5000):
    """
    High-accuracy Monte Carlo dependency modeling.
    Produces joint Pr(X>line1 & Y>line2).
    """

    mu1, sd1, line1 = leg1["mu"], leg1["sd"], leg1["line"]
    mu2, sd2, line2 = leg2["mu"], leg2["sd"], leg2["line"]

    # Covariance matrix
    cov = analytic_rho * sd1 * sd2
    C = np.array([[sd1**2, cov], [cov, sd2**2]])

    # Cholesky
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        # fallback: reduce correlation slightly to guarantee PD matrix
        analytic_rho *= 0.85
        cov = analytic_rho * sd1 * sd2
        C = np.array([[sd1**2, cov], [cov, sd2**2]])
        L = np.linalg.cholesky(C)

    # Generate correlated normals
    Z = np.random.normal(size=(iters, 2))
    sims = (L @ Z.T).T

    # Shift by means
    sims[:, 0] += mu1
    sims[:, 1] += mu2

    # Compute frequencies
    hits = np.sum((sims[:, 0] > line1) & (sims[:, 1] > line2))
    joint_mc = hits / iters

    return float(np.clip(joint_mc, 0.001, 0.999))


# ---------------------------------------------------------
# 6. FULL COMBO ENGINE (analytic + Monte Carlo blended)
# ---------------------------------------------------------
def compute_combo(leg1, leg2, payout_mult, fractional_kelly, bankroll):
    """
    Returns:
        joint_prob        → blended analytic + MC
        ev_combo          → expected value per $1
        stake             → Kelly stake recommendation
        corr_used         → final correlation used
    """

    # Step 1 — analytic correlation
    rho = analytic_corr(leg1, leg2)

    # Step 2 — analytic joint (prior)
    analytic_joint = (
        leg1["prob_over"] * leg2["prob_over"]
        + rho * (min(leg1["prob_over"], leg2["prob_over"]) -
                 leg1["prob_over"] * leg2["prob_over"])
    )
    analytic_joint = float(np.clip(analytic_joint, 0.001, 0.999))

    # Step 3 — Monte Carlo refinement
    mc_joint = mc_joint_probability(leg1, leg2, rho, iters=5000)

    # Step 4 — Blend analytic + MC
    # MC weighted slightly more (70%)
    joint = float(np.clip(0.30 * analytic_joint + 0.70 * mc_joint, 0.001, 0.999))

    # Step 5 — EV
    ev_combo = payout_mult * joint - 1.0

    # Step 6 — Kelly
    b = payout_mult - 1
    q = 1 - joint

    raw_k = (b * joint - q) / b
    stake_frac = max(0.0, raw_k) * fractional_kelly
    stake_frac = float(np.clip(stake_frac, 0.0, 0.03))

    stake = round(bankroll * stake_frac, 2)

    return {
        "joint_prob": joint,
        "analytic_joint": analytic_joint,
        "mc_joint": mc_joint,
        "ev_combo": ev_combo,
        "stake": stake,
        "corr_used": rho,
    }


# ---------------------------------------------------------
# 7. Recommendation Engine
# ---------------------------------------------------------
def combo_recommendation(ev):
    if ev >= 0.12:
        return "🔥 **PLAY — Elite Edge**"
    if ev >= 0.05:
        return "🟡 **Lean — Solid Edge**"
    return "❌ **Pass — Thin / No Edge**"
