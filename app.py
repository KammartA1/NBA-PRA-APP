# =========================================================
#  KAMAL QUANT ENGINE v5.0 — MASTER FILE (Chunk 1/10)
# =========================================================

import os, time, random, difflib
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scipy.stats import norm, beta

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

st.markdown('<p class="main-header">🏀 NBA Prop Model — Kamal Quant Engine v5.0</p>', unsafe_allow_html=True)

# =========================================================
#  SIDEBAR — USER SETTINGS
# =========================================================

st.sidebar.header("User & Bankroll")

user_id = st.sidebar.text_input("Your ID (for personal history)", value="Me").strip() or "Me"

LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0)
payout_mult = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Recent Games Sample (N)", 5, 20, 10)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)

st.sidebar.caption("Model auto-pulls NBA stats. You only enter the lines.")
# =========================================================
#  PART 2 — MODEL CONSTANTS
# =========================================================

MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists"]

MARKET_METRICS = {
    "PRA": ["PTS", "REB", "AST"],
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
}

# Heavy-tail multipliers (baseline)
HEAVY_TAIL = {
    "PRA": 1.35,
    "Points": 1.25,
    "Rebounds": 1.25,
    "Assists": 1.25,
}

MAX_KELLY_PCT = 0.03  # Hard-cap at 3%

# =========================================================
#  PART 2.1 — BASIC HELPERS
# =========================================================

def current_season():
    today = datetime.now()
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"


# =========================================================
#  PART 2.2 — PLAYER LOOKUP ENGINE
# =========================================================

@st.cache_data(show_spinner=False)
def get_players_index():
    """Caches all NBA players for fuzzy lookup."""
    return nba_players.get_players()

def _norm_name(s: str) -> str:
    """Normalize player names for fuzzy match."""
    if not isinstance(s, str):
        return ""
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
    Fuzzy player resolver:
        Input:  user text
        Output: (player_id, full_name)
    """
    if not name:
        return None, None

    players = get_players_index()
    target = _norm_name(name)

    # --- Exact match first ---
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]

    # --- Fuzzy match ---
    names = [_norm_name(p["full_name"]) for p in players]
    matches = difflib.get_close_matches(target, names, n=1, cutoff=0.7)

    if matches:
        match_norm = matches[0]
        for p in players:
            if _norm_name(p["full_name"]) == match_norm:
                return p["id"], p["full_name"]

    return None, None

def get_headshot_url(name: str):
    pid, _ = resolve_player(name)
    if not pid:
        return None
    return (
        f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/"
        f"nba/latest/260x190/{pid}.png"
    )


# =========================================================
#  PART 2.3 — TEAM CONTEXT ENGINE v2
#  Pulls pace, defensive rating, assist %, rebound %
# =========================================================

@st.cache_data(show_spinner=False, ttl=3600)
def get_team_context():
    """
    Pulls advanced NBA team data:
      • Pace
      • Defensive Rating
      • Reb% / DReb%
      • Ast%

    Returns:
       TEAM_CTX: { TEAM_ABBREV : metrics }
       LEAGUE_CTX: league averages
    """

    try:
        # Base per-game team stats
        base = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        # Advanced stats
        adv = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","REB_PCT","OREB_PCT","DREB_PCT","AST_PCT","PACE"
        ]]

        # Defensive stats
        defn = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][[
            "TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"
        ]]

        df = base.merge(adv, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
        df = df.merge(defn, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")

        # Compute league averages
        league_avg = {
            col: df[col].mean()
            for col in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]
        }

        TEAM_CTX = {
            r["TEAM_ABBREVIATION"]: {
                "PACE":      r["PACE"],
                "DEF_RATING": r["DEF_RATING"],
                "REB_PCT":   r["REB_PCT"],
                "DREB_PCT":  r["DREB_PCT"],
                "AST_PCT":   r["AST_PCT"],
            }
            for _, r in df.iterrows()
        }

        return TEAM_CTX, league_avg

    except Exception:
        return {}, {}

TEAM_CTX, LEAGUE_CTX = get_team_context()


# =========================================================
#  PART 2.4 — OPPONENT MATCHUP ENGINE v2
# =========================================================

def get_context_multiplier(opp_abbrev: str | None, market: str):
    """
    Opponent-context multiplier with v2 scaling:
      - Pace factor
      - Defensive Rating factor
      - Rebound context (REB only)
      - Assist context (AST only)

    Output is clipped for stability.
    """

    if not opp_abbrev or opp_abbrev not in TEAM_CTX or not LEAGUE_CTX:
        return 1.0

    opp = TEAM_CTX[opp_abbrev]

    pace_factor = opp["PACE"] / LEAGUE_CTX["PACE"]
    def_factor  = LEAGUE_CTX["DEF_RATING"] / opp["DEF_RATING"]

    reb_factor = (
        LEAGUE_CTX["REB_PCT"] / opp["DREB_PCT"]
        if market == "Rebounds" else 1.0
    )

    ast_factor = (
        LEAGUE_CTX["AST_PCT"] / opp["AST_PCT"]
        if market == "Assists" else 1.0
    )

    # Weighted blend
    multiplier = (
        0.40 * pace_factor +
        0.30 * def_factor +
        0.30 * (reb_factor if market == "Rebounds" else ast_factor)
    )

    return float(np.clip(multiplier, 0.80, 1.20))
# =========================================================
#  PART 3 — MARKET BASELINE LIBRARY (Option A1)
# =========================================================

MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

def load_market_library():
    """Load past market lines to build baselines."""
    if not os.path.exists(MARKET_LIBRARY_FILE):
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])
    try:
        return pd.read_csv(MARKET_LIBRARY_FILE)
    except:
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])

def save_market_library(df):
    df.to_csv(MARKET_LIBRARY_FILE, index=False)

def update_market_library(player: str, market: str, line: float):
    """Append a new market entry to historical baseline storage."""
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
    """Return (mean, median) historical market line."""
    df = load_market_library()
    if df.empty:
        return None, None
    d = df[(df["Player"] == player) & (df["Market"] == market)]
    if d.empty:
        return None, None
    return d["Line"].mean(), d["Line"].median()


# =========================================================
#  PART 3.1 — PLAYER GAME LOG ENGINE v3
#  Extracts:
#    - per-minute mean
#    - per-minute sd
#    - average minutes
#    - recent usage patterns
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_rate_and_minutes(player: str, n_games: int, market: str):
    """
    Computes:
      - mu_per_min (recent)
      - sd_per_min (recent)
      - avg_min (recent)
      - team
      - usage signals (points share, assist share, rebound share)
    """

    pid, label = resolve_player(player)
    if not pid:
        return None, None, None, None, f"No match for '{player}'."

    # Pull logs
    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception as e:
        return None, None, None, None, f"Log error: {str(e)}"

    if gl.empty:
        return None, None, None, None, "No recent games found."

    # Sort newest → oldest
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False).head(n_games)

    cols = MARKET_METRICS[market]

    per_min_vals = []
    minutes_vals = []

    # Usage signals
    pts_vals = []
    reb_vals = []
    ast_vals = []

    # -----------------------------
    # Compute per-minute values
    # -----------------------------
    for _, r in gl.iterrows():

        # Minutes (handles "mm:ss" format)
        m = 0.0
        try:
            raw_m = r.get("MIN", "0")
            if isinstance(raw_m, str) and ":" in raw_m:
                mm, ss = raw_m.split(":")
                m = float(mm) + float(ss)/60
            else:
                m = float(raw_m)
        except:
            m = 0.0

        if m > 0:
            total_val = sum(float(r.get(c, 0)) for c in cols)
            per_min_vals.append(total_val / m)
            minutes_vals.append(m)

        # Usage signals always collected
        pts_vals.append(float(r.get("PTS", 0)))
        reb_vals.append(float(r.get("REB", 0)))
        ast_vals.append(float(r.get("AST", 0)))

    if not per_min_vals:
        return None, None, None, None, "Insufficient data."

    per_min_vals = np.array(per_min_vals)
    minutes_vals = np.array(minutes_vals)

    # Baseline per-min stats
    mu_per_min = float(np.mean(per_min_vals))
    sd_per_min = float(max(np.std(per_min_vals, ddof=1), 0.15 * max(mu_per_min, 0.5)))
    avg_min = float(np.mean(minutes_vals))

    # Team detection
    try:
        team = gl["TEAM_ABBREVIATION"].mode().iloc[0]
    except:
        team = None

    # -----------------------------
    # Usage & role signals (normalized)
    # -----------------------------
    pts_vals = np.array(pts_vals)
    reb_vals = np.array(reb_vals)
    ast_vals = np.array(ast_vals)

    # Normalize so they sum to 1 → usage shares
    total_usage = pts_vals + reb_vals + ast_vals
    total_usage = np.where(total_usage == 0, 1, total_usage)

    usage_points  = float(np.mean(pts_vals / total_usage))
    usage_reb     = float(np.mean(reb_vals / total_usage))
    usage_ast     = float(np.mean(ast_vals / total_usage))

    usage_profile = {
        "PTS_share": usage_points,
        "REB_share": usage_reb,
        "AST_share": usage_ast
    }

    msg = f"{label}: {len(per_min_vals)} games • {avg_min:.1f} min"

    return (mu_per_min, sd_per_min, avg_min, team, usage_profile, msg)


# =========================================================
#  PART 3.2 — INJURY IMPACT ENGINE (Upgrade G)
# =========================================================

def apply_injury_usage_boost(mu_min, usage_profile, teammate_out):
    """
    Boosts per-minute production if a key teammate is out.
    The direction of the boost depends on player role:
        - High scorer   → biggest increase to points-heavy markets
        - High passer   → assists bump
        - High rebounder→ rebounds bump
    """

    if not teammate_out:
        return mu_min  # no change

    # Weighted boost based on dominant usage type
    score_weight = usage_profile["PTS_share"]
    pass_weight  = usage_profile["AST_share"]
    reb_weight   = usage_profile["REB_share"]

    # Dynamic scaling — scorers get +8–12%, others less
    role_boost = (
        1.00 +
        0.06 * score_weight +
        0.04 * pass_weight +
        0.03 * reb_weight
    )

    # clamp boost range
    role_boost = float(np.clip(role_boost, 1.04, 1.14))

    return mu_min * role_boost
# =========================================================
#  PART 4 — ADVANCED VOLATILITY ENGINE (Upgrade C)
# =========================================================

def adaptive_volatility(sd_min, minutes, market, opp_abbrev, teammate_out, blowout):
    """
    Produces an upgraded SD value that incorporates:
      - base per-minute volatility
      - opponent defense & pace (nonlinear)
      - role inconsistency (rebounds/assists more volatile)
      - blowout probability bump
      - injury role bumps
    """

    # Start from baseline
    sd = sd_min * np.sqrt(max(minutes, 1))

    # ---------------------------------------------
    # Opponent context non-linear adjustments
    # ---------------------------------------------
    if opp_abbrev in TEAM_CTX and LEAGUE_CTX:

        opp_def = TEAM_CTX[opp_abbrev]["DEF_RATING"]
        avg_def = LEAGUE_CTX["DEF_RATING"]

        opp_pace = TEAM_CTX[opp_abbrev]["PACE"]
        avg_pace = LEAGUE_CTX["PACE"]

        # Defense → higher def rating = higher volatility
        def_factor = np.clip((opp_def / avg_def) ** 0.65, 0.85, 1.25)

        # Pace → high pace = more chaos & possessions
        pace_factor = np.clip((opp_pace / avg_pace) ** 0.55, 0.90, 1.22)

        sd *= def_factor
        sd *= pace_factor

    # ---------------------------------------------
    # Market-specific volatility weights
    # ---------------------------------------------
    if market == "Points":
        sd *= 1.08
    elif market == "Assists":
        sd *= 1.12
    elif market == "Rebounds":
        sd *= 1.14
    elif market == "PRA":
        sd *= 1.18

    # ---------------------------------------------
    # Blowout risk → fewer minutes, more variance
    # ---------------------------------------------
    if blowout:
        sd *= 1.12

    # ---------------------------------------------
    # Teammate out = more minutes + more volatility
    # ---------------------------------------------
    if teammate_out:
        sd *= 1.10

    # Clamp to safe bounds
    return float(np.clip(sd, 0.8 * sd_min, 2.0 * sd_min))


# =========================================================
#  PART 5 — HEAVY-TAIL ENGINE v2 (Upgrade B)
# =========================================================

def heavy_tail_prob(line, mu, sd, market):
    """
    Computes a right-tail probability boost using lognormal
    + exponential correction.

    Returns probability 0–1.
    """

    # Normal baseline
    p_norm = 1 - norm.cdf(line, mu, sd)

    if mu <= 0 or sd <= 0 or np.isnan(mu) or np.isnan(sd):
        return float(np.clip(p_norm, 0.01, 0.99))

    # Lognormal tail
    try:
        variance = sd**2
        phi = np.sqrt(mu**2 + variance)
        mu_log = np.log(mu**2 / phi)
        sd_log = np.sqrt(np.log(phi**2 / mu**2))
        p_ln = 1 - norm.cdf(np.log(line + 1e-9), mu_log, sd_log)
    except:
        p_ln = p_norm

    # Exponential spill-over for extreme right tail
    tail_strength = {
        "Points": 1.10,
        "Assists": 1.05,
        "Rebounds": 1.08,
        "PRA": 1.15,
    }.get(market, 1.05)

    p_exp = p_norm * np.exp((line - mu) / (3 * sd)) * 0.02
    p_exp = float(np.clip(p_exp, 0, 0.20))

    # Weighted blend (nonlinear)
    w_ln = 0.55
    w_exp = 0.10
    w_norm = 0.35

    p = w_norm * p_norm + w_ln * p_ln + w_exp * p_exp
    return float(np.clip(p, 0.01, 0.99))


# =========================================================
#  PART 6 — MULTI-DISTRIBUTION ENSEMBLE v3 (Upgrade D)
# =========================================================

def beta_estimate_prob(line, mu, sd):
    """
    Converts normal parameters to a scaled beta distribution
    approximation for bounded continuous stats.
    """

    # Negative or zero values not suitable for beta
    if mu <= 1 or sd <= 0:
        return 0.5

    # Rough beta approximation
    alpha = ((mu**2) * (1 - (mu/(mu+sd)))) / (sd**2)
    beta = alpha * ((mu/(mu+sd)) / (1 - (mu/(mu+sd))))

    alpha = max(alpha, 1.0)
    beta = max(beta, 1.0)

    # Scale line to beta domain
    scale = mu + 4 * sd
    x = np.clip(line / scale, 0.001, 0.999)

    from scipy.stats import beta as beta_dist
    p = 1 - beta_dist.cdf(x, alpha, beta)
    return float(np.clip(p, 0.01, 0.99))


def ensemble_probability(line, mu, sd, market):
    """
    Blends:
      - Normal
      - Lognormal / heavy-tail
      - Beta approximation
    """

    p_norm = 1 - norm.cdf(line, mu, sd)
    p_tail = heavy_tail_prob(line, mu, sd, market)
    p_beta = beta_estimate_prob(line, mu, sd)

    # Market-driven weights
    weights = {
        "Points":   (0.50, 0.35, 0.15),
        "Assists":  (0.45, 0.40, 0.15),
        "Rebounds": (0.40, 0.45, 0.15),
        "PRA":      (0.35, 0.55, 0.10)
    }.get(market, (0.45, 0.40, 0.15))

    wN, wT, wB = weights

    p = wN * p_norm + wT * p_tail + wB * p_beta
    return float(np.clip(p, 0.02, 0.98))


# =========================================================
#  PART 7 — MONTE-CARLO LITE ENGINE (Upgrade E)
# =========================================================

def monte_carlo_probability(n_iter, line, mu, sd):
    """
    Fast 2,000 iteration MC simulation.
    """

    # Draws from normal with extra right-tail weight
    draws = np.random.normal(mu, sd, n_iter)
    tail_extra = np.random.exponential(sd, int(n_iter*0.10))
    draws[:len(tail_extra)] += tail_extra

    p = np.mean(draws > line)
    return float(np.clip(p, 0.02, 0.98))
# =========================================================
#  PART 14 — MONTE CARLO LITE ENGINE (2,000 iterations)
# =========================================================

def mc_simulate_leg(mu: float, sd: float, market: str, line: float, 
                    n_iter: int = 2000) -> dict:
    """
    Fast Monte-Carlo simulation for a single leg.
    Uses hybrid distribution:
      - sample from Normal
      - apply right-tail correction
      - clamp negative outcomes
    Returns:
      - sim_prob_over
      - distribution (array)
      - percentiles
    """
    if sd <= 0 or np.isnan(sd):
        return {
            "sim_prob_over": 1.0 if mu > line else 0.0,
            "dist": np.array([mu]),
            "p10": mu,
            "p50": mu,
            "p90": mu,
        }

    # Base normal draws
    draws = np.random.normal(mu, sd, size=n_iter)

    # Right-tail skewing (matching hybrid engine weights)
    tail_weight = {
        "PRA": 0.18,
        "Points": 0.14,
        "Rebounds": 0.10,
        "Assists": 0.09
    }.get(market, 0.12)

    # Apply skew only to positive tail
    tail_mask = draws > mu
    draws[tail_mask] *= (1 + tail_weight)

    # Prevent negative results
    draws = np.clip(draws, 0, None)

    # Empirical probability of clearing the line
    sim_prob = float(np.mean(draws > line))

    # Percentiles (optional for graphs)
    p10 = float(np.percentile(draws, 10))
    p50 = float(np.percentile(draws, 50))
    p90 = float(np.percentile(draws, 90))

    return {
        "sim_prob_over": sim_prob,
        "dist": draws,
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }
# =========================================================
#  PART 15 — Simulation-Analytical Probability Fusion
# =========================================================

def fuse_probabilities(p_model: float, p_sim: float, sd: float) -> float:
    """
    Combines:
      - analytical hybrid probability
      - Monte Carlo simulation probability
    High volatility → simulation has more weight.
    Low volatility → analytical model has more weight.
    """

    # Convert SD into fusion weight
    sd_norm = np.clip(sd / 12, 0.1, 1.0)   # 0–12 range normalized
    w_sim = 0.25 + 0.50 * sd_norm          # 25% → 75% depending volatility
    w_model = 1.0 - w_sim

    fused = (w_model * p_model) + (w_sim * p_sim)

    return float(np.clip(fused, 0.02, 0.98))
# =========================================================
#  PART 18 — MULTI-DISTRIBUTION ENSEMBLE ENGINE
# =========================================================

def normal_prob(line, mu, sd):
    """Basic normal distribution tail probability."""
    try:
        return float(np.clip(1 - norm.cdf(line, mu, sd), 0.001, 0.999))
    except:
        return 0.5


def lognormal_prob(line, mu, sd):
    """Heavy-tail right-skew probability using lognormal transform."""
    try:
        variance = sd ** 2
        phi = np.sqrt(variance + mu ** 2)

        mu_log = np.log(mu**2 / phi)
        sd_log = np.sqrt(np.log(phi**2 / mu**2))

        if sd_log <= 0 or np.isnan(mu_log) or np.isnan(sd_log):
            return normal_prob(line, mu, sd)

        return float(
            np.clip(
                1 - norm.cdf(np.log(line + 1e-9), mu_log, sd_log),
                0.001,
                0.999
            )
        )
    except:
        return normal_prob(line, mu, sd)


def beta_prob(line, mu, sd, cap=60):
    """
    Beta distribution approximation:
    Transforms stat range into [0,1] then scores upper tail.
    """
    try:
        x = np.clip(line / cap, 0.001, 0.999)
        mean = mu / cap
        var = (sd / cap) ** 2

        # Solve for alpha & beta
        t = mean * (1 - mean) / var - 1
        if t <= 0:
            return normal_prob(line, mu, sd)

        a = mean * t
        b = (1 - mean) * t

        return float(np.clip(1 - st.beta.cdf(x, a, b), 0.001, 0.999))
    except:
        return normal_prob(line, mu, sd)


def ensemble_prob(line, mu, sd, market, volatility_score=1.0):
    """
    Master probability engine combining:
       - Normal distribution
       - Lognormal (right skew)
       - Beta distribution (rate-structured)
    
    Weights dynamically adjust by:
       - volatility (sd)
       - heavy-tail need
       - market type
       - line difficulty (distance from mu)
    """
    # Individual model probabilities
    p_norm = normal_prob(line, mu, sd)
    p_logn = lognormal_prob(line, mu, sd)
    p_beta = beta_prob(line, mu, sd)

    # Distance from mean
    diff = abs(line - mu)
    diff_norm = np.clip(diff / max(sd, 1e-6), 0, 6)

    # ---- Weight logic ----

    # Markets with heavier tail tendencies
    market_w = {
        "PRA":   (0.40, 0.45, 0.15),
        "Points": (0.50, 0.35, 0.15),
        "Rebounds": (0.45, 0.30, 0.25),
        "Assists":  (0.55, 0.25, 0.20)
    }.get(market, (0.45, 0.40, 0.15))

    w_norm_mkt, w_logn_mkt, w_beta_mkt = market_w

    # Adjust for volatility
    vol_adj = np.clip(sd / 7.0, 0.6, 1.4)

    w_norm = w_norm_mkt * (1.1 - 0.10 * vol_adj)
    w_logn = w_logn_mkt * (0.9 + 0.15 * vol_adj)
    w_beta = w_beta_mkt * (1.0 + 0.10 * (1 - diff_norm))

    # Normalize weights
    total = w_norm + w_logn + w_beta
    w_norm /= total
    w_logn /= total
    w_beta /= total

    # Final blended probability
    p_final = (
        w_norm * p_norm +
        w_logn * p_logn +
        w_beta * p_beta
    )

    return float(np.clip(p_final, 0.02, 0.98))
# =========================================================
#  PART 19 — AUTO-TUNING ENGINE + FINAL OPTIMIZATION LAYER
# =========================================================

def compute_model_drift(history_df):
    """
    Learns from past bets to detect model miscalibration.
    Adjusts:
        - variance factor
        - heavy-tail factor
        - ensemble weight dampening
    """
    try:
        comp = history_df[history_df["Result"].isin(["Hit","Miss"])].copy()
        if comp.empty or len(comp) < 25:
            return 1.0, 1.0, 1.0   # neutral (min data)

        comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
        comp = comp.dropna(subset=["EV_float"])
        if comp.empty:
            return 1.0, 1.0, 1.0

        predicted = 0.5 + comp["EV_float"].mean()
        actual = (comp["Result"] == "Hit").mean()
        diff = actual - predicted   # positive → model too conservative

        # Sensitivity knobs
        if diff < -0.04:       # too optimistic
            var_adj = 1.08
            tail_adj = 1.05
            weight_adj = 0.92

        elif diff > 0.04:      # too conservative
            var_adj = 0.92
            tail_adj = 0.95
            weight_adj = 1.06

        else:                  # well calibrated
            var_adj = 1.0
            tail_adj = 1.0
            weight_adj = 1.0

        return (
            float(np.clip(var_adj, 0.85, 1.15)),
            float(np.clip(tail_adj, 0.85, 1.15)),
            float(np.clip(weight_adj, 0.85, 1.15)),
        )

    except:
        return 1.0, 1.0, 1.0


# ---------------------------------------------------------
# Line Difficulty Modifier
# ---------------------------------------------------------
def difficulty_modifier(mu, sd, line):
    """
    Penalizes extreme outlier lines using a logistic compression.
    Helps avoid unrealistic projections.
    """
    try:
        z = abs(line - mu) / max(sd, 1e-6)
        damp = 1 / (1 + np.exp(0.6 * (z - 2.0)))  # above 2 SD → compress
        return float(np.clip(damp, 0.55, 1.0))
    except:
        return 1.0


# ---------------------------------------------------------
# Confidence Index (0–100)
# ---------------------------------------------------------
def confidence_index(sd, ctx_mult, n_games):
    """
    Measures certainty:
      - lower SD → higher confidence
      - strong context multiplier → higher confidence
      - more sample games → higher confidence
    """
    sd_score = np.clip(1 / (sd / 6), 0.3, 1.0)  # SD range ≈ 1–6
    ctx_score = np.clip(ctx_mult, 0.75, 1.25)
    sample_score = np.clip(n_games / 15, 0.4, 1.0)

    final = (0.50 * sd_score) + (0.25 * ctx_score) + (0.25 * sample_score)
    return float(np.clip(final * 100, 25, 95))


# ---------------------------------------------------------
# Final Probability Fusion Layer
# ---------------------------------------------------------
def final_probability(
    p_ensemble,
    p_sim,
    line,
    mu,
    sd,
    market,
    var_adj,
    tail_adj,
    weight_adj
):
    """
    The master fusion step:
      1) adjusts SD based on drift
      2) adjusts skew/tail need
      3) adjusts weights depending on calibration
      4) blends ensemble + simulation
      5) applies line-difficulty dampening
    """
    # 1) Adjust volatility
    sd_adj = sd * var_adj

    # 2) Re-run ensemble with adjusted tail weight
    p_a = ensemble_prob(line, mu, sd_adj, market)

    # 3) Weighting between MC sim and ensemble
    w_sim = 0.50 * weight_adj
    w_ens = 1.00 - w_sim

    p_blend = w_ens * p_a + w_sim * p_sim

    # 4) Difficulty modifier
    damp = difficulty_modifier(mu, sd_adj, line)

    p_final = p_blend * damp

    return float(np.clip(p_final, 0.02, 0.98))

# =========================================================
#  PART 8 — FULL compute_leg_projection() v5.0
# =========================================================

def compute_leg_projection(player, market, line, opp, teammate_out, blowout, n_games):
    """
    MAIN ENGINE — v5.0
    Includes:
      - usage profile (Chunk 3)
      - injury usage boost (Chunk 3)
      - adaptive volatility v2
      - heavy-tail v2
      - multi-distribution ensemble v3
      - monte-carlo stability engine
    """

    # ------------------------------
    # Load player log data
    # ------------------------------
    res = get_player_rate_and_minutes(player, n_games, market)
    if res[0] is None:
        return None, res[-1]

    mu_min, sd_min, avg_min, team, usage_profile, msg = res

    opp_abbrev = opp.strip().upper() if opp else None

    # ------------------------------
    # Apply injury usage boost (Chunk 3)
    # ------------------------------
    mu_min_adj = apply_injury_usage_boost(mu_min, usage_profile, teammate_out)

    # ------------------------------
    # Minutes projection
    # ------------------------------
    minutes = avg_min
    if teammate_out:
        minutes *= 1.06
    if blowout:
        minutes *= 0.90

    # ------------------------------
    # Raw mean outcome
    # ------------------------------
    mu = mu_min_adj * minutes

    # ------------------------------
    # Adaptive volatility
    # ------------------------------
    sd = adaptive_volatility(sd_min, minutes, market, opp_abbrev, teammate_out, blowout)

    # ------------------------------
    # Multi-distribution ensemble
    # ------------------------------
    p_model = ensemble_probability(line, mu, sd, market)

    # ------------------------------
    # Monte-Carlo stability
    # ------------------------------
    p_mc = monte_carlo_probability(2000, line, mu, sd)

    # Combined probability
    p_over = float(np.clip(0.65*p_model + 0.35*p_mc, 0.02, 0.98))

    # --------------------------------------------------------
    # 4. Hybrid analytic probability
    # --------------------------------------------------------
    p_over = ensemble_prob(line, mu, sd_final, market)
# Monte Carlo probability from Chunk 7
p_sim = mc_simulation(line, mu, sd_final, market)

# Auto-learning model drift
var_adj, tail_adj, weight_adj = compute_model_drift(load_history())

# Final fused probability
p_over_final = final_probability(
    p_ensemble = p_over,
    p_sim = p_sim,
    line = line,
    mu = mu,
    sd = sd_final,
    market = market,
    var_adj = var_adj,
    tail_adj = tail_adj,
    weight_adj = weight_adj
)

p_over = p_over_final

    # --------------------------------------------------------
    # 5. Monte-Carlo simulation probability
    # --------------------------------------------------------
    sim = mc_simulate_leg(mu, sd_final, market, line)
    p_sim = sim["sim_prob_over"]

    # --------------------------------------------------------
    # 6. Probability fusion engine
    # --------------------------------------------------------
    p_over = fuse_probabilities(p_model, p_sim, sd_final)

    # Edge vs even odds
    ev_leg_even = p_over - (1 - p_over)

    return {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": float(mu),
        "sd": float(sd),
        "prob_over": float(p_over),
        "ev_leg_even": float(ev_leg_even),
        "team": team,
        "ctx_mult": float(get_context_multiplier(opp_abbrev, market)),
        "msg": msg,
        "teammate_out": bool(teammate_out),
        "blowout": bool(blowout)
        "sim": sim,
    }, None
# =========================================================
#  PART 9 — ADVANCED CORRELATION ENGINE v2 (Upgrade F)
# =========================================================

def player_role_from_market(market):
    if market == "Points":
        return "scorer"
    if market == "Assists":
        return "playmaker"
    if market == "Rebounds":
        return "rebounder"
    if market == "PRA":
        return "hybrid"
    return "neutral"


def advanced_correlation(leg1, leg2):
    """
    Produces a robust correlation coefficient ρ
    between -0.35 and +0.55 using:

      - team & minute synergy
      - role interaction (scorer vs playmaker etc.)
      - context multiplier similarity
      - usage rate overlap
      - opponent defensive profile
      - volatility similarity
      - blowout/injury ripple interactions
    """

    ρ = 0.0

    # ---------------------------------------------------
    # 1. Team-Based Synergy
    # ---------------------------------------------------
    if leg1["team"] and leg2["team"] and leg1["team"] == leg2["team"]:
        ρ += 0.18

    # ---------------------------------------------------
    # 2. Minutes-Based Interaction
    # ---------------------------------------------------
    m1 = leg1["mu"] / max(leg1["mu"] / leg1["line"], 1e-6)
    m2 = leg2["mu"] / max(leg2["mu"] / leg2["line"], 1e-6)

    if m1 > 30 and m2 > 30:
        ρ += 0.08
    elif m1 < 22 or m2 < 22:
        ρ -= 0.05

    # ---------------------------------------------------
    # 3. Role Interaction Model
    # ---------------------------------------------------
    r1 = player_role_from_market(leg1["market"])
    r2 = player_role_from_market(leg2["market"])

    if r1 == r2:
        ρ += 0.05  # same-kind synergy
    if (r1 == "scorer" and r2 == "playmaker") or (r2 == "scorer" and r1 == "playmaker"):
        ρ -= 0.12  # scorer vs passer negative
    if (r1 == "scorer" and r2 == "rebounder") or (r2 == "scorer" and r1 == "rebounder"):
        ρ -= 0.07

    # PRA always slightly increases interaction
    if r1 == "hybrid" or r2 == "hybrid":
        ρ += 0.04

    # ---------------------------------------------------
    # 4. Context Multiplier Interaction
    # ---------------------------------------------------
    ctx1 = leg1["ctx_mult"]
    ctx2 = leg2["ctx_mult"]

    if ctx1 > 1.03 and ctx2 > 1.03:
        ρ += 0.06
    if ctx1 < 0.97 and ctx2 < 0.97:
        ρ += 0.05
    if (ctx1 > 1.05 and ctx2 < 0.95) or (ctx1 < 0.95 and ctx2 > 1.05):
        ρ -= 0.08

    # ---------------------------------------------------
    # 5. Volatility Interaction
    # ---------------------------------------------------
    sd1 = leg1["sd"]
    sd2 = leg2["sd"]

    vol_ratio = sd1 / max(sd2, 1e-6)
    if 0.85 < vol_ratio < 1.15:
        ρ += 0.05
    if vol_ratio > 1.35 or vol_ratio < 0.65:
        ρ -= 0.04

    # ---------------------------------------------------
    # 6. Injury Ripple Effects
    # ---------------------------------------------------
    if leg1["teammate_out"] and leg2["team"] == leg1["team"]:
        ρ += 0.04
    if leg2["teammate_out"] and leg1["team"] == leg2["team"]:
        ρ += 0.04

    # ---------------------------------------------------
    # Clamp the correlation
    # ---------------------------------------------------
    return float(np.clip(ρ, -0.35, 0.55))


# =========================================================
#  PART 10 — JOINT PROBABILITY ENGINE v2
# =========================================================

def joint_probability(p1, p2, rho):
    """
    Computes a mathematically correct bivariate probability:

        P(A ∩ B) = p1 p2 + ρ √(p1(1-p1)p2(1-p2))

    Clamped to [0, 1].
    """
    base = p1 * p2
    interaction = rho * np.sqrt(p1*(1-p1) * p2*(1-p2))
    return float(np.clip(base + interaction, 0.0, 1.0))


# =========================================================
#  PART 11 — COMBO EV ENGINE v2
# =========================================================

def compute_combo_ev(leg1, leg2, payout_mult):
    """
    Computes:
      - correlation
      - joint probability
      - EV
      - stake (kelly)
      - rec label
    """

    p1 = leg1["prob_over"]
    p2 = leg2["prob_over"]

    rho = advanced_correlation(leg1, leg2)
    joint = joint_probability(p1, p2, rho)

    # EV
    ev = payout_mult * joint - 1.0

    # Kelly
    b = payout_mult - 1
    q = 1 - joint
    raw_kelly = (b * joint - q) / b
    kelly = np.clip(raw_kelly * 0.25, 0, MAX_KELLY_PCT)  # 25% fractional default

    return {
        "rho": rho,
        "joint": joint,
        "ev": ev,
        "kelly_frac": float(kelly),
    }


# =========================================================
#  PART 12 — RECOMMENDATION ENGINE v2
# =========================================================

def combo_recommendation(ev, rho):
    """
    Converts EV + correlation into a human-readable recommendation.
    """

    if ev >= 0.12 and rho >= 0:
        return "🔥 **Slam — High EV + Synergistic Pairing**"

    if ev >= 0.07:
        return "💰 **Strong Play — Solid EV**"

    if ev >= 0.03:
        if rho < -0.10:
            return "🟡 **Thin Play — Negative Correlation**"
        return "🟡 **Lean — Thin Edge**"

    if ev > 0:
        return "⚪ **Marginal**"

    return "❌ **Pass — No Model Edge**"
# =========================================================
#  PART 9 — ADVANCED CORRELATION ENGINE v2 (Upgrade F)
# =========================================================

def player_role_from_market(market):
    if market == "Points":
        return "scorer"
    if market == "Assists":
        return "playmaker"
    if market == "Rebounds":
        return "rebounder"
    if market == "PRA":
        return "hybrid"
    return "neutral"


def advanced_correlation(leg1, leg2):
    """
    Produces a robust correlation coefficient ρ
    between -0.35 and +0.55 using:

      - team & minute synergy
      - role interaction (scorer vs playmaker etc.)
      - context multiplier similarity
      - usage rate overlap
      - opponent defensive profile
      - volatility similarity
      - blowout/injury ripple interactions
    """

    ρ = 0.0

    # ---------------------------------------------------
    # 1. Team-Based Synergy
    # ---------------------------------------------------
    if leg1["team"] and leg2["team"] and leg1["team"] == leg2["team"]:
        ρ += 0.18

    # ---------------------------------------------------
    # 2. Minutes-Based Interaction
    # ---------------------------------------------------
    m1 = leg1["mu"] / max(leg1["mu"] / leg1["line"], 1e-6)
    m2 = leg2["mu"] / max(leg2["mu"] / leg2["line"], 1e-6)

    if m1 > 30 and m2 > 30:
        ρ += 0.08
    elif m1 < 22 or m2 < 22:
        ρ -= 0.05

    # ---------------------------------------------------
    # 3. Role Interaction Model
    # ---------------------------------------------------
    r1 = player_role_from_market(leg1["market"])
    r2 = player_role_from_market(leg2["market"])

    if r1 == r2:
        ρ += 0.05  # same-kind synergy
    if (r1 == "scorer" and r2 == "playmaker") or (r2 == "scorer" and r1 == "playmaker"):
        ρ -= 0.12  # scorer vs passer negative
    if (r1 == "scorer" and r2 == "rebounder") or (r2 == "scorer" and r1 == "rebounder"):
        ρ -= 0.07

    # PRA always slightly increases interaction
    if r1 == "hybrid" or r2 == "hybrid":
        ρ += 0.04

    # ---------------------------------------------------
    # 4. Context Multiplier Interaction
    # ---------------------------------------------------
    ctx1 = leg1["ctx_mult"]
    ctx2 = leg2["ctx_mult"]

    if ctx1 > 1.03 and ctx2 > 1.03:
        ρ += 0.06
    if ctx1 < 0.97 and ctx2 < 0.97:
        ρ += 0.05
    if (ctx1 > 1.05 and ctx2 < 0.95) or (ctx1 < 0.95 and ctx2 > 1.05):
        ρ -= 0.08

    # ---------------------------------------------------
    # 5. Volatility Interaction
    # ---------------------------------------------------
    sd1 = leg1["sd"]
    sd2 = leg2["sd"]

    vol_ratio = sd1 / max(sd2, 1e-6)
    if 0.85 < vol_ratio < 1.15:
        ρ += 0.05
    if vol_ratio > 1.35 or vol_ratio < 0.65:
        ρ -= 0.04

    # ---------------------------------------------------
    # 6. Injury Ripple Effects
    # ---------------------------------------------------
    if leg1["teammate_out"] and leg2["team"] == leg1["team"]:
        ρ += 0.04
    if leg2["teammate_out"] and leg1["team"] == leg2["team"]:
        ρ += 0.04

    # ---------------------------------------------------
    # Clamp the correlation
    # ---------------------------------------------------
    return float(np.clip(ρ, -0.35, 0.55))


# =========================================================
#  PART 10 — JOINT PROBABILITY ENGINE v2
# =========================================================

def joint_probability(p1, p2, rho):
    """
    Computes a mathematically correct bivariate probability:

        P(A ∩ B) = p1 p2 + ρ √(p1(1-p1)p2(1-p2))

    Clamped to [0, 1].
    """
    base = p1 * p2
    interaction = rho * np.sqrt(p1*(1-p1) * p2*(1-p2))
    return float(np.clip(base + interaction, 0.0, 1.0))


# =========================================================
#  PART 11 — COMBO EV ENGINE v2
# =========================================================

def compute_combo_ev(leg1, leg2, payout_mult):
    """
    Computes:
      - correlation
      - joint probability
      - EV
      - stake (kelly)
      - rec label
    """

    p1 = leg1["prob_over"]
    p2 = leg2["prob_over"]

    rho = advanced_correlation(leg1, leg2)
    joint = joint_probability(p1, p2, rho)

    # EV
    ev = payout_mult * joint - 1.0

    # Kelly
    b = payout_mult - 1
    q = 1 - joint
    raw_kelly = (b * joint - q) / b
    kelly = np.clip(raw_kelly * 0.25, 0, MAX_KELLY_PCT)  # 25% fractional default

    return {
        "rho": rho,
        "joint": joint,
        "ev": ev,
        "kelly_frac": float(kelly),
    }


# =========================================================
#  PART 12 — RECOMMENDATION ENGINE v2
# =========================================================

def combo_recommendation(ev, rho):
    """
    Converts EV + correlation into a human-readable recommendation.
    """

    if ev >= 0.12 and rho >= 0:
        return "🔥 **Slam — High EV + Synergistic Pairing**"

    if ev >= 0.07:
        return "💰 **Strong Play — Solid EV**"

    if ev >= 0.03:
        if rho < -0.10:
            return "🟡 **Thin Play — Negative Correlation**"
        return "🟡 **Lean — Thin Edge**"

    if ev > 0:
        return "⚪ **Marginal**"

    return "❌ **Pass — No Model Edge**"
# =========================================================
#  PART 16 — Simulation Distribution Plot Engine
# =========================================================

import plotly.graph_objects as go

def plot_distribution(sim_data: dict, line: float, player: str, market: str):
    """
    Creates a density plot overlaying:
      - Monte-Carlo simulation distribution
      - Vertical line for the prop line
      - P10, P50, P90 markers
    """
    dist = sim_data["dist"]
    p10 = sim_data["p10"]
    p50 = sim_data["p50"]
    p90 = sim_data["p90"]

    fig = go.Figure()

    # Density curve
    fig.add_trace(
        go.Histogram(
            x=dist,
            histnorm="probability density",
            nbinsx=40,
            name="Sim Density",
            marker_color="#FFCC33",
            opacity=0.55,
        )
    )

    # Line marker
    fig.add_vline(
        x=line,
        line_width=3,
        line_dash="dash",
        line_color="#FF4444",
        annotation_text=f"Line ({line})",
        annotation_position="top left",
    )

    # Percentile markers
    fig.add_vline(
        x=p10,
        line_dash="dot",
        line_color="#8888FF",
        annotation_text="P10",
        annotation_position="bottom left",
    )
    fig.add_vline(
        x=p50,
        line_dash="dot",
        line_color="#33DD33",
        annotation_text="Median",
        annotation_position="bottom left",
    )
    fig.add_vline(
        x=p90,
        line_dash="dot",
        line_color="#8888FF",
        annotation_text="P90",
        annotation_position="bottom left",
    )

    fig.update_layout(
        title=f"{player} — {market} Distribution (MC Simulation)",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )

    return fig
# =========================================================
#  PART 17 — Volatility & Risk Labeling
# =========================================================

def classify_volatility(sd: float) -> str:
    if sd < 4:
        return "🟢 Low Volatility — very stable projection"
    elif sd < 7:
        return "🟡 Medium Volatility — normal prop risk"
    elif sd < 10:
        return "🟠 High Volatility — wide performance range"
    else:
        return "🔴 Extreme Volatility — boom/bust profile"
        # --------------------------------------------------
        #  Simulation Stats (only if sim data exists)
        # --------------------------------------------------
        if "sim" in leg:
            sim = leg["sim"]
            st.markdown("### 📊 Simulation Insights")
            st.write(f"**Simulated Probability Over:** {sim['sim_prob_over']*100:.1f}%")
            st.write(f"**P10 / P50 / P90:** {sim['p10']:.1f} / {sim['p50']:.1f} / {sim['p90']:.1f}")
            
            # Volatility classification
            vol_text = classify_volatility(sd)
            st.info(vol_text)

            # Distribution plot
            fig = plot_distribution(sim, line, player, market)
            st.plotly_chart(fig, use_container_width=True)
