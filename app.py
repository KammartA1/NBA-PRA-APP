# =========================================================
#  NBA PROP BETTING QUANT ENGINE — SINGLE FILE APP
#  Upgraded with:
#   - Empirical bootstrap Monte Carlo (10,000 sims)
#   - Defensive matchup engine (team + context aware)
#   - Pace–adjusted minute expectation model
#   - Usage + on/off boost system
#   - Game script simulation (pace, blowout, fouls, OT, usage variance)
#   - Ensemble projections (bootstrap mean, historical mean, market implied, usage mean, script mean)
#   - Covariance-based joint Monte Carlo via shared script factors
#   - Underdog live line integration + Edge Scanner tab
#   - Calibration & bankroll discipline
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
#  SIDEBAR — USER SETTINGS & BANKROLL
# =========================================================

st.sidebar.header("User & Bankroll")

user_id = st.sidebar.text_input("Your ID (for personal history)", value="Me").strip() or "Me"
LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0)
payout_mult = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Recent Games Sample (N)", 5, 20, 10)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)
max_daily_loss_pct = st.sidebar.slider("Max Daily Loss (% of bankroll)", 1, 50, 15)

st.sidebar.caption("Model auto-pulls NBA stats & Underdog lines. You only enter lines on the Model tab if you want manual mode.")

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

MAX_KELLY_PCT = 0.03  # 3% hard cap

MONTE_CARLO_SIMS_MODEL = 10_000
MONTE_CARLO_SIMS_SCANNER = 3_000  # lighter for scanner

UNDERDOG_NBA_URL = "https://api.underdogfantasy.com/beta/v3/over_under_lines"  # may change; configurable

# =========================================================
#  HELPERS — SEASON, CACHING, PLAYER INDEX
# =========================================================

def current_season() -> str:
    """
    Auto-detect current NBA season based on today's date.
    If month >= 10 → season begins this year (e.g. 2025-26)
    Else → season started previous year (e.g. 2024-25)
    """
    today = datetime.now()
    if today.month >= 10:
        start_year = today.year
    else:
        start_year = today.year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"

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
    import difflib
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
#  MARKET BASELINE LIBRARY
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
#  PLAYER GAME LOG ENGINE & BOOTSTRAP SAMPLES
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_gamelog(player_name: str):
    pid, label = resolve_player(player_name)
    if not pid:
        return None, None, f"No match for '{player_name}'."
    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception as e:
        return None, None, f"Game log error: {e}"
    if gl.empty:
        return None, None, "No recent game logs found."
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False)
    return gl, label, ""

def extract_market_series(gl: pd.DataFrame, market: str, n_games: int):
    cols = MARKET_METRICS[market]
    rows = gl.head(n_games).copy()
    values = []
    minutes = []
    opps = []
    for _, r in rows.iterrows():
        m_val = 0.0
        try:
            m_str = r.get("MIN", "0")
            if isinstance(m_str, str) and ":" in m_str:
                mm, ss = m_str.split(":")
                m_val = float(mm) + float(ss)/60.0
            else:
                m_val = float(m_str)
        except Exception:
            m_val = 0.0
        if m_val <= 0:
            continue
        total_val = float(sum(float(r.get(c, 0)) for c in cols))
        values.append(total_val)
        minutes.append(m_val)
        opps.append(r.get("MATCHUP","").split()[-1] if isinstance(r.get("MATCHUP",""), str) else None)
    return np.array(values, dtype=float), np.array(minutes, dtype=float), opps

def compute_usage_boost(teammate_out: bool) -> float:
    base = 1.0
    if teammate_out:
        base *= 1.08
    return base

def compute_blowout_factor(blowout_flag: bool) -> float:
    if blowout_flag:
        return 0.93
    return 1.0

def compute_game_script_factors(n_sims: int, base_minutes: float, blowout_flag: bool):
    """
    Simulate pace, foul risk, OT chance, and usage volatility.
    Returns dict of np.arrays: pace_f, foul_f, ot_f, usage_f, minutes_f.
    """
    # Pace ~ N(1, 0.06)
    pace_f = np.random.normal(1.0, 0.06, size=n_sims)
    pace_f = np.clip(pace_f, 0.88, 1.18)

    # Foul trouble → some sims with reduced minutes
    foul_events = np.random.binomial(1, 0.12, size=n_sims)
    foul_f = np.where(foul_events == 1, np.random.uniform(0.80, 0.95, size=n_sims), 1.0)

    # OT chance → rare sims with extra minutes
    ot_events = np.random.binomial(1, 0.07, size=n_sims)
    ot_f = np.where(ot_events == 1, np.random.uniform(1.03, 1.12, size=n_sims), 1.0)

    # Usage variance
    usage_f = np.random.normal(1.0, 0.07, size=n_sims)
    usage_f = np.clip(usage_f, 0.82, 1.20)

    # Minutes distribution with all factors
    blow_f = compute_blowout_factor(blowout_flag)
    minutes_f = base_minutes * pace_f * foul_f * ot_f * blow_f
    minutes_f = np.clip(minutes_f, base_minutes * 0.70, base_minutes * 1.30)

    return {
        "pace_f": pace_f,
        "foul_f": foul_f,
        "ot_f": ot_f,
        "usage_f": usage_f,
        "minutes_f": minutes_f,
    }

def simulate_leg_bootstrap(values: np.ndarray,
                           minutes: np.ndarray,
                           line: float,
                           market: str,
                           ctx_mult: float,
                           teammate_out: bool,
                           blowout_flag: bool,
                           n_sims: int):
    """
    Empirical bootstrap Monte Carlo:
    - Sample per-game outcomes with replacement
    - Apply game script factors, usage, defense adjustment
    """
    if len(values) == 0:
        return 0.5, float(line), 1.0, np.array([line])

    # Base per-minute rate from historical
    per_min = values / np.maximum(minutes, 1.0)
    base_mu_per_min = float(np.mean(per_min))
    base_minutes = float(np.mean(minutes))

    usage_boost = compute_usage_boost(teammate_out)

    # Game script factors (shared across sims for this leg)
    gs = compute_game_script_factors(n_sims, base_minutes, blowout_flag)

    # Draw game samples
    idx = np.random.randint(0, len(values), size=n_sims)
    sampled_per_min = per_min[idx]

    # Apply all factors
    out = sampled_per_min * gs["minutes_f"] * ctx_mult * gs["usage_f"] * usage_boost

    # Market-specific volatility scaling
    if market == "Rebounds":
        out *= np.random.normal(1.0, 0.06, size=n_sims)
    elif market == "Assists":
        out *= np.random.normal(1.0, 0.05, size=n_sims)
    elif market == "Points":
        out *= np.random.normal(1.0, 0.07, size=n_sims)
    elif market == "PRA":
        out *= np.random.normal(1.0, 0.08, size=n_sims)

    out = np.clip(out, 0, None)

    p_over = float(np.mean(out > line))
    proj_mean = float(np.mean(out))
    proj_sd = float(np.std(out, ddof=1))

    return p_over, proj_mean, proj_sd, out

# =========================================================
#  ENSEMBLE PROJECTIONS
# =========================================================

def compute_ensemble_projection(values: np.ndarray,
                                minutes: np.ndarray,
                                line: float,
                                market: str,
                                ctx_mult: float,
                                teammate_out: bool,
                                blowout_flag: bool,
                                market_implied_mean: float | None,
                                n_sims: int = MONTE_CARLO_SIMS_MODEL):
    """
    Ensemble of:
    - Bootstrap mean
    - Historical mean
    - Market implied mean (from Underdog if available)
    - Usage-predicted mean
    - Game-script mean
    """
    if len(values) == 0:
        return 0.5, float(line), 1.0, np.array([line])

    # Historical mean
    hist_mean = float(np.mean(values)) * ctx_mult

    # Usage-predicted mean (simple scaling of historical with usage boost)
    usage_boost = compute_usage_boost(teammate_out)
    usage_mean = hist_mean * usage_boost

    # Game script mean via a small MC (for speed)
    p_over_gs, gs_mean, _, _ = simulate_leg_bootstrap(
        values, minutes, line, market, ctx_mult, teammate_out, blowout_flag, n_sims=min(3000, n_sims)
    )

    # Bootstrap MC main distribution
    p_over_boot, boot_mean, boot_sd, samples = simulate_leg_bootstrap(
        values, minutes, line, market, ctx_mult, teammate_out, blowout_flag, n_sims=n_sims
    )

    # Market implied mean (optional)
    means = [boot_mean, hist_mean, usage_mean, gs_mean]
    weights = [0.40, 0.20, 0.20, 0.20]

    if market_implied_mean is not None and market_implied_mean > 0:
        means.append(market_implied_mean)
        weights.append(0.20)
        # Re-normalize weights
        s = sum(weights)
        weights = [w/s for w in weights]

    ensemble_mean = float(sum(m * w for m, w in zip(means, weights)))

    # Recompute probability over line based on samples but centered around ensemble_mean
    # Shift samples to match ensemble_mean while preserving shape
    shift = ensemble_mean - boot_mean
    adj_samples = samples + shift
    p_over_ensemble = float(np.mean(adj_samples > line))
    proj_sd = float(np.std(adj_samples, ddof=1))

    return p_over_ensemble, ensemble_mean, proj_sd, adj_samples

# =========================================================
#  KELLY FOR 2-PICK
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float):
    b = payout_mult - 1
    q = 1 - p_joint
    raw = (b * p_joint - q) / b
    k = raw * frac
    return float(np.clip(k, 0, MAX_KELLY_PCT))

# =========================================================
#  HISTORY HELPERS
# =========================================================

def ensure_history():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac","Joint","ModelProb"
        ])
        df.to_csv(LOG_FILE, index=False)

def load_history():
    ensure_history()
    try:
        return pd.read_csv(LOG_FILE)
    except Exception:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac","Joint","ModelProb"
        ])

def save_history(df):
    df.to_csv(LOG_FILE, index=False)

# =========================================================
#  UNDERDOG SCRAPER (BASIC JSON CLIENT)
# =========================================================

@st.cache_data(show_spinner=True, ttl=60)
def fetch_underdog_nba_props():
    """
    Fetch Underdog NBA props from an unofficial endpoint.
    NOTE: This endpoint may change. If it does, update UNDERDOG_NBA_URL.

    Returns DataFrame with:
    - player
    - market (Points/Rebounds/Assists/PRA)
    - line
    - team
    - opponent
    - implied_mean (approximated)
    """
    try:
        resp = requests.get(UNDERDOG_NBA_URL, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    # The exact JSON structure may differ. We try to be defensive.
    # Expecting list of over_under_lines with linked players & stats.
    props = []
    try:
        items = data.get("over_under_lines") or data.get("data") or []
    except AttributeError:
        items = []

    for item in items:
        try:
            ou = item.get("over_under", {}) if isinstance(item, dict) else {}
            stat_type = ou.get("over_under_type") or ou.get("stat_type","")
            player_name = ou.get("player_name") or ou.get("name") or ""
            team = ou.get("team_abbr") or ""
            opponent = ou.get("opponent_abbr") or ""

            # Map stat_type to market
            stat_type_lower = str(stat_type).lower()
            if "points" in stat_type_lower:
                market = "Points"
            elif "rebounds" in stat_type_lower:
                market = "Rebounds"
            elif "assists" in stat_type_lower:
                market = "Assists"
            elif "pra" in stat_type_lower or "points_rebounds_assists" in stat_type_lower:
                market = "PRA"
            else:
                continue

            line_val = float(ou.get("higher_or_lower", item.get("line", 0)) or 0)
            if line_val <= 0:
                continue

            # Approximate implied mean = line (neutral)
            implied_mean = line_val

            props.append({
                "player": player_name,
                "market": market,
                "line": line_val,
                "team": team,
                "opponent": opponent,
                "market_implied_mean": implied_mean,
            })
        except Exception:
            continue

    if not props:
        return pd.DataFrame()

    df = pd.DataFrame(props)
    return df

# =========================================================
#  CORE PROJECTION ENGINE (USES BOOTSTRAP + ENSEMBLE)
# =========================================================

def compute_leg_projection(player: str,
                           market: str,
                           line: float,
                           opp: str | None,
                           teammate_out: bool,
                           blowout_flag: bool,
                           n_games: int,
                           market_implied_mean: float | None = None,
                           n_sims: int = MONTE_CARLO_SIMS_MODEL):
    """
    Main engine used by both Model tab and Edge Scanner.
    Returns leg dict and error message.
    """
    gl, label, err = get_player_gamelog(player)
    if gl is None:
        return None, err

    values, minutes, opps = extract_market_series(gl, market, n_games)
    if len(values) == 0:
        return None, "Insufficient data from game logs."

    # If opponent not supplied, infer from most common opponent in recent games (rarely needed)
    opp_abbrev = (opp or "").strip().upper()
    if not opp_abbrev and len([o for o in opps if o]) > 0:
        try:
            opp_abbrev = pd.Series([o for o in opps if o]).mode().iloc[0]
        except Exception:
            opp_abbrev = None

    ctx_mult = get_context_multiplier(opp_abbrev, market)

    p_over, ensemble_mean, proj_sd, samples = compute_ensemble_projection(
        values, minutes, line, market, ctx_mult, teammate_out, blowout_flag,
        market_implied_mean=market_implied_mean,
        n_sims=n_sims
    )

    ev_leg_even = p_over - (1 - p_over)

    # Quick defensive stats for card
    def_ctx = TEAM_CTX.get(opp_abbrev, {}) if opp_abbrev else {}
    opp_def_rating = def_ctx.get("DEF_RATING")
    opp_pace = def_ctx.get("PACE")
    opp_reb_pct = def_ctx.get("REB_PCT")
    opp_ast_pct = def_ctx.get("AST_PCT")

    return {
        "player": label or player,
        "market": market,
        "line": float(line),
        "mu": float(ensemble_mean),
        "sd": float(proj_sd),
        "prob_over": float(p_over),
        "ev_leg_even": float(ev_leg_even),
        "team": str(gl["TEAM_ABBREVIATION"].mode().iloc[0]) if "TEAM_ABBREVIATION" in gl.columns else None,
        "ctx_mult": float(ctx_mult),
        "msg": f"{label}: {len(values)} games • {np.mean(minutes):.1f} min (bootstrap/ensemble)",
        "teammate_out": bool(teammate_out),
        "blowout": bool(blowout_flag),
        "opp": opp_abbrev,
        "opp_def_rating": opp_def_rating,
        "opp_pace": opp_pace,
        "opp_reb_pct": opp_reb_pct,
        "opp_ast_pct": opp_ast_pct,
        "samples": samples,
    }, None

# =========================================================
#  JOINT MONTE CARLO — COVARIANCE VIA SHARED GAME SCRIPT
# =========================================================

def joint_monte_carlo(leg1: dict,
                      leg2: dict,
                      payout_mult: float,
                      n_sims: int = MONTE_CARLO_SIMS_MODEL):
    """
    Approximate covariance-based joint MC by:
    - Resampling from each leg's empirical distribution
    - Sharing game script factors (pace, blowout) between legs
    - Adding market-type covariance tweaks (usage→assists, etc.)
    """
    s1 = leg1.get("samples")
    s2 = leg2.get("samples")
    if s1 is None or s2 is None:
        # Fallback independent estimate
        p1 = leg1["prob_over"]
        p2 = leg2["prob_over"]
        p_joint_ind = p1 * p2
        return float(p_joint_ind), 0.0

    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    if len(s1) < 100 or len(s2) < 100:
        p1 = leg1["prob_over"]
        p2 = leg2["prob_over"]
        p_joint_ind = p1 * p2
        return float(p_joint_ind), 0.0

    # Resample indices for both legs
    idx1 = np.random.randint(0, len(s1), size=n_sims)
    idx2 = np.random.randint(0, len(s2), size=n_sims)

    base1 = s1[idx1].copy()
    base2 = s2[idx2].copy()

    # Shared game script factors: pace & blowout
    pace_f = np.random.normal(1.0, 0.05, size=n_sims)
    pace_f = np.clip(pace_f, 0.90, 1.12)

    blow_events = np.random.binomial(1, 0.10, size=n_sims)
    blow_f = np.where(blow_events == 1, 0.92, 1.0)

    # Market-type covariance adjustments
    m1, m2 = leg1["market"], leg2["market"]

    # usage→assists covariance: if leg2 is assists and leg1 points
    usage_assist_cov = np.ones(n_sims)
    if (m1 == "Points" and m2 == "Assists") or (m2 == "Points" and m1 == "Assists"):
        # negative covariance: high scoring game reduces pure assist spike probability slightly
        usage_assist_cov = np.random.normal(0.97, 0.03, size=n_sims)

    # apply shared factors
    out1 = base1 * pace_f * blow_f
    out2 = base2 * pace_f * blow_f * usage_assist_cov

    p1_over = float(np.mean(out1 > leg1["line"]))
    p2_over = float(np.mean(out2 > leg2["line"]))
    p_joint = float(np.mean((out1 > leg1["line"]) & (out2 > leg2["line"])))

    # Empirical covariance
    cov_emp = float(np.cov((out1 > leg1["line"]).astype(float),
                           (out2 > leg2["line"]).astype(float))[0,1])
    # Return joint and an approximate "correlation-like" summary
    try:
        var1 = p1_over * (1 - p1_over)
        var2 = p2_over * (1 - p2_over)
        denom = (var1 * var2) ** 0.5 if var1 > 0 and var2 > 0 else 1.0
        corr_like = cov_emp / denom
    except Exception:
        corr_like = 0.0

    return p_joint, float(np.clip(corr_like, -0.4, 0.6))

# =========================================================
#  UI RENDER — LEG CARD
# =========================================================

def render_leg_card(leg: dict, container, compact=False):
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
    opp = leg.get("opp")
    opp_def = leg.get("opp_def_rating")
    opp_pace = leg.get("opp_pace")
    opp_reb_pct = leg.get("opp_reb_pct")
    opp_ast_pct = leg.get("opp_ast_pct")

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

        cols = st.columns([1,2]) if headshot and not compact else [st]

        if headshot and not compact:
            with cols[0]:
                st.image(headshot, width=110)
            main_col = cols[1]
        else:
            main_col = cols[0] if isinstance(cols, list) else st

        with main_col:
            st.write(f"📌 **Line:** {line}")
            st.write(f"📊 **Ensemble Mean (Model Line):** {mu:.2f}")
            st.write(f"📉 **Model SD:** {sd:.2f}")
            st.write(f"🎯 **Model Probability Over:** {p*100:.1f}%")
            st.write(f"💵 **Even-Money EV:** {even_ev*100:+.1f}%")
            st.write(f"⏱️ **Context Multiplier (Def+Pace):** {ctx:.3f}")
            st.caption(f"📝 {msg}")

            # Opponent quick defensive stats
            if opp:
                st.markdown("**Opponent Context:**")
                pieces = []
                if opp_def is not None:
                    pieces.append(f"DEF RTG: {opp_def:.1f}")
                if opp_pace is not None:
                    pieces.append(f"PACE: {opp_pace:.1f}")
                if opp_reb_pct is not None:
                    pieces.append(f"REB%: {opp_reb_pct:.3f}")
                if opp_ast_pct is not None:
                    pieces.append(f"AST%: {opp_ast_pct:.3f}")
                if pieces:
                    st.write(f"🛡️ {opp} → " + " | ".join(pieces))

            # Risk flags
            if teammate_out:
                st.info("⚠️ Teammate out → usage boost & volatility applied.")
            if blowout:
                st.warning("⚠️ Blowout risk → minutes reduced in script sims.")

# =========================================================
#  LOADER & DECISION LOGIC
# =========================================================

def run_loader():
    load_ph = st.empty()
    msgs = [
        "Pulling player logs…",
        "Analyzing matchup context…",
        "Simulating game scripts…",
        "Running Monte Carlo…",
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
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration", "🧭 Edge Scanner"]
)

# =========================================================
#  MODEL TAB
# =========================================================

with tab_model:

    st.subheader("2-Pick Projection & Edge (Auto stats + manual/Underdog lines)")

    c1, c2 = st.columns(2)

    with c1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="m1")
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5)
        o1 = st.text_input("P1 Opponent (abbr)", help="Example: BOS, DEN", key="o1")
        p1_teammate_out = st.checkbox("P1: Key teammate out? (override)", key="p1_to")
        p1_blowout = st.checkbox("P1: Blowout risk high? (override)", key="p1_bo")

    with c2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="m2")
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5)
        o2 = st.text_input("P2 Opponent (abbr)", help="Example: BOS, DEN", key="o2")
        p2_teammate_out = st.checkbox("P2: Key teammate out? (override)", key="p2_to")
        p2_blowout = st.checkbox("P2: Blowout risk high? (override)", key="p2_bo")

    leg1 = None
    leg2 = None

    run = st.button("Run Model ⚡")

    if run:

        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

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
        st.markdown(f"**Market Implied Probability (2-pick equal legs):** {imp_prob*100:.1f}%")

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

        if leg1 and leg2:
            p_joint, corr_like = joint_monte_carlo(leg1, leg2, payout_mult)
            ev_combo = payout_mult * p_joint - 1.0
            k_frac = kelly_for_combo(p_joint, payout_mult, fractional_kelly)
            stake = round(bankroll * k_frac, 2)
            decision = combo_decision(ev_combo)

            st.markdown("### 🎯 **2-Pick Combo Result (Covariance MC)**")
            st.markdown(f"- Joint Probability: **{p_joint*100:.1f}%**")
            st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Suggested Stake (Kelly-capped): **${stake:.2f}**")
            st.markdown(f"- Covariance signal: **{corr_like:+.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")

        for leg in [leg1, leg2]:
            if leg:
                update_market_library(leg["player"], leg["market"], leg["line"])
                mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
                if mean_b:
                    st.caption(
                        f"📊 Market Baseline for {leg['player']} {leg['market']}: "
                        f"mean={mean_b:.1f}, median={med_b:.1f}"
                    )

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
                "CLV (Closing - Entry)",
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
                "KellyFrac": fractional_kelly,
                "Joint": None,
                "ModelProb": None,
            }

            df = pd.concat(
                [df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            save_history(df)

            st.success("Result logged ✅")

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

        hit_rate = (hits / total * 100) if total > 0 else 0.0
        roi = pnl.sum() / max(bankroll, 1.0) * 100

        st.markdown(
            f"**Completed Bets:** {total}  |  "
            f"**Hit Rate:** {hit_rate:.1f}%  |  "
            f"**ROI:** {roi:+.1f}%"
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

        filt = df[df["EV"] >= min_ev]

        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        st.markdown(f"**Filtered Bets:** {len(filt)}")
        st.dataframe(filt, use_container_width=True)

        if not filt.empty:

            filt = filt.copy()

            filt["Net"] = filt.apply(
                lambda r:
                    r["Stake"] * (payout_mult - 1.0)
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
#  CALIBRATION TAB — SELF-LEARNING HOOK
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
                    r["Stake"] * (payout_mult - 1.0)
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
#  EDGE SCANNER TAB — UNDERDOG LIVE SCAN
# =========================================================

def classify_edge(ev_diff: float) -> str:
    if ev_diff >= 0.12:
        return "Elite"
    elif ev_diff >= 0.07:
        return "Medium"
    elif ev_diff >= 0.03:
        return "Thin"
    else:
        return "No Edge"

def annotate_correlation(row, df):
    """Simple auto-annotation of correlated edges: same player/team & related markets."""
    same_player = df[(df["player"] == row["player"]) & (df["market"] != row["market"])]
    same_team = df[(df["team"] == row["team"]) & (df["market"] == row["market"])]
    tags = []
    if not same_player.empty:
        tags.append("player-link")
    if not same_team.empty:
        tags.append("team-link")
    return ", ".join(tags) if tags else ""

with tab_scanner:

    st.subheader("🧭 Live Underdog Edge Scanner (NBA PTS/REB/AST/PRA)")
    st.caption("Pulls Underdog lines (where available) and runs the full ensemble + bootstrap engine to surface edges.")

    scan_button = st.button("Scan Underdog NBA Board 🔍")

    if scan_button:
        with st.spinner("Fetching Underdog board & running simulations…"):
            ud_df = fetch_underdog_nba_props()

        if ud_df.empty:
            st.error("Could not fetch Underdog NBA props or no supported props found. You may need to update UNDERDOG_NBA_URL or run later.")
        else:
            # Run model for each prop
            records = []
            for _, r in ud_df.iterrows():
                if r["market"] not in MARKET_OPTIONS:
                    continue
                leg, err = compute_leg_projection(
                    r["player"],
                    r["market"],
                    r["line"],
                    r.get("opponent"),
                    teammate_out=False,
                    blowout_flag=False,
                    n_games=games_lookback,
                    market_implied_mean=r.get("market_implied_mean"),
                    n_sims=MONTE_CARLO_SIMS_SCANNER,
                )
                if leg is None or err:
                    continue
                model_prob = leg["prob_over"]
                model_mean = leg["mu"]
                # Treat Underdog line as "fair" mean; EV difference vs 50/50 baseline
                ev_diff = model_prob - 0.5
                tier = classify_edge(ev_diff)

                records.append({
                    "player": leg["player"],
                    "team": leg["team"],
                    "market": leg["market"],
                    "line": leg["line"],
                    "model_line": model_mean,
                    "model_prob": model_prob,
                    "ev_diff": ev_diff,
                    "tier": tier,
                    "opp": leg.get("opp"),
                    "opp_def": leg.get("opp_def_rating"),
                    "opp_pace": leg.get("opp_pace"),
                })

            if not records:
                st.warning("No modelable edges found from Underdog props.")
            else:
                edge_df = pd.DataFrame(records)
                edge_df["correlation_tag"] = edge_df.apply(lambda row: annotate_correlation(row, edge_df), axis=1)
                edge_df["EV_%"] = edge_df["ev_diff"] * 100
                edge_df["ModelProb_%"] = edge_df["model_prob"] * 100

                # Sort by EV descending
                edge_df = edge_df.sort_values("ev_diff", ascending=False)

                st.markdown("### Top Edges")
                st.dataframe(
                    edge_df[[
                        "player","team","market","line","model_line","ModelProb_%","EV_%","tier","opp","opp_def","opp_pace","correlation_tag"
                    ]].reset_index(drop=True),
                    use_container_width=True,
                )

                # Highlight elite & medium edges
                elite = edge_df[edge_df["tier"] == "Elite"]
                medium = edge_df[edge_df["tier"] == "Medium"]

                if not elite.empty:
                    st.markdown("#### 🔥 Elite Edges")
                    st.dataframe(elite[["player","market","line","model_line","EV_%","ModelProb_%","opp","opp_def","correlation_tag"]])

                if not medium.empty:
                    st.markdown("#### 🟡 Medium Edges")
                    st.dataframe(medium[["player","market","line","model_line","EV_%","ModelProb_%","opp","opp_def","correlation_tag"]])

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
