# =========================================================
#  NBA Prop Model — Streamlit Single-File App
#  Upgraded with The Odds API Edge Scanner + Defense Context
# =========================================================

import os, time, random, difflib, math, json
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
    page_title="NBA Prop Model",
    page_icon="🏀",
    layout="wide"
)

# =========================================================
#  GLOBAL CONSTANTS
# =========================================================

TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
CARD_BG = "#17131C"
BG = "#0D0A12"

# The Odds API key (user confirmed; free tier)
ODDS_API_KEY = "621ec92ab709da9f9ce59cf2e513af55"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

NBA_SPORT_KEY = "basketball_nba"

# Maximum MC simulations
MC_SIMS = 10_000

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
#  PART 1 — SIDEBAR (USER SETTINGS)
# =========================================================

st.sidebar.header("User & Bankroll")

user_id = st.sidebar.text_input("Your ID (for personal history)", value="Me").strip() or "Me"
LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=100.0)
payout_mult = st.sidebar.number_input("2-Pick Payout (e.g. 3.0x)", min_value=1.5, value=3.0)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Recent Games Sample (N)", 5, 20, 10)
compact_mode = st.sidebar.checkbox("Compact Mode (mobile)", value=False)

st.sidebar.caption("Model auto-pulls NBA stats. You only enter the lines — or use live lines via Edge Scanner.")

# =========================================================
#  PART 1.1 — MODEL CONSTANTS
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

# Map The Odds API player prop market keys → internal markets
ODDS_MARKET_MAP = {
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_points_rebounds_assists": "PRA",
}

# =========================================================
#  PART 1.2 — SEASON LOGIC
# =========================================================

def current_season() -> str:
    """
    Returns NBA season string like '2025-26' based on today's date.
    Auto-rolls each October to new season.
    """
    today = datetime.now()
    # NBA regular season generally starts in October
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"

# =========================================================
#  PART 2 — PLAYER LOOKUP HELPERS
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
    """Resolves fuzzy player input → NBA API player ID & canonical full_name."""
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
#  PART 2.1 — TEAM CONTEXT (PACE, DEF, REB%, AST%)
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
            for col in ["PACE","DEF_RATING","REB_PCT","AST_PCT","DREB_PCT"]
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
#  PART 2.2 — MARKET BASELINE LIBRARY
# =========================================================

MARKET_LIBRARY_FILE = os.path.join(TEMP_DIR, "market_baselines.csv")

def load_market_library():
    """Loads market baselines; safe fallback on first run."""
    if not os.path.exists(MARKET_LIBRARY_FILE):
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])
    try:
        return pd.read_csv(MARKET_LIBRARY_FILE)
    except:
        return pd.DataFrame(columns=["Player","Market","Line","Timestamp"])

def save_market_library(df):
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
#  PART 2.3 — ODDS API CLIENT (The Odds API)
# =========================================================

@st.cache_data(show_spinner=False, ttl=300)
def fetch_nba_player_props():
    """
    Uses The Odds API to pull NBA player props for:
      - points, rebounds, assists, PRA
    Returns: DataFrame with columns:
      [player, market, line, price, implied_prob, event, bookmaker]
    """
    if not ODDS_API_KEY:
        return pd.DataFrame(columns=["player","market","line","price","implied_prob","event","bookmaker"])

    try:
        # Step 1: list in-season NBA events
        events_url = f"{ODDS_API_BASE}/sports/{NBA_SPORT_KEY}/events"
        params_events = {
            "apiKey": ODDS_API_KEY,
        }
        events_resp = requests.get(events_url, params=params_events, timeout=8)
        events_resp.raise_for_status()
        events = events_resp.json()
    except Exception as e:
        st.warning(f"Odds API events error: {e}")
        return pd.DataFrame(columns=["player","market","line","price","implied_prob","event","bookmaker"])

    all_rows = []
    markets_param = ",".join([
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_points_rebounds_assists"
    ])

    # Limit events for free tier sanity: max 15 events
    for ev in events[:15]:
        event_id = ev.get("id")
        if not event_id:
            continue
        try:
            odds_url = f"{ODDS_API_BASE}/sports/{NBA_SPORT_KEY}/events/{event_id}/odds"
            params_odds = {
                "apiKey": ODDS_API_KEY,
                "markets": markets_param,
                "regions": "us",
                "oddsFormat": "decimal"
            }
            odds_resp = requests.get(odds_url, params=params_odds, timeout=8)
            odds_resp.raise_for_status()
            data = odds_resp.json()
        except Exception:
            continue

        event_name = f"{ev.get('home_team','')} vs {ev.get('away_team','')}".strip()
        bookmakers = data.get("bookmakers", [])

        for bk in bookmakers:
            book_key = bk.get("key","")
            book_title = bk.get("title","")
            markets = bk.get("markets", [])
            for mkt in markets:
                mkey = mkt.get("key")
                if mkey not in ODDS_MARKET_MAP:
                    continue
                internal_market = ODDS_MARKET_MAP[mkey]
                outcomes = mkt.get("outcomes", [])
                for oc in outcomes:
                    player_name = oc.get("description") or oc.get("name")
                    line = oc.get("point")
                    price = oc.get("price")  # decimal odds
                    if not player_name or line is None or price is None:
                        continue
                    try:
                        dec = float(price)
                        if dec <= 1:
                            continue
                        implied = 1.0 / dec
                    except:
                        continue

                    all_rows.append({
                        "player": player_name,
                        "market": internal_market,
                        "line": float(line),
                        "price": float(dec),
                        "implied_prob": float(implied),
                        "event": event_name,
                        "bookmaker": book_title or book_key
                    })

    if not all_rows:
        return pd.DataFrame(columns=["player","market","line","price","implied_prob","event","bookmaker"])

    df = pd.DataFrame(all_rows)
    return df

# =========================================================
#  PART 3 — PLAYER GAME LOG ENGINE & PROJECTION MODEL
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    """
    Pulls recent player logs, computes:
      - per-minute production (mu_per_min)
      - per-minute std dev (sd_per_min)
      - average minutes
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
        except:
            m = 0

        if m <= 0:
            continue

        total_val = sum(float(r.get(c, 0)) for c in cols)
        per_min_vals.append(total_val / m)
        minutes_vals.append(m)

    if not per_min_vals:
        return None, None, None, None, "Insufficient data."

    per_min_vals = np.array(per_min_vals)
    minutes_vals = np.array(minutes_vals)

    mu_per_min = float(np.mean(per_min_vals))
    avg_min = float(np.mean(minutes_vals))
    sd_per_min = max(
        np.std(per_min_vals, ddof=1),
        0.15 * max(mu_per_min, 0.5)
    )

    team = None
    try:
        team = gl["TEAM_ABBREVIATION"].mode().iloc[0]
    except:
        team = None

    return mu_per_min, sd_per_min, avg_min, team, f"{label}: {len(per_min_vals)} games • {avg_min:.1f} min"

@st.cache_data(show_spinner=False, ttl=900)
def get_player_game_samples(name: str, n_games: int, market: str):
    """
    Returns per-game totals for last N games (for bootstrap MC).
    """
    pid, label = resolve_player(name)
    if not pid:
        return np.array([]), label

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season"
        ).get_data_frames()[0]
    except Exception:
        return np.array([]), label

    if gl.empty:
        return np.array([]), label

    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False).head(n_games)

    cols = MARKET_METRICS[market]
    totals = []

    for _, r in gl.iterrows():
        total_val = sum(float(r.get(c, 0)) for c in cols)
        totals.append(total_val)

    return np.array(totals, dtype=float), label

def hybrid_prob_over(line, mu, sd, market):
    """
    Stable hybrid distribution:
    - Normal core
    - Log-normal-ish adjustment guarded
    - Market weighting
    """
    normal_p = 1 - norm.cdf(line, mu, sd)

    if mu <= 0 or sd <= 0 or np.isnan(mu) or np.isnan(sd):
        return float(np.clip(normal_p, 0.01, 0.99))

    try:
        variance = sd ** 2
        phi = math.sqrt(variance + mu ** 2)
        mu_log = math.log(mu ** 2 / phi)
        sd_log = math.sqrt(math.log(phi ** 2 / mu ** 2))

        if np.isnan(mu_log) or np.isnan(sd_log) or sd_log <= 0:
            lognorm_p = normal_p
        else:
            lognorm_p = 1 - norm.cdf(math.log(line + 1e-9), mu_log, sd_log)
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

# =========================================================
#  PART 3.1 — MONTE CARLO (BOOTSTRAP) + ENSEMBLE
# =========================================================

def run_bootstrap_mc(line: float, samples: np.ndarray, ctx_mult: float) -> dict:
    """
    Bootstrap MC for a single leg using historical game totals.
    Returns:
      mean_mc, sd_mc, prob_over_mc
    """
    if samples is None or len(samples) == 0:
        return {"mean_mc": None, "sd_mc": None, "prob_over_mc": None}

    # Apply context multiplier to each sample (defense, pace, usage, etc.)
    adj_samples = samples.astype(float) * float(ctx_mult)

    draws = np.random.choice(adj_samples, size=min(MC_SIMS, len(adj_samples) * 200), replace=True)
    mean_mc = float(np.mean(draws))
    sd_mc = float(np.std(draws, ddof=1))
    prob_over = float(np.mean(draws > line))

    return {
        "mean_mc": mean_mc,
        "sd_mc": sd_mc,
        "prob_over_mc": prob_over,
    }

def ensemble_projection(mu_model: float,
                        mc_mean: float | None,
                        hist_mean: float | None,
                        market_mean: float | None,
                        game_script_mean: float | None) -> float:
    """
    Ensemble-projected mean:
      - model mean
      - MC bootstrap mean (last N games)
      - season/historical mean (if available)
      - market implied mean (The Odds API)
      - game script mean (placeholder currently ~model)
    """
    comps = []
    weights = []

    if mu_model is not None:
        comps.append(mu_model)
        weights.append(0.40)

    if mc_mean is not None:
        comps.append(mc_mean)
        weights.append(0.25)

    if hist_mean is not None:
        comps.append(hist_mean)
        weights.append(0.15)

    if market_mean is not None:
        comps.append(market_mean)
        weights.append(0.15)

    if game_script_mean is not None:
        comps.append(game_script_mean)
        weights.append(0.05)

    if not comps:
        return mu_model

    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    comps = np.array(comps, dtype=float)
    return float(np.dot(comps, weights))

# ======================================================
#  CORRELATION / COVARIANCE ENGINE (HIGH-LEVEL)
# ======================================================

def estimate_player_correlation(leg1, leg2):
    """
    Context-aware correlation estimate between two legs.
    Uses:
      - same team
      - minutes profile
      - market types
      - opponent context
    """
    corr = 0.0

    if leg1["team"] == leg2["team"] and leg1["team"] is not None:
        corr += 0.18

    # minutes proxy
    try:
        avg_min1 = leg1.get("minutes", 30.0)
        avg_min2 = leg2.get("minutes", 30.0)
    except:
        avg_min1 = avg_min2 = 30.0

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

def run_joint_covariance_mc(leg1: dict, leg2: dict) -> float:
    """
    Approximate covariance-based joint probability using:
      - leg1/leg2 modeled probabilities
      - correlation estimate
    Returns joint probability P(both hit).
    """
    p1 = leg1["prob_over"]
    p2 = leg2["prob_over"]
    corr = estimate_player_correlation(leg1, leg2)

    # Approximate bivariate Bernoulli joint probability from correlation
    # Using covariance approx: Cov = corr * sqrt(p1(1-p1)p2(1-p2))
    cov = corr * math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    joint = p1 * p2 + cov
    return float(np.clip(joint, 0.0, 1.0))

# ======================================================
#  CORE PROJECTION ENGINE (DEFENSE + BOOTSTRAP)
# ======================================================

def compute_leg_projection(player, market, line, opp, teammate_out, blowout, n_games):
    """
    Core projection engine for a single leg.
    Uses:
      - recent per-minute production (NBA API)
      - opponent pace/defense context
      - bootstrap MC over last N games
      - ensemble-projected mean
    Returns (leg_dict, error_message).
    """
    mu_min, sd_min, avg_min, team, msg = get_player_rate_and_minutes(player, n_games, market)
    if mu_min is None:
        return None, msg

    opp_abbrev = opp.strip().upper() if opp else None
    ctx_mult = get_context_multiplier(opp_abbrev, market)

    # Defensive matchup raw context for display
    opp_def_rating = None
    opp_pace = None
    opp_reb_pct = None
    opp_ast_pct = None
    if opp_abbrev and opp_abbrev in TEAM_CTX:
        ctx = TEAM_CTX[opp_abbrev]
        opp_def_rating = float(ctx["DEF_RATING"])
        opp_pace = float(ctx["PACE"])
        opp_reb_pct = float(ctx["REB_PCT"])
        opp_ast_pct = float(ctx["AST_PCT"])

    heavy = HEAVY_TAIL[market]

    minutes = avg_min
    if teammate_out:
        minutes *= 1.05
        mu_min *= 1.06
    if blowout:
        minutes *= 0.90

    # model base mean
    mu_raw = mu_min * minutes
    mu_model = mu_raw * ctx_mult

    # base std dev
    sd_base = max(1.0, sd_min * math.sqrt(max(minutes, 1.0)) * heavy)

    # volatility tweaks from defense & pace
    sd_final = sd_base
    if opp_abbrev and opp_abbrev in TEAM_CTX:
        opp = TEAM_CTX[opp_abbrev]
        def_vol = np.clip(LEAGUE_CTX["DEF_RATING"] / opp["DEF_RATING"], 0.85, 1.20)
        pace_vol = np.clip(opp["PACE"] / LEAGUE_CTX["PACE"], 0.90, 1.18)
        sd_final *= def_vol * pace_vol

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

    # bootstrap samples
    samples, _ = get_player_game_samples(player, n_games, market)
    mc_res = run_bootstrap_mc(line, samples, ctx_mult)
    mc_mean = mc_res["mean_mc"]
    mc_prob = mc_res["prob_over_mc"]

    # simple "historical mean" from season games (unweighted)
    hist_mean = float(samples.mean()) if samples is not None and samples.size > 0 else None

    # market mean placeholder; if The Odds API used externally we can pass it in
    market_mean = line  # props are set near 50/50

    # placeholder game-script mean ~ model mean
    game_script_mean = mu_model

    # ensemble mean
    mu_ens = ensemble_projection(mu_model, mc_mean, hist_mean, market_mean, game_script_mean)

    # hybrid distribution probability using ensemble mean
    p_over_model = hybrid_prob_over(line, mu_ens, sd_final, market)

    # if MC prob available, blend with model prob
    if mc_prob is not None:
        p_over = float(0.6 * p_over_model + 0.4 * mc_prob)
    else:
        p_over = p_over_model

    if (
        p_over is None
        or not isinstance(p_over, (int, float))
        or np.isnan(p_over)
        or p_over <= 0.0
        or p_over >= 1.0
    ):
        p_over = float(np.clip(1.0 - norm.cdf(line, mu_ens, sd_final), 0.02, 0.98))

    ev_leg_even = p_over - (1 - p_over)

    return {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": float(mu_ens),
        "mu_model": float(mu_model),
        "sd": float(sd_final),
        "prob_over": float(p_over),
        "prob_mc": mc_prob,
        "ev_leg_even": float(ev_leg_even),
        "team": team,
        "ctx_mult": float(ctx_mult),
        "msg": msg,
        "teammate_out": bool(teammate_out),
        "blowout": bool(blowout),
        "minutes": float(minutes),
        "opp_abbrev": opp_abbrev,
        "opp_def_rating": opp_def_rating,
        "opp_pace": opp_pace,
        "opp_reb_pct": opp_reb_pct,
        "opp_ast_pct": opp_ast_pct,
    }, None

# =========================================================
#  PART 3.2 — KELLY FORMULA FOR 2-PICK
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float):
    """
    Kelly criterion for 2-pick entries.
    """
    b = payout_mult - 1
    q = 1 - p_joint
    raw = (b * p_joint - q) / b
    k = raw * frac
    return float(np.clip(k, 0, MAX_KELLY_PCT))

# =====================================================
# SELF-LEARNING CALIBRATION HOOK (placeholder)
# =====================================================

def compute_model_drift(history_df):
    """
    Placeholder self-learning calibration hook.
    Currently returns neutral adjustments (1.0, 1.0).
    """
    return 1.0, 1.0

# =========================================================
#  PART 4 — UI RENDER ENGINE + LOADERS + DECISION LOGIC
# =========================================================

def render_leg_card(leg: dict, container, compact=False):
    """
    Displays a stylized card showing:
      - headshot
      - player + market info
      - mean, sd, ctx multiplier
      - model + MC probability
      - EV at even money
      - defensive matchup quick stats
    """
    player = leg["player"]
    market = leg["market"]
    msg = leg["msg"]
    line = leg["line"]
    mu = leg["mu"]
    sd = leg["sd"]
    p = leg["prob_over"]
    p_mc = leg["prob_mc"]
    ctx = leg["ctx_mult"]
    even_ev = leg["ev_leg_even"]
    teammate_out = leg["teammate_out"]
    blowout = leg["blowout"]

    opp_abbrev = leg.get("opp_abbrev")
    opp_def = leg.get("opp_def_rating")
    opp_pace = leg.get("opp_pace")
    opp_reb = leg.get("opp_reb_pct")
    opp_ast = leg.get("opp_ast_pct")

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

        cols = st.columns([1, 2])
        with cols[0]:
            if headshot:
                st.image(headshot, width=120)
        with cols[1]:
            st.write(f"📌 **Line:** {line}")
            st.write(f"📊 **Ensemble Mean:** {mu:.2f}")
            st.write(f"📉 **Model SD:** {sd:.2f}")
            st.write(f"⏱️ **Context Multiplier:** {ctx:.3f}")
            st.write(f"🎯 **Model Probability Over:** {p*100:.1f}%")
            if p_mc is not None:
                st.write(f"🎲 **Bootstrap MC Prob Over:** {p_mc*100:.1f}%")
            st.write(f"💵 **Even-Money EV:** {even_ev*100:+.1f}%")
            st.caption(f"📝 {msg}")

        if teammate_out:
            st.info("⚠️ Teammate out → usage + minutes boost applied.")
        if blowout:
            st.warning("⚠️ Blowout risk → minutes reduced.")

        if opp_abbrev and opp_def is not None and opp_pace is not None:
            st.markdown("#### 🛡️ Defensive Matchup Snapshot")
            st.write(f"- Opponent: **{opp_abbrev}**")
            st.write(f"- DEF Rating: **{opp_def:.1f}** (league avg {LEAGUE_CTX.get('DEF_RATING',0):.1f})")
            st.write(f"- Pace: **{opp_pace:.1f}** (league avg {LEAGUE_CTX.get('PACE',0):.1f})")
            if market == "Rebounds" and opp_reb is not None:
                st.write(f"- Opp DREB%: **{opp_reb*100:.1f}%**")
            if market == "Assists" and opp_ast is not None:
                st.write(f"- Opp AST%: **{opp_ast*100:.1f}%**")

def run_loader():
    """Friendly loading animation for model runs."""
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
    """Converts EV into a recommendation."""
    if ev_combo >= 0.10:
        return "🔥 **PLAY — Strong Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **Lean — Thin Edge**"
    else:
        return "❌ **Pass — No Edge**"

# =====================================================
#  PART 5 — TABS
# =====================================================

tab_model, tab_scanner, tab_results, tab_history, tab_calib = st.tabs(
    ["📊 Model", "🧭 Live Edge Scanner", "📓 Results", "📜 History", "🧠 Calibration"]
)

# =====================================================
#  MODEL TAB
# =====================================================

with tab_model:
    st.subheader("2-Pick Projection & Edge (Auto stats + manual or live lines)")

    c1, c2 = st.columns(2)

    with c1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS, key="m1")
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5, key="l1")
        o1 = st.text_input("P1 Opponent (abbr)", help="Example: BOS, DEN")
        p1_teammate_out = st.checkbox("P1: Key teammate out?", key="p1_to")
        p1_blowout = st.checkbox("P1: Blowout risk high?", key="p1_bo")

    with c2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS, key="m2")
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5, key="l2")
        o2 = st.text_input("P2 Opponent (abbr)", help="Example: BOS, DEN")
        p2_teammate_out = st.checkbox("P2: Key teammate out?", key="p2_to")
        p2_blowout = st.checkbox("P2: Blowout risk high?", key="p2_bo")

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
        st.markdown(f"**Market Implied Probability (2-pick):** {imp_prob*100:.1f}%")

        if leg1:
            st.markdown(
                f"**{leg1['player']} Model Prob:** {leg1['prob_over']*100:.1f}% "
                f"→ Edge vs 2-pick: {(leg1['prob_over'] - imp_prob)*100:+.1f}%"
            )
        if leg2:
            st.markdown(
                f"**{leg2['player']} Model Prob:** {leg2['prob_over']*100:.1f}% "
                f"→ Edge vs 2-pick: {(leg2['prob_over'] - imp_prob)*100:+.1f}%"
            )

        if leg1 and leg2:
            joint = run_joint_covariance_mc(leg1, leg2)
            ev_combo = payout_mult * joint - 1.0
            k_frac = kelly_for_combo(joint, payout_mult, fractional_kelly)
            stake = round(bankroll * k_frac, 2)
            decision = combo_decision(ev_combo)

            st.markdown("### 🎯 **2-Pick Combo Result**")
            st.markdown(f"- Joint Probability (covariance-based): **{joint*100:.1f}%**")
            st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Suggested Stake (Kelly-capped): **${stake:.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")

        for leg in [leg1, leg2]:
            if leg:
                mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
                if mean_b:
                    st.caption(
                        f"📊 Market Baseline for {leg['player']} {leg['market']}: "
                        f"mean={mean_b:.1f}, median={med_b:.1f}"
                    )

# =====================================================
#  EDGE SCANNER TAB — THE ODDS API
# =====================================================

with tab_scanner:
    st.subheader("🧭 Live Edge Scanner (The Odds API — NBA Player Props)")

    st.caption(
        "Scans sportsbook player props via The Odds API and runs your model to detect edges "
        "for Points, Rebounds, Assists, and PRA."
    )

    if not ODDS_API_KEY:
        st.warning("Set a valid The Odds API key in the code to use the Edge Scanner.")
    else:
        if st.button("Fetch & Scan NBA Props 🔍"):
            with st.spinner("Fetching live props and running model…"):
                props_df = fetch_nba_player_props()

                if props_df.empty:
                    st.warning("No props returned from The Odds API right now. Try again later.")
                else:
                    # Run model projections for each prop
                    rows = []
                    for _, r in props_df.iterrows():
                        player = r["player"]
                        market = r["market"]
                        line = r["line"]
                        event = r["event"]
                        book = r["bookmaker"]
                        imp_prob = r["implied_prob"]

                        # Try to detect team abbrev from player logs; if not, leave opp blank.
                        opp = ""  # For edge scanner we don't know opp abbrev trivially here.

                        leg, err = compute_leg_projection(
                            player=player,
                            market=market,
                            line=line,
                            opp=opp,
                            teammate_out=False,
                            blowout=False,
                            n_games=games_lookback
                        )

                        if err or not leg:
                            continue

                        p_model = leg["prob_over"]
                        model_mean = leg["mu"]

                        # Approx EV vs single-market implied prob
                        ev_diff = p_model - imp_prob

                        # Tier classification
                        if ev_diff >= 0.12:
                            tier = "Elite"
                        elif ev_diff >= 0.07:
                            tier = "Medium"
                        elif ev_diff >= 0.03:
                            tier = "Thin"
                        else:
                            tier = "No Edge"

                        rows.append({
                            "Player": player,
                            "Market": market,
                            "Line": line,
                            "Model Mean (Ensemble)": round(model_mean, 2),
                            "Model P(Over)": round(p_model * 100, 1),
                            "Book Implied P(Over)": round(imp_prob * 100, 1),
                            "EV Diff (Model - Market) %": round(ev_diff * 100, 1),
                            "Tier": tier,
                            "Event": event,
                            "Book": book
                        })

                    if not rows:
                        st.info("No edges could be evaluated from the current board.")
                    else:
                        edges = pd.DataFrame(rows)

                        # Highlight best edges
                        edges_sorted = edges.sort_values(
                            "EV Diff (Model - Market) %",
                            ascending=False
                        )

                        st.markdown("### 🔝 Top Edges")
                        st.dataframe(edges_sorted.head(30), use_container_width=True)

                        # Quick tier breakdown
                        tier_counts = edges["Tier"].value_counts().reset_index()
                        tier_counts.columns = ["Tier", "Count"]
                        st.markdown("### 📊 Edge Tier Breakdown")
                        st.dataframe(tier_counts, use_container_width=True)

# =====================================================
#  HISTORY HELPERS
# =====================================================

def ensure_history():
    if not os.path.exists(LOG_FILE):
        df = pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac"
        ])
        df.to_csv(LOG_FILE, index=False)

def load_history():
    ensure_history()
    try:
        return pd.read_csv(LOG_FILE)
    except:
        return pd.DataFrame(columns=[
            "Date","Player","Market","Line","EV",
            "Stake","Result","CLV","KellyFrac"
        ])

def save_history(df):
    df.to_csv(LOG_FILE, index=False)

# =====================================================
#  RESULTS TAB
# =====================================================

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
                "KellyFrac": fractional_kelly
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

# =====================================================
#  HISTORY TAB
# =====================================================

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

# =====================================================
#  CALIBRATION TAB
# =====================================================

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

# =====================================================
#  FOOTER
# =====================================================

st.markdown(
    """
    <footer style='text-align:center; margin-top:30px; color:#FFCC33; font-size:11px;'>
        © 2025 NBA Prop Model • Powered by Kamal
    </footer>
    """,
    unsafe_allow_html=True,
)
