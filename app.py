
###############################
# SEGMENT 1 — CHUNK 1
# Imports + basic config
###############################

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import re
from datetime import datetime
from math import erf, sqrt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="UltraMAX NBA Quant Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

###############################
# SEGMENT 1 — CHUNK 2A
# Global Dark Theme CSS (Part 1)
###############################

dark_css = """
<style>
/* ===== APP BACKGROUND ===== */
body, .stApp {
    background-color: #0a0a0a !important;
    color: #e0e0e0 !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #111111 !important;
    border-right: 1px solid #222 !important;
}
"""

st.markdown(dark_css, unsafe_allow_html=True)


###############################
# SEGMENT 1 — CHUNK 2B
# Global Dark Theme CSS (Part 2)
###############################

dark_css_part2 = """
/* ===== METRIC CARDS ===== */
.stMetric {
    background-color: #1b1b1b !important;
    border: 1px solid #333 !important;
    padding: 12px !important;
    border-radius: 10px !important;
}
div[data-testid="stMetricValue"] {
    color: #00d4ff !important;
}

/* ===== HEADERS ===== */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}
"""

st.markdown(dark_css_part2, unsafe_allow_html=True)


###############################
# SEGMENT 1 — CHUNK 2C
# Global Dark Theme CSS (Part 3 FINAL)
###############################

dark_css_part3 = """
/* ===== DATAFRAME STYLING ===== */
.dataframe, .stDataFrame {
    background-color: #000 !important;
    color: #ffffff !important;
}

/* ===== BUTTONS ===== */
.stButton>button {
    background-color: #006eff !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
}
.stButton>button:hover {
    background-color: #0088ff !important;
}

/* End of Global Dark Theme */
</style>
"""

st.markdown(dark_css_part3, unsafe_allow_html=True)


###############################
# SEGMENT 1 — CHUNK 3A
# Global Helpers (Part 1)
###############################

# --- Current NBA Season Helper ---
def get_current_season():
    now = datetime.now()
    y = now.year
    m = now.month
    # NBA season runs Oct–June
    if m < 10:
        return f"{y-1}-{str(y)[2:]}"
    return f"{y}-{str(y+1)[2:]}"

# --- Gaussian CDF (SciPy-free) ---
def normal_cdf(x, mu, sd):
    if sd <= 0:
        return 0.5
    z = (x - mu) / sd
    return 0.5 * (1 + erf(z / sqrt(2)))

# --- Safe Execution Wrapper ---
def safe_run(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)

###############################
# SEGMENT 1 — CHUNK 3B
# Disk Cache Layer (JSON-based)
###############################

CACHE_DIR = ".cache_ultramax"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def _cache_key(name, *args, **kwargs):
    raw = name + str(args) + str(kwargs)
    return str(abs(hash(raw)))

def disk_cache(func):
    def wrapper(*args, **kwargs):
        key = _cache_key(func.__name__, *args, **kwargs)
        path = os.path.join(CACHE_DIR, key + ".json")

        # If cached file exists, return it
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                pass

        # Compute fresh output
        out = func(*args, **kwargs)

        # Save to cache
        try:
            with open(path, "w") as f:
                json.dump(out, f)
        except:
            pass

        return out
    return wrapper

###############################
# SEGMENT 1 — CHUNK 3C
# HTML Fetcher & Utility Functions
###############################

# --- Fallback-safe HTML fetcher ---
def fetch_html(url):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.text
        return ""
    except:
        return ""

# --- Clean player name for matching ---
def clean_name(name):
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^a-zA-Z ]", "", name).strip().lower()

# --- Parse "MM:SS" or "MM" into float minutes ---
def parse_minutes(val):
    if isinstance(val, (float, int)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    if ":" in val:
        try:
            m, s = val.split(":")
            return float(m) + float(s)/60
        except:
            return 0.0
    try:
        return float(val)
    except:
        return 0.0

# --- Safe float converter (handles '—' or blanks) ---
def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

###############################
# SEGMENT 1 — CHUNK 4
# Basketball-Reference Resolver Helpers
###############################

# Convert player name to Basketball-Reference style ID guess
def clean_player_id(name):
    if not isinstance(name, str) or len(name.strip()) == 0:
        return ""
    parts = name.lower().split()
    if len(parts) == 1:
        base = parts[0][:5]
        suffix = "01"
    else:
        last = parts[-1][:5]
        first = parts[0][:2]
        base = last + first
        suffix = "01"
    return base

# Build BR URL safely
def generate_bref_url(player_id, season):
    try:
        yr = season.split("-")[0]
        letter = player_id[0]
        url = f"https://www.basketball-reference.com/players/{letter}/{player_id}/gamelog/{yr}"
        return url
    except:
        return ""

###############################
# SEGMENT 2 — CHUNK 1
# PrizePicks Live Line Engine (Primary Source)
###############################

@disk_cache
def fetch_prizepicks_lines(player_name):
    """Fetch projected stat lines from PrizePicks public API.
    Returns dict: { 'PTS': xx, 'REB': xx, 'AST': xx, 'PRA': xx }
    """
    try:
        url = "https://api.prizepicks.com/projections"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        included = {i["id"]: i for i in data.get("included", [])}
        results = {}

        target = clean_name(player_name)

        for proj in data.get("data", []):
            pid = proj.get("relationships", {}).get("new_player", {}).get("data", {}).get("id")
            stat_id = proj.get("relationships", {}).get("stat_type", {}).get("data", {}).get("id")

            if pid not in included or stat_id not in included:
                continue

            player_raw = included[pid]["attributes"].get("name", "")
            player_norm = clean_name(player_raw)
            if target not in player_norm:
                continue

            stat_name = included[stat_id]["attributes"].get("name", "").upper()
            line_val = proj.get("attributes", {}).get("line_score", None)

            if line_val is not None:
                # Map PrizePicks stat names to model stat names
                if stat_name in ["PTS", "POINTS"]:
                    results["PTS"] = float(line_val)
                elif stat_name in ["REB", "REBOUNDS"]:
                    results["REB"] = float(line_val)
                elif stat_name in ["AST", "ASSISTS"]:
                    results["AST"] = float(line_val)
                elif stat_name in ["PRA"]:
                    results["PRA"] = float(line_val)

        return results
    except Exception:
        return {}

###############################
# SEGMENT 2 — CHUNK 2
# OddsAPI Fallback (Secondary Source)
###############################

@disk_cache
def fetch_oddsapi_lines(player_name):
    """Fallback live lines using The Odds API.
    Requires ODDS_API_KEY in environment variables.
    Returns dict {PTS, REB, AST, PRA}
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {}  # No key available

    try:
        url = (
            "https://api.the-odds-api.com/v4/sports/basketball_nba/"
            f"odds?apiKey={api_key}&regions=us&markets=player_props"
        )
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        target = clean_name(player_name)
        results = {}

        # Loop games
        for game in data:
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    key = market.get("key","")
                    # expected keys like player_points, player_rebounds, etc.
                    for outcome in market.get("outcomes", []):
                        name_norm = clean_name(outcome.get("name",""))
                        if target not in name_norm:
                            continue

                        val = outcome.get("line")
                        if val is None:
                            continue

                        if "points" in key:
                            results["PTS"] = float(val)
                        elif "rebounds" in key:
                            results["REB"] = float(val)
                        elif "assists" in key:
                            results["AST"] = float(val)
                        elif "points_rebounds_assists" in key:
                            results["PRA"] = float(val)

        return results
    except Exception:
        return {}

###############################
# SEGMENT 2 — CHUNK 3
# Unified PrizePicks + OddsAPI normalization
###############################

def normalize_lines(pp_dict, oa_dict):
    """Merge PrizePicks + OddsAPI lines into a single clean dict.
    Precedence:
        1. PrizePicks
        2. OddsAPI
        3. Missing → omitted (manual fallback later)
    """
    out = {}

    # PP always wins if available
    for k in ["PTS", "REB", "AST", "PRA"]:
        if k in pp_dict:
            out[k] = float(pp_dict[k])
        elif k in oa_dict:
            out[k] = float(oa_dict[k])

    return out


def get_live_lines(player_name):
    """Main live line aggregator — PP → OddsAPI → empty dict.
    LIVE1 system uses these to auto-fill sidebar.
    """
    pp = fetch_prizepicks_lines(player_name)
    oa = fetch_oddsapi_lines(player_name)

    merged = normalize_lines(pp, oa)
    return merged  # may be partial dict (handled by sidebar)

###############################
# SEGMENT 2 — CHUNK 4
# LIVE1 Sidebar Auto-Fill + Player Resolver
###############################

def resolve_player_for_lines(name):
    """Improve player matching for PrizePicks & OddsAPI.
    Normalizes the input name into best-match form.
    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def apply_live_lines_to_sidebar(lines_dict, sidebar_state):
    """LIVE1 auto-fill logic.
    - Only auto-fills fields that are currently empty or zero.
    - User overrides remain untouched.
    """
    if not isinstance(lines_dict, dict):
        return sidebar_state

    # Mapping sidebar element names
    field_map = {
        "line_pts": "PTS",
        "line_reb": "REB",
        "line_ast": "AST",
        "line_pra": "PRA"
    }

    for side_field, live_key in field_map.items():
        # If stat available and sidebar field is empty or 0
        if live_key in lines_dict:
            if side_field not in sidebar_state or sidebar_state[side_field] in [0, None, ""]:
                sidebar_state[side_field] = float(lines_dict[live_key])

    return sidebar_state


def safe_lines_output(lines_dict):
    """Guarantee clean float lines for downstream engines.
    Missing values will remain absent and handled via manual fallback.
    """
    out = {}
    for k, v in lines_dict.items():
        try:
            out[k] = float(v)
        except:
            continue
    return out

###############################
# SEGMENT 3 — CHUNK 1
# Basketball-Reference Gamelog Fetcher
###############################

@disk_cache
def fetch_gamelog(player_name, season):
    """Fetch gamelog from Basketball Reference with safe parsing.
    Returns list of dict entries for each game.
    """
    player_id = clean_player_id(player_name)
    url = generate_bref_url(player_id, season)
    html = fetch_html(url)
    if not html:
        return []

    try:
        tables = pd.read_html(html)
    except Exception:
        return []

    # Typically, gamelog is table index 7, but fallback to first >100-row table.
    df = None
    if len(tables) > 7:
        df = tables[7]
    else:
        for t in tables:
            if isinstance(t, pd.DataFrame) and len(t) > 20:
                df = t
                break

    if df is None:
        return []

    # Remove header rows inside the table
    df = df[df[df.columns[0]] != df.columns[0]]

    # Clean and parse stats
    out = []
    for idx, row in df.iterrows():
        entry = {}
        entry["DATE"] = row.get("Date", None)
        entry["PTS"] = safe_float(row.get("PTS", 0))
        entry["REB"] = safe_float(row.get("TRB", 0))
        entry["AST"] = safe_float(row.get("AST", 0))
        entry["MP"]  = parse_minutes(row.get("MP", 0))

        # PRA derived
        entry["PRA"] = entry["PTS"] + entry["REB"] + entry["AST"]

        out.append(entry)

    return out

###############################
# SEGMENT 3 — CHUNK 2
# Player Resolver (BR-ID + Name Normalization)
###############################

def normalize_player_name(name):
    """Standardize player name to consistent lowercase format."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def resolve_player_br_id(player_name):
    """Enhanced resolver:
    - Normalizes name
    - Attempts multiple BR ID patterns
    - Returns best-guess BR ID candidate
    """
    name = normalize_player_name(player_name)
    if not name:
        return ""

    parts = name.split()
    if len(parts) == 1:
        base = parts[0][:5]
        return base + "01"

    # Standard BR pattern: lastname(5) + firstname(2) + 01
    last = parts[-1][:5]
    first = parts[0][:2]
    base = last + first
    return base + "01"


def resolve_player_for_all_sources(name):
    """Master resolver used for ALL systems:
    - PrizePicks matching
    - OddsAPI matching
    - Basketball Reference
    """
    if not isinstance(name, str):
        return {"raw": "", "clean": "", "br_id": ""}

    raw = name.strip()
    clean = normalize_player_name(name)
    br_id = resolve_player_br_id(name)

    return {
        "raw": raw,
        "clean": clean,
        "br_id": br_id
    }

###############################
# SEGMENT 3 — CHUNK 3
# Team Pace & Defensive Rating Loader
###############################

# Static fallback data (can be expanded or replaced with API in future)
TEAM_FALLBACK_DATA = {
    "pace": {
        "default_team": 100.0
    },
    "def_rating": {
        "default_team": 113.0
    }
}

@disk_cache
def load_team_pace(team_name):
    """Load team pace from cached source or fallback."""
    try:
        # Placeholder for real API / database
        t = team_name.lower().strip()
        return TEAM_FALLBACK_DATA["pace"].get(t, 100.0)
    except:
        return 100.0


@disk_cache
def load_team_def_rating(team_name):
    """Load opponent defensive rating (lower = better defense)."""
    try:
        t = team_name.lower().strip()
        return TEAM_FALLBACK_DATA["def_rating"].get(t, 113.0)
    except:
        return 113.0


def load_team_context(team, opp):
    """Unified loader returning all contextual pace + defense stats."""
    team_pace = load_team_pace(team)
    opp_pace  = load_team_pace(opp)
    opp_def   = load_team_def_rating(opp)

    return {
        "team_pace": float(team_pace),
        "opp_pace": float(opp_pace),
        "opp_def_rating": float(opp_def)
    }

###############################
# SEGMENT 3 — CHUNK 4
# Opponent Defensive Profile Loader
###############################

# Placeholder defensive suppression values.
# Expandable to a full per-team, per-position defensive DB.
DEF_PROFILE_FALLBACK = {
    "default": {
        "PTS_SUPPRESSION": 1.00,   # 1.00 = neutral, <1 = tougher defense
        "REB_SUPPRESSION": 1.00,
        "AST_SUPPRESSION": 1.00
    }
}

@disk_cache
def load_defensive_profile(team_name):
    """Return opponent defensive suppression multipliers.
    Values closer to 0.85 indicate strong defense.
    Values closer to 1.15 indicate weak defense.
    """
    try:
        t = team_name.lower().strip()
        # Future expansion: real team map
        profile = DEF_PROFILE_FALLBACK.get(t, DEF_PROFILE_FALLBACK["default"])
        return {
            "pts_mult": float(profile.get("PTS_SUPPRESSION", 1.0)),
            "reb_mult": float(profile.get("REB_SUPPRESSION", 1.0)),
            "ast_mult": float(profile.get("AST_SUPPRESSION", 1.0)),
        }
    except:
        return {
            "pts_mult": 1.0,
            "reb_mult": 1.0,
            "ast_mult": 1.0
        }

###############################
# SEGMENT 3 — CHUNK 5
# Final Data Normalization Utilities
###############################

def extract_series_from_gamelog(gamelog, stat_key):
    """Extract a clean numeric series for a given stat from a gamelog list.
    Returns a list of floats with invalid entries removed.
    """
    if not isinstance(gamelog, list):
        return []

    out = []
    for g in gamelog:
        if not isinstance(g, dict):
            continue
        val = g.get(stat_key, None)
        if isinstance(val, (int, float)):
            out.append(float(val))
        else:
            try:
                out.append(float(val))
            except:
                continue

    return out


def extract_minutes_series(gamelog):
    return extract_series_from_gamelog(gamelog, "MP")


def extract_pts_series(gamelog):
    return extract_series_from_gamelog(gamelog, "PTS")


def extract_reb_series(gamelog):
    return extract_series_from_gamelog(gamelog, "REB")


def extract_ast_series(gamelog):
    return extract_series_from_gamelog(gamelog, "AST")


def extract_pra_series(gamelog):
    return extract_series_from_gamelog(gamelog, "PRA")

###############################
# SEGMENT 4 — CHUNK 1
# Baseline Stats Engine (MU/SD)
###############################

def compute_baseline_stats(gamelog):
    """Compute baseline MU and SD for PTS, REB, AST, PRA from gamelog list."""
    pts = extract_pts_series(gamelog)
    reb = extract_reb_series(gamelog)
    ast = extract_ast_series(gamelog)
    pra = extract_pra_series(gamelog)

    def mu_sd(arr):
        if len(arr) == 0:
            return 0.0, 1.0
        a = np.array(arr, float)
        mu = float(a.mean())
        sd = float(a.std() if a.std() > 0 else 1.0)
        return mu, sd

    mu_pts, sd_pts = mu_sd(pts)
    mu_reb, sd_reb = mu_sd(reb)
    mu_ast, sd_ast = mu_sd(ast)
    mu_pra = mu_pts + mu_reb + mu_ast
    sd_pra = float(np.sqrt(sd_pts**2 + sd_reb**2 + sd_ast**2))

    return {
        "mu": {
            "PTS": mu_pts,
            "REB": mu_reb,
            "AST": mu_ast,
            "PRA": mu_pra
        },
        "sd": {
            "PTS": sd_pts,
            "REB": sd_reb,
            "AST": sd_ast,
            "PRA": sd_pra
        }
    }

###############################
# SEGMENT 4 — CHUNK 2
# Covariance & Correlation Engine
###############################

def compute_correlations(gamelog):
    """Compute correlation & covariance matrices for PTS/REB/AST.
    Returns:
        corr: 3x3 matrix
        cov:  3x3 matrix
    Fallbacks to identity matrices if insufficient samples.
    """
    pts = extract_pts_series(gamelog)
    reb = extract_reb_series(gamelog)
    ast = extract_ast_series(gamelog)

    data = []
    for i in range(min(len(pts), len(reb), len(ast))):
        data.append([pts[i], reb[i], ast[i]])

    if len(data) < 3:
        return np.eye(3).tolist(), np.eye(3).tolist()

    arr = np.array(data, float)

    try:
        corr = np.corrcoef(arr, rowvar=False)
        cov = np.cov(arr, rowvar=False)
    except Exception:
        return np.eye(3).tolist(), np.eye(3).tolist()

    return corr.tolist(), cov.tolist()

###############################
# SEGMENT 4 — CHUNK 3
# Probability Model (Gaussian Over/Under)
###############################

def compute_probabilities(mu, sd, line):
    """Compute Gaussian-based over/under probabilities.
    Returns:
        { 'over': p_over, 'under': p_under }
    """
    try:
        if sd <= 0:
            return {"over": 0.5, "under": 0.5}

        # Z-score
        z = (line - mu) / sd

        under = 0.5 * (1 + erf(z / (2**0.5)))
        over = 1 - under

        return {
            "over": float(over),
            "under": float(under)
        }
    except Exception:
        return {"over": 0.5, "under": 0.5}

###############################
# SEGMENT 4 — CHUNK 4
# Calibration Helpers (Bias Adjustments)
###############################

def apply_calibration(mu_dict, sd_dict, bias_dict):
    """Apply calibration bias corrections to MU and SD.
    - bias_dict may contain entries like:
        { 'PTS': +0.4, 'PTS_sd': +0.05, ... }
    - Missing keys default to zero / no effect.
    """
    out_mu = {}
    out_sd = {}

    for stat in mu_dict:
        # MU bias
        mu_bias = bias_dict.get(stat, 0.0)
        out_mu[stat] = float(mu_dict[stat] + mu_bias)

        # SD bias
        sd_bias = bias_dict.get(f"{stat}_sd", 0.0)
        sd_scaled = sd_dict[stat] * (1 + sd_bias)
        out_sd[stat] = float(max(0.1, sd_scaled))  # clamp for safety

    return out_mu, out_sd

###############################
# SEGMENT 4 — CHUNK 5
# Market Metric Dictionary
###############################

MARKET_METRICS = {
    "PTS": {
        "label": "Points",
        "precision": 1,
        "key": "PTS"
    },
    "REB": {
        "label": "Rebounds",
        "precision": 1,
        "key": "REB"
    },
    "AST": {
        "label": "Assists",
        "precision": 1,
        "key": "AST"
    },
    "PRA": {
        "label": "Points + Rebounds + Assists",
        "precision": 1,
        "key": "PRA"
    }
}

###############################
# SEGMENT 5 — CHUNK 1
# UltraMAX Trend Engine (EMA + Z-score Drift)
###############################

def compute_trend_engine(series):
    """Compute trend multiplier using:
    - 5-game EMA
    - Z-score of last value vs recent window
    - Drift multiplier (0.85–1.20 range)
    """
    if not isinstance(series, (list, tuple)) or len(series) < 3:
        return {
            "ema": None,
            "zscore": 0.0,
            "direction": "neutral",
            "multiplier": 1.00
        }

    arr = np.array(series, float)
    window = arr[-10:] if len(arr) >= 10 else arr
    mean = float(window.mean())
    sd = float(window.std() if window.std() > 0 else 1.0)

    # Z-score of most recent game
    z = float((arr[-1] - mean) / sd)

    # EMA(5)
    try:
        ema = float(pd.Series(arr).ewm(span=5).mean().iloc[-1])
    except Exception:
        ema = None

    # Drift scaling based on Z
    drift = float(1.0 + 0.05 * z)
    drift = max(0.85, min(1.20, drift))

    direction = (
        "up" if z > 0.5 else
        "down" if z < -0.5 else
        "neutral"
    )

    return {
        "ema": ema,
        "zscore": z,
        "direction": direction,
        "multiplier": drift
    }

###############################
# SEGMENT 5 — CHUNK 2
# UltraMAX Rotation Volatility Engine
###############################

def compute_rotation_volatility(minutes_series, foul_rate=0.15, coach_trust=75, bench_depth=3, games_back=5):
    """Compute rotation volatility multiplier.
    Inputs:
        minutes_series : list of MP values
        foul_rate      : player's foul frequency (0–1)
        coach_trust    : 0–100 scale
        bench_depth    : # of usable bench players
        games_back     : games since return from injury (conditioning)
    """
    # Need a few minute samples
    if not isinstance(minutes_series, (list, tuple)) or len(minutes_series) < 3:
        return {
            "minutes_sd": 0.0,
            "volatility": 1.00
        }

    arr = np.array(minutes_series, float)
    sd_min = float(arr.std() if arr.std() > 0 else 0.01)

    # Components of volatility
    foul_component = float(foul_rate * 0.05)
    trust_component = float((100 - coach_trust) / 200)      # lower trust → higher volatility
    bench_component = float(bench_depth * 0.02)             # deeper bench → more risk
    games_back_component = float(max(0, (5 - games_back)) * 0.05)

    # Combine raw volatility
    raw_vol = (sd_min / 6.0) + foul_component + trust_component + bench_component + games_back_component

    # Clamp multiplier
    volatility = float(max(0.85, min(1.30, 1 + raw_vol)))

    return {
        "minutes_sd": sd_min,
        "volatility": volatility
    }

###############################
# SEGMENT 5 — CHUNK 3
# UltraMAX Blowout Impact Engine
###############################

def compute_blowout_multiplier(spread, role="starter"):
    """Compute blowout risk multiplier using a smooth sigmoid curve.
    Parameters:
        spread : favorite margin (+ spread means player's team favored)
        role   : "starter" or "bench"
    Returns:
        multiplier in range [0.70, 1.10]
    """
    try:
        # Sigmoid scaling: large spreads reduce minutes for starters
        base_prob = 1 / (1 + np.exp(-spread / 6))

        if role.lower() == "starter":
            mult = 1 - base_prob * 0.22   # starters get reduced minutes
        else:
            mult = 1 - base_prob * 0.12   # bench players get slightly less reduction

        # Clamp
        mult = float(max(0.70, min(1.10, mult)))
        return mult

    except Exception:
        return 1.00

###############################
# SEGMENT 5 — CHUNK 4
# UltraMAX Defensive Multiplier Engine
###############################

def compute_defensive_multiplier(def_rating):
    """Convert opponent defensive rating into a projection multiplier.
    - League median is approx 113
    - Lower rating = tougher defense (mult < 1)
    - Higher rating = weaker defense (mult > 1)
    """
    try:
        mult = 113 / float(def_rating if def_rating > 0 else 113)
        mult = float(max(0.85, min(1.20, mult)))
        return mult
    except Exception:
        return 1.00

###############################
# SEGMENT 5 — CHUNK 5
# UltraMAX Team Context Engine
###############################

def compute_team_context_multiplier(team_pace, opp_pace):
    """Compute pace-based context multiplier.
    - Higher combined pace → more possessions → >1.0
    - Lower combined pace → fewer possessions → <1.0
    Clamped to [0.85, 1.15].
    """
    try:
        pace = (float(team_pace) + float(opp_pace)) / 200.0
        mult = max(0.85, min(1.15, pace))
        return float(mult)
    except Exception:
        return 1.00

###############################
# SEGMENT 5 — CHUNK 6
# UltraMAX Synergy Engine (Usage-Based Scaling)
###############################

def compute_synergy_multiplier(usage_rate):
    """Compute synergy multiplier based on usage rate.
    Usage is a strong indicator of offensive involvement.
    Returns a multiplier in the typical UltraMAX range:
        0.90 → 1.12
    """
    try:
        u = float(usage_rate)

        if u >= 30:
            return 1.12
        elif u >= 24:
            return 1.05
        elif u >= 20:
            return 1.00
        elif u >= 15:
            return 0.95
        else:
            return 0.90

    except Exception:
        return 1.00

###############################
# SEGMENT 5 — CHUNK 7
# UltraMAX Similarity Engine (Cosine Similarity)
###############################

def compute_similarity_score(vec_a, vec_b):
    """Compute cosine similarity between two numeric vectors.
    Returns value in [0.0, 1.0].
    """
    try:
        a = np.array(vec_a, float)
        b = np.array(vec_b, float)

        # Zero-vector safety
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0

        score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        # Clamp in case of rounding artifacts
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5  # neutral similarity

###############################
# SEGMENT 6 — CHUNK 1
# UltraMAX Projection Fusion Engine
###############################

def fuse_projection(
    base_mu, base_sd,
    trend_mult, rotation_mult, blowout_mult,
    context_mult, defense_mult, synergy_mult
):
    """Combine UltraMAX multipliers with baseline MU/SD to produce
    final adjusted projections for PTS, REB, AST, PRA.
    """

    fused_mu = {}
    fused_sd = {}

    # Combined multiplier
    adj_mult = (
        trend_mult *
        rotation_mult *
        blowout_mult *
        context_mult *
        defense_mult *
        synergy_mult
    )

    for stat in ["PTS", "REB", "AST"]:
        # Adjust MU
        fused_mu[stat] = float(base_mu[stat] * adj_mult)

        # Adjust SD (volatility applies more strongly)
        fused_sd_val = base_sd[stat] * max(0.75, rotation_mult * 0.9)
        fused_sd[stat] = float(max(0.1, fused_sd_val))  # safety clamp

    # Compute PRA
    fused_mu["PRA"] = fused_mu["PTS"] + fused_mu["REB"] + fused_mu["AST"]
    fused_sd["PRA"] = float(
        (fused_sd["PTS"]**2 + fused_sd["REB"]**2 + fused_sd["AST"]**2) ** 0.5
    )

    return fused_mu, fused_sd

###############################
# SEGMENT 6 — CHUNK 2
# UltraMAX Monte Carlo Simulation Engine
###############################

def run_monte_carlo(mu_dict, sd_dict, size=20000):
    """Monte Carlo engine using independent normal sampling
    (correlation handled earlier via fusion + optional joint EV).

    Returns dict of arrays:
        { 'PTS': arr, 'REB': arr, 'AST': arr, 'PRA': arr }
    """
    try:
        size = int(size)

        pts = np.random.normal(mu_dict["PTS"], sd_dict["PTS"], size)
        reb = np.random.normal(mu_dict["REB"], sd_dict["REB"], size)
        ast = np.random.normal(mu_dict["AST"], sd_dict["AST"], size)
        pra = pts + reb + ast

        return {
            "PTS": pts,
            "REB": reb,
            "AST": ast,
            "PRA": pra
        }
    except Exception:
        return {
            "PTS": np.array([]),
            "REB": np.array([]),
            "AST": np.array([]),
            "PRA": np.array([])
        }

###############################
# SEGMENT 6 — CHUNK 3
# UltraMAX Joint EV Engine
###############################

def evaluate_joint_ev(mc_dict, legs):
    """Evaluate the probability and EV of multiple legs hitting simultaneously.
    legs = [
        { 'market': 'PTS', 'type':'over', 'line': 24.5 },
        { 'market': 'REB', 'type':'under','line': 10.5 },
        ...
    ]
    Returns:
        { 'probability': p, 'ev': ev, 'payout': payout }
    """
    try:
        size = len(mc_dict["PTS"])
        if size == 0:
            return {"probability": 0.0, "ev": -1.0, "payout": 0}

        mask = np.ones(size, dtype=bool)

        for leg in legs:
            stat = leg["market"]
            line = float(leg["line"])
            arr = mc_dict[stat]

            if leg["type"] == "over":
                mask &= (arr > line)
            else:
                mask &= (arr < line)

        prob = float(mask.mean())

        payout_table = {2: 3, 3: 5, 4: 10, 5: 25, 6: 30}
        payout = payout_table.get(len(legs), 0)

        ev = float(prob * payout - 1.0)

        return {
            "probability": prob,
            "ev": ev,
            "payout": payout
        }

    except Exception:
        return {"probability": 0.0, "ev": -1.0, "payout": 0}

###############################
# SEGMENT 6 — CHUNK 4
# UltraMAX Master Engine — compute_leg_projection()
###############################

def compute_leg_projection(player_name, team, opponent, season, usage_rate,
                           foul_rate=0.15, coach_trust=75, bench_depth=3,
                           games_back=5, spread=0, role="starter",
                           calibration_bias=None, line_dict=None):
    """Master engine that orchestrates all data loading and UltraMAX modules.

    Returns a final packet:
    {
        'fused_mu': {...},
        'fused_sd': {...},
        'probabilities': {...},
        'monte_carlo': {...},
        'trend': {...},
        'rotation': {...},
        'blowout': ...,
        'defense_mult': ...,
        'context_mult': ...,
        'synergy_mult': ...,
        'baseline': {...},
        'corr': matrix,
        'cov': matrix
    }
    """

    # 1. Resolve player
    entity = resolve_player_for_all_sources(player_name)
    br_id = entity["br_id"]
    clean_player = entity["clean"]

    # 2. Load gamelog
    gamelog = fetch_gamelog(clean_player, season)

    # 3. Extract series
    pts_series = extract_pts_series(gamelog)
    reb_series = extract_reb_series(gamelog)
    ast_series = extract_ast_series(gamelog)
    pra_series = extract_pra_series(gamelog)
    min_series = extract_minutes_series(gamelog)

    # 4. Baseline stats
    baseline = compute_baseline_stats(gamelog)

    # 5. Correlation + covariance
    corr, cov = compute_correlations(gamelog)

    # 6. UltraMAX engines
    trend = compute_trend_engine(pts_series)  # trend uses points by default
    rotation = compute_rotation_volatility(min_series, foul_rate, coach_trust, bench_depth, games_back)
    blow_mult = compute_blowout_multiplier(spread, role)

    # Defense + context loaders
    team_context = load_team_context(team, opponent)
    defense_prof = load_defensive_profile(opponent)

    defense_mult = compute_defensive_multiplier(team_context["opp_def_rating"])
    context_mult = compute_team_context_multiplier(team_context["team_pace"],
                                                   team_context["opp_pace"])
    synergy_mult = compute_synergy_multiplier(usage_rate)

    # 7. Fuse projection
    fused_mu, fused_sd = fuse_projection(
        baseline["mu"], baseline["sd"],
        trend["multiplier"], rotation["volatility"], blow_mult,
        context_mult, defense_mult, synergy_mult
    )

    # 8. Calibration bias
    if calibration_bias:
        fused_mu, fused_sd = apply_calibration(fused_mu, fused_sd, calibration_bias)

    # 9. Monte Carlo
    mc = run_monte_carlo(fused_mu, fused_sd, size=20000)

    # 10. Single-leg probability pricing (if lines exist)
    prob_pack = {}
    if line_dict:
        for stat, line in line_dict.items():
            try:
                p = compute_probabilities(fused_mu[stat], fused_sd[stat], float(line))
                prob_pack[stat] = p
            except:
                continue

    # 11. Build final engine packet
    return {
        "fused_mu": fused_mu,
        "fused_sd": fused_sd,
        "probabilities": prob_pack,
        "monte_carlo": mc,
        "trend": trend,
        "rotation": rotation,
        "blowout": blow_mult,
        "defense_mult": defense_mult,
        "context_mult": context_mult,
        "synergy_mult": synergy_mult,
        "baseline": baseline,
        "corr": corr,
        "cov": cov
    }

###############################
# SEGMENT 6 — CHUNK 5
# Engine Packet Builder (UI-Ready Output)
###############################

def build_engine_packet(player_name, engine_output, lines_used=None):
    """Wrap the master engine output into a clean, UI-friendly packet.
    This ensures consistency across:
    - Model Tab
    - EV Tab
    - Player Card
    - Line Shopping
    - Overrides
    - Trend/Rotation/Defense Context Tabs
    """

    packet = {}

    # Player field
    packet["player"] = player_name

    # Fused projections
    packet["mu"] = engine_output.get("fused_mu", {})
    packet["sd"] = engine_output.get("fused_sd", {})

    # Probabilities (if lines were provided)
    packet["probabilities"] = engine_output.get("probabilities", {})

    # Raw baseline stats
    packet["baseline"] = engine_output.get("baseline", {})

    # UltraMAX multipliers
    packet["trend"] = engine_output.get("trend", {})
    packet["rotation"] = engine_output.get("rotation", {})
    packet["blowout"] = engine_output.get("blowout", 1.00)
    packet["defense_mult"] = engine_output.get("defense_mult", 1.00)
    packet["context_mult"] = engine_output.get("context_mult", 1.00)
    packet["synergy_mult"] = engine_output.get("synergy_mult", 1.00)

    # Correlation / Covariance
    packet["corr"] = engine_output.get("corr", [])
    packet["cov"] = engine_output.get("cov", [])

    # MC distribution
    packet["monte_carlo"] = engine_output.get("monte_carlo", {})

    # Lines used (if any)
    packet["lines_used"] = lines_used or {}

    # Metadata
    packet["timestamp"] = datetime.utcnow().isoformat()

    return packet

###############################
# SEGMENT 7 — CHUNK 1
# Global Streamlit Config & Styling
###############################

import streamlit as st

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="UltraMAX NBA Quant Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Dark Theme CSS ---
GLOBAL_CSS = """
<style>
body {
    background-color: #0d0d0d !important;
    color: #e6e6e6 !important;
}
.stApp {
    background-color: #0d0d0d !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 700 !important;
}
.sidebar .sidebar-content {
    background-color: #1a1a1a !important;
}
.stSidebar {
    background-color: #1a1a1a !important;
}
.block-container {
    padding-top: 1rem !important;
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# --- Global Header ---
def render_header():
    st.markdown(
        """
        <div style="padding:12px 0; text-align:center;">
            <h1 style="margin-bottom:0;">🏀 UltraMAX NBA Quant Engine</h1>
            <p style="font-size:14px; color:#bbbbbb; margin-top:-8px;">
                2025–2026 Automated Projection • Monte Carlo • Joint EV • Defensive Matchups
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

###############################
# SEGMENT 7 — CHUNK 2
# Sidebar UI + LIVE1 Integration
###############################

def render_sidebar():
    st.sidebar.markdown("## 🔧 Model Configuration")

    # --- Player Name ---
    player_name = st.sidebar.text_input("Player Name", "")

    # --- Team & Opponent ---
    team = st.sidebar.text_input("Player Team", "")
    opponent = st.sidebar.text_input("Opponent Team", "")

    # --- Role Selector ---
    role = st.sidebar.selectbox("Role", ["starter", "bench"])

    # --- Usage Rate ---
    usage_rate = st.sidebar.number_input("Usage Rate (%)", 5.0, 40.0, 22.0)

    # --- Foul Rate ---
    foul_rate = st.sidebar.number_input("Foul Rate (0–1)", 0.0, 1.0, 0.15)

    # --- Coach Trust ---
    coach_trust = st.sidebar.slider("Coach Trust (0–100)", 0, 100, 75)

    # --- Bench Depth ---
    bench_depth = st.sidebar.number_input("Bench Depth", 1, 12, 3)

    # --- Games Back ---
    games_back = st.sidebar.number_input("Games Back", 0, 20, 5)

    # --- Spread ---
    spread = st.sidebar.number_input("Spread (Team favored +)", -25.0, 25.0, 0.0)

    # --- Season (Auto-updating for 2025–26) ---
    season = st.sidebar.selectbox("Season", ["2025-26", "2024-25", "2023-24"], index=0)

    # --- Live Lines Section ---
    st.sidebar.markdown("### 📡 LIVE Lines (Auto-Fill)")
    enable_live = st.sidebar.checkbox("Enable LIVE1 Auto-Fill", value=False)

    # Container for live lines
    default_lines = {"PTS": None, "REB": None, "AST": None, "PRA": None}

    if enable_live and player_name.strip() != "":
        resolved = resolve_player_for_all_sources(player_name)
        live_lines = get_live_lines(resolved["raw"])
        safe_live = safe_lines_output(live_lines)
        st.sidebar.write("Live Lines:", safe_live)
    else:
        safe_live = default_lines

    # --- Manual Stat Lines (with LIVE1 Auto-Fill) ---
    line_pts = st.sidebar.number_input("PTS Line", 0.0, 100.0, float(safe_live.get("PTS") or 0.0))
    line_reb = st.sidebar.number_input("REB Line", 0.0, 50.0, float(safe_live.get("REB") or 0.0))
    line_ast = st.sidebar.number_input("AST Line", 0.0, 40.0, float(safe_live.get("AST") or 0.0))
    line_pra = st.sidebar.number_input("PRA Line", 0.0, 120.0, float(safe_live.get("PRA") or 0.0))

    # Run Button
    run_model = st.sidebar.button("🚀 Run UltraMAX Model")

    return {
        "player_name": player_name,
        "team": team,
        "opponent": opponent,
        "role": role,
        "usage_rate": usage_rate,
        "foul_rate": foul_rate,
        "coach_trust": coach_trust,
        "bench_depth": bench_depth,
        "games_back": games_back,
        "spread": spread,
        "season": season,
        "lines": {
            "PTS": line_pts,
            "REB": line_reb,
            "AST": line_ast,
            "PRA": line_pra
        },
        "run_model": run_model
    }

###############################
# SEGMENT 7 — CHUNK 3
# Model Tab UI — Core UltraMAX Output
###############################

def render_model_tab(sidebar_state):
    render_header()
    st.markdown("## 📊 UltraMAX Model Output")

    # Stop if user hasn't run model
    if not sidebar_state.get("run_model"):
        st.info("Configure parameters in the sidebar and click **Run UltraMAX Model**.")
        return

    # Prepare inputs
    player = sidebar_state["player_name"]
    team = sidebar_state["team"]
    opponent = sidebar_state["opponent"]
    season = sidebar_state["season"]
    usage = sidebar_state["usage_rate"]
    foul_rate = sidebar_state["foul_rate"]
    coach_trust = sidebar_state["coach_trust"]
    bench_depth = sidebar_state["bench_depth"]
    games_back = sidebar_state["games_back"]
    spread = sidebar_state["spread"]
    role = sidebar_state["role"]
    lines = sidebar_state["lines"]

    # Run engine
    engine_out = compute_leg_projection(
        player_name=player,
        team=team,
        opponent=opponent,
        season=season,
        usage_rate=usage,
        foul_rate=foul_rate,
        coach_trust=coach_trust,
        bench_depth=bench_depth,
        games_back=games_back,
        spread=spread,
        role=role,
        calibration_bias=None,
        line_dict=lines
    )

    packet = build_engine_packet(player, engine_out, lines_used=lines)

    # --- Display Results ---
    st.markdown(f"### 🏀 Player: **{player}**")
    st.markdown(f"**Season:** {season} | **Team:** {team} | **Opponent:** {opponent}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔮 Projected MU")
        st.write(packet["mu"])

        st.markdown("#### 📈 Baseline MU/SD")
        st.write(packet["baseline"])

    with col2:
        st.markdown("#### 📉 Projected SD")
        st.write(packet["sd"])

        st.markdown("#### 🎯 Probabilities (O/U)")
        st.write(packet["probabilities"])

    # Multipliers Section
    st.markdown("### ⚙️ UltraMAX Multipliers")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.write("**Trend**", packet["trend"])
        st.write("**Rotation**", packet["rotation"])

    with m2:
        st.write("**Defense Multiplier**", packet["defense_mult"])
        st.write("**Context Multiplier**", packet["context_mult"])

    with m3:
        st.write("**Blowout Multiplier**", packet["blowout"])
        st.write("**Synergy Multiplier**", packet["synergy_mult"])

    st.markdown("### 📊 Monte Carlo Summary")
    mc = packet["monte_carlo"]
    if mc.get("PTS", []).size > 0:
        st.write({
            "PTS Mean": float(mc["PTS"].mean()),
            "REB Mean": float(mc["REB"].mean()),
            "AST Mean": float(mc["AST"].mean()),
            "PRA Mean": float(mc["PRA"].mean())
        })
    else:
        st.warning("Monte Carlo simulation unavailable.")


###############################
# SEGMENT 7 — CHUNK 3.5
# Page Router / Navigation Core
###############################

def run_app():
    sidebar_state = render_sidebar()

    # Navigation Tabs
    tabs = st.tabs([
        "Model",
        "EV Model",
        "Joint EV",
        "Trends",
        "Rotation",
        "Defense",
        "Context",
        "Line Shopping",
        "Overrides",
        "History"
    ])

    with tabs[0]:
        render_model_tab(sidebar_state)

    # Remaining tabs will be added in future chunks

###############################
# SEGMENT 7 — CHUNK 4
# EV Model Tab
###############################

def render_ev_tab(sidebar_state):
    render_header()
    st.markdown("## 💰 EV Model (Single-Leg Expected Value)")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX model from the sidebar first.")
        return

    player = sidebar_state["player_name"]
    team = sidebar_state["team"]
    opponent = sidebar_state["opponent"]
    season = sidebar_state["season"]
    usage = sidebar_state["usage_rate"]
    foul_rate = sidebar_state["foul_rate"]
    coach_trust = sidebar_state["coach_trust"]
    bench_depth = sidebar_state["bench_depth"]
    games_back = sidebar_state["games_back"]
    spread = sidebar_state["spread"]
    role = sidebar_state["role"]
    lines = sidebar_state["lines"]

    # Compute engine
    engine_out = compute_leg_projection(
        player_name=player,
        team=team,
        opponent=opponent,
        season=season,
        usage_rate=usage,
        foul_rate=foul_rate,
        coach_trust=coach_trust,
        bench_depth=bench_depth,
        games_back=games_back,
        spread=spread,
        role=role,
        calibration_bias=None,
        line_dict=lines
    )

    packet = build_engine_packet(player, engine_out, lines_used=lines)
    mc = packet.get("monte_carlo", {})
    probs = packet.get("probabilities", {})
    fused_mu = packet.get("mu", {})
    fused_sd = packet.get("sd", {})

    st.markdown(f"### Player: **{player}** ({season})")

    markets = ["PTS", "REB", "AST", "PRA"]

    for stat in markets:
        line = lines.get(stat)
        if line is None:
            continue

        st.markdown(f"#### **{stat}** — Line: **{line}**")

        # Gaussian EV
        prob_over = probs.get(stat, {}).get("over", 0.5)
        gaussian_ev = prob_over * 1.0 - 1.0  # simple EV (PrizePicks-style placeholder)

        # MC EV (probability from simulation)
        mc_arr = mc.get(stat, [])
        if mc_arr.size > 0:
            mc_prob_over = float((mc_arr > line).mean())
            mc_ev = mc_prob_over * 1.0 - 1.0
        else:
            mc_prob_over = 0.0
            mc_ev = -1.0

        st.write({
            "Gaussian Over Probability": prob_over,
            "Gaussian EV": gaussian_ev,
            "Monte Carlo Over Probability": mc_prob_over,
            "Monte Carlo EV": mc_ev,
            "Projected MU": fused_mu.get(stat),
            "Projected SD": fused_sd.get(stat)
        })

###############################
# SEGMENT 7 — CHUNK 5
# Joint EV Tab (Multi‑Leg Evaluation)
###############################

def render_joint_ev_tab(sidebar_state):
    render_header()
    st.markdown("## 🔗 Joint EV (Multi‑Leg Correlated Evaluation)")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX Model from the sidebar first.")
        return

    player = sidebar_state["player_name"]
    team = sidebar_state["team"]
    opponent = sidebar_state["opponent"]
    season = sidebar_state["season"]
    usage = sidebar_state["usage_rate"]
    foul_rate = sidebar_state["foul_rate"]
    coach_trust = sidebar_state["coach_trust"]
    bench_depth = sidebar_state["bench_depth"]
    games_back = sidebar_state["games_back"]
    spread = sidebar_state["spread"]
    role = sidebar_state["role"]
    lines = sidebar_state["lines"]

    # Run engine again to ensure freshest packet
    engine_out = compute_leg_projection(
        player_name=player,
        team=team,
        opponent=opponent,
        season=season,
        usage_rate=usage,
        foul_rate=foul_rate,
        coach_trust=coach_trust,
        bench_depth=bench_depth,
        games_back=games_back,
        spread=spread,
        role=role,
        calibration_bias=None,
        line_dict=lines
    )

    packet = build_engine_packet(player, engine_out, lines_used=lines)
    mc = packet.get("monte_carlo", {})
    lines_dict = packet.get("lines_used", {})

    st.markdown("### 🧩 Select Legs for Joint EV")

    markets = ["PTS", "REB", "AST", "PRA"]
    selected_legs = []

    cols = st.columns(4)
    for i, stat in enumerate(markets):
        with cols[i]:
            include = st.checkbox(f"Include {stat}", value=False)
            if include:
                l = lines_dict.get(stat, None)
                if l is not None:
                    leg_type = st.selectbox(f"{stat} Type", ["over","under"], key=f"{stat}_sel")
                    selected_legs.append({
                        "market": stat,
                        "type": leg_type,
                        "line": l
                    })

    if len(selected_legs) < 2:
        st.warning("Select at least 2 legs for Joint EV evaluation.")
        return

    # Compute Joint EV
    joint_result = evaluate_joint_ev(mc, selected_legs)

    st.markdown("### 📉 Joint EV Results")
    st.write(joint_result)

###############################
# SEGMENT 7 — CHUNK 6
# Trends Tab (EMA • Z‑Score • Drift)
###############################

def render_trends_tab(sidebar_state):
    render_header()
    st.markdown("## 📈 Trends & Form Analysis")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX model from the sidebar first.")
        return

    # Basic inputs
    player = sidebar_state["player_name"]
    team = sidebar_state["team"]
    opponent = sidebar_state["opponent"]
    season = sidebar_state["season"]
    usage = sidebar_state["usage_rate"]
    foul_rate = sidebar_state["foul_rate"]
    coach_trust = sidebar_state["coach_trust"]
    bench_depth = sidebar_state["bench_depth"]
    games_back = sidebar_state["games_back"]
    spread = sidebar_state["spread"]
    role = sidebar_state["role"]
    lines = sidebar_state["lines"]

    # Engine call
    engine_out = compute_leg_projection(
        player_name=player,
        team=team,
        opponent=opponent,
        season=season,
        usage_rate=usage,
        foul_rate=foul_rate,
        coach_trust=coach_trust,
        bench_depth=bench_depth,
        games_back=games_back,
        spread=spread,
        role=role,
        calibration_bias=None,
        line_dict=lines
    )

    packet = build_engine_packet(player, engine_out, lines_used=lines)

    st.markdown(f"### Player: **{player}**")

    trend = packet.get("trend", {})
    baseline = packet.get("baseline", {})
    mc = packet.get("monte_carlo", {})

    # Display Trend Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EMA (Last 5)", f"{trend.get('ema', 0):.2f}")
    with col2:
        st.metric("Z‑Score", f"{trend.get('zscore', 0):.2f}")
    with col3:
        st.metric("Direction", trend.get("direction", "neutral"))

    st.markdown("### 🔥 Drift Multiplier")
    st.write(f"**{trend.get('multiplier', 1.0):.3f}**")

    # Raw series chart placeholder
    pts_series = engine_out.get("baseline", {}).get("mu", {}).get("PTS", None)

    gamelog_pts = extract_pts_series(fetch_gamelog(player, season))

    if len(gamelog_pts) > 0:
        st.markdown("### 📉 Recent PTS Game Log")
        st.line_chart(gamelog_pts)
    else:
        st.warning("Not enough gamelog data to visualize.")

###############################
# SEGMENT 7 — CHUNK 7
# Rotation Volatility Tab
###############################

def render_rotation_tab(sidebar_state):
    render_header()
    st.markdown("## 🔄 Rotation Volatility Analysis")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX model from the sidebar first.")
        return

    # Basic inputs
    player = sidebar_state["player_name"]
    team = sidebar_state["team"]
    opponent = sidebar_state["opponent"]
    season = sidebar_state["season"]
    usage = sidebar_state["usage_rate"]
    foul_rate = sidebar_state["foul_rate"]
    coach_trust = sidebar_state["coach_trust"]
    bench_depth = sidebar_state["bench_depth"]
    games_back = sidebar_state["games_back"]
    spread = sidebar_state["spread"]
    role = sidebar_state["role"]
    lines = sidebar_state["lines"]

    # Run engine
    engine_out = compute_leg_projection(
        player_name=player,
        team=team,
        opponent=opponent,
        season=season,
        usage_rate=usage,
        foul_rate=foul_rate,
        coach_trust=coach_trust,
        bench_depth=bench_depth,
        games_back=games_back,
        spread=spread,
        role=role,
        calibration_bias=None,
        line_dict=lines
    )

    packet = build_engine_packet(player, engine_out, lines_used=lines)
    rotation = packet.get("rotation", {})

    st.markdown(f"### Player: **{player}** — Rotation Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Minutes SD", f"{rotation.get('minutes_sd', 0):.2f}")

    with col2:
        st.metric("Volatility Multiplier", f"{rotation.get('volatility', 1.0):.3f}")

    st.markdown("### 🎯 Volatility Factors")

    st.write({
        "Foul Rate Impact": foul_rate * 0.05,
        "Coach Trust Factor": (100 - coach_trust) / 200,
        "Bench Depth Factor": bench_depth * 0.02,
        "Games‑Back Conditioning": max(0, (5 - games_back)) * 0.05
    })

    # Minutes series visualization
    gamelog = fetch_gamelog(player, season)
    mins = extract_minutes_series(gamelog)

    if len(mins) > 0:
        st.markdown("### 📉 Minutes History")
        st.line_chart(mins)
    else:
        st.warning("Not enough minutes data to visualize.")

###############################
# SEGMENT 7 — CHUNK 8
# Defensive Profile Tab
###############################

def render_defense_tab(sidebar_state):
    render_header()
    st.markdown("## 🛡️ Defensive Matchup Profile")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX model from the sidebar first.")
        return

    # Basic inputs
    player = sidebar_state["player_name"]
    team = sidebar_state["team"]
    opponent = sidebar_state["opponent"]
    season = sidebar_state["season"]
    usage = sidebar_state["usage_rate"]
    foul_rate = sidebar_state["foul_rate"]
    coach_trust = sidebar_state["coach_trust"]
    bench_depth = sidebar_state["bench_depth"]
    games_back = sidebar_state["games_back"]
    spread = sidebar_state["spread"]
    role = sidebar_state["role"]
    lines = sidebar_state["lines"]

    # Run engine
    engine_out = compute_leg_projection(
        player_name=player,
        team=team,
        opponent=opponent,
        season=season,
        usage_rate=usage,
        foul_rate=foul_rate,
        coach_trust=coach_trust,
        bench_depth=bench_depth,
        games_back=games_back,
        spread=spread,
        role=role,
        calibration_bias=None,
        line_dict=lines
    )

    packet = build_engine_packet(player, engine_out, lines_used=lines)

    st.markdown(f"### Opponent: **{opponent}** — Defensive Profile")

    defense_mult = packet.get("defense_mult", 1.00)
    context = load_team_context(team, opponent)
    profile = load_defensive_profile(opponent)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Opponent Defensive Rating", f"{context.get('opp_def_rating', 113):.1f}")
    with col2:
        st.metric("UltraMAX Defense Multiplier", f"{defense_mult:.3f}")
    with col3:
        st.metric("Team Pace Influence", f"{context.get('team_pace', 100):.1f}")

    st.markdown("### 🔒 Suppression Factors")
    st.write({
        "PTS Suppression": profile.get("pts_mult", 1.0),
        "REB Suppression": profile.get("reb_mult", 1.0),
        "AST Suppression": profile.get("ast_mult", 1.0)
    })

    # Optional: simple defensive bars
    st.markdown("### 📊 Defensive Difficulty Bar")
    try:
        st.progress(min(1.0, max(0.0, 1 / defense_mult)))
    except:
        st.warning("Unable to render defensive difficulty.")

###############################
# SEGMENT 7 — CHUNK 9
# Team Context Tab (Pace • Opp Pace • Multiplier)
###############################

def render_context_tab(sidebar_state):
    render_header()
    st.markdown("## 🏃‍♂️💨 Team Context & Pace Analysis")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX model from the sidebar first.")
        return

    # Basic inputs
    player = sidebar_state["player_name"]
    team = sidebar_state["team"]
    opponent = sidebar_state["opponent"]
    season = sidebar_state["season"]
    usage = sidebar_state["usage_rate"]
    foul_rate = sidebar_state["foul_rate"]
    coach_trust = sidebar_state["coach_trust"]
    bench_depth = sidebar_state["bench_depth"]
    games_back = sidebar_state["games_back"]
    spread = sidebar_state["spread"]
    role = sidebar_state["role"]
    lines = sidebar_state["lines"]

    # Execute engine
    engine_out = compute_leg_projection(
        player_name=player,
        team=team,
        opponent=opponent,
        season=season,
        usage_rate=usage,
        foul_rate=foul_rate,
        coach_trust=coach_trust,
        bench_depth=bench_depth,
        games_back=games_back,
        spread=spread,
        role=role,
        calibration_bias=None,
        line_dict=lines
    )

    packet = build_engine_packet(player, engine_out, lines_used=lines)

    st.markdown(f"### Matchup: **{team} vs {opponent}**")

    context = engine_out.get("context_mult", 1.00)
    team_ctx_data = load_team_context(team, opponent)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Team Pace", f"{team_ctx_data.get('team_pace', 100):.1f}")
    with col2:
        st.metric("Opponent Pace", f"{team_ctx_data.get('opp_pace', 100):.1f}")
    with col3:
        st.metric("Context Multiplier", f"{context:.3f}")

    st.markdown("### 📊 Pace-Based Interpretation")

    team_pace = team_ctx_data.get("team_pace", 100)
    opp_pace = team_ctx_data.get("opp_pace", 100)
    avg_pace = (team_pace + opp_pace) / 2

    if avg_pace > 102:
        st.success("High-paced environment → More possessions → Projection boost likely.")
    elif avg_pace < 97:
        st.warning("Low-paced environment → Fewer possessions → Projection may drop.")
    else:
        st.info("Moderate pace environment → Neutral projection flow.")

    # Simple pace bar
    try:
        pace_norm = min(1.0, max(0.0, avg_pace / 120))
        st.markdown("### 📉 Pace Indicator")
        st.progress(pace_norm)
    except:
        st.warning("Unable to render pace indicator.")

###############################
# SEGMENT 7 — CHUNK 10
# Line Shopping Analyzer (PP • OddsAPI • Manual)
###############################

def render_line_shopping_tab(sidebar_state):
    render_header()
    st.markdown("## 🛒 Line Shopping Analyzer")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX model first to populate lines.")
        return

    player = sidebar_state["player_name"]
    lines_manual = sidebar_state["lines"]

    # Fetch live lines
    resolved = resolve_player_for_all_sources(player)
    pp_lines = fetch_prizepicks_lines(resolved["raw"])
    oa_lines = fetch_oddsapi_lines(resolved["raw"])

    st.markdown(f"### Player: **{player}** — Market Comparison")

    markets = ["PTS", "REB", "AST", "PRA"]

    for stat in markets:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Manual Line", lines_manual.get(stat, 0.0))

        with col2:
            st.metric("PrizePicks", pp_lines.get(stat, None))

        with col3:
            st.metric("OddsAPI", oa_lines.get(stat, None))

    st.markdown("### 📊 Summary Table")
    summary = []
    for stat in markets:
        summary.append({
            "Market": stat,
            "Manual": lines_manual.get(stat),
            "PrizePicks": pp_lines.get(stat),
            "OddsAPI": oa_lines.get(stat)
        })

    st.table(summary)

###############################
# SEGMENT 7 — CHUNK 11
# Override Tab (Developer Controls)
###############################

def render_override_tab(sidebar_state):
    render_header()
    st.markdown("## 🛠️ Override Controls (Developer Mode)")

    if not sidebar_state.get("run_model"):
        st.info("Run the UltraMAX model first.")
        return

    # Basic local snapshot
    player = sidebar_state["player_name"]
    lines = sidebar_state["lines"]

    st.markdown(f"### Player: **{player}**")

    # Toggle overrides
    enable = st.checkbox("Enable Overrides", value=False)

    if not enable:
        st.warning("Overrides disabled. Toggle above to activate developer controls.")
        return

    st.success("Overrides Enabled — All changes below will modify projections.")

    # Override MU
    st.markdown("### 🎯 Override MU (Projected Means)")
    mu_override = {
        "PTS": st.number_input("Override MU — PTS", value=float(lines.get("PTS", 0.0))),
        "REB": st.number_input("Override MU — REB", value=float(lines.get("REB", 0.0))),
        "AST": st.number_input("Override MU — AST", value=float(lines.get("AST", 0.0))),
        "PRA": st.number_input("Override MU — PRA", value=float(lines.get("PRA", 0.0)))
    }

    # Override SD
    st.markdown("### 📉 Override SD (Standard Deviations)")
    sd_override = {
        "PTS": st.number_input("Override SD — PTS", value=3.0),
        "REB": st.number_input("Override SD — REB", value=2.5),
        "AST": st.number_input("Override SD — AST", value=2.0),
        "PRA": st.number_input("Override SD — PRA", value=5.0)
    }

    # Override multipliers
    st.markdown("### ⚙️ Override Multipliers (Trend • Rotation • Context • Defense • Blowout • Synergy)")
    mult_override = {
        "trend": st.number_input("Trend Multiplier Override", value=1.00),
        "rotation": st.number_input("Rotation Multiplier Override", value=1.00),
        "context": st.number_input("Context Multiplier Override", value=1.00),
        "defense": st.number_input("Defense Multiplier Override", value=1.00),
        "blowout": st.number_input("Blowout Multiplier Override", value=1.00),
        "synergy": st.number_input("Synergy Multiplier Override", value=1.00)
    }

    # MC sample override
    st.markdown("### 🎲 Monte Carlo Sample Size Override")
    mc_override = st.number_input("MC Samples", min_value=1000, max_value=500000, value=20000, step=1000)

    # Pace override
    st.markdown("### 🏃 Pace Overrides")
    pace_override = {
        "team_pace": st.number_input("Team Pace Override", value=100.0),
        "opp_pace": st.number_input("Opponent Pace Override", value=100.0)
    }

    # Apply button
    apply = st.button("Apply Overrides")

    if apply:
        st.success("Override data applied (developer mode).")
        st.write({
            "mu_override": mu_override,
            "sd_override": sd_override,
            "mult_override": mult_override,
            "mc_override": mc_override,
            "pace_override": pace_override
        })

###############################
# SEGMENT 7 — CHUNK 13
# Final Page Routing Integration
###############################

# Attach all tabs to router
def run_app():
    sidebar_state = render_sidebar()

    tabs = st.tabs([
        "Model",
        "EV Model",
        "Joint EV",
        "Trends",
        "Rotation",
        "Defense",
        "Context",
        "Line Shopping",
        "Overrides",
        "History"
    ])

    with tabs[0]:
        render_model_tab(sidebar_state)

    with tabs[1]:
        render_ev_tab(sidebar_state)

    with tabs[2]:
        render_joint_ev_tab(sidebar_state)

    with tabs[3]:
        render_trends_tab(sidebar_state)

    with tabs[4]:
        render_rotation_tab(sidebar_state)

    with tabs[5]:
        render_defense_tab(sidebar_state)

    with tabs[6]:
        render_context_tab(sidebar_state)

    with tabs[7]:
        render_line_shopping_tab(sidebar_state)

    with tabs[8]:
        render_override_tab(sidebar_state)

    with tabs[9]:
        render_history_tab(sidebar_state)

###############################
# FINAL APP LAUNCHER
###############################

if __name__ == "__main__":
    try:
        run_app()
    except Exception as e:
        st.error(f"UltraMAX App crashed: {e}")
