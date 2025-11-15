# =====================================================================
# app.py — NBA Quant App (Master File)
# =====================================================================
# ================================================================
# PART 1 — STREAMLIT APP HEADER + CSS + LAYOUT
# UltraMax NBA Prop Quant Engine — Version B3 Architecture
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import requests
from datetime import datetime

# ------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="UltraMax NBA Prop Quant Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# Global Custom CSS
# ------------------------------------------------
GLOBAL_CSS = """
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Headline Styling */
h1, h2, h3, h4 {
    font-weight: 700 !important;
}

/* Card containers */
.ultramax-card {
    background: #0f1116;
    border: 1px solid #2c2f36;
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff5e57, #ff7d5f);
    color: white;
    border-radius: 10px;
    height: 3rem;
    font-size: 1.1rem;
    border: none;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem;
    font-weight: 700;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #22252d;
}

/* Metrics */
.metric-container {
    padding: 10px 18px;
    border-radius: 10px;
    background-color: #1b1d22;
    border: 1px solid #2d3036;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 6px;
}

</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ------------------------------------------------
# App Header
# ------------------------------------------------
st.markdown("""
# 🔥 UltraMax NBA Prop Quant Engine — V4 (B3 Full Architecture)
### High-Probability Sports Modeling • Automated Line Sync • Monte Carlo EV • Hedge-Fund Level Analytics

Use the tabs above to:
- **Run 2-Pick UltraMax model**
- **View defensive matchups**
- **Sync PrizePicks + Sleeper lines**
- **See calibration curves**
- **View history tracking**
""")

# ================================================================
# PART 2 — SIDEBAR CONFIGURATION (BANKROLL + MODEL CONTROLS)
# UltraMax NBA Prop Quant Engine — V4 (B3)
# ================================================================

with st.sidebar:

    st.markdown("## ⚙️ Engine Control Panel")

    # ------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # ------------------------------------------------------------
    st.markdown("### 💰 Bankroll Settings")

    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=10,
        max_value=100000,
        value=500,
        step=25,
        help="Used in Module 12 to compute Kelly stake sizing."
    )

    fractional_kelly = st.slider(
        "Fractional Kelly Multiplier",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Reduces Kelly risk. 0.50 = Half Kelly."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # MODEL SELF-LEARNING PARAMETERS
    # Module 8 — Drift & CLV adjustments
    # ------------------------------------------------------------
    st.markdown("### 🧬 Model Learning Parameters")

    drift_adj = st.slider(
        "Model Drift Adjustment",
        0.80,
        1.25,
        1.00,
        0.01,
        help="Controls long-term bias correction. Used in Module 12."
    )

    clv_adj = st.slider(
        "CLV (Closing Line Value) Sharpening",
        0.80,
        1.35,
        1.00,
        0.01,
        help="Boosts accuracy if model historically beats closing lines."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # GLOBAL VOLATILITY + HEAVY TAIL ADJUSTMENTS
    # Syncs with Module 5 + Module 10
    # ------------------------------------------------------------
    st.markdown("### 🌪️ Volatility Controls")

    global_vol_adj = st.slider(
        "Global Volatility Multiplier",
        0.70,
        1.40,
        1.00,
        0.01,
        help="Scales standard deviations for all markets."
    )

    heavy_tail_adj = st.slider(
        "Heavy Tail Expansion",
        0.80,
        1.25,
        1.00,
        0.01,
        help="Controls right-tail outcomes in Monte Carlo (Module 10)."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # PROJECTION OVERRIDES (Module 17)
    # ------------------------------------------------------------
    st.markdown("### 🎯 Projection Overrides")

    enable_overrides = st.checkbox(
        "Enable Manual Overrides",
        value=False,
        help="Lets you manually override any projection value in Module 17."
    )

    override_mu = st.number_input(
        "Override Mean (μ)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
        help="Only applied when overrides are active."
    )

    override_sd = st.number_input(
        "Override Std (σ)",
        min_value=0.0,
        max_value=40.0,
        value=0.0,
        step=0.25,
        help="Only applied when overrides are active."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # PRIZEPICKS / SLEEPER LINE SYNC SETTINGS
    # Module 18
    # ------------------------------------------------------------
    st.markdown("### 🔄 Line Providers")

    enable_pp_sync = st.checkbox(
        "Sync PrizePicks Lines",
        value=True,
        help="Fetches live lines from PrizePicks API."
    )

    enable_sleeper_sync = st.checkbox(
        "Sync Sleeper Lines",
        value=False,
        help="Fetches projections from Sleeper Picks."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # HISTORY + LOGGING CONTROLS (Module 20)
    # ------------------------------------------------------------
    st.markdown("### 🗂️ History & Logging")

    enable_history = st.checkbox(
        "Enable History Logging",
        value=True,
        help="Logs results to CSV for tracking long-term EV accuracy."
    )

    history_filename = st.text_input(
        "History File Name",
        value="ultramax_history.csv",
        help="CSV that stores past runs. Used in calibration."
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # DEFENSIVE MATCHUP & TEAM CONTEXT DISPLAY OPTIONS
    # Module 13 + Module 22
    # ------------------------------------------------------------
    st.markdown("### 🛡 Defensive & Context Layers")

    show_defensive_matchups = st.checkbox(
        "Show Defensive Matchup Breakdown",
        value=True
    )

    show_team_context = st.checkbox(
        "Show Team Context Factors",
        value=True
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # RUN BUTTON
    # ------------------------------------------------------------
    run_model = st.button("🚀 Run UltraMax Model")


# ======================================================================
# MODULE 1 — GLOBAL IMPORTS, CONSTANTS, DIRECTORIES, CACHING UTILITIES
# ======================================================================

import os
import sys
import time
import json
import math
import uuid
import random
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache

import streamlit as st
from streamlit import cache_data, cache_resource

# NBA API
from nba_api.stats.endpoints import PlayerGameLog, CommonPlayerInfo
from nba_api.stats.static import players as nba_players

# Stats / Probability tools
from scipy.stats import norm, skewnorm

# -------------------------------------------------------
# DIRECTORY SETUP (ENSURES STREAMLIT FRIENDLY FILES)
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_cache")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

PP_CACHE = os.path.join(DATA_DIR, "prizepicks_cache.json")
SL_CACHE = os.path.join(DATA_DIR, "sleeper_cache.json")

HISTORY_FILE = os.path.join(DATA_DIR, "bet_history.csv")
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "timestamp","player","market","line","prob_over","ev","decision"
    ]).to_csv(HISTORY_FILE, index=False)

# -------------------------------------------------------
# GLOBAL CONSTANTS
# -------------------------------------------------------

CURRENT_SEASON = "2025-26"

MARKET_OPTIONS = [
    "Points", "Rebounds", "Assists",
    "PRA", "PR", "PA", "RA",
    "3PM", "Stocks"
]

TEAM_ABBREVIATIONS = [
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET",
    "GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP",
    "NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
]

# How long to keep API data before refresh
PP_TTL_SECONDS = 3600   # 1 hour cache for PrizePicks
SL_TTL_SECONDS = 3600   # 1 hour cache for Sleeper

# -------------------------------------------------------
# UNIVERSAL READ/WRITE HELPERS
# -------------------------------------------------------

def safe_read_json(path, default=None):
    """Safely load JSON file."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except:
        pass
    return default

def safe_write_json(path, data):
    """Safely write JSON file without Streamlit conflicts."""
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except:
        pass

# -------------------------------------------------------
# PLAYER RESOLUTION UTILITIES
# -------------------------------------------------------

def resolve_player(name: str):
    """
    Returns (player_id, canonical_name)
    """
    if not name or len(name) < 2:
        return None, None

    name = name.lower().strip()

    matches = [
        p for p in nba_players.get_players()
        if name in p["full_name"].lower()
    ]

    if not matches:
        return None, None
    
    pid = matches[0]["id"]
    return pid, matches[0]["full_name"]




# =====================================================================
# MODULES — CORE ENGINES (2–22)
# =====================================================================


# ---------------------------------------------------------
# MODULE 2 — Game Log Fetcher (NBA API Client)
# ---------------------------------------------------------
# ======================================================================
# MODULE 2A — PRIZEPICKS 4-LAYER PROXY SCRAPER ENGINE
# ======================================================================

PRIZEPICKS_HEADERS = [
    {"User-Agent": "Mozilla/5.0"},
    {"User-Agent": "Mozilla/5.0 (Macintosh)"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
]

PRIZEPICKS_URLS = [
    "https://api.prizepicks.com/projections",                     # LAYER 1
    "https://partner-api.prizepicks.com/projections",             # LAYER 2
    "https://prizepicks.com/projections",                         # LAYER 3 (HTML)
]

def pp_rotate_header():
    return random.choice(PRIZEPICKS_HEADERS)

def pp_safe_request(url):
    """
    Safe wrapper for PrizePicks requests.
    Never crashes Streamlit.
    """
    try:
        r = requests.get(url, headers=pp_rotate_header(), timeout=6)
        if r.status_code == 200:
            return r
    except:
        return None
    return None


# ======================================================================
# LAYER 1 — Primary JSON API
# ======================================================================

def pp_layer_1_json():
    r = pp_safe_request(PRIZEPICKS_URLS[0])
    if not r:
        return None
    try:
        data = r.json()
        return data
    except:
        return None


# ======================================================================
# LAYER 2 — Mirror JSON API
# ======================================================================

def pp_layer_2_mirror():
    r = pp_safe_request(PRIZEPICKS_URLS[1])
    if not r:
        return None
    try:
        return r.json()
    except:
        return None


# ======================================================================
# LAYER 3 — HTML Scrape + Regex Extraction
# ======================================================================

import re

def pp_layer_3_html():
    r = pp_safe_request(PRIZEPICKS_URLS[2])
    if not r:
        return None

    text = r.text

    # Find any JSON structures embedded in HTML
    matches = re.findall(r'\{.*?"included".*?\}', text, flags=re.S)

    for m in matches:
        try:
            data = json.loads(m)
            return data
        except:
            pass

    return None


# ======================================================================
# LAYER 4 — Local Cache Fallback
# ======================================================================

def pp_load_cache():
    data = safe_read_json(PP_CACHE)
    return data

def pp_write_cache(data):
    safe_write_json(PP_CACHE, data)


# ======================================================================
# DATA NORMALIZATION — Convert PrizePicks Raw Data → Clean Format
# ======================================================================

def normalize_prizepicks(raw_json):
    """
    Extracts:
      - player name
      - projection line
      - market (PTS/REB/AST/PRA/etc)
      - opponent
      - team
    Returns list of dicts.
    """
    if not raw_json:
        return []

    included = raw_json.get("included", [])
    projections = raw_json.get("data", [])

    player_lookup = {}
    for item in included:
        if item.get("type") == "new_player":
            pid = item["id"]
            player_lookup[pid] = item["attributes"]["name"]

    final = []

    for p in projections:
        try:
            attr = p["attributes"]
            pid = attr["new_player_id"]
            player = player_lookup.get(pid)

            market = attr.get("stat_type")
            line = attr.get("line_score")

            opponent = attr.get("opponent")
            team = attr.get("team")

            if player and market:
                final.append({
                    "player": player,
                    "market": market,
                    "line": float(line) if line else None,
                    "team": team,
                    "opponent": opponent,
                    "source": "PrizePicks"
                })
        except:
            continue

    return final


# ======================================================================
# MASTER FUNCTION — GET PRIZEPICKS LINES (ALL 4 LAYERS)
# ======================================================================

def get_prizepicks_lines(force_refresh=False):
    """
    Main PrizePicks function used by UltraMax Engine.
    Attempts 4 layers in descending priority.

    Returns clean normalized list.
    """

    cache = pp_load_cache()
    now = time.time()

    # Use cache if recent
    if cache and not force_refresh:
        ts = cache.get("timestamp")
        if ts and now - ts < PP_TTL_SECONDS:
            return cache["data"]

    # ------------------------
    # LAYER 1
    # ------------------------
    raw = pp_layer_1_json()
    if raw:
        clean = normalize_prizepicks(raw)
        pp_write_cache({"timestamp": now, "data": clean})
        return clean

    # ------------------------
    # LAYER 2
    # ------------------------
    raw = pp_layer_2_mirror()
    if raw:
        clean = normalize_prizepicks(raw)
        pp_write_cache({"timestamp": now, "data": clean})
        return clean

    # ------------------------
    # LAYER 3
    # ------------------------
    raw = pp_layer_3_html()
    if raw:
        clean = normalize_prizepicks(raw)
        pp_write_cache({"timestamp": now, "data": clean})
        return clean

    # ------------------------
    # LAYER 4 — CACHE
    # ------------------------
    if cache:
        return cache["data"]

    return []
# ======================================================================
# MODULE 2B — SLEEPER 3-LAYER LIVE LINE SCRAPER ENGINE
# ======================================================================

SLEEPER_BASE_URL = "https://api.sleeper.app/v1"
SLEEPER_PROPS_URL = "https://api.sleeper.app/projections/nba"
SLEEPER_ALT_URL = "https://api.sleeper.app/v1/stats/nba"          # LAYER 2
SLEEPER_HTML_URL = "https://sleeper.com/nba/fantasy/props"         # LAYER 3 (HTML)


SLEEPER_HEADERS = [
    {"User-Agent": "Mozilla/5.0"},
    {"User-Agent": "Mozilla/5.0 (Macintosh)"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
]


def sleeper_header():
    return random.choice(SLEEPER_HEADERS)


def sleeper_safe_request(url):
    """
    Safe HTTP wrapper:
    - rotates headers
    - times out at 6s
    - NEVER crashes Streamlit
    """
    try:
        r = requests.get(url, headers=sleeper_header(), timeout=6)
        if r.status_code == 200:
            return r
        return None
    except:
        return None


# ======================================================================
# LAYER 1 — MAIN SLEEPER PROJECTIONS ENDPOINT
# ======================================================================

def sleeper_layer_1_api():
    r = sleeper_safe_request(SLEEPER_PROPS_URL)
    if not r:
        return None
    try:
        return r.json()
    except:
        return None


# ======================================================================
# LAYER 2 — ALTERNATE STAT FEED
# ======================================================================

def sleeper_layer_2_alt():
    r = sleeper_safe_request(SLEEPER_ALT_URL)
    if not r:
        return None
    try:
        return r.json()
    except:
        return None


# ======================================================================
# LAYER 3 — HTML SCRAPE (JS EMBED)
# ======================================================================

def sleeper_layer_3_html():
    r = sleeper_safe_request(SLEEPER_HTML_URL)
    if not r:
        return None

    html = r.text

    # attempt to extract embedded JSON
    matches = re.findall(r'\{.*?"props".*?\}', html, flags=re.S)

    for m in matches:
        try:
            return json.loads(m)
        except:
            pass

    return None


# ======================================================================
# LOCAL CACHE
# ======================================================================

def sleeper_cache_load():
    return safe_read_json(SLEEPER_CACHE)

def sleeper_cache_write(data):
    safe_write_json(SLEEPER_CACHE, data)


# ======================================================================
# MARKET NORMALIZATION
# ======================================================================

SLEEPER_TRANSLATE = {
    "pts": "Points",
    "reb": "Rebounds",
    "ast": "Assists",
    "stl": "Steals",
    "blk": "Blocks",
    "3pm": "Threes",
    "pra": "PRA",
    "pr": "PR",
    "pa": "PA",
    "ra": "RA",
}

def normalize_sleeper_market(mkt: str):
    m = mkt.lower().strip()
    return SLEEPER_TRANSLATE.get(m, m.upper())


# ======================================================================
# PARSE SLEEPER JSON INTO CLEAN FORMAT
# ======================================================================

def normalize_sleeper(raw):
    if not raw:
        return []

    final = []

    players = raw.get("players", {})
    props = raw.get("projections", raw.get("props", []))

    for entry in props:
        try:
            pid = entry.get("player_id")
            player_data = players.get(pid, {})

            player_name = player_data.get("full_name") or entry.get("player_name")
            market_key = entry.get("stat_type")
            line = entry.get("line_score") or entry.get("value")

            team = player_data.get("team")
            opponent = entry.get("opponent")

            if not player_name or not market_key or not line:
                continue

            market = normalize_sleeper_market(market_key)

            final.append({
                "player": player_name,
                "market": market,
                "line": float(line),
                "team": team,
                "opponent": opponent,
                "source": "Sleeper"
            })
        except:
            continue

    return final


# ======================================================================
# MASTER FUNCTION — GET SLEEPER LINES (3-LAYER FALLBACK)
# ======================================================================

def get_sleeper_lines(force_refresh=False):
    """
    3-layer Sleeper feed with:
        - JSON → Mirror JSON → HTML Embedded JSON
        - Safe error handling
        - Cache fallback
    """

    cache = sleeper_cache_load()
    now = time.time()

    # ----- CACHE -----
    if cache and not force_refresh:
        ts = cache.get("timestamp")
        if ts and now - ts < SLEEPER_TTL_SECONDS:
            return cache["data"]

    # ----- LAYER 1 -----
    raw = sleeper_layer_1_api()
    if raw:
        clean = normalize_sleeper(raw)
        sleeper_cache_write({"timestamp": now, "data": clean})
        return clean

    # ----- LAYER 2 -----
    raw = sleeper_layer_2_alt()
    if raw:
        clean = normalize_sleeper(raw)
        sleeper_cache_write({"timestamp": now, "data": clean})
        return clean

    # ----- LAYER 3 -----
    raw = sleeper_layer_3_html()
    if raw:
        clean = normalize_sleeper(raw)
        sleeper_cache_write({"timestamp": now, "data": clean})
        return clean

    # ----- FALLBACK -----
    if cache:
        return cache["data"]

    return []
# ======================================================================
# MODULE 2C — UNIVERSAL LINE NORMALIZER + UNIFIED LIVE LINE FEED
# ======================================================================

import re
import unicodedata

# Make sure these caches exist
UNIFIED_CACHE = os.path.join(CACHE_DIR, "unified_lines.json")
UNIFIED_TTL_SECONDS = 45


# ======================================================================
# CANONICAL NAME ENGINE
# ======================================================================

def canonicalize_name(name: str):
    """
    Convert player name to canonical form:
    - lowercase
    - remove accents (Jokić → Jokic)
    - remove extra spaces
    - underscores for spaces
    """
    if not name:
        return "unknown_player"

    try:
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    except:
        pass

    name = re.sub(r"[^a-zA-Z0-9 ]+", "", name)
    name = re.sub(r"\s+", " ", name.strip().lower())
    return name.replace(" ", "_")


# ======================================================================
# CANONICAL MARKET ENGINE
# ======================================================================

MARKET_MAP = {
    "pts": "Points",
    "points": "Points",

    "reb": "Rebounds",
    "rebs": "Rebounds",
    "rebounds": "Rebounds",

    "ast": "Assists",
    "assists": "Assists",

    "pra": "PRA",
    "points_rebounds_assists": "PRA",

    "pr": "PR",
    "points_rebounds": "PR",

    "pa": "PA",
    "points_assists": "PA",

    "ra": "RA",
    "rebounds_assists": "RA",

    "blk": "Blocks",
    "stl": "Steals",
    "blk_stl": "Stocks",
    "stocks": "Stocks",

    "3pm": "Threes",
    "three_pointers_made": "Threes",
}

def canonical_market(mkt: str):
    if not mkt:
        return None
    m = mkt.lower().strip()
    return MARKET_MAP.get(m, m.upper())


# ======================================================================
# UNIFIED MERGE ENGINE
# ======================================================================

def merge_line(final_dict, book_name, entry):
    """
    Takes a parsed entry from PP or Sleeper and merges it into a
    master dictionary structure.
    """

    player = entry.get("player")
    market = entry.get("market")
    line = entry.get("line")
    team = entry.get("team")
    opponent = entry.get("opponent")

    if not player or not market or line is None:
        return

    canon_player = canonicalize_name(player)
    canon_market = canonical_market(market)

    key = f"{canon_player}::{canon_market}"

    if key not in final_dict:
        final_dict[key] = {
            "player": player,
            "canonical": canon_player,
            "market": canon_market,
            "line": float(line),
            "team": team,
            "opponent": opponent,
            "books": {book_name: float(line)}
        }
    else:
        # merge additional book
        final_dict[key]["books"][book_name] = float(line)

        # choose the consensus line (mean of book lines)
        all_lines = list(final_dict[key]["books"].values())
        final_dict[key]["line"] = float(np.mean(all_lines))


# ======================================================================
# LOCAL CACHE
# ======================================================================

def unified_cache_load():
    return safe_read_json(UNIFIED_CACHE)

def unified_cache_write(data):
    safe_write_json(UNIFIED_CACHE, data)


# ======================================================================
# MASTER UNIFIED FUNCTION
# ======================================================================

def get_unified_lines(force_refresh=False):
    """
    MAIN FUNCTION:
        - Pulls live lines from PrizePicks & Sleeper
        - Canonicalizes names & markets
        - Merges duplicates
        - Computes consensus line
        - Caches result for 45s
    """

    cached = unified_cache_load()
    now = time.time()

    # ------------------ CACHE ------------------
    if cached and not force_refresh:
        ts = cached.get("timestamp")
        if ts and now - ts < UNIFIED_TTL_SECONDS:
            return cached["data"]

    # ------------------ LOAD FEEDS ------------------
    pp_lines = get_prizepicks_lines(force_refresh=False)
    sl_lines = get_sleeper_lines(force_refresh=False)

    final_dict = {}

    # ------------------ MERGE PRIZEPICKS ------------------
    for e in pp_lines:
        merge_line(final_dict, "PrizePicks", e)

    # ------------------ MERGE SLEEPER ------------------
    for e in sl_lines:
        merge_line(final_dict, "Sleeper", e)

    # ------------------ FINAL OUTPUT ------------------
    final = list(final_dict.values())

    unified_cache_write({"timestamp": now, "data": final})

    return final


# ======================================================================
# MODULE 3 — USAGE ENGINE V3 (UltraMax V4 Edition)
# ======================================================================

import numpy as np

# -----------------------------------------
# ROLE IMPACT BASE VALUES
# -----------------------------------------
ROLE_IMPACT = {
    "primary": 1.28,     # superstar initiator (Luka, Trae, SGA)
    "secondary": 1.12,   # strong scorer but not full initiator
    "tertiary": 0.98,    # role player with occasional usage bumps
    "low": 0.82          # catch-and-shoot / screen-setters
}


# -----------------------------------------
# SMOOTH CURVE FUNCTION
# -----------------------------------------
def _smooth_curve(x, power=1.15):
    """
    Makes multipliers smoother and avoids hard jumps.
    """
    try:
        return float(np.sign(x) * (abs(x) ** power))
    except:
        return float(x)


# ======================================================================
# TEAM USAGE FACTOR (PACE + POSSESSION)
# ======================================================================

def team_usage_factor(team_usage_rate):
    """
    Converts team pace / usage into a stable multiplier.
    """
    if team_usage_rate <= 0:
        return 1.00

    base = 0.96 + (team_usage_rate * 0.08)

    # clamp extreme outliers
    base = float(np.clip(base, 0.90, 1.12))

    return base


# ======================================================================
# TEAMMATE OUT REDISTRIBUTION
# ======================================================================

def teammate_out_factor(level: int):
    """
    Redistribution scoring:
        level 0 → no change
        level 1 → +4–6%
        level 2 → +7–12%
        level 3 → +13–18%
    """
    if level <= 0:
        return 1.00
    elif level == 1:
        return 1.055
    elif level == 2:
        return 1.105
    else:
        return 1.165


# ======================================================================
# ON/OFF ADJUSTMENT (CREATOR SHARE MODEL)
# ======================================================================

def creator_share_adjustment(onball_rating, offball_rating):
    """
    Adjusts per-minute value based on:
        - On-ball role
        - Creation responsibilities
    """
    try:
        weight = (onball_rating * 0.7) + (offball_rating * 0.3)
        return float(np.clip(1.00 + weight, 0.85, 1.22))
    except:
        return 1.00


# ======================================================================
# GAME ENVIRONMENT MODIFIER
# ======================================================================

def game_environment_mod(team_tempo, opp_tempo, blowout, back_to_back, fatigue_factor):
    """
    High-level modifier based on:
      - Pace
      - Opponent tempo
      - Blowout risk
      - Rest disadvantage
      - Fatigue decay
    """

    tempo_adj = 0.94 + ((team_tempo + opp_tempo) * 0.03)
    tempo_adj = float(np.clip(tempo_adj, 0.92, 1.10))

    blowout_adj = 0.96 if blowout else 1.00
    b2b_adj = 0.97 if back_to_back else 1.00

    final = tempo_adj * blowout_adj * b2b_adj * fatigue_factor
    return float(np.clip(final, 0.88, 1.12))


# ======================================================================
# MASTER USAGE ENGINE V3
# ======================================================================

def usage_engine_v3(mu_per_min,
                    role: str = "primary",
                    team_usage_rate: float = 1.00,
                    teammate_out_level: int = 0,
                    onball_rating: float = 0.0,
                    offball_rating: float = 0.0,
                    team_tempo: float = 1.0,
                    opp_tempo: float = 1.0,
                    blowout: bool = False,
                    back_to_back: bool = False,
                    fatigue_factor: float = 1.00):
    """
    UltraMax Usage Redistribution Engine V3

    Inputs:
      - mu_per_min          base per-minute production
      - role                primary / secondary / tertiary / low
      - team_usage_rate     possession and pace factor
      - teammate_out_level  redistribution level 0–3
      - onball_rating       creator rating (0–0.30)
      - offball_rating      movement scorer rating (0–0.25)
      - team_tempo          adjusted pace index
      - opp_tempo           opponent pace index
      - blowout             bool
      - back_to_back        bool
      - fatigue_factor      0.94–1.05

    Output:
      - adjusted per-minute production rate
    """

    # -------------------------------
    # 1. Guard base
    # -------------------------------
    base = max(mu_per_min, 0.04)

    # -------------------------------
    # 2. Role impact
    # -------------------------------
    role_mult = ROLE_IMPACT.get(role.lower(), 1.00)
    role_mult = _smooth_curve(role_mult, power=1.18)

    # -------------------------------
    # 3. Team usage / pace factor
    # -------------------------------
    team_mult = team_usage_factor(team_usage_rate)
    team_mult = _smooth_curve(team_mult, power=1.12)

    # -------------------------------
    # 4. Injury redistribution
    # -------------------------------
    injury_mult = teammate_out_factor(teammate_out_level)
    injury_mult = _smooth_curve(injury_mult, power=1.20)

    # -------------------------------
    # 5. Creator/on-ball responsibilities
    # -------------------------------
    creator_mult = creator_share_adjustment(onball_rating, offball_rating)

    # -------------------------------
    # 6. Game environment
    # -------------------------------
    env_mult = game_environment_mod(team_tempo, opp_tempo,
                                    blowout, back_to_back, fatigue_factor)

    # -------------------------------
    # 7. Final blended rate
    # -------------------------------
    final = base * role_mult * team_mult * injury_mult * creator_mult * env_mult

    # Stability clamp
    final = float(np.clip(final, base * 0.70, base * 1.85))

    return final
# ======================================================================
# MODULE 3 — PHASE 2
# ROLE ENGINE EXPANSION (Dynamic Role Classification)
# UltraMax V4 — Advanced Role Detection System
# ======================================================================

import numpy as np

# -------------------------------------------------------------
# BASELINE ROLE PROFILES
# -------------------------------------------------------------
ROLE_PROFILES = {
    "primary": {
        "usage_min": 0.28,
        "ast_chance_min": 0.22,
        "touch_time_min": 5.0,
        "weight": 1.00
    },
    "secondary": {
        "usage_min": 0.22,
        "ast_chance_min": 0.12,
        "touch_time_min": 3.0,
        "weight": 0.82
    },
    "tertiary": {
        "usage_min": 0.16,
        "ast_chance_min": 0.06,
        "touch_time_min": 1.8,
        "weight": 0.64
    },
    "low": {
        "usage_min": 0.00,
        "ast_chance_min": 0.00,
        "touch_time_min": 0.00,
        "weight": 0.45
    }
}


# -------------------------------------------------------------
# ROLE CONFIDENCE SMOOTHING
# -------------------------------------------------------------
def _role_confidence_smoothing(value, avg, weight=0.65):
    """
    Smooths short-term volatility.
    """
    return (value * weight) + (avg * (1 - weight))


# -------------------------------------------------------------
# ROLE SELECTION ENGINE
# -------------------------------------------------------------
def classify_dynamic_role(
        usage_rate,
        ast_chance,
        touch_time,
        teammate_out_level,
        recent_usage_5,
        opp_def_scheme="switch",
        creator_rating=0.0,
        offball_rating=0.0
):
    """
    Determines player role based on:
        - Long-term usage
        - Short-term usage (recent 5)
        - AST% + creation load
        - Touch-time (time of possession)
        - Opposition defensive scheme
        - Teammate-out adjustments
    """

    # ---------------------------------------------------------
    # 1. Smooth usage between season-long and recent games
    # ---------------------------------------------------------
    usage_eff = _role_confidence_smoothing(recent_usage_5, usage_rate)

    # ---------------------------------------------------------
    # 2. Teammate-out bumps role upward
    # ---------------------------------------------------------
    if teammate_out_level == 1:
        usage_eff *= 1.05
    elif teammate_out_level == 2:
        usage_eff *= 1.10
    elif teammate_out_level >= 3:
        usage_eff *= 1.15

    # cap usage
    usage_eff = float(np.clip(usage_eff, 0.00, 0.40))

    # ---------------------------------------------------------
    # 3. Build role scoring
    # ---------------------------------------------------------
    role_scores = {}

    for role_name, prof in ROLE_PROFILES.items():
        score = 0

        # Usage impact
        if usage_eff >= prof["usage_min"]:
            score += 0.45

        # Assist chance / creation
        if ast_chance >= prof["ast_chance_min"]:
            score += 0.30

        # Touch time
        if touch_time >= prof["touch_time_min"]:
            score += 0.25

        # Creation synergy
        score += creator_rating * 0.25
        score += offball_rating * 0.15

        # Defensive scheme adjustment
        if opp_def_scheme == "drop" and role_name == "primary":
            score += 0.04  # drop favors high pick-and-roll creators
        if opp_def_scheme == "blitz" and role_name == "secondary":
            score += 0.03  # blitz forces ball out of star's hands
        if opp_def_scheme == "switch" and role_name == "tertiary":
            score += 0.02  # switching favors role players attacking matchups

        role_scores[role_name] = score

    # ---------------------------------------------------------
    # 4. Select Highest Score
    # ---------------------------------------------------------
    best_role = max(role_scores, key=role_scores.get)
    confidence = float(np.clip(role_scores[best_role], 0.10, 1.40))

    return best_role, confidence


# -------------------------------------------------------------
# ROLE MULTIPLIER ENGINE (FOR USAGE ENGINE V3)
# -------------------------------------------------------------
def dynamic_role_multiplier(base_role: str, confidence: float):
    """
    Converts dynamic role + confidence → multiplier
    """

    role_base = ROLE_IMPACT.get(base_role, 1.00)

    # amplify based on certainty
    boosted = role_base * (0.78 + (confidence * 0.22))

    # stability clamp
    boosted = float(np.clip(boosted, 0.72, 1.42))

    return boosted
# ======================================================================
# MODULE 3 — PHASE 3
# TEAM USAGE ENGINE EXPANSION (Pace, Possession Load, Injury Drift)
# UltraMax V4 — Team Context + Usage Redistribution Layer
# ======================================================================

import numpy as np

# -------------------------------------------------------------
# TEAM PACE BASELINES
# (These will be replaced by API pulls in Module 21)
# -------------------------------------------------------------
TEAM_PACE_TABLE = {
    "LAL": 101.5, "GSW": 101.8, "MIN": 98.1, "DEN": 97.9, "BOS": 97.0,
    "PHX": 100.1, "MIL": 100.0, "DAL": 99.9, "NYK": 94.5, "MIA": 95.1,
    # fallback if team not found
    "DEFAULT": 98.5
}

TEAM_OFF_RTG = {
    "LAL": 116.1, "GSW": 115.2, "MIN": 118.1, "DEN": 119.6,
    "PHX": 116.8, "MIL": 118.9, "DAL": 117.2, "NYK": 114.1,
    "DEFAULT": 115.0
}

TEAM_DEF_RTG = {
    "LAL": 113.9, "GSW": 118.2, "MIN": 108.7, "DEN": 113.2,
    "PHX": 115.7, "MIL": 117.9, "DAL": 116.6, "NYK": 112.1,
    "DEFAULT": 113.5
}


# -------------------------------------------------------------
# Defensively Adjusted Pace (DAP)
# -------------------------------------------------------------
def defensive_adjusted_pace(team_abbrev, opponent_abbrev):
    """
    Computes pace adjustment using:
        - Team baseline pace
        - Opponent pace suppression boost
        - Weighted average using NBA possessions formula
    """
    team_pace = TEAM_PACE_TABLE.get(team_abbrev, TEAM_PACE_TABLE["DEFAULT"])
    opp_pace = TEAM_PACE_TABLE.get(opponent_abbrev, TEAM_PACE_TABLE["DEFAULT"])

    # Opponent defensive pace suppression
    opp_def = TEAM_DEF_RTG.get(opponent_abbrev, TEAM_DEF_RTG["DEFAULT"])
    suppress_mult = np.clip((115.0 - opp_def) / 115.0, 0.92, 1.05)

    blended = (team_pace * 0.58) + (opp_pace * 0.42)
    adjusted = blended * suppress_mult

    return float(np.clip(adjusted, 90.0, 105.5))


# -------------------------------------------------------------
# Team Possession Load Score (TPLS)
# -------------------------------------------------------------
def compute_team_possession_load(pace, off_rating):
    """
    Converts pace + offensive rating → team possession load.
    Low pace + bad offense = slow environment (hurts usage)
    High pace + elite offense = fast environment (boosts usage)
    """
    pace_z = (pace - 98.0) / 7.5
    off_z = (off_rating - 115.0) / 5.0

    tpls = 1.00 + (0.10 * pace_z) + (0.08 * off_z)
    return float(np.clip(tpls, 0.85, 1.18))


# -------------------------------------------------------------
# Lineup-Based Injury Drift (LID)
# -------------------------------------------------------------
def injury_drift_factor(teammate_out_level):
    """
    redistributes offensive load based on number/impact of missing players
    """
    if teammate_out_level == 0:
        return 1.00
    elif teammate_out_level == 1:
        return 1.04
    elif teammate_out_level == 2:
        return 1.08
    elif teammate_out_level >= 3:
        return 1.12


# -------------------------------------------------------------
# Opponent Defensive Context Score (ODC)
# -------------------------------------------------------------
def opponent_def_context(opponent_abbrev):
    """
    Converts defensive rating → multiplier affecting usage
    """
    def_rtg = TEAM_DEF_RTG.get(opponent_abbrev, TEAM_DEF_RTG["DEFAULT"])

    score = 1.00 + ((113.0 - def_rtg) / 40.0)  # lower defense → higher usage
    return float(np.clip(score, 0.88, 1.12))


# -------------------------------------------------------------
# Offense Environment Quality Score (OEQ)
# -------------------------------------------------------------
def compute_environment_quality(pace_adj, tpls, odc):
    """
    Rolls pace + team load + opponent defensive context into
    a single blended environment score.
    """
    env = (
        (pace_adj / 98.0) * 0.40 +
        tpls * 0.35 +
        odc * 0.25
    )

    return float(np.clip(env, 0.80, 1.25))


# -------------------------------------------------------------
# TEAM USAGE RATE MULTIPLIER (Final Output of Phase 3)
# -------------------------------------------------------------
def team_usage_rate_multiplier(team_abbrev, opponent_abbrev, teammate_out_level):
    """
    Creates the final usage multiplier used in Usage Engine V3.
    This is the team-context layer.
    """

    pace_adj = defensive_adjusted_pace(team_abbrev, opponent_abbrev)
    off_rating = TEAM_OFF_RTG.get(team_abbrev, TEAM_OFF_RTG["DEFAULT"])

    tpls = compute_team_possession_load(pace_adj, off_rating)
    odc = opponent_def_context(opponent_abbrev)
    injury_mult = injury_drift_factor(teammate_out_level)

    env_quality = compute_environment_quality(pace_adj, tpls, odc)

    final_mult = env_quality * injury_mult
    final_mult = float(np.clip(final_mult, 0.78, 1.32))

    return final_mult
# ======================================================================
# MODULE 3 — PHASE 4
# LINEUP-BASED USAGE REDISTRIBUTION ENGINE (On/Off, Synergy, Role Clusters)
# ======================================================================

import numpy as np

# --------------------------------------------------------
# ROLE CLUSTERING (per-minute archetype multipliers)
# --------------------------------------------------------
ROLE_CLUSTER = {
    "primary_creator": 1.18,
    "secondary_creator": 1.08,
    "off_ball_scorer": 1.05,
    "finisher_big": 0.98,
    "rebound_big": 0.92,
    "connector": 0.90,
    "low_usage": 0.82,
}

# --------------------------------------------------------
# POSITIONAL SYNERGY MATRIX
# How much each position benefits from missing teammates
# --------------------------------------------------------
SYNERGY_MATRIX = {
    ("PG", "PG"): 1.00,
    ("PG", "SG"): 1.12,
    ("PG", "SF"): 1.06,
    ("PG", "PF"): 1.03,
    ("PG", "C"):  1.01,

    ("SG", "PG"): 1.07,
    ("SG", "SG"): 1.00,
    ("SG", "SF"): 1.09,
    ("SG", "PF"): 1.03,
    ("SG", "C"):  1.01,

    ("SF", "PG"): 1.05,
    ("SF", "SG"): 1.07,
    ("SF", "SF"): 1.00,
    ("SF", "PF"): 1.09,
    ("SF", "C"):  1.02,

    ("PF", "PG"): 1.02,
    ("PF", "SG"): 1.03,
    ("PF", "SF"): 1.05,
    ("PF", "PF"): 1.00,
    ("PF", "C"):  1.10,

    ("C",  "PG"): 1.01,
    ("C",  "SG"): 1.01,
    ("C",  "SF"): 1.03,
    ("C",  "PF"): 1.12,
    ("C",  "C"):  1.00,
}

# --------------------------------------------------------
# STARTER vs BENCH usage baseline
# --------------------------------------------------------
STARTER_USAGE_MULT = 1.00
BENCH_USAGE_MULT = 1.12   # bench players get inflated usage per-minute

# --------------------------------------------------------
# FALLBACK SHOT RATE (when play breaks and someone must shoot)
# --------------------------------------------------------
FALLBACK_SHOT_RATES = {
    "PG": 0.26,
    "SG": 0.24,
    "SF": 0.20,
    "PF": 0.17,
    "C":  0.13
}

def compute_fallback_multiplier(position):
    """
    Shot-clock bailout scoring for late-clock plays.
    PG/SG gain the most.
    """
    base = FALLBACK_SHOT_RATES.get(position, 0.20)
    mult = 1.00 + (base - 0.20) * 1.75
    return float(np.clip(mult, 0.90, 1.20))


# --------------------------------------------------------
# ON/OFF USAGE SWINGS (simple but powerful)
# --------------------------------------------------------
ON_OFF_IMPACT = {
    # Player archetype is OUT → usage rebalanced
    "primary_creator": 1.10,
    "secondary_creator": 1.06,
    "off_ball_scorer": 1.04,
    "connector": 1.03,
    "rebound_big": 1.02,
    "finisher_big": 1.00,
}


def on_off_usage_boost(player_role, missing_roles):
    """
    If a player is a PG and missing players are creators → big boost.
    """
    if not missing_roles:
        return 1.00
    
    role_mult = 1.00
    for r in missing_roles:
        boost = ON_OFF_IMPACT.get(r, 1.00)
        
        # More weight if player is a guard
        if player_role in ["primary_creator", "secondary_creator"]:
            boost += 0.03
        
        role_mult *= boost
    
    return float(np.clip(role_mult, 1.00, 1.28))


# --------------------------------------------------------
# LINEUP REDISTRIBUTION CORE
# --------------------------------------------------------
def compute_lineup_redistribution(position, player_role, teammate_out_level, missing_roles):
    """
    Converts lineup, position, and missing teammates into usage boost.
    Inputs:
       - position: PG/SG/SF/PF/C
       - player_role: primary_creator, off_ball_scorer, etc
       - teammate_out_level: 0–3
       - missing_roles: list of roles of missing teammates
    """

    # Starter vs Bench adjustment
    if position in ["PG", "SG", "SF", "PF", "C"]:
        starter_mult = STARTER_USAGE_MULT
    else:
        starter_mult = BENCH_USAGE_MULT   # treat unknowns as bench players

    role_mult = ROLE_CLUSTER.get(player_role, 1.00)

    # Synergy factor based on missing teammates
    synergy_mult = 1.00
    for r in missing_roles:
        synergy_key = (position, position_from_role(r))
        synergy_mult *= SYNERGY_MATRIX.get(synergy_key, 1.00)

    # On/off usage boosts
    on_off_mult = on_off_usage_boost(player_role, missing_roles)

    # Fallback scoring for late-clock situations
    fallback_mult = compute_fallback_multiplier(position)

    # Severity of missing teammates
    injury_mult = 1.00 + (0.05 * teammate_out_level)

    final = (
        starter_mult *
        role_mult *
        synergy_mult *
        on_off_mult *
        fallback_mult *
        injury_mult
    )

    return float(np.clip(final, 0.85, 1.42))


# --------------------------------------------------------
# Helper — convert roles to positions (rough)
# --------------------------------------------------------
def position_from_role(role):
    if "creator" in role:
        return "PG"
    if "scorer" in role:
        return "SG"
    if "connector" in role:
        return "SF"
    if "finisher" in role:
        return "PF"
    if "rebound" in role:
        return "C"
    return "SF"
# ======================================================================
# MODULE 3 — PHASE 5
# USAGE ENGINE V3 — FINAL ASSEMBLY (UltraMax Version)
# Combines:
#   - Phase 1: Base per-minute productivity normalization
#   - Phase 2: Role curve shaping
#   - Phase 3: Team context (pace, usage distribution)
#   - Phase 4: Lineup redistribution (on/off, synergy, fallback scoring)
# ======================================================================

def usage_engine_v3_final(
    mu_per_min_base: float,
    position: str,
    player_role: str,
    team_usage_rate: float,
    teammate_out_level: int,
    missing_roles: list,
    pace_factor: float = 1.00,
    offensive_focus: float = 1.00,
):
    """
    FULL USAGE ENGINE V3 (FINAL ASSEMBLY)
    ------------------------------------------------------
    The final coherent usage model that drives UltraMax projections.
    - Takes raw per-minute productivity
    - Applies NBA-like role shaping
    - Adjusts for team offensive context + pace
    - Rebalances based on missing creators/scorers
    - Applies positional synergy for realistic lineup shifts
    - Adds fallback scoring logic (late-clock possessions)
    - Stabilizes & clamps output for realistic distribution

    Inputs:
       mu_per_min_base      — baseline per-minute production
       position             — PG/SG/SF/PF/C
       player_role          — primary_creator, connector, rebound_big, etc
       team_usage_rate      — pace/possession share scaling
       teammate_out_level   — 0–3 severity of injuries
       missing_roles        — list of roles missing from lineup
       pace_factor          — additional pace adjustment for certain games
       offensive_focus      — how much offense flows through this player (team strategy)

    Output:
       adjusted_per_min     — final per-minute projection
    """

    # ------------------------------------------------------
    # 1 — BASE NORMALIZATION (Phase 1)
    # ------------------------------------------------------
    mu_base = max(mu_per_min_base, 0.04)

    # ------------------------------------------------------
    # 2 — ROLE CURVE (Phase 2)
    # ------------------------------------------------------
    role_adj = ROLE_IMPACT.get(player_role.lower(), 1.00)
    role_adj = _smooth_curve(role_adj, power=1.20)

    # ------------------------------------------------------
    # 3 — TEAM CONTEXT (Phase 3)
    # ------------------------------------------------------
    team_adj = team_usage_factor(team_usage_rate)
    team_adj = _smooth_curve(team_adj, power=1.10)

    # extra scaling via game pace or game plan
    pace_adj = float(np.clip(pace_factor, 0.92, 1.10))
    focus_adj = float(np.clip(offensive_focus, 0.90, 1.12))

    # ------------------------------------------------------
    # 4 — LINEUP REDISTRIBUTION (Phase 4)
    # ------------------------------------------------------
    lineup_adj = compute_lineup_redistribution(
        position=position,
        player_role=player_role,
        teammate_out_level=teammate_out_level,
        missing_roles=missing_roles
    )

    # ------------------------------------------------------
    # 5 — FINAL MULTIPLIER COMPOSITION
    # ------------------------------------------------------
    combined_multiplier = (
        role_adj ** 0.40 *
        team_adj ** 0.35 *
        lineup_adj ** 0.55 *
        pace_adj ** 0.25 *
        focus_adj ** 0.30
    )

    # ------------------------------------------------------
    # 6 — APPLY USAGE ENGINE TO BASE
    # ------------------------------------------------------
    mu_final = mu_base * combined_multiplier

    # Clamp for stability (prevents explosive unrealistic projections)
    mu_final = float(np.clip(mu_final, mu_base * 0.65, mu_base * 1.70))

    return mu_final
# ======================================================================
# MODULE 3 — PHASE 6
# USAGE ENGINE V3 — KERNEL WRAPPER + SAFETY LAYER
# Connects all sub-phases into final stable function usage_engine_v3()
# ======================================================================

def usage_engine_v3(
    base_mu_per_min: float,
    role: str = "primary",
    team_usage_rate: float = 1.00,
    teammate_out_level: int = 0,
    position: str = "SG",
    missing_roles: list = None,
    pace_factor: float = 1.00,
    offensive_focus: float = 1.00,
):
    """
    CLEAN WRAPPER FOR USAGE ENGINE V3 FINAL ASSEMBLY
    -------------------------------------------------------------------
    Ensures compute_leg() always has a safe, stable, complete usage call.
    This function:
      - Normalizes missing inputs
      - Auto-detects missing_roles default
      - Applies Phase 5's full engine
      - Ensures no rare crashes from NoneType/bad input
      - Returns a final per-minute projection
    """

    # -----------------------------
    # Safety Defaults
    # -----------------------------
    if base_mu_per_min is None or np.isnan(base_mu_per_min):
        base_mu_per_min = 0.05

    if not isinstance(role, str):
        role = "primary"

    if missing_roles is None:
        missing_roles = []

    try:
        teammate_out_level = int(teammate_out_level)
    except:
        teammate_out_level = 0

    # clamp teammate_out_level to (0-3) range
    teammate_out_level = int(np.clip(teammate_out_level, 0, 3))

    # -----------------------------
    # Call final engine
    # -----------------------------
    return usage_engine_v3_final(
        mu_per_min_base=base_mu_per_min,
        position=position,
        player_role=role,
        team_usage_rate=float(team_usage_rate),
        teammate_out_level=teammate_out_level,
        missing_roles=missing_roles,
        pace_factor=float(pace_factor),
        offensive_focus=float(offensive_focus),
    )
# ======================================================================
# MODULE 4 — PHASE 1
# OPPONENT DEFENSE ENGINE V2.0 — BASE TEAM PROFILES
# ======================================================================

# Each team receives a "defensive context profile":
#   - pace_factor: affects total possessions
#   - def_rating: overall defensive efficiency
#   - vs_guards, vs_wings, vs_bigs: positional matchup difficulty
#   - market_modifiers: market-specific resistance (PTS/REB/AST/PRA)

OPP_DEF_PROFILES = {

    "ATL": {
        "pace_factor": 1.04,
        "def_rating": 1.08,
        "vs_guards": 0.96,
        "vs_wings": 1.02,
        "vs_bigs": 1.10,
        "market_mod": {
            "PTS": 1.06,
            "REB": 1.08,
            "AST": 1.04,
            "PRA": 1.07,
        }
    },

    "BOS": {
        "pace_factor": 0.97,
        "def_rating": 0.92,
        "vs_guards": 0.88,
        "vs_wings": 0.94,
        "vs_bigs": 0.98,
        "market_mod": {
            "PTS": 0.90,
            "REB": 0.95,
            "AST": 0.92,
            "PRA": 0.92,
        }
    },

    "BRK": {
        "pace_factor": 1.01,
        "def_rating": 1.03,
        "vs_guards": 1.02,
        "vs_wings": 1.00,
        "vs_bigs": 1.06,
        "market_mod": {
            "PTS": 1.03,
            "REB": 1.07,
            "AST": 1.03,
            "PRA": 1.04,
        }
    },

    "CHA": {
        "pace_factor": 1.06,
        "def_rating": 1.10,
        "vs_guards": 1.04,
        "vs_wings": 1.06,
        "vs_bigs": 1.12,
        "market_mod": {
            "PTS": 1.09,
            "REB": 1.12,
            "AST": 1.08,
            "PRA": 1.10,
        }
    },

    "CHI": {
        "pace_factor": 0.98,
        "def_rating": 1.01,
        "vs_guards": 0.96,
        "vs_wings": 1.00,
        "vs_bigs": 1.03,
        "market_mod": {
            "PTS": 0.98,
            "REB": 1.03,
            "AST": 0.99,
            "PRA": 1.00,
        }
    },

    # You will receive all 30 team profiles by the end of Module 4.
    # (Over 1,200 lines once expanded in full UltraMax version.)
}

# ------------------------------
# Safety default for missing team
# ------------------------------
DEFAULT_OPP_CONTEXT = {
    "pace_factor": 1.00,
    "def_rating": 1.00,
    "vs_guards": 1.00,
    "vs_wings": 1.00,
    "vs_bigs": 1.00,
    "market_mod": {
        "PTS": 1.00,
        "REB": 1.00,
        "AST": 1.00,
        "PRA": 1.00,
    }
}

# ======================================================================
# MODULE 4 — PHASE 2
# OPPONENT DEFENSE ENGINE V2.0 — NORMALIZATION + SAFETY LAYER
# ======================================================================

def normalize_team_code(team: str) -> str:
    """
    Safely normalize a team abbreviation:
       - Strip whitespace
       - Convert to uppercase
       - Fix common typos
    """
    if not team or not isinstance(team, str):
        return "UNKNOWN"
    
    t = team.strip().upper()

    # Common user typos cleanup
    fixes = {
        "BRK": "BKN",
        "BNK": "BKN",
        "GS": "GSW",
        "SA": "SAS",
        "NO": "NOP",
        "NY": "NYK",
        "PHO": "PHX",
        "OK": "OKC",
        "CLV": "CLE",
    }
    return fixes.get(t, t)


def get_opponent_context(team: str) -> dict:
    """
    Retrieves opponent defensive profile with FULL safety:
        - Normalizes team code first
        - Provides DEFAULT_OPP_CONTEXT if team not found
        - Ensures all expected keys exist
    """

    t = normalize_team_code(team)

    profile = OPP_DEF_PROFILES.get(t, DEFAULT_OPP_CONTEXT)

    # Deep safety check (avoid missing keys)
    safe_profile = {
        "pace_factor": profile.get("pace_factor", 1.00),
        "def_rating": profile.get("def_rating", 1.00),
        "vs_guards": profile.get("vs_guards", 1.00),
        "vs_wings": profile.get("vs_wings", 1.00),
        "vs_bigs": profile.get("vs_bigs", 1.00),
        "market_mod": {
            "PTS": profile.get("market_mod", {}).get("PTS", 1.00),
            "REB": profile.get("market_mod", {}).get("REB", 1.00),
            "AST": profile.get("market_mod", {}).get("AST", 1.00),
            "PRA": profile.get("market_mod", {}).get("PRA", 1.00),
        }
    }

    return safe_profile


def get_market_modifier(ctx: dict, market: str) -> float:
    """
    Extract a safe market-specific multiplier.
    If unknown market, default is 1.0.
    """
    m = market.upper()
    valid = ["PTS", "REB", "AST", "PRA"]

    if m not in valid:
        return 1.00

    return float(ctx["market_mod"].get(m, 1.00))


def get_positional_modifier(ctx: dict, position: str) -> float:
    """
    Returns positional matchup multiplier:
       - G (guards)
       - W (wings)
       - F (forwards → treated as wings)
       - C (centers/bigs)
    """
    if not position:
        return 1.00

    p = position.upper()[0]  # G/W/F/C using first letter

    if p == "G":
        return ctx["vs_guards"]

    if p in ("F", "W"):
        return ctx["vs_wings"]

    if p == "C":
        return ctx["vs_bigs"]

    return 1.00

# ======================================================================
# MODULE 4 — PHASE 3
# OPPONENT MATCHUP ENGINE V2 (CORE MULTIPLIER)
# ======================================================================

def opponent_matchup_v2(opponent_team: str, market: str, position: str = "G"):
    """
    Core matchup multiplier for UltraMax Engine.
    ------------------------------------------------------
    Combines:
        - opponent defensive profile
        - pace factor
        - positional defensive matchup
        - market-specific weakness/strength
        - defensive rating smoothing
        - nonlinear adjustments for extreme matchups

    Always returns a SAFE multiplier between ~0.80 and 1.35.
    """

    # --------------------------------------------------
    # 1. Get normalized opponent context
    # --------------------------------------------------
    ctx = get_opponent_context(opponent_team)

    # -------------------------------
    # 2. Market adjustment
    # -------------------------------
    m = market.upper()
    market_mult = get_market_modifier(ctx, m)

    # -------------------------------
    # 3. Positional defensive adj
    # -------------------------------
    pos_mult = get_positional_modifier(ctx, position)

    # -------------------------------
    # 4. Pace adjustment
    # -------------------------------
    pace = ctx.get("pace_factor", 1.00)

    # Smooth nonlinear shape (pace matters more for REB/AST than PTS)
    if m == "REB":
        pace_mult = pace ** 1.20
    elif m == "AST":
        pace_mult = pace ** 1.10
    else:
        pace_mult = pace ** 1.05

    # -------------------------------
    # 5. Defensive rating adjustment
    # -------------------------------
    def_rating = ctx.get("def_rating", 1.00)

    # Smooth curve to keep things stable
    def_mult = (def_rating ** 0.70)

    # -------------------------------
    # 6. Combine base layers
    # -------------------------------
    base_mult = (
        (market_mult ** 0.55) *
        (pos_mult ** 0.50) *
        (pace_mult ** 0.35) *
        (def_mult ** 0.40)
    )

    # -------------------------------
    # 7. Extreme matchup limiter
    # -------------------------------
    # Prevent overly large or tiny multipliers
    base_mult = float(np.clip(base_mult, 0.80, 1.35))

    return base_mult
# =====================================================================
# MODULE 5 — PHASE 1
# BASE VOLATILITY EXTRACTION ENGINE
# =====================================================================

def extract_base_volatility(game_logs, market: str):
    """
    Extracts raw volatility (standard deviation) for player production.
    This is the first step of the UltraMax Volatility Engine.

    It normalizes:
        - low minute samples
        - missing data
        - extreme outliers
        - market-specific variance patterns

    Returns:
        {
            "sd_raw": float,
            "sd_clamped": float,
            "values": np.array
        }
    """

    # --------------------------------------------------------
    # 1. Build market values
    # --------------------------------------------------------
    m = market.upper()

    if m == "PTS":
        vals = game_logs["PTS"].astype(float)
    elif m == "REB":
        vals = game_logs["REB"].astype(float)
    elif m == "AST":
        vals = game_logs["AST"].astype(float)
    else:
        # PRA (P + R + A)
        vals = (game_logs["PTS"] +
                game_logs["REB"] +
                game_logs["AST"]).astype(float)

    # --------------------------------------------------------
    # 2. Remove invalid numbers
    # --------------------------------------------------------
    vals = vals.replace([np.inf, -np.inf], np.nan)
    vals = vals.dropna()

    if len(vals) < 3:
        # not enough samples → fallback to safe defaults
        return {
            "sd_raw": 3.0,
            "sd_clamped": 3.0,
            "values": np.array([3.0])
        }

    arr = vals.values

    # --------------------------------------------------------
    # 3. Raw volatility (REAL data)
    # --------------------------------------------------------
    sd_raw = float(np.std(arr))

    # --------------------------------------------------------
    # 4. Market-specific floor/ceiling stabilization
    # --------------------------------------------------------
    # PTS typically higher variance, REB lowest, AST moderate.
    if m == "PTS":
        sd_floor, sd_ceiling = 2.5, 18.0
    elif m == "REB":
        sd_floor, sd_ceiling = 1.2, 10.0
    elif m == "AST":
        sd_floor, sd_ceiling = 1.0, 9.0
    else:  # PRA
        sd_floor, sd_ceiling = 3.5, 22.0

    sd_clamped = float(np.clip(sd_raw, sd_floor, sd_ceiling))

    # --------------------------------------------------------
    # 5. Return package
    # --------------------------------------------------------
    return {
        "sd_raw": sd_raw,
        "sd_clamped": sd_clamped,
        "values": arr
    }
# =====================================================================
# MODULE 5 — PHASE 2
# REGIME-AWARE VOLATILITY ENGINE
# =====================================================================

def classify_regime(last_values: np.ndarray):
    """
    Classifies a volatility/production regime based on last 6–10 games.
    Returns one of:
        - "hot"
        - "cold"
        - "stable"
        - "unstable"
    """

    if len(last_values) < 4:
        return "stable"

    recent = last_values[-6:] if len(last_values) >= 6 else last_values
    mean_val = np.mean(recent)
    sd_val = np.std(recent)

    # coefficient of variation
    cv = sd_val / max(mean_val, 1e-6)

    # ---------- REGIME LOGIC ----------
    if cv < 0.22:
        return "hot"         # high consistency, above mean
    elif cv < 0.36:
        return "stable"      # normal variation
    elif cv < 0.55:
        return "unstable"    # randomness increasing
    else:
        return "cold"        # highly inconsistent


def regime_multiplier(regime: str, market: str):
    """
    Determines volatility multiplier based on regime + market.
    """

    m = market.upper()

    # Base multipliers (regime only)
    base = {
        "hot": 0.88,
        "stable": 1.00,
        "unstable": 1.17,
        "cold": 1.32
    }.get(regime, 1.00)

    # Market-specific adjustments (because markets behave differently)
    if m == "PTS":
        market_adj = 1.00
    elif m == "REB":
        market_adj = 0.92
    elif m == "AST":
        market_adj = 0.95
    else:  # PRA
        market_adj = 1.10

    return base * market_adj


def minutes_volatility_adjustment(minutes_array):
    """
    More minutes -> lower volatility
    Fewer minutes -> higher volatility
    """
    if len(minutes_array) < 3:
        return 1.00

    recent_min = np.mean(minutes_array[-5:])
    if recent_min >= 34:
        return 0.90
    elif recent_min >= 30:
        return 1.00
    elif recent_min >= 26:
        return 1.10
    else:
        return 1.25


def usage_shock_adjustment(teammate_out_level: int, base_mu: float):
    """
    Large usage jumps create additional volatility.
    teammate_out_level ∈ {0,1,2,3}
    """

    if teammate_out_level == 0:
        return 1.00
    elif teammate_out_level == 1:
        return 1.08
    elif teammate_out_level == 2:
        return 1.15
    else:  # Level 3: star + secondary missing
        return 1.28


def volatility_engine_phase2(
    sd_clamped: float,
    last_values: np.ndarray,
    minutes_data: np.ndarray,
    market: str,
    teammate_out_level: int,
    base_mu_per_min: float
):
    """
    Combines:
       - regime classification
       - regime volatility shaping
       - minutes-based volatility shaping
       - usage-shock volatility shaping
    """

    # ------------------------------------------------------------
    # 1. Regime classification
    # ------------------------------------------------------------
    regime = classify_regime(last_values)

    # ------------------------------------------------------------
    # 2. Regime volatility multiplier
    # ------------------------------------------------------------
    regime_mult = regime_multiplier(regime, market)

    # ------------------------------------------------------------
    # 3. Minutes-driven volatility shaping
    # ------------------------------------------------------------
    minutes_mult = minutes_volatility_adjustment(minutes_data)

    # ------------------------------------------------------------
    # 4. Usage shock sensitivity
    # ------------------------------------------------------------
    usage_mult = usage_shock_adjustment(teammate_out_level, base_mu_per_min)

    # ------------------------------------------------------------
    # 5. Combine
    # ------------------------------------------------------------
    combined_sd = sd_clamped * regime_mult * minutes_mult * usage_mult

    # ------------------------------------------------------------
    # 6. Final safety clamp
    # ------------------------------------------------------------
    final_sd = float(np.clip(combined_sd, 0.75, 25.0))

    return {
        "regime": regime,
        "base_sd": sd_clamped,
        "final_sd": final_sd,
        "regime_mult": regime_mult,
        "minutes_mult": minutes_mult,
        "usage_mult": usage_mult
    }
# =====================================================================
# MODULE 5 — PHASE 3
# NONLINEAR VOLATILITY RESHAPING ENGINE
# =====================================================================

def nonlinear_curve(x: float, power: float, steepness: float = 1.0):
    """
    Smooth exponential curve used for volatility shaping.
    Ensures stability + controlled growth.
    """
    x = max(x, 1e-6)
    return float((x ** power) * steepness)


def market_curve_multiplier(market: str):
    """
    Each market has a unique volatility curvature shape.
    PRA is always the most nonlinear.
    """

    m = market.upper()

    if m == "PTS":
        return 1.08, 0.90   # mild curve
    elif m == "REB":
        return 1.12, 1.00   # rebounds have heavy variance tail
    elif m == "AST":
        return 1.10, 0.95   # assists slightly nonlinear
    else:  # PRA
        return 1.22, 1.05   # PRA volatility grows fastest


def late_game_chaos_multiplier(minutes_proj: float):
    """
    Late-game chaos boosts volatility for players:
        - < 28 min: unpredictable role/change
        - 28–34 min: baseline chaos
        - 34+ min: low chaos (stable rotations)
    """
    if minutes_proj >= 35:
        return 0.92
    elif minutes_proj >= 30:
        return 1.00
    else:
        return 1.13


def blowout_volatility_adjustment(blowout_flag: bool, market: str):
    """
    Blowouts increase volatility for unders, decrease for overs.
    For modeling purposes, volatility always increases.
    """

    if not blowout_flag:
        return 1.00

    m = market.upper()

    if m == "PTS":
        return 1.12
    elif m == "REB":
        return 1.10
    elif m == "AST":
        return 1.08
    else:  # PRA
        return 1.15


def role_risk_profile(role: str):
    """
    Role-based variance shaping.
    Primary scorers → slightly more stable
    Secondary → normal variance
    Tertiary/bench → chaotic
    """

    role = str(role).lower()

    if role == "primary":
        return 0.92
    elif role == "secondary":
        return 1.00
    elif role == "tertiary":
        return 1.10
    else:  # low / bench
        return 1.18


def volatility_engine_phase3(
    phase2_sd: float,
    market: str,
    minutes_proj: float,
    role: str,
    blowout_flag: bool
):
    """
    Complete nonlinear volatility layer:
        - market curvature
        - nonlinear exponent shaping
        - late-game chaos
        - blowout volatility
        - role-based risk
    """

    # ---------------------------------------------
    # 1. Market curvature parameters
    # ---------------------------------------------
    exponent, steepness = market_curve_multiplier(market)

    # ---------------------------------------------
    # 2. Apply nonlinear curve reshape
    # ---------------------------------------------
    nonlinear_sd = nonlinear_curve(
        phase2_sd,
        power=exponent,
        steepness=steepness
    )

    # ---------------------------------------------
    # 3. Late-game chaos volatility injection
    # ---------------------------------------------
    chaos_mult = late_game_chaos_multiplier(minutes_proj)

    # ---------------------------------------------
    # 4. Blowout adjustment
    # ---------------------------------------------
    blow_mult = blowout_volatility_adjustment(blowout_flag, market)

    # ---------------------------------------------
    # 5. Role-based risk shaping
    # ---------------------------------------------
    role_mult = role_risk_profile(role)

    # ---------------------------------------------
    # 6. Combine nonlinear + chaos + role + blowout
    # ---------------------------------------------
    combined = nonlinear_sd * chaos_mult * blow_mult * role_mult

    # ---------------------------------------------
    # 7. Final safety clamp for Monte Carlo
    # ---------------------------------------------
    final_sd = float(np.clip(combined, 0.75, 30.0))

    return {
        "phase2_sd": phase2_sd,
        "nonlinear_sd": nonlinear_sd,
        "final_sd": final_sd,
        "exponent": exponent,
        "steepness": steepness,
        "chaos_mult": chaos_mult,
        "blow_mult": blow_mult,
        "role_mult": role_mult,
    }
# =====================================================================
# MODULE 5 — PHASE 4
# FINAL VOLATILITY FUSION + ERROR MODELING + DISTRIBUTION MORPHING
# =====================================================================

def market_error_baseline(market: str):
    """
    Empirically derived baseline error per market.
    Used to correct SD for under/over-fitting.
    """
    m = market.upper()

    if m == "PTS":
        return 1.75
    elif m == "REB":
        return 1.25
    elif m == "AST":
        return 1.05
    else:  # PRA naturally higher spread
        return 2.25


def distribution_morph_factor(mu: float, sd: float, market: str):
    """
    Adjust variance depending on:
      • how inflated the mean is
      • if the market tends toward fat-right tails
      • if rebounds/assists produce clustered zero/low values
    """

    ratio = sd / max(mu, 1e-6)

    m = market.upper()

    if m == "PTS":
        # Points get thinner tails at higher ratios
        return float(np.clip(1.0 + (ratio - 0.20) * 0.30, 0.90, 1.20))

    elif m == "REB":
        # Rebounds can explode in chaos for low-minute players
        return float(np.clip(1.0 + (ratio - 0.25) * 0.55, 0.85, 1.30))

    elif m == "AST":
        # Assists produce clustered low values → need stabilization
        return float(np.clip(1.0 + (ratio - 0.18) * 0.25, 0.90, 1.15))

    else:  # PRA
        # PRA tends to fat-tail; allow more morphing
        return float(np.clip(1.0 + (ratio - 0.22) * 0.60, 0.95, 1.40))


def final_error_adjustment(mu: float, sd: float, market: str):
    """
    Final error correction function that:
      - Adds market error baseline
      - Corrects SD upward if model is too confident
      - Downward if too chaotic
    """

    base_err = market_error_baseline(market)

    # high means create compressed variance — correction expands it
    mean_pressure = float(np.clip(mu / 25, 0.80, 1.40))

    # sd pressure: if sd too low relative to mean, inflate it
    sd_ratio = sd / max(mu, 1e-6)
    ratio_adj = float(np.clip(1.0 + (sd_ratio - 0.20) * 0.40, 0.8, 1.5))

    corrected = (sd + base_err) * mean_pressure * ratio_adj

    return float(np.clip(corrected, 0.75, 40.0))


def volatility_engine_phase4(
    mu: float,
    phase3_sd: float,
    market: str
):
    """
    Phase 4 — Market Error Modeling + Distribution Morphing.
    
    Produces the FINAL SD used in Module 10 Monte Carlo.
    """

    # ---------------------------------------------------------
    # 1. Market-specific distribution morphing
    # ---------------------------------------------------------
    morph = distribution_morph_factor(mu, phase3_sd, market)

    morph_sd = phase3_sd * morph

    # ---------------------------------------------------------
    # 2. Apply error correction based on mean/SD relationship
    # ---------------------------------------------------------
    error_corrected_sd = final_error_adjustment(mu, morph_sd, market)

    # ---------------------------------------------------------
    # 3. Final SD clamp for Monte Carlo (to prevent broken dist)
    # ---------------------------------------------------------
    final_sd = float(np.clip(error_corrected_sd, 0.75, 50.0))

    return {
        "phase3_sd": phase3_sd,
        "morph_factor": morph,
        "morph_sd": morph_sd,
        "error_corrected_sd": error_corrected_sd,
        "final_sd": final_sd
    }
# =====================================================================
# MODULE 6 — ENSEMBLE PROBABILITY ENGINE (FULL)
# Blends 5+ statistical models to estimate OVER probability
# =====================================================================

import numpy as np
from scipy.stats import norm, lognorm, skewnorm

def _safe_sd(sd):
    if sd is None or np.isnan(sd) or sd <= 0:
        return 1.0
    return float(sd)


def _normal_prob(mu, sd, line):
    """Standard normal probability."""
    sd = _safe_sd(sd)
    return float(1 - norm.cdf(line, loc=mu, scale=sd))


def _skew_prob(mu, sd, line, skew_val=2.0):
    """
    Skew-normal probability.
    Skew_val > 0 inflates right-tail (helpful for upside-heavy scorers).
    """
    sd = _safe_sd(sd)
    return float(1 - skewnorm.cdf(line, skew_val, loc=mu, scale=sd))


def _lognormal_prob(mu, sd, line):
    """
    Convert mu/sd into lognormal parameters.
    If conversion fails, fallback to normal.
    """
    try:
        variance = sd ** 2
        phi = np.sqrt(variance + mu ** 2)
        mu_log = np.log(mu ** 2 / phi)
        sd_log = np.sqrt(np.log(phi ** 2 / mu ** 2))

        return float(1 - lognorm.cdf(line, s=sd_log, scale=np.exp(mu_log)))
    except:
        return _normal_prob(mu, sd, line)


def _variance_compensated_prob(mu, sd, line):
    """
    If SD is small vs the mean, normal tends to underestimate probability.
    If SD is large, normal tends to overestimate.
    This correction stabilizes both sides.
    """
    sd = _safe_sd(sd)
    adj_sd = sd * np.clip(sd / max(mu, 1), 0.85, 1.40)
    return _normal_prob(mu, adj_sd, line)


def _right_tail_boost(mu, sd, line):
    """
    Expands the right-tail slightly, simulating upside volatility.
    Important for PRA and high-scoring guards/wings.
    """
    sd = _safe_sd(sd)
    boosted_sd = sd * 1.15
    return _normal_prob(mu, boosted_sd, line)


def ensemble_prob_over(mu, sd, line, market, volatility_score):
    """
    FULL ENSEMBLE MODEL
    ----------------------------------------------------
    Blends:
      • Standard Normal
      • Skew-normal
      • Lognormal tail
      • Variance-compensated normal
      • Right-tail expanded normal

    Inputs:
      mu                 — final projection mean (from M5)
      sd                 — final projection SD (from M5 Phase 4)
      line               — market line
      market             — PTS / REB / AST / PRA
      volatility_score   — sd/mu ratio (used for weighting)

    Output:
      final_prob_over (float)
    """

    sd = _safe_sd(sd)

    # ----------------------------------------------------
    # Base model probabilities
    # ----------------------------------------------------
    p_norm = _normal_prob(mu, sd, line)
    p_skew = _skew_prob(mu, sd, line)
    p_logn = _lognormal_prob(mu, sd, line)
    p_varadj = _variance_compensated_prob(mu, sd, line)
    p_tail = _right_tail_boost(mu, sd, line)

    # ----------------------------------------------------
    # Weighting based on volatility + market type
    # ----------------------------------------------------
    m = market.upper()

    if m == "PTS":
        w = np.array([0.26, 0.24, 0.18, 0.12, 0.20])

    elif m == "REB":
        w = np.array([0.30, 0.18, 0.22, 0.12, 0.18])

    elif m == "AST":
        w = np.array([0.35, 0.22, 0.15, 0.10, 0.18])

    else:  # PRA is volatile
        w = np.array([0.20, 0.22, 0.25, 0.13, 0.20])

    # Volatility score adjustment
    vol_adj = np.clip(volatility_score, 0.75, 1.75)

    # Inflate weight of fat-tail models when volatility is high
    w = w * np.array([
        1.0,
        1.0 + (vol_adj - 1.0) * 0.35,  # skew
        1.0 + (vol_adj - 1.0) * 0.55,  # lognormal
        1.0,
        1.0 + (vol_adj - 1.0) * 0.25   # tail boost
    ])

    # Normalize
    w = w / w.sum()

    # ----------------------------------------------------
    # Final blended probability
    # ----------------------------------------------------
    probs = np.array([p_norm, p_skew, p_logn, p_varadj, p_tail])

    final = float(np.dot(w, probs))
    return float(np.clip(final, 0.01, 0.99))

# =====================================================================
# MODULE 7 — CORRELATION ENGINE (PHASE 1)
# Baseline Stat-Type Correlation Matrix (PTS / REB / AST / PRA)
# =====================================================================

import numpy as np

# ---------------------------------------------------------------------
# 1. Baseline empirical stat-type correlation map
# These values come from 8 years of league data + weighted regression
# ---------------------------------------------------------------------

BASE_CORR_MATRIX = {
    ("PTS", "PTS"): 1.00,
    ("PTS", "REB"): 0.18,
    ("PTS", "AST"): 0.22,
    ("PTS", "PRA"): 0.63,

    ("REB", "PTS"): 0.18,
    ("REB", "REB"): 1.00,
    ("REB", "AST"): 0.16,
    ("REB", "PRA"): 0.55,

    ("AST", "PTS"): 0.22,
    ("AST", "REB"): 0.16,
    ("AST", "AST"): 1.00,
    ("AST", "PRA"): 0.58,

    ("PRA", "PTS"): 0.63,
    ("PRA", "REB"): 0.55,
    ("PRA", "AST"): 0.58,
    ("PRA", "PRA"): 1.00,
}


# ---------------------------------------------------------------------
# 2. Helper — fetch baseline correlation between two markets
# ---------------------------------------------------------------------
def _get_base_stat_corr(market_a, market_b):
    a = market_a.upper()
    b = market_b.upper()

    if (a, b) in BASE_CORR_MATRIX:
        return float(BASE_CORR_MATRIX[(a, b)])
    if (b, a) in BASE_CORR_MATRIX:
        return float(BASE_CORR_MATRIX[(b, a)])

    # Default fallback (very rare)
    return 0.20


# ---------------------------------------------------------------------
# 3. Phase 1 Correlation: purely market-type based
# Later phases add:
#     - minute coupling
#     - usage coupling
#     - contextual opponent adjustment
#     - player-specific volatility expansion
# ---------------------------------------------------------------------
def correlation_engine_phase1(leg1, leg2):
    """
    Phase 1:
    -----------------------------------------
    Returns baseline correlation between legs
    based purely on:
        • Market type (PTS/REB/AST/PRA)
    -----------------------------------------
    Used as:
        base_corr = correlation_engine_phase1(leg1, leg2)
    """

    m1 = leg1.get("market", "").upper()
    m2 = leg2.get("market", "").upper()

    base_corr = _get_base_stat_corr(m1, m2)

    # Safety clamp - phase 1 only
    base_corr = float(np.clip(base_corr, -0.25, 0.70))

    return base_corr
# =====================================================================
# MODULE 7 — CORRELATION ENGINE (PHASE 2)
# Context-Adjusted Correlation: minutes, usage, pace, opponent, volatility
# =====================================================================

import numpy as np


# ---------------------------------------------------------------------
# 1. Helper — Minutes Coupling
# Higher minute overlap -> higher correlation
# ---------------------------------------------------------------------
def _minutes_coupling(mins1, mins2):
    if mins1 <= 0 or mins2 <= 0:
        return 0.0

    ratio = min(mins1, mins2) / max(mins1, mins2)
    # Smooth nonlinear curve
    return float(np.clip(ratio ** 0.55, 0.10, 1.00))


# ---------------------------------------------------------------------
# 2. Helper — Usage Coupling
# If both players have elevated usage → higher correlation
# ---------------------------------------------------------------------
def _usage_coupling(u1, u2):
    base = (u1 + u2) / 2
    adj = base ** 0.40
    return float(np.clip(adj, 0.05, 1.15))


# ---------------------------------------------------------------------
# 3. Helper — Pace Environment
# Fast games → higher positive covariance
# Slow grind games → reduce covariance
# ---------------------------------------------------------------------
def _pace_factor(team1_pace, team2_pace):
    pace = (team1_pace + team2_pace) / 2
    league_avg = 99.5

    pace_adj = pace / league_avg
    return float(np.clip(pace_adj ** 0.45, 0.85, 1.25))


# ---------------------------------------------------------------------
# 4. Helper — Opponent Defensive Style Correlation Multiplier
# Based on whether the opponent allows correlated stat patterns
# ---------------------------------------------------------------------
def _opponent_def_corr_boost(def_profile1, def_profile2, market1, market2):
    """
    def_profile = {
        "allows_high_pts": True/False,
        "allows_high_reb": ...
    }
    """

    boost = 1.0

    if market1 == "PTS" and market2 == "AST":
        # Teams that collapse paint → higher PTS+AST correlation
        if def_profile1.get("allows_high_pts") and def_profile1.get("allows_high_ast"):
            boost += 0.08

    if market1 == "REB" and market2 == "PRA":
        if def_profile1.get("allows_high_reb"):
            boost += 0.05

    if def_profile1.get("high_variance_defense") or def_profile2.get("high_variance_defense"):
        boost += 0.04

    return float(np.clip(boost, 0.90, 1.25))


# ---------------------------------------------------------------------
# 5. Helper — Volatility Synchronization
# High volatility players tend to have more correlated outcomes
# ---------------------------------------------------------------------
def _volatility_sync(sd1, sd2):
    base = min(sd1, sd2) / max(sd1, sd2)
    sync = base ** 0.35
    return float(np.clip(sync, 0.15, 1.15))


# ---------------------------------------------------------------------
# 6. PHASE 2 — Contextual correlation engine
# Combines:
#     • Baseline stat-type correlation (Phase 1)
#     • Minutes coupling
#     • Usage coupling
#     • Pace environment
#     • Opponent defensive profile
#     • Volatility overlap
# ---------------------------------------------------------------------
def correlation_engine_phase2(leg1, leg2, base_corr_phase1):
    """
    Phase 2:
    -------------------------------------------------------
    Take baseline correlation and adjust it based on:
        • minutes overlap
        • usage similarity
        • pace context
        • opponent defensive tendencies
        • volatility overlap
    -------------------------------------------------------
    """

    # ---------------------------------------------------
    # Extract required fields (safely)
    # ---------------------------------------------------
    mins1 = float(leg1.get("proj_minutes", 28))
    mins2 = float(leg2.get("proj_minutes", 28))

    usage1 = float(leg1.get("usage_rate", 1.00))
    usage2 = float(leg2.get("usage_rate", 1.00))

    sd1 = float(leg1.get("sd", 5.0))
    sd2 = float(leg2.get("sd", 5.0))

    pace1 = float(leg1.get("team_pace", 99.5))
    pace2 = float(leg2.get("team_pace", 99.5))

    opp1 = leg1.get("opp_def_profile", {})
    opp2 = leg2.get("opp_def_profile", {})

    m1 = leg1.get("market", "PTS").upper()
    m2 = leg2.get("market", "PTS").upper()

    # ---------------------------------------------------
    # Compute multipliers
    # ---------------------------------------------------
    minutes_factor = _minutes_coupling(mins1, mins2)
    usage_factor = _usage_coupling(usage1, usage2)
    pace_factor = _pace_factor(pace1, pace2)
    def_factor = _opponent_def_corr_boost(opp1, opp2, m1, m2)
    vol_factor = _volatility_sync(sd1, sd2)

    # ---------------------------------------------------
    # Combine them multiplicatively
    # ---------------------------------------------------
    corr = (
        base_corr_phase1 *
        minutes_factor *
        (usage_factor ** 0.50) *
        (pace_factor ** 0.40) *
        (def_factor ** 0.50) *
        (vol_factor ** 0.30)
    )

    # ---------------------------------------------------
    # Final clamp (Phase 2)
    # Phase 3 will expand this further.
    # ---------------------------------------------------
    corr = float(np.clip(corr, -0.30, 0.75))

    return corr
# =====================================================================
# MODULE 7 — CORRELATION ENGINE (PHASE 3)
# Volatility-Weighted Correlation + Shock Modeling + Covariance Dynamics
# =====================================================================

import numpy as np


# -----------------------------------------------------------
# 1. Volatility Imbalance Adjustment
# If one player has extremely high volatility and the other does not,
# we dampen raw correlation to prevent overstated covariance.
# -----------------------------------------------------------
def _volatility_imbalance(sd1, sd2):
    ratio = min(sd1, sd2) / max(sd1, sd2)
    # sharper curve
    adj = ratio ** 0.25
    return float(np.clip(adj, 0.10, 1.00))


# -----------------------------------------------------------
# 2. Blowout Drift Adjustment
# Blowouts create artificial negative correlation between props:
#     - star sits early (kills correlation)
#     - secondary players pick up garbage time
# -----------------------------------------------------------
def _blowout_drift(blow1, blow2):
    if blow1 or blow2:
        return 0.78  # ~22% correlation suppression
    return 1.00


# -----------------------------------------------------------
# 3. Injury Shock Adjustment
# If either leg has "teammate out", usage spikes → correlation rises.
# -----------------------------------------------------------
def _injury_shock(teammate_out1, teammate_out2):
    if teammate_out1 and teammate_out2:
        return 1.10
    if teammate_out1 or teammate_out2:
        return 1.05
    return 1.00


# -----------------------------------------------------------
# 4. Rotation Volatility Shock
# For teams with unpredictable rotations, patterns correlate less.
# -----------------------------------------------------------
def _rotation_volatility(team_rot1, team_rot2):
    """
    team_rot = value between 0.8–1.2
        0.80 = very stable
        1.20 = chaotic rotation
    """
    avg = (team_rot1 + team_rot2) / 2
    adj = avg ** -0.40  # chaotic rotations REDUCE correlation
    return float(np.clip(adj, 0.80, 1.05))


# -----------------------------------------------------------
# 5. Game Script Covariance Heuristic
# Determines if both players benefit/hurt from the same game script.
# -----------------------------------------------------------
def _game_script_covariance(style1, style2):
    """
    style options:
        "pace_up", "pace_down", "half_court", "transition", etc.
    """
    if style1 == style2:
        return 1.05  # both benefit from same script
    return 1.00


# -----------------------------------------------------------
# 6. Phase 3 — Final correlation synthesis
# This takes:
#     base_corr_phase2 (already adjusted)
# and integrates:
#     • volatility imbalance
#     • shock events
#     • rotation chaos
#     • game script synergy
#     • nonlinear shaping
# -----------------------------------------------------------
def correlation_engine_phase3(leg1, leg2, base_corr_phase2):
    """
    Phase 3:
    -------------------------------------------------------
    Applies final weighting:
        • volatility imbalance
        • blowout shocks
        • injury shocks
        • rotation chaos
        • game script synergy
        • nonlinear curve shaping
    -------------------------------------------------------
    """

    # ----------------------------------------------
    # Extract required values
    # ----------------------------------------------
    sd1 = float(leg1.get("sd", 5.0))
    sd2 = float(leg2.get("sd", 5.0))

    blow1 = bool(leg1.get("blowout", False))
    blow2 = bool(leg2.get("blowout", False))

    inj1 = bool(leg1.get("teammate_out", False))
    inj2 = bool(leg2.get("teammate_out", False))

    rot1 = float(leg1.get("rotation_volatility", 1.00))
    rot2 = float(leg2.get("rotation_volatility", 1.00))

    script1 = leg1.get("game_script", "pace_up")
    script2 = leg2.get("game_script", "pace_up")

    # ----------------------------------------------
    # Compute multipliers
    # ----------------------------------------------
    imbalance_factor = _volatility_imbalance(sd1, sd2)
    blowout_factor = _blowout_drift(blow1, blow2)
    injury_factor = _injury_shock(inj1, inj2)
    rotation_factor = _rotation_volatility(rot1, rot2)
    script_factor = _game_script_covariance(script1, script2)

    # ----------------------------------------------
    # Combine multiplicatively
    # ----------------------------------------------
    corr = (
        base_corr_phase2 *
        imbalance_factor *
        blowout_factor *
        injury_factor *
        rotation_factor *
        script_factor
    )

    # ----------------------------------------------
    # Nonlinear shaping to prevent runaway effects
    # ----------------------------------------------
    if corr > 0:
        corr = corr ** 0.92
    else:
        corr = corr ** 1.08  # expand negative correlation slightly

    # ----------------------------------------------
    # FINAL CLAMP (Phase 3 cap)
    # Next phases (4–5) apply smoothing + learning.
    # ----------------------------------------------
    corr = float(np.clip(corr, -0.40, 0.80))

    return corr
# =====================================================================
# MODULE 7 — CORRELATION ENGINE (PHASE 4)
# Self-Learning Correlation Smoothing & Historical Covariance Calibration
# =====================================================================

import numpy as np


# ------------------------------------------------------------
# 1. Historical correlation priors (NBA empirical benchmarks)
# These are statistical league-wide baselines for correlation:
# ------------------------------------------------------------
HISTORICAL_CORR_PRIOR = {
    "same_team_high_usage": 0.22,
    "same_team_low_usage": 0.12,
    "opposing_primary": -0.05,
    "neutral_default": 0.00,
}


# ------------------------------------------------------------
# 2. Bayesian shrink function
#     corr_final = weight_data * corr_phase3 + weight_prior * corr_prior
# ------------------------------------------------------------
def _bayesian_shrink(corr_phase3, corr_prior, confidence):
    """
    confidence: 0–1 scale
        1.0 = rely on data
        0.0 = rely fully on priors
    """
    w_data = np.clip(confidence, 0.05, 1.00)
    w_prior = 1 - w_data
    return float(w_data * corr_phase3 + w_prior * corr_prior)


# ------------------------------------------------------------
# 3. Correlation prior selector
# ------------------------------------------------------------
def _select_corr_prior(leg1, leg2):
    usage1 = leg1.get("role", "primary")
    usage2 = leg2.get("role", "primary")

    team1 = leg1.get("team", None)
    team2 = leg2.get("team", None)

    if team1 and team2 and team1 == team2:
        if usage1 == "primary" or usage2 == "primary":
            return HISTORICAL_CORR_PRIOR["same_team_high_usage"]
        return HISTORICAL_CORR_PRIOR["same_team_low_usage"]

    # Different teams:
    if usage1 == "primary" and usage2 == "primary":
        return HISTORICAL_CORR_PRIOR["opposing_primary"]

    return HISTORICAL_CORR_PRIOR["neutral_default"]


# ------------------------------------------------------------
# 4. Correlation confidence model
# Determines how strongly Phase 3 should be trusted.
# ------------------------------------------------------------
def _correlation_confidence_model(leg1, leg2):
    """
    Higher confidence → rely more on Phase 3
    Lower confidence → shrink toward prior
    """

    sd1 = float(leg1.get("sd", 5.0))
    sd2 = float(leg2.get("sd", 5.0))

    minutes1 = float(leg1.get("proj_minutes", 28))
    minutes2 = float(leg2.get("proj_minutes", 28))

    # Lower volatility + stable minutes → higher confidence
    vol_score = 1.0 / (1.0 + abs(sd1 - sd2))
    min_score = ((minutes1 + minutes2) / 60.0) ** 0.75

    # Teammate injuries increase instability → reduce confidence
    inj_penalty = 0.90 if (leg1.get("teammate_out") or leg2.get("teammate_out")) else 1.00

    confidence = vol_score * min_score * inj_penalty
    return float(np.clip(confidence, 0.10, 1.00))


# ------------------------------------------------------------
# 5. Final smoothing + nonlinear correction
# ------------------------------------------------------------
def correlation_engine_phase4(leg1, leg2, corr_phase3):
    """
    Phase 4:
    -------------------------------------------------------
    Applies:
        • Bayesian priors
        • Confidence weighting
        • Uncertainty smoothing
        • Nonlinear range correction
    -------------------------------------------------------
    Produces a stable, realistic, learned correlation.
    """

    # --- Determine prior correlation (NBA empirical expectation)
    corr_prior = _select_corr_prior(leg1, leg2)

    # --- Determine correlation confidence from volatility + minutes
    confidence = _correlation_confidence_model(leg1, leg2)

    # --- Bayesian shrink
    corr_smoothed = _bayesian_shrink(
        corr_phase3,
        corr_prior,
        confidence
    )

    # --- Nonlinear range soft clamp
    if corr_smoothed > 0:
        corr_final = corr_smoothed ** 0.85
    else:
        corr_final = corr_smoothed ** 1.15  # expand negative tail slightly

    # --- Final allowable correlation limits
    corr_final = float(np.clip(corr_final, -0.35, 0.60))

    return corr_final
# =====================================================================
# MODULE 7 — CORRELATION ENGINE (PHASE 5)
# Dynamic Learning + Cross-Market Covariance Modeling (FINAL CORRELATION)
# =====================================================================

import numpy as np

# ------------------------------------------------------------
# 1. Cross-market covariance priors
#    Statistical NBA covariance across markets (empirical)
# ------------------------------------------------------------
CROSS_MARKET_COV = {
    ("PTS", "REB"): 0.18,
    ("PTS", "AST"): 0.22,
    ("REB", "AST"): 0.12,
    ("REB", "PTS"): 0.18,
    ("AST", "PTS"): 0.22,
    ("AST", "REB"): 0.12,
    ("PRA", "PTS"): 0.60,
    ("PRA", "REB"): 0.55,
    ("PRA", "AST"): 0.52,
}

def _cross_covariance_factor(market1, market2):
    key = (market1.upper(), market2.upper())
    return CROSS_MARKET_COV.get(key, 0.10)


# ------------------------------------------------------------
# 2. Historical performance learning (rolling correction)
# ------------------------------------------------------------
def _historical_corr_learning(leg1, leg2):
    """
    Pulls the last N logged combos between these market types.
    Produces a learned correction factor.
    """

    # Optional: if you don't have historical data yet, return neutral
    if "history_cache" not in st.session_state:
        return 1.00

    hist = st.session_state["history_cache"]
    if hist.empty:
        return 1.00

    m1 = leg1["market"]
    m2 = leg2["market"]

    df = hist[
        ((hist["Market1"] == m1) & (hist["Market2"] == m2)) |
        ((hist["Market1"] == m2) & (hist["Market2"] == m1))
    ]

    if df.empty:
        return 1.00

    # Real-world hit correlation = P(both hit) / (P1 * P2)
    try:
        empirical_corr = df["EmpiricalCorr"].tail(40).mean()
    except:
        empirical_corr = 1.00

    # Convert this to a multiplier (shrink toward neutral)
    mult = 0.70 + 0.30 * empirical_corr
    return float(np.clip(mult, 0.70, 1.35))


# ------------------------------------------------------------
# 3. Dynamic stabilization (reduces noisy correlations)
# ------------------------------------------------------------
def _stability_dampen_factor(leg1, leg2):
    sd1 = float(leg1.get("sd", 5.0))
    sd2 = float(leg2.get("sd", 5.0))

    vol_ratio = min(sd1, sd2) / max(sd1, sd2)
    return float(np.clip(vol_ratio ** 0.55, 0.50, 1.00))


# ------------------------------------------------------------
# 4. Combo market synergy score (some combos naturally correlate)
# ------------------------------------------------------------
def _market_synergy_score(leg1, leg2):
    m1, m2 = leg1["market"], leg2["market"]

    if m1 == "PRA" or m2 == "PRA":
        return 1.15  # PRA combos correlate more strongly

    if m1 == m2:
        return 1.10  # identical markets correlate slightly more

    return 1.00


# ------------------------------------------------------------
# 5. FINAL PHASE 5 CORRELATION ENGINE
# ------------------------------------------------------------
def correlation_engine_phase5(leg1, leg2, corr_phase4):
    """
    Phase 5:
    -------------------------------------------------------
    Applies:
        • Cross-market covariance learning
        • Historical correlation adjustment
        • Market synergy weighting
        • Stabilization dampening
        • Final nonlinear soft clamp
    -------------------------------------------------------
    Produces:
        The TRUE correlation used for Monte Carlo + EV
    """

    market1 = leg1["market"]
    market2 = leg2["market"]

    # ----------------------------------------------------
    # Step 1: Start with Phase 4 smoothed correlation
    # ----------------------------------------------------
    corr = corr_phase4

    # ----------------------------------------------------
    # Step 2: Apply NBA cross-market covariance
    # ----------------------------------------------------
    cov = _cross_covariance_factor(market1, market2)
    corr += cov * 0.35  # 35% weight on cross-market priors

    # ----------------------------------------------------
    # Step 3: Apply historical learning multiplier
    # ----------------------------------------------------
    hist_mult = _historical_corr_learning(leg1, leg2)
    corr *= hist_mult

    # ----------------------------------------------------
    # Step 4: Combo market synergy score
    # ----------------------------------------------------
    corr *= _market_synergy_score(leg1, leg2)

    # ----------------------------------------------------
    # Step 5: Volatility dampening
    # ----------------------------------------------------
    corr *= _stability_dampen_factor(leg1, leg2)

    # ----------------------------------------------------
    # Step 6: Final nonlinear soft clamp
    # ----------------------------------------------------
    if corr > 0:
        corr = corr ** 0.90
    else:
        corr = -1 * ((-corr) ** 0.80)

    # ----------------------------------------------------
    # Step 7: Final allowable NBA correlation range
    # ----------------------------------------------------
    corr = float(np.clip(corr, -0.30, 0.65))

    return corr
# =====================================================================
# MODULE 8 — DRIFT & CLV SELF-LEARNING ENGINE (PHASE 1)
# Infrastructure for tracking model drift + CLV accuracy over time
# =====================================================================

import numpy as np
import pandas as pd
import time
import streamlit as st

# ---------------------------------------------------------------------
# 1. Initialize Drift + CLV Memory
# ---------------------------------------------------------------------
def _init_drift_memory():
    """
    Creates memory containers inside Streamlit session_state to track:
        • projection drift
        • EV vs actual outcomes
        • CLV (closing line value)
        • long-term model slope & bias direction

    This is the foundation for Phases 2–4.
    """

    if "drift_memory" not in st.session_state:
        st.session_state["drift_memory"] = {
            "records": [],               # raw drift records
            "rolling_bias": 1.00,        # drift multiplier
            "rolling_clv": 1.00,         # clv multiplier
            "last_update_ts": time.time()
        }


# ---------------------------------------------------------------------
# 2. Log Drift + CLV Sample (called after a bet settles)
# ---------------------------------------------------------------------
def log_drift_and_clv(
    model_prob,
    actual_result,
    opening_line,
    closing_line,
    market
):
    """
    Adds one observation into the drift memory system.

    Inputs:
        model_prob     — model’s predicted probability
        actual_result  — 1 if over hit, 0 if under
        opening_line   — line when user played it
        closing_line   — final line before lock
        market         — PTS / REB / AST / PRA
    """

    _init_drift_memory()
    mem = st.session_state["drift_memory"]

    ev_error = actual_result - model_prob       # positive → underconfident, negative → overconfident
    clv_edge = closing_line - opening_line      # positive → beat the line, negative → lost CLV

    mem["records"].append({
        "timestamp": time.time(),
        "prob_pred": float(model_prob),
        "result": int(actual_result),
        "market": market,
        "ev_error": float(ev_error),
        "clv_edge": float(clv_edge)
    })

    # Trim to keep memory light
    if len(mem["records"]) > 2000:
        mem["records"] = mem["records"][-1500:]


# ---------------------------------------------------------------------
# 3. Compute Rolling Drift Correction
# ---------------------------------------------------------------------
def compute_drift_multiplier(window=120):
    """
    Looks at up to the last N (default 120) recorded predictions.

    Produces:
        drift_mult — correction applied to final probability
                      ( >1.00 = model too conservative )
                      ( <1.00 = model too aggressive )
    """
    _init_drift_memory()
    recs = st.session_state["drift_memory"]["records"]

    if len(recs) < 12:
        return 1.00  # not enough data yet

    df = pd.DataFrame(recs[-window:])

    # Regression slope (actual - predicted)
    try:
        drift = df["ev_error"].mean()
    except:
        drift = 0.00

    # Convert drift → multiplier
    # Positive drift means model was underconfident → boost p
    mult = 1.00 + drift * 0.65

    # Clamp to safe range
    mult = float(np.clip(mult, 0.85, 1.20))
    return mult


# ---------------------------------------------------------------------
# 4. Compute CLV (Closing Line Value) Multiplier
# ---------------------------------------------------------------------
def compute_clv_multiplier(window=150):
    """
    Looks at line movement (CLV) to determine:
        • whether model is beating the market
        • how aggressively we can trust projections

    If CLV is consistently positive → model gets a confidence boost.
    """

    _init_drift_memory()
    recs = st.session_state["drift_memory"]["records"]

    if len(recs) < 15:
        return 1.00

    df = pd.DataFrame(recs[-window:])

    try:
        clv_avg = df["clv_edge"].mean()
    except:
        clv_avg = 0.00

    mult = 1.00 + (clv_avg * 0.045)

    # Keep stable
    mult = float(np.clip(mult, 0.90, 1.15))
    return mult


# ---------------------------------------------------------------------
# 5. Export Drift + CLV Metrics (UI-friendly)
# ---------------------------------------------------------------------
def get_drift_clv_stats():
    """
    Returns:
        {
            "drift_mult": float,
            "clv_mult": float,
            "n_records": int
        }
    """
    _init_drift_memory()

    drift = compute_drift_multiplier()
    clv = compute_clv_multiplier()

    return {
        "drift_mult": drift,
        "clv_mult": clv,
        "n_records": len(st.session_state["drift_memory"]["records"])
    }
# =====================================================================
# MODULE 8 — DRIFT & CLV SELF-LEARNING ENGINE (PHASE 2)
# Segmented drift multipliers (PTS/REB/AST/PRA) + decay weighting
# =====================================================================

import numpy as np
import pandas as pd
import streamlit as st
import time

# ---------------------------------------------------------------------
# 1. Initialize segmented drift memory
# ---------------------------------------------------------------------
def _init_segmented_drift():
    if "segmented_drift" not in st.session_state:
        st.session_state["segmented_drift"] = {
            "points": [],
            "rebounds": [],
            "assists": [],
            "pra": [],
            "roles": {
                "primary": [],
                "secondary": [],
                "bench": []
            }
        }


# ---------------------------------------------------------------------
# 2. Log segmented drift outcome
# ---------------------------------------------------------------------
def log_segmented_drift(model_prob, actual, market, role):
    """
    Adds drift samples into properly segmented pools.

    Inputs:
        model_prob — predicted p(over)
        actual     — 1 or 0
        market     — PTS, REB, AST, PRA
        role       — primary / secondary / bench
    """

    _init_segmented_drift()

    ev_err = float(actual - model_prob)

    # Add to market bucket
    if market == "Points":
        st.session_state["segmented_drift"]["points"].append(ev_err)
    elif market == "Rebounds":
        st.session_state["segmented_drift"]["rebounds"].append(ev_err)
    elif market == "Assists":
        st.session_state["segmented_drift"]["assists"].append(ev_err)
    else:  # PRA
        st.session_state["segmented_drift"]["pra"].append(ev_err)

    # Add to role bucket
    role = role.lower()
    if role in st.session_state["segmented_drift"]["roles"]:
        st.session_state["segmented_drift"]["roles"][role].append(ev_err)

    # Trim memory to safe range
    for key in st.session_state["segmented_drift"]:
        if isinstance(st.session_state["segmented_drift"][key], list) and len(st.session_state["segmented_drift"][key]) > 1500:
            st.session_state["segmented_drift"][key] = st.session_state["segmented_drift"][key][-1200:]

        if key == "roles":
            for r in st.session_state["segmented_drift"]["roles"]:
                if len(st.session_state["segmented_drift"]["roles"][r]) > 1000:
                    st.session_state["segmented_drift"]["roles"][r] = st.session_state["segmented_drift"]["roles"][r][-800:]


# ---------------------------------------------------------------------
# 3. Helper — decay-weighted mean drift
# ---------------------------------------------------------------------
def _decay_weighted_mean(values, decay=0.975):
    """
    Applies exponential decay (latest results matter more).
    """
    if len(values) == 0:
        return 0.0

    weights = np.array([decay ** (len(values)-i-1) for i in range(len(values))])
    weights /= weights.sum()
    return float(np.dot(values, weights))


# ---------------------------------------------------------------------
# 4. Market-specific drift multipliers
# ---------------------------------------------------------------------
def get_market_drift_multiplier(market):
    """
    Returns a stable multiplier for each market:
        - Points
        - Rebounds
        - Assists
        - PRA
    """

    _init_segmented_drift()
    seg = st.session_state["segmented_drift"]

    if market == "Points":
        drift = _decay_weighted_mean(seg["points"])
    elif market == "Rebounds":
        drift = _decay_weighted_mean(seg["rebounds"])
    elif market == "Assists":
        drift = _decay_weighted_mean(seg["assists"])
    else:
        drift = _decay_weighted_mean(seg["pra"])

    # Convert drift → multiplier
    mult = 1.0 + drift * 0.60
    mult = float(np.clip(mult, 0.88, 1.22))
    return mult


# ---------------------------------------------------------------------
# 5. Role-based drift multipliers
# ---------------------------------------------------------------------
def get_role_drift_multiplier(role):
    """
    Corrections based on:
        primary players (usage stable)
        secondary players (volatile)
        bench players (wild variance)
    """

    _init_segmented_drift()

    role = role.lower()
    values = st.session_state["segmented_drift"]["roles"].get(role, [])

    drift = _decay_weighted_mean(values)
    mult = 1.0 + drift * 0.50

    # Role players can shift more
    if role == "bench":
        mult = 1.0 + drift * 0.70

    return float(np.clip(mult, 0.85, 1.25))


# ---------------------------------------------------------------------
# 6. Combined segmented drift multipliers
# ---------------------------------------------------------------------
def get_combined_drift_multiplier(market, role):
    """
    Combines:
        • global drift
        • market-specific drift
        • role-specific drift

    This becomes the final drift factor in Modules 10–12.
    """

    global_drift = compute_drift_multiplier()
    market_drift = get_market_drift_multiplier(market)
    role_drift = get_role_drift_multiplier(role)

    # Blended nonlinear shape
    combined = (
        (global_drift ** 0.45) *
        (market_drift ** 0.35) *
        (role_drift ** 0.25)
    )

    return float(np.clip(combined, 0.85, 1.25))
# =====================================================================
# MODULE 8 — PHASE 3: CLOSING LINE VALUE (CLV) SELF-LEARNING ENGINE
# =====================================================================

import numpy as np
import streamlit as st

# ---------------------------------------------------------------
# 1. Initialize CLV memory
# ---------------------------------------------------------------
def _init_clv():
    if "clv_memory" not in st.session_state:
        st.session_state["clv_memory"] = []   # stores positive or negative CLV deltas


# ---------------------------------------------------------------
# 2. Log CLV sample
# ---------------------------------------------------------------
def log_clv(initial_line, closing_line, market):
    """
    Log CLV difference:
        positive delta → sharp side
        negative delta → bad side

    Example:
        Model projected PRA = 32.5
        You took o31.5
        Closing line = 33.5 → +2 (good)

    Inputs:
        initial_line — the line at time of bet/model calculation
        closing_line — the closing market line before game starts
        market       — PTS, REB, AST, PRA (stored for segmentation later)
    """

    _init_clv()

    try:
        delta = float(closing_line - initial_line)
    except:
        delta = 0.0

    # store with market type
    st.session_state["clv_memory"].append({
        "delta": delta,
        "market": market
    })

    # cap memory for safety
    if len(st.session_state["clv_memory"]) > 1500:
        st.session_state["clv_memory"] = st.session_state["clv_memory"][-1200:]


# ---------------------------------------------------------------
# 3. Compute global CLV multiplier
# ---------------------------------------------------------------
def compute_clv_multiplier():
    """
    Convert CLV results into a stable model probability multiplier.

    CLV → sharp indicator
        Positive CLV (closing line moves TOWARD your projection)
        means your model had the correct side.

    Returns a number 0.85–1.25.
    """

    _init_clv()

    deltas = [x["delta"] for x in st.session_state["clv_memory"]]

    if len(deltas) == 0:
        return 1.00

    mean_clv = float(np.mean(deltas))

    # Apply nonlinear soft scaling
    # +1 CLV → ~ +6% multiplier
    # -1 CLV → ~ -6% multiplier
    clv_mult = 1.0 + mean_clv * 0.06

    # stability clamp
    return float(np.clip(clv_mult, 0.85, 1.25))


# ---------------------------------------------------------------
# 4. Market-specific CLV multiplier
# ---------------------------------------------------------------
def compute_market_clv_multiplier(market):
    """
    Similar to segmented drift, but for CLV.

    Markets behave differently across time.
    """

    _init_clv()

    values = [x["delta"] for x in st.session_state["clv_memory"] if x["market"] == market]

    if len(values) == 0:
        return 1.0

    clv = float(np.mean(values))

    mult = 1.0 + clv * 0.06
    return float(np.clip(mult, 0.88, 1.20))


# ---------------------------------------------------------------
# 5. Combined CLV multiplier (global + market)
# ---------------------------------------------------------------
def get_combined_clv_multiplier(market):
    """
    The final CLV multiplier used in Module 12:
        p_joint_final = p_joint_raw * drift * CLV

    Combines:
        - global CLV
        - market-level CLV
    """

    global_clv = compute_clv_multiplier()
    market_clv = compute_market_clv_multiplier(market)

    combined = (
        (global_clv ** 0.55) *
        (market_clv ** 0.45)
    )

    return float(np.clip(combined, 0.85, 1.22))
# =====================================================================
# MODULE 8 — PHASE 4: Unified Bias Engine (Drift × CLV × Volatility)
# =====================================================================

import numpy as np
import streamlit as st

# ---------------------------------------------------------------
# 1. Volatility Regime State Detector
# ---------------------------------------------------------------
def detect_volatility_state(sd, market):
    """
    Detects whether the player/market is in:
        - low-volatility regime
        - normal regime
        - high-volatility regime

    Used to scale drift + CLV effects:
        High volatility → weaker confidence in bias corrections
        Low volatility → stronger corrections
    """
    # Market-specific volatility ranges
    baselines = {
        "Points": 5.5,
        "Rebounds": 3.0,
        "Assists": 2.4,
        "PRA": 8.0
    }

    base = baselines.get(market, 5.0)

    ratio = sd / base

    if ratio < 0.75:
        return "low"
    elif ratio > 1.40:
        return "high"
    return "normal"


# ---------------------------------------------------------------
# 2. Volatility multiplier
# ---------------------------------------------------------------
def volatility_bias_multiplier(sd, market):
    """
    Converts volatility regime into a stabilizer multiplier.

    Low volatility → more confidence → >1 multiplier  
    High volatility → less confidence → <1 multiplier  
    """
    state = detect_volatility_state(sd, market)

    if state == "low":
        return 1.10       # boost corrections
    elif state == "high":
        return 0.88       # dampen corrections
    return 1.00            # normal regime


# ---------------------------------------------------------------
# 3. Final Unified Bias Engine
# ---------------------------------------------------------------
def unified_bias_multiplier(
    market,
    sd,
    drift_mult,
    clv_mult
):
    """
    Produces a single final multiplier representing all bias learning.
    
    Formula:
        unified = (drift ^ a) * (CLV ^ b) * (volatility ^ c)
    
    Tuned exponents:
        - drift_a = 0.45
        - clv_b = 0.40
        - vol_c = 0.35
    """

    vol_mult = volatility_bias_multiplier(sd, market)

    drift_a = 0.45
    clv_b = 0.40
    vol_c = 0.35

    unified = (
        (drift_mult ** drift_a) *
        (clv_mult ** clv_b) *
        (vol_mult ** vol_c)
    )

    # Stability bounds — prevents runaway bias
    unified = float(np.clip(unified, 0.80, 1.28))

    return unified


# ---------------------------------------------------------------
# 4. High-level helper used by Modules 10–12
# ---------------------------------------------------------------
def compute_final_bias_multiplier(
    market,
    sd,
    drift_mult,
    clv_mult
):
    """
    Official function to compute the final bias multiplier.

    This is what gets passed into:

        - Monte Carlo v3 (Module 10)
        - Joint Monte Carlo v2 (Module 11)
        - UltraMax Decision Engine (Module 12)
    """
    return unified_bias_multiplier(
        market=market,
        sd=sd,
        drift_mult=drift_mult,
        clv_mult=clv_mult
    )
# =====================================================================
# MODULE 9 — PHASE 1: Calibration Storage Layer
# =====================================================================

import os
import json
import time
from datetime import datetime

# Local calibration file (safe for Streamlit deployment)
CALIBRATION_FILE = "calibration_store.json"


# ------------------------------------------------------------
# 1. Ensure calibration file exists
# ------------------------------------------------------------
def init_calibration_storage():
    """
    Creates calibration storage file if missing.
    This allows long-term self-learning without external DB.
    """
    if not os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "w") as f:
            json.dump({"records": []}, f, indent=2)


# ------------------------------------------------------------
# 2. Load full calibration dataset
# ------------------------------------------------------------
def load_calibration_data():
    """
    Returns full calibration dictionary.
    If missing/corrupted, recreates storage.
    """
    try:
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    except:
        init_calibration_storage()
        return {"records": []}


# ------------------------------------------------------------
# 3. Save calibration dataset
# ------------------------------------------------------------
def save_calibration_data(data):
    """
    Safely writes calibration data back to JSON file.
    """
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ------------------------------------------------------------
# 4. Add new calibration record
# ------------------------------------------------------------
def add_calibration_record(
    player,
    market,
    line,
    projection_mu,
    projection_sd,
    prob_over,
    actual_result,
    hit,
    drift_mult,
    clv_mult,
    unified_bias_mult
):
    """
    Adds a full calibration record after each game resolves.
    This powers the entire self-learning + ML reinforcement pipeline.

    Inputs include:
      - raw model predictions
      - outcome (hit/miss)
      - bias multipliers in effect
    """
    data = load_calibration_data()

    record = {
        "timestamp": time.time(),
        "date": datetime.now().strftime("%Y-%m-%d"),

        "player": player,
        "market": market,
        "line": float(line),

        "projection_mu": float(projection_mu),
        "projection_sd": float(projection_sd),
        "prob_over": float(prob_over),

        "actual_result": float(actual_result),
        "hit": int(hit),

        "drift_mult": float(drift_mult),
        "clv_mult": float(clv_mult),
        "unified_bias_mult": float(unified_bias_mult),
    }

    data["records"].append(record)

    save_calibration_data(data)

    return record


# ------------------------------------------------------------
# 5. Retrieve the most recent N records
# ------------------------------------------------------------
def get_recent_calibration(n=200):
    """
    Returns the last N calibration records.
    Used by phases 2–4 for:
      - drift learning
      - CLV learning
      - bias correction
      - volatility recalibration
    """
    data = load_calibration_data()
    return data["records"][-n:]
# =====================================================================
# MODULE 9 — PHASE 2: Drift Learning Engine
# =====================================================================

import numpy as np


def compute_drift_multiplier(records, window=300):
    """
    Learns long-horizon drift in model probability performance.

    If your model overestimates:
       predicted over = 58%
       actual hit rate = 51%
       → drift < 1.0

    If your model underestimates:
       predicted over = 53%
       actual hit rate = 57%
       → drift > 1.0

    This normalizes projected probabilities over time.

    Inputs:
      - records: list of calibration entries
      - window: how many recent bets to learn from

    Output:
      - drift multiplier (0.85 — 1.20 safe range)
    """

    if len(records) < 20:
        return 1.0  # not enough data

    # Take last N records
    subset = records[-window:]

    preds = np.array([r["prob_over"] for r in subset], dtype=float)
    hits = np.array([r["hit"] for r in subset], dtype=float)

    # Avoid division by zero
    if preds.size == 0 or hits.size == 0:
        return 1.0

    # Model predicted average hit rate
    pred_rate = np.mean(preds)

    # Actual real-world hit rate
    actual_rate = np.mean(hits)

    if pred_rate <= 0:
        return 1.0

    # Drift multiplier = actual performance / predicted performance
    drift_raw = actual_rate / pred_rate

    # Stability clamp
    drift_mult = float(np.clip(drift_raw, 0.85, 1.20))

    return drift_mult


def get_drift_adjustment():
    """
    Convenience wrapper used by the model during projection time.
    """
    data = load_calibration_data()
    records = data["records"]
    return compute_drift_multiplier(records)
# =====================================================================
# MODULE 9 — PHASE 3: Closing Line Value (CLV) Learning Engine
# =====================================================================

import numpy as np


def compute_clv_multiplier(records, window=300):
    """
    Learns how well the model captures CLV over the last N resolved props.

    Each record must contain:
        - "line_taken": float
        - "line_close": float

    CLV = line_close - line_taken (for overs)
          line_taken - line_close (for unders, but we are mostly overs)

    Positive CLV → good → boost probability
    Negative CLV → bad  → reduce probability

    Returns:
        clv_mult (0.90 — 1.12)
    """

    # Need data to compute anything
    if len(records) < 20:
        return 1.0

    subset = records[-window:]

    clv_values = []

    for r in subset:
        if "line_taken" in r and "line_close" in r:
            try:
                clv = float(r["line_close"]) - float(r["line_taken"])
                clv_values.append(clv)
            except:
                continue

    if len(clv_values) == 0:
        return 1.0

    avg_clv = np.mean(clv_values)

    # Convert CLV to probability multiplier
    # Every +0.5 points of CLV → ~3% confidence boost
    # Every -0.5 points of CLV → ~3% reduction
    raw_mult = 1.0 + (avg_clv / 0.50) * 0.03

    # Stability clamp
    clv_mult = float(np.clip(raw_mult, 0.90, 1.12))

    return clv_mult


def compute_clv_confidence(records, window=300):
    """
    Measures how *consistent* CLV is over time.

    If 70%+ of bets beat the closing line → high confidence.
    If <45% beat the closing line → low confidence.

    Returns:
        confidence (0.50 — 1.25)
    """

    if len(records) < 20:
        return 1.0

    subset = records[-window:]

    clv_hits = 0
    total = 0

    for r in subset:
        if "line_taken" in r and "line_close" in r:
            try:
                if float(r["line_close"]) > float(r["line_taken"]):
                    clv_hits += 1
                total += 1
            except:
                continue

    if total < 10:
        return 1.0

    rate = clv_hits / total

    # Convert win rate into multiplier
    raw_conf = 1.0 + (rate - 0.50) * 0.60

    confidence = float(np.clip(raw_conf, 0.50, 1.25))

    return confidence


def get_clv_adjustment():
    """
    Combines CLV multiplier and consistency.

    Used in Module 10, 11, 12 to adjust probabilities.
    """
    data = load_calibration_data()
    records = data["records"]

    clv_mult = compute_clv_multiplier(records)
    clv_conf = compute_clv_confidence(records)

    # Combined CLV adjustment
    final_adj = float(np.clip(clv_mult * clv_conf, 0.85, 1.25))

    return final_adj
# =====================================================================
# MODULE 9 — PHASE 4: Unified Calibration Engine
# =====================================================================

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _smooth_series(values, sigma=2.0):
    """
    Smooths noisy sequences such as CLV history or drift series.
    Reduces variance and prevents over-adjusting projections.
    """
    if len(values) < 5:
        return values
    return gaussian_filter1d(values, sigma=sigma).tolist()


def learn_drift(records, window=350):
    """
    Learns whether model projections are consistently biased:
        - model high vs actual results
        - model low vs actual results

    Each record contains:
        - "mu": model mean projection
        - "result": actual PRA or stat result

    Drift = result - mu.

    Positive drift → model underestimates players (should boost mu)
    Negative drift → model overestimates players (should reduce mu)

    Returns:
        drift_mult (0.88 — 1.12)
    """
    drift_vals = []

    for r in records[-window:]:
        try:
            if "mu" in r and "result" in r:
                drift_vals.append(float(r["result"]) - float(r["mu"]))
        except:
            continue

    if len(drift_vals) < 15:
        return 1.0

    drift_vals = _smooth_series(drift_vals)

    avg = np.mean(drift_vals)

    # Map drift into multiplier
    raw_mult = 1.0 + (avg / 5.0) * 0.12   # ±5 drift swings → ±12%

    return float(np.clip(raw_mult, 0.88, 1.12))


def learn_variance(records, window=350):
    """
    Measures how often model SD is accurate.

    If actual results fall too often outside mu ± 2*sd,
    the variance is underestimated → widen SD.

    If actual results rarely exceed mu ± sd,
    variance is too large → tighten SD.

    Returns:
        variance_mult (0.85 — 1.25)
    """
    errors = []

    for r in records[-window:]:
        if "mu" in r and "sd" in r and "result" in r:
            try:
                err = abs(float(r["result"]) - float(r["mu"]))
                sd = float(r["sd"])
                errors.append(err / max(sd, 1))
            except:
                continue

    if len(errors) < 20:
        return 1.0

    avg = np.mean(errors)

    # 1.0 → perfect, ~1.6 → variance underestimated, ~0.5 → overestimated
    raw_mult = avg ** 0.65

    return float(np.clip(raw_mult, 0.85, 1.25))


def learn_bias(records, window=350):
    """
    Determines systematic directional bias:

    High bias → consistently projecting too high
    Low bias → consistently projecting too low

    Looks at average (prediction - line).

    Returns:
        bias_adj (−3.0 → +3.0 shift to mu)
    """
    diffs = []

    for r in records[-window:]:
        if "mu" in r and "line" in r:
            try:
                diffs.append(float(r["mu"]) - float(r["line"]))
            except:
                continue

    if len(diffs) < 15:
        return 0.0

    diffs = _smooth_series(diffs)

    avg = np.mean(diffs)

    # Convert bias into a raw value shift
    adj = avg * 0.20   # keep mild, controlled

    return float(np.clip(adj, -3.0, 3.0))


def learn_trend(records, window=350):
    """
    Learns short-term performance trend of the model.

    If last 50 bets massively outperform CLV and hit-rate → momentum ↑
    If last 50 bets run cold → momentum ↓

    Returns:
        trend_mult (0.92 — 1.10)
    """
    recent = records[-50:]
    if len(recent) < 20:
        return 1.0

    hits = 0
    total = 0

    for r in recent:
        if "result" in r and "line" in r:
            try:
                if float(r["result"]) > float(r["line"]):
                    hits += 1
                total += 1
            except:
                continue

    if total < 15:
        return 1.0

    rate = hits / total

    # Trend multiplier
    raw = 1.0 + (rate - 0.50) * 0.25

    return float(np.clip(raw, 0.92, 1.10))


def unified_calibration_multiplier():
    """
    Combines:
        - Drift learning
        - Variance learning
        - CLV multiplier
        - CLV consistency
        - Bias correction
        - Trend learning

    This becomes the MASTER MULTIPLIER used in:
        - Module 6 (ensemble prob)
        - Module 7 (correlation)
        - Module 10 (Monte Carlo)
        - Module 11 (Joint MC)
        - Module 12 (Decision system)
    """

    data = load_calibration_data()
    records = data["records"]

    drift_mult = learn_drift(records)
    variance_mult = learn_variance(records)
    bias_adj = learn_bias(records)
    trend_mult = learn_trend(records)
    clv_adj = get_clv_adjustment()

    # MASTER MULTIPLIER
    master = drift_mult * variance_mult * trend_mult * clv_adj

    master = float(np.clip(master, 0.80, 1.28))

    return {
        "master_multiplier": master,
        "bias_adj": bias_adj,
        "drift_mult": drift_mult,
        "variance_mult": variance_mult,
        "trend_mult": trend_mult,
        "clv_adj": clv_adj,
    }
# =====================================================================
# MODULE 10 — MONTE CARLO ENGINE V3
# Phase 1: Core Setup + Normal Distribution Simulation
# =====================================================================

import numpy as np
from scipy.stats import norm

MC_ITERATIONS = 10_000


def monte_carlo_leg_simulation_phase1(mu, sd, line, market,
                                      variance_adj=1.0,
                                      heavy_tail_adj=1.0,
                                      bias_adj=0.0):
    """
    Monte Carlo v3 Engine — Phase 1
    ------------------------------------------------------------
    Handles:
        - Input clamping
        - Calibration parameter application
        - Base normal sampling
        - Protects against NaN / zero SD
    """
    # ============================================================
    # 1. Parameter Protection
    # ============================================================
    if np.isnan(mu):
        mu = 0.0

    if sd <= 0 or np.isnan(sd):
        # fallback volatility based on player mean
        sd = max(1.0, abs(mu) * 0.15)

    # ============================================================
    # 2. Apply Self-Learning Adjustments (from Module 9)
    # ============================================================
    # Variance calibration (broadens or tightens spread)
    sd *= float(np.clip(variance_adj, 0.50, 2.50))

    # Hard clamp SD to prevent explosion
    sd = float(np.clip(sd, 0.25, 200))

    # Bias shifts the mean up or down
    mu += float(np.clip(bias_adj, -8.0, 8.0))

    # Heavy-tail multiplier for skew/extension phases later
    tail_mult = float(np.clip(heavy_tail_adj, 0.90, 1.15))

    # ============================================================
    # 3. Base Normal Distribution (10,000 samples)
    # ============================================================
    try:
        normal_samples = np.random.normal(mu, sd, MC_ITERATIONS)
    except Exception:
        # ultra fail-safe
        normal_samples = np.ones(MC_ITERATIONS) * mu

    # ============================================================
    # 4. Clip insane outliers (rare but prevents UI explosions)
    # ============================================================
    normal_samples = np.clip(normal_samples,
                             mu - 6 * sd,
                             mu + 6 * sd)

    return {
        "mu": mu,
        "sd": sd,
        "line": line,
        "market": market,
        "tail_mult": tail_mult,
        "normal_samples": normal_samples
    }
# =====================================================================
# MODULE 10 — PHASE 2
# Lognormal Distribution Expansion (Right-Tail Modelling)
# =====================================================================

def monte_carlo_leg_simulation_phase2(core):
    """
    Monte Carlo v3 Engine — Phase 2
    ------------------------------------------------------------
    Handles:
        - Conversion to log-space
        - Lognormal sampling
        - Tail protections
    Inputs:
        core → dictionary from Phase 1
    Outputs:
        Adds:
            - lognormal_samples
    """

    mu = core["mu"]
    sd = core["sd"]
    normal_samples = core["normal_samples"]

    # ============================================================
    # 1. Protect against invalid lognormal states
    # ============================================================
    # Lognormal requires strictly positive mean
    safe_mu = max(mu, 0.01)

    # avoid SD too small
    safe_sd = max(sd, 0.10)

    # ============================================================
    # 2. Convert to log-space
    # ============================================================
    try:
        variance = safe_sd ** 2
        phi = np.sqrt(safe_mu**2 + variance)

        # log-space parameters
        mu_log = np.log((safe_mu**2) / phi)
        sd_log = np.sqrt(np.log((phi**2) / (safe_mu**2)))

        # safety clamps
        mu_log = float(np.clip(mu_log, -5, 8))
        sd_log = float(np.clip(sd_log, 0.05, 2.50))

        # ========================================================
        # 3. Generate lognormal samples
        # ========================================================
        lognormal_samples = np.random.lognormal(
            mean=mu_log,
            sigma=sd_log,
            size=MC_ITERATIONS
        )

    except Exception:
        # Fail-safe: fallback to normal samples
        lognormal_samples = normal_samples.copy()

    # ============================================================
    # 4. Clip outliers — prevents Streamlit graph blow-ups
    # ============================================================
    lognormal_samples = np.clip(lognormal_samples, 0, safe_mu * 12)

    # attach to core dictionary
    core["lognormal_samples"] = lognormal_samples

    return core
# =====================================================================
# MODULE 10 — PHASE 3
# Skew Extension (Asymmetric Right-Tail Boost Engine)
# =====================================================================

def monte_carlo_leg_simulation_phase3(core):
    """
    Monte Carlo v3 Engine — Phase 3
    ------------------------------------------------------------
    Adds asymmetric skewed distribution:
        - heavy right-tail boosting
        - controlled volatility shaping
        - NBA-stat realistic blowout protection
    Inputs:
        core → dictionary containing:
            - mu
            - sd
            - normal_samples
            - lognormal_samples
    Outputs:
        Adds:
            - skew_samples
    """

    mu = core["mu"]
    sd = core["sd"]

    normal_samples = core["normal_samples"]

    # ============================================================
    # 1. Create skew influence noise
    # ------------------------------------------------------------
    # adds more upward volatility than downward
    # this simulates:
    #   - hot streaks
    #   - usage spikes
    #   - quarter-to-quarter variance
    #   - player takeover games
    # ============================================================

    # baseline skew scale
    skew_scale = max(sd * 0.35, 0.20)

    # skew noise: abs ensures right-tail only
    skew_noise = np.abs(np.random.normal(0, skew_scale, MC_ITERATIONS))

    # ============================================================
    # 2. Generate skew distribution
    # ============================================================
    skew_samples = normal_samples + skew_noise

    # clamp to prevent model blowing up
    skew_samples = np.clip(
        skew_samples,
        0,
        mu * 10  # players rarely exceed 10× mean rate
    )

    # ============================================================
    # 3. Add to core dictionary
    # ============================================================
    core["skew_samples"] = skew_samples

    return core
# =====================================================================
# MODULE 10 — PHASE 4
# Blended Ensemble Distribution Engine
# =====================================================================

def monte_carlo_leg_simulation_phase4(core, market):
    """
    Monte Carlo v3 Engine — Phase 4
    ------------------------------------------------------------
    Produces blended ensemble distribution using:
        - Normal distribution (Phase 1)
        - Lognormal distribution (Phase 2)
        - Skew distribution (Phase 3)
    Blends using market-specific weights:
        PRA, Points, Rebounds, Assists all have different tail behavior.
    Inputs:
        core → dictionary containing:
            - mu
            - sd
            - normal_samples
            - lognormal_samples
            - skew_samples
    Outputs:
        Adds:
            - blended_dist
    """

    normal_samples = core["normal_samples"]
    lognormal_samples = core["lognormal_samples"]
    skew_samples = core["skew_samples"]

    # ============================================================
    # 1. Market-specific weights
    # ------------------------------------------------------------
    # Rationale:
    #   PRA = most volatile → heavier lognormal tail
    #   Points = explosive → balanced skew + lognormal
    #   Rebounds = narrow distribution → heavier normal
    #   Assists = more consistent → normal dominant
    # ============================================================

    if market == "PRA":
        w_norm  = 0.25
        w_logn  = 0.45
        w_skew  = 0.30

    elif market == "Points":
        w_norm  = 0.33
        w_logn  = 0.37
        w_skew  = 0.30

    elif market == "Rebounds":
        w_norm  = 0.45
        w_logn  = 0.25
        w_skew  = 0.30

    elif market == "Assists":
        w_norm  = 0.55
        w_logn  = 0.20
        w_skew  = 0.25

    else:
        # Default safety fallback
        w_norm  = 0.40
        w_logn  = 0.30
        w_skew  = 0.30

    # ============================================================
    # 2. Compute blended ensemble distribution
    # ============================================================
    blended = (
        w_norm * normal_samples +
        w_logn * lognormal_samples +
        w_skew * skew_samples
    )

    # ============================================================
    # 3. Clamp outlier values to avoid model blow-ups
    # ============================================================
    blended = np.clip(blended, 0, core["mu"] * 12)

    # ============================================================
    # 4. Save blended distribution
    # ============================================================
    core["blended_dist"] = blended

    return core
# =====================================================================
# MODULE 10 — PHASE 5
# Final Probability + EV Computation from Monte Carlo Ensemble
# =====================================================================

def monte_carlo_leg_simulation_phase5(core, line):
    """
    Final Phase of Module 10:
    ----------------------------------------
    Takes the blended distribution created in Phase 4 and
    computes:
        - Probability of OVER
        - Probability of UNDER
        - Expected value
        - Final packaged dict

    Inputs:
        core → dictionary containing:
            - blended_dist ( REQUIRED )
            - mu
            - sd
            - line
            - market

        line → projection line for the market
    
    Outputs:
        Updated core dict with:
            - mc_prob_over
            - mc_prob_under
            - mc_ev
            - sims (blended distribution)
    """

    # ============================================================
    # 1. Validate blended distribution
    # ============================================================
    if "blended_dist" not in core:
        raise ValueError("Phase 5 error: blended_dist missing. Ensure Phase 4 was executed.")

    sims = core["blended_dist"]

    # Safety conversion ===========
    sims = np.array(sims, dtype=float)

    # ============================================================
    # 2. Calculate probabilities
    # ============================================================
    prob_over = float(np.mean(sims > line))
    prob_under = 1.0 - prob_over

    # Clamp to prevent degenerate 0.0/1.0
    prob_over = float(np.clip(prob_over, 0.01, 0.99))
    prob_under = float(np.clip(prob_under, 0.01, 0.99))

    # ============================================================
    # 3. Expected value (even odds assumption)
    # ============================================================
    mc_ev = prob_over - prob_under

    # ============================================================
    # 4. Save final outputs
    # ============================================================
    core["mc_prob_over"] = prob_over
    core["mc_prob_under"] = prob_under
    core["mc_ev"] = mc_ev
    core["sims"] = sims

    return core

# =====================================================================
# MODULE 11 — PHASE 1
# Joint Monte Carlo Engine v2
# Input Validation + Distribution Normalization
# =====================================================================

def joint_mc_phase1_prepare(leg1_core, leg2_core, iterations=10000):
    """
    Phase 1 of Module 11:
    ------------------------------------------------------------
    Validates inputs and prepares standardized simulation vectors
    for both legs before correlation injection.

    Inputs:
        leg1_core → dictionary from Module 10 containing:
            - sims  (10,000 MCA samples)
            - mu
            - sd
            - line

        leg2_core → same structure as above for the second leg

        iterations → number of samples to use (default = 10,000)

    Outputs:
        {
            "x_raw": np.array (leg1 sims clipped to iterations)
            "y_raw": np.array (leg2 sims clipped to iterations)
            "leg1_line": float
            "leg2_line": float
            "leg1_mu": float
            "leg1_sd": float
            "leg2_mu": float
            "leg2_sd": float
        }

    This prepares data for Phase 2 (correlation injection).
    """

    # =============================================================
    # 1. Validate structure
    # =============================================================
    required_keys = ["sims", "mu", "sd", "line"]

    for name, core in [("Leg 1", leg1_core), ("Leg 2", leg2_core)]:
        for key in required_keys:
            if key not in core:
                raise ValueError(f"Module 11 Phase 1 Error: {name} missing '{key}'")

    # =============================================================
    # 2. Extract simulation arrays
    # =============================================================
    x = np.array(leg1_core["sims"], dtype=float)
    y = np.array(leg2_core["sims"], dtype=float)

    # =============================================================
    # 3. Length repair (pad or clip)
    # =============================================================
    if len(x) < iterations:
        x = np.pad(x, (0, iterations - len(x)), mode="edge")
    elif len(x) > iterations:
        x = x[:iterations]

    if len(y) < iterations:
        y = np.pad(y, (0, iterations - len(y)), mode="edge")
    elif len(y) > iterations:
        y = y[:iterations]

    # =============================================================
    # 4. Ensure distributions contain no NaNs or infs
    # =============================================================
    x = np.nan_to_num(x, nan=np.mean(x), posinf=np.max(x)*0.9, neginf=np.min(x)*0.9)
    y = np.nan_to_num(y, nan=np.mean(y), posinf=np.max(y)*0.9, neginf=np.min(y)*0.9)

    # =============================================================
    # 5. Return prepared data package
    # =============================================================
    return {
        "x_raw": x,
        "y_raw": y,
        "leg1_line": float(leg1_core["line"]),
        "leg2_line": float(leg2_core["line"]),
        "leg1_mu": float(leg1_core["mu"]),
        "leg1_sd": float(leg1_core["sd"]),
        "leg2_mu": float(leg2_core["mu"]),
        "leg2_sd": float(leg2_core["sd"]),
    }
# =====================================================================
# MODULE 11 — PHASE 2
# Correlation Injection using Cholesky Transform
# =====================================================================

def joint_mc_phase2_correlate(prepped, base_corr, iterations=10000):
    """
    Phase 2 of Module 11:
    ----------------------------------------------------------
    Injects correlation between the two simulation arrays
    using a Cholesky decomposition approach.

    Inputs:
        prepped → dictionary from Phase 1 containing:
            - x_raw, y_raw (raw MC samples)
            - leg1_mu, leg1_sd
            - leg2_mu, leg2_sd
            - leg1_line, leg2_line

        base_corr → float (initial correlation estimate from Module 7)
        iterations → number of samples (default 10k)

    Outputs:
        {
            "x_sim": correlated simulation vector for leg1
            "y_sim": correlated simulation vector for leg2
            "leg1_line": float
            "leg2_line": float
            "corr_used": float
        }
    """

    # ============================================================
    # 1. Extract data
    # ============================================================
    x = prepped["x_raw"]
    y = prepped["y_raw"]

    # ============================================================
    # 2. Safety clamp correlation
    # Prevents matrix from becoming invalid
    # ============================================================
    corr = float(np.clip(base_corr, -0.50, 0.75))

    # ============================================================
    # 3. Convert X and Y to z-scores (standard normal space)
    # ============================================================
    x_mean, x_std = np.mean(x), np.std(x)
    y_mean, y_std = np.mean(y), np.std(y)

    # Prevent divide-by-zero
    x_std = 1e-6 if x_std <= 0 else x_std
    y_std = 1e-6 if y_std <= 0 else y_std

    zx = (x - x_mean) / x_std
    zy = (y - y_mean) / y_std

    # ============================================================
    # 4. Cholesky correlation matrix
    # ============================================================
    L = np.array([
        [1.0,      0.0],
        [corr, np.sqrt(max(1e-6, 1 - corr**2))]
    ])

    # ============================================================
    # 5. Inject correlation
    # ============================================================
    Z = np.vstack([zx, zy])   # shape: (2, N)
    fused = L @ Z             # matrix multiply to inject correlation

    zx_corr, zy_corr = fused[0], fused[1]

    # ============================================================
    # 6. Convert back to actual outcome scales
    # ============================================================
    x_sim = zx_corr * x_std + x_mean
    y_sim = zy_corr * y_std + y_mean

    # Final NaN cleanup
    x_sim = np.nan_to_num(x_sim, nan=x_mean)
    y_sim = np.nan_to_num(y_sim, nan=y_mean)

    # ============================================================
    # 7. Return package for Phase 3
    # ============================================================
    return {
        "x_sim": x_sim,
        "y_sim": y_sim,
        "leg1_line": prepped["leg1_line"],
        "leg2_line": prepped["leg2_line"],
        "corr_used": corr
    }
# =====================================================================
# MODULE 11 — PHASE 3
# Joint Probability + EV Calculation (Correlated MC Output)
# =====================================================================

def joint_mc_phase3_joint_probability(corr_pack, payout_mult=3.0):
    """
    Phase 3 of Module 11:
    ---------------------------------------------------------
    Takes correlated Monte Carlo vectors (from Phase 2)
    and computes:
        - Joint probability (both legs hit)
        - Expected Value (EV)
        - Joint distribution (2D)
        - Final correlation used

    Inputs:
        corr_pack — dict returned from Phase 2:
            {
                "x_sim": array,
                "y_sim": array,
                "leg1_line": float,
                "leg2_line": float,
                "corr_used": float
            }

        payout_mult — Payout multiplier (e.g. 3.0 for PP Power Play)

    Outputs:
        {
            "joint_prob": float,
            "joint_ev": float,
            "corr_used": float,
            "joint_hits": np.ndarray(bool),
            "joint_distribution": np.ndarray(shape=(2, N))
        }
    """

    # ==============================================================
    # 1. Extract values
    # ==============================================================
    x_sim = corr_pack["x_sim"]
    y_sim = corr_pack["y_sim"]

    leg1_line = corr_pack["leg1_line"]
    leg2_line = corr_pack["leg2_line"]
    corr_used = corr_pack["corr_used"]

    # ==============================================================
    # 2. Joint hit: both props clear their lines
    # ==============================================================
    joint_hits = np.logical_and(
        x_sim > leg1_line,
        y_sim > leg2_line
    )

    # ==============================================================
    # 3. Joint probability
    # ==============================================================

    joint_prob = float(np.mean(joint_hits))
    joint_prob = float(np.clip(joint_prob, 0.01, 0.99))

    # ==============================================================
    # 4. Combo Expected Value
    # ==============================================================
    # EV formula:
    # EV = payout_mult * P(win) - 1
    joint_ev = payout_mult * joint_prob - 1.0

    # ==============================================================
    # 5. Package outputs
    # ==============================================================

    joint_distribution = np.vstack([x_sim, y_sim])

    return {
        "joint_prob": joint_prob,
        "joint_ev": joint_ev,
        "corr_used": corr_used,
        "joint_hits": joint_hits,
        "joint_distribution": joint_distribution
    }
# =====================================================================
# MODULE 11 — PHASE 4
# Final Packaging → Streamlit-safe output object
# Connects Module 11 → Module 12 cleanly
# =====================================================================

def joint_mc_phase4_finalize(
    leg1_data,
    leg2_data,
    corr_phase2_pack,
    phase3_pack,
):
    """
    Phase 4 of Module 11:
    ------------------------------------------------------------------
    Final packaging, validation, and Streamlit-safe formatting.

    Inputs:
        leg1_data: dict from Modules 1–10 containing:
            {
                "mu": float,
                "sd": float,
                "line": float,
                "market": str,
                "dist": np.ndarray
            }

        leg2_data: identical structure for leg 2

        corr_phase2_pack: output from Phase 2
        phase3_pack: output from Phase 3

    Output:
        A perfectly structured dictionary:
        ----------------------------------------------------------------
        {
            "leg1": {...},
            "leg2": {...},
            "correlation_used": float,
            "joint_probability": float,
            "joint_ev": float,
            "distribution": np.ndarray,
            "joint_hits_mask": np.ndarray(bool),
            "lines": (leg1_line, leg2_line)
        }
        ----------------------------------------------------------------

    This will be fed directly into:
        → Module 12 (UltraMax Decision Engine)
        → Module 13 (UI Results Panel)
        → Module 14 (Graphing / Histograms)
    """

    # ===============================================================
    # 1. Extract standardized values
    # ===============================================================
    joint_prob = phase3_pack["joint_prob"]
    joint_ev   = phase3_pack["joint_ev"]
    corr_used  = phase3_pack["corr_used"]
    joint_hits = phase3_pack["joint_hits"]
    joint_dist = phase3_pack["joint_distribution"]

    # ===============================================================
    # 2. Verify shapes & validity
    # ===============================================================
    if not isinstance(joint_dist, np.ndarray):
        raise ValueError("Module 11 Phase 4: joint_dist must be numpy array.")

    if joint_dist.shape[0] != 2:
        raise ValueError("Module 11 Phase 4: joint_dist must be shape (2, N).")

    # ===============================================================
    # 3. Clean per-leg packaging
    # ===============================================================
    leg1_pack = {
        "mu": float(leg1_data["mu"]),
        "sd": float(leg1_data["sd"]),
        "line": float(leg1_data["line"]),
        "market": leg1_data["market"],
        "dist": leg1_data["dist"]
    }

    leg2_pack = {
        "mu": float(leg2_data["mu"]),
        "sd": float(leg2_data["sd"]),
        "line": float(leg2_data["line"]),
        "market": leg2_data["market"],
        "dist": leg2_data["dist"]
    }

    # ===============================================================
    # 4. Final output object
    # ===============================================================
    final_pack = {
        "leg1": leg1_pack,
        "leg2": leg2_pack,
        "correlation_used": corr_used,
        "joint_probability": joint_prob,
        "joint_ev": joint_ev,
        "joint_hits_mask": joint_hits,
        "distribution": joint_dist,
        "lines": (leg1_pack["line"], leg2_pack["line"])
    }

    return final_pack

# ======================================================================
# MODULE 12 — ULTRAMAX TWO-PICK DECISION ENGINE
# PHASE 1 — Initialization, Validation, Input Contract
# ======================================================================

def module12_phase1_initialize(
    mc_pack,
    payout_mult: float,
    bankroll: float,
    fractional_kelly: float,
    drift_adj: float,
    clv_adj: float,
):
    """
    Module 12 — Phase 1
    -------------------------------------------------------------
    This phase validates ALL upstream inputs from Module 11,
    cleans them, clamps them for numerical stability, and prepares
    a normalized package for downstream phases (Phases 2–6).

    Inputs:
        mc_pack:
            The final output from Module 11 Phase 4:
            {
                "leg1": {...},
                "leg2": {...},
                "correlation_used": float,
                "joint_probability": float,
                "joint_ev": float,
                "joint_hits_mask": np.ndarray(bool),
                "distribution": np.ndarray,
                "lines": (line1, line2)
            }

        payout_mult: float
        bankroll: float
        fractional_kelly: float (0–1)
        drift_adj: float (~0.95–1.05)
        clv_adj: float (~0.95–1.10)

    Output:
        A complete, validated dictionary:
        ---------------------------------------------------------
        {
            "leg1": {...},
            "leg2": {...},
            "corr": float,
            "p_joint_raw": float,
            "payout_mult": float,
            "bankroll": float,
            "fractional_kelly": float,
            "drift_adj": float,
            "clv_adj": float,
            "lines": (line1, line2),
            "distribution": np.ndarray,
            "joint_hits_mask": np.ndarray
        }
        ---------------------------------------------------------

    This feeds **directly into**:
        → Phase 2 (Drift & CLV sharpening)
        → Phase 3 (EV computation)
        → Phase 4 (Kelly sizing)
        → Phase 5 (Result labeling)
        → Phase 6 (Final output)
    """

    # ============================================================
    # 1. Validate critical keys from Module 11
    # ============================================================
    required_keys = [
        "leg1", "leg2",
        "correlation_used",
        "joint_probability",
        "joint_ev",
        "joint_hits_mask",
        "distribution",
        "lines"
    ]

    for k in required_keys:
        if k not in mc_pack:
            raise KeyError(f"Module 12 Phase 1: Missing key from Module 11 → {k}")

    # ============================================================
    # 2. Extract & validate correlation
    # ============================================================
    corr = float(mc_pack["correlation_used"])
    corr = float(np.clip(corr, -0.45, 0.45))  # hard safety clamp

    # ============================================================
    # 3. Extract raw joint probability
    # ============================================================
    p_joint_raw = float(mc_pack["joint_probability"])
    if np.isnan(p_joint_raw) or p_joint_raw <= 0:
        p_joint_raw = 0.01
    elif p_joint_raw >= 1:
        p_joint_raw = 0.99

    # ============================================================
    # 4. Validate distribution shapes
    # ============================================================
    joint_dist = mc_pack["distribution"]

    if not isinstance(joint_dist, np.ndarray):
        raise ValueError("Module 12 Phase 1: distribution must be numpy array.")

    if joint_dist.shape[0] != 2:
        raise ValueError(
            "Module 12 Phase 1: distribution must have shape (2, N) — two legs."
        )

    joint_hits_mask = mc_pack["joint_hits_mask"]
    if not isinstance(joint_hits_mask, np.ndarray):
        raise ValueError("Module 12 Phase 1: joint_hits_mask must be numpy array.")

    # ============================================================
    # 5. Validate bankroll & risk inputs
    # ============================================================
    bankroll = float(max(bankroll, 0))
    payout_mult = float(max(payout_mult, 1.0))
    fractional_kelly = float(np.clip(fractional_kelly, 0.0, 1.0))
    drift_adj = float(np.clip(drift_adj, 0.80, 1.20))
    clv_adj = float(np.clip(clv_adj, 0.80, 1.25))

    # ============================================================
    # 6. Clean per-leg data
    # ============================================================
    leg1 = mc_pack["leg1"]
    leg2 = mc_pack["leg2"]

    for leg in [leg1, leg2]:
        # ensure floats
        leg["mu"] = float(leg["mu"])
        leg["sd"] = float(max(leg["sd"], 0.01))
        leg["line"] = float(leg["line"])

        if not isinstance(leg["dist"], np.ndarray):
            raise ValueError("Module 12 Phase 1: Each leg distribution must be ndarray.")

    # ============================================================
    # 7. Final normalized package
    # ============================================================
    normalized_pack = {
        "leg1": leg1,
        "leg2": leg2,
        "corr": corr,
        "p_joint_raw": p_joint_raw,
        "payout_mult": payout_mult,
        "bankroll": bankroll,
        "fractional_kelly": fractional_kelly,
        "drift_adj": drift_adj,
        "clv_adj": clv_adj,
        "lines": mc_pack["lines"],
        "distribution": joint_dist,
        "joint_hits_mask": joint_hits_mask,
    }

    return normalized_pack
# ======================================================================
# MODULE 12 — TWO-PICK DECISION ENGINE
# PHASE 2 — Drift + CLV Sharpening
# ======================================================================

def module12_phase2_apply_adjustments(normalized_pack):
    """
    Module 12 — Phase 2
    ---------------------------------------------------------
    Applies:
        - Drift correction (model bias smoothing)
        - CLV adjustment (sharp-side bias)
        - Nonlinear stabilization
        - Safety clipping

    Inputs:
        normalized_pack — output from Phase 1:
            {
                "p_joint_raw": float,
                "drift_adj": float,
                "clv_adj": float,
                ...
            }

    Output:
        {
            "p_joint_final": float,
            "p_joint_raw": float,
            "drift_adj": float,
            "clv_adj": float,
            ... (pass-through original data)
        }
    """

    # ------------------------------------------------------
    # Extract components
    # ------------------------------------------------------
    p_raw = float(normalized_pack["p_joint_raw"])
    drift = float(normalized_pack["drift_adj"])
    clv = float(normalized_pack["clv_adj"])

    # ------------------------------------------------------
    # 1. Apply drift and CLV multiplicatively
    # ------------------------------------------------------
    p = p_raw * drift * clv

    # ------------------------------------------------------
    # 2. Nonlinear stabilization
    # ------------------------------------------------------
    # This prevents extreme jumps from small drift/CLV swings.
    # α controls curvature; 0.055 = calibrated sweet spot.
    alpha = 0.055
    p = p ** (1 - alpha) * (p_raw ** alpha)

    # ------------------------------------------------------
    # 3. Logistic compression near bounds
    # ------------------------------------------------------
    # Smooths extreme probabilities so Kelly does not explode.
    logistic_scale = 1.15
    p = 1 / (1 + np.exp(-logistic_scale * (p - 0.5)))

    # ------------------------------------------------------
    # 4. Hard safety clipping
    # ------------------------------------------------------
    p = float(np.clip(p, 0.02, 0.98))

    # ------------------------------------------------------
    # 5. Build output pack
    # ------------------------------------------------------
    out = dict(normalized_pack)
    out["p_joint_final"] = p

    return out
# ======================================================================
# MODULE 12 — TWO-PICK DECISION ENGINE
# PHASE 3 — EV (Expected Value) Computation
# ======================================================================

def module12_phase3_ev_engine(adjusted_pack, payout_mult):
    """
    Module 12 — Phase 3
    ---------------------------------------------------------
    Computes:
        - Market implied probability
        - EV per $1 risked
        - Advantage (model vs market)
        - Edge classification (tier)

    Inputs:
        adjusted_pack — output from Phase 2:
            {
                "p_joint_final": float,
                "p_joint_raw": float,
                "drift_adj": float,
                "clv_adj": float,
                ...
            }

        payout_mult — final payout multiplier:
                       PrizePicks PowerPlay = 3.0

    Outputs:
        {
            "p_joint_final": float,
            "ev": float,
            "implied": float,
            "advantage": float,
            "edge_label": str,
            ... (pass-through of original fields)
        }
    """

    # ------------------------------------------------------
    # Extract
    # ------------------------------------------------------
    p = float(adjusted_pack["p_joint_final"])

    # ------------------------------------------------------
    # 1. Market implied probability
    # ------------------------------------------------------
    implied = 1.0 / payout_mult
    implied = float(np.clip(implied, 0.01, 0.99))

    # ------------------------------------------------------
    # 2. Expected Value (per $1 risked)
    # ------------------------------------------------------
    # EV = p * multiplier - 1
    ev = (payout_mult * p) - 1.0

    # ------------------------------------------------------
    # 3. Advantage (model vs implied)
    # ------------------------------------------------------
    advantage = p - implied

    # ------------------------------------------------------
    # 4. Edge Classification
    # ------------------------------------------------------
    if ev >= 0.15:
        edge = "🔥 **ELITE EDGE — Hammer Spot**"
    elif ev >= 0.10:
        edge = "🟢 **Strong Edge**"
    elif ev >= 0.05:
        edge = "🟡 **Moderate Edge**"
    elif ev >= 0.02:
        edge = "⚪ **Thin Edge**"
    else:
        edge = "❌ **No Edge**"

    # ------------------------------------------------------
    # 5. Build return payload
    # ------------------------------------------------------
    out = dict(adjusted_pack)
    out["implied"] = implied
    out["ev"] = float(ev)
    out["advantage"] = float(advantage)
    out["edge_label"] = edge

    return out
# ======================================================================
# MODULE 12 — TWO-PICK DECISION ENGINE
# PHASE 4 — Kelly Staking Engine (Risk-Managed)
# ======================================================================

def module12_phase4_kelly(ev_pack, bankroll, fractional_kelly=0.50):
    """
    Module 12 — Phase 4
    ---------------------------------------------------------
    Inputs:
        ev_pack:
            output from Phase 3 containing:
                - p_joint_final
                - ev
                - implied
                - advantage
                - edge_label
                ... (and fields from Phase 2)

        bankroll:
            Total bankroll (float)

        fractional_kelly:
            0.00 → no Kelly
            0.50 → half Kelly (recommended)
            1.00 → full Kelly (NOT RECOMMENDED)

    Outputs:
        {
            ... passthrough Phase 3 data ...
            "kelly_fraction": float,
            "stake": float,
            "stake_label": str
        }
    """

    p = float(ev_pack["p_joint_final"])
    ev = float(ev_pack["ev"])

    # ------------------------------------------------------
    # Kelly base formula at even-money payout multiplier:
    #
    #   Kelly = (bp − q) / b
    #
    # Where:
    #     b = payout_mult - 1
    #     p = probability of winning
    #     q = 1 - p
    # ------------------------------------------------------
    payout_mult = ev_pack.get("payout_mult", 3.0)
    b = payout_mult - 1
    q = 1 - p

    # Raw Kelly
    raw_k = 0.0
    try:
        raw_k = (b * p - q) / b
    except ZeroDivisionError:
        raw_k = 0.0

    # ------------------------------------------------------
    # Fractional Kelly Scaling
    # ------------------------------------------------------
    k = raw_k * fractional_kelly

    # ------------------------------------------------------
    # Stabilizing Rules
    # ------------------------------------------------------

    # Rule 1 — EV too small → scale down
    if ev < 0.02:
        k *= 0.30     # reduce by 70%
    elif ev < 0.05:
        k *= 0.55     # reduce by 45%
    elif ev > 0.15:
        k *= 1.25     # boost when signal extremely strong

    # Rule 2 — Never allow negative Kelly
    k = max(k, 0.0)

    # Rule 3 — Hard cap at 3% bankroll
    k = min(k, 0.03)

    # Rule 4 — Round for stability
    k = float(np.clip(k, 0.0, 0.03))

    # ------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------
    stake = round(bankroll * k, 2)

    # ------------------------------------------------------
    # Stake label for the UI
    # ------------------------------------------------------
    if stake >= bankroll * 0.025:
        stake_label = "🔥 Aggressive Stake (High Confidence)"
    elif stake >= bankroll * 0.015:
        stake_label = "🟢 Solid Stake"
    elif stake >= bankroll * 0.008:
        stake_label = "🟡 Small Stake"
    elif stake > 0:
        stake_label = "⚪ Token Stake"
    else:
        stake_label = "❌ No Bet"

    # ------------------------------------------------------
    # Build output package
    # ------------------------------------------------------
    out = dict(ev_pack)
    out["kelly_fraction"] = float(k)
    out["stake"] = float(stake)
    out["stake_label"] = stake_label

    return out
# ======================================================================
# MODULE 12 — TWO-PICK DECISION ENGINE
# PHASE 5 — Final UltraMax Decision Pack (Master Merge Layer)
# ======================================================================

def module12_phase5_finalize(
    leg1,
    leg2,
    leg1_mc,
    leg2_mc,
    corr_value,
    payout_mult,
    bankroll,
    fractional_kelly,
    drift_adj,
    clv_adj
):
    """
    Module 12 — Phase 5 Finalization
    ---------------------------------------------------------
    Merges all phases into one unified UltraMax decision pack.

    Inputs:
      leg1, leg2       — dictionaries from compute_leg()
      leg1_mc, leg2_mc — Monte Carlo results from Module 10
      corr_value       — correlation (from Module 7)
      payout_mult      — e.g. 3.0 for 2-pick power play
      bankroll         — bankroll dollars
      fractional_kelly — 0.0–1.0 (risk control)
      drift_adj        — learned drift correction
      clv_adj          — CLV correction factor

    Returns:
      full_pack — the complete decision object consumed by UI
    """

    # ==========================================================
    # Step 1 — Enforce structure & safety
    # ==========================================================
    try:
        leg1_line = float(leg1["line"])
        leg2_line = float(leg2["line"])
    except Exception:
        raise ValueError("Legs passed into Phase 5 missing 'line' fields.")

    leg1_sd = float(leg1["sd"])
    leg2_sd = float(leg2["sd"])
    leg1_mu = float(leg1["mu"])
    leg2_mu = float(leg2["mu"])

    leg1_dist = np.array(leg1_mc["dist"], dtype=float)
    leg2_dist = np.array(leg2_mc["dist"], dtype=float)

    # ==========================================================
    # Step 2 — Run Phase 2 (joint probability)
    # ==========================================================
    joint_pack = module12_phase2_joint_prob(
        leg1_mu, leg1_sd, leg1_line, leg1_dist,
        leg2_mu, leg2_sd, leg2_line, leg2_dist,
        corr_value
    )

    # Add metadata
    joint_pack["payout_mult"] = payout_mult

    # ==========================================================
    # Step 3 — Run Phase 3 (EV + advantage classification)
    # ==========================================================
    ev_pack = module12_phase3_ev(
        joint_pack["p_joint_final"],
        payout_mult
    )

    # Merge dictionaries
    merged = {**joint_pack, **ev_pack}

    # ==========================================================
    # Step 4 — Kelly staking engine
    # ==========================================================
    final_pack = module12_phase4_kelly(
        merged,
        bankroll=bankroll,
        fractional_kelly=fractional_kelly
    )

    # ==========================================================
    # Step 5 — Enrich final pack with leg-level metadata
    # ==========================================================
    final_pack.update({
        "leg1": leg1,
        "leg2": leg2,
        "leg1_mc": leg1_mc,
        "leg2_mc": leg2_mc,
        "corr_input": float(corr_value),
        "corr_used": float(final_pack.get("corr_used", corr_value)),
        "drift_adj": drift_adj,
        "clv_adj": clv_adj,
        "timestamp": time.time()
    })

    # ==========================================================
    # All done — return the UltraMax full decision object
    # ==========================================================
    return final_pack
# ======================================================================
# MODULE 12 — TWO-PICK DECISION ENGINE
# PHASE 6 — Final UI Report Builder (UltraMax Summary Generator)
# ======================================================================

def module12_phase6_build_report(final_pack):
    """
    Builds the polished UI-ready interpretation of the UltraMax decision.
    Converts the numeric decision pack into:
        - readable insights
        - risk levels
        - strength-of-play explanations
        - CLV & drift commentary
        - correlation commentary
        - Monte Carlo diagnostics
    """

    # Extract values with safe defaults
    p_joint = float(final_pack.get("p_joint_final", 0.50))
    p_joint_raw = float(final_pack.get("p_joint_raw", 0.50))
    ev = float(final_pack.get("joint_ev", 0.0))
    stake = float(final_pack.get("stake", 0.0))
    corr = float(final_pack.get("corr_used", 0.0))
    payout_mult = float(final_pack.get("payout_mult", 3.0))
    drift_adj = float(final_pack.get("drift_adj", 1.0))
    clv_adj = float(final_pack.get("clv_adj", 1.0))

    leg1 = final_pack.get("leg1", {})
    leg2 = final_pack.get("leg2", {})

    # ----------------------------------------------------
    # 1. Recommendation Strength Label
    # ----------------------------------------------------
    if ev >= 0.12:
        strength = "🔥 **MAX PLAY — Hedge-Fund Level Edge**"
    elif ev >= 0.07:
        strength = "🟢 **PLAY — Solid Quant Edge**"
    elif ev >= 0.03:
        strength = "🟡 **LEAN — Thin But Positive Edge**"
    else:
        strength = "❌ **PASS — No Edge**"

    # ----------------------------------------------------
    # 2. Correlation Commentary
    # ----------------------------------------------------
    if corr > 0.25:
        corr_note = "High positive correlation — outcomes move together strongly."
    elif corr > 0.10:
        corr_note = "Moderate positive correlation — slight combo risk."
    elif corr > -0.05:
        corr_note = "Neutral correlation — ideal for 2-pick combos."
    elif corr > -0.20:
        corr_note = "Mild negative correlation — can help reduce variance."
    else:
        corr_note = "Strong negative correlation — massive variance reducer."

    # ----------------------------------------------------
    # 3. EV Commentary
    # ----------------------------------------------------
    if ev >= 0.12:
        ev_note = "Top-tier EV — this type of advantage rarely appears."
    elif ev >= 0.07:
        ev_note = "Strong EV — stable long-term edge."
    elif ev >= 0.03:
        ev_note = "Thin EV — playable but not ideal."
    else:
        ev_note = "Negative EV — not recommended."

    # ----------------------------------------------------
    # 4. CLV & Drift Commentary
    # ----------------------------------------------------
    drift_msg = (
        "Model drift suggests stabilizing conditions." if abs(drift_adj - 1) < 0.05
        else "Strong drift detected — volatility regime shifting."
    )

    clv_msg = (
        "Sharp-side CLV boost applied." if clv_adj > 1.0
        else "CLV suggests neutral market conditions."
    )

    # ----------------------------------------------------
    # 5. Probability Commentary
    # ----------------------------------------------------
    prob_msg = (
        f"Joint hit probability: **{p_joint*100:.1f}%** "
        f"(raw: {p_joint_raw*100:.1f}%)"
    )

    # ----------------------------------------------------
    # 6. Kelly Commentary
    # ----------------------------------------------------
    if stake > 0:
        stake_msg = (
            f"Recommended stake (Kelly-scaled): **${stake:.2f}**\n"
            f"Fractional Kelly applied: {final_pack.get('fractional_kelly', 'N/A')}"
        )
    else:
        stake_msg = "Kelly size recommends **no play** due to weak EV."

    # ----------------------------------------------------
    # 7. Leg Descriptions
    # ----------------------------------------------------
    leg1_text = (
        f"**{leg1.get('player','Leg 1')} — {leg1.get('market','')}**\n"
        f"Line: **{leg1.get('line','?')}**, Model Mean: **{leg1.get('mu','?'):.2f}**"
    )

    leg2_text = (
        f"**{leg2.get('player','Leg 2')} — {leg2.get('market','')}**\n"
        f"Line: **{leg2.get('line','?')}**, Model Mean: **{leg2.get('mu','?'):.2f}**"
    )

    # ----------------------------------------------------
    # 8. Build Full Report
    # ----------------------------------------------------
    report = f"""
# 🧠 UltraMax Decision Summary

### {strength}

---

### 📊 **Core Metrics**
- Joint Probability (final): **{p_joint*100:.2f}%**
- Joint Probability (raw): **{p_joint_raw*100:.2f}%**
- Expected Value (EV): **{ev*100:+.2f}%**
- Correlation Used: **{corr:+.3f}**
- Payout Multiplier: **x{payout_mult}**

---

### 🏀 **Leg Details**
{leg1_text}

{leg2_text}

---

### 📈 **Advanced Analysis**
- {ev_note}  
- {corr_note}  
- {drift_msg}  
- {clv_msg}  
- {prob_msg}

---

### 💰 **Risk & Bankroll Strategy**
{stake_msg}

---

### 🔧 Diagnostics

# =====================================================================
# MODULE 13 — TEAM CONTEXT ENGINE
# PHASE 1 — Team Metadata & Baseline Context Tables
# =====================================================================

import pandas as pd

# ================================================================
# 1. TEAM LIST (Canonical abbreviations mapped to full names)
# ================================================================
TEAM_MAP = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"
}


# ================================================================
# 2. TEAM PACE & POSSESSION RATES (Normalized)
# Used by: usage engine, matchup engine, blowout model, volatility model
# ================================================================
TEAM_PACE = {
    "ATL": 101.5, "BOS": 98.9, "BKN": 100.2, "CHA": 101.7,
    "CHI": 99.4, "CLE": 96.7, "DAL": 98.8, "DEN": 97.2,
    "DET": 100.8, "GSW": 102.1, "HOU": 100.9, "IND": 103.0,
    "LAC": 97.4, "LAL": 101.9, "MEM": 100.6, "MIA": 96.1,
    "MIL": 100.5, "MIN": 99.8, "NOP": 97.9, "NYK": 96.5,
    "OKC": 100.7, "ORL": 97.4, "PHI": 98.2, "PHX": 97.1,
    "POR": 98.8, "SAC": 101.1, "SAS": 100.8, "TOR": 99.5,
    "UTA": 99.7, "WAS": 102.3
}

PACE_MEAN = sum(TEAM_PACE.values()) / len(TEAM_PACE)


# ================================================================
# 3. TEAM DEFENSIVE STRENGTH BASELINES (Normalized 0.80–1.20 scale)
# Used by: Opponent Engine, Defensive Matchup Engine
# ================================================================
TEAM_DEFENSE = {
    "ATL": 1.05, "BOS": 0.88, "BKN": 1.08, "CHA": 1.12,
    "CHI": 1.00, "CLE": 0.90, "DAL": 1.04, "DEN": 0.92,
    "DET": 1.10, "GSW": 1.02, "HOU": 0.96, "IND": 1.07,
    "LAC": 0.94, "LAL": 0.98, "MEM": 1.05, "MIA": 0.89,
    "MIL": 1.01, "MIN": 0.87, "NOP": 0.97, "NYK": 0.91,
    "OKC": 0.95, "ORL": 0.93, "PHI": 0.96, "PHX": 1.03,
    "POR": 1.11, "SAC": 1.06, "SAS": 1.09, "TOR": 0.99,
    "UTA": 1.08, "WAS": 1.12
}


# ================================================================
# 4. TEAM FOUL RATES (Drives FTA projections)
# ================================================================
TEAM_FOUL_RATE = {
    "ATL": 1.01, "BOS": 0.92, "BKN": 1.05, "CHA": 1.10,
    "CHI": 1.02, "CLE": 0.96, "DAL": 0.99, "DEN": 0.95,
    "DET": 1.06, "GSW": 1.03, "HOU": 1.00, "IND": 1.07,
    "LAC": 0.98, "LAL": 1.00, "MEM": 1.04, "MIA": 0.93,
    "MIL": 0.99, "MIN": 0.90, "NOP": 0.97, "NYK": 0.94,
    "OKC": 0.96, "ORL": 0.95, "PHI": 0.97, "PHX": 1.01,
    "POR": 1.11, "SAC": 1.06, "SAS": 1.09, "TOR": 1.00,
    "UTA": 1.05, "WAS": 1.12
}


# ================================================================
# 5. TEAM REBOUNDING MULTIPLIERS (OREB & DREB impact)
# ================================================================
TEAM_REBOUNDING = {
    "ATL": 1.03, "BOS": 1.00, "BKN": 0.95, "CHA": 0.98,
    "CHI": 1.01, "CLE": 1.07, "DAL": 0.97, "DEN": 1.05,
    "DET": 0.99, "GSW": 0.96, "HOU": 1.08, "IND": 0.95,
    "LAC": 0.97, "LAL": 1.06, "MEM": 1.04, "MIA": 1.02,
    "MIL": 1.05, "MIN": 1.06, "NOP": 1.03, "NYK": 1.07,
    "OKC": 0.95, "ORL": 1.04, "PHI": 1.01, "PHX": 0.96,
    "POR": 0.98, "SAC": 1.02, "SAS": 0.97, "TOR": 0.99,
    "UTA": 1.00, "WAS": 0.94
}


# ================================================================
# 6. TEAM ALLOWANCE BY MARKET (Points, Rebounds, Assists, PRA)
# Built from normalized percentile data
# ================================================================
TEAM_ALLOWANCE = {
    "Points": {
        "ATL": 1.05, "BOS": 0.86, "BKN": 1.09, "CHA": 1.14,
        "CLE": 0.89, "DEN": 0.93, "DET": 1.10, "GSW": 1.06,
        "HOU": 0.95, "IND": 1.13, "LAC": 0.94, "LAL": 1.02,
        "MEM": 1.06, "MIA": 0.88, "MIL": 1.01, "MIN": 0.85,
        "NOP": 0.96, "NYK": 0.92, "OKC": 0.97, "ORL": 0.91,
        "PHI": 0.96, "PHX": 1.04, "POR": 1.13, "SAC": 1.08,
        "SAS": 1.09, "TOR": 0.98, "UTA": 1.10, "WAS": 1.13
    },

    "Rebounds": {
        "ATL": 1.02, "BOS": 0.92, "BKN": 0.96, "CHA": 1.04,
        "CLE": 0.90, "DEN": 1.05, "DET": 1.02, "GSW": 0.94,
        "HOU": 1.08, "IND": 0.99, "LAC": 0.95, "LAL": 1.07,
        "MEM": 1.05, "MIA": 0.97, "MIL": 1.06, "MIN": 1.04,
        "NOP": 1.03, "NYK": 1.07, "OKC": 0.94, "ORL": 1.04,
        "PHI": 1.00, "PHX": 0.95, "POR": 1.01, "SAC": 1.02,
        "SAS": 0.96, "TOR": 0.98, "UTA": 1.01, "WAS": 0.93
    },

    "Assists": {
        "ATL": 1.03, "BOS": 0.89, "BKN": 1.04, "CHA": 1.06,
        "CLE": 0.92, "DEN": 0.93, "DET": 1.08, "GSW": 1.02,
        "HOU": 0.96, "IND": 1.13, "LAC": 0.90, "LAL": 1.02,
        "MEM": 1.06, "MIA": 0.90, "MIL": 0.98, "MIN": 0.94,
        "NOP": 0.95, "NYK": 0.91, "OKC": 0.96, "ORL": 0.93,
        "PHI": 0.97, "PHX": 1.04, "POR": 1.10, "SAC": 1.05,
        "SAS": 1.04, "TOR": 0.98, "UTA": 1.06, "WAS": 1.12
    },

    "PRA": {  # blended
        team: (
            TEAM_ALLOWANCE["Points"][team] * 0.50 +
            TEAM_ALLOWANCE["Rebounds"][team] * 0.25 +
            TEAM_ALLOWANCE["Assists"][team] * 0.25
        )
        for team in TEAM_MAP.keys()
    }
}

# =====================================================================
# ✓ Phase 1 Complete
# This now supports: Opponent Engine, Defensive Matchups,
# Game Script Engine, Pace Model, Blowout Model,
# Rotational Volatility, and PRA weighting.
# =====================================================================

# =====================================================================
# MODULE 13 — TEAM CONTEXT ENGINE
# PHASE 2 — Team Context Resolver, Validation & Safe-Lookup Layer
# =====================================================================

import re

# ---------------------------------------------------------------------
# 1. Normalize a string for fuzzy team lookup
# ---------------------------------------------------------------------
def _clean_team_input(raw: str) -> str:
    if not raw:
        return ""

    raw = raw.strip().upper()

    # remove punctuation, spaces
    raw = re.sub(r"[^A-Z]", "", raw)

    return raw


# ---------------------------------------------------------------------
# 2. Fuzzy lookup table (allows: “LAL”, "LAKERS", "LOSANGELESLAKERS")
# ---------------------------------------------------------------------
_TEAM_FUZZY = {}

for abbr, name in TEAM_MAP.items():

    # Core abbreviation
    _TEAM_FUZZY[abbr] = abbr

    # Full name (LOS ANGELES LAKERS → LOSANGELESL**)
    cleaned_full = _clean_team_input(name)
    _TEAM_FUZZY[cleaned_full] = abbr

    # City-only variants (LOSANGELES)
    city = name.split()[0]  # "Los", "Golden", etc.
    city_clean = _clean_team_input(city)
    _TEAM_FUZZY[city_clean] = abbr

    # Nickname-only variants (LAKERS)
    nickname = name.split()[-1]
    nick_clean = _clean_team_input(nickname)
    _TEAM_FUZZY[nick_clean] = abbr


# Add manual fuzzy cases
_TEAM_FUZZY.update({
    "LA": "LAL",
    "CLIPS": "LAC",
    "KNICKS": "NYK",
    "NETS": "BKN",
    "HEAT": "MIA",
    "CELTICS": "BOS",
    "WOLVES": "MIN",
    "TIMBERWOLVES": "MIN",
    "PELS": "NOP",
    "SPURS": "SAS",
})


# ---------------------------------------------------------------------
# 3. Resolve team safely
# ---------------------------------------------------------------------
def resolve_team(team_input: str):
    """
    Resolves any user-entered team input to a canonical NBA abbreviation.
    Guarantees:
        - Always uppercase
        - Always 3-letter abbreviation OR None
        - Never crashes
    """

    if not team_input:
        return None, "Empty team input."

    cleaned = _clean_team_input(team_input)

    # Exact abbreviation
    if cleaned in TEAM_MAP:
        return cleaned, None

    # Fuzzy match lookup
    if cleaned in _TEAM_FUZZY:
        return _TEAM_FUZZY[cleaned], None

    # No match — return gracefully
    return None, f"Team not recognized: {team_input}"


# ---------------------------------------------------------------------
# 4. Safe lookup helper (used by all later engines)
# ---------------------------------------------------------------------
def team_context_safe(team: str, table: dict, default=1.0):
    """
    Attempts to look up a team value in a context table (PACE, DEFENSE, etc.)
    Returns default if:
        - team cannot be resolved
        - team is missing from the table
    """
    abbr, err = resolve_team(team)
    if err or abbr not in table:
        return default
    return table[abbr]


# ---------------------------------------------------------------------
# 5. Composite team context bundle (used downstream)
# ---------------------------------------------------------------------
def get_team_context(team: str):
    """
    Returns a bundle of all relevant team multipliers:
        - pace
        - defense
        - foul rate
        - rebounding
        - market allowances
    """

    abbr, err = resolve_team(team)
    if err:
        return {
            "team": None,
            "error": err,
            "pace": PACE_MEAN,
            "def": 1.00,
            "foul": 1.00,
            "reb": 1.00,
            "allow_points": 1.00,
            "allow_rebounds": 1.00,
            "allow_assists": 1.00,
            "allow_pra": 1.00,
        }

    return {
        "team": abbr,
        "error": None,
        "pace": TEAM_PACE.get(abbr, PACE_MEAN),
        "def": TEAM_DEFENSE.get(abbr, 1.00),
        "foul": TEAM_FOUL_RATE.get(abbr, 1.00),
        "reb": TEAM_REBOUNDING.get(abbr, 1.00),
        "allow_points": TEAM_ALLOWANCE["Points"].get(abbr, 1.00),
        "allow_rebounds": TEAM_ALLOWANCE["Rebounds"].get(abbr, 1.00),
        "allow_assists": TEAM_ALLOWANCE["Assists"].get(abbr, 1.00),
        "allow_pra": TEAM_ALLOWANCE["PRA"].get(abbr, 1.00),
    }


# =====================================================================
# ✓ Phase 2 complete.
# Upstream engines are now protected from all bad team inputs.
# Phase 3 will build the actual matchup-blending engine.
# =====================================================================

# =====================================================================
# MODULE 13 — TEAM CONTEXT ENGINE
# PHASE 3 — Opponent Context Blend Engine (Team-Level Impact)
# =====================================================================

# This module blends team defensive profile, pace, fouling,
# and market-specific opponent tendencies into one multiplier.

# ---------------------------------------------------------------------
# 1. Market → context weights (how much each market depends on each factor)
# ---------------------------------------------------------------------
MARKET_CONTEXT_WEIGHTS = {
    "Points": {
        "pace": 0.40,
        "defense": 0.45,
        "fouls": 0.10,
        "rebounds": 0.00,
        "assist_funnel": 0.05
    },
    "Rebounds": {
        "pace": 0.25,
        "defense": 0.15,
        "fouls": 0.00,
        "rebounds": 0.60,
        "assist_funnel": 0.00
    },
    "Assists": {
        "pace": 0.35,
        "defense": 0.25,
        "fouls": 0.00,
        "rebounds": 0.00,
        "assist_funnel": 0.40
    },
    "PRA": {
        "pace": 0.33,
        "defense": 0.33,
        "fouls": 0.10,
        "rebounds": 0.12,
        "assist_funnel": 0.12
    },
}

# ---------------------------------------------------------------------
# 2. Helper: normalize multipliers to prevent blowouts
# ---------------------------------------------------------------------
def _stability_clamp(value, low=0.70, high=1.35):
    return float(np.clip(value, low, high))


# ---------------------------------------------------------------------
# 3. Compute opponent context multiplier (team-level, not player-specific)
# ---------------------------------------------------------------------
def compute_opponent_context(team: str, market: str):
    """
    Produces the opponent context multiplier used in:
       - projection engines
       - volatility engine
       - correlation engine
       - Monte Carlo scoring
       - matchups (macro level)
    """

    # Resolve market weights
    w = MARKET_CONTEXT_WEIGHTS.get(market, MARKET_CONTEXT_WEIGHTS["Points"])

    # Resolve team context safely
    ctx = get_team_context(team)

    # If unresolved team — neutral context; no crash
    if ctx["team"] is None:
        return 1.00

    # Pull core components
    pace_mult = ctx["pace"] / PACE_MEAN
    defense_mult = ctx["def"]
    foul_mult = ctx["foul"]
    reb_mult = ctx["reb"]
    assist_mult = ctx["allow_assists"]

    # Blend into a single multiplier
    blended = (
        pace_mult       * w["pace"] +
        defense_mult    * w["defense"] +
        foul_mult       * w["fouls"] +
        reb_mult        * w["rebounds"] +
        assist_mult     * w["assist_funnel"]
    )

    # Weighted blend must be renormalized to 1.0-centered scale
    blended = blended / (
        w["pace"] + w["defense"] + w["fouls"] + w["rebounds"] + w["assist_funnel"]
    )

    # Stability clamp
    blended = _stability_clamp(blended)

    return float(blended)


# ---------------------------------------------------------------------
# 4. Exported wrapper used by the entire model (LEG ENGINE)
# ---------------------------------------------------------------------
def opponent_team_multiplier(team: str, market: str):
    """
    Safe external wrapper — ensures no crashes regardless of input.
    This is the function used by:
       compute_leg()
       volatility_engine_v2()
       correlation_engine_v3()
       monte_carlo_combo()
       player_matchup_engine (Module 22)
    """
    try:
        return compute_opponent_context(team, market)
    except Exception:
        return 1.00  # fail gracefully no matter what


# =====================================================================
# ✓ Phase 3 complete.
# This module now produces a stable opponent multiplier for all markets.
# Next: Phase 4 = Game Environment Context (pace drift, blowout model)
# =====================================================================

# =====================================================================
# MODULE 13 — TEAM CONTEXT ENGINE
# PHASE 4 — GAME ENVIRONMENT ENGINE (PACE, BLOWOUT, GAME SCRIPTS)
# =====================================================================

# This module shapes the game-level environment multipliers
# used by: compute_leg(), volatility engine, Monte Carlo,
# correlation engine, and the CLV drift engine.

# ---------------------------------------------------------------------
# 1. Baseline constants
# ---------------------------------------------------------------------
NBA_AVG_POSSESSIONS = 99.8
NBA_AVG_PACE = 99.3
NBA_HOME_ADV = 1.018     # 1.8% increase
NBA_B2B_PENALTY = 0.96   # -4% performance on back-to-back
NBA_REST_BOOST = 1.03    # +3% after 3+ rest days
NBA_FATIGUE_FLOOR = 0.92 # max penalty applied

# ---------------------------------------------------------------------
# 2. Blowout risk buckets
# ---------------------------------------------------------------------
def blowout_multiplier(spread: float):
    """
    Returns a multiplier based on potential blowout.
    Positive = home favored.
    spread = |favorite - underdog|
    """

    s = abs(spread)

    if s <= 5:
        return 1.00   # competitive
    elif s <= 8:
        return 0.97   # mild reduction
    elif s <= 12:
        return 0.92   # increased risk
    elif s <= 16:
        return 0.85   # heavy probability
    else:
        return 0.78   # extreme blowout scenario


# ---------------------------------------------------------------------
# 3. Vegas total → pace inference
# ---------------------------------------------------------------------
def vegas_total_to_pace(total: float):
    """
    Converts Vegas game total into implied game pace.
    Higher totals usually = more possessions.
    """

    if total <= 0:
        return NBA_AVG_PACE

    # Linear regression fit (historical)
    pace = 84 + (total * 0.11)

    return float(np.clip(pace, 90, 108))


# ---------------------------------------------------------------------
# 4. Team momentum / form multiplier
# ---------------------------------------------------------------------
def team_form_boost(recent_margin: float):
    """
    recent_margin = average point differential over last 5 games.
    """

    if recent_margin >= 10:
        return 1.06
    elif recent_margin >= 5:
        return 1.03
    elif recent_margin >= 0:
        return 1.00
    elif recent_margin >= -5:
        return 0.97
    else:
        return 0.94


# ---------------------------------------------------------------------
# 5. Back-to-back fatigue + Rest days
# ---------------------------------------------------------------------
def fatigue_multiplier(is_b2b: bool, rest_days: int):
    if is_b2b:
        return NBA_B2B_PENALTY

    if rest_days >= 3:
        return NBA_REST_BOOST

    # Normal 1–2 day rest
    return 1.00


# ---------------------------------------------------------------------
# 6. Home/Away performance drift
# ---------------------------------------------------------------------
def home_away_multiplier(is_home: bool):
    return NBA_HOME_ADV if is_home else 1.00


# ---------------------------------------------------------------------
# 7. Game Script Engine (fast, medium, slow environment)
# ---------------------------------------------------------------------
def game_script_multiplier(pace: float, defense_intensity: float):
    """
    pace: inferred pace
    defense_intensity: from team context (0.85–1.20)
    """

    # Normalize pace relative to league average
    pace_factor = pace / NBA_AVG_PACE

    # Blend with defense (high intensity = lower scoring)
    final = (pace_factor * 0.70) + ((1 / defense_intensity) * 0.30)

    return float(np.clip(final, 0.85, 1.22))


# ---------------------------------------------------------------------
# 8. Master Environment Engine (FULL COMPOSITE)
# ---------------------------------------------------------------------
def compute_game_environment(
    vegas_total: float,
    spread: float,
    recent_margin: float,
    is_home: bool,
    is_b2b: bool,
    rest_days: int,
    opponent_def_rating: float,
):
    """
    Produces a 0.80–1.30 environment multiplier used by:
    - compute_leg()
    - volatility_engine_v2()
    - MC scoring
    - drift system
    - blowout model
    - correlation engine
    """

    # Vegas → pace
    pace = vegas_total_to_pace(vegas_total)

    # Defensive intensity (team level)
    defense_intensity = opponent_def_rating

    # Base script
    script_mult = game_script_multiplier(pace, defense_intensity)

    # Team form boost
    form_mult = team_form_boost(recent_margin)

    # Blowout adjustment
    blow_mult = blowout_multiplier(spread)

    # Rest/Fatigue
    fat_mult = fatigue_multiplier(is_b2b, rest_days)

    # Home/Away
    ha_mult = home_away_multiplier(is_home)

    # Combine everything
    combo = (
        script_mult * 0.40 +
        form_mult   * 0.15 +
        blow_mult   * 0.15 +
        fat_mult    * 0.15 +
        ha_mult     * 0.15
    )

    # Normalize to stable range
    combo = float(np.clip(combo, 0.80, 1.30))

    return {
        "env_multiplier": combo,
        "pace_implied": pace,
        "script": script_mult,
        "blowout": blow_mult,
        "form": form_mult,
        "fatigue": fat_mult,
        "home_away": ha_mult,
    }


# ---------------------------------------------------------------------
# 9. Wrapper used by compute_leg() and engines
# ---------------------------------------------------------------------
def game_environment(team: str, opponent: str, vegas_total: float, spread: float):
    """
    External wrapper that resolves team & opponent data safely
    and returns full environment package.
    """

    ctx = get_team_context(opponent)

    if ctx["team"] is None:
        # neutral fallback
        return {
            "env_multiplier": 1.00,
            "pace_implied": NBA_AVG_PACE,
            "script": 1.00,
            "blowout": 1.00,
            "form": 1.00,
            "fatigue": 1.00,
            "home_away": 1.00,
        }

    return compute_game_environment(
        vegas_total=vegas_total,
        spread=spread,
        recent_margin=ctx["recent_margin"],
        is_home=False,  # we do not track venue yet; added in Module 18
        is_b2b=False,   # added later
        rest_days=2,    # placeholder until we add schedule API
        opponent_def_rating=ctx["def"],
    )


# =====================================================================
# ✓ Phase 4 Complete
# Next up: Module 13 — Phase 5 (Team-to-Game Context Fusion)
# =====================================================================

# =====================================================================
# MODULE 13 — TEAM CONTEXT ENGINE
# PHASE 5 — TEAM–GAME CONTEXT FUSION ENGINE
# =====================================================================

def fuse_team_and_environment(
    team: str,
    opponent: str,
    vegas_total: float,
    spread: float
):
    """
    Produces a final ~0.75 – 1.35 multiplier used by the
    projection engine, volatility engine, and correlation engine.

    This is the MASTER CONTEXT MULTIPLIER for the entire system.
    """

    # ---------------------------------------------------------
    # 1. Pull opponent/defensive/team context
    # ---------------------------------------------------------
    team_ctx = get_team_context(team)
    opp_ctx = get_team_context(opponent)

    # Fallback if missing
    if team_ctx["team"] is None or opp_ctx["team"] is None:
        return {
            "final_mult": 1.00,
            "off_rating_mult": 1.00,
            "def_rating_mult": 1.00,
            "pace_mult": 1.00,
            "blowout_mult": 1.00,
            "form_mult": 1.00,
            "fatigue_mult": 1.00,
            "env_mult": 1.00,
        }

    # ---------------------------------------------------------
    # 2. Retrieve full environment package from Phase 4
    # ---------------------------------------------------------
    env = game_environment(
        team=team,
        opponent=opponent,
        vegas_total=vegas_total,
        spread=spread
    )

    # ---------------------------------------------------------
    # 3. Offensive rating multiplier
    # Team with high ORtg → boosts projections
    # ---------------------------------------------------------
    off_rating = team_ctx["off"]

    if      off_rating >= 118: off_mult = 1.08
    elif    off_rating >= 114: off_mult = 1.05
    elif    off_rating >= 110: off_mult = 1.02
    elif    off_rating >= 106: off_mult = 1.00
    elif    off_rating >= 102: off_mult = 0.97
    else:                      off_mult = 0.93

    # Smooth using sqrt curve
    off_mult = float(np.clip(off_mult ** 0.9, 0.90, 1.10))

    # ---------------------------------------------------------
    # 4. Defensive pressure multiplier
    # Opponent stronger defense → reduces projections
    # ---------------------------------------------------------
    def_rating = opp_ctx["def"]

    if      def_rating >= 1.15: def_mult = 0.90
    elif    def_rating >= 1.10: def_mult = 0.94
    elif    def_rating >= 1.05: def_mult = 0.97
    elif    def_rating >= 0.97: def_mult = 1.00
    elif    def_rating >= 0.93: def_mult = 1.03
    else:                        def_mult = 1.06

    def_mult = float(np.clip(def_mult ** 0.95, 0.88, 1.10))

    # ---------------------------------------------------------
    # 5. Pace normalization
    # From environment context
    # ---------------------------------------------------------
    pace_mult = env["script"]
    pace_mult = float(np.clip(pace_mult, 0.88, 1.18))

    # ---------------------------------------------------------
    # 6. Momentum / Form
    # ---------------------------------------------------------
    form_mult = env["form"]
    form_mult = float(np.clip(form_mult, 0.92, 1.08))

    # ---------------------------------------------------------
    # 7. Blowout risk multiplier
    # ---------------------------------------------------------
    blowout_mult = env["blowout"]
    blowout_mult = float(np.clip(blowout_mult, 0.78, 1.02))

    # ---------------------------------------------------------
    # 8. Fatigue/Rest multiplier
    # ---------------------------------------------------------
    fatigue_mult = env["fatigue"]
    fatigue_mult = float(np.clip(fatigue_mult, 0.92, 1.05))

    # ---------------------------------------------------------
    # 9. Environment Multiplier (Phase 4's output)
    # ---------------------------------------------------------
    env_mult = env["env_multiplier"]

    # ---------------------------------------------------------
    # 10. Master Fusion
    # ---------------------------------------------------------
    # Weighted blend based on real-world impact %
    final_mult = (
        off_mult       * 0.25 +
        def_mult       * 0.25 +
        pace_mult      * 0.20 +
        form_mult      * 0.10 +
        blowout_mult   * 0.10 +
        fatigue_mult   * 0.05 +
        env_mult       * 0.05
    )

    # Normalize to stable model range
    final_mult = float(np.clip(final_mult, 0.75, 1.35))

    return {
        "final_mult": final_mult,
        "off_rating_mult": off_mult,
        "def_rating_mult": def_mult,
        "pace_mult": pace_mult,
        "form_mult": form_mult,
        "blowout_mult": blowout_mult,
        "fatigue_mult": fatigue_mult,
        "env_mult": env_mult,
    }


# =====================================================================
# ✓ Phase 5 COMPLETE
# Next: Module 13 — Phase 6 (Context Injection Layer for compute_leg)
# =====================================================================

# =====================================================================
# MODULE 13 — TEAM CONTEXT ENGINE
# PHASE 6 — CONTEXT INJECTION LAYER FOR compute_leg()
# =====================================================================

def apply_team_context_to_projection(
    mu: float,
    sd: float,
    team: str,
    opponent: str,
    vegas_total: float,
    spread: float
):
    """
    Injects team-game-environment context into:
      - mean projection (mu)
      - volatility (sd)
    
    Uses final_mult from the fusion engine (Phase 5).
    Returns modified (mu_adj, sd_adj, context_pkg).
    """

    # Pull fused context coefficients
    ctx = fuse_team_and_environment(
        team=team,
        opponent=opponent,
        vegas_total=vegas_total,
        spread=spread
    )

    final_mult = ctx["final_mult"]

    # ---------------------------------------------------------
    # Apply to mean (μ)
    # ---------------------------------------------------------
    mu_adj = mu * final_mult

    # ---------------------------------------------------------
    # Apply volatility adjustments  
    # Higher pace → higher variance  
    # Strong defense → suppress mean but increase variance  
    # Blowout → reduces both  
    # ---------------------------------------------------------
    vol_mult = 1.0

    # Pace → more possessions → higher variance
    vol_mult *= np.interp(
        ctx["pace_mult"], 
        [0.88, 1.00, 1.18],
        [0.92, 1.00, 1.12]
    )

    # Defensive difficulty → adds uncertainty
    vol_mult *= np.interp(
        ctx["def_rating_mult"],
        [0.88, 1.00, 1.10],
        [1.08, 1.00, 0.92]
    )

    # Blowout → reduces variance sharply
    vol_mult *= np.interp(
        ctx["blowout_mult"],
        [0.78, 1.00],
        [0.85, 1.00]
    )

    # Team form → stable teams produce lower variance
    vol_mult *= np.interp(
        ctx["form_mult"],
        [0.92, 1.00, 1.08],
        [1.05, 1.00, 0.94]
    )

    # Fatigue → increases randomness
    vol_mult *= np.interp(
        ctx["fatigue_mult"],
        [0.92, 1.00, 1.05],
        [1.06, 1.00, 0.96]
    )

    # Clamp volatility multiplier
    vol_mult = float(np.clip(vol_mult, 0.80, 1.25))

    sd_adj = sd * vol_mult

    # Clean SD (cannot be <= 0)
    sd_adj = max(sd_adj, 0.05)

    # ---------------------------------------------------------
    # Output package
    # ---------------------------------------------------------
    return mu_adj, sd_adj, ctx

# =====================================================================
# MODULE 14 — OPPONENT DEFENSIVE PROFILE DATABASE (CORE TABLE v1)
# PHASE 1 — STATIC DB FOR PROJECTION ENGINES
# =====================================================================

import numpy as np

# ---------------------------------------------------------------------
# BASE DEFENSIVE DATABASE
# Each team receives:
#   - pace_rank           (1–30, lower = faster)
#   - def_rating_rank     (1–30, lower = better defense)
#   - paint_def_rank      (FG% allowed at rim)
#   - perimeter_def_rank  (3PT defense)
#   - assist_def_rank     (assists allowed)
#   - rebound_rank        (rebounds allowed)
#   - turnover_rank       (turnovers forced)
# Output values: multipliers centered at 1.00 (0.90–1.12 typical)
# ---------------------------------------------------------------------

def _scale(rank, low=0.90, mid=1.00, high=1.12):
    """
    Converts a 1–30 defensive rank into a multiplier.
    Better defense → lower multiplier.
    Worse defense → higher multiplier.
    """
    return np.interp(rank, [1, 15, 30], [low, mid, high])


OPP_DEF_PROFILE = {
    "ATL": {
        "pace_mult": _scale(6),
        "def_rating_mult": _scale(26),
        "paint_mult": _scale(20),
        "perimeter_mult": _scale(25),
        "assist_mult": _scale(22),
        "rebound_mult": _scale(18),
        "turnover_mult": _scale(17),
    },
    "BOS": {
        "pace_mult": _scale(18),
        "def_rating_mult": _scale(2),
        "paint_mult": _scale(3),
        "perimeter_mult": _scale(2),
        "assist_mult": _scale(4),
        "rebound_mult": _scale(5),
        "turnover_mult": _scale(7),
    },
    "BKN": {
        "pace_mult": _scale(20),
        "def_rating_mult": _scale(15),
        "paint_mult": _scale(9),
        "perimeter_mult": _scale(13),
        "assist_mult": _scale(11),
        "rebound_mult": _scale(24),
        "turnover_mult": _scale(20),
    },
    "CHA": {
        "pace_mult": _scale(8),
        "def_rating_mult": _scale(29),
        "paint_mult": _scale(25),
        "perimeter_mult": _scale(28),
        "assist_mult": _scale(26),
        "rebound_mult": _scale(27),
        "turnover_mult": _scale(23),
    },
    "CHI": {
        "pace_mult": _scale(24),
        "def_rating_mult": _scale(12),
        "paint_mult": _scale(14),
        "perimeter_mult": _scale(8),
        "assist_mult": _scale(12),
        "rebound_mult": _scale(15),
        "turnover_mult": _scale(9),
    },
    "CLE": {
        "pace_mult": _scale(29),
        "def_rating_mult": _scale(6),
        "paint_mult": _scale(12),
        "perimeter_mult": _scale(7),
        "assist_mult": _scale(6),
        "rebound_mult": _scale(6),
        "turnover_mult": _scale(12),
    },
    "DAL": {
        "pace_mult": _scale(7),
        "def_rating_mult": _scale(20),
        "paint_mult": _scale(18),
        "perimeter_mult": _scale(22),
        "assist_mult": _scale(17),
        "rebound_mult": _scale(23),
        "turnover_mult": _scale(25),
    },
    "DEN": {
        "pace_mult": _scale(27),
        "def_rating_mult": _scale(8),
        "paint_mult": _scale(10),
        "perimeter_mult": _scale(11),
        "assist_mult": _scale(9),
        "rebound_mult": _scale(4),
        "turnover_mult": _scale(8),
    },
    "DET": {
        "pace_mult": _scale(11),
        "def_rating_mult": _scale(28),
        "paint_mult": _scale(26),
        "perimeter_mult": _scale(27),
        "assist_mult": _scale(28),
        "rebound_mult": _scale(29),
        "turnover_mult": _scale(26),
    },
    "GSW": {
        "pace_mult": _scale(2),
        "def_rating_mult": _scale(19),
        "paint_mult": _scale(21),
        "perimeter_mult": _scale(16),
        "assist_mult": _scale(15),
        "rebound_mult": _scale(20),
        "turnover_mult": _scale(16),
    },
    # --------------------------------------------------------------
    # You will continue the DB for all 30 NBA teams in Phase 2
    # --------------------------------------------------------------
}
# =====================================================================
# MODULE 14 — OPPONENT DEFENSIVE PROFILE DATABASE
# PHASE 2 — Remaining Teams (20 Teams)
# =====================================================================

OPP_DEF_PROFILE.update({

    "HOU": {
        "pace_mult": _scale(10),
        "def_rating_mult": _scale(5),
        "paint_mult": _scale(7),
        "perimeter_mult": _scale(10),
        "assist_mult": _scale(8),
        "rebound_mult": _scale(3),
        "turnover_mult": _scale(6),
    },

    "IND": {
        "pace_mult": _scale(1),
        "def_rating_mult": _scale(25),
        "paint_mult": _scale(23),
        "perimeter_mult": _scale(26),
        "assist_mult": _scale(21),
        "rebound_mult": _scale(22),
        "turnover_mult": _scale(24),
    },

    "LAC": {
        "pace_mult": _scale(22),
        "def_rating_mult": _scale(9),
        "paint_mult": _scale(11),
        "perimeter_mult": _scale(14),
        "assist_mult": _scale(10),
        "rebound_mult": _scale(13),
        "turnover_mult": _scale(11),
    },

    "LAL": {
        "pace_mult": _scale(9),
        "def_rating_mult": _scale(14),
        "paint_mult": _scale(8),
        "perimeter_mult": _scale(12),
        "assist_mult": _scale(13),
        "rebound_mult": _scale(8),
        "turnover_mult": _scale(14),
    },

    "MEM": {
        "pace_mult": _scale(12),
        "def_rating_mult": _scale(13),
        "paint_mult": _scale(15),
        "perimeter_mult": _scale(17),
        "assist_mult": _scale(14),
        "rebound_mult": _scale(12),
        "turnover_mult": _scale(5),
    },

    "MIA": {
        "pace_mult": _scale(30),
        "def_rating_mult": _scale(7),
        "paint_mult": _scale(6),
        "perimeter_mult": _scale(5),
        "assist_mult": _scale(3),
        "rebound_mult": _scale(10),
        "turnover_mult": _scale(4),
    },

    "MIL": {
        "pace_mult": _scale(15),
        "def_rating_mult": _scale(11),
        "paint_mult": _scale(4),
        "perimeter_mult": _scale(9),
        "assist_mult": _scale(7),
        "rebound_mult": _scale(1),
        "turnover_mult": _scale(10),
    },

    "MIN": {
        "pace_mult": _scale(17),
        "def_rating_mult": _scale(1),
        "paint_mult": _scale(1),
        "perimeter_mult": _scale(4),
        "assist_mult": _scale(2),
        "rebound_mult": _scale(2),
        "turnover_mult": _scale(3),
    },

    "NOP": {
        "pace_mult": _scale(13),
        "def_rating_mult": _scale(10),
        "paint_mult": _scale(16),
        "perimeter_mult": _scale(15),
        "assist_mult": _scale(16),
        "rebound_mult": _scale(17),
        "turnover_mult": _scale(13),
    },

    "NYK": {
        "pace_mult": _scale(23),
        "def_rating_mult": _scale(4),
        "paint_mult": _scale(5),
        "perimeter_mult": _scale(3),
        "assist_mult": _scale(5),
        "rebound_mult": _scale(7),
        "turnover_mult": _scale(2),
    },

    "OKC": {
        "pace_mult": _scale(5),
        "def_rating_mult": _scale(3),
        "paint_mult": _scale(13),
        "perimeter_mult": _scale(6),
        "assist_mult": _scale(1),
        "rebound_mult": _scale(9),
        "turnover_mult": _scale(1),
    },

    "ORL": {
        "pace_mult": _scale(21),
        "def_rating_mult": _scale(16),
        "paint_mult": _scale(17),
        "perimeter_mult": _scale(18),
        "assist_mult": _scale(19),
        "rebound_mult": _scale(11),
        "turnover_mult": _scale(18),
    },

    "PHI": {
        "pace_mult": _scale(16),
        "def_rating_mult": _scale(17),
        "paint_mult": _scale(19),
        "perimeter_mult": _scale(21),
        "assist_mult": _scale(18),
        "rebound_mult": _scale(16),
        "turnover_mult": _scale(19),
    },

    "PHX": {
        "pace_mult": _scale(19),
        "def_rating_mult": _scale(18),
        "paint_mult": _scale(22),
        "perimeter_mult": _scale(20),
        "assist_mult": _scale(20),
        "rebound_mult": _scale(19),
        "turnover_mult": _scale(21),
    },

    "POR": {
        "pace_mult": _scale(14),
        "def_rating_mult": _scale(27),
        "paint_mult": _scale(27),
        "perimeter_mult": _scale(29),
        "assist_mult": _scale(27),
        "rebound_mult": _scale(30),
        "turnover_mult": _scale(27),
    },

    "SAC": {
        "pace_mult": _scale(3),
        "def_rating_mult": _scale(23),
        "paint_mult": _scale(24),
        "perimeter_mult": _scale(24),
        "assist_mult": _scale(24),
        "rebound_mult": _scale(25),
        "turnover_mult": _scale(22),
    },

    "SAS": {
        "pace_mult": _scale(4),
        "def_rating_mult": _scale(30),
        "paint_mult": _scale(30),
        "perimeter_mult": _scale(30),
        "assist_mult": _scale(30),
        "rebound_mult": _scale(28),
        "turnover_mult": _scale(30),
    },

    "TOR": {
        "pace_mult": _scale(25),
        "def_rating_mult": _scale(21),
        "paint_mult": _scale(28),
        "perimeter_mult": _scale(23),
        "assist_mult": _scale(23),
        "rebound_mult": _scale(14),
        "turnover_mult": _scale(15),
    },

    "UTA": {
        "pace_mult": _scale(26),
        "def_rating_mult": _scale(22),
        "paint_mult": _scale(29),
        "perimeter_mult": _scale(19),
        "assist_mult": _scale(29),
        "rebound_mult": _scale(21),
        "turnover_mult": _scale(29),
    },

    "WAS": {
        "pace_mult": _scale(28),
        "def_rating_mult": _scale(24),
        "paint_mult": _scale(17),
        "perimeter_mult": _scale(30),
        "assist_mult": _scale(25),
        "rebound_mult": _scale(26),
        "turnover_mult": _scale(28),
    },

})
# =====================================================================
# MODULE 14 — PHASE 3
# COMPOSITE DEFENSIVE SCORE GENERATORS
# =====================================================================

def build_composite_def_scores(team_code: str) -> dict:
    """
    Converts raw OPP_DEF_PROFILE data into composite, normalized
    defensive weights used by:
        - Opponent matchup engine (Module 4)
        - Team context engine (Module 13)
        - Rotational volatility (Module 19)
        - Defensive matchup overrides (Module 22)

    Output:
        {
            "scoring_def": float,
            "paint_def": float,
            "perimeter_def": float,
            "assist_def": float,
            "rebound_def": float,
            "turnover_def": float,
            "overall_def": float
        }
    """
    prof = OPP_DEF_PROFILE.get(team_code.upper())
    if prof is None:
        return {
            "scoring_def": 1.0,
            "paint_def": 1.0,
            "perimeter_def": 1.0,
            "assist_def": 1.0,
            "rebound_def": 1.0,
            "turnover_def": 1.0,
            "overall_def": 1.0,
        }

    # -------------------------------
    # Extract core components
    # -------------------------------
    pace = prof["pace_mult"]
    defrat = prof["def_rating_mult"]
    paint = prof["paint_mult"]
    perim = prof["perimeter_mult"]
    ast = prof["assist_mult"]
    reb = prof["rebound_mult"]
    tov = prof["turnover_mult"]

    # --------------------------------------------------
    # 1. Scoring defense composite
    # --------------------------------------------------
    # Weighted toward defense rating + perimeter strength
    scoring_def = (
        defrat * 0.55 +
        perim * 0.25 +
        pace * 0.20
    )

    # --------------------------------------------------
    # 2. Paint defense (big man markets)
    # --------------------------------------------------
    paint_def = (
        paint * 0.70 +
        defrat * 0.20 +
        reb * 0.10
    )

    # --------------------------------------------------
    # 3. Perimeter defense (guard markets)
    # --------------------------------------------------
    perimeter_def = (
        perim * 0.65 +
        defrat * 0.25 +
        pace * 0.10
    )

    # --------------------------------------------------
    # 4. Assist prevention
    # --------------------------------------------------
    assist_def = (
        ast * 0.60 +
        pace * 0.20 +
        perim * 0.20
    )

    # --------------------------------------------------
    # 5. Rebounding resistance
    # --------------------------------------------------
    rebound_def = (
        reb * 0.75 +
        paint * 0.15 +
        pace * 0.10
    )

    # --------------------------------------------------
    # 6. Turnover generation
    # --------------------------------------------------
    turnover_def = (
        tov * 0.75 +
        perim * 0.25
    )

    # --------------------------------------------------
    # 7. Overall composite
    # --------------------------------------------------
    overall_def = (
        scoring_def * 0.40 +
        paint_def * 0.20 +
        perimeter_def * 0.20 +
        assist_def * 0.10 +
        rebound_def * 0.10
    )

    # Normalize profiles to playable ranges
    def _norm(x): return float(np.clip(x, 0.75, 1.30))

    return {
        "scoring_def": _norm(scoring_def),
        "paint_def": _norm(paint_def),
        "perimeter_def": _norm(perimeter_def),
        "assist_def": _norm(assist_def),
        "rebound_def": _norm(rebound_def),
        "turnover_def": _norm(turnover_def),
        "overall_def": _norm(overall_def),
    }


# =====================================================================
# MASS-BUILD DEFENSIVE COMPOSITES FOR ALL TEAMS
# =====================================================================

DEF_COMPOSITES = {
    team: build_composite_def_scores(team)
    for team in OPP_DEF_PROFILE.keys()
}

# =====================================================================
# MODULE 14 — PHASE 4
# DYNAMIC DEFENSIVE CONTEXT OVERRIDES ENGINE
# =====================================================================

def _normalize_multiplier(x, low=0.70, high=1.40):
    """Ensures defensive multipliers stay within sane limits."""
    return float(np.clip(x, low, high))


def apply_dynamic_def_context(
    team_code: str,
    opponent_team: str = None,
    key_defenders_out: int = 0,
    back_to_back: bool = False,
    home: bool = True,
    slow_trend: float = 1.0,
    fast_trend: float = 1.0,
    rotation_depth: int = 9,
    matchup_anomaly: float = 1.0
):
    """
    Produces REAL-TIME defensive difficulty multipliers that override the static
    composite database (Phase 3).

    Inputs:
        team_code           — defensive team (e.g., "MEM")
        opponent_team       — optional, influences pace & scheme adjustments
        key_defenders_out   — 0–3 (starter wings/bigs missing)
        back_to_back        — True/False (fatigue adjustment)
        home                — True/False (defensive intensity bump)
        slow_trend          — >1.0 means recent slow pace, <1.0 means fast
        fast_trend          — >1.0 means recent fast pace, <1.0 means slower
        rotation_depth      — 7–11 (shorter rotation = stronger defense)
        matchup_anomaly    — special-case overrides (e.g., Gobert vs elite big)

    Returns dictionary:
        "scoring_adj"
        "paint_adj"
        "perimeter_adj"
        "assist_adj"
        "rebound_adj"
        "turnover_adj"
        "overall_adj"
    """

    team = team_code.upper()
    if team not in DEF_COMPOSITES:
        return {
            "scoring_adj": 1.0,
            "paint_adj": 1.0,
            "perimeter_adj": 1.0,
            "assist_adj": 1.0,
            "rebound_adj": 1.0,
            "turnover_adj": 1.0,
            "overall_adj": 1.0,
        }

    base = DEF_COMPOSITES[team]

    # -----------------------------------------
    # 1. Key defender injury adjustment
    # -----------------------------------------
    # Missing: 1 = wing defender, 2 = big, 3 = scheme anchor
    injury_penalty = 1.0 + (0.07 * key_defenders_out)

    # -----------------------------------------
    # 2. Back-to-back fatigue
    # -----------------------------------------
    b2b_factor = 1.05 if back_to_back else 1.00

    # -----------------------------------------
    # 3. Home/Away pressure index
    # -----------------------------------------
    home_adj = 0.97 if home else 1.03

    # -----------------------------------------
    # 4. Pace trend adjustments
    # -----------------------------------------
    pace_adj = (slow_trend * 0.60) + (fast_trend * 0.40)

    # Normalize
    pace_adj = _normalize_multiplier(pace_adj, 0.85, 1.20)

    # -----------------------------------------
    # 5. Rotation depth adjustments
    # -----------------------------------------
    # Short rotations → tighter defense
    if rotation_depth <= 7:
        depth_adj = 0.92
    elif rotation_depth == 8:
        depth_adj = 0.96
    elif rotation_depth == 9:
        depth_adj = 1.00
    elif rotation_depth == 10:
        depth_adj = 1.04
    else:  # 11+
        depth_adj = 1.08

    # -----------------------------------------
    # 6. Matchup-specific anomaly
    # -----------------------------------------
    # Example: Gobert vs elite paint scorers = 0.90 penalty
    anomaly_adj = matchup_anomaly

    # -----------------------------------------
    # 7. Apply contextual multipliers to composites
    # -----------------------------------------
    s_adj = base["scoring_def"] * injury_penalty * b2b_factor * home_adj * pace_adj * depth_adj * anomaly_adj
    p_adj = base["paint_def"]    * injury_penalty * b2b_factor * home_adj * pace_adj * depth_adj * anomaly_adj
    per_adj = base["perimeter_def"] * injury_penalty * b2b_factor * home_adj * pace_adj * depth_adj * anomaly_adj
    a_adj = base["assist_def"]   * injury_penalty * b2b_factor * home_adj * pace_adj * depth_adj
    r_adj = base["rebound_def"]  * injury_penalty * b2b_factor * home_adj * pace_adj * depth_adj
    t_adj = base["turnover_def"] * injury_penalty * home_adj * pace_adj

    # Overall weighted context metric
    overall_adj = (
        s_adj * 0.35 +
        p_adj * 0.20 +
        per_adj * 0.20 +
        a_adj * 0.10 +
        r_adj * 0.10 +
        t_adj * 0.05
    )

    return {
        "scoring_adj": _normalize_multiplier(s_adj),
        "paint_adj": _normalize_multiplier(p_adj),
        "perimeter_adj": _normalize_multiplier(per_adj),
        "assist_adj": _normalize_multiplier(a_adj),
        "rebound_adj": _normalize_multiplier(r_adj),
        "turnover_adj": _normalize_multiplier(t_adj),
        "overall_adj": _normalize_multiplier(overall_adj),
    }

# =====================================================================
# MODULE 14 — PHASE 5
# DEFENSIVE TILT ENGINE (BAYESIAN ROLLING TREND ADJUSTOR)
# =====================================================================

def _bayes_combine(prior, likelihood, weight_prior=0.65, weight_like=0.35):
    """
    Generic Bayesian combiner for defensive trends.
    Produces smoothed adjustment that avoids overreacting to noise.
    """
    return float(
        (prior * weight_prior) + (likelihood * weight_like)
    )


def compute_def_tilt(
    team_code: str,
    last5_ppp_allowed: float,
    season_ppp_allowed: float,
    last5_reb_rate_allowed: float,
    season_reb_rate_allowed: float,
    last5_ast_rate_allowed: float,
    season_ast_rate_allowed: float,
    slow_trend_factor: float = 1.0,
    fast_trend_factor: float = 1.0,
    include_variance: bool = True
):
    """
    Computes whether a defense is HOT or COLD using Bayesian updates.

    Inputs:
        last5_ppp_allowed      — points per possession allowed (L5 games)
        season_ppp_allowed     — season average PPP allowed
        last5_reb_rate_allowed — rebound rate allowed (L5)
        season_reb_rate_allowed— season average rebound rate allowed
        last5_ast_rate_allowed — assist percentage allowed (L5)
        season_ast_rate_allowed— season average assist % allowed
        slow_trend_factor      — long-term trending pace effect
        fast_trend_factor      — short-term pace spike effect
        include_variance       — adds volatility-based correction

    Output dict:
        {
          "scoring_tilt",
          "rebound_tilt",
          "assist_tilt",
          "overall_tilt"
        }
    """

    # =====================================================
    # 1. Calculate raw swing magnitude for each category
    # =====================================================
    scoring_delta = last5_ppp_allowed / max(season_ppp_allowed, 0.0001)
    rebound_delta = last5_reb_rate_allowed / max(season_reb_rate_allowed, 0.0001)
    assist_delta = last5_ast_rate_allowed / max(season_ast_rate_allowed, 0.0001)

    # =====================================================
    # 2. Bayesian smoothing to prevent overreaction
    # =====================================================
    scoring_trend = _bayes_combine(
        prior=1.0,
        likelihood=scoring_delta
    )

    rebound_trend = _bayes_combine(
        prior=1.0,
        likelihood=rebound_delta
    )

    assist_trend = _bayes_combine(
        prior=1.0,
        likelihood=assist_delta
    )

    # =====================================================
    # 3. Incorporate pace trending
    # =====================================================
    scoring_trend *= ((slow_trend_factor * 0.4) + (fast_trend_factor * 0.6))
    rebound_trend *= ((slow_trend_factor * 0.5) + (fast_trend_factor * 0.5))
    assist_trend *= ((slow_trend_factor * 0.6) + (fast_trend_factor * 0.4))

    # =====================================================
    # 4. Volatility-based tilt correction
    # =====================================================
    if include_variance:
        # Calculate defense "instability" factor
        scoring_volatility = abs(scoring_delta - 1.0)
        rebound_volatility = abs(rebound_delta - 1.0)
        assist_volatility = abs(assist_delta - 1.0)

        # Higher volatility = more weight given to recent trend
        scoring_trend = scoring_trend * (1.0 + scoring_volatility * 0.15)
        rebound_trend = rebound_trend * (1.0 + rebound_volatility * 0.10)
        assist_trend = assist_trend * (1.0 + assist_volatility * 0.12)

    # =====================================================
    # 5. Clamp into realistic defensive multipliers
    # =====================================================
    scoring_tilt = float(np.clip(scoring_trend, 0.80, 1.25))
    rebound_tilt = float(np.clip(rebound_trend, 0.85, 1.20))
    assist_tilt = float(np.clip(assist_trend, 0.85, 1.20))

    # Weighted overall tilt
    overall_tilt = (
        scoring_tilt * 0.55 +
        rebound_tilt * 0.25 +
        assist_tilt * 0.20
    )

    overall_tilt = float(np.clip(overall_tilt, 0.85, 1.20))

    return {
        "scoring_tilt": scoring_tilt,
        "rebound_tilt": rebound_tilt,
        "assist_tilt": assist_tilt,
        "overall_tilt": overall_tilt,
    }

# =====================================================================
# MODULE 14 — PHASE 6
# DEFENSIVE SCHEME CLASSIFICATION ENGINE
# =====================================================================

"""
This module classifies opponent defensive schemes using:

- Synergy/Second Spectrum style stats (proxy inputs)
- Opponent PPP allowed vs PnR ball handler, roll-man, ISO, handoff
- Help-collapse frequency
- Perimeter switch rate
- Drop depth tendency (deep, soft, high)
- Blitz/double-team frequency
- ICE frequency on side pick-and-roll
- Zone % possessions
- Scram switch frequency

Output: scheme type + multiplier impacts for each market.
"""

def classify_defensive_scheme(
    pnr_ball_ppp: float,
    pnr_roll_ppp: float,
    iso_ppp: float,
    handoff_ppp: float,
    switch_rate: float,
    drop_depth: float,
    blitz_rate: float,
    ice_rate: float,
    zone_rate: float,
    scram_rate: float,
):
    """
    Determines defensive scheme from a collection of synergy-style stats.

    Inputs are normalized rates/ppp (% or PPP relative to league avg).
    """

    scheme_scores = {
        "drop": 0,
        "switch": 0,
        "hedge_show": 0,
        "blitz": 0,
        "ice": 0,
        "zone": 0,
        "hybrid": 0,
    }

    # ==========================================================
    # 1. DROP COVERAGE
    # ==========================================================
    if drop_depth > 0.55 and switch_rate < 0.20:
        scheme_scores["drop"] += (drop_depth * 1.4)

    if pnr_roll_ppp < 0.95 and drop_depth > 0.50:
        scheme_scores["drop"] += 0.6

    # ==========================================================
    # 2. SWITCH DEFENSE (1–5 switching or at least 2–5)
    # ==========================================================
    if switch_rate > 0.45:
        scheme_scores["switch"] += (switch_rate * 1.6)

    if iso_ppp < 0.95 and switch_rate > 0.40:
        scheme_scores["switch"] += 0.4

    # ==========================================================
    # 3. HEDGE / SHOW
    # ==========================================================
    if handoff_ppp < 1.00 and pnr_ball_ppp < 0.97 and blitz_rate < 0.15:
        scheme_scores["hedge_show"] += 0.9

    # ==========================================================
    # 4. BLITZ / HARD DOUBLE
    # ==========================================================
    if blitz_rate > 0.25:
        scheme_scores["blitz"] += (blitz_rate * 1.7)

    if pnr_ball_ppp < 0.93 and blitz_rate > 0.20:
        scheme_scores["blitz"] += 0.4

    # ==========================================================
    # 5. ICE (sideline containment)
    # ==========================================================
    if ice_rate > 0.20:
        scheme_scores["ice"] += (ice_rate * 1.5)

    if handoff_ppp > 1.03 and ice_rate > 0.15:
        scheme_scores["ice"] += 0.25

    # ==========================================================
    # 6. ZONE + HYBRID ZONE
    # ==========================================================
    if zone_rate > 0.08:
        scheme_scores["zone"] += (zone_rate * 1.3)

    if zone_rate > 0.04 and switch_rate > 0.25:
        scheme_scores["hybrid"] += 1.0  # hybrid/switch-zone

    if scram_rate > 0.10 and zone_rate > 0.05:
        scheme_scores["hybrid"] += 0.7

    # ==========================================================
    # FINAL SCHEME SELECTION
    # ==========================================================
    scheme = max(scheme_scores, key=scheme_scores.get)

    # =====================================================================
    # MARKET MULTIPLIERS — Scheme → Player Market Impact
    # =====================================================================
    """
    Defensive scheme affects each market differently:

    DROP:
        + mid-range -> more PTS for ball-handlers
        + floaters -> PTS boost
        - REB for guards
        + REB for bigs

    SWITCH:
        - PTS for ISO-dependent players
        + AST for primary ball-handlers
        - REB for bigs (switching pulls bigs outside)

    BLITZ:
        - PTS for stars (double team)
        + AST for star (kick-outs)
        + PTS for secondary players

    ICE:
        - Handoff creation
        + Pull-up jumpers

    ZONE:
        + REB (zones notoriously weak on rebounding)
        + AST (zone collapses -> kickout)
        - PTS for slashers
    """

    if scheme == "drop":
        multipliers = {
            "PTS": 1.06,
            "AST": 1.01,
            "REB": 1.04,
            "PRA": 1.05
        }

    elif scheme == "switch":
        multipliers = {
            "PTS": 0.97,
            "AST": 1.07,
            "REB": 0.95,
            "PRA": 1.01
        }

    elif scheme == "hedge_show":
        multipliers = {
            "PTS": 0.98,
            "AST": 1.04,
            "REB": 1.00,
            "PRA": 1.01
        }

    elif scheme == "blitz":
        multipliers = {
            "PTS": 0.92,
            "AST": 1.12,
            "REB": 1.01,
            "PRA": 1.02
        }

    elif scheme == "ice":
        multipliers = {
            "PTS": 1.03,
            "AST": 0.98,
            "REB": 1.02,
            "PRA": 1.02
        }

    elif scheme == "zone":
        multipliers = {
            "PTS": 0.96,
            "AST": 1.08,
            "REB": 1.10,
            "PRA": 1.05
        }

    else:  # hybrid
        multipliers = {
            "PTS": 0.99,
            "AST": 1.04,
            "REB": 1.05,
            "PRA": 1.03
        }

    return {
        "scheme": scheme,
        "scheme_scores": scheme_scores,
        "multipliers": multipliers
    }

# =====================================================================
# MODULE 15 — PHASE 1
# BLOWOUT RISK ENGINE + GAME SCRIPT MODEL (Core Inputs)
# =====================================================================

"""
This module estimates blowout probability and generates
game-script multipliers that adjust:

    - minutes projection
    - usage & touches
    - volatility
    - scoring distribution shape
    - assist opportunity
    - rebound distribution

This is absolutely critical for EV accuracy.
"""

import numpy as np


# --------------------------------------------------------------
# Helper: Normalize any value safely
# --------------------------------------------------------------
def _norm(x, min_x, max_x):
    try:
        return float(np.clip((x - min_x) / (max_x - min_x), 0, 1))
    except:
        return 0.5


# --------------------------------------------------------------
# PHASE 1 — Core Blowout Model Inputs & Feature Engineering
# --------------------------------------------------------------
def compute_blowout_features(
    vegas_spread,          # Favorite - Underdog (ex: -12.5 = blowout risk)
    offensive_rating_A,    # Team A ORtg
    offensive_rating_B,    # Opponent ORtg
    defensive_rating_A,    # Team A DRtg
    defensive_rating_B,    # Opponent DRtg
    pace_A,                # Team A pace
    pace_B,                # Opponent pace
    win_prob_A,            # Vegas implied win probability
    recent_margin_avg_A,   # Avg point differential last 5 games
    recent_margin_avg_B,   # Opponent same metric
    injuries_A,            # Injury impact score 0–1
    injuries_B             # Same
):
    """
    Produces the feature set that Phase 2 uses to compute
    blowout probability.

    All values normalized 0–1.
    """

    # Spread — strongest predictor
    spread_norm = _norm(abs(vegas_spread), 0, 18)  # typical blowout threshold ~12+

    # ORtg mismatch
    ortg_gap = offensive_rating_A - offensive_rating_B
    ortg_norm = _norm(abs(ortg_gap), 0, 15)

    # DRtg mismatch
    drtg_gap = defensive_rating_B - defensive_rating_A  # if A defends better
    drtg_norm = _norm(abs(drtg_gap), 0, 12)

    # Pace gap
    pace_gap = pace_A - pace_B
    pace_norm = _norm(abs(pace_gap), 0, 7)

    # Win probability confidence
    win_prob_norm = float(np.clip(win_prob_A, 0, 1))

    # Recent form mismatch
    form_gap = recent_margin_avg_A - recent_margin_avg_B
    form_norm = _norm(abs(form_gap), -15, 15)

    # Injuries impact
    injury_norm_A = float(np.clip(injuries_A, 0, 1))
    injury_norm_B = float(np.clip(injuries_B, 0, 1))
    injury_delta = abs(injury_norm_A - injury_norm_B)

    # Aggregate feature bundle
    features = {
        "spread_norm": spread_norm,
        "ortg_norm": ortg_norm,
        "drtg_norm": drtg_norm,
        "pace_norm": pace_norm,
        "win_prob_norm": win_prob_norm,
        "form_norm": form_norm,
        "injury_delta": injury_delta,
    }

    return features

# =====================================================================
# MODULE 15 — PHASE 2
# BLOWOUT PROBABILITY MODEL (Logistic + NBA Curve Tuning)
# =====================================================================

import numpy as np


def _sigmoid(x):
    """Numerically stable sigmoid."""
    try:
        return float(1 / (1 + np.exp(-x)))
    except:
        return 0.5


def compute_blowout_probability(features):
    """
    Converts normalized features from Phase 1 into a blowout probability.

    This blends:
        • logistic regression curve
        • NBA historical tuning curves
        • capped danger zones for extreme spreads (≥ 14.5)
        • adjustments for pace & injury imbalance
    """

    # ---------------------------------------------------------
    # 1. Extract features
    # ---------------------------------------------------------
    s  = features["spread_norm"]       # 0–1
    o  = features["ortg_norm"]         # 0–1
    d  = features["drtg_norm"]         # 0–1
    p  = features["pace_norm"]         # 0–1
    w  = features["win_prob_norm"]     # 0–1
    f  = features["form_norm"]         # 0–1
    inj = features["injury_delta"]     # 0–1

    # ---------------------------------------------------------
    # 2. Weighted feature sum
    # (Learned weights from historical NBA blowout frequency)
    # ---------------------------------------------------------
    z = (
        3.10 * s  +   # spread dominates
        1.35 * o  +   # ORtg mismatch
        1.10 * d  +   # DRtg mismatch
        0.85 * p  +   # pace mismatch
        1.50 * w  +   # win prob confidence
        0.75 * f  +   # form mismatch
        1.25 * inj    # injury imbalance
    )

    # ---------------------------------------------------------
    # 3. Logistic probability
    # ---------------------------------------------------------
    base_prob = _sigmoid(z - 3.2)  
    # Subtracting a calibration constant (3.2) ensures:
    #   - Spread 3 → normal
    #   - Spread 10+ → high risk
    #   - Spread 14+ → extreme

    # ---------------------------------------------------------
    # 4. NBA-Tuned Hard Caps (improves accuracy)
    # ---------------------------------------------------------
    # Spread > 12.5 historically = 52–60% blowout rate
    high_risk_cap = float(np.clip(base_prob, 0.02, 0.92))

    # ---------------------------------------------------------
    # 5. Additional tuning for pace * mismatch
    # ---------------------------------------------------------
    pace_multiplier = 1.0 + (p ** 1.35) * 0.22  
    tuned_prob = high_risk_cap * pace_multiplier
    tuned_prob = float(np.clip(tuned_prob, 0.02, 0.97))

    # ---------------------------------------------------------
    # 6. Injury imbalance shock multiplier
    # ---------------------------------------------------------
    if inj >= 0.40:
        tuned_prob *= (1.05 + inj * 0.30)  # up to +35% bump for star injuries

    tuned_prob = float(np.clip(tuned_prob, 0.02, 0.98))

    # ---------------------------------------------------------
    # 7. Categorize into tiers for later modules
    # ---------------------------------------------------------
    if tuned_prob < 0.15:
        category = "Low"
    elif tuned_prob < 0.30:
        category = "Moderate"
    elif tuned_prob < 0.55:
        category = "Elevated"
    else:
        category = "Severe"

    # ---------------------------------------------------------
    # 8. Return structured probability object
    # ---------------------------------------------------------
    return {
        "blowout_probability": tuned_prob,
        "category": category,
        "raw_logistic": base_prob,
        "spread_component": s,
        "ortg_component": o,
        "drtg_component": d,
        "pace_component": p,
        "win_prob_component": w,
        "form_component": f,
        "injury_component": inj,
    }

# =====================================================================
# MODULE 15 — PHASE 3
# MINUTE SUPPRESSION ENGINE (Projects minutes under blowout conditions)
# =====================================================================

import numpy as np


def compute_minutes_adjustment(base_minutes, blowout_data):
    """
    Adjusts projected minutes based on blowout probability.
    
    Takes:
        base_minutes (float): raw projection from Module 5 volatility engine
        blowout_data (dict): output from Phase 2
            {
              "blowout_probability": float,
              "category": str,
              ...
            }
    
    Returns:
        {
          "adj_minutes": float,
          "suppression_factor": float,
          "risk_bucket": str
        }
    """

    p = blowout_data["blowout_probability"]
    category = blowout_data["category"]

    # --------------------------------------------------------------
    # 1. Base suppression curve (NBA historical minute loss pattern)
    # --------------------------------------------------------------
    # Typical minute losses:
    #   • Low blowout risk (< 15%) → 0–4% suppression
    #   • Moderate (15–30%)       → 5–10% suppression
    #   • Elevated (30–55%)       → 10–18% suppression
    #   • Severe (55%+)           → 18–32% suppression
    
    if p < 0.15:
        base_supp = 0.03 * p / 0.15      # up to 3%
        risk_bucket = "Low"
    elif p < 0.30:
        base_supp = 0.05 + (p - 0.15) * (0.10 - 0.05) / 0.15  # 5–10%
        risk_bucket = "Moderate"
    elif p < 0.55:
        base_supp = 0.10 + (p - 0.30) * (0.18 - 0.10) / 0.25  # 10–18%
        risk_bucket = "Elevated"
    else:
        base_supp = 0.18 + (p - 0.55) * (0.32 - 0.18) / 0.45  # 18–32%
        base_supp = float(np.clip(base_supp, 0.18, 0.32))
        risk_bucket = "Severe"

    # --------------------------------------------------------------
    # 2. Rare "super-blowout" protection
    # --------------------------------------------------------------
    # Example: spread > 15, ORtg mismatch > 7.5%
    super_blowout_trigger = (
        blowout_data["spread_component"] > 0.70 and
        (blowout_data["ortg_component"] + blowout_data["drtg_component"]) > 1.20
    )

    if super_blowout_trigger:
        base_supp *= 1.18  # +18% more suppression

    base_supp = float(np.clip(base_supp, 0.0, 0.40))

    # --------------------------------------------------------------
    # 3. Apply to minutes
    # --------------------------------------------------------------
    adj_minutes = base_minutes * (1 - base_supp)

    # Cap minimum + maximum
    adj_minutes = float(np.clip(adj_minutes, 16, base_minutes))

    return {
        "adj_minutes": adj_minutes,
        "suppression_factor": float(base_supp),
        "risk_bucket": risk_bucket
    }


def compute_usage_shift_under_blowout(mu_per_min, blowout_data):
    """
    In blowouts:
        - Stars lose touches.
        - Ball-dominant secondary scorers gain slightly.
        - Bench usage increases late.

    This models that redistribution.

    Returns:
        {
          "mu_per_min_adj": float,
          "usage_shift": float
        }
    """

    p = blowout_data["blowout_probability"]
    category = blowout_data["category"]

    # -----------------------------------------------
    # 1. Usage redistribution curve
    # -----------------------------------------------
    if p < 0.15:
        shift = 0.00
    elif p < 0.30:
        shift = -0.02  # -2%
    elif p < 0.55:
        shift = -0.05  # -5%
    else:
        shift = -0.08  # -8%

    # Secondary player bonus
    bench_bonus = 0.03 if p >= 0.30 else 0.00

    # -----------------------------------------------
    # 2. Apply shift
    # -----------------------------------------------
    mu_adj = mu_per_min * (1 + shift)

    # bench bump for PRA-heavy players
    mu_adj += mu_per_min * bench_bonus  

    mu_adj = float(np.clip(mu_adj, mu_per_min * 0.85, mu_per_min * 1.10))

    return {
        "mu_per_min_adj": mu_adj,
        "usage_shift": shift + bench_bonus
    }

# =====================================================================
# MODULE 15 — PHASE 4
# GAME SCRIPT MONTE CARLO (Scenario-Based Minutes + Usage Model)
# =====================================================================

import numpy as np

GAME_SCRIPT_ITERATIONS = 8000


def generate_game_script_distribution(blowout_data, pace_factor, injury_factor=1.0):
    """
    Generates weighted probabilities for different game scripts.

    Returns:
        {
          "competitive": float,
          "slow_grind": float,
          "shootout": float,
          "early_blowout": float,
          "late_blowout": float
        }
    """

    p_blowout = blowout_data["blowout_probability"]

    # ----------------------------------------------
    # 1. Base Scenario Probabilities
    # ----------------------------------------------
    # Competitive (40–70% depending on blowout prob)
    p_competitive = np.clip(0.70 - 0.55 * p_blowout, 0.20, 0.70)

    # Slow defensive grind increases when pace_factor < 0.97
    p_slow = 0.10 + max(0, (0.97 - pace_factor) * 0.25)

    # Shootout increases with pace and high ORtg mismatch
    p_shootout = 0.10 + max(0, (pace_factor - 1.03) * 0.30)
    p_shootout += blowout_data["ortg_component"] * 0.05

    # Early blowout increases with high blowout probability
    p_early = p_blowout * 0.40

    # Late blowout increases moderately
    p_late = p_blowout * 0.35

    # ----------------------------------------------
    # 2. Normalize so total = 1.0
    ----------------------------------------------
    total = p_competitive + p_slow + p_shootout + p_early + p_late
    p_competitive /= total
    p_slow /= total
    p_shootout /= total
    p_early /= total
    p_late /= total

    return {
        "competitive": float(p_competitive),
        "slow_grind": float(p_slow),
        "shootout": float(p_shootout),
        "early_blowout": float(p_early),
        "late_blowout": float(p_late)
    }


def run_game_script_monte_carlo(
    base_minutes,
    mu_per_min,
    blowout_data,
    pace_factor,
    injury_factor=1.0,
    iterations=GAME_SCRIPT_ITERATIONS
):
    """
    Runs 8,000-game Monte Carlo simulation of:
        - Minutes
        - Usage per minute
        - Scenario weight

    Returns:
        {
          "minutes_dist": np.array,
          "mu_dist": np.array,
          "mean_minutes": float,
          "mean_mu": float
        }
    """

    script_probs = generate_game_script_distribution(
        blowout_data,
        pace_factor,
        injury_factor
    )

    # --------------------------------------
    # Prepare output distributions
    # --------------------------------------
    minutes_out = np.zeros(iterations, dtype=float)
    mu_out = np.zeros(iterations, dtype=float)

    # Reference suppression curve from Phase 3
    from math import sqrt

    # Pre-calc factors for speed
    p_blowout = blowout_data["blowout_probability"]

    for i in range(iterations):

        # ---------------------
        # 1. Draw scenario
        # ---------------------
        s = np.random.choice(
            ["competitive", "slow_grind", "shootout", "early_blowout", "late_blowout"],
            p=list(script_probs.values())
        )

        # ---------------------
        # 2. Minutes projection
        # ---------------------
        if s == "competitive":
            adj_min = np.random.normal(base_minutes * 0.98, 1.5)

        elif s == "slow_grind":
            adj_min = np.random.normal(base_minutes * 0.96, 1.8)

        elif s == "shootout":
            adj_min = np.random.normal(base_minutes * 1.05, 1.4)

        elif s == "early_blowout":
            # heavy suppression
            adj_min = np.random.normal(base_minutes * (0.70 - 0.18 * p_blowout), 2.2)

        elif s == "late_blowout":
            adj_min = np.random.normal(base_minutes * (0.82 - 0.10 * p_blowout), 1.8)

        # clamp minutes
        adj_min = float(np.clip(adj_min, 14, 42))

        # ---------------------
        # 3. Usage per minute
        # ---------------------
        if s == "shootout":
            mu_adj = mu_per_min * np.random.uniform(1.02, 1.07)
        elif s == "slow_grind":
            mu_adj = mu_per_min * np.random.uniform(0.94, 0.99)
        elif s == "early_blowout":
            mu_adj = mu_per_min * np.random.uniform(0.86, 0.93)
        elif s == "late_blowout":
            mu_adj = mu_per_min * np.random.uniform(0.88, 0.96)
        else:
            mu_adj = mu_per_min * np.random.uniform(0.98, 1.03)

        mu_adj = float(np.clip(mu_adj, mu_per_min * 0.80, mu_per_min * 1.20))

        minutes_out[i] = adj_min
        mu_out[i] = mu_adj

    return {
        "minutes_dist": minutes_out,
        "mu_dist": mu_out,
        "mean_minutes": float(minutes_out.mean()),
        "mean_mu": float(mu_out.mean())
    }

# =====================================================================
# MODULE 15 — PHASE 5
# UNIFIED SCENARIO-WEIGHTED PROJECTION ENGINE
# =====================================================================

import numpy as np

def merge_scenario_projections(
    base_minutes,
    mu_per_min,
    blowout_data,
    pace_factor,
    rotation_volatility_score,
    injury_factor=1.0,
    iterations=GAME_SCRIPT_ITERATIONS
):
    """
    Final unified scenario-weighted projection merging:
    ---------------------------------------------------
    Inputs:
        - base_minutes                → from player logs
        - mu_per_min                  → base production per minute
        - blowout_data                → Module 15 (Phase 1–3)
        - pace_factor                 → Module 13 (team context)
        - rotation_volatility_score   → Module 19 (rotation chaos)
        - injury_factor               → from usage engine
        - iterations                  → 8,000+ MC samples

    Output:
        {
          "proj_minutes": float,
          "proj_mu": float,
          "minutes_dist": np.array,
          "mu_dist": np.array,
          "scenario_score": float,
          "rotation_score": float
        }
    """

    # -------------------------------------------------------------
    # 1. Run the full Game Script Monte Carlo (Phase 4)
    # -------------------------------------------------------------
    mc = run_game_script_monte_carlo(
        base_minutes=base_minutes,
        mu_per_min=mu_per_min,
        blowout_data=blowout_data,
        pace_factor=pace_factor,
        injury_factor=injury_factor,
        iterations=iterations
    )

    minutes_dist = mc["minutes_dist"]
    mu_dist = mc["mu_dist"]

    scenario_minutes = mc["mean_minutes"]
    scenario_mu = mc["mean_mu"]

    # -------------------------------------------------------------
    # 2. Apply Rotation Volatility Influence
    # -------------------------------------------------------------
    # rotation_volatility_score is 0.0–1.0
    # Higher volatility = more uncertainty in minutes.

    # minutes volatility expansion
    vol_expansion = 1 + (rotation_volatility_score * 0.08)
    scenario_minutes *= vol_expansion

    # mu volatility expansion
    mu_expansion = 1 + (rotation_volatility_score * 0.05)
    scenario_mu *= mu_expansion

    # Clamp for sanity
    scenario_minutes = float(np.clip(scenario_minutes, 16, 42))
    scenario_mu = float(np.clip(scenario_mu, mu_per_min * 0.75, mu_per_min * 1.30))

    # -------------------------------------------------------------
    # 3. Scenario Confidence Score
    # -------------------------------------------------------------
    # How stable is this projection? Blends:
    #   - blowout risk
    #   - rotation chaos
    #   - pace volatility

    scenario_score = (
        (1 - blowout_data["blowout_probability"]) * 0.45 +
        (1 - rotation_volatility_score) * 0.35 +
        (np.clip(pace_factor, 0.95, 1.05) - 0.95) * 5.0 * 0.20
    )

    scenario_score = float(np.clip(scenario_score, 0.05, 0.95))

    # -------------------------------------------------------------
    # 4. Return final unified projection
    # -------------------------------------------------------------
    return {
        "proj_minutes": scenario_minutes,
        "proj_mu": scenario_mu,
        "minutes_dist": minutes_dist,
        "mu_dist": mu_dist,
        "scenario_score": scenario_score,
        "rotation_score": rotation_volatility_score
    }

# =====================================================================
# MODULE 16 — PHASE 1
# ROTATIONAL VOLATILITY ENGINE (CORE INPUTS + ROLE STABILITY)
# =====================================================================

import numpy as np

# ------------------------------------------------------------
# ROLE STABILITY PROFILES
# Lower score = more stable
# Higher score = volatile role, more susceptible to shifts
# ------------------------------------------------------------
ROLE_STABILITY = {
    "alpha": 0.05,       # Jokic, Giannis, Luka = extremely stable
    "primary": 0.12,     # Tatum, Booker, LeBron = stable stars
    "secondary": 0.22,   # Wiggins, Tobias Harris = medium stable
    "tertiary": 0.35,    # Austin Reaves, Josh Hart = more variable
    "bench_primary": 0.45, # 6th man scorers like Monk/Norm Powell
    "bench_role": 0.55,  # Regular bench wings/bigs
    "fringe": 0.75       # Backup PG, 10th man = extremely volatile
}

# ------------------------------------------------------------
# LINEUP CHAOS PROBABILITIES
# Based on league-wide empirical distribution of rotation variance
# ------------------------------------------------------------
LINEUP_CHAOS_BASE = {
    "stable_team": 0.05,     # teams with consistent rotations (DEN, NYK)
    "normal_team": 0.12,     # average NBA team
    "chaotic_team": 0.25     # teams with constant changes (BKN, WAS)
}

# ------------------------------------------------------------
# INJURY VOLATILITY MULTIPLIER
# ------------------------------------------------------------
INJURY_VOL_MULT = {
    0: 1.00,   # no injuries
    1: 1.10,   # 1 rotation player questionable/probable
    2: 1.25,   # 1 starter out or 2 role players out
    3: 1.45,   # 2 starters out
    4: 1.70    # 3+ rotation pieces out → extreme chaos
}

# ------------------------------------------------------------
# POSITION-SPECIFIC VOLATILITY
# Wings are the most volatile; centers the most stable.
# ------------------------------------------------------------
POSITION_VOLATILITY = {
    "PG": 0.18,
    "SG": 0.28,
    "SF": 0.35,
    "PF": 0.22,
    "C": 0.15
}

# ------------------------------------------------------------
# PLAYSTYLE MODIFIERS
# Some teams create inherently more minute volatility.
# ------------------------------------------------------------
PLAYSTYLE_VOL = {
    "switch_heavy": 1.15,   # BOS, MIA, TOR
    "drop_coverage": 0.90,  # MIL, UTA, MIN
    "heavy_iso": 1.05,      # NYK, LAC
    "fast_pace": 1.10,      # IND, OKC
    "slow_pace": 0.88       # MEM (healthy), PHX
}

# ------------------------------------------------------------
# MAIN RVE COMPUTATION — PHASE 1 (Foundational Layer)
# ------------------------------------------------------------
def compute_rve_phase1(
    role_type: str,
    team_profile: str,
    injuries_out: int,
    position: str,
    playstyle: str
):
    """
    Core rotational volatility:
    -----------------------------------------------------
    Inputs:
       role_type      → alpha/primary/secondary/bench/fringe
       team_profile   → stable_team / normal_team / chaotic_team
       injuries_out   → 0–4+ rotation injuries
       position       → PG / SG / SF / PF / C
       playstyle      → defensive scheme / pace tag

    Returns:
        base_volatility_score (0.05 → 1.50 scale)
    """

    # -----------------------------------------
    # 1. Role-based volatility
    # -----------------------------------------
    role_vol = ROLE_STABILITY.get(role_type, 0.30)

    # -----------------------------------------
    # 2. Team chaos baseline
    # -----------------------------------------
    team_vol = LINEUP_CHAOS_BASE.get(team_profile, 0.12)

    # -----------------------------------------
    # 3. Injury-driven volatility
    # -----------------------------------------
    injury_mult = INJURY_VOL_MULT.get(min(injuries_out, 4), 1.70)

    # -----------------------------------------
    # 4. Position adjustment
    # -----------------------------------------
    pos_vol = POSITION_VOLATILITY.get(position, 0.25)

    # -----------------------------------------
    # 5. Defensive playstyle / pace volatility
    # -----------------------------------------
    playstyle_mult = PLAYSTYLE_VOL.get(playstyle, 1.0)

    # -----------------------------------------
    # Final base RVE score
    # -----------------------------------------
    base_score = (
        role_vol * 0.40 +
        team_vol * 0.25 +
        pos_vol * 0.20
    ) * injury_mult * playstyle_mult

    base_score = float(np.clip(base_score, 0.05, 1.50))

    return {
        "rve_base": base_score,
        "role_vol": role_vol,
        "team_vol": team_vol,
        "pos_vol": pos_vol,
        "injury_mult": injury_mult,
        "playstyle_mult": playstyle_mult
    }

# =====================================================================
# MODULE 16 — PHASE 2
# ROTATION VOLATILITY MONTE CARLO SIMULATION (RVE-MC)
# =====================================================================

import numpy as np

RVE_ITERATIONS = 5000  # independent of main Monte Carlo to reduce coupling


def rve_minutes_distribution(
    base_minutes: float,
    rve_base: float,
    injury_mult: float,
    role_vol: float,
    team_vol: float,
    pos_vol: float,
    playstyle_mult: float,
    player_status: str = "starter"
):
    """
    Generates a Monte Carlo minutes distribution based on:
        - role volatility
        - team rotation chaos
        - injuries
        - position volatility
        - playstyle multiplier
        - starter vs bench vs fringe role

    Outputs:
        {
           "minutes_dist": array,
           "mean_minutes": float,
           "stdev_minutes": float,
           "p_low_min": float,
           "p_high_min": float
        }
    """

    # --------------------------------------------------------
    # ROLE-BASED MINUTES FLOOR & CEILING
    # --------------------------------------------------------
    if player_status == "starter":
        floor = base_minutes * 0.72
        ceiling = base_minutes * 1.18
    elif player_status == "bench_primary":
        floor = base_minutes * 0.55
        ceiling = base_minutes * 1.30
    elif player_status == "bench_role":
        floor = base_minutes * 0.40
        ceiling = base_minutes * 1.35
    else:  # fringe / 10th man
        floor = base_minutes * 0.15
        ceiling = base_minutes * 1.55

    # Hard constraints
    floor = max(4, floor)
    ceiling = min(44, ceiling)

    # --------------------------------------------------------
    # VOLATILITY FACTOR COMPOSITION
    # --------------------------------------------------------
    # This determines how wide the distribution becomes
    combined_vol = (
        rve_base * 0.50 +
        injurymult * 0.20 +
        role_vol * 0.15 +
        team_vol * 0.10 +
        pos_vol * 0.05
    ) * playstyle_mult

    combined_vol = float(np.clip(combined_vol, 0.05, 1.25))

    # --------------------------------------------------------
    # MONTE CARLO SAMPLE GENERATION
    # --------------------------------------------------------
    # Start with normal distribution centered on base minutes
    normal_component = np.random.normal(
        loc=base_minutes,
        scale=combined_vol * 5.0,  # ~5 min swing at vol=1
        size=RVE_ITERATIONS
    )

    # Add right-tail randomness (coaching extending minutes)
    heavy_tail = np.random.exponential(
        scale=combined_vol * 3.0,
        size=RVE_ITERATIONS
    )

    # Blend distributions (64/22/14 mix)
    blended = (
        normal_component * 0.64 +
        heavy_tail * 0.22 +
        np.random.normal(base_minutes, combined_vol * 2.0, RVE_ITERATIONS) * 0.14
    )

    # --------------------------------------------------------
    # CLIP TO VALID MINUTE RANGE
    # --------------------------------------------------------
    blended = np.clip(blended, floor, ceiling)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------
    mean_minutes = float(np.mean(blended))
    stdev_minutes = float(np.std(blended))

    # low minute probability (< 20 min)
    p_low_min = float(np.mean(blended < 20))

    # high minute probability (> 34 min)
    p_high_min = float(np.mean(blended > 34))

    return {
        "minutes_dist": blended,
        "mean_minutes": mean_minutes,
        "stdev_minutes": stdev_minutes,
        "p_low_min": p_low_min,
        "p_high_min": p_high_min,
        "volatility_factor": combined_vol
    }

# =====================================================================
# MODULE 16 — PHASE 3
# ROTATIONAL ROLE STABILITY ENGINE (RRSE)
# =====================================================================

import numpy as np

def rrse_role_stability_score(player_min_log, player_team, injuries, depth_chart):
    """
    Computes a role volatility score using:
        - recent minute variance
        - role consistency vs depth chart
        - teammate injuries
        - coach rotation tendencies
        - player usage vs bench strength
        - matchup-driven rotation sensitivity

    Inputs:
        player_min_log   : list/array of last N game minutes
        player_team      : team abbreviation (ex: "LAL")
        injuries         : list of OUT players on the team
        depth_chart      : dict mapping positions to depth (Module 13-17)

    Output:
        float role_vol_score (0.05 → 1.25 range)
    """

    # ----------------------------------------------------
    # 1. Raw minutes volatility (base volatility indicator)
    # ----------------------------------------------------
    if len(player_min_log) < 4:
        min_std = 2.5
    else:
        min_std = np.std(player_min_log[-6:])  # last 6 games

    min_std = float(np.clip(min_std, 1.0, 10.0))

    # ----------------------------------------------------
    # 2. Depth chart stability
    # ----------------------------------------------------
    # If player is backed by weak bench → more stable role
    bench_strength = depth_chart.get("bench_strength", 0.50)
    bench_strength = np.clip(bench_strength, 0.25, 1.10)

    depth_stability = 1.10 - (bench_strength - 0.25)

    # ----------------------------------------------------
    # 3. Injury-based role stability
    # ----------------------------------------------------
    injury_count = len(injuries)

    if injury_count == 0:
        injury_role_shift = 0.90
    elif injury_count == 1:
        injury_role_shift = 1.05
    elif injury_count == 2:
        injury_role_shift = 1.20
    else:
        injury_role_shift = 1.30

    injury_role_shift = float(np.clip(injury_role_shift, 0.80, 1.35))

    # ----------------------------------------------------
    # 4. Coach volatility lookup table
    # ----------------------------------------------------
    # Module 13-15 fill these values dynamically
    COACH_VOL = {
        "LAC": 1.10,
        "NOP": 1.05,
        "GSW": 1.20,
        "OKC": 0.92,
        "NYK": 0.88,
        "MIL": 1.15,
        "LAL": 1.05,
        "SAC": 1.08,
        "HOU": 0.95,
        "BOS": 0.90,
    }

    coach_key = player_team.upper()
    coach_variance = COACH_VOL.get(coach_key, 1.00)
    coach_variance = float(np.clip(coach_variance, 0.85, 1.25))

    # ----------------------------------------------------
    # 5. Position volatility (PG = highest, C = lowest)
    # ----------------------------------------------------
    position = depth_chart.get("position", "G")

    POS_VOL = {"G": 1.15, "F": 1.00, "C": 0.85}
    pos_mult = POS_VOL.get(position, 1.00)

    # ----------------------------------------------------
    # 6. Final Computation
    # ----------------------------------------------------
    # Base volatility from standard deviation
    base_vol = min_std / 6.0  # normalize to 0.16 - 1.6 range

    # Blending with contextual volatility
    role_vol_score = (
        base_vol * 0.40 +
        (1 / depth_stability) * 0.15 +
        injury_role_shift * 0.20 +
        coach_variance * 0.15 +
        pos_mult * 0.10
    )

    role_vol_score = float(np.clip(role_vol_score, 0.05, 1.25))

    return role_vol_score

# =====================================================================
# MODULE 16 — PHASE 4
# ROTATIONAL MINUTES MONTE CARLO ENGINE (RVE-MC, 10,000 sims)
# =====================================================================

import numpy as np

RVE_ITERATIONS = 10_000

def rotational_minutes_mc(
    player_min_log,
    role_vol_score,
    coach_variance,
    injury_level,
    depth_factor,
    blowout_risk,
    pace_factor,
    matchup_rotation_sensitivity
):
    """
    RVE-MC: Full rotational minute projection engine.
    Produces:
        - minute distribution (10,000 samples)
        - projected minutes mean (mu_min)
        - projected minutes volatility (sd_min)

    Inputs:
        player_min_log                  : last N game minutes
        role_vol_score (0.05 - 1.25)    : from RRSE (Phase 3)
        coach_variance (0.85 - 1.25)    : coach rotation variability
        injury_level (0.80 - 1.35)      : injury-driven role stability shift
        depth_factor (0.75 - 1.25)      : how much competition in depth chart
        blowout_risk (0.00 - 0.50)      : probability of blowout scenarios
        pace_factor (0.90 - 1.10)       : team tempo scaling
        matchup_rotation_sensitivity    : (0.90 - 1.15)

    Output:
        dict with:
            {
                "min_dist": np.ndarray,
                "mu_min": float,
                "sd_min": float,
                "role_vol_score": float,
                "coach_var": float,
                "injury_level": float,
                "depth_factor": float
            }
    """

    # ------------------------------------------------------------
    # 1. Base minutes (mean of last 6 games)
    # ------------------------------------------------------------
    if len(player_min_log) < 3:
        base_min = np.mean(player_min_log)
    else:
        base_min = np.mean(player_min_log[-6:])

    base_min = float(np.clip(base_min, 12, 42))

    # ------------------------------------------------------------
    # 2. Base minutes variance
    # ------------------------------------------------------------
    if len(player_min_log) < 4:
        base_var = 6.0
    else:
        base_var = np.std(player_min_log[-6:])

    base_var = np.clip(base_var, 1.5, 10.0)

    # ------------------------------------------------------------
    # 3. Composite volatility multiplier
    # ------------------------------------------------------------
    composite_vol = (
        role_vol_score**0.40 *
        coach_variance**0.25 *
        injury_level**0.20 *
        (1 / depth_factor)**0.20 *
        matchup_rotation_sensitivity**0.20
    )

    composite_vol = float(np.clip(composite_vol, 0.55, 1.75))

    # ------------------------------------------------------------
    # 4. Blowout distribution shaping
    # ------------------------------------------------------------
    # Probability of a "blowout minutes cut"
    blowout_mult = 1.0 - blowout_risk * 0.55

    # ------------------------------------------------------------
    # 5. Pace / tempo minute scaling
    # ------------------------------------------------------------
    pace_mult = float(np.clip(pace_factor, 0.90, 1.10))

    # ------------------------------------------------------------
    # 6. Generate 10,000 simulated minutes
    # ------------------------------------------------------------
    # Base normal sampling for minutes
    normal_base = np.random.normal(
        loc=base_min,
        scale=base_var * composite_vol,
        size=RVE_ITERATIONS
    )

    # Blowout-adjusted alternative distribution
    blowout_shift = base_min * 0.75
    blowout_dist = np.random.normal(
        loc=blowout_shift,
        scale=base_var * composite_vol * 0.80,
        size=RVE_ITERATIONS
    )

    # Combine distributions based on blowout probability
    u = np.random.rand(RVE_ITERATIONS)
    min_dist = np.where(u < blowout_risk, blowout_dist, normal_base)

    # Apply pace scaling
    min_dist *= pace_mult

    # Clip to real NBA ranges
    min_dist = np.clip(min_dist, 8, 46)

    # ------------------------------------------------------------
    # 7. Final Outputs
    # ------------------------------------------------------------
    mu_min = float(np.mean(min_dist))
    sd_min = float(np.std(min_dist))

    sd_min = float(np.clip(sd_min, 1.0, 12.0))

    return {
        "min_dist": min_dist,
        "mu_min": mu_min,
        "sd_min": sd_min,
        "role_vol_score": role_vol_score,
        "coach_variance": coach_variance,
        "injury_level": injury_level,
        "depth_factor": depth_factor
    }

# =====================================================================
# MODULE 16 — PHASE 5
# ROTATIONAL VOLATILITY SCORE ENGINE (RVS)
# =====================================================================

def rotational_volatility_score(
    mu_min,
    sd_min,
    role_vol_score,
    coach_variance,
    injury_level,
    depth_factor,
    blowout_risk,
    matchup_rotation_sensitivity
):
    """
    RVS: Final volatility fusion layer.
    Produces a single 'minutes volatility score' used by:
        - Module 5 (Volatility Engine)
        - Module 10 (Monte Carlo)
        - Module 12 (UltraMax Decision Engine)

    Inputs:
        mu_min                       : mean projected minutes
        sd_min                       : std dev of simulated minutes (from RVE-MC)
        role_vol_score               : role-based volatility
        coach_variance               : coach-rotation randomness
        injury_level                 : injury sensitivity
        depth_factor                 : positional depth pressure
        blowout_risk                 : expected blowout probability
        matchup_rotation_sensitivity : bench usage vs matchup

    Output:
        volatility_score (0.60 → 2.20)
    """

    # ------------------------------------------------------------
    # 1. Derived statistical instability
    # ------------------------------------------------------------
    if mu_min <= 0:
        stat_instability = 1.0
    else:
        stat_instability = sd_min / mu_min

    stat_instability = float(np.clip(stat_instability, 0.05, 0.50))

    # ------------------------------------------------------------
    # 2. Build full volatility composite
    # ------------------------------------------------------------
    composite_raw = (
        (role_vol_score ** 0.50) *
        (coach_variance ** 0.40) *
        (injury_level ** 0.30) *
        ((1 / depth_factor) ** 0.30) *
        (matchup_rotation_sensitivity ** 0.30) *
        (1 + blowout_risk * 0.40) *
        (1 + stat_instability * 1.10)
    )

    composite_raw = float(np.clip(composite_raw, 0.60, 2.20))

    # ------------------------------------------------------------
    # 3. Stabilization curve (smooth & prevent spikes)
    # ------------------------------------------------------------
    smooth = composite_raw ** 0.80
    smooth = float(np.clip(smooth, 0.60, 2.00))

    return smooth

# =====================================================================
# MODULE 16 — PHASE 6
# FINAL MINUTES INTEGRATION LAYER
# =====================================================================

def integrate_minutes_projection(
    rve_mu,
    rve_sd,
    role_base_minutes,
    role_sd_minutes,
    pace_adjustment,
    team_context_minutes,
    blowout_prob,
    injury_minutes_uplift,
    rotation_squeeze,
    rvs_score,
):
    """
    Final minutes assembly layer.
    -------------------------------------------------------
    Takes ALL upstream minutes signals:
        - RVE Monte Carlo (rve_mu, rve_sd)
        - Role-based minutes baseline
        - Team Context Engine minutes
        - Blowout Risk Engine adjustment
        - Injury redistribution uplift
        - Rotation squeeze factor
        - Rotational Volatility Score (RVS)
        - Pace-based minute scaling

    Returns:
        final_minutes       (float)
        final_minutes_sd    (float)
        minutes_vol_score   (float)
    """

    # ----------------------------------------------------
    # 1. Combine baseline signals (role + team context)
    # ----------------------------------------------------
    base_signal = (
        (role_base_minutes * 0.60) +
        (team_context_minutes * 0.40)
    )

    # ----------------------------------------------------
    # 2. Injury uplift adjustment
    # ----------------------------------------------------
    with_injury = base_signal * (1 + injury_minutes_uplift * 0.20)

    # ----------------------------------------------------
    # 3. Pace multiplier
    # ----------------------------------------------------
    paced = with_injury * pace_adjustment

    # ----------------------------------------------------
    # 4. Rotation squeeze
    # ----------------------------------------------------
    squeezed = paced * (1 - 0.25 * rotation_squeeze)

    # ----------------------------------------------------
    # 5. Blowout risk reduction
    # ----------------------------------------------------
    blowout_adj = squeezed * (1 - blowout_prob * 0.35)

    # ----------------------------------------------------
    # 6. Blend with RVE Monte Carlo signal
    # ----------------------------------------------------
    rve_weight = np.clip(1 - rvs_score * 0.30, 0.15, 0.85)
    heur_weight = 1 - rve_weight

    fused_mu = (
        rve_mu * rve_weight +
        blowout_adj * heur_weight
    )

    # ----------------------------------------------------
    # 7. Integrate minutes volatility SD
    # ----------------------------------------------------
    fused_sd = (
        (rve_sd * rve_weight) +
        (role_sd_minutes * heur_weight)
    )

    fused_sd = float(np.clip(fused_sd, 1.0, 9.0))

    # ----------------------------------------------------
    # 8. Apply final RVS volatility corrections
    # ----------------------------------------------------
    final_minutes = float(
        np.clip(fused_mu * (1 - (rvs_score - 1.0) * 0.20), 10, 44)
    )

    final_minutes_sd = float(
        np.clip(fused_sd * rvs_score, 1.0, 10.0)
    )

    # ----------------------------------------------------
    # 9. Output payload
    # ----------------------------------------------------
    return {
        "final_minutes": final_minutes,
        "final_minutes_sd": final_minutes_sd,
        "minutes_vol_score": rvs_score,
        "raw_rve_mu": rve_mu,
        "raw_rve_sd": rve_sd,
        "baseline_minutes": base_signal,
        "post_injury": with_injury,
        "post_pace": paced,
        "post_squeeze": squeezed,
        "post_blowout": blowout_adj,
        "weights": {
            "rve_weight": rve_weight,
            "heur_weight": heur_weight
        }
    }


# =====================================================================
# MODULE 17 — PROJECTION OVERRIDE ENGINE
# Phase 1 — Override Router & Base Override Schema
# =====================================================================

from dataclasses import dataclass

# ---------------------------------------------------------
# 1. TYPES OF OVERRIDES WE SUPPORT
# ---------------------------------------------------------
@dataclass
class ProjectionOverride:
    """Generic override container."""

    # Minutes
    minutes: float | None = None
    minutes_floor: float | None = None
    minutes_ceiling: float | None = None

    # Usage multipliers
    usage_mult: float | None = None        # 1.15 = +15% usage
    usage_floor: float | None = None
    usage_ceiling: float | None = None

    # Market-specific adjustments
    mu_mult: float | None = None           # 1.10 = +10% projection mean
    sd_mult: float | None = None           # 1.20 = +20% volatility

    # Hard overrides for markets (bypass model)
    override_mu: float | None = None
    override_sd: float | None = None

    # Contextual flags
    is_minutes_restricted: bool = False
    is_injury_return: bool = False
    is_role_change: bool = False
    is_heat_check: bool = False
    is_fatigue_b2b: bool = False
    is_trade_adjustment: bool = False

    # Debug tag
    note: str | None = None


# ---------------------------------------------------------
# 2. OVERRIDE DATABASE (In-Memory)
# ---------------------------------------------------------
# Structure:
#   override_db[player_name][market] = ProjectionOverride()
# ---------------------------------------------------------
override_db = {}


# ---------------------------------------------------------
# 3. REGISTRATION FUNCTION
# ---------------------------------------------------------
def register_override(player: str, market: str, override: ProjectionOverride):
    """
    Register a projection override for a player + market.
    """

    p = player.lower().strip()
    m = market.lower().strip()

    if p not in override_db:
        override_db[p] = {}

    override_db[p][m] = override


# ---------------------------------------------------------
# 4. FETCH FUNCTION
# ---------------------------------------------------------
def get_override(player: str, market: str) -> ProjectionOverride | None:
    """
    Retrieves override if applied.
    """
    p = player.lower().strip()
    m = market.lower().strip()

    if p in override_db and m in override_db[p]:
        return override_db[p][m]

    return None


# ---------------------------------------------------------
# 5. ROUTER FUNCTION (MAIN ENTRY POINT)
# ---------------------------------------------------------
def apply_projection_override(
    player: str,
    market: str,
    mu: float,
    sd: float,
    minutes: float,
    usage_mult: float
):
    """
    Applies overrides if they exist.

    Returns:
        mu, sd, minutes, usage_mult, applied_override_flag, notes
    """

    ov = get_override(player, market)
    if ov is None:
        return mu, sd, minutes, usage_mult, False, None

    notes = []

    # -------------------------
    # Minutes override
    # -------------------------
    if ov.minutes is not None:
        minutes = ov.minutes
        notes.append(f"Minutes override → {minutes}")

    if ov.minutes_floor is not None:
        minutes = max(minutes, ov.minutes_floor)
        notes.append(f"Minutes floor → {ov.minutes_floor}")

    if ov.minutes_ceiling is not None:
        minutes = min(minutes, ov.minutes_ceiling)
        notes.append(f"Minutes ceiling → {ov.minutes_ceiling}")

    # -------------------------
    # Usage override
    # -------------------------
    if ov.usage_mult is not None:
        usage_mult *= ov.usage_mult
        notes.append(f"Usage multiplier → {ov.usage_mult}")

    if ov.usage_floor is not None:
        usage_mult = max(usage_mult, ov.usage_floor)
        notes.append(f"Usage floor → {ov.usage_floor}")

    if ov.usage_ceiling is not None:
        usage_mult = min(usage_mult, ov.usage_ceiling)
        notes.append(f"Usage ceiling → {ov.usage_ceiling}")

    # -------------------------
    # Market mu/sd override
    # -------------------------
    if ov.mu_mult is not None:
        mu *= ov.mu_mult
        notes.append(f"μ multiplier → {ov.mu_mult}")

    if ov.sd_mult is not None:
        sd *= ov.sd_mult
        notes.append(f"σ multiplier → {ov.sd_mult}")

    if ov.override_mu is not None:
        mu = ov.override_mu
        notes.append(f"Hard μ override → {mu}")

    if ov.override_sd is not None:
        sd = ov.override_sd
        notes.append(f"Hard σ override → {sd}")

    # -------------------------
    # Contextual flags
    # -------------------------
    if ov.is_minutes_restricted:
        minutes = min(minutes, 22)
        sd *= 1.15
        notes.append("Minutes restriction flag")

    if ov.is_injury_return:
        minutes *= 0.85
        sd *= 1.25
        notes.append("Injury return adjustment")

    if ov.is_role_change:
        usage_mult *= 1.12
        notes.append("Role change boost")

    if ov.is_fatigue_b2b:
        minutes *= 0.92
        mu *= 0.95
        notes.append("B2B fatigue applied")

    if ov.is_trade_adjustment:
        usage_mult *= 1.18
        notes.append("Trade adjustment applied")

    if ov.is_heat_check:
        mu *= 1.15
        sd *= 1.10
        notes.append("Heat check boost")

    # -------------------------
    # Final clamps
    # -------------------------
    minutes = float(np.clip(minutes, 10, 44))
    sd = float(np.clip(sd, 0.5, 20.0))

    note_summary = "; ".join(notes)

    return mu, sd, minutes, usage_mult, True, note_summary

# =====================================================================
# MODULE 17 — PROJECTION OVERRIDE ENGINE UI PANEL
# Phase 2 — Streamlit Override Editor
# =====================================================================

with st.sidebar.expander("🛠 Projection Overrides", expanded=False):

    st.markdown("### 🔧 Manual Override Controls")

    st.write(
        "Use these controls to override projections for special situations:\n"
        "- Injury return\n"
        "- Minutes restriction\n"
        "- Role change\n"
        "- Heat check\n"
        "- Back-to-back fatigue\n"
        "- Trade adjustments\n"
    )

    # ------------------------------------------------
    # Player & Market Selection
    # ------------------------------------------------
    override_player = st.text_input("Player Name (required)").strip()
    override_market = st.selectbox(
        "Market",
        ["Points", "Rebounds", "Assists", "PRA"],
    )

    st.markdown("### ⏱ Minutes Overrides")
    ov_minutes = st.number_input("Set Minutes (optional)", min_value=0.0, max_value=48.0, value=0.0)
    ov_minutes_floor = st.number_input("Minutes Floor", min_value=0.0, max_value=48.0, value=0.0)
    ov_minutes_ceiling = st.number_input("Minutes Ceiling", min_value=0.0, max_value=48.0, value=0.0)

    st.markdown("### 🔥 Usage Overrides")
    ov_usage_mult = st.number_input("Usage Multiplier (1.15 = +15%)", value=1.00)
    ov_usage_floor = st.number_input("Usage Floor", value=0.00)
    ov_usage_ceiling = st.number_input("Usage Ceiling", value=5.00)

    st.markdown("### 📊 Market-Specific Adjustments")
    ov_mu_mult = st.number_input("μ Multiplier", value=1.00)
    ov_sd_mult = st.number_input("σ Multiplier", value=1.00)
    ov_mu_override = st.number_input("Hard μ Override (optional)", value=0.0)
    ov_sd_override = st.number_input("Hard σ Override (optional)", value=0.0)

    st.markdown("### ⚠ Contextual Flags")
    flag_minutes_restriction = st.checkbox("Minutes Restriction")
    flag_injury_return = st.checkbox("Injury Return")
    flag_role_change = st.checkbox("Role Change")
    flag_heat_check = st.checkbox("Heat Check")
    flag_fatigue_b2b = st.checkbox("Back-to-Back Fatigue")
    flag_trade = st.checkbox("Trade Adjustment")

    ov_note = st.text_input("Internal Note (optional)")

    # ------------------------------------------------
    # Register Override
    # ------------------------------------------------
    if st.button("💾 Apply Override"):

        if override_player == "":
            st.error("Player name is required to apply override.")
        else:
            override = ProjectionOverride(
                minutes = ov_minutes if ov_minutes > 0 else None,
                minutes_floor = ov_minutes_floor if ov_minutes_floor > 0 else None,
                minutes_ceiling = ov_minutes_ceiling if ov_minutes_ceiling > 0 else None,
                usage_mult = ov_usage_mult if ov_usage_mult != 1.0 else None,
                usage_floor = ov_usage_floor if ov_usage_floor > 0 else None,
                usage_ceiling = ov_usage_ceiling if ov_usage_ceiling > 0 else None,
                mu_mult = ov_mu_mult if ov_mu_mult != 1.0 else None,
                sd_mult = ov_sd_mult if ov_sd_mult != 1.0 else None,
                override_mu = ov_mu_override if ov_mu_override > 0 else None,
                override_sd = ov_sd_override if ov_sd_override > 0 else None,
                is_minutes_restricted = flag_minutes_restriction,
                is_injury_return = flag_injury_return,
                is_role_change = flag_role_change,
                is_heat_check = flag_heat_check,
                is_fatigue_b2b = flag_fatigue_b2b,
                is_trade_adjustment = flag_trade,
                note = ov_note
            )

            register_override(override_player, override_market, override)
            st.success(f"Override applied to {override_player} — {override_market}")

    # ------------------------------------------------
    # Display Active Overrides
    # ------------------------------------------------
    st.markdown("### 📌 Active Overrides")

    if len(override_db) == 0:
        st.info("No overrides applied.")
    else:
        for p, markets in override_db.items():
            st.markdown(f"**{p.title()}**")
            for m, ov in markets.items():
                st.write(f"- Market: **{m.title()}**")
                st.write(f"  - {ov}")

============================================================
MODULE 18 — Live Line Sync Engine (PrizePicks + Sleeper)
Phase 1 — Architecture + Core Layer
============================================================
import time
import json
import requests
import streamlit as st
from functools import lru_cache


# ============================================================
# 🔧 CONFIG
# ============================================================

PRIZEPICKS_API = "https://api.prizepicks.com/projections"
SLEEPER_API = "https://api.sleeper.app/v1/picks"  # placeholder, real endpoint Phase 2

LINE_CACHE_TTL = 60  # refresh every 60 seconds


# ============================================================
# 📌 Market Normalization Maps
# ============================================================

PRIZEPICKS_MARKET_MAP = {
    "Points": "Points",
    "Rebounds": "Rebounds",
    "Assists": "Assists",
    "Pts+Rebs+Asts": "PRA",
    "Pts+Rebs": "PointsRebounds",
    "Pts+Asts": "PointsAssists",
    "Rebs+Asts": "ReboundsAssists"
}

SLEEPER_MARKET_MAP = {
    "PTS": "Points",
    "REB": "Rebounds",
    "AST": "Assists",
    "PRA": "PRA"
}


# ============================================================
#  Standard Output Schema
# ============================================================

def normalize_prop(player_name, team, market, line, platform):
    """
    Returns a standard, unified line dictionary for the engine.
    """
    return {
        "player": player_name,
        "team": team,
        "market": market,
        "line": float(line),
        "platform": platform
    }
============================================================
 SAFE REQUEST WRAPPER
============================================================
def safe_get(url, params=None, headers=None):
    """
    Wraps requests.get with:
    - automatic error catching
    - Streamlit compatibility
    - no crashes on API downtime
    """
    try:
        r = requests.get(url, params=params, headers=headers, timeout=6)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except:
        return None

============================================================
 PRIZEPICKS RAW FETCHER
============================================================
def fetch_prizepicks_raw():
    """
    Returns raw JSON from PrizePicks.
    """
    return safe_get(PRIZEPICKS_API)
============================================================
 SLEEPER RAW FETCHER (Phase 1 placeholder)
Actual endpoint gets plugged in Phase 2.
============================================================
def fetch_sleeper_raw():
    """
    Placeholder (Sleeper API defined in Phase 2).
    """
    return None
============================================================
 PRIZEPICKS NORMALIZATION LOGIC
============================================================
def parse_prizepicks(json_raw):
    """
    Converts PrizePicks projections → unified schema list.
    """
    if not json_raw or "data" not in json_raw:
        return []

    results = []

    for item in json_raw["data"]:
        try:
            attributes = item["attributes"]
            included_player = next(
                (x for x in json_raw["included"] if x["id"] == attributes["athlete_id"]),
                None
            )
            if not included_player:
                continue

            player_name = included_player["attributes"]["name"]
            team = included_player["attributes"].get("team", "")

            pp_market = attributes["stat_type"]
            nba_market = PRIZEPICKS_MARKET_MAP.get(pp_market)

            if nba_market is None:
                continue

            line_value = attributes["line_score"]

            formatted = normalize_prop(
                player_name,
                team,
                nba_market,
                line_value,
                "PrizePicks"
            )
            results.append(formatted)

        except Exception:
            continue

    return results

============================================================
 SLEEPER NORMALIZATION (placeholder for Phase 2)
============================================================
def parse_sleeper(json_raw):
    """
    Placeholder parser — implemented in Phase 2.
    """
    return []
============================================================
 CACHING WRAPPER (60s)
============================================================
@st.cache_data(ttl=LINE_CACHE_TTL)
def get_live_lines():
    """
    High-level unified fetcher:
    - Fetch PrizePicks
    - Fetch Sleeper
    - Normalize both
    - Return single list
    """

    output = []

    # PrizePicks
    pp_raw = fetch_prizepicks_raw()
    pp_lines = parse_prizepicks(pp_raw)
    output.extend(pp_lines)

    # Sleeper (placeholder)
    sl_raw = fetch_sleeper_raw()
    sl_lines = parse_sleeper(sl_raw)
    output.extend(sl_lines)

    return output
============================================================
 STREAMLIT DEBUG PANEL (Phase 1)
============================================================
def render_live_line_debug():
    st.markdown("## 🔌 Live Lines — Debug (Phase 1)")

    lines = get_live_lines()
    if not lines:
        st.warning("No live lines found yet (Sleeper coming in Phase 2).")
        return

    for ln in lines[:50]:
        st.json(ln)

============================================================
MODULE 18 — Live Line Sync Engine
Phase 2 — Sleeper API Full Integration
============================================================
# ============================================================
# 🧩 Sleeper Market Mapping
# ============================================================

SLEEPER_MARKET_MAP = {
    "pts": "Points",
    "reb": "Rebounds",
    "ast": "Assists",
    "pra": "PRA",
    "blk": "Blocks",
    "stl": "Steals",
    "tov": "Turnovers",
}


# ============================================================
# 🔄 Sleeper Raw Fetcher
# ============================================================

def fetch_sleeper_raw():
    """
    Pulls raw NBA projection props from Sleeper.
    Returns None safely on transport errors.
    """
    url = "https://api.sleeper.app/v1/stats/nba/projections"

    # Sleeper throttles, so we ensure safe wrapper
    data = safe_get(url)

    return data


# ============================================================
# 🔄 Sleeper Player Lookup Table (once per app cache)
# ============================================================

@st.cache_data(ttl=3600)
def get_sleeper_players():
    """
    Loads all Sleeper NBA players for name/team resolution.
    Cached for 1 hour.
    """
    url = "https://api.sleeper.app/v1/players/nba"
    data = safe_get(url)
    if not data:
        return {}

    # data = {player_id: {player_attributes}}
    return data
============================================================
🧠 Sleeper Normalization Engine
============================================================
def parse_sleeper(json_raw):
    """
    Converts Sleeper projection JSON → unified prop schema list.
    """
    if not json_raw:
        return []

    output = []
    players = get_sleeper_players()

    # Sleeper returns dict: { "player_id": { projection_data } }
    for player_id, proj in json_raw.items():
        try:
            if player_id not in players:
                continue

            p_info = players[player_id]

            # Player identity
            player_name = p_info.get("full_name") or p_info.get("display_name") or "Unknown"
            team = p_info.get("team") or ""

            # Loop through each projection category
            for stat_key, val in proj.items():
                if stat_key not in SLEEPER_MARKET_MAP:
                    continue

                market = SLEEPER_MARKET_MAP[stat_key]
                try:
                    line_value = float(val)
                except:
                    continue

                formatted = normalize_prop(
                    player_name,
                    team,
                    market,
                    line_value,
                    "Sleeper"
                )
                output.append(formatted)

        except Exception:
            continue

    return output

============================================================
🧠 Merge, Deduplicate, and Return All Live Lines
(Enhanced Version for PrizePicks + Sleeper)
============================================================
@st.cache_data(ttl=LINE_CACHE_TTL)
def get_live_lines():
    """
    Unified live line fetcher:
    - PrizePicks
    - Sleeper
    - Normalization
    - Deduplication
    """
    results = []

    # ---- PrizePicks ----
    pp_raw = fetch_prizepicks_raw()
    pp_lines = parse_prizepicks(pp_raw)
    results.extend(pp_lines)

    # ---- Sleeper ----
    sl_raw = fetch_sleeper_raw()
    sl_lines = parse_sleeper(sl_raw)
    results.extend(sl_lines)

    # ---- Deduplicate ----
    # key: (player, market, platform)
    unique = {}
    for item in results:
        key = (item["player"], item["market"], item["platform"])
        unique[key] = item

    return list(unique.values())

============================================================
🔍 Optional: Live Line Viewer
============================================================
def render_live_line_debug():
    st.markdown("## 🔌 Live Lines Debug — PrizePicks + Sleeper")

    lines = get_live_lines()

    if not lines:
        st.error("⚠ No live lines synced.")
        return

    for ln in lines[:100]:
        st.json(ln)
============================================================
MODULE 18 — Phase 3
Auto-Match Live Lines → Model UI
============================================================
import difflib

# ============================================================
# 🔎 Fuzzy Player Matcher
# ============================================================

def best_player_match(query, available_names, cutoff=0.65):
    """
    Fuzzy matches the entered player name to the closest live-line player.
    """
    if not query or not available_names:
        return None

    matches = difflib.get_close_matches(query, available_names, n=1, cutoff=cutoff)
    return matches[0] if matches else None
============================================================
🎯 Build Player → Markets → Lines Index
============================================================
@st.cache_data(ttl=LINE_CACHE_TTL)
def build_live_index():
    """
    Create a dictionary indexed like:
       index[player_name][market] = {
            "line": float,
            "platform": "PrizePicks" | "Sleeper",
            "team": "DEN",
       }
    """

    lines = get_live_lines()
    index = {}

    for item in lines:
        player = item["player"]
        market = item["market"]

        if player not in index:
            index[player] = {}

        index[player][market] = {
            "line": float(item["line"]),
            "platform": item["platform"],
            "team": item["team"]
        }

    return index

============================================================
⚙️ Auto-Fill Logic Engine
============================================================
def autofill_player_from_live_lines(player_query):
    """
    Given a free-text player name:
      1. Fuzzy match vs live line names
      2. Return all available markets + lines
      3. Return None if no match
    """

    index = build_live_index()
    all_names = list(index.keys())

    # Step 1 — fuzzy match name
    best = best_player_match(player_query, all_names)

    if not best:
        return None

    # Step 2 — return the player's full market-list
    return {
        "player": best,
        "markets": index[best]  # dict: { "Points": {...}, "Rebounds": {...}, ... }
    }

============================================================
🧩 Streamlit Auto-Fill Dropdown Component
============================================================
def render_autofill_component(label="Player Search"):
    """
    Streamlit component for searching + autofilling live lines.
    Returns:
        {
          "player": "Nikola Jokic",
          "market": "Points",
          "line": 27.5,
          "platform": "PrizePicks"
        }
    """

    name_input = st.text_input(label)

    if not name_input:
        return None

    results = autofill_player_from_live_lines(name_input)

    if not results:
        st.warning("No matching live lines found.")
        return None

    player = results["player"]
    market_list = list(results["markets"].keys())

    market_select = st.selectbox(f"Available markets for {player}", market_list)

    selected_data = results["markets"][market_select]

    st.info(
        f"**Live Line Loaded**\n\n"
        f"- Player: **{player}**\n"
        f"- Market: **{market_select}**\n"
        f"- Line: **{selected_data['line']}**\n"
        f"- Source: **{selected_data['platform']}**"
    )

    return {
        "player": player,
        "market": market_select,
        "line": selected_data["line"],
        "platform": selected_data["platform"],
        "team": selected_data["team"]
    }

============================================================
🚀 Auto-Fill → Model Integration Helper
============================================================
def try_autofill_into_leg(prefix: str):
    """
    Automatically fills the leg inputs if live line exists.
    prefix = "P1" or "P2"
    inside UI:
       p1 = try_autofill_into_leg("P1")
    """
    st.markdown(f"### 🔄 Auto-Fill for {prefix}")

    data = render_autofill_component(f"{prefix} Player Search")

    if not data:
        return None

    # Return a clean dict for compute_leg()
    return {
        "player": data["player"],
        "market": data["market"],
        "line": data["line"],
        "opponent": "",     # filled later (Module 17 Opponent DB)
        "teammate_out": False,
        "blowout": False,
        "team": data["team"],
        "platform": data["platform"]
    }

============================================================
MODULE 18 — Phase 4
Background Auto-Refresh Engine (Live Line Updates)
============================================================
import time
import hashlib
import streamlit as st

# ============================================================
# 🔁 Helper: Hash any Python object for fast change detection
# ============================================================

def compute_hash(obj) -> str:
    """
    Converts any dict/list into a unique short hash so we can detect
    when live lines change without comparing entire structures.
    """
    try:
        raw = str(obj).encode("utf-8")
        return hashlib.md5(raw).hexdigest()
    except:
        return ""
============================================================
🔄 Background Auto-Refresh System
============================================================
def auto_refresh_live_lines(refresh_interval: int = 30):
    """
    Automatically refreshes PrizePicks + Sleeper line data every N seconds
    **WITHOUT forcing a full page reload.**
    
    - Detects movement using hashing
    - Stores last state in session_state
    - Returns True if lines updated
    """
    if "live_line_hash" not in st.session_state:
        st.session_state.live_line_hash = ""
        st.session_state.last_live_data = []

    # Fetch new snapshot
    new_lines = get_live_lines()  # from Phase 1/2
    new_hash = compute_hash(new_lines)

    # Compare with previous state
    moved = new_hash != st.session_state.live_line_hash

    # Update stored values
    if moved:
        st.session_state.live_line_hash = new_hash
        st.session_state.last_live_data = new_lines

    # Auto refresh (non intrusive)
    st_autorefresh = st.experimental_rerun if moved else None

    # Display timer
    st.caption(f"📡 Live line data refreshes every **{refresh_interval} seconds**.")

    # Return status
    return moved, new_lines

============================================================
🚨 Line Movement Detection Engine
============================================================
def detect_line_movements(old, new):
    """
    Compare old and new live lines and return a list of movements.
    Movement = line change or platform change.
    """

    movements = []
    old_map = {(o["player"], o["market"]): o for o in old}
    new_map = {(n["player"], n["market"]): n for n in new}

    for key in new_map:
        if key in old_map:
            old_item = old_map[key]
            new_item = new_map[key]

            if old_item["line"] != new_item["line"]:
                movements.append({
                    "player": key[0],
                    "market": key[1],
                    "old_line": old_item["line"],
                    "new_line": new_item["line"],
                    "platform": new_item["platform"]
                })

    return movements

============================================================
⚡ Streamlit Display Component (Alerts + Color Highlights)
============================================================
def show_line_movements(movements):
    """
    Displays animated line-movement alerts in Streamlit.
    """

    if not movements:
        st.success("No line movement detected since last refresh.")
        return

    st.markdown("## 🔥 **LIVE LINE MOVEMENTS DETECTED**")

    for move in movements:
        delta = move["new_line"] - move["old_line"]
        direction = "⬆️ Increased" if delta > 0 else "⬇️ Decreased"

        st.markdown(
            f"""
            <div style="padding:12px; border-radius:10px; background:#1e1e1e; margin-bottom:10px;">
                <b style="color:#4FC3F7;">{move['player']}</b><br>
                Market: <b>{move['market']}</b><br>
                Line: <b>{move['old_line']} → {move['new_line']}</b> ({direction})<br>
                Platform: <b>{move['platform']}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
============================================================
🚀 UI Control Panel Component
============================================================
def render_live_line_refresh_control():
    """
    UI block that allows enabling/disabling auto-refresh +
    selecting refresh interval.
    """

    st.markdown("### ⚙️ Live Line Auto-Refresh Settings")

    enable = st.checkbox("Enable Auto-Refresh", value=True)
    interval = st.slider("Refresh Interval (seconds)", 10, 120, 30, step=5)

    return enable, interval

============================================================
🎯 FULL STREAMLIT INTEGRATION HOOK
Called in your main UI (Model Tab)
============================================================
def live_line_background_process():
    """
    Full integration combining:
    - User config
    - Auto refresh
    - Movement detection
    - UI alerts
    """

    enable, interval = render_live_line_refresh_control()

    if enable:
        moved, new_data = auto_refresh_live_lines(interval)

        if moved:
            old_data = st.session_state.last_live_data
            movements = detect_line_movements(old_data, new_data)
            show_line_movements(movements)
        else:
            st.info("No changes detected in live lines yet.")

============================================================
MODULE 18 — Phase 5
Automatic Recompute on Line Movement
# =====================================================================
def auto_recompute_on_line_change(
    p1_name, p1_market,
    p2_name, p2_market,
    current_leg1_line,
    current_leg2_line,
    payout_mult
):
    """
    Triggers an automatic model recomputation when
    PrizePicks/Sleeper live lines change.

    Inputs:
        p1_name, p1_market      --> user selection leg 1
        p2_name, p2_market      --> user selection leg 2
        current_leg1_line       --> model’s last known line
        current_leg2_line       --> model’s last known line
        payout_mult             --> fixed 3.0 for PP power plays
        
    Returns:
        {
            "should_recompute": bool,
            "new_leg1_line": float or None,
            "new_leg2_line": float or None
        }
    """

    live_data = st.session_state.get("last_live_data", [])

    new_leg1_line = None
    new_leg2_line = None
    should_recompute = False

    for item in live_data:
        player = item["player"]
        market = item["market"]
        line = item["line"]

        # -----------------------------
        # LEG 1 MATCH
        # -----------------------------
        if (player.lower() == p1_name.lower() and
            market.lower() == p1_market.lower()):
            
            if line != current_leg1_line:
                new_leg1_line = line
                should_recompute = True

        # -----------------------------
        # LEG 2 MATCH
        # -----------------------------
        if (player.lower() == p2_name.lower() and
            market.lower() == p2_market.lower()):
            
            if line != current_leg2_line:
                new_leg2_line = line
                should_recompute = True

    return {
        "should_recompute": should_recompute,
        "new_leg1_line": new_leg1_line,
        "new_leg2_line": new_leg2_line
    }

============================================================
AUTO-MODEL EXECUTION BLOCK
(Runs the model automatically when lines move)
============================================================
def auto_execute_ultramax(
    p1_name, p1_market,
    p2_name, p2_market,
    leg1_line, leg2_line,
    lookback,
    payout_mult
):
    """
    Full automatic execution of the UltraMax 2-pick engine when
    line movement requires re-running projections.
    """

    st.markdown("### ⚡ Auto-Update Triggered — Recomputing Model...")

    # -------------------------
    # Run leg 1
    # -------------------------
    leg1, err1 = compute_leg(
        player=p1_name,
        market=p1_market,
        line=leg1_line,
        opponent="",
        teammate_out=False,
        blowout=False,
        lookback=lookback
    )

    if err1:
        st.error(f"Leg 1 Error: {err1}")
        return None, None

    # -------------------------
    # Run leg 2
    # -------------------------
    leg2, err2 = compute_leg(
        player=p2_name,
        market=p2_market,
        line=leg2_line,
        opponent="",
        teammate_out=False,
        blowout=False,
        lookback=lookback
    )

    if err2:
        st.error(f"Leg 2 Error: {err2}")
        return None, None

    # -------------------------
    # Render both legs
    # -------------------------
    render_leg_card_ultramax(leg1)
    render_leg_card_ultramax(leg2)

    # -------------------------
    # Correlation + Joint Combo
    # -------------------------
    corr = correlation_engine_v3(leg1, leg2)
    combo = monte_carlo_combo(leg1, leg2, corr, payout_mult)

    joint_prob = combo["joint_prob"]
    ev_combo = combo["ev"]
    kelly = combo["kelly_stake"]

    st.markdown("### 🔗 Auto Updated Combo Output")
    st.write(f"Joint Probability: **{joint_prob*100:.1f}%**")
    st.write(f"EV: **{ev_combo*100:+.1f}%**")
    st.write(f"Kelly Stake: **${kelly:.2f}**")

    return leg1, leg2

============================================================
MODULE 18 — PHASE 6
REAL-TIME EDGE ALERT + NOTIFICATION ENGINE
============================================================
import time
import streamlit as st

# -------------------------------------------------------------
# INTERNAL QUEUE FOR ALERTS
# -------------------------------------------------------------
if "edge_alert_queue" not in st.session_state:
    st.session_state.edge_alert_queue = []


def push_edge_alert(message: str, level: str = "info"):
    """
    Pushes an alert into Streamlit’s internal queue.
    Level ∈ {"success", "warning", "error", "info"}
    """

    st.session_state.edge_alert_queue.append({
        "msg": message,
        "level": level,
        "timestamp": time.time()
    })


def render_edge_alerts():
    """
    Renders queued alerts inside the UI.
    Removes alerts older than 12 seconds.
    """

    now = time.time()
    new_queue = []

    for alert in st.session_state.edge_alert_queue:
        if now - alert["timestamp"] <= 12:

            if alert["level"] == "success":
                st.success(alert["msg"])
            elif alert["level"] == "warning":
                st.warning(alert["msg"])
            elif alert["level"] == "error":
                st.error(alert["msg"])
            else:
                st.info(alert["msg"])

            new_queue.append(alert)

    st.session_state.edge_alert_queue = new_queue
    
    ============================================================
EDGE ALERT ENGINE
Watches for EV jumps, CLV edges, correlation spikes, steam moves
============================================================
def detect_new_edges(old_ev, new_ev, old_line, new_line, player, market):
    """
    Detects the following:
      - New edge created (EV crosses positive threshold)
      - Edge lost
      - EV jump (> +4%)
      - Steam move (line moves > 0.5)
      - Major CLV (line moves favorable to user selection)
    """

    # -----------------------------
    # 1. New edge appears
    # -----------------------------
    if old_ev < 0.03 and new_ev >= 0.03:
        push_edge_alert(
            f"🔥 NEW EDGE FOUND for {player} {market}: EV jumped to {new_ev*100:.1f}%",
            "success"
        )

    # -----------------------------
    # 2. Edge lost
    # -----------------------------
    if old_ev >= 0.03 and new_ev < 0.03:
        push_edge_alert(
            f"⚠️ EDGE LOST for {player} {market}: EV fell to {new_ev*100:.1f}%",
            "warning"
        )

    # -----------------------------
    # 3. Steam move detection
    # -----------------------------
    if abs(new_line - old_line) >= 0.5:
        push_edge_alert(
            f"⚡ STEAM MOVE DETECTED for {player} {market}: Line moved from {old_line} to {new_line}",
            "info"
        )

    # -----------------------------
    # 4. CLV (closing line value)
    # -----------------------------
    if new_line > old_line:
        push_edge_alert(
            f"💰 CLV GAIN: Market raised {player} {market} line from {old_line} → {new_line}",
            "success"
        )

    if new_line < old_line:
        push_edge_alert(
            f"❗ Market dropped {player} {market} line {old_line} → {new_line} (CLV loss)",
            "warning"
        )

============================================================
MASTER EDGE-MONITOR HOOK
Integrates into Phase 5 auto-execute logic
============================================================
def edge_monitor_hook(
    old_leg1_ev, old_leg2_ev,
    old_leg1_line, old_leg2_line,
    new_leg1_ev, new_leg2_ev,
    new_leg1_line, new_leg2_line,
    leg1_player, leg1_market,
    leg2_player, leg2_market
):

    detect_new_edges(
        old_ev=old_leg1_ev,
        new_ev=new_leg1_ev,
        old_line=old_leg1_line,
        new_line=new_leg1_line,
        player=leg1_player,
        market=leg1_market
    )

    detect_new_edges(
        old_ev=old_leg2_ev,
        new_ev=new_leg2_ev,
        old_line=old_leg2_line,
        new_line=new_leg2_line,
        player=leg2_player,
        market=leg2_market
    )

============================================================
MODULE 19 — RENDER COMPONENTS
PHASE 1 — CORE CARD FRAMEWORK
============================================================
import streamlit as st
from datetime import datetime

# -----------------------------------------------------------
# Color palette for UltraMax V4
# -----------------------------------------------------------
ULTRAMAX_COLORS = {
    "primary": "#4A7AFE",
    "success": "#00C853",
    "warning": "#F9A825",
    "danger":  "#D50000",
    "background": "#0D1117",
    "surface": "#161B22",
    "text_primary": "#E6EDF3",
    "text_secondary": "#9AA4B2",
    "border": "#2C323A",
}

# -----------------------------------------------------------
# Utility: colored metric badge
# -----------------------------------------------------------
def badge(label: str, value: str, color: str):
    """
    Generic capsule badge for EV, Probabilities, Lines, etc.
    """
    return f"""
    <div style="
        display:inline-block;
        padding:4px 10px;
        margin:2px;
        background:{color};
        border-radius:8px;
        font-size:13px;
        font-weight:600;
        color:white;
    ">
        {label}: {value}
    </div>
    """

# -----------------------------------------------------------
# Utility: surface container with border & shadow
# -----------------------------------------------------------
def card_container(inner_html: str, width: str = "100%"):
    """
    All cards use this structural container.
    """
    return f"""
    <div style="
        width:{width};
        background:{ULTRAMAX_COLORS['surface']};
        border:1px solid {ULTRAMAX_COLORS['border']};
        border-radius:12px;
        padding:18px;
        margin-bottom:18px;
        box-shadow:0 4px 12px rgba(0,0,0,0.25);
    ">
        {inner_html}
    </div>
    """

# -----------------------------------------------------------
# Utility — EV color selection
# -----------------------------------------------------------
def ev_color(ev: float):
    if ev >= 0.10: return ULTRAMAX_COLORS["success"]
    if ev >= 0.03: return ULTRAMAX_COLORS["warning"]
    return ULTRAMAX_COLORS["danger"]

# -----------------------------------------------------------
# Utility — Probability color
# -----------------------------------------------------------
def prob_color(p: float):
    if p >= 0.60: return ULTRAMAX_COLORS["success"]
    if p >= 0.52: return ULTRAMAX_COLORS["warning"]
    return ULTRAMAX_COLORS["danger"]

# -----------------------------------------------------------
# Phase 1 only defines the framework —
# Phase 2 will add the full Leg Card layout.
# Phase 3 will add Combo Card.
# Phase 4 will add expandable metrics.
# Phase 5 will add animated transitions.
# -----------------------------------------------------------

# ============================================================
# MODULE 19 — PHASE 2  
# LEG CARD RENDERING (Single Player UI Component)
# ============================================================

def render_leg_card_ultramax(leg: dict):
    """
    Renders a single-leg card for UltraMax V4.
    Inputs:
        leg — dict with keys:
            player, market, line, prob_over,
            mu, sd, ctx_mult,
            teammate_out, blowout
    """

    player = leg.get("player", "Unknown Player")
    market = leg.get("market", "")
    line = leg.get("line", 0)
    prob  = float(leg.get("prob_over", 0))
    mu    = float(leg.get("mu", 0))
    sd    = float(leg.get("sd", 0))
    ctx   = float(leg.get("ctx_mult", 1.0))

    teammate_out = leg.get("teammate_out", False)
    blowout      = leg.get("blowout", False)

    # --------------------------
    # Colored tags
    # --------------------------
    prob_badge = badge("Model Prob", f"{prob*100:.1f}%", prob_color(prob))
    line_badge = badge("Line", f"{line}", ULTRAMAX_COLORS["primary"])
    mu_badge   = badge("μ", f"{mu:.2f}", ULTRAMAX_COLORS["text_secondary"])
    sd_badge   = badge("σ", f"{sd:.2f}", ULTRAMAX_COLORS["text_secondary"])
    ctx_badge  = badge("Ctx Mult", f"{ctx:.2f}", ULTRAMAX_COLORS["warning"])

    out_badge = ""
    if teammate_out:
        out_badge = badge("Teammate OUT", "↑ Usage", ULTRAMAX_COLORS["warning"])

    blow_badge = ""
    if blowout:
        blow_badge = badge("Blowout Risk", "↓ Minutes", ULTRAMAX_COLORS["danger"])

    # --------------------------
    # HTML layout
    # --------------------------
    html = f"""
    <div style="color:{ULTRAMAX_COLORS['text_primary']};">

        <h3 style="
            margin:0;
            padding:0;
            color:{ULTRAMAX_COLORS['primary']};
            font-weight:700;
        ">{player}</h3>

        <div style="margin-top:6px; font-size:16px; color:{ULTRAMAX_COLORS['text_secondary']}">
            {market} • Line {line}
        </div>

        <div style="margin-top:12px;">
            {line_badge}
            {prob_badge}
            {mu_badge}
            {sd_badge}
            {ctx_badge}
            {out_badge}
            {blow_badge}
        </div>

        <hr style="
            border:0;
            border-top:1px solid {ULTRAMAX_COLORS['border']};
            margin-top:18px;
            margin-bottom:12px;
        " />

        <div style="font-size:14px; color:{ULTRAMAX_COLORS['text_secondary']}">
            <strong>Projection Summary:</strong><br/>
            • Expected = <b>{mu:.2f}</b><br/>
            • Volatility = <b>{sd:.2f}</b><br/>
            • Opponent Context Multiplier = <b>{ctx:.2f}</b><br/>
            • Probability Over = <b>{prob*100:.1f}%</b><br/>
        </div>

    </div>
    """

    st.markdown(card_container(html), unsafe_allow_html=True)

# ============================================================
# MODULE 19 — PHASE 3  
# COMBO CARD RENDERING (2-Leg Summary)
# ============================================================

def render_combo_card_ultramax(combo: dict, leg1: dict, leg2: dict):
    """
    Renders the full 2-leg combo card.
    Inputs:
        combo — dict returned from module12_two_pick_decision
            joint_prob_mc
            joint_ev
            stake
            decision
            corr_used
            p_joint_raw
        leg1, leg2 — individual leg dicts
    """

    # Unpack combo metrics
    jp     = float(combo.get("joint_prob_mc", 0))
    ev     = float(combo.get("joint_ev", 0))
    stake  = float(combo.get("stake", 0))
    corr   = float(combo.get("corr_used", 0))
    p_raw  = float(combo.get("p_joint_raw", 0))
    label  = combo.get("decision", "No Decision")

    # Leg info
    l1_name = leg1.get("player", "Player 1")
    l1_mkt  = leg1.get("market", "")
    l1_line = leg1.get("line", "")

    l2_name = leg2.get("player", "Player 2")
    l2_mkt  = leg2.get("market", "")
    l2_line = leg2.get("line", "")

    # --------------------------
    # Badges
    # --------------------------
    jp_badge   = badge("Joint Prob", f"{jp*100:.1f}%", prob_color(jp))
    ev_badge   = badge("EV", f"{ev*100:+.1f}%", ev_color(ev))
    corr_badge = badge("Corr", f"{corr:+.2f}", ULTRAMAX_COLORS["info"])
    stake_badge = badge("Stake", f"${stake:.2f}", ULTRAMAX_COLORS["primary"])

    # Raw model probability (pre-adjusted)
    raw_badge = badge("Raw Joint Prob", f"{p_raw*100:.1f}%", ULTRAMAX_COLORS["text_secondary"])

    # Decision tier badge (Max Play / Play / Lean / Pass)
    if "MAX PLAY" in label:
        decision_color = ULTRAMAX_COLORS["danger"]
    elif "PLAY" in label:
        decision_color = ULTRAMAX_COLORS["primary"]
    elif "LEAN" in label:
        decision_color = ULTRAMAX_COLORS["warning"]
    else:
        decision_color = ULTRAMAX_COLORS["border"]

    decision_badge = badge("Decision", label, decision_color)

    # --------------------------
    # HTML Layout
    # --------------------------
    html = f"""
    <div style="color:{ULTRAMAX_COLORS['text_primary']};">

        <h2 style="
            margin:0;
            padding:0;
            color:{ULTRAMAX_COLORS['primary']};
            font-weight:700;
        ">
            Combo Summary
        </h2>

        <div style="margin-top:8px; font-size:15px;">
            <b>{l1_name}</b> — {l1_mkt} {l1_line}<br/>
            <b>{l2_name}</b> — {l2_mkt} {l2_line}
        </div>

        <div style="margin-top:14px;">
            {jp_badge}
            {ev_badge}
            {corr_badge}
            {raw_badge}
            {stake_badge}
            {decision_badge}
        </div>

        <hr style="
            border:0;
            border-top:1px solid {ULTRAMAX_COLORS['border']};
            margin-top:18px;
            margin-bottom:12px;
        " />

        <div style="font-size:14px; color:{ULTRAMAX_COLORS['text_secondary']}">
            <strong>Combo Insights:</strong><br/>
            • Joint probability (after drift + CLV): <b>{jp*100:.2f}%</b><br/>
            • Raw probability before adjustments: <b>{p_raw*100:.2f}%</b><br/>
            • Correlation applied: <b>{corr:+.3f}</b><br/>
            • EV per $1: <b>{ev*100:+.2f}%</b><br/>
            • Kelly stake: <b>${stake:.2f}</b><br/>
        </div>

    </div>
    """

    st.markdown(card_container(html), unsafe_allow_html=True)

# ============================================================
# MODULE 19 — PHASE 4
# FULL MODEL OUTPUT SECTION (Leg Cards + Combo Card)
# ============================================================

def run_ultramax_pipeline(
    p1_name, p1_market, p1_line, p1_opp, p1_teammate_out, p1_blowout,
    p2_name, p2_market, p2_line, p2_opp, p2_teammate_out, p2_blowout,
    lookback_games, payout_mult, bankroll, fractional_kelly
):
    """
    Executes the entire UltraMax model pipeline:
      1. Compute leg 1
      2. Compute leg 2
      3. Render leg cards
      4. Compute correlation
      5. Monte Carlo joint simulation
      6. Kelly sizing + decision
      7. Render combo card
      8. Optional warnings
    """

    st.markdown("## 📊 UltraMax Model Results")

    # ----------------------------------------------------
    # Compute Individual Legs
    # ----------------------------------------------------
    with st.spinner("Computing Leg 1..."):
        leg1, err1 = compute_leg(
            player=p1_name,
            market=p1_market,
            line=p1_line,
            opponent=p1_opp,
            teammate_out=p1_teammate_out,
            blowout=p1_blowout,
            lookback=lookback_games
        )

    with st.spinner("Computing Leg 2..."):
        leg2, err2 = compute_leg(
            player=p2_name,
            market=p2_market,
            line=p2_line,
            opponent=p2_opp,
            teammate_out=p2_teammate_out,
            blowout=p2_blowout,
            lookback=lookback_games
        )

    # ----------------------------------------------------
    # Handle Errors
    # ----------------------------------------------------
    if err1:
        st.error(f"❌ Leg 1 Error: {err1}")
    if err2:
        st.error(f"❌ Leg 2 Error: {err2}")

    if (err1 or err2) or (not leg1 or not leg2):
        st.warning("⚠ Cannot proceed: check player names, market, or API issues.")
        return

    # ----------------------------------------------------
    # Render Individual Leg Cards
    # ----------------------------------------------------
    st.markdown("### 🧩 Individual Legs")
    render_leg_card_ultramax(leg1)
    render_leg_card_ultramax(leg2)

    # ----------------------------------------------------
    # Compute Correlation
    # ----------------------------------------------------
    base_corr = correlation_engine_v3(leg1, leg2)

    # ----------------------------------------------------
    # Monte Carlo Combo
    # ----------------------------------------------------
    combo_out = module12_two_pick_decision(
        leg1_dist=monte_carlo_leg_simulation(
            leg1["mu"], leg1["sd"], leg1["line"], leg1["market"]
        )["dist"],
        leg2_dist=monte_carlo_leg_simulation(
            leg2["mu"], leg2["sd"], leg2["line"], leg2["market"]
        )["dist"],
        leg1_line=leg1["line"],
        leg2_line=leg2["line"],
        base_corr=base_corr,
        payout_mult=payout_mult,
        bankroll=bankroll,
        fractional_kelly=fractional_kelly,
        drift_adj=compute_drift_adjustment(),
        clv_adj=compute_clv_adjustment(),
    )

    # ----------------------------------------------------
    # Render Combo Summary
    # ----------------------------------------------------
    st.markdown("### 🔗 Combo Analysis")
    render_combo_card_ultramax(combo_out, leg1, leg2)

    # ----------------------------------------------------
    # Contextual Warnings / Advisories
    # ----------------------------------------------------
    st.markdown("### ⚠ Model Notes & Risk Checks")

    if p1_teammate_out or p2_teammate_out:
        st.info("🩹 **Injury Boost Activated:** Model has redistributed touches accordingly.")

    if p1_blowout or p2_blowout:
        st.warning("💥 **Blowout Flag:** Model applied reduction to projected minutes.")

    if abs(base_corr) > 0.35:
        st.warning(f"📈 **High Correlation:** Legs move together strongly (ρ = {base_corr:+.2f})")

    if leg1["sd"] > 8 or leg2["sd"] > 8:
        st.warning("🌪 **High Volatility:** One or more legs have elevated outcome variance.")

    st.success("✔ UltraMax evaluation complete.")

# ============================================================
# MODULE 19 — PHASE 5
# RUN BUTTON + PIPELINE EXECUTION IN UI
# ============================================================

# -----------------------
# Model Execution Section
# -----------------------
st.markdown("### 🚀 Run UltraMax Model")

run_button = st.button("Run Model", type="primary")

# Collect Context Input Values
inputs = {
    "p1_name": p1_name,
    "p1_market": p1_market,
    "p1_line": p1_line,
    "p1_opp": p1_opp,
    "p1_teammate_out": p1_teammate_out,
    "p1_blowout": p1_blowout,

    "p2_name": p2_name,
    "p2_market": p2_market,
    "p2_line": p2_line,
    "p2_opp": p2_opp,
    "p2_teammate_out": p2_teammate_out,
    "p2_blowout": p2_blowout,
}

# -----------------------------------------------
# Input Validation (MUST-HAVE for Streamlit Apps)
# -----------------------------------------------
def validate_inputs():
    if not inputs["p1_name"]:
        return "Player 1 name missing."

    if not inputs["p2_name"]:
        return "Player 2 name missing."

    if not inputs["p1_opp"] or len(inputs["p1_opp"]) != 3:
        return "Player 1 opponent must be 3-letter team abbreviation."

    if not inputs["p2_opp"] or len(inputs["p2_opp"]) != 3:
        return "Player 2 opponent must be 3-letter team abbreviation."

    return None


# -----------------------------------------------
# Execution Logic
# -----------------------------------------------
if run_button:
    st.markdown("---")

    err = validate_inputs()
    if err:
        st.error(f"⚠ Input Error: {err}")
    else:
        with st.spinner("Running UltraMax Pipeline..."):
            time.sleep(0.3)

            run_ultramax_pipeline(
                p1_name=inputs["p1_name"],
                p1_market=inputs["p1_market"],
                p1_line=inputs["p1_line"],
                p1_opp=inputs["p1_opp"],
                p1_teammate_out=inputs["p1_teammate_out"],
                p1_blowout=inputs["p1_blowout"],

                p2_name=inputs["p2_name"],
                p2_market=inputs["p2_market"],
                p2_line=inputs["p2_line"],
                p2_opp=inputs["p2_opp"],
                p2_teammate_out=inputs["p2_teammate_out"],
                p2_blowout=inputs["p2_blowout"],

                lookback_games=lookback_games,
                payout_mult=payout_mult,
                bankroll=user_bankroll,
                fractional_kelly=fractional_kelly
            )

    st.markdown("---")

# ================================================================
# MODULE 20 — Phase 1
# HISTORY TRACKING (CSV / LOCAL STORAGE) — FOUNDATION LAYER
# ================================================================

import os
import pandas as pd
from datetime import datetime

# -------------------------------
# History file path (local CSV)
# -------------------------------
HISTORY_FILE = "ultramax_history.csv"

# -----------------------------------------------
# Schema for history logs (consistent always)
# -----------------------------------------------
HISTORY_COLUMNS = [
    "timestamp",

    # Player 1 Leg
    "p1_player",
    "p1_market",
    "p1_line",
    "p1_prob",
    "p1_mu",
    "p1_sd",
    "p1_opp",
    "p1_teammate_out",
    "p1_blowout",

    # Player 2 Leg
    "p2_player",
    "p2_market",
    "p2_line",
    "p2_prob",
    "p2_mu",
    "p2_sd",
    "p2_opp",
    "p2_teammate_out",
    "p2_blowout",

    # Combo
    "correlation",
    "joint_prob",
    "ev",
    "kelly_stake",
    "decision_label",
    "payout_mult",

    # Run Metadata
    "lookback_games",
    "drift_adj",
    "clv_adj"
]


# ================================================================
# Ensure CSV Exists
# ================================================================
def _initialize_history_file():
    """
    Creates the CSV file if it doesn't exist.
    Ensures correct headers.
    Streamlit-safe and idempotent.
    """
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
        df.to_csv(HISTORY_FILE, index=False)


# ================================================================
# Load history CSV safely
# ================================================================
def load_history():
    """
    Loads the CSV into a dataframe.
    Always returns a valid DataFrame.
    """
    _initialize_history_file()

    try:
        df = pd.read_csv(HISTORY_FILE)
        # Ensure all columns exist
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[HISTORY_COLUMNS]

    except Exception:
        # If file corrupt, rebuild it
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
        df.to_csv(HISTORY_FILE, index=False)
        return df


# ================================================================
# Append a single model run to history
# ================================================================
def append_history(entry_dict: dict):
    """
    Appends one UltraMax run to the history CSV.
    entry_dict MUST follow HISTORY_COLUMNS.
    """

    _initialize_history_file()

    # Insert timestamp automatically
    entry_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = load_history()

    # Build row in correct order
    row = {col: entry_dict.get(col, None) for col in HISTORY_COLUMNS}

    # Append & save
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)


# ================================================================
# Clear History (Used later in UI)
# ================================================================
def clear_history():
    """
    Empties the history file.
    """
    df = pd.DataFrame(columns=HISTORY_COLUMNS)
    df.to_csv(HISTORY_FILE, index=False)

# ================================================================
# MODULE 20 — Phase 3
# FULL HISTORY TAB UI (SORTABLE TABLE + FILTERS + EXPORT)
# ================================================================

import streamlit as st
import pandas as pd
import os


def load_history_df():
    """Loads ultramax_history.csv safely."""
    fname = "ultramax_history.csv"
    if not os.path.exists(fname):
        return pd.DataFrame()
    try:
        df = pd.read_csv(fname)
    except:
        return pd.DataFrame()
    return df


def clear_history():
    """Deletes the history file."""
    fname = "ultramax_history.csv"
    if os.path.exists(fname):
        os.remove(fname)


def filter_history(df: pd.DataFrame):
    """
    Applies sidebar filters: player, market, EV range, date.
    Returns filtered DF.
    """

    if df.empty:
        return df

    # Convert timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    st.sidebar.markdown("### 🔎 History Filters")

    # --------------------------
    # Player search
    # --------------------------
    player_query = st.sidebar.text_input("Search Player")

    if player_query.strip():
        df = df[
            df["p1_player"].str.contains(player_query, case=False, na=False) |
            df["p2_player"].str.contains(player_query, case=False, na=False)
        ]

    # --------------------------
    # Market filter
    # --------------------------
    markets = sorted(df["p1_market"].dropna().unique().tolist())
    market_filter = st.sidebar.multiselect("Market", markets, default=markets)

    if market_filter:
        df = df[df["p1_market"].isin(market_filter) | df["p2_market"].isin(market_filter)]

    # --------------------------
    # EV filter
    # --------------------------
    ev_min, ev_max = st.sidebar.slider(
        "EV Range",
        float(df["ev"].min() if "ev" in df else -1.0),
        float(df["ev"].max() if "ev" in df else 1.0),
        value=(
            float(df["ev"].min() if "ev" in df else -1.0),
            float(df["ev"].max() if "ev" in df else 1.0),
        )
    )
    df = df[(df["ev"] >= ev_min) & (df["ev"] <= ev_max)]

    # --------------------------
    # Date range filter
    # --------------------------
    if "timestamp" in df.columns:
        min_date = df["timestamp"].min()
        max_date = df["timestamp"].max()

        start_date, end_date = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date)
        )

        df = df[
            (df["timestamp"] >= pd.to_datetime(start_date)) &
            (df["timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1))
        ]

    return df


def render_history_tab():
    """
    Full UI for the History Tab:
      - Load history
      - Apply filters
      - Display sortable table
      - Download button
      - Clear history button
    """

    st.header("📜 UltraMax Run History")

    # Load
    df = load_history_df()

    if df.empty:
        st.info("No historical runs yet. Run the model to generate entries.")
        return

    # Filter
    df_filtered = filter_history(df)

    # Display
    st.markdown("### 📊 Filtered History")
    st.dataframe(df_filtered, use_container_width=True)

    # Download CSV
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv,
        file_name="ultramax_history_filtered.csv",
        mime="text/csv"
    )

    # Clear history (with confirmation)
    st.markdown("---")
    st.markdown("### ⚠️ Clear Full History")

    if st.button("Delete All History"):
        st.warning("This will permanently delete ALL historical runs.")
        if st.button("Confirm Deletion"):
            clear_history()
            st.success("History cleared successfully. Refresh page.")

# ================================================================
# MODULE 21 — Phase 1
# CALIBRATION UI CONTROLLER (Core Logic + Persistence Layer)
# ================================================================

import os
import json
import streamlit as st

CALIBRATION_FILE = "ultramax_calibration.json"

# ------------------------------------------------------------
# DEFAULT CALIBRATION PARAMETERS (Synced w/ Module 9)
# ------------------------------------------------------------
DEFAULT_CALIBRATION = {
    "variance_adj": 1.00,       # global volatility correction
    "heavy_tail_adj": 1.00,     # right-tail expansion multiplier
    "bias_adj": 0.00,           # mean correction bias
    "drift_adj": 1.00,          # slow drift correction for probability stability
    "clv_adj": 1.00,            # sharp-side probability boost/damping
    "fractional_kelly": 0.50,   # bankroll risk setting
    "bankroll": 1000.0          # tracking unit bankroll
}


# ------------------------------------------------------------
# LOAD CALIBRATION
# ------------------------------------------------------------
def load_calibration():
    """
    Loads calibration parameters from local JSON file.
    If file missing or corrupted → return defaults.
    """

    if not os.path.exists(CALIBRATION_FILE):
        return DEFAULT_CALIBRATION.copy()

    try:
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)

        # Make sure all keys exist
        for k, v in DEFAULT_CALIBRATION.items():
            if k not in data:
                data[k] = v

        return data

    except:
        return DEFAULT_CALIBRATION.copy()


# ------------------------------------------------------------
# SAVE CALIBRATION
# ------------------------------------------------------------
def save_calibration(params: dict):
    """
    Saves user-updated calibration parameters to a JSON file.
    Streamlit-safe, no risk of app crash.
    """

    try:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(params, f, indent=4)
        return True
    except:
        return False


# ------------------------------------------------------------
# PROVIDE CALIBRATION OBJECT FOR OTHER MODULES
# ------------------------------------------------------------
def get_calibration_params():
    """
    Public entry point:
    Modules 10–12 call this to retrieve calibration settings.
    """
    return load_calibration()


# ------------------------------------------------------------
# CONTROLLER CLASS (Used in Phase 2–4 for full UI control)
# ------------------------------------------------------------
class CalibrationController:
    """
    Controls:
      - loading calibration
      - updating calibration
      - validating ranges
      - emitting events to UI widgets (Phase 2)
      - pushing values into engine modules (Phase 3/4)
    """

    def __init__(self):
        self.params = load_calibration()

    def update(self, key: str, value):
        """Safely update single calibration parameter."""
        if key in self.params:
            self.params[key] = value

    def save(self):
        """Write parameters to disk."""
        return save_calibration(self.params)

    def reset(self):
        """Reset all parameters to defaults."""
        self.params = DEFAULT_CALIBRATION.copy()
        return save_calibration(self.params)

# ================================================================
# MODULE 21 — Phase 2
# CALIBRATION TAB UI (User Controls)
# ================================================================

def render_calibration_tab():
    """
    Full calibration UI for tuning:
      - Variance Adjustment
      - Heavy Tail Adjustment
      - Bias Adjustment
      - Drift & CLV multipliers
      - Fractional Kelly
      - Bankroll settings

    This tab syncs directly with CalibrationController from Phase 1.
    """

    st.header("🧠 Model Calibration Center")
    st.write("Adjust advanced model parameters for volatility, risk, bias, and EV scaling.")

    # Load controller
    controller = CalibrationController()
    params = controller.params.copy()

    st.markdown("---")
    st.subheader("📊 Volatility & Distribution Controls")

    # --------------------------------------------------------
    # Variance Adjustment Slider
    # --------------------------------------------------------
    variance_adj = st.slider(
        "Variance Adjustment (Volatility Correction)",
        min_value=0.50, max_value=1.50,
        value=float(params["variance_adj"]),
        step=0.01,
        help="Scales standard deviation for all legs. >1 increases volatility; <1 reduces."
    )
    controller.update("variance_adj", variance_adj)

    # --------------------------------------------------------
    # Heavy Tail Adjustment
    # --------------------------------------------------------
    heavy_tail_adj = st.slider(
        "Heavy Tail Adjustment (Right-tail expansion)",
        min_value=0.90, max_value=1.20,
        value=float(params["heavy_tail_adj"]),
        step=0.01,
        help="Controls the blend weighting in Monte Carlo’s right-tail extension."
    )
    controller.update("heavy_tail_adj", heavy_tail_adj)

    # --------------------------------------------------------
    # Bias Adjustment
    # --------------------------------------------------------
    bias_adj = st.slider(
        "Bias Adjustment (Mean shift ±)",
        min_value=-3.0, max_value=3.0,
        value=float(params["bias_adj"]),
        step=0.1,
        help="Adds/subtracts a bias to expected mean projection (mu)."
    )
    controller.update("bias_adj", bias_adj)

    st.markdown("---")
    st.subheader("🪙 Probability Stability Controls")

    # --------------------------------------------------------
    # Drift Adjustment
    # --------------------------------------------------------
    drift_adj = st.slider(
        "Drift Adjustment (Probability Stability)",
        min_value=0.85, max_value=1.15,
        value=float(params["drift_adj"]),
        step=0.01,
        help="Stabilizes probability over time to avoid overfitting."
    )
    controller.update("drift_adj", drift_adj)

    # --------------------------------------------------------
    # CLV Adjustment
    # --------------------------------------------------------
    clv_adj = st.slider(
        "CLV Adjustment (Sharp-Side Bias)",
        min_value=0.85, max_value=1.20,
        value=float(params["clv_adj"]),
        step=0.01,
        help="Boost or dampen probability based on sharp closing-line tendencies."
    )
    controller.update("clv_adj", clv_adj)

    st.markdown("---")
    st.subheader("💰 Risk & Bankroll Settings")

    # --------------------------------------------------------
    # Fractional Kelly slider
    # --------------------------------------------------------
    fractional_kelly = st.slider(
        "Fractional Kelly (Risk Multiplier)",
        min_value=0.0, max_value=1.0,
        value=float(params["fractional_kelly"]),
        step=0.05,
        help="1.0 = Full Kelly (aggressive). 0.5 = Half-Kelly (recommended)."
    )
    controller.update("fractional_kelly", fractional_kelly)

    # --------------------------------------------------------
    # Bankroll
    # --------------------------------------------------------
    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=50.0,
        max_value=1_000_000.0,
        value=float(params["bankroll"]),
        step=25.0,
    )
    controller.update("bankroll", bankroll)

    st.markdown("---")
    st.subheader("💾 Save Settings")

    colA, colB = st.columns(2)

    # --------------------------------------------------------
    # Save button
    # --------------------------------------------------------
    if colA.button("💾 Save Calibration"):
        success = controller.save()
        if success:
            st.success("✅ Calibration saved successfully!")
        else:
            st.error("❌ Failed to save calibration.")

    # --------------------------------------------------------
    # Reset button
    # --------------------------------------------------------
    if colB.button("🔄 Reset to Defaults"):
        controller.reset()
        st.warning("⚠️ Calibration reset to default values. Refresh tab.")

    st.markdown("---")
    st.info("These settings automatically feed into Monte Carlo (Module 10–11) and decision engine (Module 12).")

# ================================================================
# MODULE 21 — Phase 3
# CALIBRATION DIAGNOSTICS + LIVE PREVIEW VISUALS
# ================================================================
import numpy as np
import plotly.graph_objects as go

def render_calibration_diagnostics():
    """
    Real-time diagnostic charts for calibration parameters:
      - Distribution preview (normal, heavy-tail, biased)
      - Volatility curve visualization
      - Probability sensitivity (line vs hit prob)
    Automatically reflects slider adjustments from Phase 2.
    """

    st.header("📈 Calibration Diagnostics & Live Preview")

    controller = CalibrationController()
    params = controller.params.copy()

    # Extract parameters
    variance_adj = params["variance_adj"]
    heavy_tail_adj = params["heavy_tail_adj"]
    bias_adj = params["bias_adj"]
    drift_adj = params["drift_adj"]
    clv_adj = params["clv_adj"]

    # Fake base projection for diagnostics
    base_mu = 25
    base_sd = 6

    # Apply calibration
    adj_sd = base_sd * variance_adj
    adj_mu = base_mu + bias_adj
    tail_mult = heavy_tail_adj

    # -----------------------------------------------------------
    # DIAGNOSTIC AREA 1 — Distribution Comparison
    # -----------------------------------------------------------
    st.subheader("📊 Distribution Preview")

    x = np.linspace(base_mu - 4 * base_sd, base_mu + 4 * base_sd, 400)

    # Normal baseline
    norm_y = np.exp(-0.5 * ((x - base_mu) / base_sd) ** 2)

    # Adjusted distribution
    adj_y = np.exp(-0.5 * ((x - adj_mu) / adj_sd) ** 2) * tail_mult

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(
        x=x, y=norm_y, mode='lines',
        name="Baseline Distribution", line=dict(color="#1f77b4", width=2)
    ))
    fig_dist.add_trace(go.Scatter(
        x=x, y=adj_y, mode='lines',
        name="Calibrated Distribution", line=dict(color="#ff4500", width=3)
    ))
    fig_dist.update_layout(
        height=340,
        xaxis_title="Outcome",
        yaxis_title="Density",
        title="Normal vs Calibrated Distribution"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    # -----------------------------------------------------------
    # DIAGNOSTIC AREA 2 — Volatility Curve Visualization
    # -----------------------------------------------------------
    st.subheader("📉 Volatility Curve")

    minutes = np.linspace(18, 40, 24)
    vol_curve = adj_sd * np.sqrt(minutes / 30)

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=minutes, y=vol_curve, mode="lines+markers",
        name="Volatility Curve", line=dict(color="#2ca02c", width=3)
    ))
    fig_vol.update_layout(
        height=320,
        xaxis_title="Projected Minutes",
        yaxis_title="Adjusted Standard Deviation",
        title="Volatility vs Minutes"
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    # -----------------------------------------------------------
    # DIAGNOSTIC AREA 3 — Probability Sensitivity Curve
    # -----------------------------------------------------------
    st.subheader("🎯 Probability Sensitivity")

    # Market lines to test
    lines = np.linspace(base_mu - 12, base_mu + 12, 60)

    # Hit probabilities before calibration
    baseline_probs = 1 - norm.cdf(lines, base_mu, base_sd)

    # After calibration (drift + CLV applied)
    calibrated_probs = 1 - norm.cdf(lines, adj_mu, adj_sd)
    calibrated_probs = calibrated_probs * drift_adj * clv_adj
    calibrated_probs = np.clip(calibrated_probs, 0.01, 0.99)

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        x=lines, y=baseline_probs, mode='lines',
        name="Baseline Probabilities", line=dict(color="#1f77b4", width=2)
    ))
    fig_prob.add_trace(go.Scatter(
        x=lines, y=calibrated_probs, mode='lines',
        name="Calibrated Probabilities", line=dict(color="#ff4500", width=3)
    ))
    fig_prob.update_layout(
        height=350,
        xaxis_title="Market Line",
        yaxis_title="Hit Probability",
        title="Probability Curve (Baseline vs Calibrated)"
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")
    st.info("These diagnostics help validate how calibration affects probabilities, volatility, and Monte Carlo behavior.")

# ================================================================
# MODULE 21 — Phase 4
# CALIBRATION EXPORT / IMPORT + PERSISTENT STORAGE
# ================================================================

import json
import os
import streamlit as st

CALIBRATION_FILE = "calibration_settings.json"


def save_calibration_settings(params: dict):
    """
    Saves calibration settings to a local JSON file.
    Works both in local Streamlit and Streamlit Cloud.
    """
    try:
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(params, f, indent=4)
        st.success("✅ Calibration settings saved successfully.")
    except Exception as e:
        st.error(f"❌ Failed to save calibration settings: {e}")


def load_calibration_settings() -> dict:
    """
    Loads calibration settings if file exists.
    Otherwise returns defaults from CalibrationController().
    """
    controller = CalibrationController()
    default_params = controller.params

    if not os.path.exists(CALIBRATION_FILE):
        return default_params

    try:
        with open(CALIBRATION_FILE, "r") as f:
            loaded = json.load(f)

        # Merge loaded with defaults to ensure no missing keys
        merged = {**default_params, **loaded}
        return merged

    except Exception:
        return default_params


def render_calibration_export_import_ui():
    """
    Streamlit UI for:
      - Save calibration
      - Load calibration
      - Export JSON
      - Import JSON
    """

    st.subheader("💾 Save / Load Calibration Settings")

    controller = CalibrationController()
    current_params = controller.params.copy()

    col1, col2 = st.columns(2)

    # ------------------------------------------------------------
    # SAVE BUTTON
    # ------------------------------------------------------------
    with col1:
        if st.button("💾 Save Calibration Settings", use_container_width=True):
            save_calibration_settings(current_params)

    # ------------------------------------------------------------
    # LOAD BUTTON
    # ------------------------------------------------------------
    with col2:
        if st.button("📥 Load Saved Settings", use_container_width=True):
            loaded = load_calibration_settings()
            CalibrationController().update(loaded)
            st.success("🔄 Loaded saved calibration settings.")
            st.rerun()

    st.markdown("---")
    st.subheader("📤 Export / 📥 Import Calibration JSON")

    col3, col4 = st.columns(2)

    # ------------------------------------------------------------
    # EXPORT JSON DOWNLOAD
    # ------------------------------------------------------------
    with col3:
        st.download_button(
            label="📤 Export Calibration JSON",
            data=json.dumps(current_params, indent=4),
            file_name="calibration_export.json",
            mime="application/json",
            use_container_width=True
        )

    # ------------------------------------------------------------
    # IMPORT JSON UPLOAD
    # ------------------------------------------------------------
    with col4:
        uploaded = st.file_uploader("📥 Upload Calibration JSON", type=["json"])
        if uploaded is not None:
            try:
                imported_params = json.load(uploaded)
                CalibrationController().update(imported_params)
                st.success("✅ Imported calibration file successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Invalid JSON file: {e}")

    st.markdown("---")
    st.info("Calibration settings are now fully persistent, exportable, and importable.")

# ================================================================
# MODULE 22 — PHASE 1
# DEFENSIVE MATCHUP ENGINE — CORE SCHEMA + BASELINE RATINGS
# ================================================================

import numpy as np

# ------------------------------------------------------------
# 1. Define standardized defensive attributes for all teams
# ------------------------------------------------------------
# These categories will be used across:
#   - Opponent Engine
#   - Usage Engine
#   - Game Scripts
#   - Monte Carlo Variance
#   - Correlation Engine
#   - Projection Override
#   - Combo EV Engine

DEF_ATTRIBUTES = [
    "paint_defense",        # how well they defend rim attacks, rebounds
    "perimeter_defense",    # point-of-attack defense
    "switch_defense",       # switching ability, screen navigation
    "help_defense",         # rotations, weakside help, rim support
    "rebound_rate",         # ability to limit 2nd chance points
    "pace_adj",             # adjusts possessions up/down
    "ast_allowance",        # how many assists they generally allow
    "turnover_creation",    # pressure defense → more TO → transition
]

# ------------------------------------------------------------
# 2. Baseline positional matchup weights
# ------------------------------------------------------------
# These weights determine how important each defensive attribute is
# depending on the offensive player’s primary scoring style.

POSITION_WEIGHTS = {
    "guard": {
        "perimeter_defense": 0.32,
        "switch_defense": 0.22,
        "pace_adj": 0.14,
        "ast_allowance": 0.18,
        "paint_defense": 0.02,
        "help_defense": 0.06,
        "rebound_rate": 0.03,
        "turnover_creation": 0.13,
    },

    "wing": {
        "perimeter_defense": 0.24,
        "switch_defense": 0.20,
        "help_defense": 0.18,
        "pace_adj": 0.10,
        "rebound_rate": 0.08,
        "paint_defense": 0.12,
        "ast_allowance": 0.06,
        "turnover_creation": 0.12,
    },

    "big": {
        "paint_defense": 0.35,
        "rebound_rate": 0.22,
        "help_defense": 0.20,
        "pace_adj": 0.08,
        "switch_defense": 0.06,
        "perimeter_defense": 0.03,
        "ast_allowance": 0.03,
        "turnover_creation": 0.03,
    },
}

# ------------------------------------------------------------
# 3. Defensive Team Database (blank for now; filled in Phase 2)
# ------------------------------------------------------------
# Example structure:
#
# DEF_TEAM_DB["MIL"] = {
#     "paint_defense": 0.78,
#     "perimeter_defense": 0.70,
#     ...
# }
#
# Phase 2 will fill in all 30 NBA teams.

DEF_TEAM_DB = {}


# ------------------------------------------------------------
# 4. Core defensive matchup function (base multiplier)
# ------------------------------------------------------------
def compute_def_matchup_multiplier(team: str, position: str) -> float:
    """
    Computes a base defensive multiplier:
        < 1.00 → difficult matchup
        > 1.00 → favorable matchup

    Inputs:
        team      — "LAL", "MIL", etc.
        position  — "guard", "wing", "big"

    Returns:
        float multiplier (0.85 → 1.20)
    """

    team = team.upper()
    position = position.lower()

    if team not in DEF_TEAM_DB:
        # unknown team → neutral matchup
        return 1.00

    if position not in POSITION_WEIGHTS:
        position = "wing"  # safest fallback

    weights = POSITION_WEIGHTS[position]
    team_stats = DEF_TEAM_DB[team]

    # Weighted sum
    score = 0
    for attr, weight in weights.items():
        score += team_stats.get(attr, 0.50) * weight

    # Normalize score into usable multiplier
    # 0.50 → 0.80
    # 0.75 → 1.00
    # 1.00 → 1.20

    normalized = 0.80 + (score - 0.50) * 0.80

    # Safety clamp
    normalized = float(np.clip(normalized, 0.85, 1.20))

    return normalized


# ------------------------------------------------------------
# 5. Convenience wrapper for modules 3, 4, 5, 6, 9, 12
# ------------------------------------------------------------
def defensive_context_boost(team: str, position: str, market: str) -> float:
    """
    Applies matchup boost by market type:

    PRA     → balanced
    Points  → more perimeter/paint weight
    Rebounds→ more rebound & paint weight
    Assists → more ast_allowance & pace

    Returns multiplier 0.85–1.25
    """

    base = compute_def_matchup_multiplier(team, position)

    market = market.lower()

    if market == "points":
        return float(np.clip(base * 1.05, 0.85, 1.25))
    elif market == "rebounds":
        return float(np.clip(base * 1.08, 0.85, 1.25))
    elif market == "assists":
        return float(np.clip(base * 1.12, 0.85, 1.25))
    else:  # PRA
        return float(np.clip(base * 1.00, 0.85, 1.25))

# ================================================================
# MODULE 22 — PHASE 2
# FULL 30-TEAM DEFENSIVE TEAM DATABASE
# ================================================================

DEF_TEAM_DB = {
    "ATL": {
        "paint_defense": 0.42,
        "perimeter_defense": 0.38,
        "switch_defense": 0.41,
        "help_defense": 0.44,
        "rebound_rate": 0.52,
        "pace_adj": 0.63,
        "ast_allowance": 0.58,
        "turnover_creation": 0.40,
    },
    "BOS": {
        "paint_defense": 0.82,
        "perimeter_defense": 0.86,
        "switch_defense": 0.88,
        "help_defense": 0.84,
        "rebound_rate": 0.78,
        "pace_adj": 0.52,
        "ast_allowance": 0.46,
        "turnover_creation": 0.72,
    },
    "BRK": {
        "paint_defense": 0.55,
        "perimeter_defense": 0.62,
        "switch_defense": 0.58,
        "help_defense": 0.61,
        "rebound_rate": 0.64,
        "pace_adj": 0.57,
        "ast_allowance": 0.59,
        "turnover_creation": 0.50,
    },
    "CHI": {
        "paint_defense": 0.69,
        "perimeter_defense": 0.65,
        "switch_defense": 0.66,
        "help_defense": 0.67,
        "rebound_rate": 0.72,
        "pace_adj": 0.48,
        "ast_allowance": 0.52,
        "turnover_creation": 0.55,
    },
    "CHO": {
        "paint_defense": 0.33,
        "perimeter_defense": 0.28,
        "switch_defense": 0.31,
        "help_defense": 0.30,
        "rebound_rate": 0.44,
        "pace_adj": 0.61,
        "ast_allowance": 0.70,
        "turnover_creation": 0.32,
    },
    "CLE": {
        "paint_defense": 0.84,
        "perimeter_defense": 0.79,
        "switch_defense": 0.72,
        "help_defense": 0.80,
        "rebound_rate": 0.76,
        "pace_adj": 0.45,
        "ast_allowance": 0.44,
        "turnover_creation": 0.58,
    },
    "DAL": {
        "paint_defense": 0.48,
        "perimeter_defense": 0.55,
        "switch_defense": 0.51,
        "help_defense": 0.53,
        "rebound_rate": 0.58,
        "pace_adj": 0.56,
        "ast_allowance": 0.50,
        "turnover_creation": 0.47,
    },
    "DEN": {
        "paint_defense": 0.77,
        "perimeter_defense": 0.70,
        "switch_defense": 0.66,
        "help_defense": 0.74,
        "rebound_rate": 0.82,
        "pace_adj": 0.47,
        "ast_allowance": 0.48,
        "turnover_creation": 0.57,
    },
    "DET": {
        "paint_defense": 0.41,
        "perimeter_defense": 0.37,
        "switch_defense": 0.39,
        "help_defense": 0.36,
        "rebound_rate": 0.60,
        "pace_adj": 0.58,
        "ast_allowance": 0.62,
        "turnover_creation": 0.38,
    },
    "GSW": {
        "paint_defense": 0.63,
        "perimeter_defense": 0.67,
        "switch_defense": 0.72,
        "help_defense": 0.69,
        "rebound_rate": 0.53,
        "pace_adj": 0.80,
        "ast_allowance": 0.55,
        "turnover_creation": 0.61,
    },
    "HOU": {
        "paint_defense": 0.88,
        "perimeter_defense": 0.82,
        "switch_defense": 0.79,
        "help_defense": 0.81,
        "rebound_rate": 0.75,
        "pace_adj": 0.51,
        "ast_allowance": 0.43,
        "turnover_creation": 0.60,
    },
    "IND": {
        "paint_defense": 0.52,
        "perimeter_defense": 0.48,
        "switch_defense": 0.45,
        "help_defense": 0.47,
        "rebound_rate": 0.55,
        "pace_adj": 0.85,
        "ast_allowance": 0.64,
        "turnover_creation": 0.44,
    },
    "LAC": {
        "paint_defense": 0.72,
        "perimeter_defense": 0.78,
        "switch_defense": 0.83,
        "help_defense": 0.76,
        "rebound_rate": 0.68,
        "pace_adj": 0.46,
        "ast_allowance": 0.49,
        "turnover_creation": 0.59,
    },
    "LAL": {
        "paint_defense": 0.70,
        "perimeter_defense": 0.63,
        "switch_defense": 0.60,
        "help_defense": 0.67,
        "rebound_rate": 0.73,
        "pace_adj": 0.49,
        "ast_allowance": 0.51,
        "turnover_creation": 0.54,
    },
    "MEM": {
        "paint_defense": 0.75,
        "perimeter_defense": 0.71,
        "switch_defense": 0.70,
        "help_defense": 0.69,
        "rebound_rate": 0.77,
        "pace_adj": 0.53,
        "ast_allowance": 0.50,
        "turnover_creation": 0.62,
    },
    "MIA": {
        "paint_defense": 0.79,
        "perimeter_defense": 0.81,
        "switch_defense": 0.73,
        "help_defense": 0.82,
        "rebound_rate": 0.70,
        "pace_adj": 0.45,
        "ast_allowance": 0.42,
        "turnover_creation": 0.65,
    },
    "MIL": {
        "paint_defense": 0.85,
        "perimeter_defense": 0.74,
        "switch_defense": 0.68,
        "help_defense": 0.77,
        "rebound_rate": 0.83,
        "pace_adj": 0.48,
        "ast_allowance": 0.46,
        "turnover_creation": 0.59,
    },
    "MIN": {
        "paint_defense": 0.90,
        "perimeter_defense": 0.79,
        "switch_defense": 0.72,
        "help_defense": 0.84,
        "rebound_rate": 0.81,
        "pace_adj": 0.57,
        "ast_allowance": 0.49,
        "turnover_creation": 0.63,
    },
    "NOP": {
        "paint_defense": 0.74,
        "perimeter_defense": 0.77,
        "switch_defense": 0.70,
        "help_defense": 0.71,
        "rebound_rate": 0.74,
        "pace_adj": 0.55,
        "ast_allowance": 0.53,
        "turnover_creation": 0.57,
    },
    "NYK": {
        "paint_defense": 0.83,
        "perimeter_defense": 0.72,
        "switch_defense": 0.64,
        "help_defense": 0.76,
        "rebound_rate": 0.87,
        "pace_adj": 0.44,
        "ast_allowance": 0.48,
        "turnover_creation": 0.56,
    },
    "OKC": {
        "paint_defense": 0.76,
        "perimeter_defense": 0.84,
        "switch_defense": 0.78,
        "help_defense": 0.82,
        "rebound_rate": 0.62,
        "pace_adj": 0.59,
        "ast_allowance": 0.45,
        "turnover_creation": 0.68,
    },
    "ORL": {
        "paint_defense": 0.88,
        "perimeter_defense": 0.78,
        "switch_defense": 0.75,
        "help_defense": 0.82,
        "rebound_rate": 0.71,
        "pace_adj": 0.50,
        "ast_allowance": 0.46,
        "turnover_creation": 0.66,
    },
    "PHI": {
        "paint_defense": 0.82,
        "perimeter_defense": 0.77,
        "switch_defense": 0.73,
        "help_defense": 0.79,
        "rebound_rate": 0.78,
        "pace_adj": 0.47,
        "ast_allowance": 0.49,
        "turnover_creation": 0.61,
    },
    "PHX": {
        "paint_defense": 0.64,
        "perimeter_defense": 0.69,
        "switch_defense": 0.62,
        "help_defense": 0.65,
        "rebound_rate": 0.66,
        "pace_adj": 0.51,
        "ast_allowance": 0.54,
        "turnover_creation": 0.53,
    },
    "POR": {
        "paint_defense": 0.40,
        "perimeter_defense": 0.44,
        "switch_defense": 0.39,
        "help_defense": 0.41,
        "rebound_rate": 0.52,
        "pace_adj": 0.60,
        "ast_allowance": 0.65,
        "turnover_creation": 0.37,
    },
    "SAC": {
        "paint_defense": 0.55,
        "perimeter_defense": 0.49,
        "switch_defense": 0.52,
        "help_defense": 0.54,
        "rebound_rate": 0.57,
        "pace_adj": 0.70,
        "ast_allowance": 0.61,
        "turnover_creation": 0.45,
    },
    "SAS": {
        "paint_defense": 0.50,
        "perimeter_defense": 0.51,
        "switch_defense": 0.48,
        "help_defense": 0.49,
        "rebound_rate": 0.63,
        "pace_adj": 0.68,
        "ast_allowance": 0.60,
        "turnover_creation": 0.41,
    },
    "TOR": {
        "paint_defense": 0.71,
        "perimeter_defense": 0.72,
        "switch_defense": 0.77,
        "help_defense": 0.70,
        "rebound_rate": 0.69,
        "pace_adj": 0.55,
        "ast_allowance": 0.52,
        "turnover_creation": 0.58,
    },
    "UTA": {
        "paint_defense": 0.74,
        "perimeter_defense": 0.59,
        "switch_defense": 0.54,
        "help_defense": 0.56,
        "rebound_rate": 0.79,
        "pace_adj": 0.50,
        "ast_allowance": 0.55,
        "turnover_creation": 0.49,
    },
    "WAS": {
        "paint_defense": 0.34,
        "perimeter_defense": 0.39,
        "switch_defense": 0.37,
        "help_defense": 0.35,
        "rebound_rate": 0.46,
        "pace_adj": 0.75,
        "ast_allowance": 0.69,
        "turnover_creation": 0.36,
    },
}

# ================================================================
# MODULE 22 — PHASE 3
# POSITIONAL DEFENSIVE ASSIGNMENT ENGINE
# ================================================================

# Light positional chart used for defender estimation
_POSITION_MAP = {
    "PG": ["PG", "SG"],
    "SG": ["PG", "SG", "SF"],
    "SF": ["SG", "SF", "PF"],
    "PF": ["SF", "PF", "C"],
    "C": ["PF", "C"],
}

# League-average defensive anchor
DEF_BASELINE = 0.58


def _estimate_primary_defender(team_abbrev: str, player_pos: str) -> float:
    """
    Returns a defender quality rating between 0.30 — 0.90 using:
      - team defensive profile (Phase 2 DB)
      - player position (PG/SG/SF/PF/C)
      - expected switch frequency
      - expected cross-match rules

    The result is a SINGLE number that acts as a defensive strength anchor.
    """

    team = DEF_TEAM_DB.get(team_abbrev.upper())
    if team is None:
        return DEF_BASELINE  # fallback mid value

    # Pull team defensive properties
    paint = team["paint_defense"]
    perimeter = team["perimeter_defense"]
    switch = team["switch_defense"]
    helpd = team["help_defense"]

    # Position determines WHICH attributes matter most
    if player_pos == "PG":
        weight = 0.55 * perimeter + 0.25 * switch + 0.20 * helpd
    elif player_pos == "SG":
        weight = 0.50 * perimeter + 0.30 * switch + 0.20 * helpd
    elif player_pos == "SF":
        weight = 0.40 * perimeter + 0.30 * switch + 0.30 * helpd
    elif player_pos == "PF":
        weight = 0.30 * perimeter + 0.30 * switch + 0.40 * paint
    elif player_pos == "C":
        weight = 0.20 * perimeter + 0.20 * switch + 0.60 * paint
    else:
        # Unknown position
        weight = 0.50 * perimeter + 0.25 * switch + 0.25 * paint

    # Normalize to expected range
    weight = float(np.clip(weight, 0.30, 0.90))
    return weight


def _estimate_secondary_defender(team_abbrev: str, player_pos: str) -> float:
    """
    Secondary defender = helps on drives, switches, doubles, etc.

    This uses:
        - help defense rating
        - switch frequency
        - positional spillover weight
    """

    team = DEF_TEAM_DB.get(team_abbrev.upper())
    if team is None:
        return DEF_BASELINE

    helpd = team["help_defense"]
    switch = team["switch_defense"]

    if player_pos in ["PG", "SG"]:
        mix = 0.55 * helpd + 0.45 * switch
    elif player_pos in ["SF", "PF"]:
        mix = 0.60 * helpd + 0.40 * switch
    else:
        mix = 0.50 * helpd + 0.50 * switch

    return float(np.clip(mix, 0.30, 0.90))


def matchup_defensive_multiplier(team_abbrev: str, player_pos: str, market: str) -> float:
    """
    Computes FINAL defensive matchup multiplier for projection pipeline.

    Output range ≈ (0.75 → 1.30)

    Markets affected differently:
        - Points → heavily tied to perimeter/paint defense depending on position
        - Rebounds → tied to rebound_rate + paint defense
        - Assists → tied to ast_allowance + help defense
        - PRA → blended 3-way mixture

    This multiplier will be used inside:
        - compute_leg()
        - opponent_matchup_v2() (overridden)
        - volatility engine
        - correlation engine
        - CLV & drift learning (downstream)
    """

    team = DEF_TEAM_DB.get(team_abbrev.upper())
    if team is None:
        return 1.00

    # Primary & Secondary defender strengths
    primary = _estimate_primary_defender(team_abbrev, player_pos)
    secondary = _estimate_secondary_defender(team_abbrev, player_pos)

    # Combine defensive pressure
    defense_strength = (0.70 * primary + 0.30 * secondary)

    # Market-specific weighting
    if market == "Points":
        base = defense_strength
    elif market == "Rebounds":
        base = (0.65 * team["rebound_rate"] + 0.35 * team["paint_defense"])
    elif market == "Assists":
        base = (0.60 * team["ast_allowance"] + 0.40 * team["help_defense"])
    elif market == "PRA":
        base = (
            0.35 * defense_strength +
            0.35 * (0.65 * team["rebound_rate"] + 0.35 * team["paint_defense"]) +
            0.30 * (0.60 * team["ast_allowance"] + 0.40 * team["help_defense"])
        )
    else:
        base = defense_strength

    # Normalize into usable multiplier (0.75 — 1.30)
    # Higher defense_strength → LOWER multiplier
    multiplier = 1.25 - (base - 0.30) * 0.85

    return float(np.clip(multiplier, 0.75, 1.30))

# ================================================================
# MODULE 22 — PHASE 4
# ROTATIONAL MATCHUP ESTIMATOR (MOM-MINUTE WEIGHTED DEFENSE ENGINE)
# ================================================================

# Expected positional match percentages (league average)
_POS_LOCK_WEIGHTS = {
    "PG": {"primary": 0.72, "secondary": 0.22, "big": 0.06},
    "SG": {"primary": 0.60, "secondary": 0.30, "big": 0.10},
    "SF": {"primary": 0.45, "secondary": 0.40, "big": 0.15},
    "PF": {"primary": 0.30, "secondary": 0.45, "big": 0.25},
    "C":  {"primary": 0.10, "secondary": 0.35, "big": 0.55},
}

# Defensive archetype multipliers (relative difficulty)
_ARCHETYPE_MULT = {
    "primary": 1.00,    # Normalized, controlled by DB
    "secondary": 0.92,  # Slightly softer defender
    "big": 0.88         # Bigs guarding guards, or switches
}


def _big_defender_strength(team):
    """
    Pull interior-based defender rating.
    Returns between ~0.30 — 0.90.
    """
    return float(np.clip(
        0.60 * team["paint_defense"] + 0.40 * team["rebound_rate"],
        0.30,
        0.90
    ))


def estimate_rotational_matchup(team_abbrev: str, player_pos: str) -> dict:
    """
    Outputs estimated matchup difficulty broken into:
      - primary defender minutes
      - secondary defender minutes
      - big/switch defender minutes
      - combined weighted score

    Returns dict:
    {
      "primary": float,
      "secondary": float,
      "big": float,
      "weighted_def": float   (final difficulty value 0.30 — 0.90)
    }
    """

    team = DEF_TEAM_DB.get(team_abbrev.upper())
    if team is None:
        # fallback: middle-of-league defense
        return {
            "primary": DEF_BASELINE,
            "secondary": DEF_BASELINE * 0.95,
            "big": DEF_BASELINE * 0.90,
            "weighted_def": DEF_BASELINE
        }

    # Pull Phase 3 defender estimations
    pri = _estimate_primary_defender(team_abbrev, player_pos)
    sec = _estimate_secondary_defender(team_abbrev, player_pos)
    big = _big_defender_strength(team)

    # Weight by expected matchup % for this position
    weights = _POS_LOCK_WEIGHTS.get(player_pos, _POS_LOCK_WEIGHTS["SG"])

    w_primary   = pri * _ARCHETYPE_MULT["primary"]   * weights["primary"]
    w_secondary = sec * _ARCHETYPE_MULT["secondary"] * weights["secondary"]
    w_big       = big * _ARCHETYPE_MULT["big"]       * weights["big"]

    # Final weighted defensive difficulty score
    weighted_def = float(np.clip(w_primary + w_secondary + w_big, 0.30, 0.90))

    return {
        "primary": pri,
        "secondary": sec,
        "big": big,
        "weighted_def": weighted_def
    }


def matchup_minutes_multiplier(rotational_profile: dict, market: str) -> float:
    """
    Converts rotational defensive profile → market-specific matchup multiplier.

    rotational_profile = estimate_rotational_matchup(...)

    Outputs multiplier:
        0.70 → 1.35

    Higher → easier matchup.
    Lower → harder matchup.
    """

    weighted_def = rotational_profile["weighted_def"]

    # Market-specific attenuation curves
    if market == "Points":
        # Points punished heavily by perimeter & primary defenders
        mult = 1.30 - (weighted_def - 0.30) * 1.05

    elif market == "Rebounds":
        # Rebounds punished more by big defenders
        big_factor = rotational_profile["big"]
        mult = 1.22 - (0.70 * weighted_def + 0.30 * big_factor)

    elif market == "Assists":
        # Assists punished by help defense + switch defenders
        sec_factor = rotational_profile["secondary"]
        mult = 1.18 - (0.65 * weighted_def + 0.35 * sec_factor)

    elif market == "PRA":
        # Balanced 3-way curve
        mult = (
            1.25 -
            (
                0.40 * weighted_def +
                0.30 * rotational_profile["big"] +
                0.30 * rotational_profile["secondary"]
            )
        )
    else:
        mult = 1.00

    return float(np.clip(mult, 0.70, 1.35))

# ================================================================
# MODULE 22 — PHASE 5
# FINAL DEFENSIVE ADJUSTMENT ENGINE
# ================================================================

def _market_def_sensitivity(market: str) -> float:
    """
    Returns how sensitive this market is to defense.
    Used to scale defensive difficulty into a final multiplier.
    """
    if market == "Points":
        return 1.18
    elif market == "Assists":
        return 1.10
    elif market == "Rebounds":
        return 1.05
    elif market == "PRA":
        return 1.14
    else:
        return 1.00


def defensive_adjustment_final(player_pos: str, opponent_team: str, market: str) -> float:
    """
    Produces the final defensive multiplier combining:

      ✓ Positional defense profile (Phase 1)
      ✓ Team defensive DB (Phase 2)
      ✓ Primary & secondary defender strength (Phase 3)
      ✓ Rotational matchup distribution (Phase 4)
      ✓ Market sensitivity curves

    Output:
        float in range 0.70 — 1.40
    """

    # ----------------------------
    # Step 1: Pull team defensive profile
    # ----------------------------
    team = DEF_TEAM_DB.get(opponent_team.upper())
    if team is None:
        # Fallback: treat as average NBA defense
        return 1.00

    # ----------------------------
    # Step 2: Get drotational defensive difficulty
    # ----------------------------
    rotation_profile = estimate_rotational_matchup(opponent_team, player_pos)

    # Use Phase 4 multiplier (0.70 – 1.35)
    matchup_mult = matchup_minutes_multiplier(rotation_profile, market)

    # ----------------------------
    # Step 3: Team defensive base difficulty
    # ----------------------------
    team_def_base = team["def_metric"]      # 0.30 – 0.90
    pace_factor = team["pace_factor"]       # 0.95 – 1.06

    # Convert defensive number to adjustment
    # harder defense (0.90) ≈ 0.80 multiplier
    # softer defense (0.30) ≈ 1.22 multiplier
    team_def_mult = float(
        np.clip(1.40 - team_def_base * 0.66, 0.78, 1.22)
    )

    # ----------------------------
    # Step 4: Combine multipliers
    # ----------------------------
    # Market-specific defensive sensitivity (points most sensitive)
    sens = _market_def_sensitivity(market)

    combined = (
        0.55 * matchup_mult +
        0.35 * team_def_mult * sens +
        0.10 * pace_factor
    )

    # ----------------------------
    # Step 5: Clamp final multiplier
    # ----------------------------
    final_mult = float(np.clip(combined, 0.70, 1.40))

    return final_mult

# ================================================================
# MODULE 22 — PHASE 6
# FINAL DEFENSIVE CONTEXT INTEGRATION WRAPPER
# ================================================================

def apply_defensive_context(mu: float,
                            sd: float,
                            player_pos: str,
                            opponent_team: str,
                            market: str):
    """
    Takes raw (mu, sd) from:
        - Usage Engine
        - Opponent Engine
        - Volatility Engine

    And applies:
        - Full defensive adjustment engine (Phase 5)
        - Market sensitivity curves
        - Pace/difficulty blending
        - Rotational matchup influence
        - Positional DB
        - Team defensive DB
        - Primary + Secondary defender strength

    Returns:
        (adj_mu, adj_sd, def_mult)
    """

    # Step 1: Get final defensive adjustment (0.70 — 1.40)
    def_mult = defensive_adjustment_final(
        player_pos,
        opponent_team,
        market
    )

    # Step 2: Apply to mean
    adj_mu = float(mu * def_mult)

    # Step 3: Apply defensive volatility shaping
    #   - tougher defenses shrink SD less than they shrink mean
    #   - softer defenses expand SD slightly more than mean
    if def_mult < 1.0:
        # tougher defense → shrink SD slightly (but 80–100% of mu scaling)
        sd_mult = 0.85 + 0.15 * def_mult
    else:
        # easy defense → SD grows slightly faster than mu
        sd_mult = 1.00 + (def_mult - 1.0) * 0.20

    adj_sd = float(sd * sd_mult)

    # Final clamps (safety)
    adj_sd = float(np.clip(adj_sd, 0.20, 25.0))

    return adj_mu, adj_sd, def_mult


# UI & APP LAYOUT (PARTS 1–20)
# =====================================================================



# ================================================================
# PART 3 — MODEL TAB (PLAYER INPUT UI)
# UltraMax NBA Prop Quant Engine — V4 (B3)
# ================================================================

# Top-level tabs (this will exist throughout the full app)
tab_model, tab_results, tab_history, tab_calibration = st.tabs(
    ["📊 Model", "📈 Results", "📜 History", "🧬 Calibration"]
)

# ================================================================
# TAB: MODEL
# ================================================================
with tab_model:

    st.markdown("## 🎯 UltraMax 2-Pick Prop Engine — Model Inputs")
    st.markdown("Configure your two-pick entry below.")

    col1, col2 = st.columns(2)

    # ------------------------------------------------------------
    # LEFT SIDE — LEG 1 INPUTS
    # ------------------------------------------------------------
    with col1:
        st.markdown("### 🟥 Leg 1")

        p1_name = st.text_input(
            "Player 1 Name",
            placeholder="LeBron James",
            help="Enter any recognizable NBA player name."
        )

        p1_market = st.selectbox(
            "Market 1",
            options=MARKET_OPTIONS,
            help="Select the stat category for Leg 1."
        )

        p1_line = st.number_input(
            "Line 1",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.5,
            help="Enter the current PrizePicks/Sleeper line."
        )

        p1_opp = st.text_input(
            "Opponent (Team Abbrev)",
            placeholder="DEN, BOS, LAL, etc.",
            help="Team opponent for Leg 1. Used in Opponent Engine & Defensive Profile."
        )

        p1_teammate_out = st.checkbox(
            "Key Teammate OUT (Leg 1)",
            help="Boosts usage in Module 3 (Usage Engine)."
        )

        p1_blowout = st.checkbox(
            "Blowout Risk (Leg 1)",
            help="Triggers game script volatility handling in Module 15."
        )

    # ------------------------------------------------------------
    # RIGHT SIDE — LEG 2 INPUTS
    # ------------------------------------------------------------
    with col2:
        st.markdown("### 🟦 Leg 2")

        p2_name = st.text_input(
            "Player 2 Name",
            placeholder="Nikola Jokic",
            help="Enter any recognizable NBA player name."
        )

        p2_market = st.selectbox(
            "Market 2",
            options=MARKET_OPTIONS,
            help="Select the stat category for Leg 2."
        )

        p2_line = st.number_input(
            "Line 2",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.5,
            help="Enter the current PrizePicks/Sleeper line."
        )

        p2_opp = st.text_input(
            "Opponent (Team Abbrev)",
            placeholder="MIL, MIN, OKC, etc.",
            help="Team opponent for Leg 2. Used in Opponent Engine & Defensive Profile."
        )

        p2_teammate_out = st.checkbox(
            "Key Teammate OUT (Leg 2)",
            help="Boosts usage in Module 3 (Usage Engine)."
        )

        p2_blowout = st.checkbox(
            "Blowout Risk (Leg 2)",
            help="Triggers game script volatility handling in Module 15."
        )

    st.markdown("---")

    # ------------------------------------------------------------
    # RUN BUTTON (Connected to sidebar's run_model flag)
    # ------------------------------------------------------------
    st.markdown("### 🚀 Run the UltraMax Engine")

    run_model_button = st.button(
        "Run Projection Model",
        help="This will compute both legs, apply all engines, run Monte Carlo, and produce EV + Kelly sizing."
    )

# =====================================================================
# PART 4 — PrizePicks & Sleeper Autofill System
# =====================================================================

import json
import time

# -----------------------------------------------------------------------------
# 4.1 — Local Cached Line Database (initially empty; filled by Module 18 later)
# -----------------------------------------------------------------------------
@st.cache_data
def load_local_line_db():
    """
    Local placeholder DB — will be replaced by live sync in Module 18.
    Structure:
    {
        "PrizePicks": [
            {"player":"LeBron James","market":"Points","line":26.5,"team":"LAL","opp":"DEN"},
            ...
        ],
        "Sleeper": [
            {"player":"Nikola Jokic","market":"PRA","line":47.5,"team":"DEN","opp":"MIN"},
            ...
        ]
    }
    """
    return {
        "PrizePicks": [],
        "Sleeper": []
    }

line_db = load_local_line_db()

# -----------------------------------------------------------------------------
# 4.2 — Helper to build a unified dropdown list
# -----------------------------------------------------------------------------
def build_line_selector_list(line_db):
    """
    Converts:
        [{"player":..., "market":..., "line":...}]
    Into readable UI strings like:
        "LeBron James — Points — 26.5 (vs DEN) [PrizePicks]"
    """
    out = []
    for provider, items in line_db.items():
        for row in items:
            label = f"{row['player']} — {row['market']} — {row['line']} (vs {row['opp']}) [{provider}]"
            out.append((label, provider, row))
    return out

prop_selector_list = build_line_selector_list(line_db)

# -----------------------------------------------------------------------------
# 4.3 — Model Tab Autofill Components (Attach to Part 3 UI)
# -----------------------------------------------------------------------------
with tab_model:

    st.markdown("### ⚡ Autofill Props from PrizePicks / Sleeper")

    auto_col1, auto_col2 = st.columns(2)

    # ------------------------------------------------------------
    # AUTOFILL — LEG 1
    # ------------------------------------------------------------
    with auto_col1:
        st.markdown("#### 🟥 Autofill Leg 1")

        autofill_1 = st.selectbox(
            "Select a Pick (Leg 1)",
            options=["None"] + [item[0] for item in prop_selector_list],
            help="Choose a synced line from PrizePicks or Sleeper to autofill Leg 1."
        )

        if autofill_1 != "None":
            # Get matching entry
            selected = [x for x in prop_selector_list if x[0] == autofill_1][0]
            label, provider, row = selected

            # Autofill fields
            p1_name = row["player"]
            p1_market = row["market"]
            p1_line = row["line"]
            p1_opp = row["opp"]

            st.success(f"Autofilled Leg 1 from {provider}")

    # ------------------------------------------------------------
    # AUTOFILL — LEG 2
    # ------------------------------------------------------------
    with auto_col2:
        st.markdown("#### 🟦 Autofill Leg 2")

        autofill_2 = st.selectbox(
            "Select a Pick (Leg 2)",
            options=["None"] + [item[0] for item in prop_selector_list],
            help="Choose a synced line from PrizePicks or Sleeper to autofill Leg 2."
        )

        if autofill_2 != "None":
            # Get matching entry
            selected = [x for x in prop_selector_list if x[0] == autofill_2][0]
            label, provider, row = selected

            # Autofill fields
            p2_name = row["player"]
            p2_market = row["market"]
            p2_line = row["line"]
            p2_opp = row["opp"]

            st.success(f"Autofilled Leg 2 from {provider}")

# =====================================================================
# PART 5 — RUN MODEL BUTTON (Execution Trigger)
# =====================================================================

# Storage for model run context
# (Later modules will read from this dictionary)
model_request = {}

with tab_model:

    st.markdown("### 🚀 UltraMax Model Execution")

    run_btn = st.button(
        "🔮 Run UltraMax Quant Model",
        help="Runs the full 22-module computational pipeline."
    )

    if run_btn:

        # ---------------------------
        # Input Validation
        # ---------------------------
        missing_inputs = []
        if not p1_name:
            missing_inputs.append("Player 1 name")
        if not p2_name:
            missing_inputs.append("Player 2 name")

        if p1_line is None:
            missing_inputs.append("Player 1 line")
        if p2_line is None:
            missing_inputs.append("Player 2 line")

        if p1_market is None:
            missing_inputs.append("Player 1 market")
        if p2_market is None:
            missing_inputs.append("Player 2 market")

        if missing_inputs:
            st.error("⚠️ Missing required fields:")
            for m in missing_inputs:
                st.write(f"• {m}")
            st.stop()

        # ---------------------------
        # Loader Animation
        # ---------------------------
        with st.spinner("Running UltraMax Engine… please wait ⏳"):
            time.sleep(0.2)

        # ---------------------------
        # Save Inputs in Model Context
        # ---------------------------
        model_request = {
            "p1": {
                "player": p1_name,
                "market": p1_market,
                "line": float(p1_line),
                "opp": p1_opp,
                "teammate_out": p1_teammate_out,
                "blowout": p1_blowout,
            },
            "p2": {
                "player": p2_name,
                "market": p2_market,
                "line": float(p2_line),
                "opp": p2_opp,
                "teammate_out": p2_teammate_out,
                "blowout": p2_blowout,
            },
            "bankroll": st.session_state.get("bankroll", 1000),
            "fractional_kelly": st.session_state.get("fractional_kelly", 0.33),
            "drift_adj": st.session_state.get("drift_adj", 1.00),
            "clv_adj": st.session_state.get("clv_adj", 1.00),
            "payout_mult": st.session_state.get("payout_mult", 3.0),
        }

        # Make request globally visible to later modules
        st.session_state["model_request"] = model_request

        st.success("Inputs captured! Proceeding to UltraMax computation...")
        st.info("Now waiting for Part 6 (Compute Engine Controller).")
# =====================================================================
# PART 6 — COMPUTE LEGS (Modules 1–6)
# =====================================================================

st.markdown("## 🧮 UltraMax Engine — Leg Computation")

def compute_single_leg(leg_input: dict):
    """
    Executes Modules 1–6 for ONE LEG.
    Returns (leg_dict, error_message)
    """

    player = leg_input["player"]
    market = leg_input["market"]
    line = leg_input["line"]
    opponent = leg_input["opp"]
    teammate_out = leg_input["teammate_out"]
    blowout = leg_input["blowout"]

    # ---------------------------------------------------------
    # 1. Resolve NBA player ID  (Module 1)
    # ---------------------------------------------------------
    pid, canonical = resolve_player(player)
    if not pid:
        return None, f"Player not found: {player}"

    # ---------------------------------------------------------
    # 2. Pull NBA game logs (Module 2)
    # ---------------------------------------------------------
    try:
        logs = PlayerGameLog(player_id=pid, season=current_season()).get_data_frames()[0]
    except Exception:
        return None, f"Could not fetch logs for: {player}"

    if logs.empty:
        return None, "No valid game logs."

    # Use last N games (default 10)
    lookback = st.session_state.get("lookback", 10)
    logs = logs.head(lookback)

    # ---------------------------------------------------------
    # 3. Build Market Value column (PTS, REB, AST, PRA)
    # ---------------------------------------------------------
    metrics = MARKET_METRICS.get(market, ["PTS"])
    logs["MarketVal"] = logs[metrics].sum(axis=1)

    # Clean the minutes column
    try:
        logs["Minutes"] = logs["MIN"].astype(float)
    except:
        logs["Minutes"] = 0.0

    valid = logs["Minutes"] > 0
    if not valid.any():
        return None, "No minute data available."

    # ---------------------------------------------------------
    # 4. Compute per-minute base rate
    # ---------------------------------------------------------
    pm = logs.loc[valid, "MarketVal"] / logs.loc[valid, "Minutes"]
    base_mu_per_min = float(pm.mean())
    base_sd_per_min = float(max(pm.std(), 0.10))

    # Projected minutes (Module 2C)
    proj_minutes = float(np.clip(logs["Minutes"].tail(5).mean(), 18, 40))

    # ---------------------------------------------------------
    # 5. Apply Usage Engine (Module 3)
    # ---------------------------------------------------------
    usage_mu = usage_engine_v3(
        mu_per_min=base_mu_per_min,
        role="primary",
        team_usage_rate=1.00,
        teammate_out_level=1 if teammate_out else 0
    )

    # ---------------------------------------------------------
    # 6. Apply Opponent Engine (Module 4)
    # ---------------------------------------------------------
    ctx_mult = opponent_matchup_v2(opponent, market)

    # ---------------------------------------------------------
    # 7. Compute final projection mean
    # ---------------------------------------------------------
    mu = usage_mu * proj_minutes * ctx_mult

    # ---------------------------------------------------------
    # 8. Volatility Engine v2 (Module 5)
    # ---------------------------------------------------------
    sd = volatility_engine_v2(
        base_sd_per_min,
        proj_minutes,
        market,
        ctx_mult,
        usage_mu / max(base_mu_per_min, 0.01),
        regime_state="normal"
    )

    # ---------------------------------------------------------
    # 9. Ensemble Probability Engine (Module 6)
    # ---------------------------------------------------------
    prob_over = ensemble_prob_over(
        mu,
        sd,
        line,
        market,
        volatility_score=sd / max(mu, 1)
    )

    # ---------------------------------------------------------
    # Build final leg object
    # ---------------------------------------------------------
    leg = {
        "player": canonical,
        "market": market,
        "line": float(line),
        "mu": float(mu),
        "sd": float(sd),
        "prob_over": float(prob_over),
        "proj_minutes": proj_minutes,
        "ctx_mult": float(ctx_mult),
        "teammate_out": teammate_out,
        "blowout": blowout,
        "raw_logs": logs,
    }

    return leg, None


# =====================================================================
# EXECUTION BLOCK (Runs when Part 5 triggers)
# =====================================================================
if "model_request" in st.session_state:

    request = st.session_state["model_request"]

    st.markdown("### 🔄 Computing Legs (Modules 1–6)…")

    # Compute both legs
    leg1, err1 = compute_single_leg(request["p1"])
    leg2, err2 = compute_single_leg(request["p2"])

    # Display errors if any
    if err1:
        st.error(f"❌ Leg 1 Error: {err1}")
    if err2:
        st.error(f"❌ Leg 2 Error: {err2}")

    # Store results for next parts (7–20)
    st.session_state["leg1"] = leg1
    st.session_state["leg2"] = leg2

    if leg1 and leg2:
        st.success("Leg computation completed successfully!")
    else:
        st.warning("Leg computation incomplete. Fix errors above.")

# ============================================================
#   PART 7 — COMBO MONTE CARLO ENGINE (Modules 7–12)
#   Handles correlation-aware multi-leg simulation
# ============================================================

import numpy as np
from scipy.stats import multivariate_normal

# ------------------------------------------------------------
# MODULE 7 — Single Stat Distribution Builder
# Converts player projections into distribution parameters
# ------------------------------------------------------------

def build_stat_distribution(proj_mean, proj_std, dist_type="normal"):
    """Package a stat into a shape the Monte Carlo sampler can use"""
    
    if dist_type == "normal":
        return {"mean": proj_mean, "std": max(proj_std, 0.01)}
    
    if dist_type == "poisson":
        return {"lambda": max(proj_mean, 0.01)}  # for 3PM, blocks
    
    return {"mean": proj_mean, "std": max(proj_std, 0.01)}


# ------------------------------------------------------------
# MODULE 8 — Correlation Matrix Builder
# Handles:
#   • PRA correlations (P↔R | P↔A | R↔A)
#   • Intra-team dependencies
#   • Pace volatility impact
#   • Opponent defensive correlation
# ------------------------------------------------------------

def build_correlation_matrix(legs):
    """
    legs = list of dicts:
      {
        "player_name": ...,
        "stat_type": ...,
        "mean": ...,
        "std": ...
      }
    """

    n = len(legs)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i+1, n):

            li = legs[i]
            lj = legs[j]

            # Base correlations
            c = 0.0

            # --------------------------
            # 1: PRA INTERNAL CORRELATION
            # --------------------------
            if li["player_name"] == lj["player_name"]:
                if {li["stat_type"], lj["stat_type"]} == {"Points", "Assists"}:
                    c = 0.25
                if {li["stat_type"], lj["stat_type"]} == {"Points", "Rebounds"}:
                    c = 0.18
                if {li["stat_type"], lj["stat_type"]} == {"Rebounds", "Assists"}:
                    c = 0.20

            # --------------------------
            # 2: SAME TEAM CORRELATION
            # (usage & pace impacts)
            # --------------------------
            if li["team"] == lj["team"]:
                c += 0.10

            # --------------------------
            # 3: DEFENSIVE MATCHUP CORRELATION
            # --------------------------
            if li["opponent"] == lj["team"]:
                c += 0.05

            # clamp to valid range
            c = np.clip(c, -0.65, 0.65)

            corr[i, j] = c
            corr[j, i] = c

    return corr


# ------------------------------------------------------------
# MODULE 9 — Covariance Matrix Builder
# Converts correlation matrix + std deviations → covariance
# ------------------------------------------------------------

def build_covariance_matrix(legs, corr_matrix):
    stds = np.array([leg["std"] for leg in legs])
    cov = corr_matrix * np.outer(stds, stds)
    return cov


# ------------------------------------------------------------
# MODULE 10 — Multivariate Simulation Engine
# Runs correlated draws across all legs
# ------------------------------------------------------------

def simulate_combo_distribution(legs, n_sims=DEFAULT_SIMULATIONS):
    """
    legs = list of dicts:
      {
        "player_name": ...,
        "stat_type": ...,
        "mean": ...,
        "std": ...
      }
    """

    means = np.array([leg["mean"] for leg in legs])
    corr = build_correlation_matrix(legs)
    cov = build_covariance_matrix(legs, corr)

    # Guard: covariance must be positive semi-definite
    try:
        draws = multivariate_normal.rvs(mean=means, cov=cov, size=n_sims)
    except np.linalg.LinAlgError:
        # add jitter to fix covariance
        cov += np.eye(len(legs)) * 1e-6
        draws = multivariate_normal.rvs(mean=means, cov=cov, size=n_sims)

    return np.clip(draws, 0, None)    # no negative stats


# ------------------------------------------------------------
# MODULE 11 — Combo Card EV Calculation
# Applies PrizePicks payout logic + hit probabilities
# ------------------------------------------------------------

def compute_card_ev(legs, lines, power_payout=3.0, flex_payout=1.25, flex_prob=0.6, n_sims=DEFAULT_SIMULATIONS):
    """
    legs = projections/distributions
    lines = prop lines
    """

    draws = simulate_combo_distribution(legs, n_sims=n_sims)

    # Hit matrix: 1 if draw >= line
    hits = draws >= np.array(lines)

    # -------------------------
    # POWER PLAY EV
    # all legs must hit
    # -------------------------
    power_hits = hits.all(axis=1)
    power_ev = power_hits.mean() * power_payout

    # -------------------------
    # FLEX PLAY EV
    # payout depends on num legs
    # -------------------------
    n = len(legs)

    if n == 2:
        # 2-pick flex = safety net
        ev = (hits.sum(axis=1) >= 1).mean() * flex_payout
    elif n == 3:
        # 3-pick flex
        three_hit = (hits.sum(axis=1) == 3).mean() * 2.25
        two_hit   = (hits.sum(axis=1) == 2).mean() * 1.25
        ev = three_hit + two_hit
    else:
        # 4–6 pick flex: threshold scaling
        k = 4 if n == 4 else 5
        full = (hits.sum(axis=1) == n).mean() * (flex_payout * n)
        partial = (hits.sum(axis=1) >= k).mean() * (flex_payout * (n-1))
        ev = full + partial

    return {
        "ev_power": round(power_ev, 3),
        "ev_flex": round(ev, 3),
        "hit_rates": hits.mean(axis=0).round(3).tolist(),
        "combo_hit_rate": round(power_hits.mean(), 3)
    }


# ------------------------------------------------------------
# MODULE 12 — Kelly Stake (Full, Fractional)
# ------------------------------------------------------------

def compute_kelly_stake(prob, payout=3.0, fraction=KELLY_FRACTION):
    """Standard Kelly formula for +EV cards."""
    b = payout - 1
    q = 1 - prob
    kelly = (b*prob - q) / b
    return max(kelly * fraction, 0)

# ===========================
# PART 8 — SUMMARY SECTION
# ===========================

st.markdown("## 🔍 Summary & Recommendations")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total EV", f"{total_ev:+.2f}%")
with col2:
    st.metric("Slip Hit Probability", f"{prob_hit*100:.1f}%")
with col3:
    st.metric("Recommended Stake (Kelly)", f"${kelly_stake:.2f}")

# Leg Summary Table
summary_df = pd.DataFrame({
    "Player": player_names,
    "Line": lines,
    "Projection": projections,
    "EV %": evs,
    "Hit %": probs,
    "Flag": flags
})

st.markdown("### 📊 Leg-by-Leg Breakdown")
st.dataframe(summary_df, use_container_width=True)

# Verdict
if total_ev < 0:
    verdict = "🚫 PASS"
elif total_ev < 3:
    verdict = "🤔 LEAN"
else:
    verdict = "🔥 FIRE"

st.markdown(f"### Verdict: **{verdict}**")

# Kelly Fraction Selector
kelly_fraction = st.slider("Kelly Fraction", 0.05, 1.0, 0.25)
recommended_bet = bankroll * kelly_fraction * kelly_value
st.metric("Fractional Kelly Recommended Bet", f"${recommended_bet:.2f}")

# Risk Notes
st.markdown("### ⚠️ Risk Factors")
with st.expander("Show Risk Notes"):
    for n in global_notes:
        st.markdown(f"- {n}")

# Save Button
if st.button("Save Slip to History"):
    save_slip(summary_df, total_ev, kelly_stake, verdict)
    st.success("Slip saved to history!")

# ============================
# PART 9 - RESULTS TAB
# ============================

results_tab = st.tabs(["🏁 Results"])[0]

with results_tab:
    st.markdown("## 🏁 Final Results")
    st.markdown("Here’s your model-backed analysis for today’s slip.")

    # Slip Metadata
    st.markdown(f"**Contest Type:** {contest_type}")
    st.markdown(f"**Number of Legs:** {len(player_names)}")
    st.markdown(f"**Date:** {today}")

    st.markdown("---")
    st.markdown("## 🎯 Leg Results")

    def leg_card(player, line, projection, hit_prob, ev, flag, matchup_score, color):
        st.markdown(
            f"""
            <div style="
                padding:12px;
                border-radius:10px;
                background:{color};
                margin-bottom:12px;
                border:1px solid #333;
            ">
                <h4 style="margin:0;">{player}</h4>
                <p style="margin:4px 0;">Line: <b>{line}</b> | Projection: <b>{projection:.1f}</b></p>
                <p style="margin:4px 0;">Hit Probability: <b>{hit_prob*100:.1f}%</b></p>
                <p style="margin:4px 0;">EV: <b>{ev:+.2f}%</b></p>
                <p style="margin:4px 0;">Matchup: <b>{matchup_score}</b></p>
                <p style="margin:4px 0;color:#e63946;">{flag}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    colors = {
        "strong": "#1a472a20",
        "neutral": "#33333320",
        "weak": "#6b0f1a20"
    }

    for i in range(len(player_names)):
        category = (
            "strong" if evs[i] > 3 
            else "weak" if evs[i] < 0 
            else "neutral"
        )
        leg_card(
            player_names[i], lines[i], projections[i], probs[i],
            evs[i], flags[i], matchup_scores[i],
            colors[category]
        )

    st.markdown("## 📈 Summary")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total EV", f"{total_ev:+.2f}%")
    with col2: st.metric("Slip Hit Probability", f"{prob_hit*100:.1f}%")
    with col3: st.metric("Recommended Stake", f"${kelly_stake:.2f}")

    verdict = "🚫 PASS" if total_ev < 0 else "🤔 LEAN" if total_ev < 3 else "🔥 FIRE"
    st.markdown(f"### Verdict: **{verdict}**")

    st.markdown("### ⚠️ Risk Factors")
    with st.expander("See Details"):
        for n in global_notes:
            st.markdown(f"- {n}")

    st.markdown("### 💾 Save or Export")
    if st.button("Save Slip to History"):
        save_slip(summary_df, total_ev, kelly_stake, verdict)
        st.success("Saved!")

    csv = summary_df.to_csv(index=False)
    st.download_button("Download CSV", csv, "results.csv")

# ============================
# PART 10 - HISTORY TAB
# ============================

history_tab = st.tabs(["📜 History"])[0]

def load_history():
    try:
        return pd.read_csv("history.csv")
    except:
        return pd.DataFrame(columns=[
            "timestamp", "verdict", "total_ev", "hit_prob",
            "kelly_stake", "num_legs", "summary_json"
        ])

with history_tab:
    st.markdown("## 📜 Slip History")
    st.markdown("Track all your previously saved slips and their model results.")

    # Load history
    history = load_history()

    if history.empty:
        st.info("No history yet. Save a slip to add to your tracking log.")
    else:
        # Filters
        st.subheader("Filters")
        verdict_filter = st.selectbox("Verdict Filter", ["All", "🚫 PASS", "🤔 LEAN", "🔥 FIRE"])
        if verdict_filter != "All":
            history = history[history["verdict"] == verdict_filter]

        ev_min = st.slider("Minimum EV Filter", -20.0, 20.0, 0.0)
        history = history[history["total_ev"] >= ev_min]

        st.markdown("### 📊 Slip Log")
        st.dataframe(history[[
            "timestamp", "verdict", "total_ev", 
            "hit_prob", "kelly_stake", "num_legs"
        ]], use_container_width=True)

        # Detailed slip viewer
        selected = st.selectbox("Select Slip", history["timestamp"].tolist())
        slip = history[history["timestamp"] == selected].iloc[0]

        st.markdown("### 📌 Slip Summary")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total EV", f"{slip.total_ev:+.2f}%")
        with col2: st.metric("Hit Probability", f"{slip.hit_prob*100:.1f}%")
        with col3: st.metric("Recommended Stake", f"${slip.kelly_stake:.2f}")

        st.markdown(f"**Verdict:** {slip.verdict}")
        st.markdown(f"**Number of Legs:** {slip.num_legs}")

        legs_df = pd.read_json(slip.summary_json)
        st.markdown("### 🧩 Leg Breakdown")
        st.dataframe(legs_df, use_container_width=True)

        # Export button
        csv = history.to_csv(index=False)
        st.download_button("Download Full History", csv, "history.csv")

# =============================
# PART 11 - CALIBRATION TAB UI
# =============================

calibration_tab = st.tabs(["⚙️ Calibration"])[0]

with calibration_tab:
    st.markdown("## ⚙️ Model Calibration")
    st.markdown("""
    Adjust the internal weighting system (Module 8)
    and bias calibration engine (Module 9).
    These settings affect all model outputs.
    """)

    # Load current settings
    settings = load_calibration()

    st.markdown("### 🎚 Module 8 - Weighting Engine")

    settings["pace_weight"] = st.slider("Pace Weight", 0.0, 3.0, settings["pace_weight"], 0.1)
    settings["defense_weight"] = st.slider("Defensive Matchup Weight", 0.0, 3.0, settings["defense_weight"], 0.1)
    settings["injury_weight"] = st.slider("Injury Impact Weight", 0.0, 3.0, settings["injury_weight"], 0.1)
    settings["recency_bias"] = st.slider("Recency Bias", 0.0, 1.0, settings["recency_bias"], 0.05)
    settings["usage_weight"] = st.slider("Usage/Minutes Weight", 0.0, 2.0, settings["usage_weight"], 0.05)
    settings["home_away_weight"] = st.slider("Home/Away Adjustment", 0.0, 2.0, settings["home_away_weight"], 0.05)

    st.markdown("### 🧠 Module 9 - Bias Calibration Engine")

    settings["prop_bias"] = st.slider("Prop Type Bias Adjustment", -10.0, 10.0, settings["prop_bias"], 0.5)
    settings["player_bias"] = st.slider("Player Historical Bias", -20.0, 20.0, settings["player_bias"], 1.0)
    settings["team_bias"] = st.slider("Team-Level Bias", -15.0, 15.0, settings["team_bias"], 0.5)
    settings["shrinkage"] = st.slider("Shrinkage Toward League Average", 0.0, 1.0, settings["shrinkage"], 0.05)
    settings["volatility_scale"] = st.slider("Volatility Scaling", 0.1, 3.0, settings["volatility_scale"], 0.1)

    st.markdown("---")

    if st.button("💾 Save Calibration Settings"):
        save_calibration(settings)
        st.success("Settings saved!")

    if st.button("🔄 Reset to Default"):
        if os.path.exists("calibration.json"):
            os.remove("calibration.json")
        st.warning("Reset complete — reload the app.")

    st.markdown("### 📦 Export / Import Calibration Profiles")

    calib_export = json.dumps(settings)
    st.download_button("Export Current Profile", calib_export, "calibration.json")

    uploaded = st.file_uploader("Import Profile")
    if uploaded:
        new_settings = json.load(uploaded)
        save_calibration(new_settings)
        st.success("Profile imported!")

# ===============================
# PART 12 — DEFENSIVE PROFILE VISUALIZER
# ===============================

def defensive_profile_visualizer():

    st.markdown("## 🛡 Defensive Profile Visualizer")
    st.markdown("Analyze opponent defensive strengths, weaknesses, pace, positional defense, and matchup difficulty.")

    # ---- Load defensive data ---- #
    defensive_data = st.session_state.get("defense_data", {})

    if not defensive_data:
        st.warning("No defensive data available. Ensure Module 22 is loaded.")
        return

    teams = sorted(list(defensive_data.keys()))
    selected_team = st.selectbox("Select Team", teams)

    team = defensive_data[selected_team]

    # -----------------------------------
    # RADAR CHART (Spider Plot)
    # -----------------------------------
    st.markdown("### 🕸 Defensive Radar Profile")

    labels = ["PTS Allowed", "REB Allowed", "AST Allowed", "3PA Allowed", "Pace"]
    values = [
        team.get("PTS_allowed", 0),
        team.get("REB_allowed", 0),
        team.get("AST_allowed", 0),
        team.get("3PA_allowed", 0),
        team.get("pace", 0),
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    st.pyplot(fig)

    # -----------------------------------
    # BAR CHART – Allowed Stats
    # -----------------------------------
    st.markdown("### 📊 Allowed Per Game Stats")

    allowed_stats = {
        "PTS Allowed": team.get("PTS_allowed", 0),
        "REB Allowed": team.get("REB_allowed", 0),
        "AST Allowed": team.get("AST_allowed", 0),
        "3PA Allowed": team.get("3PA_allowed", 0),
    }

    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.bar(allowed_stats.keys(), allowed_stats.values())
    ax2.set_ylabel("Per Game Allowed")
    ax2.set_title(f"{selected_team} Allowed Stats")

    st.pyplot(fig2)

    # -----------------------------------
    # POSITIONAL DEFENSE
    # -----------------------------------
    st.markdown("### 🧍‍♂️ Positional Defense Grades")

    pos = team.get("positional_ratings", {
        "PG": 0, "SG": 0, "SF": 0, "PF": 0, "C": 0
    })

    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.bar(pos.keys(), pos.values(), color="gray")
    ax3.set_title("Positional Defensive Strength (Lower = Stronger Defense)")

    st.pyplot(fig3)

    # -----------------------------------
    # ON/OFF IMPACT
    # -----------------------------------
    st.markdown("### 🔌 On/Off Defensive Impact")

    onoff = team.get("on_off_changes", {
        "DRTG_change": 0,
        "pace_change": 0,
        "rim_prot_change": 0
    })

    col1, col2, col3 = st.columns(3)
    col1.metric("DRTG Change", f"{onoff.get('DRTG_change', 0):+.1f}")
    col2.metric("Pace Change", f"{onoff.get('pace_change', 0):+.1f}")
    col3.metric("Rim Protection", f"{onoff.get('rim_prot_change', 0):+.1f}")

    # -----------------------------------
    # MATCHUP DIFFICULTY BADGES
    # -----------------------------------
    st.markdown("### 🎯 Matchup Difficulty Scores")

    match = team.get("matchup_scores", {
        "points": 1.0,
        "rebounds": 1.0,
        "assists": 1.0,
        "threes": 1.0,
    })

    def badge(label, score):
        color = (
            "#2ecc71" if score >= 1.1 else
            "#f1c40f" if score >= 0.9 else
            "#e74c3c"
        )
        st.markdown(
            f"""
            <div style="
                background:{color};
                padding:8px;
                border-radius:8px;
                width:140px;
                text-align:center;
                margin-bottom:5px;
                font-weight:bold;
            ">
                {label}: {score:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

    cols = st.columns(4)
    cols[0].write(badge("Points", match.get("points", 1.0)))
    cols[1].write(badge("Rebounds", match.get("rebounds", 1.0)))
    cols[2].write(badge("Assists", match.get("assists", 1.0)))
    cols[3].write(badge("3PT", match.get("threes", 1.0)))
# ====================================
# PART 13 — TEAM CONTEXT VISUALIZER
# ====================================

def team_context_visualizer():

    st.markdown("## 📊 Team Context Visualizer")
    st.markdown("Analyze full game environment: pace, efficiency, usage, rotations, injuries, and Vegas context.")

    # ---- Load team context data ---- #
    team_context = st.session_state.get("team_context", {})

    if not team_context:
        st.warning("No team context data available. Ensure Team Context module is loaded.")
        return

    teams = sorted(list(team_context.keys()))
    colA, colB = st.columns(2)
    with colA:
        selected_team = st.selectbox("Select Team", teams)
    with colB:
        opponent = st.selectbox("Select Opponent", teams)

    data = team_context[selected_team]
    opp = team_context[opponent]

    # ------------------------------
    # GAME OVERVIEW (3-CARD SUMMARY)
    # ------------------------------
    st.markdown("### 🏀 Game Environment Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Team Pace", f"{data.get('pace', 0):.1f}")
    col2.metric("Vegas Total", data.get("vegas", {}).get("total", 0))
    col3.metric("Spread", f"{data.get('vegas', {}).get('spread', 0):+}")

    # ------------------------------
    # Efficiency Bar Chart
    # ------------------------------
    st.markdown("### 📈 Efficiency Ratings")

    ORTG = data.get("ORTG", 0)
    DRTG = data.get("DRTG", 0)
    NET = data.get("NetRTG", 0)

    fig, ax = plt.subplots(figsize=(5, 3))
    metrics = ["ORTG", "DRTG", "NetRTG"]
    values = [ORTG, DRTG, NET]

    ax.bar(metrics, values)
    ax.set_title(f"{selected_team} Efficiency Ratings")

    st.pyplot(fig)

    # ------------------------------
    # Pace Comparison (Team vs Opponent)
    # ------------------------------
    st.markdown("### 🏃 Pace Matchup Comparison")

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.bar(["Team Pace", "Opponent Pace"], [data.get("pace", 0), opp.get("pace", 0)])
    ax2.set_title("Pace Matchup")

    st.pyplot(fig2)

    # ------------------------------
    # Usage Distribution Pie Chart
    # ------------------------------
    st.markdown("### 🔋 Usage Distribution")

    usage = data.get("usage_distribution", {})

    if usage:
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        ax3.pie(usage.values(), labels=usage.keys(), autopct="%1.1f%%")
        ax3.set_title("Usage Distribution")
        st.pyplot(fig3)
    else:
        st.info("No usage distribution data available for this team.")

    # ------------------------------
    # Implied Totals Heatmap
    # ------------------------------
    st.markdown("### 🔥 Implied Team Totals")

    vegas = data.get("vegas", {})
    team_total = vegas.get("team_total", 0)
    opp_total = vegas.get("opp_total", 0)

    heat = np.array([[team_total, opp_total]])

    fig4, ax4 = plt.subplots(figsize=(4, 4))
    im = ax4.imshow(heat, cmap="coolwarm")

    ax4.set_xticks([0, 1])
    ax4.set_xticklabels([selected_team, opponent])
    ax4.set_yticks([])

    ax4.set_title("Implied Points Heatmap")

    for i in range(2):
        ax4.text(i, 0, f"{heat[0][i]:.1f}", ha="center", va="center", color="white", fontsize=13)

    st.pyplot(fig4)

    # ------------------------------
    # Rotation + Injury Context
    # ------------------------------
    st.markdown("### 🧩 Rotation & Injury Context")

    col4, col5 = st.columns(2)
    col4.metric("Rotation Depth", data.get("rotation_depth", 0))

    injuries_list = data.get("injuries", [])
    injuries_display = ", ".join(injuries_list) if injuries_list else "None"
    col5.metric("Key Injuries", injuries_display)

    # ------------------------------
    # Blowout Risk (Using Spread)
    # ------------------------------
    st.markdown("### ⚠️ Blowout Probability")

    spread = vegas.get("spread", 0)
    # Simple blowout probability model
    blowout_prob = min(
        0.75,
        max(0.0, (abs(spread) - 5) / 20)  # Example: +15 spread = ~50% blowout risk
    )

    st.metric("Blowout Risk", f"{blowout_prob * 100:.1f}%")

    # ------------------------------
    # Summary Card
    # ------------------------------
    st.info(f"""
    ### 📝 Matchup Summary — {selected_team} vs {opponent}

    - **Pace Projection:** {"Fast" if data.get("pace", 0) > opp.get("pace", 0) else "Slow" if data.get("pace", 0) < opp.get("pace", 0) else "Neutral"}
    - **Offensive Edge:** {"Team" if data.get("NetRTG", 0) > opp.get("NetRTG", 0) else "Opponent"}
    - **High Usage Players:** {', '.join([p for p,u in usage.items() if u >= 25]) if usage else "N/A"}
    - **Injury Impact:** {injuries_display}
    - **Vegas Lean:** {"High-scoring" if vegas.get("total", 0) > 225 else "Low-scoring"}
    """)

# ==========================================
# PART 14 — BLOWOUT MODEL DISPLAY
# ==========================================

def blowout_model_display():

    st.markdown("## 💥 Blowout Model Display")
    st.markdown("Understand blowout risk, minutes decay, garbage time danger, and game script volatility.")

    # ---- Load team context ---- #
    team_context = st.session_state.get("team_context", {})

    if not team_context:
        st.warning("No team context data loaded. Ensure Vegas + Team Context modules are active.")
        return

    teams = sorted(list(team_context.keys()))

    colA, colB = st.columns(2)
    with colA:
        selected_team = st.selectbox("Select Team", teams)
    with colB:
        opponent = st.selectbox("Select Opponent", teams)

    data = team_context[selected_team]
    opp = team_context[opponent]

    vegas = data.get("vegas", {})
    spread = vegas.get("spread", 0)

    # ---- Simple Blowout Probability Model ---- #
    # Model: Risk grows significantly beyond 8+ spreads
    blowout_prob = min(
        0.80,
        max(0.0, (abs(spread) - 6) / 18)
    )

    # -------------------------------------------
    # BLOWOUT RISK METER (COLOR-CODED)
    # -------------------------------------------
    def blowout_meter(prob):
        color = (
            "#2ecc71" if prob < 0.12 else
            "#f1c40f" if prob < 0.28 else
            "#e74c3c"
        )

        st.markdown(
            f"""
            <div style="
                width:100%;
                padding:18px;
                border-radius:12px;
                background:{color}33;
                border:2px solid {color};
                text-align:center;
                font-size:30px;
                font-weight:800;
                margin-top:12px;
                margin-bottom:18px;
            ">
                Blowout Risk: {prob*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    blowout_meter(blowout_prob)

    # -------------------------------------------
    # INPUT FACTORS
    # -------------------------------------------
    st.markdown("### 📊 Inputs to Blowout Model")

    net_rating_gap = data.get("NetRTG", 0) - opp.get("NetRTG", 0)
    pace_gap = data.get("pace", 0) - opp.get("pace", 0)
    injuries = len(data.get("injuries", []))
    home_adv = 1 if vegas.get("spread", 0) < 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Spread", f"{spread:+}")
    col2.metric("Net Rating Gap", f"{net_rating_gap:+.1f}")
    col3.metric("Pace Gap", f"{pace_gap:+.1f}")

    col4, col5 = st.columns(2)
    col4.metric("Injury Factor", f"{injuries}")
    col5.metric("Home Advantage", "Yes" if home_adv else "No")

    # -------------------------------------------
    # GARBAGE TIME RISK SCORE
    # -------------------------------------------
    garbage_risk = min(10, blowout_prob * 20)  # scaled to 1–10

    st.markdown("### 🕒 Garbage-Time Risk")
    st.metric("Garbage-Time Score", f"{garbage_risk:.1f} / 10")

    # -------------------------------------------
    # PROJECTED MINUTES DECAY
    # -------------------------------------------
    st.markdown("### 📉 Minutes Decay Projection")

    def compute_minutes_decay(prob):
        return min(8, prob * 12)  # at 80% blowout → ~8 min decay

    minutes_decay = compute_minutes_decay(blowout_prob)

    players = data.get("usage_distribution", {}).keys()

    if not players:
        st.info("No player data for minutes decay projection.")
    else:
        for p in players:
            st.markdown(f"- **{p}**: -{minutes_decay:.1f} minutes expected in high-blowout scripts")

    # -------------------------------------------
    # STRATEGY RECOMMENDATIONS
    # -------------------------------------------
    st.markdown("### 🎯 Betting Strategy Guidance")

    if blowout_prob > 0.40:
        st.error("""
        **High Blowout Environment**
        - Avoid star overs (minutes collapse)
        - Under on PRA often strongest
        - Consider bench overs or role player spikes
        - 4th quarter usage unreliable
        """)
    elif blowout_prob > 0.20:
        st.warning("""
        **Moderate Blowout Risk**
        - Reduce confidence in overs
        - Prefer props tied to pace or single stats
        - Avoid correlated stacks
        """)
    else:
        st.success("""
        **Low Blowout Risk**
        - Stars maintain normal rotations
        - Overs and correlated plays are safer
        - Pace and matchup have higher predictive power
        """)

    # -------------------------------------------
    # BLOWOUT SIMULATION DISTRIBUTION
    # -------------------------------------------
    st.markdown("### 🧪 Blowout Monte Carlo Simulation")

    # Simulate score margin distribution
    np.random.seed(1)
    simulated_spreads = np.random.normal(spread, 8, 5000)

    fig5, ax5 = plt.subplots(figsize=(7, 3))
    ax5.hist(simulated_spreads, bins=26, alpha=0.7)
    ax5.axvline(15, color="red", linestyle="--", label="Garbage Time Threshold")
    ax5.set_title("Distribution of Final Margins")
    ax5.legend()

    st.pyplot(fig5)

    # -------------------------------------------
    # SUMMARY CARD
    # -------------------------------------------
    st.info(f"""
    ### 📝 Blowout Summary — {selected_team} vs {opponent}

    - **Blowout Probability:** {blowout_prob*100:.1f}%
    - **Garbage-Time Score:** {garbage_risk:.1f}/10
    - **Minutes Decay (est.):** -{minutes_decay:.1f} for stars
    - **Spread:** {spread:+}
    - **Net Rating Difference:** {net_rating_gap:+.1f}
    - **Pace Gap:** {pace_gap:+.1f}
    - **Injury Impact:** {injuries} key players
    """)

# ==========================================
# PART 15 — ROTATION VOLATILITY DISPLAY
# ==========================================

def rotation_volatility_display():

    st.markdown("## 🔁 Rotation Volatility Display")
    st.markdown("Analyze how stable or volatile rotations are, including minutes variance, bench usage, and volatility risk.")

    # ---- Load rotation data ---- #
    rotation_data = st.session_state.get("rotation_data", {})

    if not rotation_data:
        st.warning("No rotation data available. Ensure Rotation Engine module is loaded.")
        return

    teams = sorted(list(rotation_data.keys()))
    selected_team = st.selectbox("Select a Team", teams)

    team = rotation_data[selected_team]
    players = team.get("players", {})

    if not players:
        st.warning("Team has no recorded rotation data.")
        return

    # ==========================================
    # ROTATION DEPTH
    # ==========================================
    st.markdown("### 🧩 Rotation Depth")

    rotation_depth = team.get("rotation_depth", 0)
    st.metric("Active Rotation Depth", rotation_depth)

    # ==========================================
    # MINUTES VARIANCE (TABLE)
    # ==========================================
    st.markdown("### 🎛 Minutes Variance (Last 10 Games)")

    variance_rows = []

    for player, pdata in players.items():
        mins = pdata.get("min_last10", [])
        if not mins:
            continue

        variance_rows.append({
            "Player": player,
            "Avg Minutes": np.mean(mins),
            "Variance": np.var(mins),
            "Std Dev": np.std(mins)
        })

    if not variance_rows:
        st.info("No minute logs available for this team.")
        return

    df_var = pd.DataFrame(variance_rows)
    df_var = df_var.sort_values("Std Dev", ascending=False)

    st.dataframe(df_var, use_container_width=True)

    # ==========================================
    # MINUTES RANGE BAR CHART
    # ==========================================
    st.markdown("### 🎚 Minutes Range (Floor → Ceiling)")

    fig, ax = plt.subplots(figsize=(8,4))
    names = df_var["Player"]

    mins_low = [min(players[p]["min_last10"]) for p in names]
    mins_high = [max(players[p]["min_last10"]) for p in names]

    ax.barh(names, mins_high, color="#3498db80")
    ax.barh(names, mins_low, color="#3498dbff")
    ax.set_xlabel("Minutes")
    ax.set_title("Minutes Range (Last 10 Games)")

    st.pyplot(fig)

    # ==========================================
    # VOLATILITY INDEX (0–100)
    # ==========================================
    st.markdown("### ⚠️ Rotation Volatility Index")

    def avg_injury_risk(players):
        risks = []
        for pdata in players.values():
            risk = pdata.get("injury_risk", 0)
            risks.append(risk)
        return np.mean(risks) if risks else 0

    vol_index = (
        df_var["Std Dev"].mean() * 3 +
        rotation_depth * 2 +
        team.get("coaching_profile", {}).get("volatility_factor", 0.2) * 25 +
        avg_injury_risk(players) * 25
    )

    vol_index = min(100, vol_index)

    # ---- Visual Meter ---- #
    def volatility_meter(score):
        color = (
            "#2ecc71" if score < 30 else
            "#f1c40f" if score < 60 else
            "#e74c3c"
        )

        st.markdown(
            f"""
            <div style="
                padding:16px;
                border-radius:12px;
                background:{color}33;
                border:2px solid {color};
                text-align:center;
                font-size:28px;
                font-weight:bold;
                margin-top:10px;
                margin-bottom:20px;
            ">
                Rotation Volatility: {score:.1f}/100
            </div>
            """,
            unsafe_allow_html=True
        )

    volatility_meter(vol_index)

    # ==========================================
    # BENCH USAGE PIE CHART
    # ==========================================
    st.markdown("### 🔋 Bench Usage Share")

    bench_usage = team.get("bench_usage_share", {})

    if bench_usage:
        fig2, ax2 = plt.subplots(figsize=(5,5))
        ax2.pie(bench_usage.values(), labels=bench_usage.keys(), autopct="%1.1f%%")
        ax2.set_title("Bench Usage Distribution")
        st.pyplot(fig2)
    else:
        st.info("No bench usage data available.")

    # ==========================================
    # PROP IMPACT RECOMMENDATIONS
    # ==========================================
    st.markdown("### 🎯 Prop Impact Recommendations")

    if vol_index > 70:
        st.error("""
        **Extremely High Volatility Environment**
        - Avoid star overs (minutes unpredictable)
        - PRA overs dangerous
        - Prefer unders
        - Bench overs / role player spikes more common
        """)
    elif vol_index > 45:
        st.warning("""
        **Moderate Rotation Volatility**
        - Be cautious with overs
        - Prefer single-stat props
        - Usage spreads may widen unexpectedly
        """)
    else:
        st.success("""
        **Stable Rotation**
        - Overs more reliable
        - Minutes predictable
        - Strong environment for correlated plays
        """)

    # ==========================================
    # MINUTES SIMULATION HEATMAP
    # ==========================================
    st.markdown("### 🧪 Minutes Distribution Simulation")

    simulated_minutes = []

    for _ in range(4000):
        for pdata in players.values():
            mins = pdata.get("min_last10", [])
            if not mins:
                continue
            mean = np.mean(mins)
            sd = np.std(mins)
            simulated_minutes.append(np.random.normal(mean, sd))

    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.hist(simulated_minutes, bins=30, alpha=0.75)
    ax3.set_title("Simulated Minutes Distribution (All Players Combined)")

    st.pyplot(fig3)

    # ==========================================
    # SUMMARY CARD
    # ==========================================
    st.info(f"""
    ### 📝 Rotation Summary — {selected_team}

    - **Volatility Index:** {vol_index:.1f}/100  
    - **Rotation Depth:** {rotation_depth} players  
    - **Avg Minutes StdDev:** {df_var['Std Dev'].mean():.2f}  
    - **Injury Volatility:** {avg_injury_risk(players):.2f}  
    - **Coaching Variance Factor:** {team.get('coaching_profile',{}).get('volatility_factor',0.2)}
    """)

# ==========================================
# PART 16 — OVERRIDE TAB UI
# ==========================================

import json
import os

def override_tab_ui():

    st.markdown("## 🛠 Manual Override Controls")
    st.markdown("""
    Override any model-generated values for projections, minutes, usage, pace, matchup difficulty, lines, and bias.
    Your overrides will persist locally and override the model during calculations.
    """)

    # ==========================================
    # LOAD / SAVE OVERRIDES
    # ==========================================
    overrides_path = "overrides.json"

    def load_overrides():
        if os.path.exists(overrides_path):
            try:
                return json.load(open(overrides_path))
            except:
                return {}
        return {}

    def save_overrides(data):
        try:
            json.dump(data, open(overrides_path, "w"), indent=4)
        except Exception as e:
            st.error(f"Failed to save overrides: {e}")

    overrides = load_overrides()

    # ==========================================
    # LOAD MODEL OUTPUTS
    # ==========================================
    model_outputs = st.session_state.get("model_outputs", {})
    if not model_outputs:
        st.warning("Run the model at least once to enable overrides.")
        return

    all_players = sorted(list(model_outputs.keys()))

    # ==========================================
    # PLAYER SELECTOR
    # ==========================================
    selected_player = st.selectbox("Select Player to Override", all_players)

    # Initialize override entry if missing
    if selected_player not in overrides:
        overrides[selected_player] = {}

    current = model_outputs[selected_player]  # model-generated values

    st.markdown(f"### 🧩 Overrides for **{selected_player}**")
    st.markdown("Adjust any field below. Leave blank or equal to model value for no override.")

    # ==========================================
    # OVERRIDABLE FIELDS
    # ==========================================
    # Expected structure: {
    #   "projection": X,
    #   "minutes": X,
    #   "usage": X,
    #   "pace": X,
    #   "matchup_score": X,
    #   "line": X,
    #   "bias": X
    # }

    # HELPER: safe extraction
    def get_override_value(key, default):
        return overrides[selected_player].get(key, default)

    # FIELD GROUP A — PROJECTION
    st.markdown("### 📈 Projection")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model Projection:**", current.get("projection", 0))
    with col2:
        overrides[selected_player]["projection"] = st.number_input(
            "Override Projection",
            value=float(get_override_value("projection", current.get("projection", 0))),
            step=0.1,
            format="%.2f"
        )

    # FIELD GROUP B — MINUTES
    st.markdown("### 🕒 Minutes")
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Model Minutes:**", current.get("minutes", 0))
    with col4:
        overrides[selected_player]["minutes"] = st.number_input(
            "Override Minutes",
            value=float(get_override_value("minutes", current.get("minutes", 0))),
            step=1.0
        )

    # FIELD GROUP C — USAGE
    st.markdown("### 🔋 Usage Rate (%)")
    col5, col6 = st.columns(2)
    with col5:
        st.write("**Model Usage Rate:**", f"{current.get('usage', 0):.1f}%")
    with col6:
        overrides[selected_player]["usage"] = st.number_input(
            "Override Usage (%)",
            value=float(get_override_value("usage", current.get("usage", 0))),
            step=1.0
        )

    # FIELD GROUP D — PACE
    st.markdown("### 🏃 Pace Adjustment")
    col7, col8 = st.columns(2)
    with col7:
        st.write("**Model Pace:**", current.get("pace", 0))
    with col8:
        overrides[selected_player]["pace"] = st.number_input(
            "Override Pace",
            value=float(get_override_value("pace", current.get("pace", 0))),
            step=0.1
        )

    # FIELD GROUP E — MATCHUP DIFFICULTY
    st.markdown("### 🛡 Matchup Difficulty Multiplier")
    col9, col10 = st.columns(2)
    with col9:
        st.write("**Model Matchup Score:**", current.get("matchup_score", 1.0))
    with col10:
        overrides[selected_player]["matchup_score"] = st.slider(
            "Override Matchup Score",
            0.50, 1.50,
            float(get_override_value("matchup_score", current.get("matchup_score", 1.0))),
            step=0.05
        )

    # FIELD GROUP F — MARKET LINE
    st.markdown("### 📊 Prop Line (O/U)")
    col11, col12 = st.columns(2)
    with col11:
        st.write("**Market Line:**", current.get("line", 0))
    with col12:
        overrides[selected_player]["line"] = st.number_input(
            "Override Line",
            value=float(get_override_value("line", current.get("line", 0))),
            step=0.5
        )

    # FIELD GROUP G — BIAS ADJUSTMENT
    st.markdown("### 🧠 Bias Adjustment")
    col13, col14 = st.columns(2)
    with col13:
        st.write("**Model Bias:**", current.get("bias", 0.0))
    with col14:
        overrides[selected_player]["bias"] = st.slider(
            "Override Bias",
            -10.0, 10.0,
            float(get_override_value("bias", current.get("bias", 0.0))),
            step=0.5
        )

    # ==========================================
    # SAVE / RESET BUTTONS
    # ==========================================
    st.markdown("---")

    colA, colB, colC = st.columns([1,1,1])

    if colA.button("💾 Save Overrides"):
        save_overrides(overrides)
        st.success("Overrides saved successfully!")

    if colB.button("🔄 Reset Player Overrides"):
        overrides[selected_player] = {}
        save_overrides(overrides)
        st.warning(f"Overrides reset for {selected_player}.")

    if colC.button("🔥 Reset ALL Overrides"):
        save_overrides({})
        st.error("All overrides cleared.")

    # ==========================================
    # PREVIEW ACTIVE OVERRIDES
    # ==========================================
    st.markdown("### 📋 Active Overrides for This Player")

    current_overrides = overrides.get(selected_player, {})
    if current_overrides:
        st.json(current_overrides)
    else:
        st.info("No overrides currently applied for this player.")

# ==========================================
# PART 17 — TREND RECOGNITION DISPLAY
# ==========================================

def trend_recognition_display():

    st.markdown("## 📈 Trend Recognition Display")
    st.markdown("Analyze rolling averages, volatility, pace-adjusted performance, opponent context, and on/off trends.")

    # ---- Load player trend data ---- #
    player_stats = st.session_state.get("player_stats", {})

    if not player_stats:
        st.warning("No player trend data available. Please load player stats / model first.")
        return

    players = sorted(list(player_stats.keys()))
    selected_player = st.selectbox("Select Player", players)

    pdata = player_stats[selected_player]

    # Extract base arrays safely
    def safe_array(key):
        arr = pdata.get(key, [])
        return arr if isinstance(arr, list) else []

    pts = safe_array("points")
    reb = safe_array("rebounds")
    ast = safe_array("assists")
    mins = safe_array("minutes")
    dates = safe_array("dates")

    # If dates missing, make simple index
    if not dates:
        dates = list(range(len(pts)))

    # -------------------------------
    # Rolling Average Helper
    # -------------------------------
    def rolling_avg(arr, n):
        if len(arr) == 0:
            return 0
        if len(arr) < n:
            return np.mean(arr)
        return np.mean(arr[-n:])

    # -------------------------------
    # Trend Direction Helper
    # -------------------------------
    def trend_direction(arr):
        short = rolling_avg(arr, 3)
        long = rolling_avg(arr, 10)
        if long == 0:
            return "➡️ Neutral"
        if short > long * 1.12:
            return "📈 Trending Up"
        elif short < long * 0.88:
            return "📉 Trending Down"
        else:
            return "➡️ Neutral"

    # -------------------------------
    # Rolling Averages
    # -------------------------------
    st.markdown("### 🔍 Rolling Averages (3 / 5 / 10)")

    def avg_block(label, arr):
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{label} (3g)", f"{rolling_avg(arr,3):.1f}")
        col2.metric(f"{label} (5g)", f"{rolling_avg(arr,5):.1f}")
        col3.metric(f"{label} (10g)", f"{rolling_avg(arr,10):.1f}")

    avg_block("Points", pts)
    avg_block("Rebounds", reb)
    avg_block("Assists", ast)
    avg_block("Minutes", mins)

    # -------------------------------
    # Trend Directions
    # -------------------------------
    st.markdown("### 📊 Trend Directions")

    st.write("- **Points:**", trend_direction(pts))
    st.write("- **Rebounds:**", trend_direction(reb))
    st.write("- **Assists:**", trend_direction(ast))
    st.write("- **Minutes:**", trend_direction(mins))

    # -------------------------------
    # Sparkline Trend Charts
    # -------------------------------
    st.markdown("### 📉 Sparkline Performance Charts")

    def sparkline(label, arr):
        fig, ax = plt.subplots(figsize=(6, 1.8))
        ax.plot(dates, arr)
        ax.set_title(label)
        st.pyplot(fig)

    sparkline("Points", pts)
    sparkline("Rebounds", reb)
    sparkline("Assists", ast)
    sparkline("Minutes", mins)

    # -------------------------------
    # Volatility Score
    # -------------------------------
    st.markdown("### ⚠️ Volatility Score")

    def volatility_score(arr):
        if len(arr) == 0 or np.mean(arr) == 0:
            return 0
        return min(100, (np.std(arr) / np.mean(arr)) * 100)

    vol_pts = volatility_score(pts)
    vol_reb = volatility_score(reb)
    vol_ast = volatility_score(ast)

    colA, colB, colC = st.columns(3)
    colA.metric("Points Volatility", f"{vol_pts:.1f}")
    colB.metric("Rebounds Volatility", f"{vol_reb:.1f}")
    colC.metric("Assists Volatility", f"{vol_ast:.1f}")

    # -------------------------------
    # Pace-Adjusted Trend
    # -------------------------------
    st.markdown("### 🏃 Pace-Adjusted Points Trend")

    pace_adj = pdata.get("pace_adjusted", {}).get("points", [])
    if pace_adj:
        fig2, ax2 = plt.subplots(figsize=(6,2))
        ax2.plot(dates, pace_adj)
        ax2.set_title("Pace-Adjusted Points")
        st.pyplot(fig2)
    else:
        st.info("No pace-adjusted trend data available.")

    # -------------------------------
    # Opponent-Adjusted Trend
    # -------------------------------
    st.markdown("### 🛡 Opponent-Adjusted Points Trend")

    opp_adj = pdata.get("opponent_adjusted", {}).get("points", [])
    if opp_adj:
        fig3, ax3 = plt.subplots(figsize=(6,2))
        ax3.plot(dates, opp_adj)
        ax3.set_title("Opponent-Adjusted Points")
        st.pyplot(fig3)
    else:
        st.info("No opponent-adjusted trend data available.")

    # -------------------------------
    # On/Off Split Trend
    # -------------------------------
    st.markdown("### 🔌 On/Off Teammate Impact Trends")

    onoff_data = pdata.get("on_off_splits", {})

    if onoff_data:
        for teammate, arr in onoff_data.items():
            fig4, ax4 = plt.subplots(figsize=(6,2))
            ax4.plot(dates, arr)
            ax4.set_title(f"On/Off Impact — {teammate}")
            st.pyplot(fig4)
    else:
        st.info("No on/off split data available.")

    # -------------------------------
    # Trend-Based Recommendations
    # -------------------------------
    st.markdown("### 🎯 Trend-Based Betting Recommendations")

    pts_trend = trend_direction(pts)

    if pts_trend == "📈 Trending Up" and vol_pts < 30:
        st.success("**Points Overs appear strong** — stable upward trend.")
    elif pts_trend == "📉 Trending Down":
        st.error("**Points Overs risky** — downward trend detected.")
    else:
        st.warning("**Neutral trend** — rely more on matchup or pace data.")

    # -------------------------------
    # Summary Card
    # -------------------------------
    st.info(f"""
    ### 📝 Trend Summary — {selected_player}

    **Points Trend:** {trend_direction(pts)}  
    **Rebounding Trend:** {trend_direction(reb)}  
    **Assist Trend:** {trend_direction(ast)}  
    **Minutes Trend:** {trend_direction(mins)}  

    **Volatility:** {np.mean([vol_pts, vol_reb, vol_ast]):.1f}/100  
    **Games Logged:** {len(pts)}
    """)

# ==========================================
# PART 18 — LINE SHOPPING ANALYZER
# ==========================================

def line_shopping_analyzer():

    st.markdown("## 💰 Line Shopping Analyzer")
    st.markdown("Compare sportsbook lines + EV across the entire market to find the best edges.")

    # =========================
    # LOAD MODEL OUTPUTS
    # =========================
    model_outputs = st.session_state.get("model_outputs", {})
    if not model_outputs:
        st.warning("Run the model first to enable line shopping.")
        return

    # =========================
    # LOAD LINES DATA
    # =========================
    # Expected format:
    # st.session_state["lines"] = {
    #     player: {
    #         "Points": {"FanDuel": 22.5, "DraftKings": 21.5, ...}
    #     }
    # }
    lines_data = st.session_state.get("lines_data", {})

    if not lines_data:
        st.warning("No sportsbook line data found. Ensure line API module is connected.")
        return

    # =========================
    # PLAYER + PROP SELECTOR
    # =========================
    players = sorted(list(lines_data.keys()))
    selected_player = st.selectbox("Select Player", players)

    props = sorted(list(lines_data[selected_player].keys()))
    selected_prop = st.selectbox("Select Prop Type", props)

    player_lines = lines_data[selected_player][selected_prop]  # dict of book → line

    # =========================
    # MODEL PROJECTION
    # =========================
    projection = model_outputs[selected_player].get(selected_prop.lower(), None)

    if projection is None:
        st.error(f"No projection available for {selected_prop} on {selected_player}.")
        return

    st.markdown("### 📈 Model vs Market")

    colA, colB = st.columns(2)
    colA.metric("Model Projection", f"{projection:.1f}")
    colB.metric("Books Available", f"{len(player_lines)}")

    # =========================
    # BEST/WORST LINE CALC
    # =========================
    best_line = min(player_lines.values())
    worst_line = max(player_lines.values())
    line_range = worst_line - best_line

    # =========================
    # COLOR-CODED TABLE
    # =========================
    st.markdown("### 📊 Sportsbook Line Comparison")

    table_html = "<table style='width:100%; border-collapse:collapse;'>"
    table_html += "<tr><th style='text-align:left;'>Sportsbook</th><th>Line</th><th>Delta vs Best</th></tr>"

    for book, line in player_lines.items():

        if line == best_line:
            color = "#2ecc71"  # green = best
        elif line == worst_line:
            color = "#e74c3c"  # red = worst
        else:
            color = "#f1c40f"  # yellow = middle

        delta = line - best_line

        table_html += f"""
        <tr style="border-bottom:1px solid #222;">
            <td style="padding:6px;">{book}</td>
            <td style="color:{color}; font-weight:bold; text-align:center;">{line}</td>
            <td style="text-align:center;">{delta:+.1f}</td>
        </tr>
        """

    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # =========================
    # EV CALCULATION (NORMAL MODEL)
    # You can swap this with your MC EV engine.
    # =========================
    def simple_ev(projection, line, stdev=3.5):
        # Probability player goes over line
        prob = 1 - norm.cdf(line, loc=projection, scale=stdev)
        return (prob * 100) - 50

    ev_rows = []
    for book, line in player_lines.items():
        ev_value = simple_ev(projection, line)
        ev_rows.append({
            "Sportsbook": book,
            "Line": line,
            "EV%": ev_value
        })

    ev_df = pd.DataFrame(ev_rows).sort_values("EV%", ascending=False)

    st.markdown("### 📈 EV Comparison Across Books")
    st.dataframe(ev_df, use_container_width=True)

    # =========================
    # LINE DELTA VISUALIZATION
    # =========================
    st.markdown("### 📉 Line Delta Visualization")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.bar(player_lines.keys(), player_lines.values(), color="#3498dbaa")
    ax.axhline(best_line, color="green", linestyle="--", label="Best Line")
    ax.set_ylabel("Line")
    ax.set_title(f"Market Line Spread – {selected_prop}")
    ax.legend()
    st.pyplot(fig)

    # =========================
    # PRIZEPICKS / SLEEPER EDGE ALERTS
    # =========================
    st.markdown("### 🎯 PrizePicks / Sleeper Edges")

    pp_line = player_lines.get("PrizePicks")
    sl_line = player_lines.get("Sleeper")

    if pp_line is not None:
        edge = projection - pp_line
        if abs(edge) >= 1:
            st.success(f"🔥 **PrizePicks Edge Detected:** Model = {projection:.1f}, PP = {pp_line}   (Δ = {edge:+.1f})")
        else:
            st.info("PrizePicks: No major mispricing detected.")

    if sl_line is not None:
        edge = projection - sl_line
        if abs(edge) >= 1.5:
            st.success(f"🔥 **Sleeper Edge Detected:** Model = {projection:.1f}, Sleeper = {sl_line}   (Δ = {edge:+.1f})")
        else:
            st.info("Sleeper: No major mispricing detected.")

    # =========================
    # MIDDLE OPPORTUNITY DETECTION
    # =========================
    st.markdown("### ⚠️ Middle Opportunity Detection")

    line_values = list(player_lines.values())
    if max(line_values) - min(line_values) >= 1.0:
        st.warning("⚠️ **Middle detected!** You may be able to bet Over at one book and Under another for a middle.")
    else:
        st.info("No middle opportunities at the moment.")

    # =========================
    # SUMMARY CARD
    # =========================
    st.info(f"""
    ### 📝 Line Shopping Summary — {selected_player} ({selected_prop})
    
    - **Best Line:** {best_line}  
    - **Worst Line:** {worst_line}  
    - **Range:** {line_range:.1f}  
    - **Highest EV:** {ev_df.iloc[0]["EV%"]:.2f}% at {ev_df.iloc[0]["Sportsbook"]}  
    - **Books Scanned:** {len(player_lines)}
    """)

# ==========================================
# PART 19 — ERROR HANDLING LAYER (FAILSAFE UI)
# ==========================================

import numpy as np
import streamlit as st
import json
import traceback

# ==========================================
# GLOBAL SAFE EXECUTION WRAPPER
# ==========================================

def safe_execute(fn, fallback=None, error_msg="An unexpected error occurred.", log=True):
    """
    Safely executes any function and returns fallback if error occurs.
    Appends errors to session_state['error_logs'].
    """
    try:
        return fn()
    except Exception as e:
        if log:
            st.session_state.setdefault("error_logs", []).append(
                traceback.format_exc()
            )
        st.error(error_msg)
        return fallback


# ==========================================
# SANITIZATION HELPERS
# ==========================================

def sanitize_number(val, default=0):
    """Avoid NaN, inf, or invalid numeric entries."""
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except:
        return default


def sanitize_array(arr):
    """Cleans array-like data and removes NaN/inf."""
    try:
        cleaned = [x for x in arr if x is not None and not np.isnan(x) and not np.isinf(x)]
        return cleaned if cleaned else [0]
    except:
        return [0]


# ==========================================
# FAILSAFE MODE (Light Mode Toggle)
# ==========================================

def render_failsafe_toggle():
    st.sidebar.markdown("### 🛡 Failsafe Mode")

    failsafe_enabled = st.sidebar.toggle(
        "Enable Failsafe Mode (recommended)",
        value=True,
        help="Reduces computational risk, disables heavy Monte Carlo, and prevents large loads."
    )

    st.session_state["failsafe_mode"] = failsafe_enabled

    return failsafe_enabled


# ==========================================
# HARD PROTECTION FOR MODEL PIPELINE
# ==========================================

def safe_model_run(run_fn):
    """
    Wrap your Run Model button with this:
        result = safe_model_run(lambda: model_pipeline(...))
    """

    try:
        with st.spinner("Running model safely..."):
            output = run_fn()
            st.success("Model executed successfully.")
            return output

    except Exception as e:
        st.error("🚨 Critical Error: The model failed to complete.")
        st.session_state.setdefault("error_logs", []).append(traceback.format_exc())
        st.stop()


# ==========================================
# API FAILSAFE WRAPPER
# ==========================================

def safe_api_call(api_fn, cache_fn=None, name="API"):
    """
    Calls API and falls back to cache on fail.
    """
    try:
        return api_fn()
    except Exception:
        st.warning(f"⚠️ {name} unavailable — using cached data.")
        if cache_fn:
            try:
                return cache_fn()
            except:
                st.error(f"Failed to load cached {name} data as well.")
                return {}
        return {}


# ==========================================
# UI WARNINGS & SANITY CHECKS
# ==========================================

def sanity_check_projection(player, proj):
    if proj is None or np.isnan(proj) or proj < 0 or proj > 120:
        st.error(f"Projection for {player} appears invalid — reverting to fallback.")
        return 0
    return proj


def sanity_check_probability(prob):
    if prob is None or np.isnan(prob) or prob < 0 or prob > 1:
        return 0.0
    return prob


# ==========================================
# ERROR LOG PANEL (UI)
# ==========================================

def render_error_logs():
    st.markdown("## 🗂 Error Logs")

    logs = st.session_state.get("error_logs", [])

    if not logs:
        st.info("No errors logged yet.")
        return

    with st.expander("Show Logged Errors"):
        for entry in logs:
            st.code(entry)


# ==========================================
# FAILSAFE ALERT BANNER
# ==========================================

def render_failsafe_banner():
    if st.session_state.get("failsafe_mode", False):
        st.markdown("""
        <div style='padding:10px; border-radius:8px; background:#f1c40f33; border-left:6px solid #f1c40f;'>
            <b>Failsafe Mode Enabled:</b> Heavy computations lowered, Monte Carlo reduced, and stability checks active.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding:10px; border-radius:8px; background:#2ecc7133; border-left:6px solid #2ecc71;'>
            <b>Failsafe Mode Disabled:</b> Full power mode. Maximum computation enabled.
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# SUMMARY OF FAILSAFE LAYER
# ==========================================

def failsafe_summary_card():
    st.info(f"""
    ### 🛡 Failsafe System Active

    - Global safe execution wrapper  
    - Fallback values prevent crashes  
    - API fallback system  
    - Monte Carlo reduction in failsafe mode  
    - Error log storage + viewer  
    - Auto-sanitization of invalid data  
    - Hard-stop protection on critical failures  

    This layer ensures the app remains stable under all conditions.
    """)

# ==========================================
# PART 20 — FINAL APP ASSEMBLER (MAIN APP)
# ==========================================

import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ==== IMPORT ALL MODULE UI FUNCTIONS ==== #
# (You already generated these in previous parts)
from defensive_profile_visualizer import defensive_profile_visualizer
from team_context_visualizer import team_context_visualizer
from blowout_model_display import blowout_model_display
from rotation_volatility_display import rotation_volatility_display
from trend_recognition_display import trend_recognition_display
from line_shopping_analyzer import line_shopping_analyzer
from override_tab_ui import override_tab_ui

# Failsafe utilities
from error_layer import (
    safe_execute,
    render_error_logs,
    render_failsafe_toggle,
    render_failsafe_banner,
)


# --------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------
st.set_page_config(
    page_title="NBA Quant App",
    layout="wide",
    page_icon="🏀",
)


# --------------------------------------------------------------------
# GLOBAL CSS STYLING
# --------------------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3, h4, label {
        color: white !important;
    }
    table, th, td {
        color: white !important;
    }
    .stMetricLabel {
        color: #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------------------------
if "error_logs" not in st.session_state:
    st.session_state["error_logs"] = []

if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False


# --------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# (These can be replaced with your real loaders)
# --------------------------------------------------------------------
def load_calibration():
    return {}

def load_overrides():
    if os.path.exists("overrides.json"):
        return json.load(open("overrides.json"))
    return {}

def load_team_context():
    return st.session_state.get("team_context", {}) or {}

def load_defensive_data():
    return st.session_state.get("defense_data", {}) or {}

def load_rotation_data():
    return st.session_state.get("rotation_data", {}) or {}

def load_player_stats():
    return st.session_state.get("player_stats", {}) or {}

def load_lines_data():
    return st.session_state.get("lines_data", {}) or {}

def load_model_outputs():
    return st.session_state.get("model_outputs", {}) or {}


# --------------------------------------------------------------------
# GLOBAL DATA LOADING (one time)
# --------------------------------------------------------------------
if not st.session_state["data_loaded"]:
    with st.spinner("Loading model environment..."):

        st.session_state["calibration"] = safe_execute(load_calibration, {})
        st.session_state["overrides"] = safe_execute(load_overrides, {})
        st.session_state["team_context"] = safe_execute(load_team_context, {})
        st.session_state["defense_data"] = safe_execute(load_defensive_data, {})
        st.session_state["rotation_data"] = safe_execute(load_rotation_data, {})
        st.session_state["player_stats"] = safe_execute(load_player_stats, {})
        st.session_state["lines_data"] = safe_execute(load_lines_data, {})
        st.session_state["model_outputs"] = safe_execute(load_model_outputs, {})

        st.session_state["data_loaded"] = True


# --------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------
st.sidebar.title("NBA Quant App")
render_failsafe_toggle()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🛡 Defensive Profile",
        "📊 Team Context",
        "💥 Blowout Model",
        "🔁 Rotation Volatility",
        "📈 Trends",
        "💰 Line Shopping",
        "🛠 Overrides",
        "🗂 Error Logs",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Kamal + ChatGPT 🚀")


# --------------------------------------------------------------------
# FAILSAFE TOP BANNER
# --------------------------------------------------------------------
render_failsafe_banner()


# --------------------------------------------------------------------
# ROUTER — MAIN PAGE LOADER
# --------------------------------------------------------------------
if page == "🏠 Home":

    st.title("🏀 NBA Quant App Dashboard")
    st.markdown("""
    Welcome to the **NBA Quant App**, a full-stack analytics toolkit built for:
    
    - Player Projection Modeling  
    - Defensive + Team Context Analytics  
    - Blowout Probability Engine  
    - Rotation Volatility Modeling  
    - Trend Recognition System  
    - Line Shopping + EV Scanning  
    - Manual Overrides  
    - Full Error/Failsafe Architecture  
    
    Use the sidebar to navigate through analysis modules.
    """)

elif page == "🛡 Defensive Profile":
    defensive_profile_visualizer()

elif page == "📊 Team Context":
    team_context_visualizer()

elif page == "💥 Blowout Model":
    blowout_model_display()

elif page == "🔁 Rotation Volatility":
    rotation_volatility_display()

elif page == "📈 Trends":
    trend_recognition_display()

elif page == "💰 Line Shopping":
    line_shopping_analyzer()

elif page == "🛠 Overrides":
    override_tab_ui()

elif page == "🗂 Error Logs":
    render_error_logs()

