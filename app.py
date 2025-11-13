# =========================================================
#  PART 1 — IMPORTS & GLOBAL CONFIGURATION
# =========================================================

import os, time, random, difflib
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

def current_season():
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
#  PART 2.4 — MARKET BASELINE LIBRARY (Option A1)
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
#  PART 3 — PLAYER GAME LOG ENGINE & PROJECTION MODEL
# =========================================================

@st.cache_data(show_spinner=False, ttl=900)
def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    """
    Pulls recent player logs, computes:
      - per-minute production (mu_per_min)
      - per-minute standard deviation (sd_per_min)
      - average minutes
      - team abbreviation
    """
    pid, label = resolve_player(name)
    if not pid:
        return None, None, None, None, f"No match for '{name}'."

    # Try requesting game logs
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

    # Team abbreviation
    team = None
    try:
        team = gl["TEAM_ABBREVIATION"].mode().iloc[0]
    except:
        team = None

    return mu_per_min, sd_per_min, avg_min, team, f"{label}: {len(per_min_vals)} games • {avg_min:.1f} min"

# =====================================================
# SKEW-NORMAL PROBABILITY (Final version)
# =====================================================

def skew_normal_prob(mu, sd, skew, line):
    """Right-tailed skew-normal probability."""
    try:
        z = (line - mu) / sd
        base = 1 - norm.cdf(z)

        # Apply skew factor (heavier tail → increases p_over)
        adj = base * (1 + 0.15 * (skew - 1))
        return float(np.clip(adj, 0.01, 0.99))
    except:
        return float(np.clip(base, 0.01, 0.99))


# =====================================================
# HYBRID ENGINE
# =====================================================

def hybrid_prob_over(line, mu, sd, market):
    """
    Stable hybrid distribution:
    - Normal core
    - Log-normal right tail for skew (but guarded)
    - Market weighting
    """
    normal_p = 1 - norm.cdf(line, mu, sd)

    # Guard for invalid parameters
    if mu <= 0 or sd <= 0 or np.isnan(mu) or np.isnan(sd):
        return float(np.clip(normal_p, 0.01, 0.99))

    # ---------- LOGNORMAL BLOCK ----------
    try:
        variance = sd ** 2
        phi = np.sqrt(variance + mu ** 2)

        mu_log = np.log(mu ** 2 / phi)
        sd_log = np.sqrt(np.log(phi ** 2 / mu ** 2))

        # check validity
        if np.isnan(mu_log) or np.isnan(sd_log) or sd_log <= 0:
            lognorm_p = normal_p
        else:
            lognorm_p = 1 - norm.cdf(np.log(line + 1e-9), mu_log, sd_log)

    except:
        lognorm_p = normal_p

    # ---------- MARKET WEIGHTS ----------
    w = {
        "PRA": 0.70,
        "Points": 0.55,
        "Rebounds": 0.40,
        "Assists": 0.30
    }.get(market, 0.50)

    hybrid = w * lognorm_p + (1 - w) * normal_p

    return float(np.clip(hybrid, 0.02, 0.98))

# ======================================================
# ADVANCED PLAYER CORRELATION ENGINE (Upgrade 4 — Part 6)
# ======================================================
def estimate_player_correlation(leg1, leg2):
    """
    Produces a dynamic, data-driven correlation estimate.

    Factors used:
    - Shared team → strongly increases correlation
    - Shared minutes → synergy boost
    - Points vs Assists → negative correlation
    - Rebounds vs Points → mild negative
    - PRA → slightly positive
    - Opponent context → affects both legs together
    """
    corr = 0.0

    # -----------------------
    # 1. Same-team baseline
    # -----------------------
    if leg1["team"] == leg2["team"] and leg1["team"] is not None:
        corr += 0.18

    # -----------------------
    # 2. Minutes dependency
    # -----------------------
    try:
        avg_min1 = leg1["mu"] / max(leg1["mu"]/leg1["line"], 1e-6)
        avg_min2 = leg2["mu"] / max(leg2["mu"]/leg2["line"], 1e-6)
    except:
        avg_min1 = avg_min2 = 28

    if avg_min1 > 30 and avg_min2 > 30:
        corr += 0.05
    elif avg_min1 < 22 or avg_min2 < 22:
        corr -= 0.04

    # -----------------------
    # 3. Market-type interactions
    # -----------------------
    m1, m2 = leg1["market"], leg2["market"]

    if m1 == "Points" and m2 == "Points":
        corr += 0.08

    if (m1 == "Points" and m2 == "Assists") or (m1 == "Assists" and m2 == "Points"):
        corr -= 0.10

    if (m1 == "Rebounds" and m2 == "Points") or (m1 == "Points" and m2 == "Rebounds"):
        corr -= 0.06

    if m1 == "PRA" or m2 == "PRA":
        corr += 0.03

    # -----------------------
    # 4. Opponent-defense adjustment
    # -----------------------
    ctx1, ctx2 = leg1["ctx_mult"], leg2["ctx_mult"]

    if ctx1 > 1.03 and ctx2 > 1.03:
        corr += 0.04
    if ctx1 < 0.97 and ctx2 < 0.97:
        corr += 0.03
    if (ctx1 > 1.03 and ctx2 < 0.97) or (ctx1 < 0.97 and ctx2 > 1.03):
        corr -= 0.05

    # -----------------------
    # 5. Clamp for stability
    # -----------------------
    corr = float(np.clip(corr, -0.25, 0.40))
    return corr




def compute_leg_projection(player, market, line, opp, teammate_out, blowout, n_games):
    # --------------------------------------------------------
    # GET PLAYER RATES
    # --------------------------------------------------------
    mu_min, sd_min, avg_min, team, msg = get_player_rate_and_minutes(player, n_games, market)
    if mu_min is None:
        return None, msg

    # --------------------------------------------------------
    # CONTEXT MULTIPLIER
    # --------------------------------------------------------
    ctx_mult = get_context_multiplier(opp.strip().upper() if opp else None, market)
    heavy = HEAVY_TAIL[market]

    # Baseline minutes + scaling
    minutes = avg_min
    if teammate_out:
        minutes *= 1.05
        mu_min *= 1.06
    if blowout:
        minutes *= 0.90

    # Raw mean
    mu = mu_min * minutes * ctx_mult

    # Raw SD
    sd = max(1.0, sd_min * np.sqrt(max(minutes, 1.0)) * heavy)

    # --------------------------------------------------------
    # ADAPTIVE VOLATILITY ENGINE
    # --------------------------------------------------------
    sd_final = sd

    if opp:
        opp_abbrev = opp.strip().upper()
        if opp_abbrev in TEAM_CTX:
            opp_def = TEAM_CTX[opp_abbrev]["DEF_RATING"]
            league_def = LEAGUE_CTX["DEF_RATING"]
            def_vol = np.clip(opp_def / league_def, 0.85, 1.20)
            sd_final *= def_vol

            opp_pace = TEAM_CTX[opp_abbrev]["PACE"]
            league_pace = LEAGUE_CTX["PACE"]
            pace_vol = np.clip(opp_pace / league_pace, 0.90, 1.18)
            sd_final *= pace_vol

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
    sd_final = float(np.clip(sd_final, sd * 0.80, sd * 1.60))

    # --------------------------------------------------------
    # HYBRID PROBABILITY ENGINE
    # --------------------------------------------------------
    p_over = hybrid_prob_over(line, mu, sd_final, market)
    
    # Safety guard — ensure p_over valid
    if (
        p_over is None
        or not isinstance(p_over, (int, float))
        or np.isnan(p_over)
        or p_over <= 0
        or p_over >= 1
    ):
        p_over = 0.5  # fallback neutral only if invalid
    
    # EV vs even odds
    ev_leg_even = p_over - (1 - p_over)


    

    # --------------------------------------------------------
    # ALWAYS RETURN EXACTLY TWO VALUES
    # --------------------------------------------------------
    return {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": mu,
        "sd": sd_final,
        "prob_over": p_over,
        "ev_leg_even": ev_leg_even,
        "team": team,
        "ctx_mult": ctx_mult,
        "msg": msg,
        "teammate_out": teammate_out,
        "blowout": blowout,
    }, None




# =========================================================
#  PART 3.2 — KELLY FORMULA FOR 2-PICK
# =========================================================

def kelly_for_combo(p_joint: float, payout_mult: float, frac: float):
    # ======================================================
# SELF-LEARNING CALIBRATION ENGINE (Upgrade 4 — Part 7)
# ======================================================

def compute_model_drift(history_df):
    """
    Uses completed bets (Hit/Miss) to detect model miscalibration.
    Returns recommended adjustments:
    - variance_factor_adj
    - heavy_tail_adj
    """

    comp = history_df[history_df["Result"].isin(["Hit", "Miss"])].copy()
    if comp.empty or len(comp) < 20:
        return 1.0, 1.0  # no adjustments until sample size is decent

    comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
    comp = comp.dropna(subset=["EV_float"])

    if comp.empty:
        return 1.0, 1.0

    # Expected hit rate based on EV distribution
    predicted = 0.5 + comp["EV_float"].mean()

    # Actual hit rate
    actual = (comp["Result"] == "Hit").mean()

    # Gap → model miscalibration signal
    gap = actual - predicted  # positive = model too conservative

    # -----------------------------------------------------
    # Variance Adjustment Logic
    # -----------------------------------------------------
    # If actual < predicted → model too optimistic → increase variance
    # If actual > predicted → model too conservative → reduce variance

    if gap < -0.03:
        # too optimistic
        variance_adj = 1.06  # widen sd
        ht_adj = 1.03        # reinforce tail heaviness
    elif gap > 0.03:
        # too conservative
        variance_adj = 0.94  # tighten sd
        ht_adj = 0.97        # reduce tail
    else:
        variance_adj = 1.0
        ht_adj = 1.0

    # clamp for safety
    variance_adj = float(np.clip(variance_adj, 0.90, 1.10))
    ht_adj = float(np.clip(ht_adj, 0.90, 1.10))

    return variance_adj, ht_adj

    """
    Kelly for 2-pick:
      - p_joint = joint probability both legs hit
      - b = payout - 1
      - q = 1 - p_joint
      - raw Kelly = (b*p - q)/b
      - Apply fractional Kelly & max cap
    """
    b = payout_mult - 1
    q = 1 - p_joint

    raw = (b * p_joint - q) / b
    k = raw * frac

    return float(np.clip(k, 0, MAX_KELLY_PCT))  # apply hard cap
# =========================================================
# PART 4 — UI RENDER ENGINE + LOADERS + DECISION LOGIC
# =========================================================

# ---------------------------------------------------------
# 4.1 — RENDER LEG CARD (fully upgraded version)
# ---------------------------------------------------------
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
        st.write(f"📊 **Model Mean:** {mu:.2f}")
        st.write(f"📉 **Model SD:** {sd:.2f}")
        st.write(f"⏱️ **Context Multiplier:** {ctx:.3f}")
        st.write(f"🎯 **Model Probability Over:** {p*100:.1f}%")
        st.write(f"💵 **Even-Money EV:** {even_ev*100:+.1f}%")
        st.caption(f"📝 {msg}")

        # Risk flags
        if teammate_out:
            st.info("⚠️ Teammate out → usage boost applied.")
        if blowout:
            st.warning("⚠️ Blowout risk → minutes reduced.")


# ---------------------------------------------------------
# 4.2 — RUN LOADER ANIMATION
# ---------------------------------------------------------
def run_loader():
    """
    Friendly loading animation for model runs.
    Avoids blocking, visually clean.
    """
    load_ph = st.empty()
    msgs = [
        "Pulling player logs…",
        "Analyzing matchup context…",
        "Calculating distribution…",
        "Simulating outcomes…",
        "Finalizing edge…",
    ]
    for m in msgs:
        load_ph.markdown(
            f"<p style='color:#FFCC33;font-size:20px;font-weight:600;'>{m}</p>",
            unsafe_allow_html=True,
        )
        time.sleep(0.35)
    load_ph.empty()


# ---------------------------------------------------------
# 4.3 — DECISION LOGIC FOR COMBO PICK
# ---------------------------------------------------------
def combo_decision(ev_combo: float) -> str:
    """
    Converts EV into a recommendation.
    """
    if ev_combo >= 0.10:
        return "🔥 **PLAY — Strong Edge**"
    elif ev_combo >= 0.03:
        return "🟡 **Lean — Thin Edge**"
    else:
        return "❌ **Pass — No Edge**"

# =====================================================
# APP TABS (this must appear BEFORE any "with tab_*")
# =====================================================

tab_model, tab_results, tab_history, tab_calib = st.tabs(
    ["📊 Model", "📓 Results", "📜 History", "🧠 Calibration"]
)

with tab_model:

    st.subheader("2-Pick Projection & Edge (Auto stats + manual lines)")

    c1, c2 = st.columns(2)

    # ------------------------------------------------------
    # LEFT LEG — PLAYER 1 INPUTS
    # ------------------------------------------------------
    with c1:
        p1 = st.text_input("Player 1 Name")
        m1 = st.selectbox("P1 Market", MARKET_OPTIONS)
        l1 = st.number_input("P1 Line", min_value=0.0, value=25.0, step=0.5)
        o1 = st.text_input("P1 Opponent (abbr)", help="Example: BOS, DEN")
        p1_teammate_out = st.checkbox("P1: Key teammate out?")
        p1_blowout = st.checkbox("P1: Blowout risk high?")

    # ------------------------------------------------------
    # RIGHT LEG — PLAYER 2 INPUTS
    # ------------------------------------------------------
    with c2:
        p2 = st.text_input("Player 2 Name")
        m2 = st.selectbox("P2 Market", MARKET_OPTIONS)
        l2 = st.number_input("P2 Line", min_value=0.0, value=25.0, step=0.5)
        o2 = st.text_input("P2 Opponent (abbr)", help="Example: BOS, DEN")
        p2_teammate_out = st.checkbox("P2: Key teammate out?")
        p2_blowout = st.checkbox("P2: Blowout risk high?")

    # Safety defaults
    leg1 = None
    leg2 = None

    run = st.button("Run Model ⚡")

    # =========================================================
    # MODEL RUN
    # =========================================================
    if run:

        # Validate
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        run_loader()

        # ------------------------------------------------------
        # Compute legs (safe)
        # ------------------------------------------------------
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

        # ------------------------------------------------------
        # Render Legs
        # ------------------------------------------------------
        colL, colR = st.columns(2)

        if leg1:
            render_leg_card(leg1, colL, compact_mode)
        if leg2:
            render_leg_card(leg2, colR, compact_mode)

        # =========================================================
        # IMPLIED PROBABILITY VS MODEL
        # =========================================================
        st.markdown("---")
        st.subheader("📈 Market vs Model Probability Check")

        def implied_probability(mult):
            return 1.0 / mult

        imp_prob = implied_probability(payout_mult)
        st.markdown(f"**Market Implied Probability:** {imp_prob*100:.1f}%")

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

        # =========================================================
        # =========================================================
        # 2-PICK COMBO — FINAL MODEL OUTPUT
        # =========================================================
        if leg1 and leg2:

            # ---------------------------
            # Correlation
            # ---------------------------
            corr = 0.0
            if leg1["team"] and leg2["team"] and leg1["team"] == leg2["team"]:
                corr = 0.25  # same-team bump

            base_joint = leg1["prob_over"] * leg2["prob_over"]
            joint = base_joint + corr * (
                min(leg1["prob_over"], leg2["prob_over"]) - base_joint
            )
            joint = float(np.clip(joint, 0.0, 1.0))

            # ---------------------------
            # EV + Kelly stake
            # ---------------------------
            ev_combo = payout_mult * joint - 1.0
            k_frac = kelly_for_combo(joint, payout_mult, fractional_kelly)
            stake = round(bankroll * k_frac, 2)
            decision = combo_decision(ev_combo)

            st.markdown("### 🎯 **2-Pick Combo Result**")
            st.markdown(f"- Correlation: **{corr:+.2f}**")
            st.markdown(f"- Joint Probability: **{joint*100:.1f}%**")
            st.markdown(f"- EV (per $1): **{ev_combo*100:+.1f}%**")
            st.markdown(f"- Suggested Stake (Kelly-capped): **${stake:.2f}**")
            st.markdown(f"- **Recommendation:** {decision}")


        # =========================================================
        # MARKET BASELINE LIBRARY HOOK
        # =========================================================
        for leg in [leg1, leg2]:
            if leg:
                mean_b, med_b = get_market_baseline(leg["player"], leg["market"])
                if mean_b:
                    st.caption(
                        f"📊 Market Baseline for {leg['player']} {leg['market']}: "
                        f"mean={mean_b:.1f}, median={med_b:.1f}"
                    )
# =====================================================
# HISTORY HELPERS
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

# =========================================================
# PART 6 — RESULTS TAB
# =========================================================

with tab_results:

    st.subheader("Results & Personal Tracking")

    df = load_history()

    # ------------------------------
    # Display Logged History Table
    # ------------------------------
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No bets logged yet. Log entries after you place bets.")

    # ------------------------------
    # LOG RESULT FORM
    # ------------------------------
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

        # ---------------------------------------------------
        # SUBMIT LOG ENTRY
        # ---------------------------------------------------
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

    # ------------------------------
    # SUMMARY METRICS
    # ------------------------------
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

        # ------------------------------
        # PLOT PROFIT TREND
        # ------------------------------
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
# PART 7 — HISTORY TAB
# =========================================================

with tab_history:

    st.subheader("History & Filters")

    df = load_history()

    # -------------------------
    # If empty, show message
    # -------------------------
    if df.empty:
        st.info("No logged bets yet.")
    else:

        # --------------------------------------
        # FILTER CONTROLS
        # --------------------------------------
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

        # --------------------------------------
        # APPLY FILTERS
        # --------------------------------------
        filt = df[df["EV"] >= min_ev]

        if market_filter != "All":
            filt = filt[filt["Market"] == market_filter]

        st.markdown(f"**Filtered Bets:** {len(filt)}")

        # Show table
        st.dataframe(filt, use_container_width=True)

        # --------------------------------------
        # PROFIT CURVE BASED ON FILTER
        # --------------------------------------
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
# PART 8 — CALIBRATION TAB
# =========================================================

with tab_calib:

    st.subheader("Calibration & Edge Integrity Check")

    df = load_history()
    comp = df[df["Result"].isin(["Hit", "Miss"])]

    # -------------------------------------------
    # REQUIRE ENOUGH SAMPLES
    # -------------------------------------------
    if comp.empty or len(comp) < 15:
        st.info("Log at least 15 completed bets with EV to start calibration.")
    else:

        comp = comp.copy()
        comp["EV_float"] = pd.to_numeric(comp["EV"], errors="coerce") / 100.0
        comp = comp.dropna(subset=["EV_float"])

        if comp.empty:
            st.info("No valid EV values yet.")
        else:

            # -------------------------------------------
            # CALCULATE MODEL PREDICTION VS REALITY
            # -------------------------------------------
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

            # -------------------------------------------
            # 📊 DISTRIBUTION OF MODEL EDGE VS MARKET
            # -------------------------------------------
            st.markdown("---")
            st.subheader("Market vs Model Performance Trend")

            comp["Edge_vs_Market"] = comp["EV_float"] * 100

            fig2 = px.histogram(
                comp,
                x="Edge_vs_Market",
                nbins=20,
                title="Distribution of Model Edge vs Market (EV%)",
                color_discrete_sequence=["#FFCC33"]
            )
            st.plotly_chart(fig2, use_container_width=True)

            # -------------------------------------------
            # SUMMARY VALUES
            # -------------------------------------------
            st.markdown(
                f"**Predicted Avg Win Prob (approx):** {pred_win_prob*100:.1f}%"
            )
            st.markdown(
                f"**Actual Hit Rate:** {actual_win_prob*100:.1f}%"
            )
            st.markdown(
                f"**Calibration Gap:** {gap:+.1f}% | **ROI:** {roi:+.1f}%"
            )

            # -------------------------------------------
            # MODEL INTEGRITY CHECK
            # -------------------------------------------
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
# PART 9 — FOOTER & FINAL ASSEMBLY
# =========================================================

st.markdown(
    """
    <footer style='text-align:center; margin-top:30px; color:#FFCC33; font-size:11px;'>
        © 2025 NBA Prop Model • Powered by Kamal
    </footer>
    """,
    unsafe_allow_html=True,
)
