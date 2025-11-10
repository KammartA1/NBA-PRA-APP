import os
import csv
import difflib
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
from scipy.stats import norm

# =========================
# BASE CONFIG
# =========================

st.set_page_config(
    page_title="NBA 2-Pick Prop Edge Model",
    page_icon="🏀",
    layout="wide",
)
st.set_option("client.toolbarMode", "minimal")

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
DARK_BG = "#0C0B10"
CARD_BG = "#17131C"

MAX_BANKROLL_PCT = 0.03  # 3% max stake per entry

MARKET_OPTIONS = [
    "PRA (Points + Rebounds + Assists)",
    "Points",
    "Rebounds",
    "Assists",
]

market_key_map = {
    "PRA (Points + Rebounds + Assists)": "pra",
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
}

metric_map = {
    "pra": ["PTS", "REB", "AST"],
    "pts": ["PTS"],
    "reb": ["REB"],
    "ast": ["AST"],
}

# Usage model assumptions
LEAGUE_USAGE_PCT = 20.0
BASE_USAGE_PER_MIN = 0.39  # proxy FGA+0.44*FTA+TOV per minute at ~20% usage

# Default model parameters (can be session-tuned)
if "ht_pra" not in st.session_state:
    st.session_state["ht_pra"] = 1.35
if "ht_other" not in st.session_state:
    st.session_state["ht_other"] = 1.25
if "usage_min" not in st.session_state:
    st.session_state["usage_min"] = 0.7
if "usage_max" not in st.session_state:
    st.session_state["usage_max"] = 1.4
if "ev_threshold_play" not in st.session_state:
    st.session_state["ev_threshold_play"] = 0.10  # 10%+
if "ev_threshold_thin" not in st.session_state:
    st.session_state["ev_threshold_thin"] = 0.05  # 5–10% thin edge

# =========================
# SESSION STATE DEFAULTS
# =========================

defaults = {
    "user_id": "Me",
    "bankroll": 1000.0,
    "payout_mult": 3.0,
    "fractional_kelly": 0.25,
    "games_lookback": 10,
    "p1_name": "RJ Barrett",
    "p2_name": "Jaylen Brown",
    "p1_line": 33.5,
    "p2_line": 34.5,
    "p1_market_label": "PRA (Points + Rebounds + Assists)",
    "p2_market_label": "PRA (Points + Rebounds + Assists)",
    "compact_mode": True,
    "last_run": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# GLOBAL STYLE
# =========================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {DARK_BG};
        color: #FFFFFF;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, -sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: radial-gradient(circle at top, {PRIMARY_MAROON} 0%, #2b0b14 55%, #12060a 100%);
        border-right: 1px solid {GOLD}33;
    }}
    h1, h2, h3, h4, h5 {{
        color: {GOLD};
        font-weight: 700;
    }}
    .prop-card {{
        background: {CARD_BG};
        border-radius: 18px;
        padding: 16px 14px 12px 14px;
        border: 1px solid {GOLD}22;
        box-shadow: 0 8px 20px rgba(0,0,0,0.45);
        margin-bottom: 12px;
    }}
    .divider-gold {{
        height: 2px;
        background: linear-gradient(90deg, transparent, {GOLD}, transparent);
        margin: 10px 0 14px 0;
    }}
    .info-icon {{
        display: inline-block;
        margin-left: 4px;
        color: {GOLD};
        cursor: help;
        font-weight: 700;
    }}
    .prob-bar-outer {{
        width: 100%;
        background: #2a222f;
        border-radius: 999px;
        height: 8px;
        margin-top: 4px;
    }}
    .prob-bar-inner {{
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, {GOLD}, #ffec99);
    }}
    .stTextInput input, .stNumberInput input, .stSelectbox select {{
        background-color: #221925 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }}
    .stSlider > div > div > div {{
        background-color: {PRIMARY_MAROON}55 !important;
    }}
    @media (max-width: 768px) {{
        h1 {{
            font-size: 1.6rem;
        }}
        h2 {{
            font-size: 1.2rem;
        }}
        .prop-card {{
            padding: 14px 10px;
        }}
        div[data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
        button[kind="primary"] {{
            width: 100% !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TITLE
# =========================

st.markdown(
    """
    <h1>🏀 NBA 2-Pick Prop Edge & Risk Engine</h1>
    <div class="divider-gold"></div>
    <p>
      Manual-line, usage-aware, context-adjusted model for 2-pick props:
      <ul>
        <li>Per-player markets: <b>PRA, Points, Rebounds, Assists</b></li>
        <li>Usage, minutes, pace & opponent defense integrated</li>
        <li>Context toggles: key teammate out, blowout risk</li>
        <li>Heavy-tail variance, realistic probabilities & Play/Pass logic</li>
        <li>Personal bet history, results, CLV, and calibration-driven tuning</li>
      </ul>
    </p>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR (USER & SETTINGS)
# =========================

st.sidebar.header("User & Bankroll")

user_input = st.sidebar.text_input(
    "Your ID (for personal bet history)",
    value=st.session_state["user_id"],
    help="Use something unique (e.g. your name or initials)."
).strip() or "Me"

safe_id = "".join(c for c in user_input if c.isalnum() or c in ("_", "-")).strip() or "Me"
st.session_state["user_id"] = user_input

LOG_FILE = f"bet_history_{safe_id}.csv"
BACKUP_FILE = f"bet_history_{safe_id}_backup.csv"

st.sidebar.caption(f"Your bets are logged only to **{LOG_FILE}** on this app.")

st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input(
    "Bankroll ($)",
    min_value=10.0,
    value=float(st.session_state["bankroll"]),
    step=10.0,
    key="bankroll",
    help="Starting bankroll for stake sizing & bankroll curve."
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout Multiplier",
    min_value=1.01,
    value=float(st.session_state["payout_mult"]),
    step=0.1,
    key="payout_mult",
    help="Total payout for a winning 2-pick (e.g., 3.0x Power Play)."
)

fractional_kelly = st.sidebar.slider(
    "Fractional Kelly",
    0.0,
    1.0,
    value=float(st.session_state["fractional_kelly"]),
    step=0.05,
    key="fractional_kelly",
    help="Use 0.1–0.3 for safer growth. Each stake is capped at 3% of bankroll."
)

games_lookback = st.sidebar.slider(
    "Recent Games (N)",
    5,
    20,
    value=int(st.session_state["games_lookback"]),
    step=1,
    key="games_lookback",
    help="Recent games used for per-minute rates, minutes & usage."
)

compact_mode = st.sidebar.checkbox(
    "Compact Mode (mobile-friendly)",
    value=bool(st.session_state["compact_mode"]),
    key="compact_mode",
    help="Hide some details for a cleaner look."
)

st.sidebar.caption("Lines & decisions are yours. Math keeps you honest. 🧮")

# =========================
# HELPERS
# =========================

def _norm_name(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )

def current_season() -> str:
    today = datetime.now()
    year = today.year
    start = year if today.month >= 10 else year - 1
    end = start + 1
    return f"{start}-{str(end)[-2:]}"

@st.cache_data(show_spinner=False, ttl=600)
def nba_lookup_player(name: str):
    players = nba_players.get_players()
    target = _norm_name(name)
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]
    norm_names = [_norm_name(p["full_name"]) for p in players]
    best = difflib.get_close_matches(target, norm_names, n=1, cutoff=0.6)
    if best:
        chosen = best[0]
        for p in players:
            if _norm_name(p["full_name"]) == chosen:
                return p["id"], p["full_name"]
    return None, f"No NBA player match for '{name}'."

def headshot_url(player_id: int | None):
    if not player_id:
        return None
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

@st.cache_data(show_spinner=False, ttl=600)
def get_team_context():
    """
    Returns dict: {TEAM_ABBREV: {"PACE": pace, "DEF_RATING": def_rating}}
    """
    try:
        stats = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Defense"
        ).get_data_frames()[0]
        pace_stats = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "PACE"]]

        df = stats.merge(
            pace_stats[["TEAM_ABBREVIATION", "PACE"]],
            on="TEAM_ABBREVIATION",
            suffixes=("", "_PACE"),
            how="left"
        )

        league_pace = df["PACE"].mean()
        league_def = df["DEF_RATING"].mean()

        ctx = {}
        for _, r in df.iterrows():
            abbr = r["TEAM_ABBREVIATION"]
            pace = float(r.get("PACE", league_pace))
            d = float(r.get("DEF_RATING", league_def))
            ctx[abbr] = {
                "PACE": pace,
                "DEF_RATING": d,
            }
        return ctx, league_pace, league_def
    except Exception:
        return {}, None, None

TEAM_CTX, LEAGUE_PACE, LEAGUE_DEF = get_team_context()

@st.cache_data(show_spinner=False, ttl=600)
def get_player_rate_and_minutes(name: str, n_games: int, market_key: str):
    """
    Returns:
      mu_per_min, sd_per_min, avg_minutes,
      msg, team_abbrev, usage_pct, player_id
    """
    cols = metric_map[market_key]
    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, None, f"Could not find player '{name}'.", None, None, None

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
    except Exception as e:
        return None, None, None, f"Error fetching logs for {label}: {e}", None, None, pid

    if df.empty:
        return None, None, None, f"No logs found for {label}.", None, None, pid

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    per_min_vals, minutes_list, usg_like_vals = [], [], []

    for _, row in df.iterrows():
        total_val = sum(float(row.get(c, 0)) for c in cols)
        mins_raw = row.get("MIN", 0)
        minutes = 0.0
        try:
            if isinstance(mins_raw, str) and ":" in mins_raw:
                mm, ss = mins_raw.split(":")
                minutes = float(mm) + float(ss) / 60.0
            else:
                minutes = float(mins_raw)
        except Exception:
            minutes = 0.0
        if minutes <= 0:
            continue

        per_min_vals.append(total_val / minutes)
        minutes_list.append(minutes)

        fga = float(row.get("FGA", 0))
        fta = float(row.get("FTA", 0))
        tov = float(row.get("TOV", 0))
        usg_like = (fga + 0.44 * fta + tov) / minutes
        usg_like_vals.append(usg_like)

    if len(per_min_vals) < 3:
        return None, None, None, f"Not enough valid recent games for {label}.", None, None, pid

    per_min_arr = np.array(per_min_vals)
    mins_arr = np.array(minutes_list)

    weights = np.linspace(0.5, 1.5, len(per_min_arr))
    weights /= weights.sum()

    mu_per_min = float(np.average(per_min_arr, weights=weights))
    avg_min = float(np.average(mins_arr, weights=weights))

    sd_per_min = float(per_min_arr.std(ddof=1))
    if sd_per_min <= 0:
        sd_per_min = max(0.05, 0.1 * max(mu_per_min, 0.5))

    if usg_like_vals:
        usg_arr = np.array(usg_like_vals)
        usg_per_min = float(np.average(usg_arr, weights=weights))
        approx_usage_pct = (usg_per_min / BASE_USAGE_PER_MIN) * LEAGUE_USAGE_PCT
        usage_pct = float(np.clip(approx_usage_pct, 10.0, 40.0))
    else:
        usage_pct = LEAGUE_USAGE_PCT

    team_abbrev = None
    if "TEAM_ABBREVIATION" in df.columns:
        try:
            team_abbrev = df["TEAM_ABBREVIATION"].mode().iloc[0]
        except Exception:
            team_abbrev = None

    msg = (
        f"{label}: {len(per_min_vals)} recent games (weighted), "
        f"avg minutes {avg_min:.1f}"
    )
    return mu_per_min, sd_per_min, avg_min, msg, team_abbrev, usage_pct, pid

def get_context_multipliers(team_abbrev: str | None, opp_abbrev: str | None):
    """
    Pace + opponent defense multiplier.
    If no opp provided or context missing -> 1.0
    """
    if not opp_abbrev or opp_abbrev not in TEAM_CTX or LEAGUE_PACE is None or LEAGUE_DEF is None:
        return 1.0
    opp = TEAM_CTX[opp_abbrev]
    pace_factor = float(opp["PACE"] / LEAGUE_PACE) if LEAGUE_PACE else 1.0
    # Def rating: lower = tougher defense → reduce projection
    def_factor = float(LEAGUE_DEF / opp["DEF_RATING"]) if opp["DEF_RATING"] else 1.0
    # Slight soften so it's not extreme
    def_factor = 0.5 + 0.5 * def_factor
    return max(0.85, min(1.15, pace_factor * def_factor))

def compute_leg(
    line: float,
    mu_per_min: float,
    sd_per_min: float,
    minutes: float,
    usage_pct: float,
    payout_mult: float,
    bankroll: float,
    kelly_frac: float,
    heavy_tail_factor: float,
    context_mult: float,
    key_teammate_out: bool,
    blowout_risk: bool,
):
    """
    Usage-adjusted, context-adjusted heavy-tailed normal model:
      - applies usage factor
      - pace/defense/context multipliers
      - clamps probabilities [5%, 95%]
      - returns raw EV for math, smoothed EV for display
    """
    # Usage factor
    u_min = st.session_state["usage_min"]
    u_max = st.session_state["usage_max"]
    usage_factor = 1.0 + (usage_pct - LEAGUE_USAGE_PCT) / 100.0
    usage_factor = float(np.clip(usage_factor, u_min, u_max))

    # Manual context toggles
    if key_teammate_out:
        usage_factor *= 1.08  # +8% usage
        minutes *= 1.04       # +4% minutes
    if blowout_risk:
        minutes *= 0.90       # -10% minutes

    # Apply pace/defense and context to mean
    mu_per_min_adj = mu_per_min * usage_factor * context_mult
    mu = mu_per_min_adj * minutes

    # Volatility
    base_sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    volatility_factor = 1.0 + (abs(usage_pct - LEAGUE_USAGE_PCT) / 100.0) * 0.3
    sd = max(1.0, base_sd * heavy_tail_factor * volatility_factor)

    # Probability of over
    p_over = 1.0 - norm.cdf(line, mu, sd)
    p_over = float(np.clip(p_over, 0.05, 0.95))

    b = payout_mult - 1.0
    ev_raw = p_over * b - (1.0 - p_over)

    full_kelly = max(0.0, (b * p_over - (1.0 - p_over)) / b) if b > 0 else 0.0

    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    ev_display = float(np.tanh(ev_raw * 0.5))

    return p_over, ev_raw, ev_display, full_kelly, stake, mu, sd

def adjust_joint_probability(p1_prob: float, p2_prob: float, corr: float):
    base = p1_prob * p2_prob
    adj = base + corr * (min(p1_prob, p2_prob) - base)
    return float(np.clip(adj, 0.0, 1.0))

def ensure_log_files():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Timestamp", "Player", "Market", "Line",
                "ModelMean", "ProbOVER", "EV", "Stake",
                "Type", "Extra", "PayoutMult", "Context",
                "ClosingLine", "CLV", "Result"
            ])
    if not os.path.exists(BACKUP_FILE):
        with open(BACKUP_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Timestamp", "Player", "Market", "Line",
                "ModelMean", "ProbOVER", "EV", "Stake",
                "Type", "Extra", "PayoutMult", "Context",
                "ClosingLine", "CLV", "Result"
            ])

def append_row(row):
    ensure_log_files()
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)
    with open(BACKUP_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

def load_history():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def save_history(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)
    df.to_csv(BACKUP_FILE, index=False)

def compute_profit(row):
    res = str(row.get("Result", "")).strip()
    try:
        stake = float(row.get("Stake", 0) or 0)
    except Exception:
        stake = 0.0
    if res not in ["Hit", "Miss", "Push"] or stake <= 0:
        return 0.0

    bet_type = str(row.get("Type", "Single"))
    try:
        payout_mult_row = float(row.get("PayoutMult", np.nan))
    except Exception:
        payout_mult_row = np.nan

    if "Combo" in bet_type:
        b = payout_mult_row - 1.0 if not np.isnan(payout_mult_row) else (payout_mult - 1.0)
    else:
        b = 1.0  # assume ~even for singles

    if res == "Push":
        return 0.0
    if res == "Hit":
        return stake * b
    if res == "Miss":
        return -stake
    return 0.0

def compute_clv(row):
    """Closing line value: positive if you beat the close on an over."""
    try:
        line = float(row.get("Line", np.nan))
        closing = float(row.get("ClosingLine", np.nan))
    except Exception:
        return np.nan
    if np.isnan(line) or np.isnan(closing):
        return np.nan
    return closing - line

# =========================
# TABS
# =========================

tab_model, tab_results, tab_history, tab_calib = st.tabs([
    "📊 Model",
    "📓 Results / Tracking",
    "📜 History",
    "🧠 Calibration & Auto-Tuning",
])

# =========================
# MODEL TAB
# =========================

with tab_model:
    st.subheader("🎯 Player Inputs & Context")

    col_left, col_right = st.columns(2)

    with col_left:
        p1_name = st.text_input("Player 1 Name", value=st.session_state["p1_name"], key="p1_name")
        p1_market_label = st.selectbox(
            "P1 Market",
            MARKET_OPTIONS,
            index=MARKET_OPTIONS.index(st.session_state["p1_market_label"]),
            key="p1_market_label",
        )
        p1_line = st.number_input(
            "P1 Line (manual)",
            min_value=1.0, max_value=100.0,
            value=float(st.session_state["p1_line"]),
            step=0.5,
            key="p1_line",
        )
        p1_opp = st.text_input(
            "P1 Opponent (Team Abbrev, optional)",
            value="",
            help="e.g. BOS, DEN. Used for pace/defense adjustment."
        )
        p1_teammate_out = st.checkbox(
            "P1: Key teammate out?",
            value=False,
            help="Boosts usage & minutes slightly."
        )
        p1_blowout = st.checkbox(
            "P1: Blowout risk high?",
            value=False,
            help="Reduces projected minutes."
        )

    with col_right:
        p2_name = st.text_input("Player 2 Name", value=st.session_state["p2_name"], key="p2_name")
        p2_market_label = st.selectbox(
            "P2 Market",
            MARKET_OPTIONS,
            index=MARKET_OPTIONS.index(st.session_state["p2_market_label"]),
            key="p2_market_label",
        )
        p2_line = st.number_input(
            "P2 Line (manual)",
            min_value=1.0, max_value=100.0,
            value=float(st.session_state["p2_line"]),
            step=0.5,
            key="p2_line",
        )
        p2_opp = st.text_input(
            "P2 Opponent (Team Abbrev, optional)",
            value="",
            help="e.g. BOS, DEN. Used for pace/defense adjustment."
        )
        p2_teammate_out = st.checkbox(
            "P2: Key teammate out?",
            value=False,
            help="Boosts usage & minutes slightly."
        )
        p2_blowout = st.checkbox(
            "P2: Blowout risk high?",
            value=False,
            help="Reduces projected minutes."
        )

    c1, c2 = st.columns(2)
    with c1:
        run_clicked = st.button("Run Model", use_container_width=True)
    with c2:
        quick_refresh = st.button(
            "Quick Refresh Last Bet",
            use_container_width=True,
            help="Re-run using your last successful inputs."
        )

    trigger, use_last = False, False
    if run_clicked:
        trigger = True
        use_last = False
    elif quick_refresh and st.session_state["last_run"]:
        trigger = True
        use_last = True

    if trigger:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
        else:
            if use_last:
                params = st.session_state["last_run"]
                p1_name_i = params["p1_name"]
                p2_name_i = params["p2_name"]
                p1_line_i = params["p1_line"]
                p2_line_i = params["p2_line"]
                p1_market_label_i = params["p1_market_label"]
                p2_market_label_i = params["p2_market_label"]
                p1_opp_i = params.get("p1_opp", "")
                p2_opp_i = params.get("p2_opp", "")
                p1_teammate_out_i = params.get("p1_teammate_out", False)
                p2_teammate_out_i = params.get("p2_teammate_out", False)
                p1_blowout_i = params.get("p1_blowout", False)
                p2_blowout_i = params.get("p2_blowout", False)
                payout_i = params["payout_mult"]
                bank_i = params["bankroll"]
                kelly_i = params["fractional_kelly"]
                gl_i = params["games_lookback"]
            else:
                p1_name_i, p2_name_i = p1_name, p2_name
                p1_line_i, p2_line_i = p1_line, p2_line
                p1_market_label_i, p2_market_label_i = p1_market_label, p2_market_label
                p1_opp_i, p2_opp_i = p1_opp.strip().upper(), p2_opp.strip().upper()
                p1_teammate_out_i, p2_teammate_out_i = p1_teammate_out, p2_teammate_out
                p1_blowout_i, p2_blowout_i = p1_blowout, p2_blowout
                payout_i, bank_i, kelly_i, gl_i = payout_mult, bankroll, fractional_kelly, games_lookback

                st.session_state["last_run"] = {
                    "p1_name": p1_name_i,
                    "p2_name": p2_name_i,
                    "p1_line": float(p1_line_i),
                    "p2_line": float(p2_line_i),
                    "p1_market_label": p1_market_label_i,
                    "p2_market_label": p2_market_label_i,
                    "p1_opp": p1_opp_i,
                    "p2_opp": p2_opp_i,
                    "p1_teammate_out": p1_teammate_out_i,
                    "p2_teammate_out": p2_teammate_out_i,
                    "p1_blowout": p1_blowout_i,
                    "p2_blowout": p2_blowout_i,
                    "payout_mult": float(payout_i),
                    "bankroll": float(bank_i),
                    "fractional_kelly": float(kelly_i),
                    "games_lookback": int(gl_i),
                }

            p1_key = market_key_map[p1_market_label_i]
            p2_key = market_key_map[p2_market_label_i]
            ht_pra = st.session_state["ht_pra"]
            ht_other = st.session_state["ht_other"]
            p1_ht = ht_pra if p1_key == "pra" else ht_other
            p2_ht = ht_pra if p2_key == "pra" else ht_other

            # Fetch stats
            p1_mu_min, p1_sd_min, p1_avg_min, p1_msg, p1_team, p1_usg, p1_pid = \
                get_player_rate_and_minutes(p1_name_i, gl_i, p1_key)
            if p1_mu_min is None:
                st.error(f"P1 stats error: {p1_msg}")
            else:
                p2_mu_min, p2_sd_min, p2_avg_min, p2_msg, p2_team, p2_usg, p2_pid = \
                    get_player_rate_and_minutes(p2_name_i, gl_i, p2_key)
                if p2_mu_min is None:
                    st.error(f"P2 stats error: {p2_msg}")
                else:
                    # Context multipliers
                    p1_ctx_mult = get_context_multipliers(p1_team, p1_opp_i)
                    p2_ctx_mult = get_context_multipliers(p2_team, p2_opp_i)

                    # Compute legs
                    p1_prob, p1_ev_raw, p1_ev_disp, p1_kelly, p1_stake, p1_mu, p1_sd = compute_leg(
                        p1_line_i, p1_mu_min, p1_sd_min, p1_avg_min,
                        p1_usg, payout_i, bank_i, kelly_i, p1_ht,
                        p1_ctx_mult, p1_teammate_out_i, p1_blowout_i
                    )
                    p2_prob, p2_ev_raw, p2_ev_disp, p2_kelly, p2_stake, p2_mu, p2_sd = compute_leg(
                        p2_line_i, p2_mu_min, p2_sd_min, p2_avg_min,
                        p2_usg, payout_i, bank_i, kelly_i, p2_ht,
                        p2_ctx_mult, p2_teammate_out_i, p2_blowout_i
                    )

                    # Correlation
                    corr = 0.0
                    corr_reason = "0.00 (Independent)"
                    if p1_team and p2_team and p1_team == p2_team:
                        corr = 0.35
                        corr_reason = f"+0.35 (Same team: {p1_team})"

                    joint_prob = adjust_joint_probability(p1_prob, p2_prob, corr)
                    b_combo = payout_i - 1.0
                    combo_ev_raw = payout_i * joint_prob - 1.0
                    combo_ev_disp = float(np.tanh(combo_ev_raw * 0.5))
                    combo_full_kelly = max(
                        0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo
                    ) if b_combo > 0 else 0.0
                    combo_stake = bank_i * kelly_i * combo_full_kelly
                    combo_stake = min(combo_stake, bank_i * MAX_BANKROLL_PCT)
                    combo_stake = max(0.0, round(combo_stake, 2))

                    # PLAY/PASS FLAGS
                    ev_play = st.session_state["ev_threshold_play"]
                    ev_thin = st.session_state["ev_threshold_thin"]

                    def play_label(ev_raw):
                        if ev_raw >= ev_play:
                            return "✅ PLAY"
                        elif ev_raw >= ev_thin:
                            return "⚠️ Thin edge"
                        else:
                            return "🚫 PASS"

                    # ===== Single-Leg Cards =====
                    st.markdown("## 📊 Single-Leg Results")
                    col_a, col_b = st.columns(2)

                    # P1
                    with col_a:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        hc1, hc2 = st.columns([1, 4])
                        with hc1:
                            url = headshot_url(p1_pid)
                            if url:
                                st.image(url, width=56)
                        with hc2:
                            st.markdown(f"### {p1_name_i}")
                            st.caption(p1_msg)
                        st.markdown(f"**Market:** {p1_market_label_i}")
                        st.markdown(f"**Line:** {p1_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Usage (weighted):** {p1_usg:.1f}%")
                            st.markdown(f"**Context Multiplier:** {p1_ctx_mult:.3f}")
                            st.markdown(f"**Proj Minutes (pre-context):** {p1_avg_min:.1f}")
                            st.markdown(f"**Model Mean (final):** {p1_mu:.1f}")
                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Usage + context adjusted probability.'>ℹ️</span>: "
                            f"{p1_prob * 100:.1f}%", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' "
                            f"style='width:{max(4,min(96,p1_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $ (display)** <span class='info-icon' title='Smoothed EV; staking uses raw edge.'>ℹ️</span>: "
                            f"{p1_ev_disp * 100:.1f}%", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"**Suggested Stake:** ${p1_stake:.2f}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Decision:** {play_label(p1_ev_raw)}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # P2
                    with col_b:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        hc1, hc2 = st.columns([1, 4])
                        with hc1:
                            url = headshot_url(p2_pid)
                            if url:
                                st.image(url, width=56)
                        with hc2:
                            st.markdown(f"### {p2_name_i}")
                            st.caption(p2_msg)
                        st.markdown(f"**Market:** {p2_market_label_i}")
                        st.markdown(f"**Line:** {p2_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Usage (weighted):** {p2_usg:.1f}%")
                            st.markdown(f"**Context Multiplier:** {p2_ctx_mult:.3f}")
                            st.markdown(f"**Proj Minutes (pre-context):** {p2_avg_min:.1f}")
                            st.markdown(f"**Model Mean (final):** {p2_mu:.1f}")
                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Usage + context adjusted probability.'>ℹ️</span>: "
                            f"{p2_prob * 100:.1f}%", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' "
                            f"style='width:{max(4,min(96,p2_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $ (display)**: {p2_ev_disp * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Suggested Stake:** ${p2_stake:.2f}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Decision:** {play_label(p2_ev_raw)}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ===== Combo =====
                    st.markdown("---")
                    st.markdown("## 🎯 2-Pick Combo (Both Must Hit)")

                    st.markdown(
                        f"**Correlation Applied** <span class='info-icon' title='Same-team legs assumed positively correlated.'>ℹ️</span>: "
                        f"{corr_reason}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Joint Prob:** {joint_prob * 100:.1f}%",
                    )
                    st.markdown(
                        f"**EV per $ (display):** {combo_ev_disp * 100:.1f}%",
                    )
                    st.markdown(
                        f"**Suggested Combo Stake:** ${combo_stake:.2f}",
                    )

                    combo_decision = (
                        "✅ PLAY"
                        if combo_ev_raw >= ev_play and combo_stake > 0
                        else "⚠️ Thin edge"
                        if combo_ev_raw >= ev_thin and combo_stake > 0
                        else "🚫 PASS"
                    )
                    st.markdown(f"**Combo Decision:** {combo_decision}")

                    # ===== Logging =====
                    ensure_log_files()
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    p1_context_str = f"Opp={p1_opp_i or '-'}, TeamOut={p1_teammate_out_i}, Blowout={p1_blowout_i}"
                    p2_context_str = f"Opp={p2_opp_i or '-'}, TeamOut={p2_teammate_out_i}, Blowout={p2_blowout_i}"
                    combo_context_str = f"{p1_context_str} | {p2_context_str}; Corr={corr_reason}"

                    append_row([
                        ts, p1_name_i, p1_market_label_i, p1_line_i,
                        round(p1_mu, 2), round(p1_prob, 4),
                        round(p1_ev_raw, 4), p1_stake,
                        "Single", f"Usage {p1_usg:.1f}%", payout_i,
                        p1_context_str, "", "", ""
                    ])

                    append_row([
                        ts, p2_name_i, p2_market_label_i, p2_line_i,
                        round(p2_mu, 2), round(p2_prob, 4),
                        round(p2_ev_raw, 4), p2_stake,
                        "Single", f"Usage {p2_usg:.1f}%", payout_i,
                        p2_context_str, "", "", ""
                    ])

                    append_row([
                        ts, f"{p1_name_i} + {p2_name_i}",
                        f"{p1_market_label_i} & {p2_market_label_i}",
                        f"{p1_line_i} & {p2_line_i}",
                        "-", round(joint_prob, 4),
                        round(combo_ev_raw, 4), combo_stake,
                        "Combo", "", payout_i,
                        combo_context_str, "", "", ""
                    ])

                    # Daily exposure warning
                    df_today = load_history()
                    if df_today is not None and "Timestamp" in df_today.columns:
                        try:
                            df_today["Timestamp"] = pd.to_datetime(df_today["Timestamp"])
                            today_mask = df_today["Timestamp"].dt.date == date.today()
                            spent_today = df_today.loc[today_mask, "Stake"].fillna(0).sum()
                            if spent_today > bankroll * 0.10:
                                st.warning(
                                    f"Risk Alert: Today's logged stakes total ${spent_today:.2f}, "
                                    f"which exceeds 10% of your bankroll."
                                )
                        except Exception:
                            pass

                    st.info(f"💾 Logged this run to {LOG_FILE} (plus backup). Use Results/History tabs to review.")

# =========================
# RESULTS / TRACKING TAB
# =========================

with tab_results:
    st.subheader("📓 Results & Bankroll Tracking")

    df = load_history()
    if df is None:
        st.info("No bets logged yet. Run the model in the 'Model' tab.")
    else:
        # Ensure columns exist
        for col in ["Result", "ClosingLine", "CLV"]:
            if col not in df.columns:
                df[col] = "" if col == "Result" else np.nan

        # Show editable table for Result & ClosingLine
        st.markdown("#### Update Outcomes & Closing Lines")
        editable = df.copy()
        allowed_results = ["", "Hit", "Miss", "Push"]

        editable = st.data_editor(
            editable,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Result": st.column_config.SelectboxColumn(
                    "Result",
                    options=allowed_results,
                    help="Set outcome after game settles.",
                    required=False,
                ),
                "ClosingLine": st.column_config.NumberColumn(
                    "ClosingLine",
                    help="Optional: final market line (for CLV tracking).",
                    format="%.2f",
                ),
            },
            disabled=[
                c for c in editable.columns if c not in ["Result", "ClosingLine"]
            ],
            key="results_editor",
        )

        # Recompute CLV
        editable["CLV"] = editable.apply(compute_clv, axis=1)

        if not editable.equals(df):
            save_history(editable)
            st.success("✅ Saved updates to results & CLV.")
            df = editable

        # Metrics
        eval_df = df[df["Result"].isin(["Hit", "Miss", "Push"])].copy()
        if not eval_df.empty:
            st.markdown("### 📈 Performance Summary")

            total_bets = len(eval_df)
            pushes = (eval_df["Result"] == "Push").sum()
            settled = total_bets - pushes
            hits = (eval_df["Result"] == "Hit").sum()
            hit_rate = hits / settled if settled > 0 else 0.0

            eval_df = eval_df.sort_values("Timestamp")
            profits = eval_df.apply(compute_profit, axis=1).values
            bankroll_series = bankroll + np.cumsum(profits)
            total_profit = bankroll_series[-1] - bankroll
            roi = total_profit / bankroll if bankroll > 0 else 0.0

            st.markdown(f"- **Bets Tracked:** {total_bets}")
            st.markdown(f"- **Hit Rate (excl. Push):** {hit_rate * 100:.1f}%")
            st.markdown(f"- **Net Profit:** ${total_profit:.2f}")
            st.markdown(f"- **ROI:** {roi * 100:.1f}%")

            # CLV stats
            if "CLV" in eval_df.columns:
                clv_vals = pd.to_numeric(eval_df["CLV"], errors="coerce").dropna()
                if not clv_vals.empty:
                    pos_clv = (clv_vals > 0).mean()
                    avg_clv = clv_vals.mean()
                    st.markdown(f"- **Avg CLV:** {avg_clv:+.2f}")
                    st.markdown(f"- **% Bets Beating Close:** {pos_clv * 100:.1f}%")

            # Bankroll curve
            st.markdown("#### 📉 Bankroll Over Time")
            try:
                chart_df = pd.DataFrame({
                    "Timestamp": pd.to_datetime(eval_df["Timestamp"]),
                    "Bankroll": bankroll_series,
                }).set_index("Timestamp")
                st.line_chart(chart_df, height=220)
            except Exception:
                pass
        else:
            st.info("Mark some bets as Hit/Miss/Push above to unlock performance tracking.")

# =========================
# HISTORY TAB
# =========================

with tab_history:
    st.subheader("📜 Bet History & Filters")

    df = load_history()
    if df is None:
        st.info("No history yet. Run the model to start logging.")
    else:
        # Filters
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            days = st.selectbox(
                "Lookback Window",
                ["All", "7 days", "30 days", "90 days"],
                index=1,
            )
        with colf2:
            min_ev = st.number_input(
                "Min EV (raw, e.g. 0.05 = 5%)",
                value=0.0,
                step=0.01,
            )
        with colf3:
            market_filter = st.selectbox(
                "Market Filter",
                ["All", "PRA", "Points", "Rebounds", "Assists"],
                index=0,
            )

        name_search = st.text_input(
            "Filter by Player (optional)",
            value="",
        ).strip().lower()

        hist = df.copy()
        # Time filter
        if "Timestamp" in hist.columns and days != "All":
            try:
                hist["Timestamp"] = pd.to_datetime(hist["Timestamp"])
                now = datetime.now()
                if days == "7 days":
                    cutoff = now - pd.Timedelta(days=7)
                elif days == "30 days":
                    cutoff = now - pd.Timedelta(days=30)
                else:
                    cutoff = now - pd.Timedelta(days=90)
                hist = hist[hist["Timestamp"] >= cutoff]
            except Exception:
                pass

        # EV filter
        if "EV" in hist.columns:
            hist = hist[pd.to_numeric(hist["EV"], errors="coerce").fillna(0) >= min_ev]

        # Market filter
        if market_filter != "All":
            if "Market" in hist.columns:
                if market_filter == "PRA":
                    hist = hist[hist["Market"].str.contains("PRA", na=False)]
                else:
                    hist = hist[hist["Market"].str.contains(market_filter, na=False)]

        # Name search
        if name_search:
            if "Player" in hist.columns:
                hist = hist[hist["Player"].str.lower().str.contains(name_search, na=False)]

        st.markdown(f"**Filtered Bets:** {len(hist)}")

        # Summary
        if len(hist) > 0 and "EV" in hist.columns:
            avg_ev = pd.to_numeric(hist["EV"], errors="coerce").mean()
            st.markdown(f"- **Avg EV (filtered):** {avg_ev * 100:.1f}%")

        # Trendline by day (net profit)
        if len(hist) > 0:
            try:
                hist["Timestamp"] = pd.to_datetime(hist["Timestamp"])
                hist["Profit"] = hist.apply(compute_profit, axis=1)
                daily = hist.groupby(hist["Timestamp"].dt.date)["Profit"].sum().cumsum()
                daily_df = pd.DataFrame({
                    "Date": daily.index,
                    "Cumulative Profit": daily.values
                }).set_index("Date")
                st.markdown("#### 📈 Cumulative Profit (Filtered Window)")
                st.line_chart(daily_df, height=220)
            except Exception:
                pass

        st.markdown("#### 📋 Detailed Table")
        st.dataframe(hist, use_container_width=True)

        # Export
        if st.button("Export Filtered to CSV", use_container_width=True):
            export_name = f"bet_history_filtered_{safe_id}.csv"
            hist.to_csv(export_name, index=False)
            st.success(f"Saved filtered history to {export_name} in app directory.")

# =========================
# CALIBRATION & AUTO-TUNING TAB
# =========================

with tab_calib:
    st.subheader("🧠 Calibration & Auto-Tuning (Manual Apply)")

    df = load_history()
    if df is None:
        st.info("No data yet. Log some bets first.")
    else:
        calib = df.copy()
        calib = calib[calib["Result"].isin(["Hit", "Miss"])]
        calib["ProbOVER"] = pd.to_numeric(calib.get("ProbOVER", np.nan), errors="coerce")
        calib = calib.dropna(subset=["ProbOVER"])
        if len(calib) < 20:
            st.info("Need at least 20 settled bets with ProbOVER to run calibration.")
        else:
            st.markdown(f"**Sample Size:** {len(calib)} settled bets")

            # Calibration curve: bin by predicted prob
            calib["bin"] = (calib["ProbOVER"] * 10).astype(int) / 10.0
            grp = calib.groupby("bin")
            bin_pred = grp["ProbOVER"].mean()
            bin_actual = (grp["Result"].apply(lambda s: (s == "Hit").mean()))
            calib_df = pd.DataFrame({
                "Pred": bin_pred,
                "Actual": bin_actual
            }).dropna()

            if not calib_df.empty:
                st.markdown("#### 🎯 Calibration Curve (Predicted vs Actual)")
                st.line_chart(calib_df[["Pred", "Actual"]], height=220)

            # EV vs ROI
            profits = calib.apply(compute_profit, axis=1)
            roi = profits.sum() / bankroll if bankroll > 0 else 0.0
            avg_ev = calib["EV"].mean() if "EV" in calib.columns else np.nan

            st.markdown("#### 📊 Calibration Metrics")
            st.markdown(f"- **Avg Model Prob:** {calib['ProbOVER'].mean() * 100:.1f}%")
            st.markdown(
                f"- **Actual Hit Rate:** {(calib['Result'] == 'Hit').mean() * 100:.1f}%"
            )
            if not np.isnan(avg_ev):
                st.markdown(f"- **Avg EV (model):** {avg_ev * 100:.1f}%")
            st.markdown(f"- **ROI (based on logged stakes):** {roi * 100:.1f}%")

            # CLV insight
            if "CLV" in calib.columns:
                clv_vals = pd.to_numeric(calib["CLV"], errors="coerce").dropna()
                if not clv_vals.empty:
                    st.markdown(f"- **Avg CLV (all bets):** {clv_vals.mean():+.2f}")
                    st.markdown(
                        f"- **% Bets Beating Close:** {(clv_vals > 0).mean() * 100:.1f}%"
                    )

            # ===== Recommendations =====
            pred = calib["ProbOVER"].mean()
            actual = (calib["Result"] == "Hit").mean()
            conf_gap = actual - pred  # positive = model underconfident

            recs = []
            new_ht_pra = st.session_state["ht_pra"]
            new_ht_other = st.session_state["ht_other"]
            new_usage_min = st.session_state["usage_min"]
            new_usage_max = st.session_state["usage_max"]

            # Overconfident: predicted > actual
            if pred - actual > 0.07:
                recs.append("Model is overconfident. Increase variance slightly.")
                new_ht_pra = min(1.6, new_ht_pra + 0.05)
                new_ht_other = min(1.5, new_ht_other + 0.05)
            # Underconfident
            elif actual - pred > 0.07:
                recs.append("Model is too conservative. Decrease variance slightly.")
                new_ht_pra = max(1.15, new_ht_pra - 0.05)
                new_ht_other = max(1.10, new_ht_other - 0.05)
            else:
                recs.append("Overall calibration is within a reasonable band.")

            # ROI vs EV alignment
            if not np.isnan(avg_ev):
                if avg_ev > 0 and roi < 0:
                    recs.append(
                        "Positive EV but negative ROI → edges likely overstated. "
                        "Consider slightly widening usage or context variance."
                    )
                    new_usage_max = max(new_usage_min + 0.5, new_usage_max - 0.05)
                elif avg_ev < 0 and roi > 0:
                    recs.append(
                        "Negative EV but positive ROI → model too pessimistic. "
                        "Slightly relax variance or EV thresholds."
                    )

            st.markdown("#### 🧩 Tuning Recommendations")
            st.markdown(
                f"""
                <div style="background-color:{CARD_BG};padding:12px;border-radius:10px;border:1px solid {GOLD}55;">
                <ul>
                {''.join(f'<li>{r}</li>' for r in recs)}
                </ul>
                <p>Proposed session-only parameters:</p>
                <ul>
                  <li>Heavy-tail (PRA): <b>{st.session_state['ht_pra']:.2f}</b> → <b>{new_ht_pra:.2f}</b></li>
                  <li>Heavy-tail (Other): <b>{st.session_state['ht_other']:.2f}</b> → <b>{new_ht_other:.2f}</b></li>
                  <li>Usage clamp: <b>{st.session_state['usage_min']:.2f} - {st.session_state['usage_max']:.2f}</b> → <b>{new_usage_min:.2f} - {new_usage_max:.2f}</b></li>
                </ul>
                <p style="font-size:0.85rem;color:#aaa;">
                These changes only affect this session and keep projections anchored to live stats.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_apply, col_reset = st.columns(2)
            with col_apply:
                if st.button("✅ Apply Suggested Changes (This Session Only)", use_container_width=True):
                    st.session_state["ht_pra"] = new_ht_pra
                    st.session_state["ht_other"] = new_ht_other
                    st.session_state["usage_min"] = new_usage_min
                    st.session_state["usage_max"] = new_usage_max
                    st.success("Updated model parameters for this session. Re-run bets in the Model tab.")
            with col_reset:
                if st.button("♻️ Reset to Default Parameters", use_container_width=True):
                    st.session_state["ht_pra"] = 1.35
                    st.session_state["ht_other"] = 1.25
                    st.session_state["usage_min"] = 0.7
                    st.session_state["usage_max"] = 1.4
                    st.success("Reset to baseline settings for this session.")

# =========================
# FOOTER
# =========================

st.caption(
    "This engine will never be 100% — and that's the point. "
    "Edges come from disciplined +EV spots, context, and honest tracking, not guarantees."
)
import os
import csv
import difflib
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats
from scipy.stats import norm

# =========================
# BASE CONFIG
# =========================

st.set_page_config(
    page_title="NBA 2-Pick Prop Edge Model",
    page_icon="🏀",
    layout="wide",
)
st.set_option("client.toolbarMode", "minimal")

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
DARK_BG = "#0C0B10"
CARD_BG = "#17131C"

MAX_BANKROLL_PCT = 0.03  # 3% max stake per entry

MARKET_OPTIONS = [
    "PRA (Points + Rebounds + Assists)",
    "Points",
    "Rebounds",
    "Assists",
]

market_key_map = {
    "PRA (Points + Rebounds + Assists)": "pra",
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
}

metric_map = {
    "pra": ["PTS", "REB", "AST"],
    "pts": ["PTS"],
    "reb": ["REB"],
    "ast": ["AST"],
}

# Usage model assumptions
LEAGUE_USAGE_PCT = 20.0
BASE_USAGE_PER_MIN = 0.39  # proxy FGA+0.44*FTA+TOV per minute at ~20% usage

# Default model parameters (can be session-tuned)
if "ht_pra" not in st.session_state:
    st.session_state["ht_pra"] = 1.35
if "ht_other" not in st.session_state:
    st.session_state["ht_other"] = 1.25
if "usage_min" not in st.session_state:
    st.session_state["usage_min"] = 0.7
if "usage_max" not in st.session_state:
    st.session_state["usage_max"] = 1.4
if "ev_threshold_play" not in st.session_state:
    st.session_state["ev_threshold_play"] = 0.10  # 10%+
if "ev_threshold_thin" not in st.session_state:
    st.session_state["ev_threshold_thin"] = 0.05  # 5–10% thin edge

# =========================
# SESSION STATE DEFAULTS
# =========================

defaults = {
    "user_id": "Me",
    "bankroll": 1000.0,
    "payout_mult": 3.0,
    "fractional_kelly": 0.25,
    "games_lookback": 10,
    "p1_name": "RJ Barrett",
    "p2_name": "Jaylen Brown",
    "p1_line": 33.5,
    "p2_line": 34.5,
    "p1_market_label": "PRA (Points + Rebounds + Assists)",
    "p2_market_label": "PRA (Points + Rebounds + Assists)",
    "compact_mode": True,
    "last_run": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# GLOBAL STYLE
# =========================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {DARK_BG};
        color: #FFFFFF;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, -sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: radial-gradient(circle at top, {PRIMARY_MAROON} 0%, #2b0b14 55%, #12060a 100%);
        border-right: 1px solid {GOLD}33;
    }}
    h1, h2, h3, h4, h5 {{
        color: {GOLD};
        font-weight: 700;
    }}
    .prop-card {{
        background: {CARD_BG};
        border-radius: 18px;
        padding: 16px 14px 12px 14px;
        border: 1px solid {GOLD}22;
        box-shadow: 0 8px 20px rgba(0,0,0,0.45);
        margin-bottom: 12px;
    }}
    .divider-gold {{
        height: 2px;
        background: linear-gradient(90deg, transparent, {GOLD}, transparent);
        margin: 10px 0 14px 0;
    }}
    .info-icon {{
        display: inline-block;
        margin-left: 4px;
        color: {GOLD};
        cursor: help;
        font-weight: 700;
    }}
    .prob-bar-outer {{
        width: 100%;
        background: #2a222f;
        border-radius: 999px;
        height: 8px;
        margin-top: 4px;
    }}
    .prob-bar-inner {{
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, {GOLD}, #ffec99);
    }}
    .stTextInput input, .stNumberInput input, .stSelectbox select {{
        background-color: #221925 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }}
    .stSlider > div > div > div {{
        background-color: {PRIMARY_MAROON}55 !important;
    }}
    @media (max-width: 768px) {{
        h1 {{
            font-size: 1.6rem;
        }}
        h2 {{
            font-size: 1.2rem;
        }}
        .prop-card {{
            padding: 14px 10px;
        }}
        div[data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
        button[kind="primary"] {{
            width: 100% !important;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TITLE
# =========================

st.markdown(
    """
    <h1>🏀 NBA 2-Pick Prop Edge & Risk Engine</h1>
    <div class="divider-gold"></div>
    <p>
      Manual-line, usage-aware, context-adjusted model for 2-pick props:
      <ul>
        <li>Per-player markets: <b>PRA, Points, Rebounds, Assists</b></li>
        <li>Usage, minutes, pace & opponent defense integrated</li>
        <li>Context toggles: key teammate out, blowout risk</li>
        <li>Heavy-tail variance, realistic probabilities & Play/Pass logic</li>
        <li>Personal bet history, results, CLV, and calibration-driven tuning</li>
      </ul>
    </p>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR (USER & SETTINGS)
# =========================

st.sidebar.header("User & Bankroll")

user_input = st.sidebar.text_input(
    "Your ID (for personal bet history)",
    value=st.session_state["user_id"],
    help="Use something unique (e.g. your name or initials)."
).strip() or "Me"

safe_id = "".join(c for c in user_input if c.isalnum() or c in ("_", "-")).strip() or "Me"
st.session_state["user_id"] = user_input

LOG_FILE = f"bet_history_{safe_id}.csv"
BACKUP_FILE = f"bet_history_{safe_id}_backup.csv"

st.sidebar.caption(f"Your bets are logged only to **{LOG_FILE}** on this app.")

st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input(
    "Bankroll ($)",
    min_value=10.0,
    value=float(st.session_state["bankroll"]),
    step=10.0,
    key="bankroll",
    help="Starting bankroll for stake sizing & bankroll curve."
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout Multiplier",
    min_value=1.01,
    value=float(st.session_state["payout_mult"]),
    step=0.1,
    key="payout_mult",
    help="Total payout for a winning 2-pick (e.g., 3.0x Power Play)."
)

fractional_kelly = st.sidebar.slider(
    "Fractional Kelly",
    0.0,
    1.0,
    value=float(st.session_state["fractional_kelly"]),
    step=0.05,
    key="fractional_kelly",
    help="Use 0.1–0.3 for safer growth. Each stake is capped at 3% of bankroll."
)

games_lookback = st.sidebar.slider(
    "Recent Games (N)",
    5,
    20,
    value=int(st.session_state["games_lookback"]),
    step=1,
    key="games_lookback",
    help="Recent games used for per-minute rates, minutes & usage."
)

compact_mode = st.sidebar.checkbox(
    "Compact Mode (mobile-friendly)",
    value=bool(st.session_state["compact_mode"]),
    key="compact_mode",
    help="Hide some details for a cleaner look."
)

st.sidebar.caption("Lines & decisions are yours. Math keeps you honest. 🧮")

# =========================
# HELPERS
# =========================

def _norm_name(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )

def current_season() -> str:
    today = datetime.now()
    year = today.year
    start = year if today.month >= 10 else year - 1
    end = start + 1
    return f"{start}-{str(end)[-2:]}"

@st.cache_data(show_spinner=False, ttl=600)
def nba_lookup_player(name: str):
    players = nba_players.get_players()
    target = _norm_name(name)
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]
    norm_names = [_norm_name(p["full_name"]) for p in players]
    best = difflib.get_close_matches(target, norm_names, n=1, cutoff=0.6)
    if best:
        chosen = best[0]
        for p in players:
            if _norm_name(p["full_name"]) == chosen:
                return p["id"], p["full_name"]
    return None, f"No NBA player match for '{name}'."

def headshot_url(player_id: int | None):
    if not player_id:
        return None
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

@st.cache_data(show_spinner=False, ttl=600)
def get_team_context():
    """
    Returns dict: {TEAM_ABBREV: {"PACE": pace, "DEF_RATING": def_rating}}
    """
    try:
        stats = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Defense"
        ).get_data_frames()[0]
        pace_stats = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame"
        ).get_data_frames()[0][["TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "PACE"]]

        df = stats.merge(
            pace_stats[["TEAM_ABBREVIATION", "PACE"]],
            on="TEAM_ABBREVIATION",
            suffixes=("", "_PACE"),
            how="left"
        )

        league_pace = df["PACE"].mean()
        league_def = df["DEF_RATING"].mean()

        ctx = {}
        for _, r in df.iterrows():
            abbr = r["TEAM_ABBREVIATION"]
            pace = float(r.get("PACE", league_pace))
            d = float(r.get("DEF_RATING", league_def))
            ctx[abbr] = {
                "PACE": pace,
                "DEF_RATING": d,
            }
        return ctx, league_pace, league_def
    except Exception:
        return {}, None, None

TEAM_CTX, LEAGUE_PACE, LEAGUE_DEF = get_team_context()

@st.cache_data(show_spinner=False, ttl=600)
def get_player_rate_and_minutes(name: str, n_games: int, market_key: str):
    """
    Returns:
      mu_per_min, sd_per_min, avg_minutes,
      msg, team_abbrev, usage_pct, player_id
    """
    cols = metric_map[market_key]
    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, None, f"Could not find player '{name}'.", None, None, None

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
    except Exception as e:
        return None, None, None, f"Error fetching logs for {label}: {e}", None, None, pid

    if df.empty:
        return None, None, None, f"No logs found for {label}.", None, None, pid

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    per_min_vals, minutes_list, usg_like_vals = [], [], []

    for _, row in df.iterrows():
        total_val = sum(float(row.get(c, 0)) for c in cols)
        mins_raw = row.get("MIN", 0)
        minutes = 0.0
        try:
            if isinstance(mins_raw, str) and ":" in mins_raw:
                mm, ss = mins_raw.split(":")
                minutes = float(mm) + float(ss) / 60.0
            else:
                minutes = float(mins_raw)
        except Exception:
            minutes = 0.0
        if minutes <= 0:
            continue

        per_min_vals.append(total_val / minutes)
        minutes_list.append(minutes)

        fga = float(row.get("FGA", 0))
        fta = float(row.get("FTA", 0))
        tov = float(row.get("TOV", 0))
        usg_like = (fga + 0.44 * fta + tov) / minutes
        usg_like_vals.append(usg_like)

    if len(per_min_vals) < 3:
        return None, None, None, f"Not enough valid recent games for {label}.", None, None, pid

    per_min_arr = np.array(per_min_vals)
    mins_arr = np.array(minutes_list)

    weights = np.linspace(0.5, 1.5, len(per_min_arr))
    weights /= weights.sum()

    mu_per_min = float(np.average(per_min_arr, weights=weights))
    avg_min = float(np.average(mins_arr, weights=weights))

    sd_per_min = float(per_min_arr.std(ddof=1))
    if sd_per_min <= 0:
        sd_per_min = max(0.05, 0.1 * max(mu_per_min, 0.5))

    if usg_like_vals:
        usg_arr = np.array(usg_like_vals)
        usg_per_min = float(np.average(usg_arr, weights=weights))
        approx_usage_pct = (usg_per_min / BASE_USAGE_PER_MIN) * LEAGUE_USAGE_PCT
        usage_pct = float(np.clip(approx_usage_pct, 10.0, 40.0))
    else:
        usage_pct = LEAGUE_USAGE_PCT

    team_abbrev = None
    if "TEAM_ABBREVIATION" in df.columns:
        try:
            team_abbrev = df["TEAM_ABBREVIATION"].mode().iloc[0]
        except Exception:
            team_abbrev = None

    msg = (
        f"{label}: {len(per_min_vals)} recent games (weighted), "
        f"avg minutes {avg_min:.1f}"
    )
    return mu_per_min, sd_per_min, avg_min, msg, team_abbrev, usage_pct, pid

def get_context_multipliers(team_abbrev: str | None, opp_abbrev: str | None):
    """
    Pace + opponent defense multiplier.
    If no opp provided or context missing -> 1.0
    """
    if not opp_abbrev or opp_abbrev not in TEAM_CTX or LEAGUE_PACE is None or LEAGUE_DEF is None:
        return 1.0
    opp = TEAM_CTX[opp_abbrev]
    pace_factor = float(opp["PACE"] / LEAGUE_PACE) if LEAGUE_PACE else 1.0
    # Def rating: lower = tougher defense → reduce projection
    def_factor = float(LEAGUE_DEF / opp["DEF_RATING"]) if opp["DEF_RATING"] else 1.0
    # Slight soften so it's not extreme
    def_factor = 0.5 + 0.5 * def_factor
    return max(0.85, min(1.15, pace_factor * def_factor))

def compute_leg(
    line: float,
    mu_per_min: float,
    sd_per_min: float,
    minutes: float,
    usage_pct: float,
    payout_mult: float,
    bankroll: float,
    kelly_frac: float,
    heavy_tail_factor: float,
    context_mult: float,
    key_teammate_out: bool,
    blowout_risk: bool,
):
    """
    Usage-adjusted, context-adjusted heavy-tailed normal model:
      - applies usage factor
      - pace/defense/context multipliers
      - clamps probabilities [5%, 95%]
      - returns raw EV for math, smoothed EV for display
    """
    # Usage factor
    u_min = st.session_state["usage_min"]
    u_max = st.session_state["usage_max"]
    usage_factor = 1.0 + (usage_pct - LEAGUE_USAGE_PCT) / 100.0
    usage_factor = float(np.clip(usage_factor, u_min, u_max))

    # Manual context toggles
    if key_teammate_out:
        usage_factor *= 1.08  # +8% usage
        minutes *= 1.04       # +4% minutes
    if blowout_risk:
        minutes *= 0.90       # -10% minutes

    # Apply pace/defense and context to mean
    mu_per_min_adj = mu_per_min * usage_factor * context_mult
    mu = mu_per_min_adj * minutes

    # Volatility
    base_sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    volatility_factor = 1.0 + (abs(usage_pct - LEAGUE_USAGE_PCT) / 100.0) * 0.3
    sd = max(1.0, base_sd * heavy_tail_factor * volatility_factor)

    # Probability of over
    p_over = 1.0 - norm.cdf(line, mu, sd)
    p_over = float(np.clip(p_over, 0.05, 0.95))

    b = payout_mult - 1.0
    ev_raw = p_over * b - (1.0 - p_over)

    full_kelly = max(0.0, (b * p_over - (1.0 - p_over)) / b) if b > 0 else 0.0

    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    ev_display = float(np.tanh(ev_raw * 0.5))

    return p_over, ev_raw, ev_display, full_kelly, stake, mu, sd

def adjust_joint_probability(p1_prob: float, p2_prob: float, corr: float):
    base = p1_prob * p2_prob
    adj = base + corr * (min(p1_prob, p2_prob) - base)
    return float(np.clip(adj, 0.0, 1.0))

def ensure_log_files():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Timestamp", "Player", "Market", "Line",
                "ModelMean", "ProbOVER", "EV", "Stake",
                "Type", "Extra", "PayoutMult", "Context",
                "ClosingLine", "CLV", "Result"
            ])
    if not os.path.exists(BACKUP_FILE):
        with open(BACKUP_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "Timestamp", "Player", "Market", "Line",
                "ModelMean", "ProbOVER", "EV", "Stake",
                "Type", "Extra", "PayoutMult", "Context",
                "ClosingLine", "CLV", "Result"
            ])

def append_row(row):
    ensure_log_files()
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)
    with open(BACKUP_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)

def load_history():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return None
        return df
    except Exception:
        return None

def save_history(df: pd.DataFrame):
    df.to_csv(LOG_FILE, index=False)
    df.to_csv(BACKUP_FILE, index=False)

def compute_profit(row):
    res = str(row.get("Result", "")).strip()
    try:
        stake = float(row.get("Stake", 0) or 0)
    except Exception:
        stake = 0.0
    if res not in ["Hit", "Miss", "Push"] or stake <= 0:
        return 0.0

    bet_type = str(row.get("Type", "Single"))
    try:
        payout_mult_row = float(row.get("PayoutMult", np.nan))
    except Exception:
        payout_mult_row = np.nan

    if "Combo" in bet_type:
        b = payout_mult_row - 1.0 if not np.isnan(payout_mult_row) else (payout_mult - 1.0)
    else:
        b = 1.0  # assume ~even for singles

    if res == "Push":
        return 0.0
    if res == "Hit":
        return stake * b
    if res == "Miss":
        return -stake
    return 0.0

def compute_clv(row):
    """Closing line value: positive if you beat the close on an over."""
    try:
        line = float(row.get("Line", np.nan))
        closing = float(row.get("ClosingLine", np.nan))
    except Exception:
        return np.nan
    if np.isnan(line) or np.isnan(closing):
        return np.nan
    return closing - line

# =========================
# TABS
# =========================

tab_model, tab_results, tab_history, tab_calib = st.tabs([
    "📊 Model",
    "📓 Results / Tracking",
    "📜 History",
    "🧠 Calibration & Auto-Tuning",
])

# =========================
# MODEL TAB
# =========================

with tab_model:
    st.subheader("🎯 Player Inputs & Context")

    col_left, col_right = st.columns(2)

    with col_left:
        p1_name = st.text_input("Player 1 Name", value=st.session_state["p1_name"], key="p1_name")
        p1_market_label = st.selectbox(
            "P1 Market",
            MARKET_OPTIONS,
            index=MARKET_OPTIONS.index(st.session_state["p1_market_label"]),
            key="p1_market_label",
        )
        p1_line = st.number_input(
            "P1 Line (manual)",
            min_value=1.0, max_value=100.0,
            value=float(st.session_state["p1_line"]),
            step=0.5,
            key="p1_line",
        )
        p1_opp = st.text_input(
            "P1 Opponent (Team Abbrev, optional)",
            value="",
            help="e.g. BOS, DEN. Used for pace/defense adjustment."
        )
        p1_teammate_out = st.checkbox(
            "P1: Key teammate out?",
            value=False,
            help="Boosts usage & minutes slightly."
        )
        p1_blowout = st.checkbox(
            "P1: Blowout risk high?",
            value=False,
            help="Reduces projected minutes."
        )

    with col_right:
        p2_name = st.text_input("Player 2 Name", value=st.session_state["p2_name"], key="p2_name")
        p2_market_label = st.selectbox(
            "P2 Market",
            MARKET_OPTIONS,
            index=MARKET_OPTIONS.index(st.session_state["p2_market_label"]),
            key="p2_market_label",
        )
        p2_line = st.number_input(
            "P2 Line (manual)",
            min_value=1.0, max_value=100.0,
            value=float(st.session_state["p2_line"]),
            step=0.5,
            key="p2_line",
        )
        p2_opp = st.text_input(
            "P2 Opponent (Team Abbrev, optional)",
            value="",
            help="e.g. BOS, DEN. Used for pace/defense adjustment."
        )
        p2_teammate_out = st.checkbox(
            "P2: Key teammate out?",
            value=False,
            help="Boosts usage & minutes slightly."
        )
        p2_blowout = st.checkbox(
            "P2: Blowout risk high?",
            value=False,
            help="Reduces projected minutes."
        )

    c1, c2 = st.columns(2)
    with c1:
        run_clicked = st.button("Run Model", use_container_width=True)
    with c2:
        quick_refresh = st.button(
            "Quick Refresh Last Bet",
            use_container_width=True,
            help="Re-run using your last successful inputs."
        )

    trigger, use_last = False, False
    if run_clicked:
        trigger = True
        use_last = False
    elif quick_refresh and st.session_state["last_run"]:
        trigger = True
        use_last = True

    if trigger:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
        else:
            if use_last:
                params = st.session_state["last_run"]
                p1_name_i = params["p1_name"]
                p2_name_i = params["p2_name"]
                p1_line_i = params["p1_line"]
                p2_line_i = params["p2_line"]
                p1_market_label_i = params["p1_market_label"]
                p2_market_label_i = params["p2_market_label"]
                p1_opp_i = params.get("p1_opp", "")
                p2_opp_i = params.get("p2_opp", "")
                p1_teammate_out_i = params.get("p1_teammate_out", False)
                p2_teammate_out_i = params.get("p2_teammate_out", False)
                p1_blowout_i = params.get("p1_blowout", False)
                p2_blowout_i = params.get("p2_blowout", False)
                payout_i = params["payout_mult"]
                bank_i = params["bankroll"]
                kelly_i = params["fractional_kelly"]
                gl_i = params["games_lookback"]
            else:
                p1_name_i, p2_name_i = p1_name, p2_name
                p1_line_i, p2_line_i = p1_line, p2_line
                p1_market_label_i, p2_market_label_i = p1_market_label, p2_market_label
                p1_opp_i, p2_opp_i = p1_opp.strip().upper(), p2_opp.strip().upper()
                p1_teammate_out_i, p2_teammate_out_i = p1_teammate_out, p2_teammate_out
                p1_blowout_i, p2_blowout_i = p1_blowout, p2_blowout
                payout_i, bank_i, kelly_i, gl_i = payout_mult, bankroll, fractional_kelly, games_lookback

                st.session_state["last_run"] = {
                    "p1_name": p1_name_i,
                    "p2_name": p2_name_i,
                    "p1_line": float(p1_line_i),
                    "p2_line": float(p2_line_i),
                    "p1_market_label": p1_market_label_i,
                    "p2_market_label": p2_market_label_i,
                    "p1_opp": p1_opp_i,
                    "p2_opp": p2_opp_i,
                    "p1_teammate_out": p1_teammate_out_i,
                    "p2_teammate_out": p2_teammate_out_i,
                    "p1_blowout": p1_blowout_i,
                    "p2_blowout": p2_blowout_i,
                    "payout_mult": float(payout_i),
                    "bankroll": float(bank_i),
                    "fractional_kelly": float(kelly_i),
                    "games_lookback": int(gl_i),
                }

            p1_key = market_key_map[p1_market_label_i]
            p2_key = market_key_map[p2_market_label_i]
            ht_pra = st.session_state["ht_pra"]
            ht_other = st.session_state["ht_other"]
            p1_ht = ht_pra if p1_key == "pra" else ht_other
            p2_ht = ht_pra if p2_key == "pra" else ht_other

            # Fetch stats
            p1_mu_min, p1_sd_min, p1_avg_min, p1_msg, p1_team, p1_usg, p1_pid = \
                get_player_rate_and_minutes(p1_name_i, gl_i, p1_key)
            if p1_mu_min is None:
                st.error(f"P1 stats error: {p1_msg}")
            else:
                p2_mu_min, p2_sd_min, p2_avg_min, p2_msg, p2_team, p2_usg, p2_pid = \
                    get_player_rate_and_minutes(p2_name_i, gl_i, p2_key)
                if p2_mu_min is None:
                    st.error(f"P2 stats error: {p2_msg}")
                else:
                    # Context multipliers
                    p1_ctx_mult = get_context_multipliers(p1_team, p1_opp_i)
                    p2_ctx_mult = get_context_multipliers(p2_team, p2_opp_i)

                    # Compute legs
                    p1_prob, p1_ev_raw, p1_ev_disp, p1_kelly, p1_stake, p1_mu, p1_sd = compute_leg(
                        p1_line_i, p1_mu_min, p1_sd_min, p1_avg_min,
                        p1_usg, payout_i, bank_i, kelly_i, p1_ht,
                        p1_ctx_mult, p1_teammate_out_i, p1_blowout_i
                    )
                    p2_prob, p2_ev_raw, p2_ev_disp, p2_kelly, p2_stake, p2_mu, p2_sd = compute_leg(
                        p2_line_i, p2_mu_min, p2_sd_min, p2_avg_min,
                        p2_usg, payout_i, bank_i, kelly_i, p2_ht,
                        p2_ctx_mult, p2_teammate_out_i, p2_blowout_i
                    )

                    # Correlation
                    corr = 0.0
                    corr_reason = "0.00 (Independent)"
                    if p1_team and p2_team and p1_team == p2_team:
                        corr = 0.35
                        corr_reason = f"+0.35 (Same team: {p1_team})"

                    joint_prob = adjust_joint_probability(p1_prob, p2_prob, corr)
                    b_combo = payout_i - 1.0
                    combo_ev_raw = payout_i * joint_prob - 1.0
                    combo_ev_disp = float(np.tanh(combo_ev_raw * 0.5))
                    combo_full_kelly = max(
                        0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo
                    ) if b_combo > 0 else 0.0
                    combo_stake = bank_i * kelly_i * combo_full_kelly
                    combo_stake = min(combo_stake, bank_i * MAX_BANKROLL_PCT)
                    combo_stake = max(0.0, round(combo_stake, 2))

                    # PLAY/PASS FLAGS
                    ev_play = st.session_state["ev_threshold_play"]
                    ev_thin = st.session_state["ev_threshold_thin"]

                    def play_label(ev_raw):
                        if ev_raw >= ev_play:
                            return "✅ PLAY"
                        elif ev_raw >= ev_thin:
                            return "⚠️ Thin edge"
                        else:
                            return "🚫 PASS"

                    # ===== Single-Leg Cards =====
                    st.markdown("## 📊 Single-Leg Results")
                    col_a, col_b = st.columns(2)

                    # P1
                    with col_a:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        hc1, hc2 = st.columns([1, 4])
                        with hc1:
                            url = headshot_url(p1_pid)
                            if url:
                                st.image(url, width=56)
                        with hc2:
                            st.markdown(f"### {p1_name_i}")
                            st.caption(p1_msg)
                        st.markdown(f"**Market:** {p1_market_label_i}")
                        st.markdown(f"**Line:** {p1_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Usage (weighted):** {p1_usg:.1f}%")
                            st.markdown(f"**Context Multiplier:** {p1_ctx_mult:.3f}")
                            st.markdown(f"**Proj Minutes (pre-context):** {p1_avg_min:.1f}")
                            st.markdown(f"**Model Mean (final):** {p1_mu:.1f}")
                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Usage + context adjusted probability.'>ℹ️</span>: "
                            f"{p1_prob * 100:.1f}%", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' "
                            f"style='width:{max(4,min(96,p1_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $ (display)** <span class='info-icon' title='Smoothed EV; staking uses raw edge.'>ℹ️</span>: "
                            f"{p1_ev_disp * 100:.1f}%", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"**Suggested Stake:** ${p1_stake:.2f}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Decision:** {play_label(p1_ev_raw)}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # P2
                    with col_b:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        hc1, hc2 = st.columns([1, 4])
                        with hc1:
                            url = headshot_url(p2_pid)
                            if url:
                                st.image(url, width=56)
                        with hc2:
                            st.markdown(f"### {p2_name_i}")
                            st.caption(p2_msg)
                        st.markdown(f"**Market:** {p2_market_label_i}")
                        st.markdown(f"**Line:** {p2_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Usage (weighted):** {p2_usg:.1f}%")
                            st.markdown(f"**Context Multiplier:** {p2_ctx_mult:.3f}")
                            st.markdown(f"**Proj Minutes (pre-context):** {p2_avg_min:.1f}")
                            st.markdown(f"**Model Mean (final):** {p2_mu:.1f}")
                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Usage + context adjusted probability.'>ℹ️</span>: "
                            f"{p2_prob * 100:.1f}%", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' "
                            f"style='width:{max(4,min(96,p2_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $ (display)**: {p2_ev_disp * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Suggested Stake:** ${p2_stake:.2f}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Decision:** {play_label(p2_ev_raw)}")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # ===== Combo =====
                    st.markdown("---")
                    st.markdown("## 🎯 2-Pick Combo (Both Must Hit)")

                    st.markdown(
                        f"**Correlation Applied** <span class='info-icon' title='Same-team legs assumed positively correlated.'>ℹ️</span>: "
                        f"{corr_reason}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Joint Prob:** {joint_prob * 100:.1f}%",
                    )
                    st.markdown(
                        f"**EV per $ (display):** {combo_ev_disp * 100:.1f}%",
                    )
                    st.markdown(
                        f"**Suggested Combo Stake:** ${combo_stake:.2f}",
                    )

                    combo_decision = (
                        "✅ PLAY"
                        if combo_ev_raw >= ev_play and combo_stake > 0
                        else "⚠️ Thin edge"
                        if combo_ev_raw >= ev_thin and combo_stake > 0
                        else "🚫 PASS"
                    )
                    st.markdown(f"**Combo Decision:** {combo_decision}")

                    # ===== Logging =====
                    ensure_log_files()
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    p1_context_str = f"Opp={p1_opp_i or '-'}, TeamOut={p1_teammate_out_i}, Blowout={p1_blowout_i}"
                    p2_context_str = f"Opp={p2_opp_i or '-'}, TeamOut={p2_teammate_out_i}, Blowout={p2_blowout_i}"
                    combo_context_str = f"{p1_context_str} | {p2_context_str}; Corr={corr_reason}"

                    append_row([
                        ts, p1_name_i, p1_market_label_i, p1_line_i,
                        round(p1_mu, 2), round(p1_prob, 4),
                        round(p1_ev_raw, 4), p1_stake,
                        "Single", f"Usage {p1_usg:.1f}%", payout_i,
                        p1_context_str, "", "", ""
                    ])

                    append_row([
                        ts, p2_name_i, p2_market_label_i, p2_line_i,
                        round(p2_mu, 2), round(p2_prob, 4),
                        round(p2_ev_raw, 4), p2_stake,
                        "Single", f"Usage {p2_usg:.1f}%", payout_i,
                        p2_context_str, "", "", ""
                    ])

                    append_row([
                        ts, f"{p1_name_i} + {p2_name_i}",
                        f"{p1_market_label_i} & {p2_market_label_i}",
                        f"{p1_line_i} & {p2_line_i}",
                        "-", round(joint_prob, 4),
                        round(combo_ev_raw, 4), combo_stake,
                        "Combo", "", payout_i,
                        combo_context_str, "", "", ""
                    ])

                    # Daily exposure warning
                    df_today = load_history()
                    if df_today is not None and "Timestamp" in df_today.columns:
                        try:
                            df_today["Timestamp"] = pd.to_datetime(df_today["Timestamp"])
                            today_mask = df_today["Timestamp"].dt.date == date.today()
                            spent_today = df_today.loc[today_mask, "Stake"].fillna(0).sum()
                            if spent_today > bankroll * 0.10:
                                st.warning(
                                    f"Risk Alert: Today's logged stakes total ${spent_today:.2f}, "
                                    f"which exceeds 10% of your bankroll."
                                )
                        except Exception:
                            pass

                    st.info(f"💾 Logged this run to {LOG_FILE} (plus backup). Use Results/History tabs to review.")

# =========================
# RESULTS / TRACKING TAB
# =========================

with tab_results:
    st.subheader("📓 Results & Bankroll Tracking")

    df = load_history()
    if df is None:
        st.info("No bets logged yet. Run the model in the 'Model' tab.")
    else:
        # Ensure columns exist
        for col in ["Result", "ClosingLine", "CLV"]:
            if col not in df.columns:
                df[col] = "" if col == "Result" else np.nan

        # Show editable table for Result & ClosingLine
        st.markdown("#### Update Outcomes & Closing Lines")
        editable = df.copy()
        allowed_results = ["", "Hit", "Miss", "Push"]

        editable = st.data_editor(
            editable,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Result": st.column_config.SelectboxColumn(
                    "Result",
                    options=allowed_results,
                    help="Set outcome after game settles.",
                    required=False,
                ),
                "ClosingLine": st.column_config.NumberColumn(
                    "ClosingLine",
                    help="Optional: final market line (for CLV tracking).",
                    format="%.2f",
                ),
            },
            disabled=[
                c for c in editable.columns if c not in ["Result", "ClosingLine"]
            ],
            key="results_editor",
        )

        # Recompute CLV
        editable["CLV"] = editable.apply(compute_clv, axis=1)

        if not editable.equals(df):
            save_history(editable)
            st.success("✅ Saved updates to results & CLV.")
            df = editable

        # Metrics
        eval_df = df[df["Result"].isin(["Hit", "Miss", "Push"])].copy()
        if not eval_df.empty:
            st.markdown("### 📈 Performance Summary")

            total_bets = len(eval_df)
            pushes = (eval_df["Result"] == "Push").sum()
            settled = total_bets - pushes
            hits = (eval_df["Result"] == "Hit").sum()
            hit_rate = hits / settled if settled > 0 else 0.0

            eval_df = eval_df.sort_values("Timestamp")
            profits = eval_df.apply(compute_profit, axis=1).values
            bankroll_series = bankroll + np.cumsum(profits)
            total_profit = bankroll_series[-1] - bankroll
            roi = total_profit / bankroll if bankroll > 0 else 0.0

            st.markdown(f"- **Bets Tracked:** {total_bets}")
            st.markdown(f"- **Hit Rate (excl. Push):** {hit_rate * 100:.1f}%")
            st.markdown(f"- **Net Profit:** ${total_profit:.2f}")
            st.markdown(f"- **ROI:** {roi * 100:.1f}%")

            # CLV stats
            if "CLV" in eval_df.columns:
                clv_vals = pd.to_numeric(eval_df["CLV"], errors="coerce").dropna()
                if not clv_vals.empty:
                    pos_clv = (clv_vals > 0).mean()
                    avg_clv = clv_vals.mean()
                    st.markdown(f"- **Avg CLV:** {avg_clv:+.2f}")
                    st.markdown(f"- **% Bets Beating Close:** {pos_clv * 100:.1f}%")

            # Bankroll curve
            st.markdown("#### 📉 Bankroll Over Time")
            try:
                chart_df = pd.DataFrame({
                    "Timestamp": pd.to_datetime(eval_df["Timestamp"]),
                    "Bankroll": bankroll_series,
                }).set_index("Timestamp")
                st.line_chart(chart_df, height=220)
            except Exception:
                pass
        else:
            st.info("Mark some bets as Hit/Miss/Push above to unlock performance tracking.")

# =========================
# HISTORY TAB
# =========================

with tab_history:
    st.subheader("📜 Bet History & Filters")

    df = load_history()
    if df is None:
        st.info("No history yet. Run the model to start logging.")
    else:
        # Filters
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            days = st.selectbox(
                "Lookback Window",
                ["All", "7 days", "30 days", "90 days"],
                index=1,
            )
        with colf2:
            min_ev = st.number_input(
                "Min EV (raw, e.g. 0.05 = 5%)",
                value=0.0,
                step=0.01,
            )
        with colf3:
            market_filter = st.selectbox(
                "Market Filter",
                ["All", "PRA", "Points", "Rebounds", "Assists"],
                index=0,
            )

        name_search = st.text_input(
            "Filter by Player (optional)",
            value="",
        ).strip().lower()

        hist = df.copy()
        # Time filter
        if "Timestamp" in hist.columns and days != "All":
            try:
                hist["Timestamp"] = pd.to_datetime(hist["Timestamp"])
                now = datetime.now()
                if days == "7 days":
                    cutoff = now - pd.Timedelta(days=7)
                elif days == "30 days":
                    cutoff = now - pd.Timedelta(days=30)
                else:
                    cutoff = now - pd.Timedelta(days=90)
                hist = hist[hist["Timestamp"] >= cutoff]
            except Exception:
                pass

        # EV filter
        if "EV" in hist.columns:
            hist = hist[pd.to_numeric(hist["EV"], errors="coerce").fillna(0) >= min_ev]

        # Market filter
        if market_filter != "All":
            if "Market" in hist.columns:
                if market_filter == "PRA":
                    hist = hist[hist["Market"].str.contains("PRA", na=False)]
                else:
                    hist = hist[hist["Market"].str.contains(market_filter, na=False)]

        # Name search
        if name_search:
            if "Player" in hist.columns:
                hist = hist[hist["Player"].str.lower().str.contains(name_search, na=False)]

        st.markdown(f"**Filtered Bets:** {len(hist)}")

        # Summary
        if len(hist) > 0 and "EV" in hist.columns:
            avg_ev = pd.to_numeric(hist["EV"], errors="coerce").mean()
            st.markdown(f"- **Avg EV (filtered):** {avg_ev * 100:.1f}%")

        # Trendline by day (net profit)
        if len(hist) > 0:
            try:
                hist["Timestamp"] = pd.to_datetime(hist["Timestamp"])
                hist["Profit"] = hist.apply(compute_profit, axis=1)
                daily = hist.groupby(hist["Timestamp"].dt.date)["Profit"].sum().cumsum()
                daily_df = pd.DataFrame({
                    "Date": daily.index,
                    "Cumulative Profit": daily.values
                }).set_index("Date")
                st.markdown("#### 📈 Cumulative Profit (Filtered Window)")
                st.line_chart(daily_df, height=220)
            except Exception:
                pass

        st.markdown("#### 📋 Detailed Table")
        st.dataframe(hist, use_container_width=True)

        # Export
        if st.button("Export Filtered to CSV", use_container_width=True):
            export_name = f"bet_history_filtered_{safe_id}.csv"
            hist.to_csv(export_name, index=False)
            st.success(f"Saved filtered history to {export_name} in app directory.")

# =========================
# CALIBRATION & AUTO-TUNING TAB
# =========================

with tab_calib:
    st.subheader("🧠 Calibration & Auto-Tuning (Manual Apply)")

    df = load_history()
    if df is None:
        st.info("No data yet. Log some bets first.")
    else:
        calib = df.copy()
        calib = calib[calib["Result"].isin(["Hit", "Miss"])]
        calib["ProbOVER"] = pd.to_numeric(calib.get("ProbOVER", np.nan), errors="coerce")
        calib = calib.dropna(subset=["ProbOVER"])
        if len(calib) < 20:
            st.info("Need at least 20 settled bets with ProbOVER to run calibration.")
        else:
            st.markdown(f"**Sample Size:** {len(calib)} settled bets")

            # Calibration curve: bin by predicted prob
            calib["bin"] = (calib["ProbOVER"] * 10).astype(int) / 10.0
            grp = calib.groupby("bin")
            bin_pred = grp["ProbOVER"].mean()
            bin_actual = (grp["Result"].apply(lambda s: (s == "Hit").mean()))
            calib_df = pd.DataFrame({
                "Pred": bin_pred,
                "Actual": bin_actual
            }).dropna()

            if not calib_df.empty:
                st.markdown("#### 🎯 Calibration Curve (Predicted vs Actual)")
                st.line_chart(calib_df[["Pred", "Actual"]], height=220)

            # EV vs ROI
            profits = calib.apply(compute_profit, axis=1)
            roi = profits.sum() / bankroll if bankroll > 0 else 0.0
            avg_ev = calib["EV"].mean() if "EV" in calib.columns else np.nan

            st.markdown("#### 📊 Calibration Metrics")
            st.markdown(f"- **Avg Model Prob:** {calib['ProbOVER'].mean() * 100:.1f}%")
            st.markdown(
                f"- **Actual Hit Rate:** {(calib['Result'] == 'Hit').mean() * 100:.1f}%"
            )
            if not np.isnan(avg_ev):
                st.markdown(f"- **Avg EV (model):** {avg_ev * 100:.1f}%")
            st.markdown(f"- **ROI (based on logged stakes):** {roi * 100:.1f}%")

            # CLV insight
            if "CLV" in calib.columns:
                clv_vals = pd.to_numeric(calib["CLV"], errors="coerce").dropna()
                if not clv_vals.empty:
                    st.markdown(f"- **Avg CLV (all bets):** {clv_vals.mean():+.2f}")
                    st.markdown(
                        f"- **% Bets Beating Close:** {(clv_vals > 0).mean() * 100:.1f}%"
                    )

            # ===== Recommendations =====
            pred = calib["ProbOVER"].mean()
            actual = (calib["Result"] == "Hit").mean()
            conf_gap = actual - pred  # positive = model underconfident

            recs = []
            new_ht_pra = st.session_state["ht_pra"]
            new_ht_other = st.session_state["ht_other"]
            new_usage_min = st.session_state["usage_min"]
            new_usage_max = st.session_state["usage_max"]

            # Overconfident: predicted > actual
            if pred - actual > 0.07:
                recs.append("Model is overconfident. Increase variance slightly.")
                new_ht_pra = min(1.6, new_ht_pra + 0.05)
                new_ht_other = min(1.5, new_ht_other + 0.05)
            # Underconfident
            elif actual - pred > 0.07:
                recs.append("Model is too conservative. Decrease variance slightly.")
                new_ht_pra = max(1.15, new_ht_pra - 0.05)
                new_ht_other = max(1.10, new_ht_other - 0.05)
            else:
                recs.append("Overall calibration is within a reasonable band.")

            # ROI vs EV alignment
            if not np.isnan(avg_ev):
                if avg_ev > 0 and roi < 0:
                    recs.append(
                        "Positive EV but negative ROI → edges likely overstated. "
                        "Consider slightly widening usage or context variance."
                    )
                    new_usage_max = max(new_usage_min + 0.5, new_usage_max - 0.05)
                elif avg_ev < 0 and roi > 0:
                    recs.append(
                        "Negative EV but positive ROI → model too pessimistic. "
                        "Slightly relax variance or EV thresholds."
                    )

            st.markdown("#### 🧩 Tuning Recommendations")
            st.markdown(
                f"""
                <div style="background-color:{CARD_BG};padding:12px;border-radius:10px;border:1px solid {GOLD}55;">
                <ul>
                {''.join(f'<li>{r}</li>' for r in recs)}
                </ul>
                <p>Proposed session-only parameters:</p>
                <ul>
                  <li>Heavy-tail (PRA): <b>{st.session_state['ht_pra']:.2f}</b> → <b>{new_ht_pra:.2f}</b></li>
                  <li>Heavy-tail (Other): <b>{st.session_state['ht_other']:.2f}</b> → <b>{new_ht_other:.2f}</b></li>
                  <li>Usage clamp: <b>{st.session_state['usage_min']:.2f} - {st.session_state['usage_max']:.2f}</b> → <b>{new_usage_min:.2f} - {new_usage_max:.2f}</b></li>
                </ul>
                <p style="font-size:0.85rem;color:#aaa;">
                These changes only affect this session and keep projections anchored to live stats.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_apply, col_reset = st.columns(2)
            with col_apply:
                if st.button("✅ Apply Suggested Changes (This Session Only)", use_container_width=True):
                    st.session_state["ht_pra"] = new_ht_pra
                    st.session_state["ht_other"] = new_ht_other
                    st.session_state["usage_min"] = new_usage_min
                    st.session_state["usage_max"] = new_usage_max
                    st.success("Updated model parameters for this session. Re-run bets in the Model tab.")
            with col_reset:
                if st.button("♻️ Reset to Default Parameters", use_container_width=True):
                    st.session_state["ht_pra"] = 1.35
                    st.session_state["ht_other"] = 1.25
                    st.session_state["usage_min"] = 0.7
                    st.session_state["usage_max"] = 1.4
                    st.success("Reset to baseline settings for this session.")

# =========================
# FOOTER
# =========================

st.caption(
    "This engine will never be 100% — and that's the point. "
    "Edges come from disciplined +EV spots, context, and honest tracking, not guarantees."
)
