import os
import csv
import difflib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog
from scipy.stats import norm

import gspread
from oauth2client.service_account import ServiceAccountCredentials

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
GSHEET_NAME = "NBA_Prop_Model_Log"
CSV_LOG = "bet_history.csv"

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

# Usage baseline assumptions (approximate)
LEAGUE_USAGE_PCT = 20.0
BASE_USAGE_PER_MIN = 0.39  # approx FGA+0.44*FTA+TOV per minute for 20% usage

# =========================
# SESSION STATE DEFAULTS
# =========================

defaults = {
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
# GLOBAL STYLE (Gophers + Mobile Responsive)
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
    <h1>🏀 NBA 2-Pick Prop Edge & Risk Model</h1>
    <div class="divider-gold"></div>
    <p>
      Built for manual PrizePicks-style entries:
      <ul>
        <li>Per-player markets: <b>PRA, Points, Rebounds, Assists</b></li>
        <li>Usage-adjusted per-minute projections</li>
        <li>Weighted recency + heavy-tail variance for realistic probabilities</li>
        <li>Auto same-team correlation on 2-pick combos</li>
        <li>Kelly-based staking, logging & tracking</li>
      </ul>
    </p>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR SETTINGS
# =========================

st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input(
    "Bankroll ($)",
    min_value=10.0,
    value=st.session_state["bankroll"],
    step=10.0,
    key="bankroll",
    help="Total bankroll. Stakes are fractions of this."
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout Multiplier",
    min_value=1.01,
    value=st.session_state["payout_mult"],
    step=0.1,
    key="payout_mult",
    help="Total payout on a winning 2-pick (e.g. 3.0x Power Play)."
)

fractional_kelly = st.sidebar.slider(
    "Fractional Kelly",
    0.0,
    1.0,
    value=st.session_state["fractional_kelly"],
    step=0.05,
    key="fractional_kelly",
    help=(
        "Kelly = optimal % of bankroll when you have an edge.\n"
        "Use 0.1–0.3 for safer growth. Each stake is capped at 3% of bankroll."
    ),
)

games_lookback = st.sidebar.slider(
    "Recent Games (N)",
    5,
    20,
    value=st.session_state["games_lookback"],
    step=1,
    key="games_lookback",
    help="How many recent games to use for rates, minutes & usage."
)

compact_mode = st.sidebar.checkbox(
    "Compact Mode (mobile-friendly)",
    value=st.session_state["compact_mode"],
    key="compact_mode",
    help="Hide some details for a cleaner mobile layout."
)

st.sidebar.caption("Manual lines only. No odds API. Stable, transparent, sharp. 🧮")

# =========================
# HELPER FUNCTIONS
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
    # Standard NBA headshot pattern
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

@st.cache_data(show_spinner=False, ttl=600)
def get_player_rate_and_minutes(name: str, n_games: int, market_key: str):
    """
    Returns:
      mu_per_min, sd_per_min, avg_minutes, msg, team_abbrev, usage_pct, player_id
    - mu_per_min is for the requested market_key
    - usage_pct is an approximate usage rate based on FGA/FTA/TOV
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

    per_min_vals = []
    minutes_list = []
    usg_like_vals = []

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

        # Usage proxy: (FGA + 0.44*FTA + TOV) per minute
        fga = float(row.get("FGA", 0))
        fta = float(row.get("FTA", 0))
        tov = float(row.get("TOV", 0))
        usg_like = (fga + 0.44 * fta + tov) / minutes
        usg_like_vals.append(usg_like)

    if len(per_min_vals) < 3 or not minutes_list:
        return None, None, None, f"Not enough valid recent games for {label}.", None, None, pid

    per_min_arr = np.array(per_min_vals)
    mins_arr = np.array(minutes_list)

    # Weighted recency for per-minute and minutes
    weights = np.linspace(0.5, 1.5, len(per_min_arr))
    weights /= weights.sum()

    mu_per_min = float(np.average(per_min_arr, weights=weights))
    avg_min = float(np.average(mins_arr, weights=weights))

    sd_per_min = float(per_min_arr.std(ddof=1))
    if sd_per_min <= 0:
        sd_per_min = max(0.05, 0.1 * max(mu_per_min, 0.5))

    # Usage rate estimate
    if usg_like_vals:
        usg_arr = np.array(usg_like_vals)
        usg_per_min = float(np.average(usg_arr, weights=weights))
        # Map to usage% relative to baseline
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
):
    """
    Builds a usage-adjusted, heavy-tailed normal model for the chosen stat.
    Returns:
      p_over (clamped realistic),
      ev_raw_per_dollar,
      ev_display_per_dollar,
      full_kelly_fraction,
      suggested_stake,
      mu,
      sd
    """
    # Usage adjustment: scale per-minute rate
    usage_factor = 1.0 + (usage_pct - LEAGUE_USAGE_PCT) / 100.0
    usage_factor = float(np.clip(usage_factor, 0.7, 1.4))

    mu_per_min_adj = mu_per_min * usage_factor
    mu = mu_per_min_adj * minutes

    # Base SD from sample, then inflate for heavy tails & high usage volatility
    base_sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    volatility_factor = 1.0 + (abs(usage_pct - LEAGUE_USAGE_PCT) / 100.0) * 0.3
    sd = max(1.0, base_sd * heavy_tail_factor * volatility_factor)

    # Probability of going over
    p_over = 1.0 - norm.cdf(line, mu, sd)
    # Clamp extremes to avoid fake 99.9% edges
    p_over = float(np.clip(p_over, 0.05, 0.95))

    # EV per $ (raw, theoretical)
    b = payout_mult - 1.0
    ev_raw = p_over * b - (1.0 - p_over)

    # Kelly fraction from raw probabilities
    full_kelly = max(0.0, (b * p_over - (1.0 - p_over)) / b) if b > 0 else 0.0

    # Suggested stake using fractional Kelly, with bankroll cap
    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    # Display EV compressed so it's realistic & readable
    ev_display = float(np.tanh(ev_raw * 0.5))

    return p_over, ev_raw, ev_display, full_kelly, stake, mu, sd

def adjust_joint_probability(p1_prob: float, p2_prob: float, corr: float):
    """
    Adjust joint probability:
      base = P1 * P2
      P12 = base + corr * (min(P1,P2) - base)
    """
    base = p1_prob * p2_prob
    adj = base + corr * (min(p1_prob, p2_prob) - base)
    return float(np.clip(adj, 0.0, 1.0))

@st.cache_resource(show_spinner=False)
def connect_to_gsheet():
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "service_account.json", scope
        )
        client = gspread.authorize(creds)
        sh = client.open(GSHEET_NAME)
        return sh.sheet1
    except Exception:
        return None

def append_to_csv(row):
    header = [
        "Timestamp", "Player", "Market", "Line",
        "ModelMean", "ProbOVER", "EV", "Stake",
        "Type", "Extra", "Result"
    ]
    file_exists = os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)

def log_bet(row):
    sheet = connect_to_gsheet()
    if sheet:
        try:
            sheet.append_row(row, value_input_option="USER_ENTERED")
            return "sheet"
        except Exception:
            append_to_csv(row)
            return "csv"
    else:
        append_to_csv(row)
        return "csv"

def load_history():
    sheet = connect_to_gsheet()
    if sheet:
        try:
            records = sheet.get_all_records()
            if records:
                return pd.DataFrame(records), "sheet"
        except Exception:
            pass
    if os.path.exists(CSV_LOG):
        try:
            df = pd.read_csv(CSV_LOG)
            return df, "csv"
        except Exception:
            return None, None
    return None, None

# =========================
# TABS
# =========================

tab_model, tab_results = st.tabs(["📊 Model", "📓 Results / Tracking"])

# =========================
# MODEL TAB
# =========================

with tab_model:
    st.subheader("🎯 Player Inputs")

    col_left, col_right = st.columns(2)

    with col_left:
        p1_name = st.text_input(
            "Player 1 Name",
            value=st.session_state["p1_name"],
            key="p1_name",
            help="Enter as listed in NBA box scores."
        )
        p1_market_label = st.selectbox(
            "P1 Market",
            MARKET_OPTIONS,
            index=MARKET_OPTIONS.index(st.session_state["p1_market_label"]),
            key="p1_market_label",
            help="Select which stat line you're modeling for Player 1."
        )
        p1_line = st.number_input(
            "P1 Line (manual)",
            min_value=1.0,
            max_value=100.0,
            value=float(st.session_state["p1_line"]),
            step=0.5,
            key="p1_line",
            help="Enter the current line from PrizePicks or any book."
        )

    with col_right:
        p2_name = st.text_input(
            "Player 2 Name",
            value=st.session_state["p2_name"],
            key="p2_name",
            help="Enter as listed in NBA box scores."
        )
        p2_market_label = st.selectbox(
            "P2 Market",
            MARKET_OPTIONS,
            index=MARKET_OPTIONS.index(st.session_state["p2_market_label"]),
            key="p2_market_label",
            help="Select which stat line you're modeling for Player 2."
        )
        p2_line = st.number_input(
            "P2 Line (manual)",
            min_value=1.0,
            max_value=100.0,
            value=float(st.session_state["p2_line"]),
            step=0.5,
            key="p2_line",
            help="Enter the current line from PrizePicks or any book."
        )

    # Buttons row
    b1, b2 = st.columns(2)
    with b1:
        run_clicked = st.button("Run Model", use_container_width=True)
    with b2:
        quick_refresh = st.button(
            "Quick Refresh Last Bet",
            use_container_width=True,
            help="Re-run using last successful inputs."
        )

    trigger = False
    use_last = False
    if run_clicked:
        trigger = True
        use_last = False
    elif quick_refresh and st.session_state["last_run"]:
        trigger = True
        use_last = True

    if trigger:
        if use_last:
            params = st.session_state["last_run"]
            p1_name_i = params["p1_name"]
            p2_name_i = params["p2_name"]
            p1_line_i = params["p1_line"]
            p2_line_i = params["p2_line"]
            p1_market_label_i = params["p1_market_label"]
            p2_market_label_i = params["p2_market_label"]
            payout_i = params["payout_mult"]
            bank_i = params["bankroll"]
            kelly_i = params["fractional_kelly"]
            gl_i = params["games_lookback"]
        else:
            p1_name_i = p1_name
            p2_name_i = p2_name
            p1_line_i = p1_line
            p2_line_i = p2_line
            p1_market_label_i = p1_market_label
            p2_market_label_i = p2_market_label
            payout_i = payout_mult
            bank_i = bankroll
            kelly_i = fractional_kelly
            gl_i = games_lookback

            st.session_state["last_run"] = {
                "p1_name": p1_name_i,
                "p2_name": p2_name_i,
                "p1_line": float(p1_line_i),
                "p2_line": float(p2_line_i),
                "p1_market_label": p1_market_label_i,
                "p2_market_label": p2_market_label_i,
                "payout_mult": float(payout_i),
                "bankroll": float(bank_i),
                "fractional_kelly": float(kelly_i),
                "games_lookback": int(gl_i),
            }

        if payout_i <= 1.0:
            st.error("Payout multiplier must be > 1.0")
        else:
            # Resolve market keys
            p1_market_key = market_key_map[p1_market_label_i]
            p2_market_key = market_key_map[p2_market_label_i]

            # Heavy-tail by market
            p1_ht = 1.35 if p1_market_key == "pra" else 1.25
            p2_ht = 1.35 if p2_market_key == "pra" else 1.25

            # Stats P1
            p1_mu_min, p1_sd_min, p1_avg_min, p1_msg, p1_team, p1_usg, p1_pid = \
                get_player_rate_and_minutes(p1_name_i, gl_i, p1_market_key)
            if p1_mu_min is None:
                st.error(f"P1 stats error: {p1_msg}")
            else:
                # Stats P2
                p2_mu_min, p2_sd_min, p2_avg_min, p2_msg, p2_team, p2_usg, p2_pid = \
                    get_player_rate_and_minutes(p2_name_i, gl_i, p2_market_key)
                if p2_mu_min is None:
                    st.error(f"P2 stats error: {p2_msg}")
                else:
                    # Compute legs
                    p1_prob, p1_ev_raw, p1_ev_disp, p1_kelly, p1_stake, p1_mu, p1_sd = compute_leg(
                        p1_line_i, p1_mu_min, p1_sd_min, p1_avg_min,
                        p1_usg, payout_i, bank_i, kelly_i, p1_ht
                    )
                    p2_prob, p2_ev_raw, p2_ev_disp, p2_kelly, p2_stake, p2_mu, p2_sd = compute_leg(
                        p2_line_i, p2_mu_min, p2_sd_min, p2_avg_min,
                        p2_usg, payout_i, bank_i, kelly_i, p2_ht
                    )

                    # Auto correlation: same team only
                    corr = 0.0
                    corr_reason = "0.00 (Independent)"
                    if p1_team and p2_team and p1_team == p2_team:
                        corr = 0.35
                        corr_reason = f"+0.35 (Same team: {p1_team})"

                    joint_prob = adjust_joint_probability(p1_prob, p2_prob, corr)

                    # Combo EV
                    b_combo = payout_i - 1.0
                    combo_ev_raw = payout_i * joint_prob - 1.0
                    combo_ev_disp = float(np.tanh(combo_ev_raw * 0.5))

                    combo_full_kelly = max(
                        0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo
                    ) if b_combo > 0 else 0.0

                    combo_stake = bank_i * kelly_i * combo_full_kelly
                    combo_stake = min(combo_stake, bank_i * MAX_BANKROLL_PCT)
                    combo_stake = max(0.0, round(combo_stake, 2))

                    # =========================
                    # SINGLE-LEG RESULTS
                    # =========================
                    st.markdown("## 📊 Single-Leg Results")
                    col_a, col_b = st.columns(2)

                    # P1 Card
                    with col_a:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        ch1, ch2 = st.columns([1, 4])
                        with ch1:
                            url = headshot_url(p1_pid)
                            if url:
                                st.image(url, width=60)
                        with ch2:
                            st.markdown(f"### {p1_name_i}")
                            st.caption(p1_msg)

                        st.markdown(f"**Market:** {p1_market_label_i}")
                        st.markdown(f"**Line:** {p1_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Usage Rate (weighted):** {p1_usg:.1f}%")
                            st.markdown(f"**Model Mean:** {p1_mu:.1f}")
                            st.markdown(f"**Proj Minutes:** {p1_avg_min:.1f}")

                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Usage & volatility-adjusted probability this leg goes over.'>ℹ️</span>: "
                            f"{p1_prob * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' style='width:{max(4,min(96,p1_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $** <span class='info-icon' title='Display-smoothed EV. True EV is used for staking; this is a realistic edge signal.'>ℹ️</span>: "
                            f"{p1_ev_disp * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Suggested Stake** <span class='info-icon' title='Fractional Kelly on true edge, capped at 3% of bankroll.'>ℹ️</span>: "
                            f"${p1_stake:.2f}",
                            unsafe_allow_html=True,
                        )
                        if p1_ev_raw > 0 and p1_stake > 0:
                            st.success("✅ +EV leg (model edge detected)")
                        else:
                            st.error("❌ -EV leg (no long-term edge)")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # P2 Card
                    with col_b:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        ch1, ch2 = st.columns([1, 4])
                        with ch1:
                            url = headshot_url(p2_pid)
                            if url:
                                st.image(url, width=60)
                        with ch2:
                            st.markdown(f"### {p2_name_i}")
                            st.caption(p2_msg)

                        st.markdown(f"**Market:** {p2_market_label_i}")
                        st.markdown(f"**Line:** {p2_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Usage Rate (weighted):** {p2_usg:.1f}%")
                            st.markdown(f"**Model Mean:** {p2_mu:.1f}")
                            st.markdown(f"**Proj Minutes:** {p2_avg_min:.1f}")

                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Usage & volatility-adjusted probability this leg goes over.'>ℹ️</span>: "
                            f"{p2_prob * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' style='width:{max(4,min(96,p2_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $** <span class='info-icon' title='Smoothed EV display; staking logic uses raw edge.'>ℹ️</span>: "
                            f"{p2_ev_disp * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Suggested Stake** <span class='info-icon' title='Fractional Kelly on true edge, capped at 3% of bankroll.'>ℹ️</span>: "
                            f"${p2_stake:.2f}",
                            unsafe_allow_html=True,
                        )
                        if p2_ev_raw > 0 and p2_stake > 0:
                            st.success("✅ +EV leg (model edge detected)")
                        else:
                            st.error("❌ -EV leg (no long-term edge)")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # =========================
                    # COMBO RESULTS
                    # =========================

                    st.markdown("---")
                    st.markdown("## 🎯 2-Pick Combo (Both Must Hit)")

                    st.markdown(
                        f"**Correlation Applied** <span class='info-icon' title='Same-team legs assumed positively correlated; others independent.'>ℹ️</span>: "
                        f"{corr_reason}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Joint Prob** <span class='info-icon' title='Adjusted probability BOTH legs go over.'>ℹ️</span>: "
                        f"{joint_prob * 100:.1f}%",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**EV per $ (display)** <span class='info-icon' title='Smoothed EV for the combo. True EV used for staking and logging.'>ℹ️</span>: "
                        f"{combo_ev_disp * 100:.1f}%",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Suggested Combo Stake** <span class='info-icon' title='Fractional Kelly on true combo edge, capped at 3% of bankroll.'>ℹ️</span>: "
                        f"${combo_stake:.2f}",
                        unsafe_allow_html=True,
                    )
                    if combo_ev_raw > 0 and combo_stake > 0:
                        st.success("🔥 Combo is +EV under this model.")
                    else:
                        st.error("🚫 Combo is -EV. Only play with extra conviction.")

                    # =========================
                    # BEST BET SUMMARY
                    # =========================

                    st.markdown("---")
                    st.markdown("## 💬 Best Bet Summary")

                    # Pick best by raw EV
                    best = max(
                        [
                            ("P1", p1_name_i, p1_market_label_i, p1_line_i, p1_ev_raw, p1_prob, p1_stake),
                            ("P2", p2_name_i, p2_market_label_i, p2_line_i, p2_ev_raw, p2_prob, p2_stake),
                        ],
                        key=lambda x: x[4],
                    )

                    _, bp_name, bp_mkt, bp_line, bp_ev_raw, bp_prob, bp_stake = best

                    if bp_ev_raw > 0 and bp_stake > 0:
                        st.success(
                            f"**Best Single-Leg Edge:** {bp_name} OVER {bp_line} ({bp_mkt})  \n"
                            f"Win Probability: **{bp_prob * 100:.1f}%**  \n"
                            f"True EV per $: **{bp_ev_raw * 100:.1f}%**  \n"
                            f"Suggested Stake: **${bp_stake:.2f}**"
                        )
                    else:
                        st.warning(
                            "No strong +EV single-leg edge detected. Discipline = edge. Wait for better numbers."
                        )

                    # =========================
                    # LOGGING (Sheet or CSV)
                    # =========================

                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    srcs = []
                    srcs.append(log_bet([
                        ts, p1_name_i, p1_market_label_i, p1_line_i,
                        round(p1_mu, 2), round(p1_prob, 4),
                        round(p1_ev_raw, 4), p1_stake,
                        "Single", f"Usage {p1_usg:.1f}%", ""
                    ]))
                    srcs.append(log_bet([
                        ts, p2_name_i, p2_market_label_i, p2_line_i,
                        round(p2_mu, 2), round(p2_prob, 4),
                        round(p2_ev_raw, 4), p2_stake,
                        "Single", f"Usage {p2_usg:.1f}%", ""
                    ]))
                    srcs.append(log_bet([
                        ts, f"{p1_name_i} + {p2_name_i}",
                        f"{p1_market_label_i} & {p2_market_label_i}",
                        f"{p1_line_i} & {p2_line_i}",
                        "-", round(joint_prob, 4),
                        round(combo_ev_raw, 4), combo_stake,
                        f"Combo (corr={corr_reason})", "", ""
                    ]))

                    if "sheet" in srcs:
                        st.info("📊 Logged this run to Google Sheet (NBA_Prop_Model_Log).")
                    else:
                        st.info("💾 Logged this run locally to bet_history.csv.")

# =========================
# RESULTS / TRACKING TAB
# =========================

with tab_results:
    st.subheader("📓 Results & Tracking")

    df, source = load_history()

    if df is None or df.empty:
        st.info(
            "No logged history yet. Run the model in the 'Model' tab to start logging.\n"
            "Logs go to Google Sheets (if configured) or bet_history.csv."
        )
    else:
        st.markdown(
            f"Data source: **{'Google Sheet' if source == 'sheet' else 'Local CSV'}**"
        )
        if "Timestamp" in df.columns:
            df = df.sort_values("Timestamp", ascending=False)
        st.dataframe(df.head(150), use_container_width=True)

        if "Result" in df.columns:
            eval_df = df[df["Result"].isin(["Hit", "Miss"])].copy()
            if not eval_df.empty:
                singles = eval_df[eval_df["Type"].str.contains("Single", na=False)]
                if not singles.empty:
                    hits = (singles["Result"] == "Hit").sum()
                    total = len(singles)
                    hit_rate = hits / total if total else 0.0

                    st.markdown("### ✅ Performance Summary (Singles)")
                    st.markdown(f"**Recorded Bets:** {total}")
                    st.markdown(f"**Hit Rate:** {hit_rate * 100:.1f}%")

                    if "ProbOVER" in singles.columns:
                        avg_model_prob = singles["ProbOVER"].mean()
                        st.markdown(
                            f"**Avg Model Prob at Bet Time:** {avg_model_prob * 100:.1f}%"
                        )
                        if abs(hit_rate - avg_model_prob) > 0.05:
                            st.warning(
                                "Model may be a bit off calibration (hit rate vs predicted).\n"
                                "Consider modest tweaks to variance or required EV threshold."
                            )
                        else:
                            st.success(
                                "Model calibration vs results looks reasonably in line so far."
                            )
            else:
                st.info(
                    "Logs found but no 'Hit/Miss' yet. Add results in your sheet/CSV to unlock accuracy tracking."
                )
        else:
            st.info(
                "No 'Result' column detected.\n"
                "Add a 'Result' column (Hit/Miss) in your log to track outcomes."
            )

# =========================
# FOOTER
# =========================

st.caption(
    "Workflow: Enter lines → Run Model → Use +EV & Kelly sizing → Log auto-saves → "
    "After games, mark Hit/Miss in your log → Use Results tab to calibrate and refine. "
    "Remember: the edge is in disciplined volume on +EV spots, not forcing action."
)

