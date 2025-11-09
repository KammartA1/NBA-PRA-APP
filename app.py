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
# CONFIG & CONSTANTS
# =========================

st.set_page_config(
    page_title="NBA 2-Pick Prop Edge Model",
    page_icon="🏀",
    layout="wide",
)

PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
DARK_BG = "#0C0B10"
CARD_BG = "#17131C"

MAX_BANKROLL_PCT = 0.03  # 3% cap on any one stake

# =========================
# GLOBAL STYLE (Gophers aesthetic)
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
        padding: 18px 16px 14px 16px;
        border: 1px solid {GOLD}22;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        margin-bottom: 14px;
    }}
    .divider-gold {{
        height: 2px;
        background: linear-gradient(90deg, transparent, {GOLD}, transparent);
        margin: 12px 0 16px 0;
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
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TITLE / DESCRIPTION
# =========================

st.markdown(
    f"""
    <h1>🏀 NBA 2-Pick Prop Edge & Risk Model</h1>
    <div class="divider-gold"></div>
    <p>
      Enter live lines from <b>PrizePicks</b> or any book.<br>
      This tool:
      <ul>
        <li>Uses recent NBA logs (<b>nba_api</b>) to estimate per-minute production & minutes</li>
        <li>Supports <b>PRA, Points, Rebounds, Assists</b></li>
        <li>Computes win probability, <b>EV per $</b>, and <b>Kelly-based stake</b></li>
        <li>Evaluates the 2-pick combo and logs every run to Google Sheets</li>
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
    value=1000.0,
    step=10.0,
    help="Total bankroll you are managing. Stakes are sized as a fraction of this."
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout Multiplier",
    min_value=1.01,
    value=3.0,
    step=0.1,
    help="Total return on a winning 2-pick Power Play (e.g., 3.0x)."
)

fractional_kelly = st.sidebar.slider(
    "Fractional Kelly",
    0.0,
    1.0,
    0.25,
    0.05,
    help=(
        "Kelly Criterion = optimal fraction of bankroll to wager when you have an edge.\n"
        "1.0 = Full Kelly (aggressive), 0.1–0.3 = common safer range.\n"
        "We also cap stakes at 3% of bankroll."
    ),
)

games_lookback = st.sidebar.slider(
    "Recent Games (N)",
    5,
    20,
    10,
    1,
    help="How many recent games to use for per-minute production and minutes."
)

st.sidebar.caption("Manual-line only. No sportsbook API dependencies. Stable & sharp. 🧮")

# =========================
# MARKET SELECTOR
# =========================

market_label = st.selectbox(
    "Select Prop Market",
    [
        "PRA (Points + Rebounds + Assists)",
        "Points",
        "Rebounds",
        "Assists",
    ],
    index=0,
    help="Pick the stat type you're modeling. Then enter that stat's line for each player."
)

market_key_map = {
    "PRA (Points + Rebounds + Assists)": "pra",
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
}
selected_market = market_key_map[market_label]

metric_map = {
    "pra": ["PTS", "REB", "AST"],
    "pts": ["PTS"],
    "reb": ["REB"],
    "ast": ["AST"],
}

# =========================
# PLAYER INPUTS
# =========================

st.subheader("🎯 Player Inputs")

col1, col2 = st.columns(2)

with col1:
    p1_name = st.text_input(
        "Player 1 Name",
        "RJ Barrett",
        help="Type the player's name as listed in NBA box scores. We'll auto-pull recent stats."
    )
    p1_line = st.number_input(
        "P1 Line (manual)",
        min_value=1.0,
        max_value=100.0,
        value=33.5,
        step=0.5,
        help="Enter the current line from PrizePicks or any sportsbook for the chosen market."
    )

with col2:
    p2_name = st.text_input(
        "Player 2 Name",
        "Jaylen Brown",
        help="Type the player's name as listed in NBA box scores."
    )
    p2_line = st.number_input(
        "P2 Line (manual)",
        min_value=1.0,
        max_value=100.0,
        value=34.5,
        step=0.5,
        help="Enter the current line from PrizePicks or any sportsbook for the chosen market."
    )

run = st.button("Run Model")

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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    cols = metric_map[market]
    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, None, f"Could not find player '{name}'."

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
    except Exception as e:
        return None, None, None, f"Error fetching logs for {label}: {e}"

    if df.empty:
        return None, None, None, f"No logs found for {label}."

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    per_min_vals = []
    minutes_list = []

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

        if minutes > 0:
            minutes_list.append(minutes)
            per_min_vals.append(total_val / minutes)

    if len(per_min_vals) < 3 or not minutes_list:
        return None, None, None, f"Not enough valid recent games for {label}."

    per_min_arr = np.array(per_min_vals)
    mu_per_min = float(per_min_arr.mean())
    sd_per_min = float(per_min_arr.std(ddof=1))
    if sd_per_min <= 0:
        sd_per_min = max(0.05, 0.1 * mu_per_min)

    avg_min = float(np.mean(minutes_list))
    msg = (
        f"{label}: {len(per_min_vals)} recent games, "
        f"avg minutes {avg_min:.1f}"
    )
    return mu_per_min, sd_per_min, avg_min, msg

def compute_leg(line, mu_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    mu = mu_per_min * minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    if sd <= 0:
        sd = max(1.0, 0.15 * max(mu, 1.0))

    p_over = 1.0 - norm.cdf(line, mu, sd)

    b = payout_mult - 1.0
    ev_per_dollar = p_over * b - (1.0 - p_over)
    full_kelly = max(0.0, (b * p_over - (1.0 - p_over)) / b) if b > 0 else 0.0

    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_over, ev_per_dollar, full_kelly, stake, mu, sd

# =========================
# GOOGLE SHEETS LOGGING
# =========================

GSHEET_NAME = "NBA_Prop_Model_Log"

@st.cache_resource(show_spinner=False)
def connect_to_gsheet():
    """
    Connect to Google Sheet using local service_account.json.
    Returns sheet object or None if it fails.
    """
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
        sheet = sh.sheet1
        return sheet
    except Exception:
        return None

def log_bet_row(sheet, row):
    try:
        sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception:
        pass  # fail silently; don't break the app


# =========================
# MAIN RUN
# =========================

if run:
    if payout_mult <= 1.0:
        st.error("Payout multiplier must be > 1.0")
        st.stop()

    # Pull model inputs from nba_api
    p1_mu_min, p1_sd_min, p1_avg_min, p1_msg = get_player_rate_and_minutes(
        p1_name, games_lookback, selected_market
    )
    if p1_mu_min is None:
        st.error(f"P1 stats error: {p1_msg}")
        st.stop()

    p2_mu_min, p2_sd_min, p2_avg_min, p2_msg = get_player_rate_and_minutes(
        p2_name, games_lookback, selected_market
    )
    if p2_mu_min is None:
        st.error(f"P2 stats error: {p2_msg}")
        st.stop()

    # Compute legs
    p1_prob, ev1, k1, stake1, p1_mu, p1_sd = compute_leg(
        p1_line, p1_mu_min, p1_sd_min, p1_avg_min, payout_mult, bankroll, fractional_kelly
    )
    p2_prob, ev2, k2, stake2, p2_mu, p2_sd = compute_leg(
        p2_line, p2_mu_min, p2_sd_min, p2_avg_min, payout_mult, bankroll, fractional_kelly
    )

    # Combo
    joint_prob = p1_prob * p2_prob
    b_combo = payout_mult - 1.0
    combo_ev = payout_mult * joint_prob - 1.0
    combo_full_kelly = max(
        0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo
    ) if b_combo > 0 else 0.0

    combo_stake = bankroll * fractional_kelly * combo_full_kelly
    combo_stake = min(combo_stake, bankroll * MAX_BANKROLL_PCT)
    combo_stake = max(0.0, round(combo_stake, 2))

    # =========================
    # DISPLAY: SINGLE LEGS
    # =========================

    st.markdown("## 📊 Single-Leg Results")

    col_a, col_b = st.columns(2)

    # Player 1 card
    with col_a:
        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
        st.markdown(f"### {p1_name}")
        st.caption(p1_msg)
        st.markdown(f"**Market:** {market_label}")
        st.markdown(f"**Line:** {p1_line}")
        st.markdown(f"**Auto Projected Minutes:** {p1_avg_min:.1f}")
        st.markdown(f"**Model Mean:** {p1_mu:.1f}")

        # Prob bar
        st.markdown(
            f"**Prob OVER** "
            f"<span class='info-icon' title='Model-estimated chance this leg goes over the line.'>ℹ️</span>: "
            f"{p1_prob * 100:.1f}%",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='prob-bar-outer'><div class='prob-bar-inner' style='width:{max(3,min(97,p1_prob*100))}%;'></div></div>",
            unsafe_allow_html=True,
        )

        # EV
        st.markdown(
            f"**EV per $** "
            f"<span class='info-icon' title='Expected long-term return per $1. Higher EV = better value, not always higher hit rate.'>ℹ️</span>: "
            f"{ev1 * 100:.1f}%",
            unsafe_allow_html=True,
        )

        # Stake
        st.markdown(
            f"**Suggested Stake** "
            f"<span class='info-icon' title='Kelly-based stake using your Fractional Kelly, capped at 3% of bankroll.'>ℹ️</span>: "
            f"${stake1:.2f}",
            unsafe_allow_html=True,
        )

        if ev1 > 0 and stake1 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")

        st.markdown("</div>", unsafe_allow_html=True)

    # Player 2 card
    with col_b:
        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
        st.markdown(f"### {p2_name}")
        st.caption(p2_msg)
        st.markdown(f"**Market:** {market_label}")
        st.markdown(f"**Line:** {p2_line}")
        st.markdown(f"**Auto Projected Minutes:** {p2_avg_min:.1f}")
        st.markdown(f"**Model Mean:** {p2_mu:.1f}")

        st.markdown(
            f"**Prob OVER** "
            f"<span class='info-icon' title='Model-estimated chance this leg goes over the line.'>ℹ️</span>: "
            f"{p2_prob * 100:.1f}%",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='prob-bar-outer'><div class='prob-bar-inner' style='width:{max(3,min(97,p2_prob*100))}%;'></div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**EV per $** "
            f"<span class='info-icon' title='Expected long-term return per $1. Compares your edge vs the payout.'>ℹ️</span>: "
            f"{ev2 * 100:.1f}%",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"**Suggested Stake** "
            f"<span class='info-icon' title='Kelly-based stake using your Fractional Kelly, capped at 3% of bankroll.'>ℹ️</span>: "
            f"${stake2:.2f}",
            unsafe_allow_html=True,
        )

        if ev2 > 0 and stake2 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # DISPLAY: COMBO
    # =========================

    st.markdown("---")
    st.markdown("## 🎯 2-Pick Combo (Both Must Hit)")

    st.markdown(
        f"**Joint Prob** "
        f"<span class='info-icon' title='Probability that BOTH legs go over, assuming independence.'>ℹ️</span>: "
        f"{joint_prob * 100:.1f}%",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**EV per $** "
        f"<span class='info-icon' title='Expected long-term return for the 2-pick as a whole.'>ℹ️</span>: "
        f"{combo_ev * 100:.1f}%",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Suggested Combo Stake** "
        f"<span class='info-icon' title='Fractional Kelly stake on the combo, capped at 3% of bankroll.'>ℹ️</span>: "
        f"${combo_stake:.2f}",
        unsafe_allow_html=True,
    )

    if combo_ev > 0 and combo_stake > 0:
        st.success("🔥 Combo is +EV under this model.")
    else:
        st.error("🚫 Combo is -EV. Only fire if you have other reasons to like it.")

    # =========================
    # BEST BET SUMMARY
    # =========================

    st.markdown("---")
    st.markdown("## 💬 Best Bet Summary")

    if ev1 >= ev2:
        best_player, best_line, best_ev, best_prob, best_stake = (
            p1_name,
            p1_line,
            ev1,
            p1_prob,
            stake1,
        )
    else:
        best_player, best_line, best_ev, best_prob, best_stake = (
            p2_name,
            p2_line,
            ev2,
            p2_prob,
            stake2,
        )

    if best_ev > 0 and best_stake > 0:
        st.success(
            f"**Best Single-Leg Edge:** {best_player} OVER {best_line}  \n"
            f"Win Probability: **{best_prob * 100:.1f}%**  \n"
            f"EV per $: **{best_ev * 100:.1f}%**  \n"
            f"Suggested Stake: **${best_stake:.2f}**"
        )
    else:
        st.warning(
            "No strong +EV single-leg edge detected. "
            "Be selective. Passing is a winning strategy too."
        )

    # =========================
    # GOOGLE SHEETS LOGGING
    # =========================

    sheet = connect_to_gsheet()
    if sheet:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # log P1
        log_bet_row(sheet, [
            ts, p1_name, market_label, p1_line,
            round(p1_mu, 2), round(p1_prob, 4),
            round(ev1, 4), stake1,
            "Single"
        ])
        # log P2
        log_bet_row(sheet, [
            ts, p2_name, market_label, p2_line,
            round(p2_mu, 2), round(p2_prob, 4),
            round(ev2, 4), stake2,
            "Single"
        ])
        # log Combo
        log_bet_row(sheet, [
            ts, f"{p1_name} + {p2_name}", market_label, f"{p1_line} & {p2_line}",
            "-", round(joint_prob, 4),
            round(combo_ev, 4), combo_stake,
            "Combo"
        ])

        st.info("📊 Logged this run to Google Sheet: NBA_Prop_Model_Log")
    else:
        st.warning(
            "Could not connect to Google Sheets. "
            "Ensure 'service_account.json' is present and the sheet is shared with that service account."
        )

# FOOTER
st.caption(
    "This tool is for informational/educational use. "
    "Always layer in injuries, usage, pace, matchup, and your own judgment. "
    "Long-term success = edges + discipline."
)
