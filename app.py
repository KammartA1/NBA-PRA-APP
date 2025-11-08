import difflib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog
from scipy.stats import norm

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="NBA 2-Pick Prop Edge Model",
    page_icon="🏀",
    layout="wide",
)

# =========================
# CUSTOM STYLING (Gophers theme)
# =========================
PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
DARK_BG = "#1A0C10"

st.markdown(
    f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: {DARK_BG};
        color: #FFFFFF;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY_MAROON}, #32000F);
        color: #FFFFFF;
        border-right: 1px solid {GOLD}33;
    }}

    /* Headers */
    h1, h2, h3, h4 {{
        color: {GOLD};
        font-weight: 700;
    }}

    /* Metric cards and boxes */
    .stMetric, .stAlert {{
        color: #FFFFFF !important;
    }}

    /* Cards / containers */
    .prop-card {{
        padding: 18px 16px;
        border-radius: 14px;
        border: 1px solid {GOLD}33;
        background: rgba(122, 0, 25, 0.16);
        margin-bottom: 10px;
    }}

    /* Tooltips using title attr */
    .info-icon {{
        display: inline-block;
        margin-left: 4px;
        color: {GOLD};
        cursor: help;
        font-weight: 700;
    }}

    /* Inputs background tweak */
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stSlider > div {{
        background-color: #2B151C !important;
        color: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# TITLE & DESCRIPTION
# =========================
st.markdown(
    f"""
    <h1>🏀 NBA 2-Pick Prop Edge & Risk Model</h1>
    <p>
    Built for <b>PrizePicks-style</b> 2-pick cards.<br>
    Enter lines manually from any app. The model:
    <ul>
      <li>Uses recent NBA logs (<b>nba_api</b>) to estimate efficiency & minutes</li>
      <li>Supports <b>PRA, Points, Rebounds, Assists</b></li>
      <li>Computes win probability, EV, and <b>Kelly-based stake sizing</b></li>
    </ul>
    </p>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR: GLOBAL SETTINGS
# =========================
st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input(
    "Bankroll ($)",
    min_value=10.0,
    value=1000.0,
    step=10.0,
    help="Total bankroll you are managing. All suggested stakes are percentages of this."
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout Multiplier",
    min_value=1.01,
    value=3.0,
    step=0.1,
    help="Total return on a winning 2-pick (e.g. 3.0x for PrizePicks Power Play)."
)

fractional_kelly = st.sidebar.slider(
    "Fractional Kelly",
    0.0,
    1.0,
    0.25,
    0.05,
    help=(
        "Kelly Criterion controls how much of your bankroll to risk when you have an edge.\n"
        "- 1.0 = Full Kelly (aggressive)\n"
        "- 0.1–0.3 = Common range (safer)\n"
        "- 0.0 = No staking advice"
    ),
)

games_lookback = st.sidebar.slider(
    "Recent Games Used (N)",
    5,
    20,
    10,
    1,
    help="Number of recent games used to estimate per-minute production & minutes."
)

MAX_BANKROLL_PCT = 0.03  # cap any one stake at 3% of bankroll

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
    help="Choose which stat line you're modeling. Enter the corresponding line from PrizePicks or any book."
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
st.subheader("Player Inputs")

c1, c2 = st.columns(2)

with c1:
    p1_name = st.text_input(
        "Player 1 Name",
        "RJ Barrett",
        help="Enter the player's name as listed in NBA box scores."
    )
    p1_line = st.number_input(
        "P1 Line (manual)",
        min_value=1.0,
        max_value=100.0,
        value=33.5,
        step=0.5,
        help="Enter the prop line from PrizePicks or any sportsbook for the selected market."
    )

with c2:
    p2_name = st.text_input(
        "Player 2 Name",
        "Jaylen Brown",
        help="Enter the player's name as listed in NBA box scores."
    )
    p2_line = st.number_input(
        "P2 Line (manual)",
        min_value=1.0,
        max_value=100.0,
        value=34.5,
        step=0.5,
        help="Enter the prop line from PrizePicks or any sportsbook for the selected market."
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

    # exact
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]

    # fuzzy
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
    """
    Returns:
      (mu_per_min, sd_per_min, avg_minutes, message)
    for selected market based on last N regular-season games.
    """
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
        f"{label}: using last {len(per_min_vals)} games "
        f"({current_season()}), avg minutes {avg_min:.1f}"
    )
    return mu_per_min, sd_per_min, avg_min, msg

def compute_leg(line, mu_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    """
    Build normal distribution from per-minute rates:
      X ~ N(mu, sd^2), where:
        mu = mu_per_min * minutes
        sd = sd_per_min * sqrt(minutes)

    Returns:
      p_over, ev_per_dollar, full_kelly_fraction, stake, mu, sd
    """
    mu = mu_per_min * minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    if sd <= 0:
        sd = max(1.0, 0.15 * max(mu, 1.0))

    # Probability of going over the line
    p_over = 1.0 - norm.cdf(line, mu, sd)

    # Kelly-style EV for a bet with net odds b = payout_mult - 1
    b = payout_mult - 1.0
    ev_per_dollar = p_over * b - (1.0 - p_over)

    full_kelly = max(0.0, (b * p_over - (1.0 - p_over)) / b) if b > 0 else 0.0

    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_over, ev_per_dollar, full_kelly, stake, mu, sd

# =========================
# MAIN RUN
# =========================
if run:
    if payout_mult <= 1.0:
        st.error("Payout multiplier must be > 1.0")
        st.stop()

    # Player 1
    p1_mu_min, p1_sd_min, p1_avg_min, p1_msg = get_player_rate_and_minutes(
        p1_name, games_lookback, selected_market
    )
    if p1_mu_min is None:
        st.error(f"P1 stats error: {p1_msg}")
        st.stop()

    # Player 2
    p2_mu_min, p2_sd_min, p2_avg_min, p2_msg = get_player_rate_and_minutes(
        p2_name, games_lookback, selected_market
    )
    if p2_mu_min is None:
        st.error(f"P2 stats error: {p2_msg}")
        st.stop()

    # Single-leg calculations
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
    # SINGLE-LEG RESULTS
    # =========================
    st.markdown("## 📊 Single-Leg Results")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"<div class='prop-card'><h3>{p1_name}</h3>", unsafe_allow_html=True)
        st.caption(p1_msg)
        st.markdown(f"**Line:** {p1_line}")
        st.markdown(f"**Auto Projected Minutes:** {p1_avg_min:.1f}")
        st.markdown(f"**Model Mean ({market_label}):** {p1_mu:.1f}")
        st.markdown(
            f"**Prob OVER** <span class='info-icon' title='Model-estimated chance this leg goes over the line.'>ℹ️</span>: "
            f"{p1_prob * 100:.1f}%",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**EV per $** <span class='info-icon' title='Expected long-term return per $1 staked. Higher EV = better value, not always higher hit rate.'>ℹ️</span>: "
            f"{ev1 * 100:.1f}%",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Suggested Stake** <span class='info-icon' title='Fractional Kelly stake, capped at 3% of bankroll.'>ℹ️</span>: "
            f"${stake1:.2f}",
            unsafe_allow_html=True,
        )
        if ev1 > 0 and stake1 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"<div class='prop-card'><h3>{p2_name}</h3>", unsafe_allow_html=True)
        st.caption(p2_msg)
        st.markdown(f"**Line:** {p2_line}")
        st.markdown(f"**Auto Projected Minutes:** {p2_avg_min:.1f}")
        st.markdown(f"**Model Mean ({market_label}):** {p2_mu:.1f}")
        st.markdown(
            f"**Prob OVER** <span class='info-icon' title='Model-estimated chance this leg goes over the line.'>ℹ️</span>: "
            f"{p2_prob * 100:.1f}%",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**EV per $** <span class='info-icon' title='Expected long-term return per $1 staked. Higher EV = better value, not always higher hit rate.'>ℹ️</span>: "
            f"{ev2 * 100:.1f}%",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Suggested Stake** <span class='info-icon' title='Fractional Kelly stake, capped at 3% of bankroll.'>ℹ️</span>: "
            f"${stake2:.2f}",
            unsafe_allow_html=True,
        )
        if ev2 > 0 and stake2 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================
    # COMBO RESULTS
    # =========================
    st.markdown("---")
    st.markdown("## 🎯 2-Pick Combo (Both Must Hit)")

    st.markdown(
        f"**Joint Prob** <span class='info-icon' title='Probability that BOTH players go over, assuming independence.'>ℹ️</span>: "
        f"{joint_prob * 100:.1f}%",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**EV per $** <span class='info-icon' title='Expected long-term return on the 2-pick as a whole.'>ℹ️</span>: "
        f"{combo_ev * 100:.1f}%",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Suggested Combo Stake** <span class='info-icon' title='Fractional Kelly on the combo, capped at 3% of bankroll.'>ℹ️</span>: "
        f"${combo_stake:.2f}",
        unsafe_allow_html=True,
    )

    if combo_ev > 0 and combo_stake > 0:
        st.success("🔥 Combo is +EV under this model.")
    else:
        st.error("🚫 Combo is -EV. Consider passing.")

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
            "Stay disciplined and wait for better numbers."
        )

# FOOTER
st.caption(
    "Usage: Enter live lines from PrizePicks or any sportsbook. "
    "Model uses recent performance + minutes to estimate edge. "
    "This is NOT a guarantee — always consider news, injuries, and your own risk tolerance."
)
