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

MAX_BANKROLL_PCT = 0.03  # 3% max stake
GSHEET_NAME = "NBA_Prop_Model_Log"
CSV_LOG = "bet_history.csv"

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
    /* Mobile responsiveness */
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
      Manual-line, data-backed 2-pick builder:
      <ul>
        <li>Recent game logs via <b>nba_api</b> (weighted recency)</li>
        <li>Supports <b>PRA, Points, Rebounds, Assists</b></li>
        <li>Heavy-tail volatility + auto same-team correlation</li>
        <li>Kelly-based stake sizing with safety caps</li>
        <li>History logging & tracking for calibration</li>
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
    help="Total bankroll. Stakes are suggested as a fraction of this."
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout Multiplier",
    min_value=1.01,
    value=st.session_state["payout_mult"],
    step=0.1,
    key="payout_mult",
    help="Total return on a winning 2-pick (e.g., 3.0x Power Play)."
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
        "1.0 = full Kelly (aggressive), 0.1–0.3 = safer.\n"
        "Each stake is also capped at 3% of bankroll."
    ),
)

games_lookback = st.sidebar.slider(
    "Recent Games (N)",
    5,
    20,
    value=st.session_state["games_lookback"],
    step=1,
    key="games_lookback",
    help="How many recent games to use for per-minute rates and minutes."
)

compact_mode = st.sidebar.checkbox(
    "Compact Mode (mobile-friendly)",
    value=st.session_state["compact_mode"],
    key="compact_mode",
    help="Hide some detailed fields for a cleaner mobile view."
)

st.sidebar.caption("Manual lines only. No odds API. Stable, transparent, sharp. 🧮")

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
    help="Pick the stat category you're modeling; input that line for each player."
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

HEAVY_TAIL_FACTOR = 1.2 if selected_market == "pra" else 1.1

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

@st.cache_data(show_spinner=False, ttl=600)
def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    cols = metric_map[market]
    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, None, f"Could not find player '{name}'.", None

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
    except Exception as e:
        return None, None, None, f"Error fetching logs for {label}: {e}", None

    if df.empty:
        return None, None, None, f"No logs found for {label}.", None

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    per_min_vals, minutes_list = [], []

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
        return None, None, None, f"Not enough valid recent games for {label}.", None

    per_min_arr = np.array(per_min_vals)
    mins_arr = np.array(minutes_list)

    # Weighted recency: older ~0.5, newest ~1.5
    weights = np.linspace(0.5, 1.5, len(per_min_arr))
    weights /= weights.sum()

    mu_per_min = float(np.average(per_min_arr, weights=weights))
    avg_min = float(np.average(mins_arr, weights=weights))

    sd_per_min = float(per_min_arr.std(ddof=1))
    if sd_per_min <= 0:
        sd_per_min = max(0.05, 0.1 * mu_per_min)

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
    return mu_per_min, sd_per_min, avg_min, msg, team_abbrev

def compute_leg(line, mu_per_min, sd_per_min, minutes,
                payout_mult, bankroll, kelly_frac, heavy_tail_factor):
    mu = mu_per_min * minutes
    base_sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    sd = max(1.0, base_sd * heavy_tail_factor)

    p_over = 1.0 - norm.cdf(line, mu, sd)

    b = payout_mult - 1.0
    ev_per_dollar = p_over * b - (1.0 - p_over)

    full_kelly = max(0.0, (b * p_over - (1.0 - p_over)) / b) if b > 0 else 0.0

    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_over, ev_per_dollar, full_kelly, stake, mu, sd

def adjust_joint_probability(p1_prob, p2_prob, corr):
    base = p1_prob * p2_prob
    adj = base + corr * (min(p1_prob, p2_prob) - base)
    return max(0.0, min(1.0, adj))

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

    col_inputs = st.columns(2)

    with col_inputs[0]:
        p1_name = st.text_input(
            "Player 1 Name",
            value=st.session_state["p1_name"],
            key="p1_name",
            help="Enter as listed in NBA box scores."
        )
        p1_line = st.number_input(
            "P1 Line (manual)",
            min_value=1.0,
            max_value=100.0,
            value=float(st.session_state["p1_line"]),
            step=0.5,
            key="p1_line",
            help="Enter the line for the selected market from your book."
        )

    with col_inputs[1]:
        p2_name = st.text_input(
            "Player 2 Name",
            value=st.session_state["p2_name"],
            key="p2_name",
            help="Enter as listed in NBA box scores."
        )
        p2_line = st.number_input(
            "P2 Line (manual)",
            min_value=1.0,
            max_value=100.0,
            value=float(st.session_state["p2_line"]),
            step=0.5,
            key="p2_line",
            help="Enter the line for the selected market from your book."
        )

    # Buttons row (Run + Quick Refresh)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_clicked = st.button("Run Model", use_container_width=True)
    with btn_col2:
        quick_refresh = st.button(
            "Quick Refresh Last Bet",
            use_container_width=True,
            help="Re-run using the last successful inputs."
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
        # Choose input source
        if use_last:
            params = st.session_state["last_run"]
            p1_name_i = params["p1_name"]
            p2_name_i = params["p2_name"]
            p1_line_i = params["p1_line"]
            p2_line_i = params["p2_line"]
            payout_i = params["payout_mult"]
            bank_i = params["bankroll"]
            kelly_i = params["fractional_kelly"]
            gl_i = params["games_lookback"]
        else:
            p1_name_i = p1_name
            p2_name_i = p2_name
            p1_line_i = p1_line
            p2_line_i = p2_line
            payout_i = payout_mult
            bank_i = bankroll
            kelly_i = fractional_kelly
            gl_i = games_lookback
            st.session_state["last_run"] = {
                "p1_name": p1_name_i,
                "p2_name": p2_name_i,
                "p1_line": float(p1_line_i),
                "p2_line": float(p2_line_i),
                "payout_mult": float(payout_i),
                "bankroll": float(bank_i),
                "fractional_kelly": float(kelly_i),
                "games_lookback": int(gl_i),
                "market_label": market_label,
                "selected_market": selected_market,
            }

        if payout_i <= 1.0:
            st.error("Payout multiplier must be > 1.0")
        else:
            # Fetch stats
            p1_mu_min, p1_sd_min, p1_avg_min, p1_msg, p1_team = get_player_rate_and_minutes(
                p1_name_i, gl_i, selected_market
            )
            if p1_mu_min is None:
                st.error(f"P1 stats error: {p1_msg}")
            else:
                p2_mu_min, p2_sd_min, p2_avg_min, p2_msg, p2_team = get_player_rate_and_minutes(
                    p2_name_i, gl_i, selected_market
                )
                if p2_mu_min is None:
                    st.error(f"P2 stats error: {p2_msg}")
                else:
                    # Compute legs
                    p1_prob, ev1, k1, stake1, p1_mu, p1_sd = compute_leg(
                        p1_line_i, p1_mu_min, p1_sd_min, p1_avg_min,
                        payout_i, bank_i, kelly_i, HEAVY_TAIL_FACTOR
                    )
                    p2_prob, ev2, k2, stake2, p2_mu, p2_sd = compute_leg(
                        p2_line_i, p2_mu_min, p2_sd_min, p2_avg_min,
                        payout_i, bank_i, kelly_i, HEAVY_TAIL_FACTOR
                    )

                    # Auto correlation: same team → +0.35
                    corr = 0.0
                    corr_reason = "0.00 (Independent)"
                    if p1_team and p2_team and p1_team == p2_team:
                        corr = 0.35
                        corr_reason = f"+0.35 (Same team: {p1_team})"

                    joint_prob = adjust_joint_probability(p1_prob, p2_prob, corr)

                    b_combo = payout_i - 1.0
                    combo_ev = payout_i * joint_prob - 1.0
                    combo_full_kelly = max(
                        0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo
                    ) if b_combo > 0 else 0.0

                    combo_stake = bank_i * kelly_i * combo_full_kelly
                    combo_stake = min(combo_stake, bank_i * MAX_BANKROLL_PCT)
                    combo_stake = max(0.0, round(combo_stake, 2))

                    # ---------- Single-Leg Results ----------
                    st.markdown("## 📊 Single-Leg Results")
                    col_a, col_b = st.columns(2)

                    # P1 card
                    with col_a:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        st.markdown(f"### {p1_name_i}")
                        st.caption(p1_msg)
                        if not compact_mode:
                            st.markdown(f"**Market:** {market_label}")
                        st.markdown(f"**Line:** {p1_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Auto Projected Minutes:** {p1_avg_min:.1f}")
                            st.markdown(f"**Model Mean:** {p1_mu:.1f}")
                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Model-estimated chance this leg goes over.'>ℹ️</span>: "
                            f"{p1_prob * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' style='width:{max(3,min(97,p1_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $** <span class='info-icon' title='Long-term return per $1. Higher EV = better value.'>ℹ️</span>: "
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

                    # P2 card
                    with col_b:
                        st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
                        st.markdown(f"### {p2_name_i}")
                        st.caption(p2_msg)
                        if not compact_mode:
                            st.markdown(f"**Market:** {market_label}")
                        st.markdown(f"**Line:** {p2_line_i}")
                        if not compact_mode:
                            st.markdown(f"**Auto Projected Minutes:** {p2_avg_min:.1f}")
                            st.markdown(f"**Model Mean:** {p2_mu:.1f}")
                        st.markdown(
                            f"**Prob OVER** <span class='info-icon' title='Model-estimated chance this leg goes over.'>ℹ️</span>: "
                            f"{p2_prob * 100:.1f}%",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='prob-bar-outer'><div class='prob-bar-inner' style='width:{max(3,min(97,p2_prob*100))}%;'></div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**EV per $** <span class='info-icon' title='Long-term return per $1 at these odds.'>ℹ️</span>: "
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

                    # ---------- Combo ----------
                    st.markdown("---")
                    st.markdown("## 🎯 2-Pick Combo (Both Must Hit)")
                    st.markdown(
                        f"**Correlation Applied** <span class='info-icon' title='Same team → positive correlation; otherwise independent.'>ℹ️</span>: "
                        f"{corr_reason}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Joint Prob** <span class='info-icon' title='Adjusted probability BOTH legs go over.'>ℹ️</span>: "
                        f"{joint_prob * 100:.1f}%",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**EV per $** <span class='info-icon' title='Expected long-term return for the combo.'>ℹ️</span>: "
                        f"{combo_ev * 100:.1f}%",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"**Suggested Combo Stake** <span class='info-icon' title='Kelly-based stake on combo, capped at 3% of bankroll.'>ℹ️</span>: "
                        f"${combo_stake:.2f}",
                        unsafe_allow_html=True,
                    )
                    if combo_ev > 0 and combo_stake > 0:
                        st.success("🔥 Combo is +EV under this model.")
                    else:
                        st.error("🚫 Combo is -EV. Don’t force action.")

                    # ---------- Best Bet ----------
                    st.markdown("---")
                    st.markdown("## 💬 Best Bet Summary")

                    if ev1 >= ev2:
                        best_player, best_line, best_ev, best_prob, best_stake = (
                            p1_name_i, p1_line_i, ev1, p1_prob, stake1
                        )
                    else:
                        best_player, best_line, best_ev, best_prob, best_stake = (
                            p2_name_i, p2_line_i, ev2, p2_prob, stake2
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
                            "No strong +EV single-leg edge detected. Passing is perfectly fine."
                        )

                    # ---------- Logging ----------
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    srcs = []
                    srcs.append(log_bet([
                        ts, p1_name_i, market_label, p1_line_i,
                        round(p1_mu, 2), round(p1_prob, 4),
                        round(ev1, 4), stake1,
                        "Single", "", ""
                    ]))
                    srcs.append(log_bet([
                        ts, p2_name_i, market_label, p2_line_i,
                        round(p2_mu, 2), round(p2_prob, 4),
                        round(ev2, 4), stake2,
                        "Single", "", ""
                    ]))
                    srcs.append(log_bet([
                        ts, f"{p1_name_i} + {p2_name_i}", market_label,
                        f"{p1_line_i} & {p2_line_i}",
                        "-", round(joint_prob, 4),
                        round(combo_ev, 4), combo_stake,
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
            "Logs go to Google Sheets (if configured) or to bet_history.csv."
        )
    else:
        st.markdown(
            f"Data source: **{'Google Sheet' if source == 'sheet' else 'Local CSV'}**"
        )

        # Sort by latest
        if "Timestamp" in df.columns:
            df = df.sort_values("Timestamp", ascending=False)

        st.dataframe(df.head(100), use_container_width=True)

        # Basic evaluation where Result is Hit/Miss (you fill this in manually post-game)
        if "Result" in df.columns:
            eval_df = df[df["Result"].isin(["Hit", "Miss"])].copy()
            if not eval_df.empty:
                singles = eval_df[eval_df["Type"].str.contains("Single", na=False)]
                if not singles.empty:
                    hits = (singles["Result"] == "Hit").sum()
                    total = len(singles)
                    hit_rate = hits / total if total > 0 else 0.0

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
                                "Model may be miscalibrated (hit rate vs predicted). "
                                "Consider slightly adjusting variance or required EV threshold."
                            )
                        else:
                            st.success(
                                "Model calibration vs results looks reasonable so far."
                            )
            else:
                st.info(
                    "You have logs but no Results set. "
                    "Mark 'Hit' or 'Miss' in your sheet/CSV to unlock analytics."
                )
        else:
            st.info(
                "No 'Result' column found. "
                "Add a 'Result' column (Hit/Miss) in your sheet/CSV to track outcomes."
            )

# =========================
# FOOTER
# =========================

st.caption(
    "Mobile tips: Add this page to your home screen. Your settings & inputs persist via session state. "
    "Use Quick Refresh before locking slips. Always layer in injuries, role, pace, and your own judgment."
)
