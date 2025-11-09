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

MAX_BANKROLL_PCT = 0.03  # 3% max stake per position
GSHEET_NAME = "NBA_Prop_Model_Log"
CSV_LOG = "bet_history.csv"

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
# TITLE
# =========================

st.markdown(
    f"""
    <h1>🏀 NBA 2-Pick Prop Edge & Risk Model</h1>
    <div class="divider-gold"></div>
    <p>
      Manual-line, data-driven 2-pick modeling:
      <ul>
        <li>Recent-game priors via <b>nba_api</b></li>
        <li>Markets: <b>PRA</b>, Points, Rebounds, Assists</li>
        <li>Weighted recency + heavier tails</li>
        <li>Auto same-team correlation for combos</li>
        <li>Kelly-based stake sizing</li>
        <li>History logging to Google Sheet or CSV</li>
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
    help="Total bankroll you're managing. Stakes are sized off this."
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout Multiplier",
    min_value=1.01,
    value=3.0,
    step=0.1,
    help="Total return on a winning 2-pick (e.g., 3.0x for PrizePicks Power Play)."
)

fractional_kelly = st.sidebar.slider(
    "Fractional Kelly",
    0.0,
    1.0,
    0.25,
    0.05,
    help=(
        "Kelly = optimal % of bankroll with an edge.\n"
        "1.0 = full (aggressive), 0.1–0.3 = safer.\n"
        "Each stake also capped at 3% of bankroll."
    ),
)

games_lookback = st.sidebar.slider(
    "Recent Games (N)",
    5,
    20,
    10,
    1,
    help="Number of recent games used for per-minute production & minutes."
)

st.sidebar.caption("Manual lines only. Stable, transparent, model-based. 🧮")

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
    help="Choose which stat you're modeling; enter that line from your book."
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

@st.cache_data(show_spinner=False)
def nba_lookup_player(name: str):
    players = nba_players.get_players()
    target = _norm_name(name)

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
      mu_per_min, sd_per_min, avg_minutes, msg, team_abbrev
    Uses weighted recency.
    """
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
        return None, None, None, f"Not enough valid recent games for {label}.", None

    per_min_arr = np.array(per_min_vals)
    mins_arr = np.array(minutes_list)

    # Weighted recency: older games ~0.5, newest ~1.5
    weights = np.linspace(0.5, 1.5, len(per_min_arr))
    weights = weights / weights.sum()

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

# =========================
# LOGGING HELPERS
# =========================

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
    """
    Try Google Sheet, else fallback to local CSV.
    """
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

    col1, col2 = st.columns(2)

    with col1:
        p1_name = st.text_input(
            "Player 1 Name",
            "RJ Barrett",
            help="Enter as listed in NBA box scores."
        )
        p1_line = st.number_input(
            "P1 Line (manual)",
            min_value=1.0,
            max_value=100.0,
            value=33.5,
            step=0.5,
            help="Enter line from PrizePicks or any book for the selected market."
        )

    with col2:
        p2_name = st.text_input(
            "Player 2 Name",
            "Jaylen Brown",
            help="Enter as listed in NBA box scores."
        )
        p2_line = st.number_input(
            "P2 Line (manual)",
            min_value=1.0,
            max_value=100.0,
            value=34.5,
            step=0.5,
            help="Enter line from your book."
        )

    run = st.button("Run Model")

    if run:
        if payout_mult <= 1.0:
            st.error("Payout multiplier must be > 1.0")
            st.stop()

        # Stats
        p1_mu_min, p1_sd_min, p1_avg_min, p1_msg, p1_team = get_player_rate_and_minutes(
            p1_name, games_lookback, selected_market
        )
        if p1_mu_min is None:
            st.error(f"P1 stats error: {p1_msg}")
            st.stop()

        p2_mu_min, p2_sd_min, p2_avg_min, p2_msg, p2_team = get_player_rate_and_minutes(
            p2_name, games_lookback, selected_market
        )
        if p2_mu_min is None:
            st.error(f"P2 stats error: {p2_msg}")
            st.stop()

        # Legs
        p1_prob, ev1, k1, stake1, p1_mu, p1_sd = compute_leg(
            p1_line, p1_mu_min, p1_sd_min, p1_avg_min,
            payout_mult, bankroll, fractional_kelly, HEAVY_TAIL_FACTOR
        )
        p2_prob, ev2, k2, stake2, p2_mu, p2_sd = compute_leg(
            p2_line, p2_mu_min, p2_sd_min, p2_avg_min,
            payout_mult, bankroll, fractional_kelly, HEAVY_TAIL_FACTOR
        )

        # Auto correlation
        corr = 0.0
        corr_reason = "0.00 (Assumed independent)"
        if p1_team and p2_team and p1_team == p2_team:
            corr = 0.35
            corr_reason = f"+0.35 (Same team: {p1_team})"

        joint_prob = adjust_joint_probability(p1_prob, p2_prob, corr)

        # Combo EV
        b_combo = payout_mult - 1.0
        combo_ev = payout_mult * joint_prob - 1.0
        combo_full_kelly = max(
            0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo
        ) if b_combo > 0 else 0.0

        combo_stake = bankroll * fractional_kelly * combo_full_kelly
        combo_stake = min(combo_stake, bankroll * MAX_BANKROLL_PCT)
        combo_stake = max(0.0, round(combo_stake, 2))

        # -------- Single-Leg Results --------
        st.markdown("## 📊 Single-Leg Results")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
            st.markdown(f"### {p1_name}")
            st.caption(p1_msg)
            st.markdown(f"**Market:** {market_label}")
            st.markdown(f"**Line:** {p1_line}")
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
                f"**EV per $** <span class='info-icon' title='Long-term profit per $1. Higher EV = better value; not just hit rate.'>ℹ️</span>: "
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
            st.markdown("<div class='prop-card'>", unsafe_allow_html=True)
            st.markdown(f"### {p2_name}")
            st.caption(p2_msg)
            st.markdown(f"**Market:** {market_label}")
            st.markdown(f"**Line:** {p2_line}")
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
                f"**EV per $** <span class='info-icon' title='Long-term profit per $1 at given edge & payout.'>ℹ️</span>: "
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

        # -------- Combo Results --------
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
            st.error("🚫 Combo is -EV. Don’t force it.")

        # -------- Best Bet Summary --------
        st.markdown("---")
        st.markdown("## 💬 Best Bet Summary")

        if ev1 >= ev2:
            best_player, best_line, best_ev, best_prob, best_stake = (
                p1_name, p1_line, ev1, p1_prob, stake1
            )
        else:
            best_player, best_line, best_ev, best_prob, best_stake = (
                p2_name, p2_line, ev2, p2_prob, stake2
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
                "No strong +EV single-leg edge detected. Passing is a winning strategy."
            )

        # -------- Log Bets (Sheet or CSV) --------
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        src1 = log_bet([
            ts, p1_name, market_label, p1_line,
            round(p1_mu, 2), round(p1_prob, 4),
            round(ev1, 4), stake1,
            "Single", "", ""  # Extra, Result
        ])
        src2 = log_bet([
            ts, p2_name, market_label, p2_line,
            round(p2_mu, 2), round(p2_prob, 4),
            round(ev2, 4), stake2,
            "Single", "", ""
        ])
        src3 = log_bet([
            ts, f"{p1_name} + {p2_name}", market_label,
            f"{p1_line} & {p2_line}",
            "-", round(joint_prob, 4),
            round(combo_ev, 4), combo_stake,
            f"Combo (corr={corr_reason})", "", ""
        ])

        if src1 == "sheet" or src2 == "sheet" or src3 == "sheet":
            st.info("📊 Logged this run to Google Sheet (NBA_Prop_Model_Log).")
        else:
            st.info("💾 Logged this run locally to bet_history.csv (in app directory).")

# =========================
# RESULTS / TRACKING TAB
# =========================

with tab_results:
    st.subheader("📓 Results & Tracking")

    df, source = load_history()

    if df is None or df.empty:
        st.info(
            "No logged history yet. Run the model in the 'Model' tab to start logging.\n"
            "Logs go to Google Sheets if configured, otherwise to bet_history.csv."
        )
    else:
        st.markdown(
            f"Data source: **{'Google Sheet' if source == 'sheet' else 'Local CSV'}**"
        )

        # Show latest 50 rows (most recent first)
        df_view = df.copy()
        if "Timestamp" in df_view.columns:
            df_view = df_view.sort_values("Timestamp", ascending=False)
        st.dataframe(df_view.head(50), use_container_width=True)

        # Basic metrics (only where Result is recorded as 'Hit' or 'Miss')
        if "Result" in df_view.columns:
            eval_df = df_view[df_view["Result"].isin(["Hit", "Miss"])].copy()
            if not eval_df.empty:
                # Treat Type == 'Single' only for hit rate on legs
                single_df = eval_df[eval_df["Type"] == "Single"]
                if not single_df.empty:
                    hits = (single_df["Result"] == "Hit").sum()
                    total = len(single_df)
                    hit_rate = hits / total if total > 0 else 0.0

                    st.markdown("### ✅ Performance Summary (Singles)")
                    st.markdown(f"**Recorded Bets:** {total}")
                    st.markdown(f"**Hit Rate:** {hit_rate * 100:.1f}%")

                    # Compare to average model prob
                    if "ProbOVER" in single_df.columns:
                        avg_model_prob = single_df["ProbOVER"].mean()
                        st.markdown(
                            f"**Avg Model Prob (at time of bet):** {avg_model_prob * 100:.1f}%"
                        )

                        # Calibration hint
                        if abs(hit_rate - avg_model_prob) > 0.05:
                            st.warning(
                                "Model may be miscalibrated (hit rate vs predicted).\n"
                                "Consider adjusting variance or edge thresholds."
                            )
                        else:
                            st.success(
                                "Model calibration looks reasonable vs actual results so far."
                            )
            else:
                st.info(
                    "You have logs but no Results marked yet.\n"
                    "Add 'Hit' / 'Miss' in the 'Result' column (in your sheet or CSV) to unlock calibration."
                )
        else:
            st.info(
                "Your log does not yet include a 'Result' column.\n"
                "Add it to your Google Sheet or CSV to track Hit/Miss outcomes."
            )

# FOOTER
st.caption(
    "Workflow: Enter lines → Run Model → Model logs to sheet/CSV → "
    "You later mark Hit/Miss in the log → Track calibration & refine edges over time."
)
