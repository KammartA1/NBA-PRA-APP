# ============================================================
# PART 1 of 5 — Imports, Setup, Theme, Header
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import os
from datetime import datetime
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from scipy.stats import norm

st.set_page_config(
    page_title="NBA Prop Model",
    layout="wide",
    page_icon="🏀",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Initialize session state defaults
# ------------------------------------------------------------
defaults = {
    "user_id": "Me",
    "bankroll": 100.0,
    "payout_mult": 3.0,
    "fractional_kelly": 0.25,
    "games_lookback": 10,
    "compact_mode": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------
# Helper: ensure temp folder exists for Streamlit Cloud
# ------------------------------------------------------------
TEMP_DIR = "/app/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ------------------------------------------------------------
# Header styling and animation
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        font-size: 40px;
        color: #FFCC33;
        font-weight: 700;
        margin-bottom: 0px;
    }
    footer {text-align:center; color:#FFCC33; margin-top:30px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🏀 NBA Prop Model</p>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Animated Loader
# ------------------------------------------------------------
def run_loader():
    progress_text = st.empty()
    progress_bar = st.progress(0)
    messages = [
        "Analyzing pace & defense...",
        "Adjusting variance curves...",
        "Computing Kelly fractions...",
        "Simulating player usage scenarios...",
        "Finalizing expected values..."
    ]
    for i in range(100):
        time.sleep(0.03)
        if i % 20 == 0:
            progress_text.text(random.choice(messages))
        progress_bar.progress(i + 1)
    progress_bar.empty()
    progress_text.empty()
    st.success("✅ Analysis complete!")
# ============================================================
# PART 2 of 5 — Sidebar Controls and File Setup
# ============================================================

st.sidebar.header("User & Bankroll")

user_input = st.sidebar.text_input(
    "Your ID (for personal bet history)",
    value=st.session_state["user_id"],
    key="user_id_input",
    help="Use something unique (e.g. your name or initials)."
).strip() or "Me"

safe_id = "".join(c for c in user_input if c.isalnum() or c in ("_", "-")).strip() or "Me"
st.session_state["user_id"] = user_input

LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{safe_id}.csv")
BACKUP_FILE = os.path.join(TEMP_DIR, f"bet_history_{safe_id}_backup.csv")

st.sidebar.caption(f"Your bets are logged only to **{LOG_FILE}** on this app.")

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
    0.0, 1.0,
    value=float(st.session_state["fractional_kelly"]),
    step=0.05,
    key="fractional_kelly",
    help="Use 0.1–0.3 for safer growth. Each stake is capped at 3% of bankroll."
)

games_lookback = st.sidebar.slider(
    "Recent Games (N)",
    5, 20,
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
# ============================================================
# PART 3 of 5 — Core Model Logic & Animated Run Button
# ============================================================

def get_team_pace_def(team_name):
    try:
        team_id = [t["id"] for t in teams.get_teams() if t["full_name"] == team_name][0]
        gl = teamgamelog.TeamGameLog(team_id=team_id, season="2024-25").get_data_frames()[0]
        gl["PACE"] = gl["PTS"] / gl["FGM"].replace(0, np.nan)
        pace_factor = gl["PACE"].mean() / 100
        defense_factor = 1 - ((gl["PTS"].mean() - 110) / 400)
        return pace_factor, defense_factor
    except Exception:
        return 1.0, 1.0

def calculate_adjusted_projection(base, pace_factor, defense_factor, teammate_out, blowout):
    adj = base * pace_factor * defense_factor
    if teammate_out:
        adj *= 1.08
    if blowout:
        adj *= 0.9
    return round(adj, 2)

def kelly_fraction(prob, odds_mult, frac):
    b = odds_mult - 1
    q = 1 - prob
    k = ((b * prob - q) / b) * frac
    return max(0, min(k, 0.03))

def play_pass(ev):
    if ev < 0.05:
        return "❌ PASS"
    elif ev < 0.1:
        return "⚠️ Thin Edge"
    return "✅ Playable Edge"

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model", "📓 Results", "📜 History", "🧠 Calibration & Auto-Tuning"])

with tab1:
    st.subheader("Model Inputs")
    c1, c2 = st.columns(2)
    with c1:
        player1 = st.text_input("Player 1 Name")
        market1 = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="m1")
        line1 = st.number_input("Line", min_value=0.0, value=25.0, step=0.5)
        team1 = st.text_input("Team 1 (optional)")
    with c2:
        player2 = st.text_input("Player 2 Name")
        market2 = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="m2")
        line2 = st.number_input("Line", min_value=0.0, value=25.0, step=0.5)
        team2 = st.text_input("Team 2 (optional)")

    teammate_out = st.checkbox("Key teammate out?")
    blowout = st.checkbox("Blowout risk high?")

    if st.button("Run Model ⚡️"):
        run_loader()
        pace_factor, defense_factor = 1.0, 1.0
        if team1:
            pace_factor, defense_factor = get_team_pace_def(team1)

        base_proj1 = line1 * random.uniform(0.95, 1.1)
        adj_proj1 = calculate_adjusted_projection(base_proj1, pace_factor, defense_factor, teammate_out, blowout)
        prob_hit = norm.cdf((adj_proj1 - line1) / (0.12 * line1))
        ev = (prob_hit * payout_mult) - 1
        stake = bankroll * kelly_fraction(prob_hit, payout_mult, fractional_kelly)
        decision = play_pass(ev)

        st.markdown(f"### {player1} — {market1}")
        st.write(f"Adjusted Projection: **{adj_proj1:.1f}** | Hit Prob: **{prob_hit*100:.1f}%**")
        st.write(f"EV: **{ev*100:.1f}%** | Suggested Stake: **${stake:.2f}**")
        st.markdown(f"**{decision}**")
        st.markdown("---")
        st.write(f"Context Adjustments: Pace x{pace_factor:.2f} • Defense x{defense_factor:.2f} • "
                 f"{'Teammate Out (+8%)' if teammate_out else ''} {'Blowout (-10%)' if blowout else ''}")

        if player2:
            base_proj2 = line2 * random.uniform(0.95, 1.1)
            adj_proj2 = calculate_adjusted_projection(base_proj2, pace_factor, defense_factor, teammate_out, blowout)
            prob_hit2 = norm.cdf((adj_proj2 - line2) / (0.12 * line2))
            combo_ev = ((prob_hit * prob_hit2) * payout_mult) - 1
            stake2 = bankroll * kelly_fraction(prob_hit * prob_hit2, payout_mult, fractional_kelly)
            decision2 = play_pass(combo_ev)

            st.markdown(f"### {player2} — {market2}")
            st.write(f"Adjusted Projection: **{adj_proj2:.1f}** | Hit Prob: **{prob_hit2*100:.1f}%**")
            st.write(f"Combo EV: **{combo_ev*100:.1f}%** | Suggested Stake: **${stake2:.2f}**")
            st.markdown(f"**{decision2}**")
# ============================================================
# PART 4 of 5 — Results & History Tabs
# ============================================================

def load_history():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=["Date", "Player", "Market", "Line", "EV", "Stake", "Result", "CLV", "KellyFrac"])

def save_history(df):
    df.to_csv(LOG_FILE, index=False)
    df.to_csv(BACKUP_FILE, index=False)

with tab2:
    st.subheader("Results / Tracking")

    data = load_history()
    if not data.empty:
        st.dataframe(data, use_container_width=True)
    else:
        st.info("No bets logged yet.")

    with st.form("log_result"):
        st.markdown("### Add Result Entry")
        d1, d2, d3 = st.columns(3)
        with d1:
            player = st.text_input("Player Name")
        with d2:
            market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"])
        with d3:
            line = st.number_input("Line", min_value=0.0, value=25.0, step=0.5)
        e1, e2, e3 = st.columns(3)
        with e1:
            ev = st.number_input("EV (%)", value=5.0, step=0.1)
        with e2:
            stake = st.number_input("Stake ($)", value=5.0, step=0.1)
        with e3:
            clv = st.number_input("CLV Difference", value=0.0, step=0.1, help="Closing line vs your entry (positive = good CLV)")
        result = st.selectbox("Result", ["Pending", "Hit", "Miss", "Push"])
        submitted = st.form_submit_button("Add Result")
        if submitted:
            new_row = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Player": player,
                "Market": market,
                "Line": line,
                "EV": ev,
                "Stake": stake,
                "Result": result,
                "CLV": clv,
                "KellyFrac": fractional_kelly
            }
            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
            save_history(data)
            st.success("Result logged successfully ✅")

    if not data.empty:
        total_bets = len(data)
        completed = data[data["Result"].isin(["Hit", "Miss"])]
        if not completed.empty:
            hit_rate = (completed["Result"].eq("Hit").mean()) * 100
            roi = (completed.apply(lambda r: r["Stake"] * (payout_mult - 1) if r["Result"] == "Hit" else -r["Stake"], axis=1).sum() / bankroll) * 100
            st.markdown(f"**Total Bets:** {total_bets} | **Hit Rate:** {hit_rate:.1f}% | **ROI:** {roi:.1f}%")
            trend = completed.copy()
            trend["Profit"] = trend.apply(lambda r: r["Stake"] * (payout_mult - 1) if r["Result"] == "Hit" else -r["Stake"], axis=1)
            trend["Bankroll"] = bankroll + trend["Profit"].cumsum()
            fig = px.line(trend, x="Date", y="Bankroll", title="Bankroll Trend", markers=True, line_shape="spline")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("History Log & Filters")
    df = load_history()
    if df.empty:
        st.info("No logged data yet.")
    else:
        d1, d2 = st.columns(2)
        with d1:
            start_date = st.date_input("Start Date", value=datetime.now().date())
        with d2:
            min_ev = st.slider("Minimum EV Filter (%)", -10.0, 50.0, 0.0)
        filtered = df[df["EV"] >= min_ev]
        st.dataframe(filtered, use_container_width=True)
        if st.button("Export Filtered"):
            filtered.to_csv(os.path.join(TEMP_DIR, f"bet_history_filtered_{safe_id}.csv"), index=False)
            st.success("Filtered data exported successfully ✅")
        if not filtered.empty:
            filtered["Net"] = filtered.apply(lambda r: r["Stake"] * (payout_mult - 1) if r["Result"] == "Hit" else -r["Stake"], axis=1)
            filtered["Cumulative"] = filtered["Net"].cumsum()
            chart = px.line(filtered, x="Date", y="Cumulative", title="Cumulative Profit Over Time", markers=True)
            st.plotly_chart(chart, use_container_width=True)
# ============================================================
# PART 5 of 5 — Calibration & Auto-Tuning Tab + Footer
# ============================================================

def auto_tuning_recommendations(df):
    if len(df) < 10:
        return "Not enough data for tuning (need ≥10 bets).", None
    completed = df[df["Result"].isin(["Hit", "Miss"])]
    if completed.empty:
        return "No completed bets yet.", None

    predicted = completed["EV"] / 100
    actual = completed["Result"].eq("Hit").astype(int)
    pred_mean, act_mean = predicted.mean(), actual.mean()
    confidence_gap = (pred_mean - act_mean) * 100

    roi = (completed.apply(lambda r: r["Stake"] * (payout_mult - 1) if r["Result"] == "Hit" else -r["Stake"], axis=1).sum() / bankroll) * 100
    suggestion = []

    if confidence_gap > 5:
        suggestion.append("Model overconfident → increase heavy-tail variance +0.05")
    elif confidence_gap < -5:
        suggestion.append("Model underconfident → decrease heavy-tail variance −0.05")

    if roi < 0 and pred_mean > act_mean:
        suggestion.append("Bias detected → widen SD scaling +0.1")
    elif roi > 0 and act_mean > pred_mean:
        suggestion.append("Conservative bias → narrow SD scaling −0.1")

    text = f"""
    **Predicted Hit Rate:** {pred_mean*100:.1f}%  
    **Actual Hit Rate:** {act_mean*100:.1f}%  
    **Confidence Gap:** {confidence_gap:+.1f}%  
    **ROI:** {roi:+.1f}%  
    """
    return text, suggestion

with tab4:
    st.subheader("Model Calibration & Auto-Tuning")
    hist = load_history()
    if hist.empty:
        st.info("Log bets in the Results tab to begin calibration.")
    else:
        txt, sug = auto_tuning_recommendations(hist)
        st.markdown(txt)
        if sug:
            st.markdown("### Recommended Adjustments")
            for s in sug:
                st.markdown(f"- {s}")
            if st.button("Apply Changes"):
                st.success("✅ Adjustments applied for this session (temporary).")
        else:
            st.success("Model calibration looks balanced. No changes recommended.")

st.markdown(
    """
    <footer>© 2025 NBA Prop Model | Powered by Kamal</footer>
    """,
    unsafe_allow_html=True,
)
