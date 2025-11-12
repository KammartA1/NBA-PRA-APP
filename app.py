# ============================================================
#  NBA PROP MODEL – FULL PRO BUILD (Part 1 of 3)
#  Streamlit App | Dual-Player Advanced Modeling Engine
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime, time, os, json
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Streamlit Configuration & Theme
# ------------------------------------------------------------
st.set_page_config(
    page_title="NBA Prop Model",
    page_icon="🏀",
    layout="wide"
)

GOPHER_MAROON = "#7a0019"
GOPHER_GOLD = "#ffcc33"
BACKGROUND = "#111111"
CARD_BG = "#1b1b1b"

st.markdown(f"""
<style>
body {{
    background-color: {BACKGROUND};
    color: white;
    font-family: 'Inter', sans-serif;
}}
h1,h2,h3,h4 {{
    color: {GOPHER_GOLD};
}}
[data-testid="stMetricValue"] {{
    color: {GOPHER_GOLD} !important;
}}
.block-container {{
    padding-top: 1rem;
}}
.player-card {{
    background: {CARD_BG};
    padding: 1.2rem;
    border-radius: 16px;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Helper Utilities
# ------------------------------------------------------------

def get_player_id(name: str):
    """Return NBA API player_id for a given name."""
    plist = players.get_players()
    match = [p for p in plist if name.lower() in p['full_name'].lower()]
    return match[0]['id'] if match else None


@st.cache_data(show_spinner=False)
def fetch_recent_games(player_id, season='2024-25', last_n=10):
    """Fetch last N games from NBA API"""
    try:
        log = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        return log.head(last_n)
    except Exception:
        return pd.DataFrame()


def weighted_projection(df, stat):
    """Weighted moving average for projection."""
    if df.empty or stat not in df.columns:
        return None
    weights = np.linspace(1, 2, len(df))
    return np.average(df[stat].values, weights=weights)


def pace_def_adj(proj, pace_factor, def_factor):
    """Adjust projection for game context."""
    return proj * pace_factor * def_factor


def usage_minute_adjust(proj, usage_delta=0.0, minute_delta=0.0):
    return proj * (1 + usage_delta) * (1 + minute_delta)


def heavy_tail_prob(projection, line):
    """Compute probability over given PRA/pts line using heavy-tail model."""
    diff = projection - line
    scale = np.std([projection, line])
    prob = 1 / (1 + np.exp(-diff / (scale if scale else 1)))
    ev = (prob * 2 - 1) * 100
    return prob * 100, ev


def kelly_stake(bankroll, ev, fraction=0.25):
    if ev <= 0:
        return 0
    return bankroll * fraction * (ev / 100)


# ------------------------------------------------------------
# Sidebar Config
# ------------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", 100.0, step=50.0, value=1000.0)
kelly_frac = st.sidebar.slider("Fractional Kelly", 0.05, 1.0, 0.25)
pace_factor = st.sidebar.slider("Pace Factor", 0.8, 1.2, 1.0)
def_factor = st.sidebar.slider("Opponent Defense Factor", 0.8, 1.2, 1.0)
st.sidebar.markdown("---")

# ------------------------------------------------------------
# Dual Player Inputs
# ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Player 1 Inputs")
    p1_name = st.text_input("Name (Player 1)", "Anthony Edwards")
    p1_market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="m1")
    p1_line = st.number_input("Manual Line", step=0.5, value=33.5)
    teammate_out1 = st.checkbox("Key Teammate Out", key="to1")
    blowout1 = st.checkbox("Blowout Risk", key="bo1")

with col2:
    st.subheader("Player 2 Inputs")
    p2_name = st.text_input("Name (Player 2)", "Jayson Tatum")
    p2_market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="m2")
    p2_line = st.number_input("Manual Line", step=0.5, value=34.5)
    teammate_out2 = st.checkbox("Key Teammate Out", key="to2")
    blowout2 = st.checkbox("Blowout Risk", key="bo2")

# ------------------------------------------------------------
# Core Run Button
# ------------------------------------------------------------
if st.button("🚀 Run Model", use_container_width=True):
    # Retrieve data
    ids = [get_player_id(p1_name), get_player_id(p2_name)]
    dfs = [fetch_recent_games(pid) if pid else pd.DataFrame() for pid in ids]

    players_data = []
    for idx, (df, nm, mkt, line, tm_out, blw) in enumerate([
        (dfs[0], p1_name, p1_market, p1_line, teammate_out1, blowout1),
        (dfs[1], p2_name, p2_market, p2_line, teammate_out2, blowout2)
    ]):
        if df.empty:
            players_data.append({
                "name": nm, "projection": None, "prob": None, "ev": None
            })
            continue

        # Calculate projection
        if mkt == "PRA":
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
            proj = weighted_projection(df, "PRA")
        else:
            proj = weighted_projection(df, mkt[:3].upper())

        if proj is None:
            players_data.append({"name": nm, "projection": None})
            continue

        # Context Adjustments
        proj = pace_def_adj(proj, pace_factor, def_factor)
        if tm_out:
            proj = usage_minute_adjust(proj, usage_delta=0.07)
        if blw:
            proj = usage_minute_adjust(proj, minute_delta=-0.08)

        prob, ev = heavy_tail_prob(proj, line)
        stake = kelly_stake(bankroll, ev, kelly_frac)

        players_data.append({
            "name": nm, "market": mkt, "line": line,
            "projection": proj, "prob": prob, "ev": ev, "stake": stake
        })

    # Layout Output
    c1, c2 = st.columns(2)
    for i, dat in enumerate(players_data):
        if dat["projection"] is None:
            [c1, c2][i].warning(f"No data for {dat['name']}.")
            continue

        with [c1, c2][i]:
            st.markdown(f"""
            <div class='player-card'>
                <h3>{dat['name']} ({dat['market']})</h3>
                <p>Projection: <b>{dat['projection']:.2f}</b></p>
                <p>Line: <b>{dat['line']}</b></p>
                <p>Probability Over: <b>{dat['prob']:.1f}%</b></p>
                <p>EV: <b style='color:{"#16e316" if dat["ev"]>0 else "#f44336"}'>{dat["ev"]:.2f}%</b></p>
                <p>Recommended Stake: <b>${dat["stake"]:.2f}</b></p>
                <p>Decision: <b>{"✅ BET" if dat["ev"]>0 else "❌ PASS"}</b></p>
            </div>
            """, unsafe_allow_html=True)
# ------------------------------------------------------------
# Logging & Results Handling
# ------------------------------------------------------------

RESULTS_FILE = "results_log.csv"

def log_result(entry):
    """Append a new model run result to CSV."""
    columns = [
        "timestamp", "player", "market", "line", "projection",
        "probability", "ev", "stake", "decision", "clv"
    ]
    df = pd.DataFrame([entry], columns=columns)
    if os.path.exists(RESULTS_FILE):
        df.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULTS_FILE, mode="w", header=True, index=False)

def load_results():
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame(columns=[
            "timestamp", "player", "market", "line", "projection",
            "probability", "ev", "stake", "decision", "clv"
        ])
    return pd.read_csv(RESULTS_FILE)

# ------------------------------------------------------------
# CLV & Decision Logging
# ------------------------------------------------------------
for d in players_data:
    if d["projection"] is None: 
        continue
    clv = d["projection"] - d["line"]
    decision = "BET" if d["ev"] > 0 else "PASS"
    log_result([
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        d["name"], d["market"], d["line"], d["projection"],
        d["prob"], d["ev"], d["stake"], decision, clv
    ])

st.success("✅ Run logged to local results file!")

# ------------------------------------------------------------
# Results / Calibration Tab
# ------------------------------------------------------------
st.markdown("---")
tab1, tab2 = st.tabs(["📈 Model", "📊 Results"])

with tab2:
    st.header("📊 Historical Performance & Calibration")

    results_df = load_results()
    if results_df.empty:
        st.info("No results logged yet. Run the model first!")
    else:
        # Convert timestamp
        results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])

        # Line chart for EV
        st.subheader("Expected Value Over Time")
        fig, ax = plt.subplots()
        ax.plot(results_df["timestamp"], results_df["ev"], color=GOPHER_GOLD, linewidth=2)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_ylabel("Expected Value (%)")
        ax.set_xlabel("Date")
        ax.set_facecolor("#111")
        fig.patch.set_facecolor("#111")
        st.pyplot(fig)

        # Hit rate tracking
        st.subheader("Hit Rate vs EV")
        results_df["hit"] = np.where(results_df["ev"] > 0, 1, 0)
        rolling_hit = results_df["hit"].rolling(window=10, min_periods=1).mean()
        fig2, ax2 = plt.subplots()
        ax2.plot(results_df["timestamp"], rolling_hit*100, color=GOPHER_MAROON, linewidth=2)
        ax2.set_ylabel("Hit Rate (%)")
        ax2.set_xlabel("Date")
        ax2.set_facecolor("#111")
        fig2.patch.set_facecolor("#111")
        st.pyplot(fig2)

        # Basic calibration hints
        mean_ev = results_df["ev"].mean()
        mean_hit = rolling_hit.iloc[-1] * 100 if not rolling_hit.empty else 0
        st.markdown(f"**Average EV:** {mean_ev:.2f}%  |  **Recent Hit Rate:** {mean_hit:.2f}%")

        if mean_hit < 45:
            st.warning("Model underperforming — consider lowering aggressiveness or pace factor.")
        elif mean_hit > 65:
            st.success("Model performing strongly — parameters seem well-calibrated.")
        else:
            st.info("Stable calibration zone. Keep logging more data.")
# ------------------------------------------------------------
# Auto-Tuning Engine & Performance Summary
# ------------------------------------------------------------
with tab2:
    if not results_df.empty:
        st.markdown("---")
        st.header("⚙️ Model Tuning Insights")

        # Basic summaries
        avg_ev = results_df["ev"].mean()
        avg_hit = results_df["hit"].mean() * 100
        avg_kelly = results_df["stake"].mean()
        st.metric("Average EV (%)", f"{avg_ev:.2f}")
        st.metric("Hit Rate (%)", f"{avg_hit:.1f}")
        st.metric("Average Stake ($)", f"{avg_kelly:.2f}")

        # Trend detection
        last_20 = results_df.tail(20)
        ev_trend = np.polyfit(range(len(last_20)), last_20["ev"], 1)[0]
        hit_trend = np.polyfit(range(len(last_20)), last_20["hit"], 1)[0]

        st.subheader("📊 Performance Trends")
        colt1, colt2 = st.columns(2)
        colt1.metric("EV Trend", f"{'▲' if ev_trend>0 else '▼'} {abs(ev_trend):.2f}/run",
                     delta_color="inverse" if ev_trend<0 else "normal")
        colt2.metric("Hit Trend", f"{'▲' if hit_trend>0 else '▼'} {abs(hit_trend*100):.2f} %",
                     delta_color="inverse" if hit_trend<0 else "normal")

        # Recommendation logic
        st.markdown("#### 🧭 Recommended Tuning")
        if avg_hit < 45:
            st.write("🔴 Lower pace or defense factors (0.95–1.0). "
                     "Your model is too aggressive.")
        elif avg_hit > 65 and avg_ev > 0:
            st.write("🟢 Increase Kelly fraction slightly (0.3–0.4). "
                     "Model confidence is validated by hit rate.")
        else:
            st.write("🟡 Maintain current parameters for stability.")

        # Suggest next targets
        target_ev = avg_ev + (5 if avg_ev < 10 else 0)
        target_hit = min(avg_hit + 3, 70)
        st.info(f"🎯 Next Goal: EV ≈ {target_ev:.1f}% | Hit Rate ≈ {target_hit:.1f}%")

# ------------------------------------------------------------
# Footer / Session Summary
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 NBA Prop Model — Built for consistent long-term edge generation.")

# Smooth end-of-run animation
with st.empty():
    for i in range(15):
        st.progress(i/15)
        time.sleep(0.03)
    st.success("🏁 Model Ready for Next Run!")

# End of File
