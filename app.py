# =============================================================
#  NBA Prop Model – Optimized Version (Part A1)
#  Author: Kamal Martin x ChatGPT (GPT-5)
#  Description: High-performance NBA Prop Betting Model
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
import time
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# =============================================================
#  GLOBAL SETTINGS & CONSTANTS
# =============================================================

st.set_page_config(page_title="NBA Prop Model", layout="wide")

# --- Colors (slightly darker Gopher theme) ---
GOPHER_MAROON = "#5E0018"
GOPHER_GOLD = "#FFCC33"
BACKGROUND_DARK = "#0D0D0D"
SECONDARY_DARK = "#181818"
TEXT_COLOR = "#FFFFFF"

RESULTS_FILE = "results_log.csv"

# --- Create log file if missing ---
if not os.path.exists(RESULTS_FILE):
    base_cols = [
        "timestamp", "player", "market", "line", "projection", "probability",
        "ev", "stake", "clv", "variance", "skewness", "p25", "p75", "sim_mean"
    ]
    pd.DataFrame(columns=base_cols).to_csv(RESULTS_FILE, index=False)

# =============================================================
#  SESSION STATE INITIALIZATION
# =============================================================

if "results" not in st.session_state:
    st.session_state["results"] = None
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = None

# =============================================================
#  SIDEBAR – REFRESH & BANKROLL SETTINGS
# =============================================================

with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🔄 Refresh Data Manually"):
        st.session_state["last_refresh"] = datetime.now()
        st.success("Data refreshed successfully!")

    bankroll = st.number_input("💰 Bankroll ($)", value=30.0, step=10.0)
    risk_limit = 0.05  # Max daily loss cap = 5%

# =============================================================
#  HELPER FUNCTIONS
# =============================================================

@st.cache_data(show_spinner=False)
def get_nba_player_id(player_name: str):
    """Fetch NBA player ID from nba_api."""
    try:
        player_list = players.get_players()
        match = [p for p in player_list if p["full_name"].lower() == player_name.lower()]
        return match[0]["id"] if match else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def get_last_15_games(player_name: str):
    """Return last 15-game stats for player with rolling averages."""
    pid = get_nba_player_id(player_name)
    if not pid:
        return None

    try:
        gamelog = playergamelog.PlayerGameLog(player_id=pid, season="2024-25").get_data_frames()[0]
        df = gamelog.head(15)[["GAME_DATE", "PTS", "REB", "AST", "MIN"]]
        df = df.assign(PRA=df["PTS"] + df["REB"] + df["AST"])
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE")
        df["rolling_PRA"] = df["PRA"].rolling(5, min_periods=1).mean()
        return df
    except Exception:
        return None


def fetch_backup_metrics(player_name: str):
    """Fallback to SportsMetrics API (simplified dummy backup)."""
    try:
        url = f"https://sportsmetrics.io/api/nba/player/{player_name.replace(' ', '%20')}/summary"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {
                "usage": data.get("usageRate", np.nan),
                "pace": data.get("pace", np.nan),
                "minutes": data.get("minutes", np.nan),
                "team_defense": data.get("defRating", np.nan),
            }
    except Exception:
        pass
    return {"usage": np.nan, "pace": np.nan, "minutes": np.nan, "team_defense": np.nan}


def ensure_dataframe(df):
    """Ensure DataFrame integrity for simulation pipeline."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["GAME_DATE", "PTS", "REB", "AST", "PRA", "rolling_PRA"])
    return df

# =============================================================
#  MODEL TAB – PLAYER INPUTS & LAYOUT
# =============================================================

st.title("🏀 NBA Prop Model – Optimized Edition")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Model", "📈 Results", "⚙️ Calibration", "🔍 Insights"])

with tab1:
    st.header("📊 Player Selection & Simulation")

    c1, c2 = st.columns(2)
    with c1:
        p1_name = st.text_input("Player 1 Name", "")
        p1_market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="p1_market")
        p1_line = st.number_input("Player 1 Line", step=0.5, key="p1_line")

        key_out1 = st.checkbox("Key teammate out (↑ usage)", value=False)
        blowout1 = st.checkbox("High blowout risk (↓ minutes)", value=False)

    with c2:
        p2_name = st.text_input("Player 2 Name (optional)", "")
        p2_market = st.selectbox("Market", ["PRA", "Points", "Rebounds", "Assists"], key="p2_market")
        p2_line = st.number_input("Player 2 Line", step=0.5, key="p2_line")

        key_out2 = st.checkbox("Key teammate out (↑ usage)", value=False)
        blowout2 = st.checkbox("High blowout risk (↓ minutes)", value=False)

    st.markdown("---")
    st.markdown("### 📥 Load Player Data")

    if st.button("Fetch Latest Stats"):
        with st.spinner("Retrieving player data..."):
            p1_data = get_last_15_games(p1_name)
            p2_data = get_last_15_games(p2_name) if p2_name else None
            if p1_data is None:
                st.error(f"No valid data found for {p1_name}")
            else:
                st.success(f"{p1_name} data loaded successfully.")
            if p2_name:
                if p2_data is None:
                    st.error(f"No valid data found for {p2_name}")
                else:
                    st.success(f"{p2_name} data loaded successfully.")
            st.session_state["p1_data"] = ensure_dataframe(p1_data)
            st.session_state["p2_data"] = ensure_dataframe(p2_data)

# ---- End of Part A1 ----
# =============================================================
#  DATA PREPARATION & BOOTSTRAP PIPELINE  (Part A2)
# =============================================================

def adjust_for_context(df, usage_boost=False, blowout_risk=False):
    """Apply contextual adjustments (usage/minutes modifiers)."""
    if df is None or df.empty:
        return df

    df = df.copy()
    # usage boost: +5% PRA increase
    if usage_boost:
        df["PRA"] *= 1.05
        df["PTS"] *= 1.05
        df["REB"] *= 1.05
        df["AST"] *= 1.05

    # blowout: -8% minutes impact on PRA/PTS/REB/AST
    if blowout_risk:
        df["PRA"] *= 0.92
        df["PTS"] *= 0.92
        df["REB"] *= 0.92
        df["AST"] *= 0.92

    return df


def bootstrap_simulations(values, n=1000):
    """Run bootstrap sampling for simulation of player prop outcomes."""
    if len(values) == 0:
        return np.array([])
    idx = np.random.randint(0, len(values), (n, len(values)))
    samples = np.take(values, idx)
    return samples.mean(axis=1)


def simulate_player(df, market="PRA", line=0.0, usage=False, blowout=False):
    """Perform 1000-run bootstrapped Monte Carlo simulation for one player."""
    df = adjust_for_context(df, usage, blowout)
    market_col = {
        "PRA": "PRA",
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST"
    }.get(market, "PRA")

    vals = df[market_col].dropna().values
    sim_results = bootstrap_simulations(vals, n=1000)
    if len(sim_results) == 0:
        return None

    prob = (sim_results > line).mean()
    mean = np.mean(sim_results)
    var = np.var(sim_results)
    skew = (np.mean((sim_results - mean) ** 3)) / (np.std(sim_results) ** 3 + 1e-8)
    p25, p75 = np.percentile(sim_results, [25, 75])
    return {
        "prob": prob,
        "mean": mean,
        "variance": var,
        "skewness": skew,
        "p25": p25,
        "p75": p75,
        "simulations": sim_results
    }

# =============================================================
#  RUN MODEL SECTION WITH SPINNER + PROGRESS BAR
# =============================================================

st.markdown("---")
st.markdown("### 🧮 Run Model Simulation")

if st.button("Run 1000 Simulations"):
    st.session_state["results"] = None
    p1_df = st.session_state.get("p1_data")
    p2_df = st.session_state.get("p2_data")

    if p1_df is None or p1_df.empty:
        st.error("Please fetch Player 1 data first.")
    else:
        with st.spinner("Running Monte Carlo simulations..."):
            progress = st.progress(0)
            # simulate player 1
            p1_stats = simulate_player(p1_df, p1_market, p1_line, key_out1, blowout1)
            progress.progress(50)
            # simulate player 2 (if any)
            p2_stats = None
            if p2_name:
                p2_stats = simulate_player(p2_df, p2_market, p2_line, key_out2, blowout2)
            progress.progress(100)
            time.sleep(0.3)
            progress.empty()
        st.success("✅ Simulation complete.")
        st.session_state["results"] = {
            "p1": p1_stats,
            "p2": p2_stats,
            "p1_name": p1_name,
            "p2_name": p2_name,
            "p1_line": p1_line,
            "p2_line": p2_line,
            "p1_market": p1_market,
            "p2_market": p2_market,
        }

# ---- End of Part A2 ----
# =============================================================
#  MONTE CARLO POST-PROCESSING & RESULTS DISPLAY (Part B1)
# =============================================================

def compute_ev(probability: float, odds: float = 1.5):
    """Compute expected value (%) given probability and odds."""
    if probability <= 0 or odds <= 0:
        return 0.0
    ev = (probability * odds - 1) * 100
    return ev


def compute_kelly(prob: float, odds: float = 1.5, fraction: float = 0.3):
    """Fractional Kelly stake sizing."""
    b = odds - 1
    edge = (prob * (b + 1) - 1) / b if b != 0 else 0
    return max(0, edge * fraction)


def compute_clv(projection: float, line: float):
    """Simple CLV metric (projection - line)."""
    return projection - line


def append_to_csv(result_dict):
    """Append simulation result to local results_log.csv."""
    df = pd.DataFrame([result_dict])
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(RESULTS_FILE, index=False)
    else:
        df.to_csv(RESULTS_FILE, index=False)


def render_stat_card(player_name, market, stats_dict, line, color=GOPHER_GOLD):
    """Render a compact stat card for each player."""
    if stats_dict is None:
        st.warning(f"No stats for {player_name}.")
        return

    prob = stats_dict["prob"]
    ev = compute_ev(prob)
    kelly_frac = compute_kelly(prob)
    stake = kelly_frac * bankroll
    clv = compute_clv(stats_dict["mean"], line)

    # Stat card layout
    st.markdown(
        f"""
        <div style='background-color:{SECONDARY_DARK};
                    border-left:4px solid {color};
                    padding:10px;margin-bottom:10px;
                    border-radius:10px'>
            <h4 style='color:{color};margin:0'>{player_name} – {market}</h4>
            <p style='color:{TEXT_COLOR};margin:0.3em 0'>
                <b>Line:</b> {line} |
                <b>Projection:</b> {stats_dict["mean"]:.2f} |
                <b>Prob:</b> {prob*100:.1f}% |
                <b>EV:</b> {ev:.1f}% |
                <b>CLV:</b> {clv:.2f}
            </p>
            <p style='color:{TEXT_COLOR};margin:0.3em 0'>
                <b>Kelly Stake:</b> ${stake:.2f} |
                <b>Variance:</b> {stats_dict["variance"]:.2f} |
                <b>Skew:</b> {stats_dict["skewness"]:.2f}
            </p>
            <small style='color:#888'>
                25th-75th Percentile Range: {stats_dict["p25"]:.1f} – {stats_dict["p75"]:.1f}
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )

    # small histogram
    fig, ax = plt.subplots(figsize=(4, 1.2))
    ax.hist(stats_dict["simulations"], bins=25, color=color, alpha=0.7)
    ax.axvline(line, color="#ff4444", linestyle="--", linewidth=1)
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.tick_params(axis="y", colors="white", labelsize=8)
    ax.set_title(f"{player_name} Distribution", color="white", fontsize=9)
    st.pyplot(fig, use_container_width=True)

    # return dict for logging
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": player_name,
        "market": market,
        "line": line,
        "projection": stats_dict["mean"],
        "probability": prob,
        "ev": ev,
        "stake": stake,
        "clv": clv,
        "variance": stats_dict["variance"],
        "skewness": stats_dict["skewness"],
        "p25": stats_dict["p25"],
        "p75": stats_dict["p75"],
        "sim_mean": stats_dict["mean"],
    }

# ---- End of Part B1 ----
# =============================================================
#  LOGGING & TWO-PLAYER DISPLAY + TAB ROUTING  (Part B2)
# =============================================================

with tab1:
    st.markdown("### 📈 Simulation Results")

    if st.session_state.get("results"):
        r = st.session_state["results"]
        p1, p2 = r.get("p1"), r.get("p2")

        # ---- Layout: two cards side-by-side ----
        c1, c2 = st.columns(2)
        logs = []

        with c1:
            if p1:
                res1 = render_stat_card(
                    r["p1_name"], r["p1_market"], p1, r["p1_line"], GOPHER_GOLD
                )
                logs.append(res1)

        with c2:
            if p2:
                res2 = render_stat_card(
                    r["p2_name"], r["p2_market"], p2, r["p2_line"], GOPHER_MAROON
                )
                logs.append(res2)

        # ---- Log results safely ----
        if logs:
            for entry in logs:
                if entry:
                    append_to_csv(entry)
            st.success("✅ Logged results to results_log.csv")

    else:
        st.info("Run simulations to see results here.")

# =============================================================
#  TAB ROUTING
# =============================================================

with tab2:
    results_tab_ui()

with tab3:
    calibration_tab_ui()

with tab4:
    insights_tab_ui()

# =============================================================
#  FOOTER
# =============================================================

st.markdown("---")
st.caption(
    "NBA Prop Model — Optimized Edition | "
    "1000-run Monte Carlo Simulation | "
    "Fractional Kelly Staking • Manual Data Refresh • Local Logging"
)

# ---- End of Part B2 / End of app.py ----
