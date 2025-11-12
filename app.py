# =============================================================
#  NBA Prop Model – Advanced Edition (Part 1A of 4)
#  Preserves full working foundation with styling & caching
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime, time, os, json, requests, random, math
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.stats import skew, kurtosis

# =============================================================
#  STREAMLIT PAGE CONFIGURATION
# =============================================================

st.set_page_config(
    page_title="NBA Prop Model",
    page_icon="🏀",
    layout="wide"
)

# =============================================================
#  COLORWAY + UI STYLING
# =============================================================

GOPHER_MAROON = "#7A0019"
GOPHER_GOLD = "#FFCC33"
BACKGROUND = "#0F0F0F"
TEXT_COLOR = "#F5F5F5"

st.markdown(
    f"""
    <style>
        body {{
            background-color: {BACKGROUND};
            color: {TEXT_COLOR};
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background-color: {BACKGROUND};
            color: {TEXT_COLOR};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {GOPHER_GOLD};
            font-weight: 600;
        }}
        .player-card {{
            background-color: #1C1C1C;
            border-radius: 18px;
            padding: 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 0 12px rgba(255, 204, 51, 0.15);
        }}
        .metric-card {{
            background-color: #202020;
            border-radius: 14px;
            padding: 0.7rem;
            text-align: center;
            color: white;
            margin: 0.5rem;
        }}
        .adv {{
            font-size: 0.9rem;
            color: #CCCCCC;
            margin-top: 0.3rem;
        }}
        .tooltip {{
            color: {GOPHER_GOLD};
            cursor: help;
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            background-color: #181818;
            border-radius: 8px 8px 0 0;
            color: {TEXT_COLOR};
            border: 1px solid #333;
            font-weight: 500;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: {GOPHER_MAROON};
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================
#  GLOBAL CONSTANTS AND DEFAULTS
# =============================================================

RESULTS_FILE = "results_log.csv"
DAILY_BANKROLL = 30.0
DAILY_LOSS_CAP = 0.05
SIMULATIONS = 1000
BOOTSTRAP_WINDOW = 15
NBA_SEASON = "2024-25"

# =============================================================
#  HELPER FUNCTIONS & CACHE SETTINGS
# =============================================================

@st.cache_data(ttl=86400)
def cached_request(url, headers=None, params=None):
    """Generic cached GET request with 24h TTL for stability."""
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.warning(f"⚠️ Non-200 response from {url}: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Network error for {url}: {e}")
        return None

@st.cache_data(ttl=86400)
def read_local_results():
    """Load the results CSV if present; create empty if not."""
    if os.path.exists(RESULTS_FILE):
        try:
            df = pd.read_csv(RESULTS_FILE)
            return df
        except Exception:
            st.warning("⚠️ Could not read results file. Re-creating.")
    return pd.DataFrame(columns=[
        "timestamp","player","market","line","projection",
        "probability","ev","stake","decision","clv",
        "variance","skewness","p25","p75","sim_mean"
    ])

# =============================================================
#  STYLIZED HEADER
# =============================================================

st.markdown(
    """
    <div style='text-align:center; padding-top:0.5rem; padding-bottom:1rem;'>
        <h1>🏀 NBA Prop Model</h1>
        <p style='color:#ccc;'>Monte Carlo | PRA | Risk Management | Edge Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================
#  DATA REFRESH TIMER & CACHE INFO
# =============================================================

def refresh_notice():
    st.sidebar.info(
        "🔄 Data refreshes automatically every 24 hours.\n"
        "Bootstrap window = 15 games per player.\n"
        "Simulations = 1 000 per run."
    )

refresh_notice()

# ---- End of Part 1A ----
# =============================================================
#  NBA API + SportsMetrics Dual-Source Data Fetcher
# =============================================================

from nba_api.stats.endpoints import leaguedashteamstats, playergamelog, commonplayerinfo
from nba_api.stats.static import players

@st.cache_data(ttl=86400)
def fetch_team_metrics():
    """
    Pull team pace and defense metrics.
    Tries nba_api first; falls back to SportsMetrics API if unavailable.
    """
    try:
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame',
            season=NBA_SEASON
        ).get_data_frames()[0]
        df = team_stats[['TEAM_ID','TEAM_NAME','PACE','DEF_RATING']]
        df.columns = ['team_id','team_name','pace','def_rating']
        return df
    except Exception as e:
        st.warning(f"NBA API failed: {e}. Switching to SportsMetrics backup ...")
        try:
            res = cached_request("https://api.sportsmetrics.io/nba/team_metrics")
            if res:
                df = pd.DataFrame(res)
                if not {'team_name','pace','def_rating'}.issubset(df.columns):
                    st.error("SportsMetrics data missing expected columns.")
                    return pd.DataFrame(columns=['team_id','team_name','pace','def_rating'])
                return df
        except Exception as ex:
            st.error(f"SportsMetrics fallback also failed: {ex}")
    return pd.DataFrame(columns=['team_id','team_name','pace','def_rating'])

team_metrics = fetch_team_metrics()

# =============================================================
#  PLAYER LOOKUP & INFO UTILITIES
# =============================================================

@lru_cache(maxsize=512)
def find_player_id(name):
    """
    Resolve player name → ID mapping (case-insensitive).
    Returns None if no match found.
    """
    plist = players.get_players()
    for p in plist:
        if name.lower() in p['full_name'].lower():
            return p['id']
    return None

def get_player_team_pos(pid):
    """
    Get player's team and position info.
    """
    try:
        data = commonplayerinfo.CommonPlayerInfo(player_id=pid).get_data_frames()[0]
        return data.loc[0,'TEAM_NAME'], data.loc[0,'POSITION']
    except Exception:
        return "Unknown","N/A"

# =============================================================
#  PLAYER GAME LOG FETCHER (15-Game Window)
# =============================================================

@st.cache_data(ttl=3600)
def get_last_games(pid, n=BOOTSTRAP_WINDOW):
    """
    Retrieve a player's last N games from nba_api.
    Adds PRA (PTS + REB + AST) column.
    """
    try:
        df = playergamelog.PlayerGameLog(
            player_id=pid,
            season=NBA_SEASON
        ).get_data_frames()[0]
        df = df.head(n)
        df['PRA'] = df['PTS'] + df['REB'] + df['AST']
        return df[['GAME_DATE','PTS','REB','AST','PRA']]
    except Exception as e:
        st.warning(f"Game log fetch failed for {pid}: {e}")
        return pd.DataFrame(columns=['GAME_DATE','PTS','REB','AST','PRA'])

# =============================================================
#  DATA PREPROCESSING HELPERS
# =============================================================

def clean_gamelog(gamelog: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric columns are floats and drop NaNs.
    """
    for c in ['PTS','REB','AST','PRA']:
        if c in gamelog.columns:
            gamelog[c] = pd.to_numeric(gamelog[c], errors='coerce')
    return gamelog.dropna(subset=['PTS','REB','AST','PRA'])

def compute_player_trends(gamelog: pd.DataFrame) -> dict:
    """
    Compute rolling averages and volatility metrics for the last 15 games.
    """
    if gamelog.empty:
        return {}
    trends = {
        "avg_pts": gamelog['PTS'].mean(),
        "avg_reb": gamelog['REB'].mean(),
        "avg_ast": gamelog['AST'].mean(),
        "avg_pra": gamelog['PRA'].mean(),
        "var_pra": np.var(gamelog['PRA']),
        "skew_pra": skew(gamelog['PRA']),
        "p25": np.percentile(gamelog['PRA'], 25),
        "p75": np.percentile(gamelog['PRA'], 75)
    }
    return trends

# ---- End of Part 1B ----
# =============================================================
#  Monte Carlo Simulation + EV / Kelly / CLV Engine  (Part 1C-A)
# =============================================================

def run_monte_carlo(gamelog: pd.DataFrame, metric: str, line: float, bankroll: float):
    """
    Bootstrapped Monte Carlo Simulation for player prop.
    Uses last 15 games with 1000 resampled distributions.
    Returns dict with projection, EV, CLV, stake, and stats.
    """
    gamelog = clean_gamelog(gamelog)
    if gamelog.empty or metric not in gamelog.columns:
        return {
            "projection": 0,
            "prob": 0,
            "ev": 0,
            "stake": 0,
            "clv": 0,
            "variance": 0,
            "skewness": 0,
            "p25": 0,
            "p75": 0,
            "sim_mean": 0,
            "samples": []
        }

    # --- Bootstrapped resampling ---
    vals = gamelog[metric].values
    samples = np.random.choice(vals, size=(SIMULATIONS, len(vals)), replace=True)
    sim_means = samples.mean(axis=1)

    # --- Distribution statistics ---
    sim_mean = np.mean(sim_means)
    variance = np.var(sim_means)
    skewness = skew(sim_means)
    p25, p75 = np.percentile(sim_means, [25, 75])

    # --- Probabilities ---
    prob = np.mean(sim_means > line)   # % of samples beating the line
    ev = expected_value(prob, line_odds=1.5)
    clv = sim_mean - line
    stake = kelly_stake(ev, prob, bankroll)

    return {
        "projection": sim_mean,
        "prob": prob,
        "ev": ev,
        "stake": stake,
        "clv": clv,
        "variance": variance,
        "skewness": skewness,
        "p25": p25,
        "p75": p75,
        "sim_mean": sim_mean,
        "samples": sim_means.tolist()
    }

# =============================================================
#  ENHANCED LOGGING HANDLER
# =============================================================

def append_result(entry: dict):
    """
    Append single simulation result to local results_log.csv.
    Adds new statistical columns if file already exists.
    """
    row = pd.DataFrame([{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "player": entry.get("player", ""),
        "market": entry.get("market", ""),
        "line": entry.get("line", 0),
        "projection": entry.get("projection", 0),
        "probability": entry.get("prob", 0),
        "ev": entry.get("ev", 0),
        "stake": entry.get("stake", 0),
        "decision": "BET" if entry.get("ev", 0) > 0 else "PASS",
        "clv": entry.get("clv", 0),
        "variance": entry.get("variance", 0),
        "skewness": entry.get("skewness", 0),
        "p25": entry.get("p25", 0),
        "p75": entry.get("p75", 0),
        "sim_mean": entry.get("sim_mean", 0)
    }])

    if os.path.exists(RESULTS_FILE):
        try:
            row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)
        except Exception as e:
            st.warning(f"Logging append failed: {e}")
    else:
        row.to_csv(RESULTS_FILE, mode="w", header=True, index=False)

# =============================================================
#  PLAYER SIMULATION WRAPPER (CALLED FROM MODEL TAB)
# =============================================================

def simulate_player(name: str, market: str, line: float, bankroll: float):
    """
    High-level wrapper: runs simulation, logs result, and returns output dict.
    """
    pid = find_player_id(name)
    if not pid:
        st.warning(f"Player {name} not found.")
        return None

    team, pos = get_player_team_pos(pid)
    gamelog = get_last_games(pid)
    if gamelog.empty:
        st.error(f"No game data for {name}.")
        return None

    gamelog = clean_gamelog(gamelog)
    result = run_monte_carlo(gamelog, market, line, bankroll)
    result["player"] = name
    result["market"] = market
    result["line"] = line
    result["team"] = team
    result["position"] = pos

    append_result(result)
    return result

# ---- End of Part 1C-A ----
# =============================================================
#  Simulation Display & Model Tab Integration (Part 1C-B)
# =============================================================

def display_simulation(result: dict):
    """
    Draw summary cards and histogram for a single simulation result.
    """
    if not result:
        st.info("No data to display yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projection", f"{result['projection']:.1f}")
    c2.metric("Prob > Line", f"{result['prob']*100:.1f}%")
    c3.metric("Expected Value", f"{result['ev']:.1f}%")
    c4.metric("Kelly Stake", f"${result['stake']:.2f}")

    with st.expander("📊 Distribution Details", expanded=False):
        fig, ax = plt.subplots()
        ax.hist(result["samples"], bins=30, color=GOPHER_MAROON, alpha=0.75)
        ax.axvline(result["line"], color=GOPHER_GOLD, linestyle="--", label="Line")
        ax.axvline(result["projection"], color="white", linestyle="-", label="Mean Projection")
        ax.set_facecolor("#111")
        ax.legend(facecolor="#111", labelcolor="white")
        st.pyplot(fig)

        st.caption(
            f"Variance = {result['variance']:.2f} | Skew = {result['skewness']:.2f} | "
            f"25-75% Range = [{result['p25']:.1f}, {result['p75']:.1f}]"
        )

# =============================================================
#  MODEL TAB UI HOOK
# =============================================================

def model_tab_ui():
    """
    Integrates player entry boxes, line inputs, and Monte Carlo execution.
    """
    st.header("🏀 Model Simulation")

    st.markdown("Enter up to two players to simulate PRA / PTS / REB / AST props.")

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.text_input("Player 1 Name")
        market1 = st.selectbox("Market (P1)", ["PRA","PTS","REB","AST"], key="m1")
        line1 = st.number_input("Line (P1)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="l1")
    with col2:
        p2 = st.text_input("Player 2 Name")
        market2 = st.selectbox("Market (P2)", ["PRA","PTS","REB","AST"], key="m2")
        line2 = st.number_input("Line (P2)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="l2")

    st.markdown("—")

    bankroll = st.number_input("💰 Daily Bankroll", min_value=1.0, max_value=1000.0,
                                value=DAILY_BANKROLL, step=1.0)

    colA, colB = st.columns(2)
    with colA:
        blowout = st.checkbox("Blowout Risk High (-10 % Minutes)", value=False)
    with colB:
        teammate_out = st.checkbox("Key Teammate Out (+8 % Usage)", value=False)

    run = st.button("▶️ Run Model")

    if run:
        st.info("Running simulations … please wait ≈ 3 seconds per player.")
        if p1:
            st.subheader(f"Results – {p1}")
            res1 = simulate_player(p1, market1, line1, bankroll)
            if blowout:
                res1["projection"] *= 0.9
            if teammate_out:
                res1["projection"] *= 1.08
            display_simulation(res1)

        if p2:
            st.subheader(f"Results – {p2}")
            res2 = simulate_player(p2, market2, line2, bankroll)
            if blowout:
                res2["projection"] *= 0.9
            if teammate_out:
                res2["projection"] *= 1.08
            display_simulation(res2)

        st.success("✅ Simulations complete and logged to results_log.csv !")

# ---- End of Part 1C-B ----
# =============================================================
#  RESULTS TAB  (Part 2A)
# =============================================================

def results_tab_ui():
    """
    Display logged simulation results with EV trend,
    rolling hit rate, and summary performance metrics.
    """
    st.header("📊 Results Log")

    if not os.path.exists(RESULTS_FILE):
        st.info("No results logged yet. Run a simulation first.")
        return

    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        st.info("No entries yet.")
        return

    # Convert timestamps safely
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        df["timestamp"] = pd.to_datetime("today")

    # ---- Summary Metrics ----
    avg_ev = df["ev"].mean()
    avg_clv = df["clv"].mean()
    avg_stake = df["stake"].mean()
    avg_var = df["variance"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average EV", f"{avg_ev:.2f}%")
    c2.metric("Average CLV", f"{avg_clv:.2f}")
    c3.metric("Avg Stake", f"${avg_stake:.2f}")
    c4.metric("Avg Variance", f"{avg_var:.2f}")

    # ---- EV Chart ----
    st.subheader("Expected Value Over Time")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["ev"], color=GOPHER_GOLD, linewidth=2)
    ax.axhline(0, color="#555", linestyle="--")
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    ax.tick_params(colors="white")
    ax.set_ylabel("EV (%)", color="white")
    ax.set_xlabel("Date", color="white")
    st.pyplot(fig)

    # ---- Rolling Hit Rate ----
    st.subheader("Rolling Hit Rate (EV>0)")
    df["hit"] = (df["ev"] > 0).astype(int)
    df["rolling_hit"] = df["hit"].rolling(window=10, min_periods=1).mean() * 100

    fig2, ax2 = plt.subplots()
    ax2.plot(df["timestamp"], df["rolling_hit"], color=GOPHER_MAROON, linewidth=2)
    ax2.axhline(50, color="#333", linestyle="--")
    ax2.set_facecolor("#111")
    fig2.patch.set_facecolor("#111")
    ax2.tick_params(colors="white")
    ax2.set_ylabel("Hit Rate (%)", color="white")
    ax2.set_xlabel("Date", color="white")
    st.pyplot(fig2)

    # ---- Expanded Table ----
    with st.expander("📋 View Detailed Log"):
        st.dataframe(
            df[[
                "timestamp","player","market","line","projection",
                "probability","ev","stake","clv","variance","skewness",
                "p25","p75","sim_mean"
            ]].sort_values("timestamp", ascending=False),
            use_container_width=True
        )

# ---- End of Part 2A ----
# =============================================================
#  RESULTS TAB CONTINUATION – Calibration Feedback (Part 2B)
# =============================================================

    # ---- Calibration & Feedback ----
    st.markdown("---")
    st.subheader("⚙️ Model Calibration & Performance Bands")

    mean_ev = df["ev"].mean()
    recent_hit = df["rolling_hit"].iloc[-1] if "rolling_hit" in df.columns else 0
    avg_prob = df["probability"].mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Average EV (%)", f"{mean_ev:.2f}")
    c2.metric("Recent Hit Rate (%)", f"{recent_hit:.2f}")
    c3.metric("Avg Predicted Probability (%)", f"{avg_prob:.2f}")

    if recent_hit < 45:
        st.warning(
            "🔴 Under-performing band detected.\n\n"
            "Model likely too aggressive — consider slightly **reducing pace multipliers (0.95–1.0)** "
            "or **tightening variance scaling**."
        )
    elif 45 <= recent_hit <= 65:
        st.info(
            "🟡 Stable calibration band.\n\n"
            "Model is performing consistently. Continue logging data for larger sample confidence."
        )
    else:
        st.success(
            "🟢 Strong calibration band!\n\n"
            "You’re outperforming market expectation — you may cautiously **increase fractional Kelly "
            "or expand sample size per day.**"
        )

    # ---- Monthly Snapshot ----
    st.subheader("📆 Monthly Snapshot")
    df["month"] = df["timestamp"].dt.to_period("M")
    monthly = df.groupby("month").agg({
        "ev": "mean",
        "hit": "mean",
        "stake": "mean",
        "clv": "mean"
    }).reset_index()

    fig3, ax3 = plt.subplots()
    ax3.plot(monthly["month"].astype(str), monthly["ev"], color=GOPHER_GOLD, label="Avg EV (%)")
    ax3.plot(monthly["month"].astype(str), monthly["hit"]*100, color=GOPHER_MAROON, label="Hit Rate (%)")
    ax3.set_facecolor("#111")
    fig3.patch.set_facecolor("#111")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#111", labelcolor="white")
    ax3.set_xlabel("Month", color="white")
    ax3.set_ylabel("Percentage", color="white")
    st.pyplot(fig3)

    st.caption(
        "EV = Expected Value %, Hit Rate = % of positive EV bets per month. "
        "Use this chart to detect performance drift or seasonal variance."
    )

# ---- End of Part 2B ----
# =============================================================
#  CALIBRATION TAB  (Part 3)
# =============================================================

def calibration_tab_ui():
    """
    Analyze calibration across EV bins and generate tuning suggestions.
    """
    st.header("🧠 Model Calibration & Tuning")

    if not os.path.exists(RESULTS_FILE):
        st.info("No results logged yet. Run simulations to build calibration data.")
        return

    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        st.info("No data available yet.")
        return

    # Clean and compute bins
    df["hit"] = (df["ev"] > 0).astype(int)
    df["ev_bin"] = pd.cut(
        df["ev"],
        bins=[-999, 10, 20, 999],
        labels=["Thin Edge (<10%)", "Moderate Edge (10–20%)", "Strong Edge (>20%)"]
    )

    # ---- Bin Performance ----
    st.subheader("📦 EV Bin Performance")
    bin_perf = df.groupby("ev_bin").agg({
        "hit": ["mean", "count"],
        "ev": "mean",
        "clv": "mean"
    }).round(3)
    bin_perf.columns = ["Hit Rate", "Count", "Avg EV", "Avg CLV"]
    st.dataframe(bin_perf, use_container_width=True)

    # ---- Chart: Calibration Consistency ----
    st.subheader("📈 Calibration Consistency (EV → Hit Rate)")
    avg_hit = bin_perf["Hit Rate"].mean() * 100
    fig, ax = plt.subplots()
    ax.bar(bin_perf.index.astype(str), bin_perf["Hit Rate"] * 100, color=GOPHER_GOLD, alpha=0.8)
    ax.axhline(50, color="#444", linestyle="--", linewidth=1)
    ax.set_facecolor("#111")
    fig.patch.set_facecolor("#111")
    ax.tick_params(colors="white")
    ax.set_ylabel("Hit Rate (%)", color="white")
    ax.set_xlabel("EV Bin", color="white")
    st.pyplot(fig)

    # ---- Automatic Tuning Suggestions ----
    st.subheader("🧭 Tuning Recommendations")

    mean_ev = df["ev"].mean()
    mean_hit = df["hit"].mean() * 100
    clv_mean = df["clv"].mean()
    variance_avg = df["variance"].mean()

    st.write(f"**Average EV:** {mean_ev:.2f}% | **Hit Rate:** {mean_hit:.1f}% | **Avg CLV:** {clv_mean:.2f}")

    if mean_hit < 45:
        st.warning(
            "🔴 Model underperforming.\n"
            "- Decrease offensive pace weighting (–5%)\n"
            "- Increase defensive strength adjustment (+5%)\n"
            "- Reduce variance scaling slightly (0.9×)\n"
            "- Consider expanding sample size per player"
        )
    elif 45 <= mean_hit <= 65:
        st.info(
            "🟡 Model calibrated.\n"
            "- Keep parameters stable\n"
            "- Continue tracking 30+ new results before further adjustment\n"
            "- Minor fine-tune only if CLV drops for multiple days"
        )
    else:
        st.success(
            "🟢 Model overperforming.\n"
            "- Maintain current variance scaling\n"
            "- Optionally increase Kelly fraction (0.3 → 0.35)\n"
            "- Capture as much data as possible to confirm edge persistence"
        )

    # ---- Calibration Trend ----
    st.subheader("📊 Calibration Stability Over Time")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    cal_trend = df.groupby("date")["hit"].mean().rolling(7, min_periods=1).mean() * 100

    fig2, ax2 = plt.subplots()
    ax2.plot(cal_trend.index, cal_trend.values, color=GOPHER_MAROON, linewidth=2)
    ax2.axhline(50, color="#444", linestyle="--", linewidth=1)
    ax2.set_facecolor("#111")
    fig2.patch.set_facecolor("#111")
    ax2.tick_params(colors="white")
    ax2.set_ylabel("7-Day Rolling Hit Rate (%)", color="white")
    ax2.set_xlabel("Date", color="white")
    st.pyplot(fig2)

    st.caption(
        "Rolling hit rate shows how calibration evolves. Flat lines near 50% suggest balance; "
        "rising trends indicate improving predictive reliability."
    )

# ---- End of Part 3 ----
# =============================================================
#  INSIGHTS TAB – Edge Sustainability & Market Adaptation (Part 4)
# =============================================================

def insights_tab_ui():
    """
    Long-term analytics for edge sustainability, correlation,
    and market-adaptation tracking.
    """
    st.header("🔍 Insights & Edge Sustainability")

    if not os.path.exists(RESULTS_FILE):
        st.info("No results logged yet. Run a few simulations first.")
        return

    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        st.info("No data available yet.")
        return

    # --- Pre-processing ---
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df["hit"] = (df["ev"] > 0).astype(int)

    # =========================================================
    # 1️⃣  CLV Trend – Are Your Edges Shrinking?
    # =========================================================
    st.subheader("💹 CLV Trend Over Time")

    fig1, ax1 = plt.subplots()
    ax1.plot(df["timestamp"], df["clv"], color=GOPHER_GOLD, linewidth=2, label="CLV (Projection – Line)")
    ax1.axhline(0, color="#555", linestyle="--")
    ax1.set_facecolor("#111")
    fig1.patch.set_facecolor("#111")
    ax1.tick_params(colors="white")
    ax1.set_ylabel("CLV", color="white")
    ax1.set_xlabel("Date", color="white")
    ax1.legend(facecolor="#111", labelcolor="white")
    st.pyplot(fig1)

    mean_clv = df["clv"].mean()
    slope_clv = np.polyfit(range(len(df)), df["clv"], 1)[0] if len(df) > 2 else 0
    if slope_clv < 0:
        st.warning("⚠️ Your average CLV is **declining** — the market may be adjusting faster to your edges.")
    else:
        st.success("✅ CLV trend stable or improving — your projections remain ahead of market moves.")

    # =========================================================
    # 2️⃣  Market Adaptation & EV Decay
    # =========================================================
    st.subheader("📉 Market Adaptation (EV Decay Analysis)")

    df["rolling_ev"] = df["ev"].rolling(10, min_periods=3).mean()
    fig2, ax2 = plt.subplots()
    ax2.plot(df["timestamp"], df["rolling_ev"], color=GOPHER_MAROON, linewidth=2, label="10-Run Rolling EV")
    ax2.axhline(0, color="#333", linestyle="--")
    ax2.set_facecolor("#111")
    fig2.patch.set_facecolor("#111")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#111", labelcolor="white")
    ax2.set_ylabel("EV %", color="white")
    st.pyplot(fig2)

    ev_trend = np.polyfit(range(len(df)), df["rolling_ev"].fillna(0), 1)[0] if len(df) > 2 else 0
    if ev_trend < 0:
        st.warning("🔻 EV is trending down. Your model’s predictive advantage may be tightening.")
    elif ev_trend > 0:
        st.success("🟢 EV trending upward — market still mispricing your targets.")
    else:
        st.info("⏸ No clear EV trend detected yet — collect more data.")

    # =========================================================
    # 3️⃣  Correlation Learning Between Legs
    # =========================================================
    st.subheader("🔗 Correlation Learning")

    corr_data = df[["player", "ev", "clv"]].dropna()
    if len(corr_data) > 5:
        corr_matrix = corr_data[["ev", "clv"]].corr().iloc[0, 1]
        st.metric("EV ↔ CLV Correlation", f"{corr_matrix:.2f}")
        if corr_matrix > 0.6:
            st.info("High positive correlation — your projections and line movement align closely.")
        elif corr_matrix < 0:
            st.warning("Negative correlation — projections may be misaligned with market adjustments.")
        else:
            st.success("Moderate correlation — market partially reflects your model, still exploitable.")
    else:
        st.caption("Not enough logged results for correlation learning yet.")

    # =========================================================
    # 4️⃣  ROI, Volatility & Drawdown Summary
    # =========================================================
    st.subheader("💰 ROI / Volatility / Drawdown Summary")

    df["roi"] = df["ev"] / 100
    df["cum_roi"] = (1 + df["roi"]).cumprod() - 1
    df["drawdown"] = df["cum_roi"] - df["cum_roi"].cummax()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total ROI", f"{df['cum_roi'].iloc[-1]*100:.2f}%")
    c2.metric("Avg Volatility", f"{df['variance'].mean():.2f}")
    c3.metric("Max Drawdown", f"{df['drawdown'].min()*100:.2f}%")

    fig3, ax3 = plt.subplots()
    ax3.plot(df["timestamp"], df["cum_roi"]*100, color=GOPHER_GOLD, linewidth=2, label="Cumulative ROI %")
    ax3.fill_between(df["timestamp"], df["drawdown"]*100, color="#7A0019", alpha=0.3, label="Drawdown %")
    ax3.set_facecolor("#111")
    fig3.patch.set_facecolor("#111")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#111", labelcolor="white")
    ax3.set_ylabel("ROI %", color="white")
    st.pyplot(fig3)

    st.caption(
        "ROI % represents compounded expected value performance. "
        "Drawdown % helps gauge risk exposure and streak sensitivity."
    )

    # =========================================================
    # 5️⃣  Summary Health Indicator
    # =========================================================
    st.markdown("---")
    st.subheader("🏁 Model Health Indicator")

    score = (
        (1 if mean_clv > 0 else 0)
        + (1 if ev_trend > 0 else 0)
        + (1 if df['drawdown'].min() > -0.3 else 0)
    )

    if score == 3:
        st.success("🟢 Model health excellent — sustainable edge detected.")
    elif score == 2:
        st.info("🟡 Model solid but monitor for EV compression over time.")
    elif score == 1:
        st.warning("🟠 Model weakening — tighten parameters and review variance scaling.")
    else:
        st.error("🔴 Model health poor — consider full recalibration cycle.")

    st.caption(
        "Health score derived from CLV trend, EV trend, and drawdown resilience. "
        "Values auto-update with each new result."
    )

# ---- End of Part 4 ----
