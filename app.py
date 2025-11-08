import os
import difflib
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NBA Prop Edge Model", page_icon="🏀", layout="wide")
st.title("🏀 NBA 2-Pick Prop Edge & Risk Model")

st.markdown("""
This app:
- Uses **nba_api** to model player efficiency and estimate projected minutes  
- Pulls **PrizePicks lines** from The Odds API  
- Computes hit probability, expected value (EV), and optimal stake (Kelly)  
- Supports PRA, Points, Rebounds, and Assists markets
""")

# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=10.0)
payout_mult = st.sidebar.number_input("2-Pick Payout Multiplier", min_value=1.0, value=3.0, step=0.1)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Games for API lookback", 5, 20, 10, 1)
st.sidebar.caption("For PrizePicks Power Play use 3.0x. Use smaller Kelly (0.1–0.3) to reduce risk.")
MAX_BANKROLL_PCT = 0.03

# =========================
# MARKET SELECTOR
# =========================
market = st.selectbox(
    "Select Prop Market",
    ["PRA (Points + Rebounds + Assists)", "Points", "Rebounds", "Assists"],
    index=0
)
market_map = {
    "PRA (Points + Rebounds + Assists)": "player_points_rebounds_assists",
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
}
selected_market_key = market_map[market]

# =========================
# PLAYER INPUTS
# =========================
st.subheader("Player Inputs")
col1, col2 = st.columns(2)

with col1:
    p1_name = st.text_input("Player 1 Name", "RJ Barrett")
    use_p1_line = st.checkbox("Use PrizePicks line for P1", True)
    p1_manual_line = st.number_input("P1 Manual Line (fallback)", min_value=10.0, max_value=80.0, value=33.5)

with col2:
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    use_p2_line = st.checkbox("Use PrizePicks line for P2", True)
    p2_manual_line = st.number_input("P2 Manual Line (fallback)", min_value=10.0, max_value=80.0, value=34.5)

# =========================
# API KEYS
# =========================
THE_ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", os.getenv("THE_ODDS_API_KEY", ""))
ODDS_BASE = "https://api.the-odds-api.com/v4/sports"

# =========================
# HELPERS
# =========================
def _norm_name(s):
    return s.lower().replace(".", "").replace("'", "").replace("-", " ").strip()

def current_season():
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"

@st.cache_data
def nba_lookup_player(name):
    all_players = nba_players.get_players()
    norm_target = _norm_name(name)
    for p in all_players:
        if _norm_name(p["full_name"]) == norm_target:
            return p["id"], p["full_name"]
    best = difflib.get_close_matches(norm_target, [_norm_name(p["full_name"]) for p in all_players], n=1, cutoff=0.6)
    if best:
        for p in all_players:
            if _norm_name(p["full_name"]) == best[0]:
                return p["id"], p["full_name"]
    return None, f"No NBA player match for '{name}'"

@st.cache_data
def nba_get_player_params(name, n_games, market_type):
    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, None, f"Could not find player {name}."

    try:
        df = PlayerGameLog(player_id=pid, season=current_season(), season_type_all_star="Regular Season").get_data_frames()[0]
    except Exception as e:
        return None, None, None, f"Error fetching logs for {label}: {e}"

    if df.empty:
        return None, None, None, f"No logs found for {label}"

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    metric_map = {
        "player_points_rebounds_assists": ["PTS", "REB", "AST"],
        "player_points": ["PTS"],
        "player_rebounds": ["REB"],
        "player_assists": ["AST"]
    }
    cols = metric_map[market_type]

    vals_per_min, mins_list = [], []
    for _, row in df.iterrows():
        total = sum(float(row.get(c, 0)) for c in cols)
        mins_raw = row.get("MIN", 0)
        try:
            if isinstance(mins_raw, str) and ":" in mins_raw:
                m, s = mins_raw.split(":")
                minutes = float(m) + float(s)/60.0
            else:
                minutes = float(mins_raw)
        except Exception:
            minutes = 0
        if minutes > 0:
            mins_list.append(minutes)
            vals_per_min.append(total / minutes)

    if len(vals_per_min) < 3:
        return None, None, None, f"Not enough data for {label}"

    avg_min = float(np.mean(mins_list))
    mu = np.mean(vals_per_min)
    sd = np.std(vals_per_min, ddof=1)
    return mu, sd, avg_min, f"{label}: {len(vals_per_min)} games analyzed ({current_season()})"

@st.cache_data
@st.cache_data
def load_prizepicks_lines(market_key):
    if not THE_ODDS_API_KEY:
        return {}, "Missing THE_ODDS_API_KEY in secrets."

    url = f"{ODDS_BASE}/basketball_nba/odds"
    params = {
        "regions": "us",
        "markets": market_key,
        "bookmakers": "prizepicks",
        "oddsFormat": "american",
        "apiKey": THE_ODDS_API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {}, f"The Odds API error {r.status_code}: {r.text}"
        data = r.json()
    except Exception as e:
        return {}, f"Request failed: {e}"

    lines = {}
    for game in data:
        for bookmaker in game.get("bookmakers", []):
            if bookmaker.get("key") == "prizepicks":
                for market in bookmaker.get("markets", []):
                    if market.get("key") == market_key:
                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name")
                            line = outcome.get("point")
                            if name and line:
                                lines[_norm_name(name)] = float(line)

    if not lines:
        return {}, f"No PrizePicks {market_key} lines found."
    return lines, f"Loaded {len(lines)} {market_key} lines successfully."

    params = {"regions": "us", "markets": market_key, "bookmakers": "prizepicks", "apiKey": THE_ODDS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return {}, f"The Odds API error {r.status_code}: {r.text}"
        data = r.json()
    except Exception as e:
        return {}, f"Request failed: {e}"
    lines = {}
    for prop in data:
        if prop.get("bookmaker_key") == "prizepicks" and prop.get("market_key") == market_key:
            player = prop.get("player_name")
            val = prop.get("odds", {}).get("line")
            if player and val:
                lines[_norm_name(player)] = float(val)
    if not lines:
        return {}, "No lines found."
    return lines, f"Loaded {len(lines)} {market_key} lines."

def get_prizepicks_line(name, market_key):
    lines, msg = load_prizepicks_lines(market_key)
    if not lines:
        return None, msg
    key = _norm_name(name)
    if key in lines:
        return lines[key], "OK"
    best = difflib.get_close_matches(key, lines.keys(), n=1, cutoff=0.6)
    if best:
        return lines[best[0]], f"Matched '{best[0]}'"
    return None, f"No line found for {name}"

def compute_ev(line, val_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    mu = val_per_min * minutes
    sd = sd_per_min * np.sqrt(minutes)
    p_over = 1 - norm.cdf(line, mu, sd)
    ev = p_over * (payout_mult - 1) - (1 - p_over)
    b = payout_mult - 1
    full_k = max(0, (b * p_over - (1 - p_over)) / b)
    stake = min(bankroll * kelly_frac * full_k, bankroll * MAX_BANKROLL_PCT)
    return p_over, ev, stake, mu, sd

# =========================
# MAIN RUN
# =========================
if st.button("Run Live Model"):
    if not THE_ODDS_API_KEY:
        st.error("Missing THE_ODDS_API_KEY in secrets.")
        st.stop()

    p1_mu, p1_sd, p1_avg_min, msg1 = nba_get_player_params(p1_name, games_lookback, selected_market_key)
    p2_mu, p2_sd, p2_avg_min, msg2 = nba_get_player_params(p2_name, games_lookback, selected_market_key)

    if not p1_mu or not p2_mu:
        st.error("Error loading player stats.")
        st.stop()

    p1_line, msg = get_prizepicks_line(p1_name, selected_market_key) if use_p1_line else (p1_manual_line, "manual")
    if not p1_line:
        p1_line = p1_manual_line
        st.warning(f"P1 line not found ({msg}) — using manual {p1_manual_line}")

    p2_line, msg = get_prizepicks_line(p2_name, selected_market_key) if use_p2_line else (p2_manual_line, "manual")
    if not p2_line:
        p2_line = p2_manual_line
        st.warning(f"P2 line not found ({msg}) — using manual {p2_manual_line}")

    # Compute EV using auto projected minutes
    p1_prob, ev1, stake1, mu1, sd1 = compute_ev(p1_line, p1_mu, p1_sd, p1_avg_min, payout_mult, bankroll, fractional_kelly)
    p2_prob, ev2, stake2, mu2, sd2 = compute_ev(p2_line, p2_mu, p2_sd, p2_avg_min, payout_mult, bankroll, fractional_kelly)
    combo_prob = p1_prob * p2_prob
    ev_combo = payout_mult * combo_prob - 1
    stake_combo = min(bankroll * fractional_kelly * ((payout_mult * combo_prob - (1 - combo_prob)) / (payout_mult - 1)), bankroll * MAX_BANKROLL_PCT)

    # =========================
    # DISPLAY
    # =========================
    st.markdown("### 📊 Single-Leg Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(p1_name)
        st.write(msg1)
        st.metric("Auto Projected Minutes", f"{p1_avg_min:.1f}")
        st.metric("Line", p1_line)
        st.metric("Prob OVER", f"{p1_prob:.1%}")
        st.metric("EV per $", f"{ev1*100:.1f}%")
        st.metric("Suggested Stake", f"${stake1:.2f}")
        st.success("✅ +EV leg") if ev1 > 0 else st.error("❌ -EV leg")

    with col2:
        st.subheader(p2_name)
        st.write(msg2)
        st.metric("Auto Projected Minutes", f"{p2_avg_min:.1f}")
        st.metric("Line", p2_line)
        st.metric("Prob OVER", f"{p2_prob:.1%}")
        st.metric("EV per $", f"{ev2*100:.1f}%")
        st.metric("Suggested Stake", f"${stake2:.2f}")
        st.success("✅ +EV leg") if ev2 > 0 else st.error("❌ -EV leg")

    st.markdown("---")
    st.header("🎯 2-Pick Combo (Both Must Hit)")
    st.metric("Joint Prob", f"{combo_prob:.1%}")
    st.metric("EV per $", f"{ev_combo*100:.1f}%")
    st.metric("Recommended Stake", f"${stake_combo:.2f}")
    st.success("✅ Combo is +EV") if ev_combo > 0 else st.error("❌ Combo is -EV")

st.caption("App uses live Odds API lines and auto-estimates minutes from recent NBA logs.")
