import streamlit as st
import requests
import difflib
import math
import os
from datetime import datetime

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="NBA PRA Prop Model", layout="wide")

st.title("🏀 NBA PRA 2-Pick Live Edge & Risk Model")
st.caption("Pulls PrizePicks PRA lines via The Odds API and computes optimal stakes with Kelly criteria.")

# =========================
# GLOBAL SETTINGS
# =========================
st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=10.0)
payout_mult = st.sidebar.number_input("2-Pick Payout Multiplier", min_value=1.0, value=3.0, step=0.1)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Games for API lookback", 5, 20, 10, 1)

# =========================
# PLAYER INPUTS
# =========================
st.subheader("Player Inputs")

col1, col2 = st.columns(2)

with col1:
    p1_name = st.text_input("Player 1 Name", "RJ Barrett")
    p1_minutes = st.number_input("P1 Projected Minutes", min_value=20.0, max_value=40.0, value=34.0)
    use_p1_line = st.checkbox("Use PrizePicks PRA line for P1", True)
    p1_manual_line = st.number_input("P1 Manual PRA Line (fallback)", min_value=10.0, max_value=70.0, value=33.5)

with col2:
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    p2_minutes = st.number_input("P2 Projected Minutes", min_value=20.0, max_value=40.0, value=32.0)
    use_p2_line = st.checkbox("Use PrizePicks PRA line for P2", True)
    p2_manual_line = st.number_input("P2 Manual PRA Line (fallback)", min_value=10.0, max_value=70.0, value=34.5)

# =========================
# THE ODDS API CONFIG
# =========================
THE_ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", os.getenv("THE_ODDS_API_KEY", ""))
ODDS_BASE = "https://api.the-odds-api.com/v4/sports"

@st.cache_data(show_spinner=False)
def load_prizepicks_pra_lines():
    """
    Pull PrizePicks PRA lines from The Odds API.
    """
    if not THE_ODDS_API_KEY:
        return {}, "Missing THE_ODDS_API_KEY in secrets."

    url = f"{ODDS_BASE}/basketball_nba/odds"
    params = {
        "regions": "us",
        "markets": "player_points_rebounds_assists",
        "bookmakers": "prizepicks",
        "apiKey": THE_ODDS_API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
    except Exception as e:
        return {}, f"API request failed: {e}"

    if r.status_code != 200:
        return {}, f"The Odds API error {r.status_code}: {r.text}"

    try:
        data = r.json()
    except Except
