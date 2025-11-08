import streamlit as st
import requests
import difflib
import math
import os

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
        return {}, f"The Odds API request failed: {e}"

    if r.status_code != 200:
        return {}, f"The Odds API error {r.status_code}: {r.text}"

    try:
        data = r.json()
    except Exception:
        return {}, "Invalid JSON response from The Odds API."

    lines = {}
    for game in data:
        for bookmaker in game.get("bookmakers", []):
            if bookmaker.get("key") == "prizepicks":
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "player_points_rebounds_assists":
                        for outcome in market.get("outcomes", []):
                            player = outcome.get("description")
                            line_val = outcome.get("point")
                            if player and line_val:
                                lines[player.lower()] = float(line_val)
    if not lines:
        return {}, "No PrizePicks PRA lines found via The Odds API."
    return lines, f"Loaded {len(lines)} PRA lines."

def get_prizepicks_pra_line(player_name):
    lines, msg = load_prizepicks_pra_lines()
    if not lines:
        return None, msg

    name_norm = player_name.lower()
    if name_norm in lines:
        return lines[name_norm], "OK"

    best = difflib.get_close_matches(name_norm, lines.keys(), n=1, cutoff=0.6)
    if best:
        return lines[best[0]], f"Matched as '{best[0]}'"
    return None, f"No line for {player_name} — {msg}"

# =========================
# RUN MODEL
# =========================
run = st.button("Run Live Model")

if run:
    errors = []

    if payout_mult <= 1:
        errors.append("Payout multiplier must be > 1.")

    if not THE_ODDS_API_KEY:
        errors.append("Missing THE_ODDS_API_KEY in secrets.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # Try to load live lines
    p1_line, p1_msg = get_prizepicks_pra_line(p1_name) if use_p1_line else (p1_manual_line, "")
    p2_line, p2_msg = get_prizepicks_pra_line(p2_name) if use_p2_line else (p2_manual_line, "")

    if not p1_line:
        p1_line = p1_manual_line
    if not p2_line:
        p2_line = p2_manual_line

    st.warning(f"P1 line lookup: {p1_msg} — using {p1_line}")
    st.warning(f"P2 line lookup: {p2_msg} — using {p2_line}")

    # Example EV and Kelly calculation
    p1_prob_hit = 0.55
    p2_prob_hit = 0.57
    combo_prob = p1_prob_hit * p2_prob_hit
    ev = combo_prob * payout_mult - 1

    kelly_fraction = ((payout_mult * combo_prob - (1 - combo_prob)) / payout_mult)
    stake = bankroll * kelly_fraction * fractional_kelly

    st.subheader("Single-Leg Results")
    st.write(f"**{p1_name}**: {p1_prob_hit*100:.1f}% win probability, line {p1_line}")
    st.write(f"**{p2_name}**: {p2_prob_hit*100:.1f}% win probability, line {p2_line}")

    st.subheader("Combo & Risk Output")
    st.write(f"**2-Pick Probability:** {combo_prob*100:.2f}%")
    st.write(f"**Expected Value (EV):** {ev:.2f}")
    st.write(f"**Recommended Stake:** ${stake:.2f} ({fractional_kelly*100:.0f}% Kelly)")
    st.write(f"**Updated Bankroll after Win:** ${(bankroll + stake*(payout_mult-1)):.2f}")
