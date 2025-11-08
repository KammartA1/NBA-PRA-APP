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
st.set_page_config(page_title="NBA PRA 2-Pick Edge", page_icon="🏀", layout="wide")
st.title("🏀 NBA PRA 2-Pick Live Edge & Risk Model")

st.markdown("""
This app:
- Uses **nba_api** for last N games to model PRA per minute and volatility  
- Pulls **PrizePicks PRA lines** via The Odds API  
- Calculates probabilities, expected value (EV), and optimal stake using Kelly criterion
""")


# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=10.0)
payout_mult = st.sidebar.number_input("2-Pick Payout Multiplier", min_value=1.0, value=3.0, step=0.1)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Games for PRA/min lookback", 5, 20, 10, 1)
st.sidebar.caption("For PrizePicks 2-pick Power Play use 3.0x. Lower Kelly (0.1–0.3) = safer growth.")

MAX_BANKROLL_PCT = 0.03


# =========================
# PLAYER INPUTS
# =========================
st.subheader("Player Inputs")

col1, col2 = st.columns(2)

with col1:
    p1_name = st.text_input("Player 1 Name", "RJ Barrett")
    p1_minutes = st.number_input("P1 Projected Minutes", min_value=20.0, max_value=44.0, value=34.0)
    use_p1_line = st.checkbox("Use PrizePicks PRA line for P1", True)
    p1_manual_line = st.number_input("P1 Manual PRA Line (fallback)", min_value=10.0, max_value=80.0, value=33.5)

with col2:
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    p2_minutes = st.number_input("P2 Projected Minutes", min_value=20.0, max_value=44.0, value=32.0)
    use_p2_line = st.checkbox("Use PrizePicks PRA line for P2", True)
    p2_manual_line = st.number_input("P2 Manual PRA Line (fallback)", min_value=10.0, max_value=80.0, value=34.5)


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
    year = today.year
    start_year = year if today.month >= 10 else year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


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
    return None, f"No NBA player match for '{name}'."


@st.cache_data
def nba_get_pra_params(name, n_games):
    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, label

    season = current_season()
    try:
        df = PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season").get_data_frames()[0]
    except Exception as e:
        return None, None, f"Error fetching logs for {label}: {e}"

    if df.empty:
        return None, None, f"No logs found for {label}."

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    pra_per_min = []
    for _, row in df.iterrows():
        pts, reb, ast = row.get("PTS", 0), row.get("REB", 0), row.get("AST", 0)
        mins_raw = row.get("MIN", 0)
        try:
            if isinstance(mins_raw, str) and ":" in mins_raw:
                m, s = mins_raw.split(":")
                minutes = float(m) + float(s) / 60.0
            else:
                minutes = float(mins_raw)
        except Exception:
            minutes = 0

        if minutes > 0:
            pra_per_min.append((pts + reb + ast) / minutes)

    if len(pra_per_min) < 3:
        return None, None, f"Not enough valid games for {label}."

    arr = np.array(pra_per_min)
    return arr.mean(), arr.std(ddof=1), f"{label}: using last {len(pra_per_min)} games ({season})"


# =========================
# FIXED PRIZEPICKS FETCHER (props endpoint)
# =========================
@st.cache_data
def load_prizepicks_pra_lines():
    if not THE_ODDS_API_KEY:
        return {}, "Missing THE_ODDS_API_KEY in secrets."

    url = f"{ODDS_BASE}/basketball_nba/props"
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
        return {}, "Invalid JSON response."

    lines = {}
    for prop in data:
        if prop.get("bookmaker_key") == "prizepicks":
            if prop.get("market_key") == "player_points_rebounds_assists":
                player = prop.get("player_name")
                val = prop.get("odds", {}).get("line")
                if player and val:
                    lines[_norm_name(player)] = float(val)

    if not lines:
        return {}, "No PrizePicks PRA lines found."
    return lines, f"Loaded {len(lines)} PRA lines."


def get_prizepicks_pra_line(name):
    lines, msg = load_prizepicks_pra_lines()
    if not lines:
        return None, msg
    norm_name = _norm_name(name)
    if norm_name in lines:
        return lines[norm_name], "OK"
    best = difflib.get_close_matches(norm_name, lines.keys(), n=1, cutoff=0.6)
    if best:
        return lines[best[0]], f"Matched '{best[0]}'"
    return None, f"No PRA line found for {name}."


# =========================
# EV + KELLY CALC
# =========================
def compute_ev(line, pra_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    mu = pra_per_min * minutes
    sd = sd_per_min * np.sqrt(minutes)
    if sd <= 0:
        sd = max(1.0, 0.15 * mu)

    p_over = 1 - norm.cdf(line, mu, sd)
    ev = p_over * (payout_mult - 1) - (1 - p_over)
    b = payout_mult - 1
    full_k = max(0, (b * p_over - (1 - p_over)) / b) if b > 0 else 0
    stake = bankroll * kelly_frac * full_k
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    return p_over, ev, stake, mu, sd


# =========================
# MAIN RUN
# =========================
if st.button("Run Live Model"):
    errors = []

    if not THE_ODDS_API_KEY:
        errors.append("Missing THE_ODDS_API_KEY in secrets.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    p1_mu, p1_sd, msg1 = nba_get_pra_params(p1_name, games_lookback)
    p2_mu, p2_sd, msg2 = nba_get_pra_params(p2_name, games_lookback)

    if not p1_mu or not p2_mu:
        st.error("Error pulling player data. Check logs.")
        st.stop()

    # Try PrizePicks lines
    p1_line, msg = (get_prizepicks_pra_line(p1_name) if use_p1_line else (p1_manual_line, "manual"))
    if not p1_line:
        p1_line = p1_manual_line
        st.warning(f"P1 line lookup failed ({msg}) — using manual {p1_manual_line}")

    p2_line, msg = (get_prizepicks_pra_line(p2_name) if use_p2_line else (p2_manual_line, "manual"))
    if not p2_line:
        p2_line = p2_manual_line
        st.warning(f"P2 line lookup failed ({msg}) — using manual {p2_manual_line}")

    # EV computation
    p1_prob, ev1, stake1, mu1, sd1 = compute_ev(p1_line, p1_mu, p1_sd, p1_minutes, payout_mult, bankroll, fractional_kelly)
    p2_prob, ev2, stake2, mu2, sd2 = compute_ev(p2_line, p2_mu, p2_sd, p2_minutes, payout_mult, bankroll, fractional_kelly)

    combo_prob = p1_prob * p2_prob
    ev_combo = payout_mult * combo_prob - 1
    stake_combo = min(bankroll * fractional_kelly * ((payout_mult * combo_prob - (1 - combo_prob)) / (payout_mult - 1)),
                      bankroll * MAX_BANKROLL_PCT)

    # =========================
    # DISPLAY
    # =========================
    st.markdown("### 📊 Single-Leg Results")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(p1_name)
        st.write(msg1)
        st.write(f"Line: **{p1_line}**, Proj Min: **{p1_minutes}**")
        st.metric("Prob OVER", f"{p1_prob:.1%}")
        st.metric("EV per $", f"{ev1*100:.1f}%")
        st.metric("Suggested Stake", f"${stake1:.2f}")
        if ev1 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")

    with col2:
        st.subheader(p2_name)
        st.write(msg2)
        st.write(f"Line: **{p2_line}**, Proj Min: **{p2_minutes}**")
        st.metric("Prob OVER", f"{p2_prob:.1%}")
        st.metric("EV per $", f"{ev2*100:.1f}%")
        st.metric("Suggested Stake", f"${stake2:.2f}")
        if ev2 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")

    st.markdown("---")
    st.header("🎯 2-Pick Combo (Both Must Hit)")
    st.metric("Joint Prob", f"{combo_prob:.1%}")
    st.metric("EV per $", f"{ev_combo*100:.1f}%")
    st.metric("Recommended Stake", f"${stake_combo:.2f}")
    if ev_combo > 0:
        st.success("✅ Combo is +EV")
    else:
        st.error("❌ Combo is -EV")

st.caption("If the Odds API rate-limits or fails, app uses fallback manual lines.")
