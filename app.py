import os
import difflib
from datetime import datetime, timedelta, timezone

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

st.markdown(
    """
This app:
- Uses **nba_api** to pull last N games for each player and estimate PRA per minute + volatility  
- Uses **The Odds API** to pull **PrizePicks PRA** lines  
- Computes:
  - Single-leg probability (OVER)
  - EV per $1
  - Fractional Kelly stake with a safety cap
  - 2-pick combo (both must hit) probability, EV, and stake
"""
)

# =========================
# GLOBAL SETTINGS (SIDEBAR)
# =========================

st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=10.0)
payout_mult = st.sidebar.number_input("2-Pick Payout Multiplier", min_value=1.01, value=3.0, step=0.1)
kelly_frac = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
lookback = st.sidebar.slider("Games for PRA/min lookback", 5, 20, 10, 1)

st.sidebar.caption(
    "For PrizePicks 2-pick Power Play use 3.0x. "
    "Use smaller Kelly (0.1–0.3) to control variance."
)

MAX_BANKROLL_PCT = 0.03  # hard cap: 3% of bankroll on any position

# =========================
# PLAYER INPUTS
# =========================

st.subheader("Player Inputs")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Player 1")
    p1_name = st.text_input("Player 1 Name", "RJ Barrett")
    p1_proj_min = st.number_input("P1 Projected Minutes", min_value=20.0, max_value=44.0, value=34.0, step=1.0)
    use_p1_line = st.checkbox("Use PrizePicks PRA line for P1", value=True)
    p1_manual_line = st.number_input("P1 Manual PRA Line (fallback)", min_value=5.0, max_value=80.0, value=33.5, step=0.5)

with col2:
    st.markdown("### Player 2")
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    p2_proj_min = st.number_input("P2 Projected Minutes", min_value=20.0, max_value=44.0, value=32.0, step=1.0)
    use_p2_line = st.checkbox("Use PrizePicks PRA line for P2", value=True)
    p2_manual_line = st.number_input("P2 Manual PRA Line (fallback)", min_value=5.0, max_value=80.0, value=34.5, step=0.5)

run = st.button("Run Live Model")

# =========================
# THE ODDS API CONFIG
# =========================

THE_ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", os.getenv("THE_ODDS_API_KEY", ""))
ODDS_BASE = "https://api.the-odds-api.com/v4/sports"


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


def current_season_str() -> str:
    """Return NBA season string like '2024-25' based on today's date."""
    today = datetime.now()
    year = today.year
    if today.month >= 10:
        start_year = year
    else:
        start_year = year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


# ---------- nba_api: player lookup ----------

@st.cache_data(show_spinner=False)
def nba_lookup_player(name: str):
    all_players = nba_players.get_players()
    target = _norm_name(name)

    candidates = []
    for p in all_players:
        full = p.get("full_name", "")
        normed = _norm_name(full)
        candidates.append((normed, p["id"], full))

    # exact
    for normed, pid, full in candidates:
        if normed == target:
            return pid, full

    # fuzzy
    norm_names = [c[0] for c in candidates]
    best = difflib.get_close_matches(target, norm_names, n=1, cutoff=0.5)
    if best:
        chosen = best[0]
        for normed, pid, full in candidates:
            if normed == chosen:
                return pid, full

    return None, f"No NBA player match for '{name}'."


# ---------- nba_api: last N games PRA/min ----------

@st.cache_data(show_spinner=False)
def nba_get_pra_params(name: str, n_games: int):
    """
    Fetch last N games via nba_api and compute PRA/min mean & sd.
    """
    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, label

    season = current_season_str()

    try:
        gl = PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
        df = gl.get_data_frames()[0]
    except Exception as e:
        return None, None, f"Game log error for {label}: {e}"

    if df.empty:
        return None, None, f"No logs found for {label} in {season}."

    # newest first, take N
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    pra_per_min = []

    for _, row in df.iterrows():
        pts = float(row.get("PTS", 0))
        reb = float(row.get("REB", 0))
        ast = float(row.get("AST", 0))
        mins_raw = row.get("MIN", 0)

        if isinstance(mins_raw, str) and ":" in mins_raw:
            try:
                mm, ss = mins_raw.split(":")
                minutes = float(mm) + float(ss) / 60.0
            except Exception:
                minutes = 0.0
        else:
            try:
                minutes = float(mins_raw)
            except Exception:
                minutes = 0.0

        if minutes > 0:
            pra_per_min.append((pts + reb + ast) / minutes)

    if len(pra_per_min) < 3:
        return None, None, f"Not enough valid games for {label} (got {len(pra_per_min)})."

    arr = np.array(pra_per_min)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 0:
        sd = max(0.05, 0.1 * mu)

    return mu, sd, f"{label}: using last {len(pra_per_min)} games ({season})"


# ---------- The Odds API: PrizePicks PRA lines ----------

@st.cache_data(show_spinner=False)
def load_prizepicks_pra_lines():
    """
    Load PrizePicks PRA lines via The Odds API.
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

    if r.status_code == 401:
        return {}, "The Odds API 401: invalid or expired API key."
    if r.status_code == 403:
        return {}, "The Odds API 403: forbidden (check plan/usage)."
    if r.status_code != 200:
        return {}, f"The Odds API error {r.status_code}: {r.text}"

    try:
        data = r.json()
    except Exception:
        return {}, "Invalid JSON response from The Odds API."

    lines = {}

    for game in data:
        for bookmaker in game.get("bookmakers", []):
            if bookmaker.get("key") != "prizepicks":
                continue
            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_points_rebounds_assists":
                    continue
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description") or outcome.get("name")
                    line_val = outcome.get("point")
                    if player and line_val is not None:
                        lines[_norm_name(player)] = float(line_val)

    if not lines:
        return {}, "No PrizePicks PRA lines found via The Odds API."

    return lines, f"Loaded {len(lines)} PrizePicks PRA lines."


def get_prizepicks_pra_line(player_name: str):
    lines, msg = load_prizepicks_pra_lines()
    if not lines:
        return None, msg

    target = _norm_name(player_name)
    if target in lines:
        return lines[target], "OK"

    keys = list(lines.keys())
    best = difflib.get_closematches(target, keys, n=1, cutoff=0.5) if hasattr(difflib, "get_closematches") else difflib.get_close_matches(target, keys, n=1, cutoff=0.5)
    # (guard in case of weird environment; but get_close_matches exists)

    # simpler:
    best = difflib.get_close_matches(target, keys, n=1, cutoff=0.5)
    if best:
        match = best[0]
        return lines[match], f"Matched as '{match}'"

    return None, f"No PRA line for '{player_name}'. {msg}"


# ---------- EV & Kelly ----------

def compute_ev(line, pra_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    """
    Normal model:
      X ~ N(mu, sd^2)
      mu = pra_per_min * minutes
      sd = sd_per_min * sqrt(minutes)
    Returns (p_over, ev_per_dollar, stake, mu, sd, full_kelly_fraction)
    """
    mu = pra_per_min * minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    if sd <= 0:
        sd = max(1.0, 0.15 * max(mu, 1.0))

    # Probability hitting OVER
    p_over = 1.0 - norm.cdf(line, mu, sd)

    # EV per $1 with multiplier payoff (win -> payout_mult, lose -> 0)
    ev = p_over * (payout_mult - 1.0) - (1.0 - p_over)

    b = payout_mult - 1.0
    full_k = 0.0
    if b > 0:
        full_k = max(0.0, (b * p_over - (1.0 - p_over)) / b)

    stake = bankroll * kelly_frac * full_k
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_over, ev, stake, mu, sd, full_k


# =========================
# MAIN RUN LOGIC
# =========================

if run:
    errors = []

    if payout_mult <= 1.0:
        errors.append("Payout multiplier must be > 1.")
    if not THE_ODDS_API_KEY:
        errors.append("Missing THE_ODDS_API_KEY in secrets.")

    # Get nba_api stats
    if not errors:
        p1_mu_min, p1_sd_min, p1_stats_msg = nba_get_pra_params(p1_name, lookback)
        if p1_mu_min is None:
            errors.append(f"P1 stats: {p1_stats_msg}")

        p2_mu_min, p2_sd_min, p2_stats_msg = nba_get_pra_params(p2_name, lookback)
        if p2_mu_min is None:
            errors.append(f"P2 stats: {p2_stats_msg}")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ===== Lines from The Odds API (with manual fallback) =====
    # P1
    p1_line = p1_manual_line
    if use_p1_line:
        api_line, msg = get_prizepicks_pra_line(p1_name)
        if api_line is not None:
            p1_line = api_line
            st.info(f"P1 line from PrizePicks/The Odds API: {api_line} ({msg})")
        else:
            st.warning(f"P1 line lookup: {msg} — using manual {p1_manual_line}")

    # P2
    p2_line = p2_manual_line
    if use_p2_line:
        api_line, msg = get_prizepicks_pra_line(p2_name)
        if api_line is not None:
            p2_line = api_line
            st.info(f"P2 line from PrizePicks/The Odds API: {api_line} ({msg})")
        else:
            st.warning(f"P2 line lookup: {msg} — using manual {p2_manual_line}")

    # ===== Compute single-leg outputs =====
    p1_prob, ev1, stake1, mu1, sd1, fk1 = compute_ev(
        p1_line, p1_mu_min, p1_sd_min, p1_proj_min, payout_mult, bankroll, kelly_frac
    )
    p2_prob, ev2, stake2, mu2, sd2, fk2 = compute_ev(
        p2_line, p2_mu_min, p2_sd_min, p2_proj_min, payout_mult, bankroll, kelly_frac
    )

    # ===== 2-pick combo (independent assumption) =====
    joint_prob = p1_prob * p2_prob
    ev_combo = payout_mult * joint_prob - 1.0

    b_combo = payout_mult - 1.0
    full_k_combo = max(0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo) if b_combo > 0 else 0.0
    stake_combo = bankroll * kelly_frac * full_k_combo
    stake_combo = min(stake_combo, bankroll * MAX_BANKROLL_PCT)
    stake_combo = max(0.0, round(stake_combo, 2))

    # =========================
    # DISPLAY RESULTS
    # =========================

    st.markdown("---")
    st.header("📊 Single-Leg Results")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader(p1_name)
        st.write(p1_stats_msg)
        st.write(f"Line: **{p1_line}**, Proj Min: **{p1_proj_min}**")
        st.write(f"Model μ ≈ {mu1:.2f}, σ ≈ {sd1:.2f}")
        st.metric("Prob OVER", f"{p1_prob:.1%}")
        st.metric("EV per $", f"{ev1*100:.1f}%")
        st.metric("Suggested Stake", f"${stake1:.2f}")
        st.success("✅ +EV leg") if ev1 > 0 else st.error("❌ -EV leg")

    with c2:
        st.subheader(p2_name)
        st.write(p2_stats_msg)
        st.write(f"Line: **{p2_line}**, Proj Min: **{p2_proj_min}**")
        st.write(f"Model μ ≈ {mu2:.2f}, σ ≈ {sd2:.2f}")
        st.metric("Prob OVER", f"{p2_prob:.1%}")
        st.metric("EV per $", f"{ev2*100:.1f}%")
        st.metric("Suggested Stake", f"${stake2:.2f}")
        st.success("✅ +EV leg") if ev2 > 0 else st.error("❌ -EV leg")

    st.markdown("---")
    st.header("🎯 2-Pick Combo (Both Must Hit)")

    st.metric("Joint Prob (both OVER)", f"{joint_prob:.1%}")
    st.metric("EV per $", f"{ev_combo*100:.1f}%")
    st.metric("Recommended Combo Stake", f"${stake_combo:.2f}")

    if ev_combo > 0:
        st.success("✅ Combo is +EV under current model.")
    else:
        st.error("❌ Combo is -EV. Do not force it.")

st.caption(
    "If nba_api or The Odds API calls fail (rate limits, network, etc.), the app will "
    "show clear messages above. Always sanity-check projected minutes and roles."
)
