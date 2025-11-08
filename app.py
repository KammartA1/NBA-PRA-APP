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
# CONFIG
# =========================

st.set_page_config(page_title="NBA PRA 2-Pick Live Edge", page_icon="🏀", layout="centered")
st.title("🏀 NBA PRA 2-Pick Live Edge (nba_api + The Odds API)")

st.markdown(
    """
This app:
- Uses **nba_api** to pull last N games for each player and compute PRA/min + volatility
- Uses **The Odds API** to pull **PrizePicks PRA** lines
- Computes:
  - Single-leg probability of OVER
  - EV per $1
  - Fractional Kelly stake (with safety cap)
  - 2-pick combo joint probability, EV, and optimal stake
"""
)
PROP_ODDS_API_KEY = st.secrets.get("PROP_ODDS_API_KEY", os.getenv("PROP_ODDS_API_KEY", ""))


TOA_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

MAX_BANKROLL_PCT = 0.03      # max 3% of bankroll on any single / combo
DEFAULT_LOOKBACK = 10        # last N games for PRA/min

# =========================
# UTILS
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
    if today.month >= 10:  # season starts in Oct
        start_year = year
    else:
        start_year = year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


# =========================
# nba_api: PLAYER LOOKUP
# =========================

@st.cache_data(show_spinner=False)
def nba_lookup_player(name: str):
    """
    Fuzzy-match a player name using nba_api's static player list.
    No external requests needed here.
    """
    all_players = nba_players.get_players()  # list of dicts
    target = _norm_name(name)

    # Build [normalized_name -> (id, full_name)]
    candidates = []
    for p in all_players:
        full = p.get("full_name", "")
        norm = _norm_name(full)
        candidates.append((norm, p["id"], full))

    # Exact match
    for norm, pid, full in candidates:
        if norm == target:
            return pid, full

    # Fuzzy match
    norm_names = [c[0] for c in candidates]
    best = difflib.get_close_matches(target, norm_names, n=1, cutoff=0.5)
    if best:
        chosen = best[0]
        for norm, pid, full in candidates:
            if norm == chosen:
                return pid, full

    return None, f"No NBA player match for '{name}'."


# =========================
# nba_api: LAST N GAMES PRA/MIN
# =========================

@st.cache_data(show_spinner=False)
def nba_get_pra_params(name: str, n_games: int):
    """
    Use nba_api PlayerGameLog for current season, last N games.
    Returns: (pra_per_min_mean, pra_per_min_sd, message)
    """
    pid, info = nba_lookup_player(name)
    if pid is None:
        return None, None, info

    season = current_season_str()
    try:
        gl = PlayerGameLog(player_id=pid, season=season, season_type_all_star="Regular Season")
        df = gl.get_data_frames()[0]
    except Exception as e:
        return None, None, f"Error fetching game log for {info}: {e}"

    if df.empty:
        return None, None, f"No game logs found for {info} in {season}."

    # Ensure sorted by date (newest first from API, but we sort anyway)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    pra_per_min = []
    for _, row in df.iterrows():
        pts = row.get("PTS", 0)
        reb = row.get("REB", 0)
        ast = row.get("AST", 0)
        mins = row.get("MIN", 0)

        # MIN often as string "34:12" or int/float
        if isinstance(mins, str) and ":" in mins:
            try:
                mm, ss = mins.split(":")
                minutes = float(mm) + float(ss) / 60.0
            except:
                minutes = 0.0
        else:
            try:
                minutes = float(mins)
            except:
                minutes = 0.0

        if minutes > 0:
            pra = pts + reb + ast
            pra_per_min.append(pra / minutes)

    if len(pra_per_min) < 3:
        return None, None, f"Not enough valid games for {info} (got {len(pra_per_min)})."

    arr = np.array(pra_per_min)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))

    if sd <= 0:
        sd = max(0.05, 0.1 * mu)

    return mu, sd, f"OK — {info}, last {len(pra_per_min)} games in {season}"


# =========================
# PROP ODDS API: PrizePicks PRA Lines (updated endpoint)
# =========================

PROP_ODDS_API_KEY = st.secrets.get("PROP_ODDS_API_KEY", os.getenv("PROP_ODDS_API_KEY", ""))
PROP_BASE = "https://api.prop-odds.com/v1"

@st.cache_data(show_spinner=False)
def load_prizepicks_pra_lines():
    """
    Pull PrizePicks PRA lines from Prop Odds API (v1).
    """
    if not PROP_ODDS_API_KEY:
        return {}, "Missing PROP_ODDS_API_KEY in secrets."

    url = f"{PROP_BASE}/dfs/picks"
    params = {
        "sport": "nba",
        "book": "prizepicks",
        "market": "player_points_rebounds_assists",
        "api_key": PROP_ODDS_API_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=8)
    except Exception as e:
        return {}, f"Prop Odds request failed: {e}"

    if r.status_code == 401:
        return {}, "Prop Odds 401: Invalid API key. Check your PROP_ODDS_API_KEY secret."
    if r.status_code == 403:
        return {}, "Prop Odds 403: Access forbidden (plan/tier issue)."
    if r.status_code == 404:
        return {}, "Prop Odds 404: Endpoint or market not found. Try again later or check plan."
    if r.status_code != 200:
        return {}, f"Prop Odds error {r.status_code}: {r.text}"

    try:
        data = r.json()
    except Exception:
        return {}, "Prop Odds returned invalid JSON."

    lines = {}
    picks = data.get("picks") or data.get("data") or []
    for item in picks:
        player = item.get("player_name") or item.get("name")
        line_val = item.get("line") or item.get("odds_value")
        market = item.get("market_name") or ""
        book = item.get("book_name") or ""
        if player and line_val and "rebounds" in market.lower() and "assists" in market.lower():
            lines[_norm_name(player)] = float(line_val)

    if not lines:
        return {}, "No PrizePicks PRA lines found via Prop Odds API."

    return lines, f"Loaded {len(lines)} PrizePicks PRA lines from Prop Odds (v1)."

def get_prizepicks_pra_line(player_name: str):
    """
    Retrieve a specific player's PRA line using fuzzy matching.
    """
    lines, msg = load_prizepicks_pra_lines()
    if not lines:
        return None, msg

    target = _norm_name(player_name)
    if target in lines:
        return lines[target], "OK"

    keys = list(lines.keys())
    best = difflib.get_close_matches(target, keys, n=1, cutoff=0.5)
    if best:
        match = best[0]
        return lines[match], f"Matched as '{match}'"

    return None, f"No PRA line for '{player_name}'. {msg}"


# =========================
# EV / Kelly model
# =========================

def compute_ev(line, pra_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    mu = pra_per_min * minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    if sd <= 0:
        sd = max(1.0, 0.15 * max(mu, 1.0))

    # Probability over threshold
    p_hat = 1.0 - norm.cdf(line, mu, sd)

    # EV per $1 for multiplier-style payout (win -> payout_mult, lose -> 0)
    ev = p_hat * (payout_mult - 1.0) - (1.0 - p_hat)

    b = payout_mult - 1.0
    full_k = 0.0
    if b > 0:
        full_k = max(0.0, (b * p_hat - (1.0 - p_hat)) / b)

    stake = bankroll * kelly_frac * full_k
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_hat, ev, stake, mu, sd, full_k


# =========================
# UI: GLOBAL SETTINGS
# =========================

st.sidebar.header("Global Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", value=1000.0, step=50.0)
payout_mult = st.sidebar.number_input("2-Pick Payout Multiplier", value=3.0, step=0.1)
kelly_frac = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
lookback = st.sidebar.slider("Games for PRA/min lookback", 5, 20, DEFAULT_LOOKBACK)

st.sidebar.caption(
    "For PrizePicks 2-pick Power Play, use 3.0x. "
    "Kelly 0.1–0.3 is safer for long-term growth."
)

# =========================
# UI: PLAYER INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Player 1")
    p1_name = st.text_input("Player 1 Name", "RJ Barrett")
    p1_proj_min = st.number_input("P1 Projected Minutes", value=34.0, step=1.0)
    auto_p1_line = st.checkbox("Use PrizePicks PRA line for P1", value=True)
    p1_manual_line = st.number_input("P1 Manual PRA Line (fallback)", value=33.5, step=0.5)

with col2:
    st.subheader("Player 2")
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    p2_proj_min = st.number_input("P2 Projected Minutes", value=32.0, step=1.0)
    auto_p2_line = st.checkbox("Use PrizePicks PRA line for P2", value=True)
    p2_manual_line = st.number_input("P2 Manual PRA Line (fallback)", value=34.5, step=0.5)

run = st.button("Run Live Model")

# =========================
# MAIN LOGIC
# =========================

if run:
    errors = []
    if payout_mult <= 1:
        errors.append("Payout multiplier must be > 1.")

    if not PROP_ODDS_API_KEY:
        errors.append("Missing PROP_ODDS_API_KEY in secrets.")

    # Get last N game stats via nba_api
    if not errors:
        p1_mu_min, p1_sd_min, p1_msg = nba_get_pra_params(p1_name, lookback)
        if p1_mu_min is None:
            errors.append(f"P1 stats: {p1_msg}")

        p2_mu_min, p2_sd_min, p2_msg = nba_get_pra_params(p2_name, lookback)
        if p2_mu_min is None:
            errors.append(f"P2 stats: {p2_msg}")

    if errors:
        for e in errors:
            st.error(e)
    else:
        # Lines from Odds API (or manual)
        p1_line = p1_manual_line
        if auto_p1_line:
            api_line, msg = get_prizepicks_pra_line(p1_name)
            if api_line is not None:
                p1_line = api_line
                st.info(f"P1 PrizePicks PRA: {api_line} ({msg})")
            else:
                st.warning(f"P1 line lookup: {msg} — using manual {p1_manual_line}")

        p2_line = p2_manual_line
        if auto_p2_line:
            api_line, msg = get_prizepicks_pra_line(p2_name)
            if api_line is not None:
                p2_line = api_line
                st.info(f"P2 PrizePicks PRA: {api_line} ({msg})")
            else:
                st.warning(f"P2 line lookup: {msg} — using manual {p2_manual_line}")

        # Compute single-leg edges
        p1, ev1, stake1, mu1, sd1, fk1 = compute_ev(
            p1_line, p1_mu_min, p1_sd_min, p1_proj_min, payout_mult, bankroll, kelly_frac
        )
        p2, ev2, stake2, mu2, sd2, fk2 = compute_ev(
            p2_line, p2_mu_min, p2_sd_min, p2_proj_min, payout_mult, bankroll, kelly_frac
        )

        # 2-pick combo (independent assumption)
        p_joint = p1 * p2
        ev_joint = payout_mult * p_joint - 1.0
        b_joint = payout_mult - 1.0
        fk_joint = max(0.0, (b_joint * p_joint - (1.0 - p_joint)) / b_joint) if b_joint > 0 else 0.0
        stake_joint = bankroll * kelly_frac * fk_joint
        stake_joint = min(stake_joint, bankroll * MAX_BANKROLL_PCT)
        stake_joint = max(0.0, round(stake_joint, 2))

        # ===== Display =====

        st.markdown("---")
        st.header("📊 Single-Leg Results")

        c1_out, c2_out = st.columns(2)

        with c1_out:
            st.subheader(p1_name)
            st.write(p1_msg)
            st.write(f"Line: **{p1_line}**, Proj Min: **{p1_proj_min}**")
            st.write(f"Model μ ≈ {mu1:.2f}, σ ≈ {sd1:.2f}")
            st.metric("Prob OVER", f"{p1:.1%}")
            st.metric("EV per $", f"{ev1*100:.1f}%")
            st.metric("Suggested Stake", f"${stake1:.2f}")
            st.success("✅ +EV leg") if ev1 > 0 else st.error("❌ -EV leg")

        with c2_out:
            st.subheader(p2_name)
            st.write(p2_msg)
            st.write(f"Line: **{p2_line}**, Proj Min: **{p2_proj_min}**")
            st.write(f"Model μ ≈ {mu2:.2f}, σ ≈ {sd2:.2f}")
            st.metric("Prob OVER", f"{p2:.1%}")
            st.metric("EV per $", f"{ev2*100:.1f}%")
            st.metric("Suggested Stake", f"${stake2:.2f}")
            st.success("✅ +EV leg") if ev2 > 0 else st.error("❌ -EV leg")

        st.markdown("---")
        st.header("🎯 2-Pick Combo (Both Must Hit)")

        st.metric("Joint Prob (both OVER)", f"{p_joint:.1%}")
        st.metric("EV per $", f"{ev_joint*100:.1f}%")
        st.metric("Recommended Combo Stake", f"${stake_joint:.2f}")
        st.success("✅ +EV combo") if ev_joint > 0 else st.error("❌ -EV combo")

st.caption(
    "This uses nba_api for last-N game logs and The Odds API for PrizePicks lines. "
    "If nba_api calls fail (e.g. due to environment or NBA rate limits), you'll see the exact error above."
)
