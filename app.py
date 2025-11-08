import streamlit as st
import numpy as np
from scipy.stats import norm
import requests
import os

st.set_page_config(page_title="NBA PRA 2-Pick Edge", page_icon="🏀", layout="centered")
st.title("🏀 NBA PRA 2-Pick Edge Calculator")

st.markdown(
    "Type player names, auto-fill last 10-game PRA/min from the API, "
    "set minutes & lines, and get probabilities, EV, and suggested stakes."
)

# ---------- CONFIG ----------
API_BASE = "https://api.balldontlie.io/v1"
API_KEY = st.secrets.get("BALLDONTLIE_API_KEY", os.getenv("BALLDONTLIE_API_KEY", ""))

HEADERS = {"Authorization": API_KEY} if API_KEY else {}

DEFAULT_N_GAMES = 10
MAX_BANKROLL_PCT = 0.03  # 3% cap


# ---------- HELPERS ----------

def get_player_id(player_name: str):
    """Search player ID by name using balldontlie players endpoint."""
    if not API_KEY:
        return None, "No API key set. Add BALLDONTLIE_API_KEY in Streamlit secrets."
    params = {"search": player_name, "per_page": 10}
    try:
        r = requests.get(f"{API_BASE}/players", headers=HEADERS, params=params, timeout=5)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None, f"No player found for '{player_name}'."
        # naive: take first match
        p = data[0]
        full_name = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        return p.get("id"), full_name
    except Exception as e:
        return None, f"Error fetching player id for '{player_name}': {e}"


def get_last_n_game_stats(player_id: int, n: int = DEFAULT_N_GAMES):
    """
    Fetch last N games' basic stats for player via balldontlie 'stats' or 'box scores' style endpoint.
    NOTE: Some endpoints require paid tiers; adjust based on your account.
    """
    if not API_KEY:
        return None, "No API key set."
    # Using game player stats endpoint requires appropriate tier.
    # We'll request latest stats and slice last N.
    params = {
        "player_ids[]": player_id,
        "per_page": n,
        "sort": "game.date",   # latest first if supported
    }
    try:
        r = requests.get(f"{API_BASE}/stats", headers=HEADERS, params=params, timeout=5)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None, "No recent stats found (check player or tier)."
        return data[:n], None
    except Exception as e:
        return None, f"Error fetching stats: {e}"


def estimate_pra_params_from_api(player_name: str, n: int = DEFAULT_N_GAMES):
    """
    For a given player name:
    - get player id
    - fetch last N game logs
    - compute PRA_per_min mean and sd
    """
    pid, info = get_player_id(player_name)
    if pid is None:
        return None, None, info  # error message in info

    stats, err = get_last_n_game_stats(pid, n)
    if stats is None:
        return None, None, err

    pra_per_min_values = []
    for g in stats:
        # fields based on balldontlie stats schema
        pts = g.get("pts") or g.get("points") or 0
        reb = g.get("reb") or g.get("rebounds") or 0
        ast = g.get("ast") or g.get("assists") or 0
        min_str = g.get("min") or g.get("minutes") or "0"
        # convert "MM:SS" to float minutes if needed
        if isinstance(min_str, str) and ":" in min_str:
            mm, ss = min_str.split(":")
            minutes = float(mm) + float(ss) / 60.0
        else:
            try:
                minutes = float(min_str)
            except:
                minutes = 0.0

        if minutes > 0:
            pra = pts + reb + ast
            pra_per_min_values.append(pra / minutes)

    if len(pra_per_min_values) < 3:
        return None, None, f"Not enough valid games for {info}."

    arr = np.array(pra_per_min_values)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    # floor sd slightly to avoid zero
    if sd <= 0:
        sd = max(0.05, 0.1 * mu)

    return mu, sd, f"OK (using last {len(arr)} games for {info})"


def compute_ev(line, pra_per_min, sd_per_min, minutes, payout, bankroll, kelly_frac):
    mu = pra_per_min * minutes
    sd = sd_per_min * np.sqrt(minutes)
    if sd <= 0:
        sd = max(1.0, 0.15 * max(mu, 1.0))

    # Probability OVER using Normal approximation
    p_hat = 1 - norm.cdf(line, mu, sd)

    # EV per $1  (for payout multiplier on win, 0 on loss)
    ev = p_hat * (payout - 1.0) - (1.0 - p_hat)

    # Full Kelly fraction
    b = payout - 1.0
    full_kelly = 0.0
    if b > 0:
        full_kelly = max(0.0, (b * p_hat - (1.0 - p_hat)) / b)

    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_hat, ev, stake, mu, sd, full_kelly


# ---------- UI: INPUTS ----------

st.sidebar.header("Global Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", value=30.0, step=1.0)
payout = st.sidebar.number_input("2-Pick Payout Multiplier", value=3.0, step=0.1)
kelly_frac = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
n_games = st.sidebar.slider("Games for API lookback", 5, 20, DEFAULT_N_GAMES)

st.markdown("### Player Inputs")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.subheader("Player 1")
    p1_name = st.text_input("Player 1 Name", "Donovan Mitchell")
    p1_line = st.number_input("P1 PRA Line", value=33.5, step=0.5)
    p1_min = st.number_input("P1 Projected Minutes", value=34.0, step=1.0)
    auto1 = st.button("Auto-fill P1 from last N games")

with col_p2:
    st.subheader("Player 2")
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    p2_line = st.number_input("P2 PRA Line", value=34.5, step=0.5)
    p2_min = st.number_input("P2 Projected Minutes", value=32.0, step=1.0)
    auto2 = st.button("Auto-fill P2 from last N games")

# We store auto-filled params in session_state to persist across reruns
if "p1_mu" not in st.session_state:
    st.session_state.p1_mu = 1.10
    st.session_state.p1_sd = 0.15
if "p2_mu" not in st.session_state:
    st.session_state.p2_mu = 1.15
    st.session_state.p2_sd = 0.15

# Handle auto-fill for Player 1
if auto1:
    mu, sd, msg = estimate_pra_params_from_api(p1_name, n_games)
    if mu is None:
        st.warning(f"P1 auto-fill failed: {msg}")
    else:
        st.session_state.p1_mu = round(mu, 3)
        st.session_state.p1_sd = round(sd, 3)
        st.success(f"P1 auto-fill: PRA/min={mu:.3f}, SD/min={sd:.3f} ({msg})")

# Handle auto-fill for Player 2
if auto2:
    mu, sd, msg = estimate_pra_params_from_api(p2_name, n_games)
    if mu is None:
        st.warning(f"P2 auto-fill failed: {msg}")
    else:
        st.session_state.p2_mu = round(mu, 3)
        st.session_state.p2_sd = round(sd, 3)
        st.success(f"P2 auto-fill: PRA/min={mu:.3f}, SD/min={sd:.3f} ({msg})")

st.markdown("### Current Model Inputs (editable)")
col_m1, col_m2 = st.columns(2)
with col_m1:
    p1_pra_per_min = st.number_input("P1 PRA per Min", value=st.session_state.p1_mu, step=0.01)
    p1_sd_per_min = st.number_input("P1 SD per Min", value=st.session_state.p1_sd, step=0.01)
with col_m2:
    p2_pra_per_min = st.number_input("P2 PRA per Min", value=st.session_state.p2_mu, step=0.01)
    p2_sd_per_min = st.number_input("P2 SD per Min", value=st.session_state.p2_sd, step=0.01)

run_calc = st.button("Calculate Edge for Both & Combo")

# ---------- CALCULATIONS & OUTPUT ----------

if run_calc:
    if payout <= 1:
        st.error("Payout multiplier must be > 1.")
    else:
        # Player 1
        p1, ev1, stake1, mu1, sd1, fk1 = compute_ev(
            p1_line, p1_pra_per_min, p1_sd_per_min, p1_min, payout, bankroll, kelly_frac
        )

        # Player 2
        p2, ev2, stake2, mu2, sd2, fk2 = compute_ev(
            p2_line, p2_pra_per_min, p2_sd_per_min, p2_min, payout, bankroll, kelly_frac
        )

        # Combo: both have to hit (assume independent)
        p_joint = p1 * p2
        ev_joint = payout * p_joint - 1.0
        b_joint = payout - 1.0
        full_k_joint = 0.0
        if b_joint > 0:
            full_k_joint = max(0.0, (b_joint * p_joint - (1.0 - p_joint)) / b_joint)
        stake_joint = bankroll * kelly_frac * full_k_joint
        stake_joint = min(stake_joint, bankroll * MAX_BANKROLL_PCT)
        stake_joint = max(0.0, round(stake_joint, 2))

        st.markdown("---")
        st.header("📊 Results")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader(p1_name)
            st.write(f"Model μ ≈ {mu1:.2f}, σ ≈ {sd1:.2f}")
            st.metric("Prob OVER", f"{p1:.1%}")
            st.metric("EV per $", f"{ev1*100:.1f}%")
            st.metric("Suggested Stake", f"${stake1:.2f}")
            if ev1 > 0:
                st.success("✅ +EV leg")
            else:
                st.error("❌ -EV leg")

        with c2:
            st.subheader(p2_name)
            st.write(f"Model μ ≈ {mu2:.2f}, σ ≈ {sd2:.2f}")
            st.metric("Prob OVER", f"{p2:.1%}")
            st.metric("EV per $", f"{ev2*100:.1f}%")
            st.metric("Suggested Stake", f"${stake2:.2f}")
            if ev2 > 0:
                st.success("✅ +EV leg")
            else:
                st.error("❌ -EV leg")

        st.markdown("---")
        st.subheader("🎯 2-Pick Combo (Both Must Hit)")
        st.metric("Joint Prob (both over)", f"{p_joint:.1%}")
        st.metric("EV per $", f"{ev_joint*100:.1f}%")
        st.metric("Recommended Combo Stake", f"${stake_joint:.2f}")

        if ev_joint > 0:
            st.success("✅ YES: Combo is +EV under current model.")
        else:
            st.error("❌ NO: Combo is -EV. Skip or resize.")

st.caption(
    "Auto-fill uses last N games PRA/min via balldontlie API. "
    "Always adjust projected minutes & sanity-check roles, injuries, and matchup."
)
