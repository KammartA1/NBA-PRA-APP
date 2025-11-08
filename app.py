import difflib
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests  # only used by nba_api internally
import streamlit as st
from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog
from scipy.stats import norm

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NBA 2-Pick Prop Edge Model", page_icon="🏀", layout="wide")
st.title("🏀 NBA 2-Pick Prop Edge & Risk Model")

st.markdown(
    """
This app:

- Uses **nba_api** to pull last N games and estimate efficiency & minutes.
- Lets you manually enter prop lines from **PrizePicks or any book/app**.
- Supports **PRA, Points, Rebounds, Assists**.
- Computes **win probability**, **expected value (EV)**, and **Kelly-based stake**.
- Evaluates both single legs and the **2-pick combo**.
"""
)

# =========================
# SIDEBAR: GLOBAL SETTINGS
# =========================
st.sidebar.header("Global Settings")

bankroll = st.sidebar.number_input("Bankroll ($)", min_value=10.0, value=1000.0, step=10.0)
payout_mult = st.sidebar.number_input("2-Pick Payout Multiplier", min_value=1.01, value=3.0, step=0.1)
fractional_kelly = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
games_lookback = st.sidebar.slider("Games for lookback", 5, 20, 10, 1)

st.sidebar.caption(
    "For PrizePicks 2-pick Power Play use 3.0x. "
    "Use smaller Kelly (0.1–0.3) to reduce variance."
)

MAX_BANKROLL_PCT = 0.03  # hard cap per position (3% of bankroll)

# =========================
# MARKET SELECTOR
# =========================
market_label = st.selectbox(
    "Select Prop Market",
    ["PRA (Points + Rebounds + Assists)", "Points", "Rebounds", "Assists"],
    index=0,
)

market_key_map = {
    "PRA (Points + Rebounds + Assists)": "pra",
    "Points": "pts",
    "Rebounds": "reb",
    "Assists": "ast",
}
selected_market = market_key_map[market_label]

metric_map = {
    "pra": ["PTS", "REB", "AST"],
    "pts": ["PTS"],
    "reb": ["REB"],
    "ast": ["AST"],
}

# =========================
# PLAYER INPUTS
# =========================
st.subheader("Player Inputs")

col1, col2 = st.columns(2)

with col1:
    p1_name = st.text_input("Player 1 Name", "RJ Barrett")
    p1_line = st.number_input(
        "P1 Line (enter from PrizePicks / book)",
        min_value=1.0,
        max_value=100.0,
        value=33.5,
        step=0.5,
    )

with col2:
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    p2_line = st.number_input(
        "P2 Line (enter from PrizePicks / book)",
        min_value=1.0,
        max_value=100.0,
        value=34.5,
        step=0.5,
    )

run = st.button("Run Live Model")

# =========================
# HELPER FUNCS
# =========================
def _norm_name(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )


def current_season() -> str:
    """Return NBA season string like '2024-25' based on today's date."""
    today = datetime.now()
    year = today.year
    if today.month >= 10:
        start = year
    else:
        start = year - 1
    end = start + 1
    return f"{start}-{str(end)[-2:]}"


@st.cache_data(show_spinner=False)
def nba_lookup_player(name: str):
    players = nba_players.get_players()
    target = _norm_name(name)

    # Exact match
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]

    # Fuzzy match
    norm_names = [_norm_name(p["full_name"]) for p in players]
    best = difflib.get_close_matches(target, norm_names, n=1, cutoff=0.6)
    if best:
        chosen = best[0]
        for p in players:
            if _norm_name(p["full_name"]) == chosen:
                return p["id"], p["full_name"]

    return None, f"No NBA player match for '{name}'."


@st.cache_data(show_spinner=False)
def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    """
    Return (per_min_mean, per_min_sd, avg_minutes, message)
    for the chosen market based on last N games.
    """
    cols = metric_map[market]

    pid, label = nba_lookup_player(name)
    if pid is None:
        return None, None, None, f"Could not find player '{name}'."

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
    except Exception as e:
        return None, None, None, f"Error fetching logs for {label}: {e}"

    if df.empty:
        return None, None, None, f"No logs found for {label}."

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).head(n_games)

    per_min_vals = []
    minutes_list = []

    for _, row in df.iterrows():
        total_val = sum(float(row.get(c, 0)) for c in cols)
        mins_raw = row.get("MIN", 0)

        # Handle "MM:SS" or numeric
        minutes = 0.0
        try:
            if isinstance(mins_raw, str) and ":" in mins_raw:
                mm, ss = mins_raw.split(":")
                minutes = float(mm) + float(ss) / 60.0
            else:
                minutes = float(mins_raw)
        except Exception:
            minutes = 0.0

        if minutes > 0:
            minutes_list.append(minutes)
            per_min_vals.append(total_val / minutes)

    if len(per_min_vals) < 3 or len(minutes_list) == 0:
        return None, None, None, f"Not enough valid games for {label}."

    per_min_arr = np.array(per_min_vals)
    mu_per_min = float(per_min_arr.mean())
    sd_per_min = float(per_min_arr.std(ddof=1))
    if sd_per_min <= 0:
        sd_per_min = max(0.05, 0.1 * mu_per_min)

    avg_min = float(np.mean(minutes_list))

    msg = (
        f"{label}: using last {len(per_min_vals)} games "
        f"({current_season()}), avg minutes {avg_min:.1f}"
    )

    return mu_per_min, sd_per_min, avg_min, msg


def compute_leg(line, mu_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    """
    Compute leg probability, EV per $ (using 2-pick style payout edge),
    Kelly fraction, and recommended stake.
    """
    # Projected distribution
    mu = mu_per_min * minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    if sd <= 0:
        sd = max(1.0, 0.15 * max(mu, 1.0))

    # Probability this leg goes OVER its line
    p_over = 1.0 - norm.cdf(line, mu, sd)

    # EV per $ if priced like a 2-pick leg (for intuition)
    b = payout_mult - 1.0
    ev_per_dollar = p_over * b - (1.0 - p_over)

    # Kelly fraction for this leg under that structure
    full_kelly = max(0.0, (b * p_over - (1.0 - p_over)) / b) if b > 0 else 0.0

    # Recommended stake
    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_over, ev_per_dollar, full_kelly, stake, mu, sd


# =========================
# MAIN LOGIC
# =========================
if run:
    # Basic sanity
    if payout_mult <= 1.0:
        st.error("2-pick payout multiplier must be > 1.0")
        st.stop()

    # --- Player 1 model ---
    p1_mu_min, p1_sd_min, p1_avg_min, p1_msg = get_player_rate_and_minutes(
        p1_name, games_lookback, selected_market
    )
    if p1_mu_min is None:
        st.error(f"P1 stats error: {p1_msg}")
        st.stop()

    # --- Player 2 model ---
    p2_mu_min, p2_sd_min, p2_avg_min, p2_msg = get_player_rate_and_minutes(
        p2_name, games_lookback, selected_market
    )
    if p2_mu_min is None:
        st.error(f"P2 stats error: {p2_msg}")
        st.stop()

    # --- Single legs ---
    p1_prob, ev1, k1, stake1, p1_mu, p1_sd = compute_leg(
        p1_line, p1_mu_min, p1_sd_min, p1_avg_min, payout_mult, bankroll, fractional_kelly
    )
    p2_prob, ev2, k2, stake2, p2_mu, p2_sd = compute_leg(
        p2_line, p2_mu_min, p2_sd_min, p2_avg_min, payout_mult, bankroll, fractional_kelly
    )

    # --- 2-pick combo (independent legs assumption) ---
    joint_prob = p1_prob * p2_prob
    b_combo = payout_mult - 1.0
    combo_ev = payout_mult * joint_prob - 1.0
    combo_full_kelly = max(
        0.0, (b_combo * joint_prob - (1.0 - joint_prob)) / b_combo
    ) if b_combo > 0 else 0.0

    combo_stake = bankroll * fractional_kelly * combo_full_kelly
    combo_stake = min(combo_stake, bankroll * MAX_BANKROLL_PCT)
    combo_stake = max(0.0, round(combo_stake, 2))

    # =========================
    # DISPLAY: SINGLE LEGS
    # =========================
    st.markdown("## 📊 Single-Leg Results")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"### {p1_name}")
        st.caption(p1_msg)
        st.markdown(f"**Line:** {p1_line}")
        st.markdown(f"**Auto Projected Minutes:** {p1_avg_min:.1f}")
        st.markdown(f"**Model Mean ({market_label}):** {p1_mu:.1f}")
        st.markdown(f"**Prob OVER:** {p1_prob * 100:.1f}%")
        st.markdown(f"**EV per $ (2-pick style):** {ev1 * 100:.1f}%")
        st.markdown(f"**Suggested Stake:** ${stake1:.2f}")
        if ev1 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")

    with c2:
        st.markdown(f"### {p2_name}")
        st.caption(p2_msg)
        st.markdown(f"**Line:** {p2_line}")
        st.markdown(f"**Auto Projected Minutes:** {p2_avg_min:.1f}")
        st.markdown(f"**Model Mean ({market_label}):** {p2_mu:.1f}")
        st.markdown(f"**Prob OVER:** {p2_prob * 100:.1f}%")
        st.markdown(f"**EV per $ (2-pick style):** {ev2 * 100:.1f}%")
        st.markdown(f"**Suggested Stake:** ${stake2:.2f}")
        if ev2 > 0:
            st.success("✅ +EV leg")
        else:
            st.error("❌ -EV leg")

    # =========================
    # DISPLAY: COMBO
    # =========================
    st.markdown("---")
    st.markdown("## 🎯 2-Pick Combo (Both Must Hit)")

    st.markdown(f"**Joint Prob (both OVER):** {joint_prob * 100:.1f}%")
    st.markdown(f"**EV per $:** {combo_ev * 100:.1f}%")
    st.markdown(f"**Suggested Combo Stake:** ${combo_stake:.2f}")

    if combo_ev > 0:
        st.success("🔥 Combo is +EV under this model.")
    else:
        st.error("🚫 Combo is -EV. Consider passing.")

    # =========================
    # BEST BET SUMMARY
    # =========================
    st.markdown("---")
    st.markdown("## 💬 Best Bet Summary")

    # Choose best leg by EV
    if ev1 >= ev2:
        best_player, best_line, best_ev, best_prob, best_stake = (
            p1_name,
            p1_line,
            ev1,
            p1_prob,
            stake1,
        )
    else:
        best_player, best_line, best_ev, best_prob, best_stake = (
            p2_name,
            p2_line,
            ev2,
            p2_prob,
            stake2,
        )

    if best_ev > 0 and best_stake > 0:
        st.success(
            f"**Best Single-Leg Edge:** {best_player} OVER {best_line}  \n"
            f"Win Probability: **{best_prob * 100:.1f}%**  \n"
            f"EV per $: **{best_ev * 100:.1f}%**  \n"
            f"Suggested Stake: **${best_stake:.2f}**"
        )
    else:
        st.warning(
            "No clear +EV single-leg edge detected. "
            "Use discipline and wait for better numbers."
        )

# Footer
st.caption(
    "Lines are entered manually from PrizePicks or any book. "
    "Projections are based on recent-game rates and a normal approximation. "
    "Always layer in injury/news/role context before betting."
)
