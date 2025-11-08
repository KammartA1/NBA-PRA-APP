import os
import difflib
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import streamlit as st
from scipy.stats import norm

# =========================
# CONFIG & SECRETS
# =========================

st.set_page_config(page_title="NBA PRA 2-Pick Live Edge", page_icon="🏀", layout="centered")
st.title("🏀 NBA PRA 2-Pick Live Edge")

st.markdown(
    "This tool:\n"
    "- Pulls recent stats from **balldontlie**\n"
    "- Pulls PrizePicks PRA lines via **The Odds API**\n"
    "- Computes probability, EV, and recommended stakes for each leg and a 2-pick combo."
)

BALLDONTLIE_API_KEY = st.secrets.get("BALLDONTLIE_API_KEY", os.getenv("BALLDONTLIE_API_KEY", ""))
THE_ODDS_API_KEY = st.secrets.get("THE_ODDS_API_KEY", os.getenv("THE_ODDS_API_KEY", ""))

BDL_HEADERS = {"Authorization": BALLDONTLIE_API_KEY} if BALLDONTLIE_API_KEY else {}
TOA_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

MAX_BANKROLL_PCT = 0.03  # 3% max position
DEFAULT_LOOKBACK = 10    # games for PRA/min

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

# ---------- balldontlie: player lookup ----------

@st.cache_data(show_spinner=False)
def bdl_get_player_id(player_name: str):
    """Fuzzy player search via balldontlie."""
    if not BALLDONTLIE_API_KEY:
        return None, "BALLDONTLIE_API_KEY missing. Add it in Streamlit secrets."

    base_url = "https://api.balldontlie.io/v1/players"

    def _query(q):
        try:
            r = requests.get(base_url, headers=BDL_HEADERS,
                             params={"search": q, "per_page": 100}, timeout=8)
            if r.status_code != 200:
                return None, f"balldontlie error {r.status_code}: {r.text}"
            return r.json().get("data", []), None
        except Exception as e:
            return None, f"Network error calling balldontlie: {e}"

    # first try full input
    data, err = _query(player_name)
    if err:
        return None, err

    # if no results, try last name only
    if not data:
        parts = player_name.split()
        if len(parts) > 1:
            data, err = _query(parts[-1])
            if err:
                return None, err

    if not data:
        return None, f"No player found for '{player_name}' via balldontlie."

    # fuzzy match
    target = _norm_name(player_name)
    candidates = []
    for p in data:
        full = f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        candidates.append((p["id"], full, _norm_name(full)))

    names_norm = [c[2] for c in candidates]
    best = difflib.get_close_matches(target, names_norm, n=1, cutoff=0.4)
    if best:
        best_norm = best[0]
        for pid, full, normed in candidates:
            if normed == best_norm:
                return pid, full

    # fallback: first result
    pid, full, _ = candidates[0]
    return pid, full

# ---------- balldontlie: PRA/min from last N games ----------

@st.cache_data(show_spinner=False)
def bdl_get_pra_params(player_name: str, n_games: int):
    """
    Get PRA_per_min mean & sd from last N games via balldontlie stats endpoint.
    """
    pid, msg = bdl_get_player_id(player_name)
    if pid is None:
        return None, None, msg

    if not BALLDONTLIE_API_KEY:
        return None, None, "BALLDONTLIE_API_KEY missing."

    url = "https://api.balldontlie.io/v1/stats"
    params = {
        "player_ids[]": pid,
        "per_page": n_games,
        "sort": "game.date",  # newest first if supported
    }

    try:
        r = requests.get(url, headers=BDL_HEADERS, params=params, timeout=8)
    except Exception as e:
        return None, None, f"Error fetching stats: {e}"

    if r.status_code != 200:
        return None, None, f"Stats error {r.status_code}: {r.text}"

    data = r.json().get("data", [])
    if not data:
        return None, None, f"No stats returned for {player_name}."

    pra_per_min = []
    for g in data:
        pts = g.get("pts", 0)
        reb = g.get("reb", 0)
        ast = g.get("ast", 0)
        mins_raw = g.get("min") or g.get("minutes") or "0"

        # handle "MM:SS" or numeric
        if isinstance(mins_raw, str) and ":" in mins_raw:
            mm, ss = mins_raw.split(":")
            try:
                mins = float(mm) + float(ss) / 60.0
            except:
                mins = 0.0
        else:
            try:
                mins = float(mins_raw)
            except:
                mins = 0.0

        if mins > 0:
            pra_per_min.append((pts + reb + ast) / mins)

    if len(pra_per_min) < 3:
        return None, None, f"Not enough valid games for {player_name}."

    arr = np.array(pra_per_min)
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd <= 0:
        sd = max(0.05, 0.1 * mu)

    return mu, sd, f"OK — using last {len(arr)} games for {player_name}"

# ---------- The Odds API: pull all PrizePicks PRA lines ----------

@st.cache_data(show_spinner=False)
def load_prizepicks_pra_lines():
    """
    Load PrizePicks PRA lines for today's & near-future NBA games using The Odds API.
    Returns (dict: norm_name -> line, message).
    """
    if not THE_ODDS_API_KEY:
        return {}, "THE_ODDS_API_KEY missing. Add it in Streamlit secrets."

    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=2)

    events_url = f"{TOA_BASE}/sports/{SPORT_KEY}/events"
    events_params = {
        "apiKey": THE_ODDS_API_KEY,
        "commenceTimeFrom": start.isoformat(),
        "commenceTimeTo": end.isoformat(),
    }

    try:
        ev_resp = requests.get(events_url, params=events_params, timeout=8)
        if ev_resp.status_code != 200:
            return {}, f"Events error {ev_resp.status_code}: {ev_resp.text}"
        events = ev_resp.json()
    except Exception as e:
        return {}, f"Events request failed: {e}"

    lines = {}

    for ev in events:
        event_id = ev.get("id")
        if not event_id:
            continue

        odds_url = f"{TOA_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
        odds_params = {
            "apiKey": THE_ODDS_API_KEY,
            "regions": "us_dfs",  # DFS books region
            "bookmakers": "prizepicks",
            "markets": "player_points_rebounds_assists",
            "oddsFormat": "decimal",
        }

        try:
            od_resp = requests.get(odds_url, params=odds_params, timeout=8)
            if od_resp.status_code != 200:
                # silently skip if this event doesn't have props
                continue
            od = od_resp.json()
        except Exception:
            continue

        for bm in od.get("bookmakers", []):
            if bm.get("key") != "prizepicks":
                continue
            for m in bm.get("markets", []):
                if m.get("key") != "player_points_rebounds_assists":
                    continue
                for o in m.get("outcomes", []):
                    raw_name = o.get("description") or o.get("name") or ""
                    if not raw_name:
                        continue
                    point = o.get("point")
                    if point is None:
                        continue
                    norm = _norm_name(raw_name)
                    # last one wins if duplicates; usually fine
                    lines[norm] = float(point)

    if not lines:
        return {}, "No PrizePicks PRA lines found via The Odds API (check key/plan/markets)."

    return lines, f"Loaded {len(lines)} PrizePicks PRA lines."

def get_prizepicks_pra_line(player_name: str):
    lines, msg = load_prizepicks_pra_lines()
    if not lines:
        return None, msg
    target = _norm_name(player_name)
    # exact or substring match
    if target in lines:
        return lines[target], "OK"
    # fuzzy search
    keys = list(lines.keys())
    best = difflib.get_close_matches(target, keys, n=1, cutoff=0.5)
    if best:
        return lines[best[0]], f"Matched as '{best[0]}'"
    return None, f"No PRA line match for '{player_name}'. ({msg})"

# ---------- EV / Kelly model ----------

def compute_ev(line, pra_per_min, sd_per_min, minutes, payout_mult, bankroll, kelly_frac):
    mu = pra_per_min * minutes
    sd = sd_per_min * np.sqrt(max(minutes, 1.0))
    if sd <= 0:
        sd = max(1.0, 0.15 * max(mu, 1.0))

    # Probability OVER (Normal)
    p_hat = 1.0 - norm.cdf(line, mu, sd)

    # EV per $1 for fixed-multiplier payout
    ev = p_hat * (payout_mult - 1.0) - (1.0 - p_hat)

    # Full Kelly fraction (for multiplier odds)
    b = payout_mult - 1.0
    full_kelly = 0.0
    if b > 0:
        full_kelly = max(0.0, (b * p_hat - (1.0 - p_hat)) / b)

    stake = bankroll * kelly_frac * full_kelly
    stake = min(stake, bankroll * MAX_BANKROLL_PCT)
    stake = max(0.0, round(stake, 2))

    return p_hat, ev, stake, mu, sd, full_kelly

# =========================
# UI: GLOBAL SETTINGS
# =========================

st.sidebar.header("Global Settings")
bankroll = st.sidebar.number_input("Bankroll ($)", value=30.0, step=1.0)
payout_mult = st.sidebar.number_input("2-Pick Payout Multiplier", value=3.0, step=0.1)
kelly_frac = st.sidebar.slider("Fractional Kelly", 0.0, 1.0, 0.25, 0.05)
lookback = st.sidebar.slider("Games for PRA/min (lookback)", 5, 20, DEFAULT_LOOKBACK)

st.sidebar.caption(
    "For PrizePicks 2-pick Power Play use 3.0x.\n"
    "Kelly < 0.25 recommended with small bankroll."
)

# =========================
# UI: PLAYER INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Player 1")
    p1_name = st.text_input("Player 1 Name", "RJ Barrett")
    p1_proj_min = st.number_input("P1 Projected Minutes", value=34.0, step=1.0)
    auto_p1_line = st.checkbox("Use PrizePicks PRA line (P1)", value=True)
    p1_manual_line = st.number_input("P1 Manual PRA Line", value=33.5, step=0.5)

with col2:
    st.subheader("Player 2")
    p2_name = st.text_input("Player 2 Name", "Jaylen Brown")
    p2_proj_min = st.number_input("P2 Projected Minutes", value=32.0, step=1.0)
    auto_p2_line = st.checkbox("Use PrizePicks PRA line (P2)", value=True)
    p2_manual_line = st.number_input("P2 Manual PRA Line", value=34.5, step=0.5)

run = st.button("Run Live Model")

# =========================
# MAIN LOGIC
# =========================

if run:
    errors = []

    if payout_mult <= 1:
        errors.append("Payout multiplier must be > 1.")

    # P1 stats
    p1_mu_min, p1_sd_min, p1_stats_msg = bdl_get_pra_params(p1_name, lookback)
    if p1_mu_min is None:
        errors.append(f"P1 stats: {p1_stats_msg}")

    # P2 stats
    p2_mu_min, p2_sd_min, p2_stats_msg = bdl_get_pra_params(p2_name, lookback)
    if p2_mu_min is None:
        errors.append(f"P2 stats: {p2_stats_msg}")

    # show any blocking errors
    if errors:
        for e in errors:
            st.error(e)
    else:
        # Lines from The Odds API (with manual fallback)
        p1_line = p1_manual_line
        p2_line = p2_manual_line

        if auto_p1_line:
            api_line, msg = get_prizepicks_pra_line(p1_name)
            if api_line is not None:
                p1_line = api_line
                st.info(f"P1 line from PrizePicks/The Odds API: {api_line} ({msg})")
            else:
                st.warning(f"P1 line lookup: {msg} — using manual {p1_manual_line}")

        if auto_p2_line:
            api_line, msg = get_prizepicks_pra_line(p2_name)
            if api_line is not None:
                p2_line = api_line
                st.info(f"P2 line from PrizePicks/The Odds API: {api_line} ({msg})")
            else:
                st.warning(f"P2 line lookup: {msg} — using manual {p2_manual_line}")

        # Compute model outputs
        p1, ev1, stake1, mu1, sd1, fk1 = compute_ev(
            p1_line, p1_mu_min, p1_sd_min, p1_proj_min, payout_mult, bankroll, kelly_frac
        )
        p2, ev2, stake2, mu2, sd2, fk2 = compute_ev(
            p2_line, p2_mu_min, p2_sd_min, p2_proj_min, payout_mult, bankroll, kelly_frac
        )

        # Combo (independence assumed)
        p_joint = p1 * p2
        ev_joint = payout_mult * p_joint - 1.0
        b_joint = payout_mult - 1.0
        fk_joint = max(0.0, (b_joint * p_joint - (1.0 - p_joint)) / b_joint) if b_joint > 0 else 0.0
        stake_joint = bankroll * kelly_frac * fk_joint
        stake_joint = min(stake_joint, bankroll * MAX_BANKROLL_PCT)
        stake_joint = max(0.0, round(stake_joint, 2))

        # ----- DISPLAY RESULTS -----

        st.markdown("---")
        st.header("📊 Single-Leg Results")

        c1_out, c2_out = st.columns(2)

        with c1_out:
            st.subheader(p1_name)
            st.write(f"Model from stats: PRA/min ≈ {p1_mu_min:.3f}, SD/min ≈ {p1_sd_min:.3f}")
            st.write(f"Line: **{p1_line}**, Projected Minutes: **{p1_proj_min}**")
            st.metric("Prob OVER", f"{p1:.1%}")
            st.metric("EV per $", f"{ev1*100:.1f}%")
            st.metric("Suggested Stake", f"${stake1:.2f}")
            if ev1 > 0:
                st.success("✅ +EV leg under current model.")
            else:
                st.error("❌ -EV leg (skip or adjust).")

        with c2_out:
            st.subheader(p2_name)
            st.write(f"Model from stats: PRA/min ≈ {p2_mu_min:.3f}, SD/min ≈ {p2_sd_min:.3f}")
            st.write(f"Line: **{p2_line}**, Projected Minutes: **{p2_proj_min}**")
            st.metric("Prob OVER", f"{p2:.1%}")
            st.metric("EV per $", f"{ev2*100:.1f}%")
            st.metric("Suggested Stake", f"${stake2:.2f}")
            if ev2 > 0:
                st.success("✅ +EV leg under current model.")
            else:
                st.error("❌ -EV leg (skip or adjust).")

        st.markdown("---")
        st.header("🎯 2-Pick Combo (Both Must Hit)")

        st.metric("Joint Prob (both OVER)", f"{p_joint:.1%}")
        st.metric("EV per $", f"{ev_joint*100:.1f}%")
        st.metric("Recommended Combo Stake", f"${stake_joint:.2f}")

        if ev_joint > 0:
            st.success("✅ YES: Combo is +EV under your assumptions.")
        else:
            st.error("❌ NO: Combo is -EV. Do not force it.")

st.caption(
    "If a player or line doesn't load, the message will tell you exactly why "
    "(missing key, no props returned, etc). Always sanity-check minutes/role."
)
