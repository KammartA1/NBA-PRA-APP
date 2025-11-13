# =============================================================
#  SECTION 1 — Imports, Config, Styling, Sidebar, Core Helpers
# =============================================================

import os, time, random, difflib
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import PlayerGameLog, LeagueDashTeamStats

# -------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="NBA Prop Model",
    page_icon="🏀",
    layout="wide",
)

# -------------------------------------------------------------
# TEMP STORAGE DIRECTORY
# -------------------------------------------------------------
TEMP_DIR = os.path.join("/tmp", "nba_prop_temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------------------------------------
# COLORS / THEME
# -------------------------------------------------------------
PRIMARY_MAROON = "#7A0019"
GOLD = "#FFCC33"
CARD_BG = "#17131C"
BG = "#0D0A12"

# -------------------------------------------------------------
# APP STYLE
# -------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .main-header {{
        text-align:center;
        font-size:40px;
        color:{GOLD};
        font-weight:700;
        margin-bottom:0px;
    }}
    .card {{
        background-color:{CARD_BG};
        border-radius:18px;
        padding:14px;
        margin-bottom:18px;
        border:1px solid {GOLD}33;
        box-shadow:0 10px 24px rgba(0,0,0,0.75);
        transition:all 0.16s ease-in-out;
    }}
    .card:hover {{
        transform:translateY(-3px) scale(1.015);
        box-shadow:0 18px 40px rgba(0,0,0,0.9);
    }}
    .rec-play {{color:#4CAF50;font-weight:700;}}
    .rec-thin {{color:#FFC107;font-weight:700;}}
    .rec-pass {{color:#F44336;font-weight:700;}}
    .stApp {{
        background-color:{BG};
        color:white;
        font-family:system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: radial-gradient(circle at top, {PRIMARY_MAROON} 0%, #2b0b14 55%, #12060a 100%);
        border-right:1px solid {GOLD}33;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🏀 NBA Prop Model</p>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR — BANKROLL, SETTINGS
# -------------------------------------------------------------
st.sidebar.header("User & Bankroll")

user_id = st.sidebar.text_input(
    "Your ID (for personal history)",
    value="Me"
).strip() or "Me"

LOG_FILE = os.path.join(TEMP_DIR, f"bet_history_{user_id}.csv")

bankroll = st.sidebar.number_input(
    "Bankroll ($)",
    min_value=10.0,
    value=100.0
)

payout_mult = st.sidebar.number_input(
    "2-Pick Payout (e.g. 3.0x)",
    min_value=1.5,
    value=3.0
)

fractional_kelly = st.sidebar.slider(
    "Fractional Kelly",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

games_lookback = st.sidebar.slider(
    "Recent Games Sample (N)",
    min_value=5,
    max_value=20,
    value=10
)

compact_mode = st.sidebar.checkbox(
    "Compact Mode (mobile)",
    value=False
)

# -------------------------------------------------------------
# GLOBAL CONSTANTS & HELPERS
# -------------------------------------------------------------
MARKET_OPTIONS = ["PRA", "Points", "Rebounds", "Assists"]

MARKET_METRICS = {
    "PRA": ["PTS", "REB", "AST"],
    "Points": ["PTS"],
    "Rebounds": ["REB"],
    "Assists": ["AST"],
}

HEAVY_TAIL = {
    "PRA": 1.35,
    "Points": 1.25,
    "Rebounds": 1.25,
    "Assists": 1.25,
}

MAX_KELLY_PCT = 0.03  # 3% cap

def current_season():
    """Return the current NBA season (YYYY-YY)."""
    today = datetime.now()
    yr = today.year if today.month >= 10 else today.year - 1
    return f"{yr}-{str(yr+1)[-2:]}"

@st.cache_data(show_spinner=False)
def get_players_index():
    return nba_players.get_players()

def _norm_name(s: str) -> str:
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("-", " ")
        .strip()
    )

@st.cache_data(show_spinner=False)
def resolve_player(name: str):
    """Resolve fuzzy name → player ID."""
    if not name:
        return None, None

    players = get_players_index()
    target = _norm_name(name)

    # direct match
    for p in players:
        if _norm_name(p["full_name"]) == target:
            return p["id"], p["full_name"]

    # fuzzy match
    names = [_norm_name(p["full_name"]) for p in players]
    best = difflib.get_close_matches(target, names, n=1, cutoff=0.7)
    if best:
        match = best[0]
        for p in players:
            if _norm_name(p["full_name"]) == match:
                return p["id"], p["full_name"]

    return None, None

def get_headshot_url(name: str):
    pid, _ = resolve_player(name)
    if not pid:
        return None
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png"

    comp["Edge_vs_Market"] = comp["EV_float"] * 100

    fig2 = px.histogram(
        comp,
        x="Edge_vs_Market",
        nbins=20,
        title="Distribution of Model Edge vs Market (EV%)",
        color_discrete_sequence=["#FFCC33"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # Calibration summary
    # -------------------------------
    st.markdown(
        f"**Predicted Avg Win Prob (approx):** {pred_win_prob*100:.1f}%"
    )
    st.markdown(
        f"**Actual Hit Rate:** {actual_win_prob*100:.1f}%"
    )
    st.markdown(
        f"**Calibration Gap:** {gap:+.1f}% | **ROI:** {roi:+.1f}%"
    )

    if gap > 5:
        st.warning(
            "Model appears overconfident → consider requiring higher EV before firing."
        )
    elif gap < -5:
        st.info(
            "Model appears conservative → thin edges may be slightly under-trusted."
        )
    else:
        st.success("Model and results are reasonably aligned ✅")
# =============================================================
#  SECTION 2 — Team Context + Player Projection Engine
# =============================================================

# -------------------------------------------------------------
# TEAM CONTEXT: Pace, Defense, Advanced Rebound % & Assist %
# -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_team_context():
    """
    Pulls league-wide team context metrics:
    - PACE
    - DEF_RATING
    - REB_PCT, OREB_PCT, DREB_PCT
    - AST_PCT

    Returns:
        TEAM_CTX (dict): per-team metrics
        LEAGUE_CTX (dict): league-average benchmarks
    """
    try:
        # Base per-game stats
        base = LeagueDashTeamStats(
            season=current_season(),
            per_mode_detailed="PerGame"
        ).get_data_frames()[0]

        # Advanced rebound & assist metrics
        adv = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed="Advanced",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "REB_PCT",
            "OREB_PCT",
            "DREB_PCT",
            "AST_PCT",
            "PACE"
        ]]

        # Defensive rating
        defense = LeagueDashTeamStats(
            season=current_season(),
            measure_type_detailed_defense="Defense",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0][[
            "TEAM_ID",
            "TEAM_ABBREVIATION",
            "DEF_RATING"
        ]]

        # Merge
        df = base.merge(adv, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="left")
        df = df.merge(defense, on=["TEAM_ID", "TEAM_ABBREVIATION"], how="left")

        # League averages
        league_ctx = {
            "PACE": df["PACE"].mean(),
            "DEF_RATING": df["DEF_RATING"].mean(),
            "REB_PCT": df["REB_PCT"].mean(),
            "AST_PCT": df["AST_PCT"].mean(),
        }

        # Per-team dictionary
        team_ctx = {}
        for _, r in df.iterrows():
            team_ctx[r["TEAM_ABBREVIATION"]] = {
                "PACE": float(r["PACE"]),
                "DEF_RATING": float(r["DEF_RATING"]),
                "REB_PCT": float(r["REB_PCT"]),
                "DREB_PCT": float(r["DREB_PCT"]),
                "AST_PCT": float(r["AST_PCT"]),
            }

        return team_ctx, league_ctx

    except Exception:
        return {}, {}


TEAM_CTX, LEAGUE_CTX = get_team_context()

# -------------------------------------------------------------
# CONTEXT MULTIPLIER
# -------------------------------------------------------------
def get_context_multiplier(opp_abbrev: str | None, market: str = "PRA"):
    """
    Adjust projection based on opponent defense, pace, rebound %, assist %
    The weighting is proportional:
      - Pace impact
      - Defensive rating impact
      - Rebound-adjustment for REB markets
      - Assist-adjustment for AST markets
    """
    if not opp_abbrev:
        return 1.0

    opp_abbrev = opp_abbrev.upper().strip()

    if opp_abbrev not in TEAM_CTX or not LEAGUE_CTX:
        return 1.0

    opp = TEAM_CTX[opp_abbrev]

    pace_factor = opp["PACE"] / LEAGUE_CTX["PACE"]
    defense_factor = LEAGUE_CTX["DEF_RATING"] / opp["DEF_RATING"]

    # Advanced adjustments
    reb_adj = (
        LEAGUE_CTX["REB_PCT"] / opp["DREB_PCT"]
        if market == "Rebounds"
        else 1.0
    )

    ast_adj = (
        LEAGUE_CTX["AST_PCT"] / opp["AST_PCT"]
        if market == "Assists"
        else 1.0
    )

    # Blended multiplier
    mult = (
        0.4 * pace_factor +
        0.3 * defense_factor +
        0.3 * (reb_adj if market == "Rebounds" else ast_adj)
    )

    return float(np.clip(mult, 0.80, 1.20))  # prevent extremes


# -------------------------------------------------------------
# PLAYER RATE + MINUTES MODEL
# -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def get_player_rate_and_minutes(name: str, n_games: int, market: str):
    """
    Computes:
      - per-minute production rate
      - weighted average minutes
      - standard deviation of per-minute rate
    """
    pid, label = resolve_player(name)
    if not pid:
        return None, None, None, None, f"No match for '{name}'."

    try:
        gl = PlayerGameLog(
            player_id=pid,
            season=current_season(),
            season_type_all_star="Regular Season",
        ).get_data_frames()[0]
    except Exception as e:
        return None, None, None, None, f"Log error: {e}"

    if gl.empty:
        return None, None, None, None, "No recent games."

    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values("GAME_DATE", ascending=False).head(n_games)

    cols = MARKET_METRICS[market]
    per_min, mins = [], []

    for _, r in gl.iterrows():
        # Parse minutes
        m_val = r.get("MIN", "0")
        try:
            if isinstance(m_val, str) and ":" in m_val:
                mm, ss = m_val.split(":")
                mins_played = float(mm) + float(ss) / 60
            else:
                mins_played = float(m_val)
        except:
            mins_played = 0

        if mins_played <= 0:
            continue

        val = sum(float(r.get(c, 0)) for c in cols)
        per_min.append(val / mins_played)
        mins.append(mins_played)

    if not per_min:
        return None, None, None, None, "Insufficient data."

    per_min = np.array(per_min)
    mins = np.array(mins)

    # Weighted mean (more recent games slightly heavier)
    weights = np.linspace(0.6, 1.4, len(per_min))
    weights /= weights.sum()

    mu_per_min = float(np.average(per_min, weights=weights))
    avg_min = float(np.average(mins, weights=weights))

    sd_per_min = float(np.std(per_min, ddof=1))
    sd_per_min = max(sd_per_min, 0.15 * max(mu_per_min, 0.5))

    # Player team
    try:
        team = gl["TEAM_ABBREVIATION"].mode().iloc[0]
    except:
        team = None

    msg = f"{label}: {len(per_min)} games • {avg_min:.1f} min"

    return mu_per_min, sd_per_min, avg_min, team, msg


# -------------------------------------------------------------
# LEG PROJECTION ENGINE
# -------------------------------------------------------------
def compute_leg_projection(player, market, line, opp, teammate_out, blowout, n_games):
    """
    Returns a full projection object ready for the UI card.
    """
    mu_min, sd_min, avg_min, team, msg = get_player_rate_and_minutes(
        player, n_games, market
    )

    if mu_min is None:
        return None, msg

    ctx_mult = get_context_multiplier(
        opp.strip().upper() if opp else None,
        market
    )

    # Heavy-tail sim effect
    ht = HEAVY_TAIL[market]

    # Adjust minutes
    minutes = avg_min
    if teammate_out:
        minutes *= 1.05
        mu_min *= 1.06
    if blowout:
        minutes *= 0.90

    # Final distribution parameters
    mu = mu_min * minutes * ctx_mult
    sd = max(
        1.0,
        sd_min * np.sqrt(max(minutes, 1.0)) * ht
    )

    # Modeled probability
    p_over = 1.0 - norm.cdf(line, mu, sd)
    p_over = float(np.clip(p_over, 0.05, 0.95))

    # EV vs even-money
    ev_leg_even = p_over - (1 - p_over)

    return {
        "player": player,
        "market": market,
        "line": float(line),
        "mu": mu,
        "sd": sd,
        "prob_over": p_over,
        "ev_leg_even": ev_leg_even,
        "team": team,
        "ctx_mult": ctx_mult,
        "msg": msg,
        "teammate_out": teammate_out,
        "blowout": blowout,
    }, None
# =============================================================
#  SECTION 3 — Market Implied Probability + Baseline Library
# =============================================================

# -------------------------------------------------------------
# IMPLIED PROBABILITY (PrizePicks-style payout)
# -------------------------------------------------------------
def implied_probability(payout_mult: float) -> float:
    """
    Converts a PrizePicks 2-pick multiplier (e.g., 3.0x) into
    an implied probability per leg.
    """
    try:
        if payout_mult <= 1:
            return None
        return 1.0 / payout_mult
    except Exception:
        return None


# -------------------------------------------------------------
# MARKET BASELINE LIBRARY (stored under /tmp)
# -------------------------------------------------------------
MARKET_LIB_FILE = os.path.join(TEMP_DIR, "market_baselines.json")

def load_market_library():
    """Load historical market lines from /tmp safely."""
    if not os.path.exists(MARKET_LIB_FILE):
        return {}
    try:
        with open(MARKET_LIB_FILE, "r") as f:
            return json.load(f)
    except:
        return {}  # reset silently on corruption

def save_market_library(lib: dict):
    try:
        with open(MARKET_LIB_FILE, "w") as f:
            json.dump(lib, f)
    except:
        pass


# Structure:
# {
#   "LeBron James": {
#         "PRA": [42.5, 43.0, 41.5, ...],
#         "Points": [...],
#   },
#   ...
# }

MARKET_LIB = load_market_library()


# -------------------------------------------------------------
# UPDATE BASELINE LIBRARY ON EACH MODEL RUN
# -------------------------------------------------------------
def update_market_library(player: str, market: str, line: float):
    """Records every entered prop line for long-term mean/median baselines."""
    if not player or not market:
        return

    player = player.strip()
    market = market.strip()

    if player not in MARKET_LIB:
        MARKET_LIB[player] = {}

    if market not in MARKET_LIB[player]:
        MARKET_LIB[player][market] = []

    MARKET_LIB[player][market].append(float(line))

    # Save immediately
    save_market_library(MARKET_LIB)


# -------------------------------------------------------------
# RETRIEVE BASELINE (mean/median)
# -------------------------------------------------------------
def get_market_baseline(player: str, market: str):
    """Returns (mean, median) of all recorded market lines."""
    try:
        hist = MARKET_LIB.get(player, {}).get(market, [])
        if not hist:
            return None, None
        return float(np.mean(hist)), float(np.median(hist))
    except:
        return None, None
# =============================================================
# SECTION 4 — Single Player Expanded Analytics Tab
# =============================================================

tab_single = st.tabs(["🔍 Single Player Analytics"])[0]

with tab_single:
    st.header("🔎 Single Player Advanced Breakdown")

    sp_name = st.text_input("Player Name", key="single_name")
    opp_team = st.text_input("Opponent (optional, e.g. BOS, DEN)", key="single_opp")

    n_games_single = st.slider(
        "Games to analyze (recent)",
        min_value=5, max_value=20, value=10,
        key="single_n_games"
    )

    run_single = st.button("Run Player Breakdown")

    if run_single and sp_name:
        pid, label = resolve_player(sp_name)
        if not pid:
            st.error("Player not found.")
        else:
            # -------------------------------------------------------------
            # Load Game Log
            # -------------------------------------------------------------
            try:
                gl = PlayerGameLog(
                    player_id=pid,
                    season=current_season(),
                    season_type_all_star="Regular Season",
                ).get_data_frames()[0]
            except Exception as e:
                st.error(f"Unable to load game log: {e}")
                gl = None

            if gl is not None and not gl.empty:
                gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
                gl = gl.sort_values("GAME_DATE", ascending=False)

                # Trim sample
                gl_n = gl.head(n_games_single)

                # Compute minutes
                def parse_minutes(m):
                    try:
                        if isinstance(m, str) and ":" in m:
                            mm, ss = m.split(":")
                            return float(mm) + float(ss)/60
                        return float(m)
                    except:
                        return 0

                gl_n["MIN_float"] = gl_n["MIN"].apply(parse_minutes)

                # -------------------------------------------------------------
                # BASIC PER-GAME STATS
                # -------------------------------------------------------------
                ppg = gl_n["PTS"].mean()
                rpg = gl_n["REB"].mean()
                apg = gl_n["AST"].mean()
                pra = gl_n["PTS"].add(gl_n["REB"]).add(gl_n["AST"]).mean()

                st.subheader(f"📊 Recent ({n_games_single} games) Averages")
                st.markdown(
                    f"- **Points:** {ppg:.1f}\n"
                    f"- **Rebounds:** {rpg:.1f}\n"
                    f"- **Assists:** {apg:.1f}\n"
                    f"- **PRA:** {pra:.1f}\n"
                    f"- **Minutes:** {gl_n['MIN_float'].mean():.1f}"
                )

                # -------------------------------------------------------------
                # ADVANCED — Usage Rate (approx)
                # -------------------------------------------------------------
                gl_n["USG_EST"] = (
                    gl_n["PTS"] +
                    gl_n["FGA"] * 1.3 +
                    gl_n["FTA"] * 0.8 +
                    gl_n["TOV"] * 1.5
                ) / (gl_n["MIN_float"] + 1e-6)

                usg_rate = gl_n["USG_EST"].mean()

                st.subheader("⚡ Usage Influence")
                st.markdown(
                    f"- **Estimated Usage Rate:** {usg_rate:.2f} (per minute estimate)\n"
                    f"- Higher usage generally correlates with strong prop overs."
                )

                # -------------------------------------------------------------
                # ADVANCED — Shot Attempts Model
                # -------------------------------------------------------------
                # FGA/min
                gl_n["FGA_min"] = gl_n["FGA"] / (gl_n["MIN_float"] + 1e-6)
                fga_rate = gl_n["FGA_min"].mean()
                avg_min_single = gl_n["MIN_float"].mean()

                # Opponent context
                opp_mult = get_context_multiplier(opp_team.strip().upper() if opp_team else None, "Points")

                expected_fga = fga_rate * avg_min_single * opp_mult

                st.subheader("🎯 Shot Attempt Projection")
                st.markdown(
                    f"- **FGA per min:** {fga_rate:.3f}\n"
                    f"- **Expected minutes:** {avg_min_single:.1f}\n"
                    f"- **Opponent pace/def adj:** {opp_mult:.3f}\n"
                    f"- **Projected FGA today:** **{expected_fga:.1f}**"
                )

                # -------------------------------------------------------------
                # ADVANCED — Rebound & Assist Context
                # -------------------------------------------------------------
                if opp_team and opp_team in TEAM_CTX:
                    opp = TEAM_CTX[opp_team]

                    reb_mult = TEAM_CTX[opp_team]["DREB_PCT"]
                    ast_mult = TEAM_CTX[opp_team]["AST_PCT"]

                    st.subheader("🧱 Rebound & Assist Matchup Context")
                    st.markdown(
                        f"- **Opponent Defensive Rebound %:** {reb_mult:.3f}\n"
                        f"- **Opponent Assist % Allowed:** {ast_mult:.3f}"
                    )

                # -------------------------------------------------------------
                # TREND CHARTS
                # -------------------------------------------------------------
                st.subheader("📈 Performance Trends")

                trend_df = gl_n[["GAME_DATE","PTS","REB","AST"]].copy()
                trend_df["PRA"] = trend_df["PTS"] + trend_df["REB"] + trend_df["AST"]

                fig = px.line(
                    trend_df,
                    x="GAME_DATE",
                    y=["PTS","REB","AST","PRA"],
                    title="Recent Trend — Points, Rebounds, Assists, PRA",
                    markers=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("No games available for this player.")
# ================================================================
# SECTION 5 — Advanced Probability Engine (Monte Carlo + Skew Modeling)
# ================================================================

import numpy as _np

def simulate_player_distribution(mu, sd, minutes, market, trials=12000):
    """
    Creates a heavy-tail aware Monte Carlo distribution for PRA/PTS/REB/AST.
    """

    # Heavier tails for PRA and Rebounds especially
    tail_factor = {
        "PRA": 1.25,
        "Points": 1.15,
        "Rebounds": 1.20,
        "Assists": 1.10,
    }.get(market, 1.15)

    # Create base distribution
    base = _np.random.normal(mu, sd, trials)

    # Inject tail risk (right skew)
    boost_mask = _np.random.rand(trials) < 0.18  # 18% chance of tail event
    tail_boost = _np.random.gamma(shape=2.2, scale=sd * 0.35, size=trials)
    base[boost_mask] += tail_boost[boost_mask] * tail_factor

    # Force no negative stats
    base = _np.clip(base, 0, None)

    return base


def monte_carlo_probability(line, dist):
    """Probability the simulation clears the line."""
    return float((dist > line).mean())


def enriched_projection(mu, sd, minutes, line, market):
    """
    Combines:
    - Normal model
    - Monte Carlo simulation
    - Heavy tail distribution
    """

    # Run simulation
    dist = simulate_player_distribution(mu, sd, minutes, market)

    # Simulated probability
    mc_prob = monte_carlo_probability(line, dist)

    # Normal backup probability
    normal_prob = 1.0 - norm.cdf(line, mu, sd)
    normal_prob = float(np.clip(normal_prob, 0.01, 0.99))

    # Final weighted probability (simulation is primary)
    final_prob = 0.72 * mc_prob + 0.28 * normal_prob

    # Simulated EV
    ev_sim = final_prob - (1 - final_prob)

    # Export summary stats
    return {
        "dist": dist,
        "prob_sim": mc_prob,
        "prob_final": final_prob,
        "ev_sim": ev_sim,
        "p50": float(np.percentile(dist, 50)),
        "p95": float(np.percentile(dist, 95)),
        "p75": float(np.percentile(dist, 75)),
    }
# ==========================================================
# Monte Carlo upgraded projection
# ==========================================================
sim = enriched_projection(mu, sd, minutes, line, market)

p_over = float(np.clip(sim["prob_final"], 0.05, 0.97))
ev_leg_even = sim["ev_sim"]

# Attach for rendering
return {
    "player":player,"market":market,"line":line,"mu":mu,"sd":sd,
    "prob_over":p_over,"prob_sim":sim["prob_sim"],
    "p50":sim["p50"],"p75":sim["p75"],"p95":sim["p95"],
    "ev_leg_even":ev_leg_even,"team":team,
    "ctx_mult":ctx_mult,"msg":msg,
    "teammate_out":teammate_out,"blowout":blowout
},None
# ================================================================
# SECTION 6 — Advanced Correlation Engine for 2-Pick Combos
# ================================================================

def estimate_market_pair_correlation(m1, m2):
    """
    Baseline relationship between markets:
    Positive: PRA↔PTS, PRA↔REB, PTS↔AST (scoring → assists)
    Moderate: REB↔AST (pace-driven)
    Negative: PTS↔REB (scorers rebound less on high-usage nights)
    """
    pairs = {
        ("PRA", "Points"): 0.28,
        ("PRA", "Rebounds"): 0.20,
        ("PRA", "Assists"): 0.22,
        ("Points", "Assists"): 0.18,
        ("Points", "Rebounds"): -0.12,
        ("Rebounds", "Assists"): 0.10,
    }
    return pairs.get((m1, m2), pairs.get((m2, m1), 0.0))


def estimate_team_usage_corr(leg1, leg2):
    """Same team = correlated usage. Opposing teams = pace correlation."""
    if not leg1["team"] or not leg2["team"]:
        return 0.0
    if leg1["team"] == leg2["team"]:
        return 0.25  # team usage correlation
    else:
        return 0.12  # same game pace correlation (approx NBA avg +12%)


def estimate_minutes_volatility_corr(mu1, mu2, sd1, sd2):
    """Players with high SD/minutes often share volatility patterns."""
    vol1 = sd1 / max(mu1, 1e-9)
    vol2 = sd2 / max(mu2, 1e-9)
    return min(0.18, (vol1 + vol2) / 8)


def estimate_matchup_corr(leg1, leg2):
    """Pace-up or pace-down games increase cov for all stats."""
    ctx1 = leg1["ctx_mult"]
    ctx2 = leg2["ctx_mult"]
    pace_sync = (ctx1 - 1) * (ctx2 - 1)
    return np.clip(pace_sync * 0.30, -0.15, 0.20)


def synthetic_covariance(p1, p2):
    """
    Gaussian copula approximation for Monte Carlo covariance:
    Cov ≈ sqrt(p1(1-p1)p2(1-p2))
    """
    return np.sqrt(p1 * (1 - p1) * p2 * (1 - p2))


def combined_correlation(leg1, leg2):
    """
    FINAL correlation estimate combining:
    - Market pair relationship
    - Team usage / same-game pace
    - Minutes/role volatility
    - Opponent matchup alignment
    """
    
    m_corr = estimate_market_pair_correlation(leg1["market"], leg2["market"])
    t_corr = estimate_team_usage_corr(leg1, leg2)
    v_corr = estimate_minutes_volatility_corr(leg1["mu"], leg2["mu"], leg1["sd"], leg2["sd"])
    mt_corr = estimate_matchup_corr(leg1, leg2)

    raw = m_corr + t_corr + v_corr + mt_corr

    # Clip to safe NBA-supported correlation range:
    return float(np.clip(raw, -0.30, 0.55))


def joint_probability(leg1, leg2):
    """Statistically correct 2-pick joint probability."""

    p1 = leg1["prob_over"]
    p2 = leg2["prob_over"]
    rho = combined_correlation(leg1, leg2)

    base = p1 * p2
    cov = synthetic_covariance(p1, p2)

    joint = base + rho * cov
    return float(np.clip(joint, 0.0, 1.0)), rho
# =====================================================
# NEW — ADVANCED CORRELATION ENGINE
# =====================================================
joint, rho = joint_probability(leg1, leg2)

ev_combo = payout_mult * joint - 1.0
k_frac = kelly_for_combo(joint, payout_mult, fractional_kelly)
stake = round(bankroll * k_frac, 2)

st.markdown("### 🎯 2-Pick Combo (Both Must Hit)")
st.markdown(f"- Correlation coefficient (ρ): **{rho:+.3f}**")
st.markdown(f"- Joint Hit Probability: **{joint*100:.2f}%**")
st.markdown(f"- EV on 2-pick: **{ev_combo*100:+.1f}%**")
st.markdown(f"- Suggested Stake (Kelly-capped): **${stake:.2f}**")
st.markdown(f"- Recommendation: **{combo_decision(ev_combo)}**")
# ================================================================
# SECTION 7 — Line Movement Engine & CLV Predictor
# ================================================================

LINE_TRACKER_FILE = os.path.join(TEMP_DIR, "line_movement_library.csv")

def ensure_line_tracker():
    """Initialize local line-movement file."""
    if not os.path.exists(LINE_TRACKER_FILE):
        df = pd.DataFrame(columns=[
            "Timestamp","Player","Market","Line","ModelProj","ModelProb",
            "ContextMult","Opp","ExpectedDir","Strength"
        ])
        df.to_csv(LINE_TRACKER_FILE, index=False)

def log_line_snapshot(player, market, line, model_proj, model_prob, ctx_mult, opp):
    ensure_line_tracker()
    df = pd.read_csv(LINE_TRACKER_FILE)

    expected_dir = (
        "Up" if model_proj > line else
        "Down" if model_proj < line else
        "Flat"
    )

    strength = abs(model_proj - line)

    new = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Player": player,
        "Market": market,
        "Line": line,
        "ModelProj": model_proj,
        "ModelProb": model_prob,
        "ContextMult": ctx_mult,
        "Opp": opp,
        "ExpectedDir": expected_dir,
        "Strength": strength
    }

    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(LINE_TRACKER_FILE, index=False)


def get_line_analytics(player, market):
    """Pull last X historical datapoints for this player/market."""
    ensure_line_tracker()
    df = pd.read_csv(LINE_TRACKER_FILE)

    df = df[(df["Player"] == player) & (df["Market"] == market)]
    if df.empty:
        return None, None, None

    last = df.tail(10)   # last 10 entries

    avg_line = last["Line"].mean()
    avg_model = last["ModelProj"].mean()
    avg_prob = last["ModelProb"].mean()

    return avg_line, avg_model, avg_prob


def predict_clv_direction(line, model_proj):
    delta = model_proj - line
    if delta > 2.0:
        return "⬆ Strong Upward Move Expected"
    elif delta > 1.0:
        return "↗ Mild Upward Move Likely"
    elif delta < -2.0:
        return "⬇ Strong Downward Move Expected"
    elif delta < -1.0:
        return "↘ Mild Downward Move Likely"
    else:
        return "➡ Stable / No Sharp Expectation"
# Log market snapshot for line movement tracking
log_line_snapshot(
    leg["player"],
    leg["market"],
    leg["line"],
    leg["mu"],
    leg["prob_over"],
    leg["ctx_mult"],
    opp=""  # optional opponent
)
# =========================================================
# CLV Prediction & Expected Line Movement
# =========================================================
st.markdown("### 📈 Expected Line Movement (CLV Prediction)")

for label, leg in [("P1", leg1), ("P2", leg2)]:
    if leg:
        clv_msg = predict_clv_direction(leg["line"], leg["mu"])
        avg_line, avg_proj, avg_prob = get_line_analytics(leg["player"], leg["market"])

        st.markdown(f"**{label} – {leg['player']} ({leg['market']})**")
        st.markdown(f"- Current line: **{leg['line']}** | Model proj: **{leg['mu']:.1f}**")
        st.markdown(f"- Expected Movement: **{clv_msg}**")

        if avg_line:
            st.caption(
                f"Historical avg: line={avg_line:.1f}, model={avg_proj:.1f}, prob={avg_prob*100:.1f}%"
            )
