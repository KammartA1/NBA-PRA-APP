ort os
import json
import math
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# SECTION A — GLOBAL CONFIG
# =========================

APP_VERSION = "3.0.0-upgraded"
ODDS_API_KEY = "621ec92ab709da9f9ce59cf2e513af55"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA = "basketball_nba"

API_USAGE_FILE = "api_usage.json"
HISTORY_FILE = "bet_history.csv"
CALIBRATION_FILE = "calibration.json"

DEFAULT_KELLY_FRACTION = 0.25  # fractional Kelly
N_MONTE_CARLO_SIMS = 10_000

EDGE_SCANNER_MIN_EV = 0.065  # 6.5% edge threshold for scanner

# ======================
# SECTION B — DATA TYPES
# ======================

@dataclass
class PlayerContext:
    player_name: str
    team: str
    opponent: str
    market: str  # "points", "rebounds", "assists", "pra"
    line: float
    american_odds: int
    side: str  # "over" or "under"
    minutes_expectation: float
    usage_rate: float
    pace_adj_factor: float
    rebound_rate: float
    assist_rate: float
    shot_rate: float
    opponent_def_rating: float
    opponent_reb_pct_allowed: float
    opponent_ast_pct_allowed: float
    injury_teammate_out: bool
    blowout_risk: float  # 0-1
    notes: str = ""

@dataclass
class SimulationResult:
    hit_prob: float
    fair_odds: float
    ev: float
    kelly_fraction: float
    play_pass: str
    volatility: float
    simulated_mean: float
    simulated_std: float

@dataclass
class JointSimulationResult:
    hit_prob_both: float
    ev_both: float
    kelly_fraction_both: float
    play_pass_both: str

# ===============================
# SECTION C — UTILITY / PERSISTENCE
# ===============================

def _today_str() -> str:
    return datetime.date.today().isoformat()

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def _save_json(path: str, data) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def get_api_usage() -> Dict:
    return _load_json(API_USAGE_FILE, {"date": _today_str(), "odds_api_calls": 0})

def register_odds_api_call() -> None:
    usage = get_api_usage()
    today = _today_str()
    if usage.get("date") != today:
        usage = {"date": today, "odds_api_calls": 0}
    usage["odds_api_calls"] = usage.get("odds_api_calls", 0) + 1
    _save_json(API_USAGE_FILE, usage)

def can_call_odds_api() -> bool:
    usage = get_api_usage()
    today = _today_str()
    if usage.get("date") != today:
        return True
    return usage.get("odds_api_calls", 0) < 20

def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "date",
                "game",
                "player_name",
                "team",
                "opponent",
                "market",
                "side",
                "line",
                "american_odds",
                "stake",
                "result",
                "payout",
                "model_prob",
                "ensemble_prob",
                "fair_odds",
                "ev",
                "kelly_fraction",
                "actual_stat",
                "open_odds",
                "closing_odds",
                "clv",
                "notes",
            ]
        )
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame()

def save_history(df: pd.DataFrame) -> None:
    try:
        df.to_csv(HISTORY_FILE, index=False)
    except Exception:
        pass

def load_calibration() -> Dict:
    default = {
        "last_updated": None,
        "probability_scaling": {
            "thin": 1.0,
            "edge": 1.0,
            "strong": 1.0,
        },
        "notes": "Scaling factors applied to raw hit probabilities by EV bucket.",
        "version": APP_VERSION,
    }
    data = _load_json(CALIBRATION_FILE, default)
    # Ensure keys exist
    for k in default["probability_scaling"]:
        data.setdefault("probability_scaling", {})
        data["probability_scaling"].setdefault(k, 1.0)
    return data

def save_calibration(calib: Dict) -> None:
    calib["last_updated"] = datetime.datetime.utcnow().isoformat()
    calib["version"] = APP_VERSION
    _save_json(CALIBRATION_FILE, calib)

# ====================================
# SECTION D — ODDS API & NBA DATA LAYER
# ====================================

def fetch_live_player_props(markets: List[str]) -> Tuple[bool, Optional[List[Dict]], str]:
    """Fetch live player prop markets from The Odds API.

    Returns (ok, data, message).
    """
    if not can_call_odds_api():
        return False, None, "Daily The Odds API request cap (20) reached."

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
    }
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY_NBA}/odds"
    try:
        resp = requests.get(url, params=params, timeout=10)
        register_odds_api_call()
        if resp.status_code != 200:
            return False, None, f"Odds API error: {resp.status_code} — {resp.text[:200]}"
        data = resp.json()
        return True, data, "OK"
    except Exception as e:
        return False, None, f"Odds API exception: {e}"

def flatten_player_props(odds_data: List[Dict]) -> pd.DataFrame:
    """Flatten The Odds API player-prop JSON into a DataFrame.

    This is designed to be resilient to minor schema differences.
    """
    rows = []
    for event in odds_data:
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        game_label = f"{away_team} @ {home_team}"

        for book in event.get("bookmakers", []):
            book_key = book.get("key")
            last_update = book.get("last_update")
            for market in book.get("markets", []):
                mkey = market.get("key")
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description") or outcome.get("name")
                    line = outcome.get("point")
                    price = outcome.get("price")
                    if player_name is None or line is None or price is None:
                        continue
                    rows.append(
                        {
                            "event_id": event_id,
                            "game": game_label,
                            "bookmaker": book_key,
                            "last_update": last_update,
                            "player_name": player_name,
                            "market": mkey,
                            "line": line,
                            "american_odds": int(price),
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "event_id",
                "game",
                "bookmaker",
                "last_update",
                "player_name",
                "market",
                "line",
                "american_odds",
            ]
        )
    return pd.DataFrame(rows)

# Placeholder NBA advanced data fetches. To keep the app robust, these
# functions fail gracefully and fall back to user inputs if nba_api is
# not available or network calls fail.

def fetch_player_last10_stats(player_name: str) -> Optional[pd.DataFrame]:
    try:
        from nba_api.stats.static import players
        from nba_api.stats.endpoints import playergamelog

        pl = players.find_players_by_full_name(player_name)
        if not pl:
            return None
        player_id = pl[0]["id"]
        # Use current season by default; nba_api will infer if left empty
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=None)
        df = gl.get_data_frames()[0]
        return df.head(10)
    except Exception:
        return None

def estimate_usage_and_rates(player_name: str, market: str) -> Tuple[float, float, float, float]:
    """Best-effort estimate of usage, rebound rate, assist rate, shot rate.

    Falls back to generic assumptions if NBA API calls fail.
    """
    df = fetch_player_last10_stats(player_name)
    if df is None or df.empty:
        # Fallback generic assumptions
        return 24.0, 0.12, 0.18, 0.25

    # Basic per-minute rates from last 10 games
    minutes = df["MIN"].astype(str).str.split(":", expand=True).iloc[:, 0].astype(float)
    pts = df["PTS"].astype(float)
    reb = df["REB"].astype(float)
    ast = df["AST"].astype(float)

    min_mean = max(minutes.mean(), 10.0)
    usage = 24.0  # placeholder, could be improved with advanced endpoints
    reb_rate = float((reb / min_mean).mean())
    ast_rate = float((ast / min_mean).mean())
    shot_rate = float((pts / min_mean).mean())

    return usage, reb_rate, ast_rate, shot_rate

def estimate_opponent_defense(team_abbr: str) -> Tuple[float, float, float]:
    """Return (def_rating, def_reb_pct_allowed, def_ast_pct_allowed).

    For robustness this uses simple hard-coded baselines. You can later
    wire in nba_api leaguedashteamstats here.
    """
    # Baseline values: average NBA team
    base_def_rating = 113.0
    base_reb_allowed = 0.50
    base_ast_allowed = 0.50

    # Very lightweight custom tweaks can be added here per team if desired.
    # For now, we keep it simple but still pass these knobs into the engines.
    return base_def_rating, base_reb_allowed, base_ast_allowed

# ===================================
# SECTION E — PROJECTION / RISK ENGINES
# ===================================

def american_to_implied_prob(odds: int) -> float:
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)

def implied_prob_to_american(p: float) -> float:
    p = max(min(p, 0.999), 0.001)
    if p >= 0.5:
        return - (p * 100.0) / (1.0 - p)
    else:
        return (1.0 - p) * 100.0 / p

def kelly_fraction(prob: float, odds: int, frac: float = DEFAULT_KELLY_FRACTION) -> float:
    b = abs(odds) / 100.0 if odds != 0 else 1.0
    q = 1.0 - prob
    k = (prob * (b + 1.0) - 1.0) / b
    k = max(k, 0.0)
    return k * frac

def classify_ev_bucket(ev: float) -> str:
    if ev < 0.05:
        return "thin"
    elif ev < 0.20:
        return "edge"
    else:
        return "strong"

def volatility_score_from_gamelog(df: Optional[pd.DataFrame], market: str) -> float:
    if df is None or df.empty:
        return 1.0
    if market == "points":
        series = df["PTS"].astype(float)
    elif market == "rebounds":
        series = df["REB"].astype(float)
    elif market == "assists":
        series = df["AST"].astype(float)
    else:  # PRA
        series = df["PTS"].astype(float) + df["REB"].astype(float) + df["AST"].astype(float)
    return float(series.std(ddof=1))

def weighted_bootstrap_samples(df: Optional[pd.DataFrame], market: str, n: int) -> np.ndarray:
    if df is None or df.empty:
        return np.array([])
    if market == "points":
        vals = df["PTS"].astype(float).values
    elif market == "rebounds":
        vals = df["REB"].astype(float).values
    elif market == "assists":
        vals = df["AST"].astype(float).values
    else:
        vals = (df["PTS"].astype(float) + df["REB"].astype(float) + df["AST"].astype(float)).values
    # Exponential weighting: recent games heavier
    idx = np.arange(len(vals))
    weights = np.exp(idx - idx.max())
    weights = weights / weights.sum()
    samples = np.random.choice(vals, size=n, p=weights)
    return samples

def game_script_minutes_samples(base_minutes: float, blowout_risk: float, n: int) -> np.ndarray:
    # Pace factor ~ N(1.0, 0.08^2), truncated to [0.85, 1.15]
    pace = np.random.normal(loc=1.0, scale=0.08, size=n)
    pace = np.clip(pace, 0.85, 1.15)
    blowout_flags = np.random.binomial(1, blowout_risk, size=n)
    minutes = base_minutes * pace * (1.0 - 0.30 * blowout_flags)
    return minutes

def apply_defense_adjustments(base_mean: float, ctx: PlayerContext) -> float:
    # Simple linear adjustments that can be refined later.
    def_factor = (113.0 / max(ctx.opponent_def_rating, 90.0))  # tougher D -> <1
    reb_factor = 1.0 + (0.50 - ctx.opponent_reb_pct_allowed)
    ast_factor = 1.0 + (0.50 - ctx.opponent_ast_pct_allowed)
    if ctx.market == "rebounds":
        adj = base_mean * reb_factor * def_factor
    elif ctx.market == "assists":
        adj = base_mean * ast_factor * def_factor
    else:
        adj = base_mean * def_factor
    # Injury bump: if key teammate out, modest usage bump
    if ctx.injury_teammate_out:
        adj *= 1.07
    return adj

def projection_engine(ctx: PlayerContext) -> Tuple[float, float]:
    """Return (projection_mean, projection_std_per_minute)."""
    df_last10 = fetch_player_last10_stats(ctx.player_name)
    vol = volatility_score_from_gamelog(df_last10, ctx.market)
    if df_last10 is not None and not df_last10.empty:
        minutes = df_last10["MIN"].astype(str).str.split(":", expand=True).iloc[:, 0].astype(float)
        if ctx.market == "points":
            vals = df_last10["PTS"].astype(float)
        elif ctx.market == "rebounds":
            vals = df_last10["REB"].astype(float)
        elif ctx.market == "assists":
            vals = df_last10["AST"].astype(float)
        else:
            vals = (
                df_last10["PTS"].astype(float)
                + df_last10["REB"].astype(float)
                + df_last10["AST"].astype(float)
            )
        base_per_min = (vals / minutes).mean()
    else:
        # Fallback per-minute baselines tied to usage/market
        if ctx.market == "points":
            base_per_min = ctx.shot_rate * 0.5
        elif ctx.market == "rebounds":
            base_per_min = ctx.rebound_rate
        elif ctx.market == "assists":
            base_per_min = ctx.assist_rate
        else:
            base_per_min = ctx.shot_rate * 0.4 + ctx.rebound_rate + ctx.assist_rate

    minutes_proj = ctx.minutes_expectation
    base_mean = base_per_min * minutes_proj
    mean_adj = apply_defense_adjustments(base_mean, ctx)

    # Approximate per-minute std from volatility
    std_per_min = max(vol / max(minutes_proj, 1.0), 0.1)
    return mean_adj, std_per_min

def ensemble_hit_probability(ctx: PlayerContext, calib: Dict) -> Tuple[SimulationResult, np.ndarray]:
    """Monte Carlo + bootstrap + game script ensemble.

    Returns SimulationResult and the raw samples used.
    """
    df_last10 = fetch_player_last10_stats(ctx.player_name)
    # Base projection & volatility
    mean_adj, std_per_min = projection_engine(ctx)
    minutes_samples = game_script_minutes_samples(ctx.minutes_expectation, ctx.blowout_risk, N_MONTE_CARLO_SIMS)
    # Model 1: parametric normal per-minute
    per_min_mean = mean_adj / max(ctx.minutes_expectation, 1.0)
    base_std = std_per_min * np.sqrt(minutes_samples / max(ctx.minutes_expectation, 1.0))
    normal_samples = np.random.normal(
        loc=per_min_mean * minutes_samples,
        scale=np.clip(base_std, 0.5, None),
        size=N_MONTE_CARLO_SIMS,
    )

    # Model 2: weighted bootstrap of last 10 games
    boot = weighted_bootstrap_samples(df_last10, ctx.market, N_MONTE_CARLO_SIMS)
    if boot.size == 0:
        boot = normal_samples.copy()

    # Model 3: usage/pace/injury-adjusted deterministic projection + noise
    usage_factor = ctx.usage_rate / 22.0
    pace_factor = ctx.pace_adj_factor
    injury_factor = 1.10 if ctx.injury_teammate_out else 1.0
    script_mean = mean_adj * usage_factor * pace_factor * injury_factor
    script_std = max(script_mean * 0.20, 1.0)
    script_samples = np.random.normal(loc=script_mean, scale=script_std, size=N_MONTE_CARLO_SIMS)

    # Ensemble samples (simple average of three models)
    samples = (normal_samples + boot + script_samples) / 3.0

    # Side-aware hit probability
    if ctx.side == "over":
        raw_prob = float(np.mean(samples > ctx.line))
    else:
        raw_prob = float(np.mean(samples < ctx.line))

    simulated_mean = float(np.mean(samples))
    simulated_std = float(np.std(samples, ddof=1))
    volatility = float(np.std(boot if boot.size else samples, ddof=1))

    # Calibration scaling by EV bucket (we don't know EV yet, so assume "edge" then re-bucket below)
    scaling = calib.get("probability_scaling", {}).get("edge", 1.0)
    prob = max(min(raw_prob * scaling, 0.995), 0.005)

    # Fair odds and EV
    fair_odds = implied_prob_to_american(prob)
    implied = american_to_implied_prob(ctx.american_odds)
    ev = prob * (abs(ctx.american_odds) / 100.0) - (1 - prob)
    kelly_frac = kelly_fraction(prob, ctx.american_odds)

    bucket = classify_ev_bucket(max(ev, 0.0))
    bucket_scale = calib.get("probability_scaling", {}).get(bucket, 1.0)
    prob *= bucket_scale
    prob = max(min(prob, 0.995), 0.005)
    fair_odds = implied_prob_to_american(prob)
    ev = prob * (abs(ctx.american_odds) / 100.0) - (1 - prob)
    kelly_frac = kelly_fraction(prob, ctx.american_odds)

    play_pass = "PLAY" if ev > 0 and prob > implied else "PASS"

    sim_result = SimulationResult(
        hit_prob=prob,
        fair_odds=fair_odds,
        ev=ev,
        kelly_fraction=kelly_frac,
        play_pass=play_pass,
        volatility=volatility,
        simulated_mean=simulated_mean,
        simulated_std=simulated_std,
    )

    return sim_result, samples

def covariance_joint_probability(ctx1: PlayerContext, ctx2: PlayerContext, calib: Dict) -> JointSimulationResult:
    # Approximate joint MC with a correlation assumption based on markets
    base_corr = 0.1
    if ctx1.player_name == ctx2.player_name:
        if {ctx1.market, ctx2.market} == {"points", "assists"}:
            base_corr = 0.35
        elif {ctx1.market, ctx2.market} == {"points", "rebounds"}:
            base_corr = 0.25
        else:
            base_corr = 0.3
    elif ctx1.team == ctx2.team:
        base_corr = 0.15

    # Individual ensembles to get means/stds
    res1, samples1 = ensemble_hit_probability(ctx1, calib)
    res2, samples2 = ensemble_hit_probability(ctx2, calib)

    # Use rank correlation if samples available; otherwise stick with base
    if samples1.size and samples2.size:
        rho = float(np.corrcoef(samples1, samples2)[0, 1])
        corr = 0.5 * rho + 0.5 * base_corr
    else:
        corr = base_corr

    # Build correlated normal draws around the original samples' distributions
    mean1, mean2 = np.mean(samples1), np.mean(samples2)
    std1, std2 = np.std(samples1), np.std(samples2, ddof=1)
    cov = corr * std1 * std2
    cov_matrix = np.array([[std1 ** 2, cov], [cov, std2 ** 2]])
    mvn = np.random.multivariate_normal(mean=[mean1, mean2], cov=cov_matrix, size=N_MONTE_CARLO_SIMS)
    s1 = mvn[:, 0]
    s2 = mvn[:, 1]

    if ctx1.side == "over":
        leg1_hit = s1 > ctx1.line
    else:
        leg1_hit = s1 < ctx1.line
    if ctx2.side == "over":
        leg2_hit = s2 > ctx2.line
    else:
        leg2_hit = s2 < ctx2.line

    both_hit_prob = float(np.mean(leg1_hit & leg2_hit))

    # Assume user enters combined parlay odds separately; if not, approximate as product
    combo_odds = st.session_state.get("parlay_american_odds", None)
    if combo_odds is None:
        # naive approximation: multiply decimal odds
        dec1 = 1 + abs(ctx1.american_odds) / 100.0
        dec2 = 1 + abs(ctx2.american_odds) / 100.0
        combo_dec = dec1 * dec2
        if combo_dec >= 2:
            combo_odds = int((combo_dec - 1) * 100)
        else:
            combo_odds = -int(100 / (combo_dec - 1))

    implied_parlay = american_to_implied_prob(combo_odds)
    ev_both = both_hit_prob * (abs(combo_odds) / 100.0) - (1 - both_hit_prob)
    kelly_both = kelly_fraction(both_hit_prob, combo_odds)
    play_pass_both = "PLAY" if ev_both > 0 and both_hit_prob > implied_parlay else "PASS"

    return JointSimulationResult(
        hit_prob_both=both_hit_prob,
        ev_both=ev_both,
        kelly_fraction_both=kelly_both,
        play_pass_both=play_pass_both,
    )

# ======================================
# SECTION F — CALIBRATION / FEEDBACK LOOP
# ======================================

def update_calibration_from_history(hist: pd.DataFrame) -> Dict:
    calib = load_calibration()
    if hist.empty:
        return calib

    # Only use rows with results and model_prob
    df = hist.dropna(subset=["result", "model_prob"])
    if df.empty:
        return calib

    # Binary outcome: 1 for win, 0 for loss; pushes excluded
    df = df[df["result"].isin(["win", "loss"])].copy()
    if df.empty:
        return calib
    df["hit"] = (df["result"] == "win").astype(int)

    # EV buckets based on stored EV
    df["ev_bucket"] = df["ev"].apply(classify_ev_bucket)
    scales = {}
    for bucket in ["thin", "edge", "strong"]:
        sub = df[df["ev_bucket"] == bucket]
        if len(sub) < 30:
            scales[bucket] = calib["probability_scaling"].get(bucket, 1.0)
            continue
        expected_hits = (sub["model_prob"]).sum()
        actual_hits = sub["hit"].sum()
        if expected_hits <= 0:
            scales[bucket] = calib["probability_scaling"].get(bucket, 1.0)
        else:
            # scale so expected ~ actual
            scale = float(actual_hits / expected_hits)
            # clip to reasonable range
            scale = float(max(min(scale, 1.2), 0.8))
            scales[bucket] = scale

    calib["probability_scaling"].update(scales)
    save_calibration(calib)
    return calib

def compute_performance_metrics(hist: pd.DataFrame) -> Dict[str, float]:
    if hist.empty:
        return {}
    df = hist.dropna(subset=["result", "stake"])
    if df.empty:
        return {}
    df = df[df["result"].isin(["win", "loss", "push"])].copy()

    total_bets = len(df)
    wins = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    pushes = (df["result"] == "push").sum()
    hit_rate = wins / max(wins + losses, 1)

    # Profit per bet assuming American odds and stakes
    profits = []
    for _, row in df.iterrows():
        stake = row["stake"]
        odds = int(row["american_odds"])
        if row["result"] == "win":
            profits.append(stake * (abs(odds) / 100.0))
        elif row["result"] == "loss":
            profits.append(-stake)
        else:
            profits.append(0.0)
    profits = np.array(profits)
    roi = float(profits.sum() / max(df["stake"].sum(), 1.0))
    variance = float(np.var(profits, ddof=1))
    avg_clv = float(df["clv"].dropna().mean()) if "clv" in df.columns else 0.0

    return {
        "total_bets": total_bets,
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "hit_rate": hit_rate,
        "roi": roi,
        "variance": variance,
        "avg_clv": avg_clv,
    }

# ============================
# SECTION G — STREAMLIT UI LAYER
# ============================

def init_session_state():
    defaults = {
        "legs": [],
        "last_results": [],
        "parlay_american_odds": None,
        "bet_logged_for_last_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def build_player_context(index: int) -> Optional[PlayerContext]:
    st.subheader(f"Leg {index + 1}")
    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Player name", key=f"player_name_{index}")
        team = st.text_input("Team (abbr)", key=f"team_{index}")
        opponent = st.text_input("Opponent (abbr)", key=f"opponent_{index}")
    with col2:
        market = st.selectbox(
            "Market",
            options=["points", "rebounds", "assists", "pra"],
            key=f"market_{index}",
        )
        side = st.selectbox("Side", options=["over", "under"], key=f"side_{index}")
        line = st.number_input("Line", key=f"line_{index}", value=20.5)
        american_odds = st.number_input("American odds", key=f"odds_{index}", value=-115, step=5)

    adv = st.expander("Advanced context (minutes, usage, pace, etc.)", expanded=False)
    with adv:
        col3, col4, col5 = st.columns(3)
        with col3:
            minutes_expectation = st.number_input(
                "Expected minutes",
                min_value=10.0,
                max_value=45.0,
                value=32.0,
                key=f"minutes_{index}",
            )
            usage_rate = st.number_input(
                "Usage rate %",
                min_value=10.0,
                max_value=40.0,
                value=24.0,
                key=f"usage_{index}",
            )
            pace_adj_factor = st.slider(
                "Pace factor",
                min_value=0.8,
                max_value=1.2,
                value=1.0,
                step=0.01,
                key=f"pace_{index}",
            )
        with col4:
            rebound_rate = st.number_input(
                "Rebound rate per min",
                min_value=0.0,
                max_value=2.0,
                value=0.25,
                key=f"reb_rate_{index}",
            )
            assist_rate = st.number_input(
                "Assist rate per min",
                min_value=0.0,
                max_value=1.5,
                value=0.15,
                key=f"ast_rate_{index}",
            )
            shot_rate = st.number_input(
                "Shot attempts per min",
                min_value=0.0,
                max_value=2.5,
                value=0.9,
                key=f"shot_rate_{index}",
            )
        with col5:
            injury_teammate_out = st.checkbox("Key teammate OUT", key=f"injury_{index}")
            blowout_risk = st.slider(
                "Blowout risk",
                min_value=0.0,
                max_value=0.8,
                value=0.15,
                step=0.01,
                key=f"blowout_{index}",
            )
            # Opponent defense info (auto + manual)
            def_rating, reb_pct_allowed, ast_pct_allowed = estimate_opponent_defense(opponent or "")
            st.caption(
                f"Opponent def rating (approx): {def_rating:.1f} | "
                f"Defensive reb% allowed baseline: {reb_pct_allowed:.2f} | "
                f"Ast% allowed baseline: {ast_pct_allowed:.2f}"
            )
        notes = st.text_area("Notes (optional)", key=f"notes_{index}", height=60)

    if not player_name or not team or not opponent:
        return None

    ctx = PlayerContext(
        player_name=player_name,
        team=team,
        opponent=opponent,
        market=market,
        line=float(line),
        american_odds=int(american_odds),
        side=side,
        minutes_expectation=float(minutes_expectation),
        usage_rate=float(usage_rate),
        pace_adj_factor=float(pace_adj_factor),
        rebound_rate=float(rebound_rate),
        assist_rate=float(assist_rate),
        shot_rate=float(shot_rate),
        opponent_def_rating=def_rating,
        opponent_reb_pct_allowed=reb_pct_allowed,
        opponent_ast_pct_allowed=ast_pct_allowed,
        injury_teammate_out=bool(injury_teammate_out),
        blowout_risk=float(blowout_risk),
        notes=notes,
    )
    return ctx

def model_tab(calib: Dict):
    st.header("NBA Prop Quant Engine — Model")
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=50.0)
    kelly_frac_global = st.slider(
        "Global Kelly fraction",
        min_value=0.05,
        max_value=0.5,
        value=DEFAULT_KELLY_FRACTION,
        step=0.05,
    )

    st.markdown("### Legs configuration")
    num_legs = st.slider("Number of legs", min_value=1, max_value=4, value=2, step=1)
    contexts: List[PlayerContext] = []
    for i in range(num_legs):
        ctx = build_player_context(i)
        if ctx:
            contexts.append(ctx)

    st.session_state["parlay_american_odds"] = st.number_input(
        "Parlay American odds (if 2+ legs parlay)",
        value=-110,
        step=5,
    )

    run = st.button("Run model", type="primary")
    if not run:
        return

    if not contexts:
        st.warning("Please complete all required fields for at least one leg.")
        return

    # Reset last run log flag
    st.session_state["bet_logged_for_last_run"] = False

    leg_results = []
    st.markdown("### Single-leg Monte Carlo results")
    cols = st.columns(len(contexts))
    for idx, (ctx, col) in enumerate(zip(contexts, cols)):
        with col:
            sim_res, samples = ensemble_hit_probability(ctx, calib)
            # Override Kelly fraction with global
            k_frac = kelly_fraction(sim_res.hit_prob, ctx.american_odds, frac=kelly_frac_global)
            sim_res.kelly_fraction = k_frac
            stake = bankroll * k_frac
            implied = american_to_implied_prob(ctx.american_odds)
            edge_pct = (sim_res.hit_prob - implied) * 100.0

            card = st.container(border=True)
            with card:
                st.subheader(f"Leg {idx + 1}: {ctx.player_name} {ctx.market} {ctx.side} {ctx.line}")
                st.metric("Defense-adjusted projection", f"{sim_res.simulated_mean:.2f} ± {sim_res.simulated_std:.2f}")
                st.metric("Defense-adjusted hit probability", f"{sim_res.hit_prob*100:.1f}%")
                st.metric("Implied probability (market)", f"{implied*100:.1f}%")
                st.metric("Edge (model - market)", f"{edge_pct:.1f}%")
                st.metric("Fair odds (model)", f"{sim_res.fair_odds:.0f}")
                st.metric("Volatility score", f"{sim_res.volatility:.2f}")
                st.metric("Kelly stake", f"${stake:.2f}")
                st.markdown(f"**Decision:** :{'green_circle' if sim_res.play_pass=='PLAY' else 'red_circle'}: {sim_res.play_pass}")

            leg_results.append(
                {
                    "context": ctx,
                    "sim_result": sim_res,
                    "stake": stake,
                    "samples": samples,
                }
            )

    # Joint 2-leg simulation (UltraMax)
    joint_result = None
    if len(contexts) == 2:
        st.markdown("### 2-Leg Covariance-based Joint Monte Carlo")
        ctx1, ctx2 = contexts
        joint_result = covariance_joint_probability(ctx1, ctx2, calib)
        combo_odds = st.session_state.get("parlay_american_odds", -110)
        implied_parlay = american_to_implied_prob(combo_odds)
        edge_pct = (joint_result.hit_prob_both - implied_parlay) * 100.0
        stake_both = bankroll * joint_result.kelly_fraction_both

        joint_card = st.container(border=True)
        with joint_card:
            label = (
                f"{ctx1.player_name} {ctx1.market} {ctx1.side} {ctx1.line}  +  "
                f"{ctx2.player_name} {ctx2.market} {ctx2.side} {ctx2.line}"
            )
            st.subheader(label)
            st.metric("Joint hit probability", f"{joint_result.hit_prob_both*100:.1f}%")
            st.metric("Market implied prob (parlay)", f"{implied_parlay*100:.1f}%")
            st.metric("Edge (model - market)", f"{edge_pct:.1f}%")
            st.metric("Kelly stake (parlay)", f"${stake_both:.2f}")
            st.markdown(
                f"**2-Leg Decision:** :{'green_circle' if joint_result.play_pass_both=='PLAY' else 'red_circle'}: "
                f"{joint_result.play_pass_both}"
            )

    # Save into session for Results & History tabs
    st.session_state["last_results"] = [
        {
            "player_name": r["context"].player_name,
            "team": r["context"].team,
            "opponent": r["context"].opponent,
            "market": r["context"].market,
            "side": r["context"].side,
            "line": r["context"].line,
            "american_odds": r["context"].american_odds,
            "hit_prob": r["sim_result"].hit_prob,
            "fair_odds": r["sim_result"].fair_odds,
            "ev": r["sim_result"].ev,
            "kelly_fraction": r["sim_result"].kelly_fraction,
            "stake": r["stake"],
            "projection_mean": r["sim_result"].simulated_mean,
            "projection_std": r["sim_result"].simulated_std,
            "volatility": r["sim_result"].volatility,
            "notes": r["context"].notes,
        }
        for r in leg_results
    ]
    st.session_state["last_joint_result"] = (
        {
            "joint_prob": joint_result.hit_prob_both,
            "joint_ev": joint_result.ev_both,
            "joint_kelly": joint_result.kelly_fraction_both,
            "joint_play_pass": joint_result.play_pass_both,
            "parlay_odds": st.session_state.get("parlay_american_odds", None),
        }
        if joint_result is not None
        else None
    )

    st.success("Model run complete.")

    # Prompt for bet logging
    st.markdown("---")
    st.markdown("### Bet logging")
    placed = st.radio(
        "Did you place this bet?",
        options=["No", "Yes"],
        index=0,
        horizontal=True,
    )
    if placed == "Yes":
        history = load_history()
        now = datetime.datetime.now()
        for res in st.session_state["last_results"]:
            history = pd.concat(
                [
                    history,
                    pd.DataFrame(
                        [
                            {
                                "timestamp": now.isoformat(),
                                "date": now.date().isoformat(),
                                "game": f"{res['team']} vs {res['opponent']}",
                                "player_name": res["player_name"],
                                "team": res["team"],
                                "opponent": res["opponent"],
                                "market": res["market"],
                                "side": res["side"],
                                "line": res["line"],
                                "american_odds": res["american_odds"],
                                "stake": res["stake"],
                                "result": None,
                                "payout": None,
                                "model_prob": res["hit_prob"],
                                "ensemble_prob": res["hit_prob"],
                                "fair_odds": res["fair_odds"],
                                "ev": res["ev"],
                                "kelly_fraction": res["kelly_fraction"],
                                "actual_stat": None,
                                "open_odds": res["american_odds"],
                                "closing_odds": None,
                                "clv": None,
                                "notes": res["notes"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        save_history(history)
        st.session_state["bet_logged_for_last_run"] = True
        st.success("Bet(s) logged to history.")

def results_tab():
    st.header("Results & Game Scripts")
    last = st.session_state.get("last_results", [])
    joint = st.session_state.get("last_joint_result", None)
    if not last:
        st.info("Run the model in the Model tab to see results.")
        return
    df = pd.DataFrame(last)
    st.subheader("Single-leg outputs")
    st.dataframe(df.style.format({"hit_prob": "{:.3f}", "ev": "{:.3f}", "stake": "{:.2f}"}), use_container_width=True)

    if joint is not None:
        st.subheader("2-leg joint simulation")
        st.write(
            f"Joint hit probability: **{joint['joint_prob']*100:.1f}%** | "
            f"Joint EV: **{joint['joint_ev']:.3f}** | "
            f"Kelly fraction: **{joint['joint_kelly']:.3f}** | "
            f"Decision: **{joint['joint_play_pass']}** | "
            f"Parlay odds: **{joint['parlay_odds']}**"
        )

    st.markdown("### Live Edge Scanner (The Odds API)")
    st.caption("Automatically detects only high-EV edges (>6.5% EV) from current player props feed.")
    if not can_call_odds_api():
        st.warning("Daily The Odds API request cap reached — scanner disabled.")
        return

    scan = st.button("Run live edge scanner")
    if not scan:
        return

    ok, data, msg = fetch_live_player_props(
        markets=[
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_points_rebounds_assists",
        ]
    )
    if not ok or data is None:
        st.error(f"Could not fetch live props: {msg}")
        return

    props_df = flatten_player_props(data)
    if props_df.empty:
        st.info("No player props returned from The Odds API.")
        return

    calib = load_calibration()
    edges = []
    # Limit to avoid burning calls / time; sample subset
    subset = props_df.sample(min(len(props_df), 40), random_state=42)
    progress = st.progress(0.0)
    for i, row in enumerate(subset.itertuples()):
        ctx = PlayerContext(
            player_name=row.player_name,
            team="",
            opponent=row.game,
            market=row.market.replace("player_", "").replace("_rebounds", "rebounds").replace("_assists", "assists").replace("_points", "points").replace("points_rebounds_assists", "pra"),
            line=float(row.line),
            american_odds=int(row.american_odds),
            side="over",
            minutes_expectation=32.0,
            usage_rate=24.0,
            pace_adj_factor=1.0,
            rebound_rate=0.25,
            assist_rate=0.15,
            shot_rate=0.9,
            opponent_def_rating=113.0,
            opponent_reb_pct_allowed=0.5,
            opponent_ast_pct_allowed=0.5,
            injury_teammate_out=False,
            blowout_risk=0.15,
            notes="Scanner auto-evaluated.",
        )
        sim_res, _ = ensemble_hit_probability(ctx, calib)
        implied = american_to_implied_prob(ctx.american_odds)
        edge = sim_res.hit_prob - implied
        if edge >= EDGE_SCANNER_MIN_EV:
            edges.append(
                {
                    "game": row.game,
                    "player_name": row.player_name,
                    "market": ctx.market,
                    "line": ctx.line,
                    "american_odds": ctx.american_odds,
                    "model_prob": sim_res.hit_prob,
                    "market_implied": implied,
                    "edge_pct": edge * 100.0,
                    "fair_odds": sim_res.fair_odds,
                    "decision": sim_res.play_pass,
                }
            )
        progress.progress((i + 1) / len(subset))

    if not edges:
        st.info("Scanner did not find any edges above the EV threshold.")
        return

    edges_df = pd.DataFrame(edges).sort_values("edge_pct", ascending=False)
    st.subheader("Detected edges")
    st.dataframe(
        edges_df.style.format(
            {
                "model_prob": "{:.3f}",
                "market_implied": "{:.3f}",
                "edge_pct": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

def history_tab():
    st.header("History & Performance")
    hist = load_history()
    if hist.empty:
        st.info("No bets logged yet.")
        return

    st.subheader("Bet history")
    st.dataframe(hist, use_container_width=True)

    st.markdown("### Update results for open bets")
    open_bets = hist[hist["result"].isna()]
    if not open_bets.empty:
        idx = st.selectbox(
            "Select open bet to update",
            options=list(open_bets.index),
            format_func=lambda i: f"{hist.loc[i, 'date']} — {hist.loc[i, 'player_name']} {hist.loc[i, 'market']} {hist.loc[i, 'side']} {hist.loc[i, 'line']}",
        )
        result = st.selectbox("Result", options=["win", "loss", "push"])
        actual_stat = st.number_input("Actual stat value", min_value=0.0, value=0.0)
        closing_odds = st.number_input(
            "Closing American odds (for CLV)",
            value=int(hist.loc[idx, "american_odds"]),
            step=5,
        )
        if st.button("Save result update"):
            hist.loc[idx, "result"] = result
            hist.loc[idx, "actual_stat"] = actual_stat
            hist.loc[idx, "closing_odds"] = closing_odds
            open_imp = american_to_implied_prob(int(hist.loc[idx, "open_odds"]))
            close_imp = american_to_implied_prob(int(closing_odds))
            hist.loc[idx, "clv"] = close_imp - open_imp
            # Compute payout
            stake = hist.loc[idx, "stake"]
            odds = int(hist.loc[idx, "american_odds"])
            if result == "win":
                hist.loc[idx, "payout"] = stake * (abs(odds) / 100.0)
            elif result == "loss":
                hist.loc[idx, "payout"] = -stake
            else:
                hist.loc[idx, "payout"] = 0.0
            save_history(hist)
            st.success("Bet updated.")

    st.markdown("### Performance metrics")
    metrics = compute_performance_metrics(hist)
    if not metrics:
        st.info("Not enough completed bets to compute metrics.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total bets", metrics["total_bets"])
            st.metric("Wins / Losses / Pushes", f"{metrics['wins']} / {metrics['losses']} / {metrics['pushes']}")
        with col2:
            st.metric("Hit rate", f"{metrics['hit_rate']*100:.1f}%")
            st.metric("ROI", f"{metrics['roi']*100:.1f}%")
        with col3:
            st.metric("Outcome variance", f"{metrics['variance']:.2f}")
            st.metric("Avg CLV", f"{metrics['avg_clv']*100:.2f}%")

def calibration_tab():
    st.header("Calibration & Model Drift")
    hist = load_history()
    calib = load_calibration()

    st.subheader("Current calibration scaling")
    st.json(calib)

    if st.button("Recalibrate from history"):
        calib = update_calibration_from_history(hist)
        st.success("Calibration updated.")
        st.json(calib)

    st.markdown("### Edge bins performance")
    if hist.empty:
        st.info("No history to analyze.")
        return
    df = hist.dropna(subset=["result", "ev"])
    if df.empty:
        st.info("No bets with EV and result stored yet.")
        return
    df = df[df["result"].isin(["win", "loss"])].copy()
    df["ev_bucket"] = df["ev"].apply(classify_ev_bucket)
    summary = (
        df.groupby("ev_bucket")
        .agg(
            bets=("ev", "count"),
            avg_ev=("ev", "mean"),
            hit_rate=("result", lambda x: (x == "win").mean()),
            avg_clv=("clv", "mean"),
        )
        .reset_index()
    )
    st.dataframe(
        summary.style.format(
            {
                "avg_ev": "{:.3f}",
                "hit_rate": "{:.3f}",
                "avg_clv": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("### Playbook patterns")
    st.caption("Quick view of where edges tend to come from — by market type and injury context.")
    if "market" in hist.columns:
        playbook = (
            hist.dropna(subset=["result"])
            .assign(win=lambda d: (d["result"] == "win").astype(int))
            .groupby(["market", "notes"])
            .agg(
                bets=("win", "count"),
                win_rate=("win", "mean"),
                avg_ev=("ev", "mean"),
            )
            .reset_index()
            .sort_values("avg_ev", ascending=False)
            .head(50)
        )
        st.dataframe(
            playbook.style.format({"win_rate": "{:.3f}", "avg_ev": "{:.3f}"}),
            use_container_width=True,
        )

# =====================
# APP ENTRYPOINT / MAIN
# =====================

def main():
    st.set_page_config(page_title="NBA Prop Quant Engine", layout="wide")
    init_session_state()
    calib = load_calibration()

    st.sidebar.title("Bankroll & Risk Controls")
    usage = get_api_usage()
    st.sidebar.metric("Odds API calls used today", f"{usage.get('odds_api_calls', 0)}/20")
    st.sidebar.info(
        "Risk notes:\\n"
        "- Fractional Kelly sizing (configurable in Model tab).\\n"
        "- Daily Odds API calls capped at 20 for CLV & scanner use."
    )

    tab_model, tab_results, tab_history, tab_calibration = st.tabs(
        ["Model", "Results", "History", "Calibration"]
    )

    with tab_model:
        model_tab(calib)
    with tab_results:
        results_tab()
    with tab_history:
        history_tab()
    with tab_calibration:
        calibration_tab()

if __name__ == "__main__":
    main()
'''
print(len(app_code.splitlines()))
compile(app_code, "app.py", "exec")
