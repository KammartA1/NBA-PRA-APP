# ============================================================
# NBA PROP QUANT ENGINE v2.0 — Complete Rewrite
# Premium Quant Terminal UI + Hardened Model
# ============================================================

import os, re, math, time, json, difflib
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import requests
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    playergamelog, scoreboardv2, LeagueDashTeamStats, CommonPlayerInfo
)

# ──────────────────────────────────────────────
# SAFE HELPERS
# ──────────────────────────────────────────────
def safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def safe_round(v, d=2):
    try:
        return round(float(v), d) if v is not None else None
    except Exception:
        return None

def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\.\'\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
ODDS_BASE       = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA   = "basketball_nba"
REGION_US       = "us"

ODDS_MARKETS = {
    "Points":   "player_points",
    "Rebounds": "player_rebounds",
    "Assists":  "player_assists",
    "3PM":      "player_threes",
    "PRA":      "player_points_rebounds_assists",
    "PR":       "player_points_rebounds",
    "PA":       "player_points_assists",
    "RA":       "player_rebounds_assists",
    "Blocks":   "player_blocks",
    "Steals":   "player_steals",
    "Turnovers":"player_turnovers",
}

STAT_FIELDS = {
    "Points":   "PTS",
    "Rebounds": "REB",
    "Assists":  "AST",
    "3PM":      "FG3M",
    "PRA":      ("PTS","REB","AST"),
    "PR":       ("PTS","REB"),
    "PA":       ("PTS","AST"),
    "RA":       ("REB","AST"),
    "Blocks":   "BLK",
    "Steals":   "STL",
    "Turnovers":"TOV",
}

BOOK_SHARPNESS: dict[str, float] = {
    "pinnacle":0.99,"circa":0.95,"bookmaker":0.90,"betcris":0.85,
    "draftkings":0.70,"fanduel":0.70,"betmgm":0.65,"caesars":0.65,
    "betrivers":0.60,"pointsbetus":0.55,
    "betonlineag":0.45,"bovada":0.40,"mybookieag":0.30,
}

def book_sharpness(k: str|None) -> float:
    return float(BOOK_SHARPNESS.get((k or "").strip().lower(), 0.55))

# ──────────────────────────────────────────────
# POSITIONAL PRIORS (Bayesian shrinkage targets)
# Typical starter-level season averages by position/market
# ──────────────────────────────────────────────
POSITIONAL_PRIORS: dict[str, dict[str, float]] = {
    "Guard": {"Points":16.5,"Rebounds":3.4,"Assists":5.8,"3PM":2.1,
              "PRA":25.7,"PR":19.9,"PA":22.3,"RA":9.2,"Blocks":0.4,"Steals":1.2,"Turnovers":2.2},
    "Wing":  {"Points":14.8,"Rebounds":5.9,"Assists":2.9,"3PM":1.6,
              "PRA":23.6,"PR":20.7,"PA":17.7,"RA":8.8,"Blocks":0.8,"Steals":1.0,"Turnovers":1.7},
    "Big":   {"Points":13.2,"Rebounds":8.8,"Assists":2.1,"3PM":0.5,
              "PRA":24.1,"PR":22.0,"PA":15.3,"RA":10.9,"Blocks":1.4,"Steals":0.7,"Turnovers":2.0},
    "Unknown":{"Points":14.8,"Rebounds":5.5,"Assists":3.5,"3PM":1.4,
              "PRA":23.8,"PR":20.3,"PA":18.3,"RA":9.0,"Blocks":0.8,"Steals":0.9,"Turnovers":1.9},
}

# REST MULTIPLIERS (days since last game → stat multiplier)
REST_MULTIPLIERS = {0: 0.93, 1: 0.97, 2: 1.00, 3: 1.01, 4: 1.02}

# ──────────────────────────────────────────────
# PLAYER POSITION CACHE
# ──────────────────────────────────────────────
_POSITION_CACHE: dict[str, str] = {}

def get_player_position(name: str) -> str:
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key in _POSITION_CACHE:
        return _POSITION_CACHE[key]
    try:
        matches = nba_players.find_players_by_full_name(name)
    except Exception:
        matches = []
    pos = ""
    if matches:
        pid = matches[0].get("id")
        if pid:
            try:
                info = CommonPlayerInfo(player_id=pid).get_data_frames()[0]
                raw = str(info.get("POSITION", "") or info.get("POSITION_SHORT","") or "")
                pos = raw or ""
            except Exception:
                pos = ""
    _POSITION_CACHE[key] = pos
    return pos

def get_position_bucket(pos: str) -> str:
    if not pos: return "Unknown"
    p = str(pos).upper()
    if p.startswith("G"): return "Guard"
    if p.startswith("F"): return "Wing"
    if p.startswith("C"): return "Big"
    if "G" in p and "F" in p: return "Wing"
    if "F" in p and "C" in p: return "Big"
    return "Unknown"

# ──────────────────────────────────────────────
# SEASON HELPER
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)
def get_season_string(today=None):
    d = today or date.today()
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start+1)%100:02d}"

# ──────────────────────────────────────────────
# GAME LOG FETCHER (robust, multi-version nba_api)
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_player_gamelog(player_id: int, max_games: int = 15) -> tuple[pd.DataFrame, list[str]]:
    errs: list[str] = []
    season_str = get_season_string()
    for params in [{"season_nullable": season_str}, {"season": season_str}, {}]:
        try:
            gl = playergamelog.PlayerGameLog(player_id=player_id, **params)
            df = gl.get_data_frames()[0]
            if not df.empty:
                return df.head(int(max_games)).copy(), errs
            errs.append(f"Empty log with params {params}")
        except TypeError as te:
            errs.append(f"TypeError {params}: {te}")
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")
    return pd.DataFrame(), errs

# ──────────────────────────────────────────────
# REST / B2B FACTOR  ← NEW
# ──────────────────────────────────────────────
def compute_rest_factor(game_log_df: pd.DataFrame, game_date: date) -> tuple[float, int]:
    """
    Compute a rest-days multiplier from the most recent game date in the log.
    Returns (multiplier, rest_days).
    """
    if game_log_df is None or game_log_df.empty:
        return 1.00, 2  # neutral default
    try:
        dates_raw = pd.to_datetime(game_log_df["GAME_DATE"], errors="coerce").dropna()
        if dates_raw.empty:
            return 1.00, 2
        last_game = dates_raw.max().date()
        rest = (game_date - last_game).days - 1  # 0 = B2B, 1 = one day off, etc.
        rest = max(0, min(rest, 4))
        mult = REST_MULTIPLIERS.get(rest, 1.02)
        return float(mult), int(rest)
    except Exception:
        return 1.00, 2

# ──────────────────────────────────────────────
# HOME / AWAY SPLIT  ← NEW
# ──────────────────────────────────────────────
def compute_home_away_factor(game_log_df: pd.DataFrame, market: str, is_home: bool | None) -> float:
    """
    Compute home/away multiplier from historical game log splits.
    is_home=None → neutral (1.0).
    """
    if game_log_df is None or game_log_df.empty or is_home is None:
        return 1.00
    try:
        df = game_log_df.copy()
        # MATCHUP: "TOR vs. MIA" = home, "TOR @ MIA" = away
        df["_home"] = df["MATCHUP"].str.contains("vs", case=False, na=False)
        stat_col = STAT_FIELDS.get(market)
        if stat_col is None:
            return 1.00
        if isinstance(stat_col, tuple):
            df["_stat"] = sum(pd.to_numeric(df.get(c), errors="coerce").fillna(0) for c in stat_col)
        else:
            df["_stat"] = pd.to_numeric(df.get(stat_col), errors="coerce")
        home_avg = df[df["_home"]]["_stat"].mean()
        away_avg = df[~df["_home"]]["_stat"].mean()
        if pd.isna(home_avg) or pd.isna(away_avg) or away_avg == 0 or home_avg == 0:
            return 1.00
        ratio = (home_avg / away_avg) if is_home else (away_avg / home_avg)
        # Cap to avoid extreme splits from tiny samples
        return float(np.clip(ratio, 0.88, 1.12))
    except Exception:
        return 1.00

# ──────────────────────────────────────────────
# BAYESIAN SHRINKAGE  ← NEW
# ──────────────────────────────────────────────
def bayesian_shrink(observed_mu: float, n_obs: int, market: str, position_bucket: str) -> float:
    """
    Shrink observed mean toward positional prior.
    Weight on prior decreases as n_obs grows. 
    Prior weight = k / (k + n_obs), k=8 (prior effective sample size).
    """
    prior = POSITIONAL_PRIORS.get(position_bucket, POSITIONAL_PRIORS["Unknown"]).get(market)
    if prior is None or observed_mu is None:
        return observed_mu
    k = 8  # prior effective sample size
    w_prior = k / (k + max(n_obs, 1))
    w_obs   = 1.0 - w_prior
    return float(w_prior * prior + w_obs * observed_mu)

# ──────────────────────────────────────────────
# STAT SERIES COMPUTATION
# ──────────────────────────────────────────────
def compute_stat_from_gamelog(df: pd.DataFrame, market: str) -> pd.Series:
    f = STAT_FIELDS.get(market)
    if f is None:
        return pd.Series([], dtype=float)
    if isinstance(f, tuple):
        s = pd.Series(0.0, index=df.index)
        for col in f:
            s = s + pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
        return s
    return pd.to_numeric(df.get(f), errors="coerce")

# ──────────────────────────────────────────────
# VOLATILITY ENGINE
# ──────────────────────────────────────────────
def compute_volatility(series: pd.Series) -> tuple[float|None, str|None]:
    try:
        arr = pd.to_numeric(series, errors="coerce").dropna().values.astype(float)
    except Exception:
        return None, None
    if arr.size < 2:
        return None, None
    mean = arr.mean()
    if mean == 0:
        return None, None
    cv = float(arr.std(ddof=1) / mean)
    label = "Low" if cv < 0.15 else ("Moderate" if cv < 0.30 else "High")
    return cv, label

def volatility_penalty_factor(cv: float|None) -> float:
    if cv is None: return 0.0
    v = float(cv)
    if v <= 0.20: return 1.00
    if v <= 0.25: return 0.85
    if v <= 0.30: return 0.65
    if v <= 0.35: return 0.45
    return 0.0

def passes_volatility_gate(cv: float|None, ev_raw: float|None) -> tuple[bool, str]:
    if cv is None:
        return False, "no stat history (CV unavailable)"
    v = float(cv)
    if v > 0.35:
        return False, "CV>0.35 (too volatile)"
    if v > 0.25 and (ev_raw is None or float(ev_raw) < 0.06):
        return False, "CV>0.25 needs EV≥6%"
    return True, ""

# ──────────────────────────────────────────────
# BOOTSTRAP WITH PER-PLAYER NOISE  ← IMPROVED
# ──────────────────────────────────────────────
def bootstrap_prob_over(
    stat_series: pd.Series,
    line: float,
    n_sims: int = 8000,
    cv_override: float | None = None,
) -> tuple[float|None, float|None, float|None]:
    """
    Bootstrap next-game probability using historical outcomes.
    Noise is now SCALED by per-player CV (not a fixed lognormal).
    Returns (p_over, mu, sigma).
    """
    x = pd.to_numeric(stat_series, errors="coerce").dropna().values.astype(float)
    if x.size < 4:
        mu = float(np.nanmean(x)) if x.size else None
        sigma = float(np.nanstd(x, ddof=1)) if x.size > 1 else None
        return None, mu, sigma

    # Recency weights (most recent game = highest weight)
    if x.size >= 6:
        w = np.linspace(1.5, 0.5, x.size)
        w = w / w.sum()
    else:
        w = None

    rng = np.random.default_rng(42)
    # Primary bootstrap: resample from historical outcomes
    sims = rng.choice(x, size=int(n_sims), replace=True, p=w)

    # Per-player additive noise scaled by historical std
    cv = cv_override or (float(x.std(ddof=1) / x.mean()) if x.mean() != 0 else 0.20)
    noise_scale = max(0.05, min(cv * 0.40, 0.25))  # 40% of CV, capped
    noise = rng.normal(0, float(x.std(ddof=1) * noise_scale), int(n_sims))
    sims_noisy = np.clip(sims + noise, 0, None)

    p_over = float((sims_noisy > float(line)).mean())
    mu_w = float(np.average(x, weights=w) if w is not None else x.mean())
    sigma_w = float(np.sqrt(np.average((x - mu_w)**2, weights=w)) if w is not None else x.std(ddof=1))
    return float(np.clip(p_over, 1e-4, 1-1e-4)), mu_w, max(1e-9, sigma_w)

# ──────────────────────────────────────────────
# EMPIRICAL CORRELATION  ← NEW (replaces heuristic)
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60, show_spinner=False)
def empirical_leg_correlation(pid1: int, pid2: int, mkt1: str, mkt2: str, n_games: int = 20) -> float|None:
    """
    Compute Pearson correlation of same-game stat totals from overlapping game logs.
    Falls back to None if insufficient shared games.
    """
    try:
        gl1, _ = fetch_player_gamelog(pid1, max_games=n_games)
        gl2, _ = fetch_player_gamelog(pid2, max_games=n_games)
        if gl1.empty or gl2.empty:
            return None
        s1 = compute_stat_from_gamelog(gl1, mkt1).rename("s1")
        s2 = compute_stat_from_gamelog(gl2, mkt2).rename("s2")
        # Align by game date
        df1 = pd.concat([gl1["GAME_DATE"].reset_index(drop=True), s1.reset_index(drop=True)], axis=1)
        df2 = pd.concat([gl2["GAME_DATE"].reset_index(drop=True), s2.reset_index(drop=True)], axis=1)
        merged = df1.merge(df2, on="GAME_DATE", how="inner")
        if len(merged) < 6:
            return None
        corr = float(merged["s1"].corr(merged["s2"]))
        return float(np.clip(corr, -0.50, 0.70)) if not np.isnan(corr) else None
    except Exception:
        return None

def estimate_player_correlation(leg1: dict, leg2: dict) -> float:
    """Use empirical correlation when available; fall back to rule-based."""
    pid1 = leg1.get("player_id")
    pid2 = leg2.get("player_id")
    if pid1 and pid2:
        emp = empirical_leg_correlation(
            int(pid1), int(pid2), leg1.get("market","Points"), leg2.get("market","Points")
        )
        if emp is not None:
            return float(emp)
    # Rule-based fallback
    corr = 0.0
    if leg1.get("team") and leg2.get("team") and leg1["team"] == leg2["team"]:
        corr += 0.15
    m1, m2 = leg1.get("market"), leg2.get("market")
    if m1 == m2: corr += 0.10
    if set([m1,m2]) == {"Points","PRA"}: corr += 0.14
    if m1 in ["Rebounds","RA"] and m2 in ["Rebounds","RA"]: corr += 0.06
    if m1 in ["Assists","RA"] and m2 in ["Assists","RA"]: corr += 0.05
    ctx1, ctx2 = float(leg1.get("context_mult",1.0)), float(leg2.get("context_mult",1.0))
    if ctx1>1.03 and ctx2>1.03: corr += 0.04
    if ctx1<0.97 and ctx2<0.97: corr += 0.03
    if (ctx1>1.03 and ctx2<0.97) or (ctx1<0.97 and ctx2>1.03): corr -= 0.05
    return float(np.clip(corr, -0.25, 0.45))

# ──────────────────────────────────────────────
# LINE MOVEMENT ALERT  ← NEW
# ──────────────────────────────────────────────
def get_line_movement_signal(player_norm: str, market_key: str, current_line: float, side: str = "Over") -> dict:
    """
    Compare current line to opening line stored in session_state.
    Returns signal dict: {direction, pips, steam, fade, msg}
    """
    store_key = f"open_line_{player_norm}_{market_key}_{side}"
    opening = st.session_state.get(store_key)
    if opening is None:
        # First time seeing this — store as opening
        st.session_state[store_key] = float(current_line)
        return {"direction": "—", "pips": 0.0, "steam": False, "fade": False, "msg": "Opening line recorded"}
    delta = float(current_line) - float(opening)
    # For Over: line moving UP = books taking Over action (steam on Over = confirm)
    # For Over: line moving DOWN = books have Under action (fade signal on Over)
    is_over = "over" in str(side).lower()
    steam = (delta > 0.5 and is_over) or (delta < -0.5 and not is_over)
    fade  = (delta < -0.5 and is_over) or (delta > 0.5 and not is_over)
    msg = ""
    if abs(delta) >= 0.5:
        direction = "UP" if delta > 0 else "DOWN"
        msg = f"Line moved {direction} {abs(delta):.1f} pts from open ({opening:.1f} → {current_line:.1f})"
        if steam: msg += " ⚡ STEAM (confirms your side)"
        if fade:  msg += " ⚠️ FADE (sharps vs your side)"
    return {
        "direction": "UP" if delta > 0 else ("DOWN" if delta < 0 else "FLAT"),
        "pips": float(delta),
        "steam": steam,
        "fade": fade,
        "msg": msg,
        "opening": float(opening),
    }

# ──────────────────────────────────────────────
# REGIME CLASSIFIER
# ──────────────────────────────────────────────
def classify_regime(cv, blowout_prob, ctx_mult) -> tuple[str, float]:
    try: v = float(cv) if cv is not None else None
    except: v = None
    try: b = float(blowout_prob) if blowout_prob is not None else 0.10
    except: b = 0.10
    try: c = float(ctx_mult) if ctx_mult is not None else 1.0
    except: c = 1.0
    v_score = 0.0 if v is None else float(np.clip((v-0.15)/0.25, 0.0, 1.0))
    b_score = float(np.clip((b-0.08)/0.20, 0.0, 1.0))
    c_score = float(np.clip(abs(c-1.0)/0.20, 0.0, 1.0))
    score = float(np.clip(0.55*v_score + 0.30*b_score + 0.15*c_score, 0.0, 1.0))
    if score >= 0.65: return "Chaotic", score
    if score >= 0.40: return "Mixed", score
    return "Stable", score

# ──────────────────────────────────────────────
# MARKET PRICING
# ──────────────────────────────────────────────
def implied_prob_from_decimal(price: float|None) -> float|None:
    if price is None: return None
    try: return float(np.clip(1.0/float(price), 1e-6, 1.0-1e-6))
    except: return None

def ev_per_dollar(p_win: float|None, price: float|None) -> float|None:
    if p_win is None or price is None: return None
    try:
        p, o = float(p_win), float(price)
        if o <= 1.0: return None
        return float(p*(o-1.0) - (1.0-p))
    except: return None

def classify_edge(ev: float|None) -> str|None:
    if ev is None: return None
    e = float(ev)
    if e <= 0.0:   return "No Edge"
    if e < 0.04:   return "Lean Edge"
    if e < 0.08:   return "Solid Edge"
    return "Strong Edge"

def kelly_fraction(p: float, price: float) -> float:
    try:
        p, o = float(p), float(price)
        if o<=1.0 or p<=0 or p>=1: return 0.0
        b=o-1.0; q=1.0-p
        return float(max(0.0, (b*p-q)/b))
    except: return 0.0

def recommended_stake(bankroll, p, price_decimal, frac_kelly, cap_frac=0.05):
    try: br=float(bankroll)
    except: br=0.0
    if br<=0 or p is None or price_decimal is None: return 0.0, 0.0, "bankroll≤0"
    k = kelly_fraction(float(p), float(price_decimal))
    f = max(0.0, min(1.0, float(frac_kelly))) * k
    f = min(f, float(cap_frac))
    stake = br * f
    if stake <= 0: return 0.0, 0.0, "kelly≤0"
    return float(stake), float(f), "ok"

# ──────────────────────────────────────────────
# TEAM CONTEXT
# ──────────────────────────────────────────────
@st.cache_data(ttl=60*60*3, show_spinner=False)
def get_team_context():
    try:
        ss = get_season_string()
        adv = LeagueDashTeamStats(season=ss, measure_type_detailed="Advanced",
                                  per_mode_detailed="PerGame").get_data_frames()[0][
              ["TEAM_ID","TEAM_ABBREVIATION","PACE","REB_PCT","AST_PCT"]]
        try:
            defn = LeagueDashTeamStats(season=ss, measure_type_detailed_defense="Defense",
                                       per_mode_detailed="PerGame").get_data_frames()[0][
                   ["TEAM_ID","TEAM_ABBREVIATION","DEF_RATING"]]
            df = adv.merge(defn, on=["TEAM_ID","TEAM_ABBREVIATION"], how="left")
        except Exception:
            df = adv.copy()
            df["DEF_RATING"] = 113.0
        league_avg = {c: df[c].mean() for c in ["PACE","DEF_RATING","REB_PCT","AST_PCT"]}
        ctx = {}
        for _, r in df.iterrows():
            ctx[str(r["TEAM_ABBREVIATION"]).upper()] = {
                "PACE": float(r.get("PACE",0)),
                "DEF_RATING": float(r.get("DEF_RATING",113)),
                "REB_PCT": float(r.get("REB_PCT",0)),
                "AST_PCT": float(r.get("AST_PCT",0)),
            }
        return ctx, league_avg
    except Exception:
        return {}, {}

TEAM_CTX, LEAGUE_CTX = get_team_context()

def get_context_multiplier(opp: str|None, market: str, position: str|None) -> float:
    def _hash_fallback(o):
        base = 1.0
        bucket = get_position_bucket(position or "")
        if bucket=="Guard" and market in ["Assists","PA","RA"]: base *= 1.03
        elif bucket=="Big" and market in ["Rebounds","PR","RA"]: base *= 1.04
        if o:
            h = sum(ord(c) for c in str(o).upper())
            base *= (1 + ((h%15)-7)/200.0)
        return float(np.clip(base, 0.90, 1.10))
    if not LEAGUE_CTX or not TEAM_CTX or not opp:
        return _hash_fallback(opp)
    ok = str(opp).upper()
    if ok not in TEAM_CTX:
        return _hash_fallback(opp)
    o = TEAM_CTX[ok]
    lg = LEAGUE_CTX
    pace_f = o["PACE"] / (lg.get("PACE",100) or 1)
    def_f  = (lg.get("DEF_RATING",113) or 1) / (o["DEF_RATING"] or 1)
    reb_adj = (lg.get("REB_PCT",0.5) or 1) / (o.get("REB_PCT",0.5) or 1)
    ast_adj = (lg.get("AST_PCT",0.6) or 1) / (o.get("AST_PCT",0.6) or 1)
    bucket  = get_position_bucket(position or "")
    if bucket=="Guard":   pos_f = 0.5*ast_adj + 0.5*pace_f
    elif bucket=="Wing":  pos_f = 0.5*def_f + 0.5*pace_f
    elif bucket=="Big":   pos_f = 0.6*reb_adj + 0.4*def_f
    else:                 pos_f = pace_f
    if market=="Rebounds":  mult = 0.30*pace_f+0.25*def_f+0.30*reb_adj+0.15*pos_f
    elif market=="Assists": mult = 0.30*pace_f+0.25*def_f+0.30*ast_adj+0.15*pos_f
    elif market in ("RA","PA","PR"): mult = 0.25*pace_f+0.20*def_f+0.25*reb_adj+0.20*ast_adj+0.10*pos_f
    else:                   mult = 0.45*pace_f+0.40*def_f+0.15*pos_f
    return float(np.clip(mult, 0.80, 1.30))

def advanced_context_multiplier(player_name, market, opp, teammate_out) -> float:
    pos = get_player_position(player_name) or ""
    base = get_context_multiplier(opp, market, pos)
    if teammate_out: base *= 1.05
    return float(base)

def estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=None) -> float:
    if spread_abs is not None:
        s = abs(float(spread_abs))
        if s < 5: return 0.05
        if s < 8: return 0.10
        if s < 12: return 0.18
        if s < 16: return 0.26
        return 0.33
    if TEAM_CTX and LEAGUE_CTX and team_abbr and opp_abbr:
        tk, ok = str(team_abbr).upper(), str(opp_abbr).upper()
        if tk in TEAM_CTX and ok in TEAM_CTX:
            def_gap = abs(TEAM_CTX[ok].get("DEF_RATING",113)-TEAM_CTX[tk].get("DEF_RATING",113)) / (LEAGUE_CTX.get("DEF_RATING",113) or 1)
            idx = def_gap
            if idx<0.05: return 0.06
            if idx<0.10: return 0.10
            if idx<0.18: return 0.18
            return 0.24
    return 0.10

# ──────────────────────────────────────────────
# ODDS API
# ──────────────────────────────────────────────
def odds_api_key() -> str:
    return (st.secrets.get("ODDS_API_KEY","") if hasattr(st,"secrets") else "") or os.getenv("ODDS_API_KEY","")

def http_get_json(url, params, timeout=25):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        rem = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        st.session_state["_odds_headers_last"] = {"remaining":rem,"used":used,"ts":_now_iso()}
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try: detail = e.response.text[:2000]
        except: pass
        return None, f"HTTP {getattr(e.response,'status_code',None)}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_events(date_iso: str|None = None):
    key = odds_api_key()
    if not key: return [], "Missing ODDS_API_KEY"
    data, err = http_get_json(f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events", {"apiKey":key})
    if err or not isinstance(data, list): return [], err or "Unexpected events response"
    if date_iso:
        return [ev for ev in data if (ev.get("commence_time") or "")[:10] == date_iso], None
    return data, None

@st.cache_data(ttl=60*5, show_spinner=False)
def odds_get_event_odds(event_id: str, market_keys: tuple, regions: str = REGION_US):
    key = odds_api_key()
    if not key: return None, "Missing ODDS_API_KEY"
    data, err = http_get_json(
        f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{event_id}/odds",
        {"apiKey":key,"regions":regions,"markets":",".join(market_keys),
         "oddsFormat":"decimal","dateFormat":"iso"}
    )
    return data, err

@st.cache_data(ttl=60*60*24, show_spinner=False)
def lookup_player_id(name: str):
    if not name: return None
    nm = normalize_name(name)
    plist = nba_players.get_players()
    for p in plist:
        if normalize_name(p.get("full_name","")) == nm:
            return p.get("id")
    names = [p.get("full_name","") for p in plist]
    cand = difflib.get_close_matches(name, names, n=1, cutoff=0.75)
    if cand:
        for p in plist:
            if p.get("full_name") == cand[0]: return p.get("id")
    return None

def nba_headshot_url(pid: int|None) -> str|None:
    return f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png" if pid else None

@st.cache_data(ttl=60*60*24, show_spinner=False)
def get_team_maps():
    teams = nba_teams.get_teams()
    by_name = {}
    for t in teams:
        full = t.get("full_name","")
        by_name[normalize_name(full)] = {"abbr":t.get("abbreviation",""),"id":t.get("id"),"full_name":full}
    for a,tgt in [("la clippers","los angeles clippers"),("la lakers","los angeles lakers")]:
        if normalize_name(tgt) in by_name: by_name[normalize_name(a)] = by_name[normalize_name(tgt)]
    return by_name

def map_team_name_to_abbr(name: str) -> str|None:
    m = get_team_maps()
    rec = m.get(normalize_name(name))
    return rec["abbr"] if rec else None

@st.cache_data(ttl=60*60*24, show_spinner=False)
def team_id_to_abbr_map():
    return {int(t["id"]): t["abbreviation"] for t in nba_teams.get_teams()}

@st.cache_data(ttl=60*10, show_spinner=False)
def nba_scoreboard_games(game_date: date):
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=game_date.strftime("%m/%d/%Y"))
        df = sb.get_data_frames()[0]
        return [{"game_id":str(r.get("GAME_ID")),"home_team_id":int(r.get("HOME_TEAM_ID")),
                 "away_team_id":int(r.get("VISITOR_TEAM_ID"))} for _,r in df.iterrows()], None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

def opponent_from_team_abbr(team_abbr, game_date):
    games, _ = nba_scoreboard_games(game_date)
    tid_map = team_id_to_abbr_map()
    for g in (games or []):
        ha = tid_map.get(g["home_team_id"])
        aa = tid_map.get(g["away_team_id"])
        if ha == team_abbr: return aa, True   # (opponent, player_is_home)
        if aa == team_abbr: return ha, False
    return None, None

def _parse_player_prop_outcomes(event_odds, market_key, book_filter=None):
    if not event_odds: return [], None
    eid = event_odds.get("id"); home = event_odds.get("home_team"); away = event_odds.get("away_team")
    ct = event_odds.get("commence_time"); books = event_odds.get("bookmakers",[]) or []
    rows = []
    for b in books:
        bkey = b.get("key")
        if book_filter and book_filter not in ("consensus","all") and bkey != book_filter: continue
        for mk in b.get("markets",[]) or []:
            if mk.get("key") != market_key: continue
            for out in mk.get("outcomes",[]) or []:
                player = out.get("description") or out.get("name")
                if player and out.get("point") is not None:
                    rows.append({"player":player,"player_norm":normalize_name(player),
                                 "line":float(out.get("point")),"price":out.get("price"),
                                 "book":(bkey or ""),"side":(out.get("name") or ""),
                                 "market_key":market_key,"event_id":eid,
                                 "home_team":home,"away_team":away,"commence_time":ct})
    if book_filter == "consensus" and rows:
        df = pd.DataFrame(rows)
        df = df[pd.to_numeric(df["line"],errors="coerce").notna()].copy()
        if df.empty: return [], None
        df["w"] = df["book"].apply(lambda k: book_sharpness(str(k)))
        name_map = df.groupby("player_norm")["player"].agg(lambda x: x.value_counts().index[0]).to_dict()
        out_rows = []
        for (pn, side), sub in df.groupby(["player_norm","side"], dropna=False):
            sub = sub.copy().sort_values("line")
            if float(sub["w"].sum() or 0)>0:
                cw = sub["w"].cumsum(); cutoff=0.5*float(sub["w"].sum())
                line_med = float(sub.loc[cw>=cutoff,"line"].iloc[0])
            else:
                line_med = float(sub["line"].median())
            price_syn = None
            try:
                v = sub[pd.to_numeric(sub["price"],errors="coerce").notna()].copy()
                if not v.empty:
                    v["pf"] = pd.to_numeric(v["price"],errors="coerce").astype(float)
                    v = v[v["pf"]>1.0001]
                    if not v.empty:
                        v["pi"] = 1.0/v["pf"]; ws=float(v["w"].sum() or 0)
                        ps = float((v["pi"]*v["w"]).sum()/max(ws,1e-9))
                        ps = float(np.clip(ps,1e-6,1-1e-6)); price_syn=float(1.0/ps)
            except Exception: price_syn=None
            out_rows.append({"player":name_map.get(pn,pn),"player_norm":pn,"line":float(line_med),
                             "price":price_syn,"book":"consensus","side":side,"market_key":market_key,
                             "event_id":eid,"home_team":home,"away_team":away,"commence_time":ct})
        return out_rows, None
    return rows, None

def find_player_line_from_events(player_name, market_key, date_iso, book_choice):
    evs, err = odds_get_events(date_iso)
    if err: return None, None, err
    if not evs: return None, None, "No events for that date"
    target = normalize_name(player_name)
    for ev in evs:
        eid = ev.get("id")
        if not eid: continue
        odds, oerr = odds_get_event_odds(eid, (market_key,))
        if oerr or not odds: continue
        rows, _ = _parse_player_prop_outcomes(odds, market_key, book_filter=book_choice)
        for r in rows:
            if r.get("player_norm") == target: return float(r["line"]), r, None
        norms = [r.get("player_norm","") for r in rows]
        close = difflib.get_close_matches(target, norms, n=1, cutoff=0.88)
        if close:
            rr = next((x for x in rows if x.get("player_norm")==close[0]), None)
            if rr: return float(rr["line"]), rr, None
    return None, None, "Player/market not found in Odds API props"

def get_sportsbook_choices(date_iso):
    evs, err = odds_get_events(date_iso)
    if err or not evs: return ["consensus"], err
    for ev in evs[:6]:
        eid = ev.get("id")
        if not eid: continue
        odds, oerr = odds_get_event_odds(eid, (ODDS_MARKETS["Points"],))
        if odds and not oerr:
            books = sorted(list(dict.fromkeys(
                b.get("key") for b in odds.get("bookmakers",[]) if b.get("key"))))
            return ["consensus"] + books, None
    return ["consensus"], None

# ──────────────────────────────────────────────
# SHARP BOOK DIVERGENCE ALERT  ← NEW
# ──────────────────────────────────────────────
def sharp_divergence_alert(event_id, market_key, player_norm, side, model_side="Over") -> dict:
    """
    Compare sharp book (pinnacle/circa) line vs soft book consensus.
    If model agrees with sharp AND line moved in model's favor → strong confirm.
    If sharp disagrees → reduce confidence.
    """
    try:
        sharp_books = ["pinnacle","circa","bookmaker"]
        soft_odds, _ = odds_get_event_odds(str(event_id), (str(market_key),))
        if not soft_odds: return {}
        sharp_lines, soft_lines = [], []
        for b in soft_odds.get("bookmakers",[]) or []:
            bk = (b.get("key") or "").lower()
            for mk in b.get("markets",[]) or []:
                if mk.get("key") != market_key: continue
                for out in mk.get("outcomes",[]) or []:
                    pn = normalize_name(out.get("description") or out.get("name") or "")
                    if pn == player_norm and (out.get("name") or "").lower() == side.lower():
                        line = out.get("point")
                        if line is not None:
                            if bk in sharp_books: sharp_lines.append(float(line))
                            else: soft_lines.append(float(line))
        if not sharp_lines or not soft_lines: return {}
        sl = np.mean(sharp_lines); softl = np.mean(soft_lines)
        diff = sl - softl
        return {"sharp_line": sl, "soft_line": softl, "diff": diff,
                "confirm": abs(diff) < 0.3, "fade_model": (diff < -0.5 if "over" in model_side.lower() else diff > 0.5)}
    except Exception:
        return {}

# ──────────────────────────────────────────────
# CLV TRACKING
# ──────────────────────────────────────────────
def fetch_latest_market_for_leg(leg):
    try:
        eid = leg.get("event_id"); mk = leg.get("market_key")
        if not eid or not mk: return None, None, None, "missing event_id/market_key"
        pn = leg.get("player_norm") or normalize_name(leg.get("player",""))
        side = (leg.get("side") or "Over").strip()
        for bf in [(leg.get("book") or "consensus").strip().lower(), "consensus"]:
            odds, oerr = odds_get_event_odds(str(eid), (str(mk),))
            if oerr or not odds: return None, None, None, oerr or "fetch failed"
            rows, _ = _parse_player_prop_outcomes(odds, str(mk), book_filter=(bf if bf!="all" else None))
            m = next((r for r in rows if r.get("player_norm")==pn and str(r.get("side","")).strip()==side), None)
            if m:
                ln = safe_float(m.get("line")); pr = m.get("price")
                try: pr = float(pr) if pr is not None else None
                except: pr = None
                return ln, pr, (m.get("book") or bf), None
        return None, None, None, "player/side not found"
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}"

def apply_clv_update_to_legs(legs):
    errs, out = [], []
    for leg in legs:
        leg2 = dict(leg)
        line0 = safe_float(leg2.get("line")); price0 = safe_float(leg2.get("price_decimal"))
        line1, price1, book_used, err = fetch_latest_market_for_leg(leg2)
        leg2["close_ts"]=_now_iso(); leg2["line_close"]=line1
        leg2["price_close"]=price1; leg2["book_close"]=book_used
        side = (leg2.get("side") or "Over").strip().lower()
        if line0 is not None and line1 is not None:
            leg2["clv_line"]=float(line1-line0)
            leg2["clv_line_fav"]=bool(line1<line0 if "under" not in side else line1>line0)
        else:
            leg2["clv_line"]=None; leg2["clv_line_fav"]=None
        if price0 is not None and price1 is not None:
            leg2["clv_price"]=float(price1-price0); leg2["clv_price_fav"]=bool(price1>price0)
        else:
            leg2["clv_price"]=None; leg2["clv_price_fav"]=None
        if err: errs.append(f"{leg2.get('player')} {leg2.get('market')}: {err}")
        out.append(leg2)
    return out, errs

# ──────────────────────────────────────────────
# MAIN PROJECTION ENGINE  ← FULLY IMPROVED
# ──────────────────────────────────────────────
def compute_leg_projection(
    player_name: str, market_name: str, line: float, meta: dict|None,
    n_games: int, key_teammate_out: bool,
    bankroll: float=0.0, frac_kelly: float=0.25, max_risk_frac: float=0.05,
    market_prior_weight: float=0.65, exclude_chaotic: bool=True,
    game_date: date|None=None, is_home: bool|None=None,
) -> dict:
    errors = []
    game_date = game_date or date.today()
    player_id = lookup_player_id(player_name)
    if not player_id:
        errors.append("Could not resolve NBA player id.")
        return {"player":player_name,"market":market_name,"line":float(line),
                "proj":None,"p_over":None,"p_cal":None,"edge":None,
                "team":None,"opp":None,"headshot":None,"errors":errors,
                "player_id":None,"gate_ok":False,"gate_reason":"no player id"}

    # Fetch game log
    gldf, gl_errs = fetch_player_gamelog(player_id=player_id, max_games=max(6, n_games+5))
    if gl_errs: errors.extend([f"NBA API: {m}" for m in gl_errs])
    # Use requested N games
    gldf_n = gldf.head(n_games) if not gldf.empty else gldf

    # Stat series
    stat_series = compute_stat_from_gamelog(gldf_n, market_name) if not gldf_n.empty else pd.Series([], dtype=float)

    # Volatility
    vol_cv, vol_label = compute_volatility(stat_series)

    # Rest factor  ← NEW
    rest_mult, rest_days = compute_rest_factor(gldf, game_date)

    # Home/away — resolve from matchup
    team_abbr, opp_abbr, is_home_resolved = None, None, is_home
    if not gldf_n.empty:
        try:
            matchup = str(gldf_n.iloc[0].get("MATCHUP","")).strip()
            if " vs " in matchup.lower().replace("vs.","vs"):
                pts = re.split(r'\s+vs\.?\s+', matchup, flags=re.IGNORECASE)
                if len(pts)==2: team_abbr, opp_abbr = pts[0].strip(), pts[1].strip()
            elif " @ " in matchup:
                pts = matchup.split(" @ ")
                if len(pts)==2: team_abbr, opp_abbr = pts[0].strip(), pts[1].strip()
        except Exception: pass

    # Resolve opponent from scoreboard if needed
    if team_abbr and not opp_abbr:
        try:
            opp_abbr, is_home_resolved = opponent_from_team_abbr(team_abbr, game_date)
        except Exception: pass

    # Override opp from meta if available
    if meta:
        try:
            home_abbr = map_team_name_to_abbr(meta.get("home_team","") or "")
            away_abbr = map_team_name_to_abbr(meta.get("away_team","") or "")
            if team_abbr and home_abbr and away_abbr:
                if team_abbr == home_abbr: opp_abbr, is_home_resolved = away_abbr, True
                elif team_abbr == away_abbr: opp_abbr, is_home_resolved = home_abbr, False
        except Exception: pass

    # Home/away factor  ← NEW
    ha_mult = compute_home_away_factor(gldf_n, market_name, is_home_resolved)

    # Context multiplier (opponent pace/defense)
    ctx_mult = advanced_context_multiplier(player_name, market_name, opp_abbr, key_teammate_out)

    # Blowout risk
    blowout_prob = estimate_blowout_risk(team_abbr, opp_abbr, spread_abs=None)

    # Bootstrap probability
    p_over_raw, mu_raw, sigma = bootstrap_prob_over(stat_series, float(line), cv_override=vol_cv)
    if p_over_raw is None:
        errors.append(f"Insufficient history (need ≥4 games, have {len(stat_series.dropna())})")

    # Positional priors & Bayesian shrinkage  ← NEW
    pos_str = get_player_position(player_name) or ""
    pos_bucket = get_position_bucket(pos_str)
    n_valid = int(stat_series.dropna().count())
    mu_shrunk = bayesian_shrink(mu_raw, n_valid, market_name, pos_bucket) if mu_raw is not None else None

    # Composite projection = shrunk_mean × ctx × rest × home_away
    proj = mu_shrunk * ctx_mult * rest_mult * ha_mult if mu_shrunk is not None else None

    # Regime
    regime_label, regime_score = classify_regime(vol_cv, blowout_prob, ctx_mult)

    # Market price
    price_decimal = None
    try:
        if meta and meta.get("price") is not None:
            price_decimal = float(meta.get("price"))
    except Exception: pass

    p_implied = implied_prob_from_decimal(price_decimal)

    # Blend model with market prior (sharp-book-aware)
    p_model = p_over_raw
    sharp = book_sharpness(meta.get("book") if meta else None)
    w_model = float(market_prior_weight)
    w_eff = float(np.clip(w_model*(1.0-0.60*sharp)+0.15, 0.10, 0.95))
    if p_model is not None and p_implied is not None:
        p_raw = float(np.clip(w_eff*p_model + (1.0-w_eff)*p_implied, 1e-4, 1-1e-4))
    else:
        p_raw = p_model
    p_cal = p_raw  # calibration applied later in main thread

    # EV and gates
    ev_raw = ev_per_dollar(p_cal, price_decimal) if (p_cal is not None and price_decimal is not None) else None
    pen = volatility_penalty_factor(vol_cv)
    ev_adj = float(ev_raw * pen) if ev_raw is not None else None
    gate_ok, gate_reason = passes_volatility_gate(vol_cv, ev_raw)
    if exclude_chaotic and regime_label=="Chaotic":
        gate_ok, gate_reason = False, "chaotic regime (high volatility + blowout risk)"
    if not gate_ok: ev_adj = None

    # Stake
    stake_dollars, stake_frac, stake_reason = 0.0, 0.0, "gated"
    if gate_ok and p_cal is not None and price_decimal is not None and ev_adj is not None and ev_adj > 0:
        stake_dollars, stake_frac, stake_reason = recommended_stake(
            bankroll, float(p_cal), float(price_decimal), frac_kelly, max_risk_frac)

    # Line movement signal
    mk_key = meta.get("market_key") if meta else ODDS_MARKETS.get(market_name,"")
    side_str = (meta.get("side") if meta else "Over") or "Over"
    player_norm = normalize_name(player_name)
    mv_signal = get_line_movement_signal(player_norm, str(mk_key), float(line), side_str)

    # Sharp divergence (non-blocking)
    sharp_div = {}
    if meta and meta.get("event_id"):
        try:
            sharp_div = sharp_divergence_alert(meta["event_id"], mk_key, player_norm, side_str, side_str) or {}
        except Exception: sharp_div = {}

    return {
        "player":           player_name,
        "player_norm":      player_norm,
        "player_id":        player_id,
        "market":           market_name,
        "line":             float(line),
        "proj":             float(proj) if proj is not None else None,
        "proj_vs_line":     float(proj - line) if proj is not None else None,
        "p_over":           float(p_raw) if p_raw is not None else None,
        "p_raw":            float(p_raw) if p_raw is not None else None,
        "p_model":          float(p_model) if p_model is not None else None,
        "p_cal":            float(p_cal) if p_cal is not None else None,
        "p_implied":        float(p_implied) if p_implied is not None else None,
        "advantage":        float(p_cal - p_implied) if (p_cal and p_implied) else None,
        "price_decimal":    float(price_decimal) if price_decimal is not None else None,
        "book":             meta.get("book") if meta else None,
        "event_id":         meta.get("event_id") if meta else None,
        "market_key":       meta.get("market_key") if meta else None,
        "side":             side_str,
        "commence_time":    meta.get("commence_time") if meta else None,
        "regime":           regime_label,
        "regime_score":     float(regime_score),
        "ev_raw":           float(ev_raw) if ev_raw is not None else None,
        "ev_adj":           float(ev_adj) if ev_adj is not None else None,
        "ev_pct":           float(ev_adj*100) if ev_adj is not None else None,
        "stake":            float(stake_dollars),
        "stake_frac":       float(stake_frac),
        "stake_reason":     stake_reason,
        "vol_penalty":      float(pen),
        "gate_ok":          bool(gate_ok),
        "gate_reason":      gate_reason,
        "edge":             float(ev_adj) if ev_adj is not None else None,
        "edge_cat":         classify_edge(ev_adj),
        "team":             team_abbr,
        "opp":              opp_abbr,
        "is_home":          is_home_resolved,
        "headshot":         nba_headshot_url(player_id),
        "blowout_prob":     float(blowout_prob),
        "context_mult":     float(ctx_mult),
        "rest_mult":        float(rest_mult),
        "rest_days":        int(rest_days),
        "ha_mult":          float(ha_mult),
        "volatility_cv":    vol_cv,
        "volatility_label": vol_label,
        "position":         pos_str,
        "position_bucket":  pos_bucket,
        "n_games_used":     n_valid,
        "mu_raw":           float(mu_raw) if mu_raw is not None else None,
        "mu_shrunk":        float(mu_shrunk) if mu_shrunk is not None else None,
        "sigma":            float(sigma) if sigma is not None else None,
        "line_movement":    mv_signal,
        "sharp_div":        sharp_div,
        "errors":           errors,
    }

# ──────────────────────────────────────────────
# CALIBRATION ENGINE (isotonic-style)
# ──────────────────────────────────────────────
def _expand_history_legs(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df is None or history_df.empty: return pd.DataFrame()
    rows = []
    for _, r in history_df.iterrows():
        res = str(r.get("result","Pending"))
        if res not in ("HIT","MISS","PUSH"): continue
        try:
            legs = json.loads(r.get("legs","[]")) if isinstance(r.get("legs"),str) else (r.get("legs") or [])
        except: legs = []
        for leg in legs:
            if not isinstance(leg,dict): continue
            rows.append({
                "ts":r.get("ts"),"market":leg.get("market"),"player":leg.get("player"),
                "p_raw":safe_float(leg.get("p_raw") or leg.get("p_over"), default=np.nan),
                "price_decimal":safe_float(leg.get("price_decimal"), default=np.nan),
                "cv":safe_float(leg.get("volatility_cv"), default=np.nan),
                "ev_adj":safe_float(leg.get("ev_adj"), default=np.nan),
                "y":1 if res=="HIT" else 0,"result":res,
                "clv_line_fav":leg.get("clv_line_fav"),
                "clv_price_fav":leg.get("clv_price_fav"),
            })
    df = pd.DataFrame(rows)
    return df[pd.to_numeric(df["p_raw"],errors="coerce").notna()].copy() if not df.empty else df

def fit_monotone_calibrator(df_legs: pd.DataFrame, n_bins: int=12) -> dict|None:
    if df_legs is None or df_legs.empty: return None
    d = df_legs.copy()
    d = d[(d["p_raw"]>=0.01)&(d["p_raw"]<=0.99)]
    if len(d) < 80: return None
    d["bin"] = pd.cut(d["p_raw"], bins=n_bins, labels=False, include_lowest=True)
    g = d.groupby("bin",dropna=True).agg(p_mid=("p_raw","mean"),win=("y","mean"),n=("y","size")).reset_index()
    g = g[g["n"]>=5].sort_values("p_mid")
    if g.empty or len(g)<4: return None
    win_mono = np.maximum.accumulate(g["win"].values.astype(float))
    win_mono = np.clip(win_mono, 0.01, 0.99)
    return {"x":g["p_mid"].values.astype(float).tolist(),"y":win_mono.tolist(),"n":int(len(d))}

def apply_calibrator(p_raw: float|None, calib: dict|None) -> float|None:
    if p_raw is None: return None
    try: p = float(p_raw)
    except: return None
    if calib is None: return float(np.clip(p, 0.0, 1.0))
    xs = calib.get("x") or []; ys = calib.get("y") or []
    if len(xs)<2 or len(xs)!=len(ys): return float(np.clip(p, 0.0, 1.0))
    try: return float(np.clip(np.interp(p, xs, ys), 0.0, 1.0))
    except: return float(np.clip(p, 0.0, 1.0))

def recompute_pricing_fields(leg: dict, calib: dict|None) -> dict:
    p_raw = leg.get("p_raw")
    p_cal = apply_calibrator(p_raw, calib)
    leg["p_cal"] = p_cal
    price = leg.get("price_decimal")
    p_imp = implied_prob_from_decimal(price) if price is not None else None
    leg["p_implied"] = p_imp
    leg["advantage"] = float(p_cal-p_imp) if (p_cal and p_imp) else None
    ev_raw = ev_per_dollar(p_cal, price) if (p_cal and price) else None
    leg["ev_raw"] = ev_raw
    pen = volatility_penalty_factor(leg.get("volatility_cv"))
    leg["vol_penalty"] = pen
    gate_ok, gate_reason = passes_volatility_gate(leg.get("volatility_cv"), ev_raw)
    regime_label, regime_score = classify_regime(leg.get("volatility_cv"),leg.get("blowout_prob"),leg.get("context_mult"))
    leg["regime"]=regime_label; leg["regime_score"]=float(regime_score)
    if bool(st.session_state.get("exclude_chaotic",True)) and regime_label=="Chaotic":
        gate_ok, gate_reason = False, "chaotic regime"
    leg["gate_ok"]=gate_ok; leg["gate_reason"]=gate_reason
    leg["ev_adj"] = (ev_raw*pen) if (ev_raw is not None and gate_ok) else None
    leg["ev_pct"] = float(leg["ev_adj"]*100) if leg["ev_adj"] is not None else None
    leg["edge"] = leg["ev_adj"]
    leg["edge_cat"] = classify_edge(leg["ev_adj"])
    bankroll = float(st.session_state.get("bankroll",0.0) or 0)
    frac_k = float(st.session_state.get("frac_kelly",0.25) or 0.25)
    cap_frac = float(st.session_state.get("max_risk_per_bet",5.0) or 5.0)/100.0
    if gate_ok and p_cal and price and leg.get("ev_adj") and float(leg["ev_adj"])>0 and bankroll>0:
        sd, sf, sr = recommended_stake(bankroll, float(p_cal), float(price), frac_k, cap_frac)
        leg["stake"]=float(sd); leg["stake_frac"]=float(sf); leg["stake_reason"]=sr
    else:
        leg["stake"]=float(leg.get("stake",0) or 0)
        leg["stake_frac"]=float(leg.get("stake_frac",0) or 0)
        leg["stake_reason"]=leg.get("stake_reason") or "gated"
    return leg

# ──────────────────────────────────────────────
# HISTORY PERSISTENCE
# ──────────────────────────────────────────────
def user_state_path(uid): return f"user_state_{re.sub(r'[^a-zA-Z0-9_-]','_',uid or 'default')}.json"
def history_path(uid):    return f"history_{re.sub(r'[^a-zA-Z0-9_-]','_',uid or 'default')}.csv"

def load_user_state(uid):
    fp = user_state_path(uid)
    try:
        if os.path.exists(fp):
            with open(fp) as f: return json.load(f) or {}
    except Exception: pass
    return {}

def save_user_state(uid, state):
    try:
        with open(user_state_path(uid),"w") as f: json.dump(state or {}, f)
    except Exception: pass

def load_history(uid):
    fp = history_path(uid)
    try:
        if os.path.exists(fp): return pd.read_csv(fp)
    except Exception: pass
    return pd.DataFrame()

def append_history(uid, row):
    df = load_history(uid)
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(history_path(uid), index=False)

# ============================================================
# STREAMLIT UI  —  PREMIUM QUANT TERMINAL
# ============================================================
st.set_page_config(
    page_title="NBA QUANT ENGINE",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── FONTS + GLOBAL STYLES ───────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;600;700&family=Fira+Code:wght@300;400;500;700&display=swap" rel="stylesheet">

<style>
/* ── ROOT PALETTE ── */
:root {
  --bg:        #070B10;
  --bg2:       #0D1117;
  --bg3:       #111820;
  --panel:     #0F1620;
  --border:    #1E2D3D;
  --green:     #00FFB2;
  --green-dim: #00C88A;
  --blue:      #00AAFF;
  --red:       #FF3358;
  --amber:     #FFB800;
  --muted:     #4A607A;
  --text:      #C8D8E8;
  --text-hi:   #EEF4FF;
  --font-head: 'Chakra Petch', monospace;
  --font-mono: 'Fira Code', monospace;
}

/* ── APP BG ── */
.stApp {
    background: var(--bg) !important;
    font-family: var(--font-mono) !important;
    color: var(--text) !important;
}
.block-container { padding-top: 1.2rem !important; max-width: 1400px !important; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; font-family: var(--font-mono) !important; }
section[data-testid="stSidebar"] .stSlider, 
section[data-testid="stSidebar"] .stSelectbox { margin-bottom: 0.5rem; }

/* ── HEADINGS ── */
h1, h2, h3 {
    font-family: var(--font-head) !important;
    color: var(--text-hi) !important;
    letter-spacing: 0.04em;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0px;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-head) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom-color: var(--green) !important;
    background: transparent !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
    font-family: var(--font-head) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s;
    border-radius: 2px !important;
}
.stButton > button:hover {
    background: var(--green) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 18px rgba(0,255,178,0.35) !important;
}

/* ── INPUTS ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-hi) !important;
    font-family: var(--font-mono) !important;
    border-radius: 2px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 1px var(--blue) !important;
}

/* ── DATAFRAMES ── */
.stDataFrame { background: var(--panel) !important; border: 1px solid var(--border) !important; }
.stDataFrame thead th { background: var(--bg3) !important; color: var(--green) !important; font-family: var(--font-head) !important; font-size: 0.68rem !important; letter-spacing: 0.06em; text-transform: uppercase; }
.stDataFrame tbody tr:hover { background: rgba(0,170,255,0.05) !important; }
.stDataFrame td { font-family: var(--font-mono) !important; font-size: 0.72rem !important; color: var(--text) !important; }

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--green) !important;
    padding: 0.8rem 1rem !important;
    border-radius: 2px !important;
}
[data-testid="stMetricLabel"] { font-family: var(--font-head) !important; font-size: 0.65rem !important; letter-spacing: 0.10em; text-transform: uppercase; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-mono) !important; font-size: 1.4rem !important; color: var(--text-hi) !important; }

/* ── CHECKBOXES / RADIO ── */
.stCheckbox label, .stRadio label { font-family: var(--font-mono) !important; font-size: 0.78rem !important; color: var(--text) !important; }

/* ── CAPTION / SMALL TEXT ── */
.stCaption, caption { color: var(--muted) !important; font-family: var(--font-mono) !important; font-size: 0.68rem !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── WARNINGS / INFO ── */
.stAlert { background: var(--panel) !important; border: 1px solid var(--border) !important; font-family: var(--font-mono) !important; font-size: 0.74rem !important; }

/* Scanline overlay */
.stApp::after {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
    pointer-events: none;
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:1.5rem;padding:0.8rem 0 1.2rem;border-bottom:1px solid #1E2D3D;margin-bottom:1.2rem;">
  <div style="font-family:'Chakra Petch',monospace;font-size:1.7rem;font-weight:700;color:#00FFB2;letter-spacing:0.06em;">
    NBA QUANT ENGINE <span style="font-size:0.75rem;color:#4A607A;vertical-align:middle;margin-left:0.5rem;">v2.0</span>
  </div>
  <div style="flex:1;height:1px;background:linear-gradient(90deg,#00FFB2,transparent);"></div>
  <div style="font-family:'Fira Code',monospace;font-size:0.65rem;color:#4A607A;text-align:right;">
    BOOTSTRAP · BAYESIAN · KELLY · LIVE ODDS
  </div>
</div>
""", unsafe_allow_html=True)

# ─── CARD HELPER ──────────────────────────────────────────────
def make_card(content_html: str, border_color: str = "#1E2D3D", glow: bool = False) -> str:
    glow_css = f"box-shadow:0 0 20px {border_color}40;" if glow else ""
    return f"""
<div style="background:#0F1620;border:1px solid {border_color};border-radius:3px;
            padding:1.1rem 1.2rem;margin-bottom:0.8rem;{glow_css}font-family:'Fira Code',monospace;">
  {content_html}
</div>"""

def color_for_edge(cat: str|None) -> str:
    if cat == "Strong Edge": return "#00FFB2"
    if cat == "Solid Edge":  return "#00AAFF"
    if cat == "Lean Edge":   return "#FFB800"
    return "#4A607A"

def prob_bar_html(p: float|None, line_pct: float = 0.50, label: str = "") -> str:
    if p is None: return "<span style='color:#4A607A;font-size:0.72rem;'>—</span>"
    pct = int(round(p * 100))
    color = "#00FFB2" if p > 0.57 else ("#FFB800" if p > 0.52 else "#FF3358")
    return f"""
<div style="margin:0.35rem 0;">
  <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#4A607A;margin-bottom:2px;">
    <span>{label}</span><span style="color:{color};font-weight:600;">{pct}%</span>
  </div>
  <div style="background:#111820;border-radius:1px;height:6px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:{color};border-radius:1px;transition:width 0.4s;"></div>
  </div>
</div>"""

def regime_badge(label: str) -> str:
    colors = {"Stable":"#00FFB2","Mixed":"#FFB800","Chaotic":"#FF3358"}
    c = colors.get(label, "#4A607A")
    return f"<span style='background:{c}18;border:1px solid {c};color:{c};padding:1px 7px;border-radius:1px;font-size:0.60rem;letter-spacing:0.08em;font-family:Chakra Petch,monospace;'>{label.upper()}</span>"

def mv_badge(mv: dict) -> str:
    if not mv or abs(mv.get("pips",0)) < 0.25: return ""
    d = mv.get("direction","")
    pips = mv.get("pips",0)
    if mv.get("steam"): col,icon = "#00FFB2","⚡"
    elif mv.get("fade"): col,icon = "#FF3358","⚠️"
    else: col,icon = "#FFB800","→"
    arrow = "▲" if pips > 0 else "▼"
    return f"<span style='color:{col};font-size:0.65rem;'>{icon} LINE {arrow} {abs(pips):.1f}</span>"

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;
    color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;
    border-bottom:1px solid #1E2D3D;padding-bottom:0.5rem;'>
    ◈ CONTROL PANEL</div>""", unsafe_allow_html=True)

    user_id = st.text_input("Personal ID", value=st.session_state.get("user_id","trader"))
    st.session_state["user_id"] = user_id

    _active = st.session_state.get("_active_user_id")
    if _active != user_id:
        state = load_user_state(user_id)
        st.session_state["bankroll"] = safe_float(state.get("bankroll"), default=st.session_state.get("bankroll",1000.0))
        st.session_state["_active_user_id"] = user_id

    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=float(st.session_state.get("bankroll",1000.0)), step=50.0)
    st.session_state["bankroll"] = float(bankroll)
    _lb = st.session_state.get("_last_saved_bankroll")
    if _lb is None or float(_lb) != float(bankroll):
        state = load_user_state(user_id); state["bankroll"]=float(bankroll)
        save_user_state(user_id, state); st.session_state["_last_saved_bankroll"]=float(bankroll)

    st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)

    payout_multi = st.number_input("Multi-Leg Payout (×)", min_value=1.0, value=float(st.session_state.get("payout_multi",3.0)), step=0.1)
    st.session_state["payout_multi"] = payout_multi

    frac_kelly = st.slider("Fractional Kelly", 0.0, 1.0, float(st.session_state.get("frac_kelly",0.25)), 0.05)
    st.session_state["frac_kelly"] = frac_kelly

    market_prior_weight = st.slider("Model Weight (vs Market)", 0.0, 1.0, float(st.session_state.get("market_prior_weight",0.65)), 0.05,
                                    help="1.0 = pure model; 0.0 = pure market implied prob")
    st.session_state["market_prior_weight"] = float(market_prior_weight)

    max_risk_per_bet = st.slider("Max Bet Size (% BR)", 0.0, 10.0, float(st.session_state.get("max_risk_per_bet",3.0)), 0.5)
    st.session_state["max_risk_per_bet"] = float(max_risk_per_bet)

    n_games = st.slider("Sample Window (games)", 5, 30, int(st.session_state.get("n_games",10)))
    st.session_state["n_games"] = n_games

    st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)

    exclude_chaotic = st.checkbox("Block Chaotic Regime", value=bool(st.session_state.get("exclude_chaotic",True)),
                                  help="Filters high-CV / blowout-risk environments (improves hit rate)")
    st.session_state["exclude_chaotic"] = bool(exclude_chaotic)

    max_daily_loss  = st.slider("Daily Loss Stop (%)", 0, 50, int(st.session_state.get("max_daily_loss",15)))
    max_weekly_loss = st.slider("Weekly Loss Stop (%)", 0, 50, int(st.session_state.get("max_weekly_loss",25)))
    st.session_state["max_daily_loss"]  = max_daily_loss
    st.session_state["max_weekly_loss"] = max_weekly_loss

    with st.expander("Odds API", expanded=False):
        scan_book_override = st.text_input("Book override (blank=auto)", value="")
        max_req_day = st.number_input("Max requests/day", 1, 500, int(st.session_state.get("max_req_day",100)), 10)
        st.session_state["max_req_day"] = int(max_req_day)

    hdr = st.session_state.get("_odds_headers_last",{})
    if hdr:
        st.caption(f"API: used {hdr.get('used','?')} | rem {hdr.get('remaining','?')}")

# ─── SESSION STATE INIT ────────────────────────────────────────
for k in ["last_results","calibrator_map","scanner_offers"]:
    if k not in st.session_state: st.session_state[k] = None if k != "last_results" else []

MARKET_OPTIONS = list(ODDS_MARKETS.keys())

# ─── DAILY LOSS GUARD ──────────────────────────────────────────
def _daily_pnl(uid):
    h = load_history(uid)
    if h.empty: return 0.0
    try:
        h["ts_d"] = pd.to_datetime(h["ts"],errors="coerce").dt.date
        today_rows = h[h["ts_d"]==date.today()].copy()
        if today_rows.empty: return 0.0
        hits = (today_rows["result"]=="HIT").sum(); miss = (today_rows["result"]=="MISS").sum()
        return float(hits - miss)
    except: return 0.0

def _check_loss_stops(uid, bankroll):
    pnl = _daily_pnl(uid)
    if bankroll > 0 and pnl < 0 and abs(pnl)/bankroll*100 > float(st.session_state.get("max_daily_loss",15)):
        st.error(f"⛔ DAILY LOSS STOP HIT ({abs(pnl)/bankroll*100:.1f}%). No new bets recommended today.")
        return True
    return False

# ─── TABS ─────────────────────────────────────────────────────
tabs = st.tabs(["⬛ MODEL", "📊 RESULTS", "⚡ LIVE SCANNER", "🗂 HISTORY", "🧬 CALIBRATION"])

with tabs[0]:
    _check_loss_stops(user_id, bankroll)

    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;
    color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>
    ▸ CONFIGURE UP TO 4 LEGS — AUTO-FETCH OR MANUAL LINE OVERRIDE</div>""", unsafe_allow_html=True)

    date_col, book_col = st.columns([2,2])
    with date_col:
        scan_date = st.date_input("Lines Date", value=date.today(), key="model_date")
    with book_col:
        book_choices, book_err = get_sportsbook_choices(scan_date.isoformat())
        if book_err: st.caption(book_err)
        sportsbook = st.selectbox("Sportsbook", options=book_choices, index=0)

    # ── 4-leg grid ──
    leg_configs = []
    for row_idx in range(2):
        cols = st.columns(2)
        for col_idx in range(2):
            leg_n = row_idx * 2 + col_idx + 1
            tag = f"P{leg_n}"
            with cols[col_idx]:
                st.markdown(f"""<div style='font-family:Chakra Petch,monospace;font-size:0.62rem;
                color:#00FFB2;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem;'>
                ◈ LEG {leg_n}</div>""", unsafe_allow_html=True)
                pname  = st.text_input(f"Player", key=f"pname_{leg_n}", placeholder="e.g. LeBron James")
                mkt    = st.selectbox(f"Market", options=MARKET_OPTIONS, key=f"mkt_{leg_n}")
                manual = st.checkbox(f"Manual line", key=f"manual_{leg_n}")
                mline  = st.number_input(f"Line", min_value=0.0, value=float(st.session_state.get(f"line_{leg_n}",22.5)), step=0.5, key=f"mline_{leg_n}")
                out_cb = st.checkbox(f"Key teammate OUT?", key=f"out_{leg_n}")
                leg_configs.append((tag, pname, mkt, manual, mline, out_cb))

    run_btn = st.button("⚡ RUN MODEL", use_container_width=True)

    if run_btn:
        results = []; warnings = []; tasks = []
        for (tag, pname, mkt, manual, mline, teammate_out) in leg_configs:
            pname = (pname or "").strip()
            if not pname: continue
            market_key = ODDS_MARKETS.get(mkt)
            if not market_key: warnings.append(f"{tag}: unsupported market {mkt}"); continue
            line = mline; meta = None
            if not manual:
                val, m_meta, ferr = find_player_line_from_events(pname, market_key, scan_date.isoformat(), sportsbook)
                if val is not None:
                    line = float(val); meta = m_meta
                    st.success(f"✓ {tag} — {pname} {mkt}: line {line:.1f} ({sportsbook})")
                else:
                    st.warning(f"{tag} auto-line failed ({ferr}). Using manual {line:.1f}.")
            if not line or float(line) <= 0:
                warnings.append(f"{tag}: invalid line"); continue
            tasks.append((tag, pname, mkt, float(line), meta, bool(teammate_out)))

        if tasks:
            with st.spinner("Computing projections…"):
                with ThreadPoolExecutor(max_workers=min(8,len(tasks))) as ex:
                    futs = [ex.submit(compute_leg_projection, pname, mkt, line, meta,
                                      n_games=n_games, key_teammate_out=to,
                                      bankroll=bankroll, frac_kelly=frac_kelly,
                                      max_risk_frac=float(st.session_state.get("max_risk_per_bet",3.0))/100.0,
                                      market_prior_weight=market_prior_weight,
                                      exclude_chaotic=bool(exclude_chaotic),
                                      game_date=scan_date)
                            for (tag, pname, mkt, line, meta, to) in tasks]
                    results = [f.result() for f in futs]

            calib = st.session_state.get("calibrator_map")
            results = [recompute_pricing_fields(dict(leg), calib) for leg in results]
            st.session_state["last_results"] = results
            if warnings:
                for w in warnings: st.warning(w)

    # Logging
    st.markdown("<hr style='border-color:#1E2D3D;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""<span style='font-family:Chakra Petch,monospace;font-size:0.65rem;
    color:#4A607A;letter-spacing:0.12em;'>◈ LOG THIS SLATE</span>""", unsafe_allow_html=True)
    c1, c2 = st.columns([2,1])
    with c1: placed = st.radio("Did you place?", ["No","Yes"], horizontal=True, index=0)
    with c2:
        if st.button("Confirm Log"):
            res = st.session_state.get("last_results") or []
            if not res:
                st.warning("Run model first.")
            elif placed == "Yes":
                append_history(user_id, {"ts":_now_iso(),"user_id":user_id,
                                          "legs":json.dumps(res),"n_legs":len(res),
                                          "result":"Pending","notes":""})
                st.success("Logged ✓")
            else:
                st.info("Not logged.")

with tabs[1]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;
    color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>
    ▸ PROJECTION RESULTS & EDGE ANALYSIS</div>""", unsafe_allow_html=True)

    res = st.session_state.get("last_results") or []
    if not res:
        st.markdown(make_card("<span style='color:#4A607A;font-size:0.8rem;'>Run the model to see projections.</span>"), unsafe_allow_html=True)
    else:
        # ── Summary bar ──
        n_gated = sum(1 for l in res if l.get("gate_ok"))
        n_edge  = sum(1 for l in res if (l.get("ev_adj") or 0) > 0.01)
        total_stake = sum(l.get("stake",0) for l in res)
        m1c, m2c, m3c, m4c = st.columns(4)
        m1c.metric("Legs Analyzed", len(res))
        m2c.metric("Passed Gate", n_gated)
        m3c.metric("Positive EV", n_edge)
        m4c.metric("Total Rec. Stake", f"${total_stake:.2f}")

        st.markdown("")
        cols = st.columns(min(4, len(res)))
        for i, leg in enumerate(res):
            c = cols[i % len(cols)]
            with c:
                ec = color_for_edge(leg.get("edge_cat"))
                ev_pct = leg.get("ev_pct")
                ev_str = f"{ev_pct:+.1f}%" if ev_pct is not None else "—"
                proj_disp = f"{leg['proj']:.1f}" if leg.get("proj") is not None else "—"
                p_cal_v = leg.get("p_cal") or leg.get("p_over")
                n_used = leg.get("n_games_used",0)
                rest_d = leg.get("rest_days",2)
                rest_tag = "B2B" if rest_d==0 else f"{rest_d}d rest"
                mv = leg.get("line_movement") or {}
                mv_html = mv_badge(mv)
                sharp = leg.get("sharp_div") or {}
                sharp_html = ""
                if sharp.get("fade_model"):
                    sharp_html = "<div style='color:#FF3358;font-size:0.62rem;'>⚠️ SHARP FADE</div>"
                elif sharp.get("confirm") == True:
                    sharp_html = "<div style='color:#00FFB2;font-size:0.62rem;'>✓ SHARP CONFIRM</div>"

                card_html = f"""
<div style='margin-bottom:0.5rem;'>
  {"<img src='"+leg["headshot"]+"' style='width:52px;height:38px;object-fit:cover;border-radius:2px;border:1px solid #1E2D3D;float:right;'>" if leg.get("headshot") else ""}
  <div style='font-family:Chakra Petch,monospace;font-size:0.82rem;font-weight:700;color:#EEF4FF;'>{leg["player"]}</div>
  <div style='font-size:0.65rem;color:#4A607A;letter-spacing:0.08em;'>{leg.get("team","??")} vs {leg.get("opp","??")}</div>
  <div style='clear:both;'></div>
</div>
<div style='font-size:0.70rem;color:#4A607A;margin:0.15rem 0;text-transform:uppercase;letter-spacing:0.06em;'>{leg["market"]} · {rest_tag} · {leg.get("position_bucket","?")}</div>
<div style='display:flex;justify-content:space-between;margin:0.6rem 0;'>
  <div style='text-align:center;'>
    <div style='font-size:0.60rem;color:#4A607A;'>LINE</div>
    <div style='font-family:Fira Code,monospace;font-size:1.1rem;color:#EEF4FF;font-weight:500;'>{leg["line"]:.1f}</div>
  </div>
  <div style='text-align:center;'>
    <div style='font-size:0.60rem;color:#4A607A;'>PROJ</div>
    <div style='font-family:Fira Code,monospace;font-size:1.1rem;color:#00AAFF;font-weight:500;'>{proj_disp}</div>
  </div>
  <div style='text-align:center;'>
    <div style='font-size:0.60rem;color:#4A607A;'>EV</div>
    <div style='font-family:Fira Code,monospace;font-size:1.1rem;color:{ec};font-weight:600;'>{ev_str}</div>
  </div>
</div>
{prob_bar_html(p_cal_v, label="P(OVER)")}
{prob_bar_html(leg.get("p_implied"), label="IMPLIED")}
<div style='margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center;'>
  {regime_badge(leg.get("regime","?"))}
  {mv_html}
</div>
{sharp_html}
<div style='margin-top:0.7rem;font-size:0.64rem;color:#4A607A;'>
  ctx×{leg.get("context_mult",1):.3f} · rest×{leg.get("rest_mult",1):.2f} · ha×{leg.get("ha_mult",1):.2f}<br>
  CV={f"{leg['volatility_cv']:.2f}" if leg.get("volatility_cv") else "—"} · N={n_used} games<br>
  Shrunk μ: {f"{leg['mu_shrunk']:.1f}" if leg.get("mu_shrunk") else "—"}
</div>"""

                stake = safe_float(leg.get("stake"))
                if stake > 0:
                    card_html += f"<div style='margin-top:0.6rem;background:#00FFB218;border:1px solid #00FFB230;border-radius:2px;padding:0.4rem 0.6rem;font-size:0.72rem;color:#00FFB2;font-family:Fira Code,monospace;'>REC STAKE: ${stake:.2f} ({leg.get('stake_frac',0)*100:.1f}% BR)</div>"
                elif not leg.get("gate_ok"):
                    card_html += f"<div style='margin-top:0.6rem;background:#FF335818;border:1px solid #FF335830;border-radius:2px;padding:0.4rem 0.6rem;font-size:0.65rem;color:#FF3358;'>GATED: {leg.get('gate_reason','')}</div>"

                if leg.get("errors"):
                    card_html += f"<div style='margin-top:0.4rem;font-size:0.60rem;color:#FFB800;'>" + "<br>".join(leg["errors"][:2]) + "</div>"

                st.markdown(make_card(card_html, border_color=ec, glow=(leg.get("edge_cat") in ["Strong Edge","Solid Edge"])), unsafe_allow_html=True)

        # ── Multi-leg combo ──
        if len(res) >= 2:
            st.markdown("<hr style='border-color:#1E2D3D;margin:1rem 0;'>", unsafe_allow_html=True)
            st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.72rem;
            color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.8rem;'>
            ◈ MULTI-LEG JOINT MONTE CARLO (CORRELATED)</div>""", unsafe_allow_html=True)
            try:
                from scipy.stats import norm as _norm_mc
                valid_legs = [l for l in res if l.get("gate_ok") and l.get("p_cal") is not None]
                if len(valid_legs) < 2:
                    st.caption("Need ≥2 gated legs for combo.")
                else:
                    n = len(valid_legs)
                    probs = np.array([float(l["p_cal"]) for l in valid_legs])
                    corr_mat = np.eye(n)
                    for i in range(n):
                        for j in range(i+1, n):
                            c = estimate_player_correlation(valid_legs[i], valid_legs[j])
                            corr_mat[i,j] = corr_mat[j,i] = c
                    # PSD fix
                    evals, evecs = np.linalg.eigh(corr_mat)
                    evals = np.clip(evals, 1e-6, None)
                    corr_psd = evecs @ np.diag(evals) @ evecs.T
                    # Run joint MC
                    rng2 = np.random.default_rng(99)
                    z = rng2.multivariate_normal(np.zeros(n), corr_psd, 10000)
                    u = _norm_mc.cdf(z)
                    joint = float((u < probs).all(axis=1).mean())
                    payout_mult = float(st.session_state.get("payout_multi",3.0))
                    ev_combo = payout_mult * joint - 1.0
                    naive = float(np.prod(probs))
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Joint Hit Prob (MC)", f"{joint*100:.1f}%")
                    mc2.metric("Naive (uncorr)", f"{naive*100:.1f}%")
                    mc3.metric(f"Combo EV (×{payout_mult})", f"{ev_combo*100:+.1f}%")
                    st.caption("Correlation matrix built from empirical game-log overlap where available, otherwise rule-based.")
            except ImportError:
                st.caption("scipy not available — joint MC skipped.")
            except Exception as e:
                st.caption(f"Joint MC error: {type(e).__name__}: {e}")

        with st.expander("Raw Data Table", expanded=False):
            display_cols = ["player","market","line","proj","p_cal","p_implied","advantage",
                            "ev_pct","edge_cat","gate_ok","stake","vol_label","volatility_cv",
                            "regime","rest_days","position_bucket","context_mult","n_games_used"]
            disp_df = pd.DataFrame([{k:l.get(k) for k in display_cols} for l in res])
            st.dataframe(disp_df, use_container_width=True)

with tabs[2]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;
    color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>
    ▸ LIVE SCANNER — SWEEP ALL PLAYER PROPS FOR EDGES</div>""", unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns([2,2,2])
    with sc1:
        scan_date2 = st.date_input("Scan Date", value=date.today(), key="scan_date2")
    with sc2:
        markets_sel = st.multiselect("Markets", options=MARKET_OPTIONS, default=["Points","Rebounds","Assists"])
    with sc3:
        book_choices2, _ = get_sportsbook_choices(scan_date2.isoformat())
        sportsbook2 = st.selectbox("Book", options=["all"]+book_choices2, index=0)

    sf1, sf2, sf3 = st.columns(3)
    with sf1: min_prob = st.slider("Min P(Over)", 0.50, 0.80, 0.57, 0.01)
    with sf2: min_adv  = st.slider("Min Advantage vs Implied", 0.00, 0.12, 0.02, 0.005)
    with sf3: min_ev   = st.slider("Min EV (adj)", -0.05, 0.25, 0.01, 0.005)

    max_rows = st.slider("Max Results", 10, 200, 60, 10)

    fetch_col, scan_col = st.columns(2)
    if fetch_col.button("Fetch Live Lines", use_container_width=True):
        selected_keys = list(dict.fromkeys(ODDS_MARKETS.get(m) for m in markets_sel if ODDS_MARKETS.get(m)))
        if not selected_keys:
            st.warning("Select at least one market.")
        else:
            evs, err = odds_get_events(scan_date2.isoformat())
            if err: st.error(err)
            elif not evs: st.warning("No events for that date.")
            else:
                offers = []
                for ev in evs:
                    eid = ev.get("id")
                    if not eid: continue
                    odds, oerr = odds_get_event_odds(eid, tuple(selected_keys))
                    if oerr or not odds: continue
                    for m in markets_sel:
                        mk = ODDS_MARKETS.get(m)
                        if not mk: continue
                        bf = sportsbook2 if sportsbook2 != "all" else None
                        parsed, _ = _parse_player_prop_outcomes(odds, mk, book_filter=bf)
                        offers.extend([{**r,"market":m} for r in parsed])
                if offers:
                    st.session_state["scanner_offers"] = pd.DataFrame(offers)
                    st.success(f"Fetched {len(offers)} raw prop outcomes.")
                else:
                    st.warning("No offers returned — check plan includes NBA player props.")

    if scan_col.button("Run Scan ⚡", use_container_width=True):
        df = st.session_state.get("scanner_offers")
        if df is None or df.empty:
            st.warning("Fetch lines first.")
        else:
            df2 = df[df["side"].str.lower().isin(["over","o","over "])].copy()
            if df2.empty: df2 = df.copy()
            candidates = []
            for _, r in df2.iterrows():
                pname = r.get("player"); mkt = r.get("market"); line = r.get("line")
                if not pname or pd.isna(line) or not mkt: continue
                meta = {"event_id":r.get("event_id"),"home_team":r.get("home_team"),
                        "away_team":r.get("away_team"),"commence_time":r.get("commence_time"),
                        "price":r.get("price"),"book":r.get("book"),
                        "market_key":ODDS_MARKETS.get(mkt),"side":r.get("side","Over")}
                candidates.append((pname, mkt, float(line), meta))

            out_rows, dropped = [], []
            if candidates:
                with st.spinner(f"Scanning {len(candidates)} candidates…"):
                    with ThreadPoolExecutor(max_workers=8) as ex:
                        futs = [ex.submit(compute_leg_projection, pname, mkt, line, meta,
                                          n_games=n_games, key_teammate_out=False,
                                          bankroll=bankroll, frac_kelly=frac_kelly,
                                          max_risk_frac=float(st.session_state.get("max_risk_per_bet",3.0))/100.0,
                                          market_prior_weight=market_prior_weight,
                                          exclude_chaotic=bool(exclude_chaotic),
                                          game_date=scan_date2)
                                for pname, mkt, line, meta in candidates]
                        for (pname, mkt, line, meta), fut in zip(candidates, futs):
                            leg = fut.result()
                            leg = recompute_pricing_fields(leg, st.session_state.get("calibrator_map"))
                            if not leg.get("gate_ok"):
                                dropped.append({"player":pname,"market":mkt,"reason":leg.get("gate_reason","gated")}); continue
                            pc = float(leg.get("p_cal") or leg.get("p_over") or 0)
                            pi = leg.get("p_implied")
                            ev = leg.get("ev_adj")
                            if pi is None or ev is None:
                                dropped.append({"player":pname,"market":mkt,"reason":"no price/EV"}); continue
                            adv = pc - float(pi)
                            if pc < min_prob:
                                dropped.append({"player":pname,"market":mkt,"reason":f"p_cal<{min_prob:.2f}"}); continue
                            if adv < min_adv:
                                dropped.append({"player":pname,"market":mkt,"reason":f"adv<{min_adv:.3f}"}); continue
                            if float(ev) < min_ev:
                                dropped.append({"player":pname,"market":mkt,"reason":f"ev<{min_ev:.3f}"}); continue
                            mv = leg.get("line_movement") or {}
                            out_rows.append({
                                "player":pname,"market":mkt,"line":line,
                                "p_cal":round(pc,3),"p_implied":round(float(pi),3),
                                "advantage":round(adv,3),"ev_adj_pct":round(float(ev)*100,2),
                                "proj":safe_round(leg.get("proj")),
                                "edge_cat":leg.get("edge_cat",""),"regime":leg.get("regime",""),
                                "team":leg.get("team",""),"opp":leg.get("opp",""),
                                "vol_cv":safe_round(leg.get("volatility_cv")),
                                "rest_d":leg.get("rest_days",2),
                                "line_mv":mv.get("direction","—"),
                                "mv_pips":mv.get("pips",0.0),
                                "steam": "⚡" if mv.get("steam") else ("⚠️" if mv.get("fade") else ""),
                                "stake_$":round(leg.get("stake",0),2),
                                "n_games":leg.get("n_games_used",0),
                            })

            out_df = pd.DataFrame(out_rows)
            if not out_df.empty:
                out_df = out_df.sort_values("ev_adj_pct", ascending=False).head(max_rows)
                st.markdown(f"""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;
                color:#00FFB2;letter-spacing:0.10em;margin-bottom:0.6rem;'>
                ◈ {len(out_df)} EDGES FOUND</div>""", unsafe_allow_html=True)
                st.dataframe(out_df, use_container_width=True)
            else:
                st.warning("No legs met threshold criteria.")

            with st.expander(f"Excluded ({len(dropped)})", expanded=False):
                if dropped: st.dataframe(pd.DataFrame(dropped).head(200), use_container_width=True)

with tabs[3]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;
    color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>
    ▸ BET HISTORY & CLV TRACKER</div>""", unsafe_allow_html=True)

    h = load_history(user_id)
    if h.empty:
        st.markdown(make_card("<span style='color:#4A607A;'>No bets logged yet. Log from the Model tab.</span>"), unsafe_allow_html=True)
    else:
        # P&L summary
        settled = h[h["result"].isin(["HIT","MISS","PUSH"])].copy() if not h.empty else pd.DataFrame()
        n_hit = (settled["result"]=="HIT").sum() if not settled.empty else 0
        n_miss = (settled["result"]=="MISS").sum() if not settled.empty else 0
        n_push = (settled["result"]=="PUSH").sum() if not settled.empty else 0
        n_pend = (h["result"]=="Pending").sum() if not h.empty else 0
        hit_rate = n_hit/(n_hit+n_miss) if (n_hit+n_miss)>0 else None

        hc1,hc2,hc3,hc4 = st.columns(4)
        hc1.metric("Hit Rate",  f"{hit_rate*100:.1f}%" if hit_rate else "—")
        hc2.metric("Wins",      n_hit)
        hc3.metric("Losses",    n_miss)
        hc4.metric("Pending",   n_pend)

        st.dataframe(h, use_container_width=True)

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)

        uc1, uc2 = st.columns(2)
        with uc1:
            st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#4A607A;letter-spacing:0.10em;'>UPDATE RESULT</span>", unsafe_allow_html=True)
            idx = st.number_input("Row index", 0, max(0,len(h)-1), 0, 1)
            new_res = st.selectbox("Result", ["Pending","HIT","MISS","PUSH"])
            if st.button("Update Result"):
                h2 = h.copy(); h2.loc[int(idx),"result"] = new_res
                h2.to_csv(history_path(user_id), index=False)
                st.success("Updated ✓")

        with uc2:
            st.markdown("<span style='font-family:Chakra Petch,monospace;font-size:0.62rem;color:#4A607A;letter-spacing:0.10em;'>CLV UPDATE</span>", unsafe_allow_html=True)
            idx2 = st.number_input("CLV Row index", 0, max(0,len(h)-1), 0, 1, key="clv_idx")
            if st.button("Fetch & Update CLV"):
                try:
                    h2 = h.copy()
                    legs = json.loads(h2.loc[int(idx2),"legs"])
                    if not isinstance(legs,list) or not legs:
                        st.warning("No legs on that row.")
                    else:
                        legs2, errs = apply_clv_update_to_legs(legs)
                        h2.loc[int(idx2),"legs"] = json.dumps(legs2)
                        h2.to_csv(history_path(user_id), index=False)
                        for e in errs[:5]: st.warning(e)
                        st.success("CLV updated ✓")
                except Exception as e:
                    st.error(f"CLV update failed: {e}")

with tabs[4]:
    st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.68rem;
    color:#4A607A;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:1rem;'>
    ▸ CALIBRATION ENGINE — MONOTONE ISOTONIC CALIBRATION + MARKET-LEVEL ROI</div>""", unsafe_allow_html=True)

    h = load_history(user_id)
    legs_df = _expand_history_legs(h)

    if legs_df.empty:
        st.markdown(make_card("""<span style='color:#4A607A;font-size:0.78rem;'>
        No settled bets yet. Log bets and mark results to enable calibration.<br>
        <span style='font-size:0.65rem;'>Minimum ~80 settled legs needed to fit a reliable calibrator.</span>
        </span>"""), unsafe_allow_html=True)
    else:
        y = legs_df["y"].values.astype(float)
        p_raw = legs_df["p_raw"].values.astype(float)
        brier = float(np.mean((p_raw - y)**2))
        hit_rate_cal = float(y.mean())
        n_settled = len(legs_df)

        cc1,cc2,cc3,cc4 = st.columns(4)
        cc1.metric("Settled Legs", n_settled)
        cc2.metric("Actual Hit Rate", f"{hit_rate_cal*100:.1f}%")
        cc3.metric("Brier Score (raw)", f"{brier:.4f}")
        cc4.metric("Calibrator Fitted", "✓" if st.session_state.get("calibrator_map") else "✗")

        # CLV metrics
        if "clv_line_fav" in legs_df.columns and legs_df["clv_line_fav"].notna().any():
            clv_line_rate = float(legs_df["clv_line_fav"].dropna().astype(int).mean())
            st.metric("CLV (line) favorable %", f"{clv_line_rate*100:.1f}%",
                      delta="Edge exists" if clv_line_rate > 0.52 else "No edge vs closing line",
                      delta_color="normal" if clv_line_rate > 0.52 else "inverse")

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)

        # ── Market-level ROI  ← NEW ──
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;
        color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>
        ◈ ROI BY MARKET</div>""", unsafe_allow_html=True)
        if "market" in legs_df.columns:
            mkt_grp = legs_df.groupby("market").agg(
                bets=("y","size"), hit_rate=("y","mean"),
                avg_ev=("ev_adj","mean")
            ).reset_index()
            mkt_grp["hit_rate_pct"] = (mkt_grp["hit_rate"]*100).round(1)
            mkt_grp["avg_ev_pct"] = (mkt_grp["avg_ev"]*100).round(2)
            mkt_grp = mkt_grp.sort_values("hit_rate_pct", ascending=False)
            st.dataframe(mkt_grp, use_container_width=True)

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)

        # ── Reliability bins ──
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;
        color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>
        ◈ RELIABILITY TABLE (p_raw vs actual)</div>""", unsafe_allow_html=True)
        n_bins = st.slider("Bins", 6, 20, 10)
        legs_df["bin"] = pd.cut(legs_df["p_raw"], bins=n_bins, labels=False, include_lowest=True)
        rel = legs_df.groupby("bin",dropna=True).agg(
            p_mean=("p_raw","mean"), win_rate=("y","mean"), n=("y","size")).reset_index()
        rel["calibration_error"] = (rel["p_mean"] - rel["win_rate"]).abs().round(3)
        st.dataframe(rel, use_container_width=True)

        st.markdown("<hr style='border-color:#1E2D3D;margin:0.8rem 0;'>", unsafe_allow_html=True)

        # ── Fit calibrator ──
        st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;
        color:#00FFB2;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;'>
        ◈ FIT CALIBRATOR</div>""", unsafe_allow_html=True)
        st.caption("Monotone isotonic calibration maps p_raw → p_cal using your settled history. Applied globally to all model runs.")
        if st.button("Fit Calibrator from History", use_container_width=True):
            calib = fit_monotone_calibrator(legs_df, n_bins=int(n_bins))
            if calib is None:
                st.warning(f"Need ~80+ quality legs (currently {n_settled}). Identity calibration used.")
                st.session_state["calibrator_map"] = None
            else:
                st.session_state["calibrator_map"] = calib
                st.success(f"Calibrator fitted on {calib.get('n','?')} legs ✓")

        calib = st.session_state.get("calibrator_map")
        if calib:
            legs_df["p_cal_fit"] = legs_df["p_raw"].apply(lambda p: apply_calibrator(p, calib))
            brier_cal = float(np.mean((legs_df["p_cal_fit"].values.astype(float)-y)**2))
            st.metric("Brier Score (calibrated)", f"{brier_cal:.4f}",
                      delta=f"{(brier_cal-brier)*100:.2f}% vs raw",
                      delta_color="inverse")
            legs_df["bin2"] = pd.cut(legs_df["p_cal_fit"], bins=n_bins, labels=False, include_lowest=True)
            rel2 = legs_df.groupby("bin2",dropna=True).agg(
                p_mean=("p_cal_fit","mean"),win_rate=("y","mean"),n=("y","size")).reset_index()
            st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;
            color:#00AAFF;letter-spacing:0.12em;text-transform:uppercase;margin:0.6rem 0;'>
            ◈ POST-CALIBRATION RELIABILITY</div>""", unsafe_allow_html=True)
            st.dataframe(rel2, use_container_width=True)

            st.markdown("<hr style='border-color:#1E2D3D;margin:0.6rem 0;'>", unsafe_allow_html=True)
            st.markdown("""<div style='font-family:Chakra Petch,monospace;font-size:0.65rem;
            color:#4A607A;'>POLICY AUDIT</div>""", unsafe_allow_html=True)
            if hit_rate_cal < 0.48:
                st.error("⛔ Hit rate below 48% — cut volume, tighten EV threshold, review market selection.")
            elif hit_rate_cal > 0.58:
                st.success("✓ Strong hit rate — consider increasing Kelly fraction gradually.")
            else:
                st.info("Moderate hit rate. Continue collecting data, focus on CLV tracking.")
            if brier_cal > brier:
                st.warning("Calibrator is WORSENING Brier score — needs more data. Reset to identity.")

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:2rem;padding-top:0.8rem;border-top:1px solid #1E2D3D;
font-family:Fira Code,monospace;font-size:0.60rem;color:#2A3A4A;
display:flex;justify-content:space-between;'>
  <span>NBA QUANT ENGINE v2.0</span>
  <span>BOOTSTRAP · BAYESIAN SHRINKAGE · ISOTONIC CALIBRATION · EMPIRICAL CORRELATION · LINE MOVEMENT · REGIME FILTER</span>
  <span>⚡ Powered by Kamal</span>
</div>
""", unsafe_allow_html=True)
