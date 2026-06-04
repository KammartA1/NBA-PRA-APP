"""
core/mlb_projections.py — MLB projection engine for background workers.
No Streamlit dependency. Uses MLB-StatsAPI for player game logs.

Same interface as core/projections.py (NBA) so the scanner, grader,
and bet_grader can call the same four functions per sport.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

# ── PrizePicks stat_type → internal code ──────────────────────────
MLB_MARKET_MAP = {
    "Pitcher Strikeouts": "K",
    "Pitching Outs": "OUTS",
    "Earned Runs": "ER",
    "Hits Allowed": "HA",
    "Walks Allowed": "BB_A",
    "Total Bases": "TB",
    "Hits": "H",
    "Runs": "R",
    "RBIs": "RBI",
    "Stolen Bases": "SB",
    "Home Runs": "HR",
    "Hits+Runs+RBIs": "HRR",
    "Fantasy Score": "MLB_FS",
    "Walks": "BB",
    "Singles": "1B",
    "Doubles": "2B",
    "Triples": "3B",
    # Pass-through codes
    "K": "K", "OUTS": "OUTS", "ER": "ER", "HA": "HA", "BB_A": "BB_A",
    "TB": "TB", "H": "H", "R": "R", "RBI": "RBI", "SB": "SB", "HR": "HR",
    "HRR": "HRR", "MLB_FS": "MLB_FS", "BB": "BB",
}

PITCHER_MARKETS = {"K", "OUTS", "ER", "HA", "BB_A", "MLB_FS"}
BATTER_MARKETS = {"TB", "H", "R", "RBI", "SB", "HR", "HRR", "MLB_FS", "BB", "1B", "2B", "3B"}

# ── Mapping internal code → API field(s) ─────────────────────────

_BATTER_STAT_MAP = {
    "TB": "totalBases",
    "H": "hits",
    "R": "runs",
    "RBI": "rbi",
    "SB": "stolenBases",
    "HR": "homeRuns",
    "BB": "baseOnBalls",
    "2B": "doubles",
    "3B": "triples",
}

_PITCHER_STAT_MAP = {
    "K": "strikeOuts",
    "OUTS": "outs",
    "ER": "earnedRuns",
    "HA": "hits",
    "BB_A": "baseOnBalls",
}


def _detect_player_type(player_id: int) -> str:
    """Return 'pitcher' or 'batter' based on MLB primary position."""
    try:
        import statsapi
        data = statsapi.get("people", {"personIds": player_id})
        pos = data["people"][0].get("primaryPosition", {}).get("abbreviation", "")
        return "pitcher" if pos == "P" else "batter"
    except Exception:
        return "batter"


_player_type_cache: dict[int, str] = {}


def resolve_player_id(name: str) -> Optional[int]:
    try:
        import statsapi
        results = statsapi.lookup_player(name)
        if results:
            pid = results[0]["id"]
            pos = results[0].get("primaryPosition", {}).get("abbreviation", "")
            _player_type_cache[pid] = "pitcher" if pos == "P" else "batter"
            return pid
        return None
    except Exception as e:
        log.warning("resolve_player_id(%s) failed: %s", name, e)
        return None


def get_player_type(player_id: int) -> str:
    if player_id in _player_type_cache:
        return _player_type_cache[player_id]
    ptype = _detect_player_type(player_id)
    _player_type_cache[player_id] = ptype
    return ptype


def fetch_player_gamelog(
    player_id: int,
    n_games: int = 20,
    player_type: str | None = None,
    season: int | None = None,
) -> Optional[pd.DataFrame]:
    """Fetch per-game stats from MLB StatsAPI. Returns a DataFrame or None."""
    try:
        import statsapi
        time.sleep(0.3)

        if player_type is None:
            player_type = get_player_type(player_id)
        group = "pitching" if player_type == "pitcher" else "hitting"

        if season is None:
            season = date.today().year

        data = statsapi.get("people", {
            "personIds": player_id,
            "hydrate": f"stats(group=[{group}],type=[gameLog],season={season})",
        })
        splits = data["people"][0].get("stats", [{}])[0].get("splits", [])
        if not splits:
            prev = season - 1
            data = statsapi.get("people", {
                "personIds": player_id,
                "hydrate": f"stats(group=[{group}],type=[gameLog],season={prev})",
            })
            splits = data["people"][0].get("stats", [{}])[0].get("splits", [])

        if not splits:
            return None

        rows = []
        for s in splits:
            stat = s.get("stat", {})
            row = {
                "GAME_DATE": s.get("date", ""),
                "OPPONENT": s.get("opponent", {}).get("name", ""),
                "TEAM": s.get("team", {}).get("name", ""),
                "PLAYER_TYPE": player_type,
            }
            for k, v in stat.items():
                row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.head(n_games) if not df.empty else None

    except Exception as e:
        log.warning("fetch_player_gamelog(%s) failed: %s", player_id, e)
        return None


def _compute_stat(df: pd.DataFrame, market: str) -> pd.Series:
    """Extract the stat series for a given market code from a gamelog DataFrame."""
    ptype = df["PLAYER_TYPE"].iloc[0] if "PLAYER_TYPE" in df.columns else "batter"

    if market in _BATTER_STAT_MAP:
        col = _BATTER_STAT_MAP[market]
        return pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    if market in _PITCHER_STAT_MAP:
        col = _PITCHER_STAT_MAP[market]
        return pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    if market == "1B":
        h = pd.to_numeric(df.get("hits", 0), errors="coerce").fillna(0)
        d = pd.to_numeric(df.get("doubles", 0), errors="coerce").fillna(0)
        t = pd.to_numeric(df.get("triples", 0), errors="coerce").fillna(0)
        hr = pd.to_numeric(df.get("homeRuns", 0), errors="coerce").fillna(0)
        return h - d - t - hr

    if market == "HRR":
        return (pd.to_numeric(df.get("hits", 0), errors="coerce").fillna(0) +
                pd.to_numeric(df.get("runs", 0), errors="coerce").fillna(0) +
                pd.to_numeric(df.get("rbi", 0), errors="coerce").fillna(0))

    if market == "MLB_FS":
        if ptype == "pitcher":
            k = pd.to_numeric(df.get("strikeOuts", 0), errors="coerce").fillna(0)
            outs = pd.to_numeric(df.get("outs", 0), errors="coerce").fillna(0)
            er = pd.to_numeric(df.get("earnedRuns", 0), errors="coerce").fillna(0)
            ip_str = df.get("inningsPitched", "0")
            ip = pd.to_numeric(ip_str, errors="coerce").fillna(0)
            w = pd.to_numeric(df.get("wins", 0), errors="coerce").fillna(0)
            qs = ((ip >= 6) & (er <= 3)).astype(float)
            return w * 6 + qs * 4 + k * 3 + outs * 1 - er * 3
        else:
            h = pd.to_numeric(df.get("hits", 0), errors="coerce").fillna(0)
            d = pd.to_numeric(df.get("doubles", 0), errors="coerce").fillna(0)
            t = pd.to_numeric(df.get("triples", 0), errors="coerce").fillna(0)
            hr = pd.to_numeric(df.get("homeRuns", 0), errors="coerce").fillna(0)
            singles = h - d - t - hr
            r = pd.to_numeric(df.get("runs", 0), errors="coerce").fillna(0)
            rbi = pd.to_numeric(df.get("rbi", 0), errors="coerce").fillna(0)
            bb = pd.to_numeric(df.get("baseOnBalls", 0), errors="coerce").fillna(0)
            hbp = pd.to_numeric(df.get("hitByPitch", 0), errors="coerce").fillna(0)
            sb = pd.to_numeric(df.get("stolenBases", 0), errors="coerce").fillna(0)
            return singles * 3 + d * 5 + t * 8 + hr * 10 + r * 2 + rbi * 2 + bb * 2 + hbp * 2 + sb * 5

    return pd.Series([0] * len(df))


def actual_stat_for_date(
    gamelog_df: pd.DataFrame,
    market: str,
    game_date,
) -> Optional[float]:
    """Return the player's ACTUAL value for `market` on `game_date`."""
    if gamelog_df is None or gamelog_df.empty or "GAME_DATE" not in gamelog_df.columns:
        return None
    try:
        dates = pd.to_datetime(gamelog_df["GAME_DATE"], errors="coerce").dt.date
    except Exception:
        return None
    mask = dates == game_date
    if not mask.any():
        return None
    row = gamelog_df[mask]
    # Doubleheaders: sum stats across both games on same date
    stat_series = _compute_stat(row, market)
    if stat_series is None or len(stat_series) == 0:
        return None
    try:
        return float(stat_series.sum())
    except Exception:
        return None


def project_player_prop(
    gamelog_df: pd.DataFrame,
    market: str,
    line: float,
) -> dict:
    """Project a player prop and return probability/EV metrics.

    Same interface as core/projections.project_player_prop for NBA.
    """
    stat_series = _compute_stat(gamelog_df, market)
    n = len(stat_series)
    if n < 3:
        return {"error": "insufficient_games", "n_games": n}

    ptype = gamelog_df["PLAYER_TYPE"].iloc[0] if "PLAYER_TYPE" in gamelog_df.columns else "batter"
    is_pitcher = ptype == "pitcher"

    # Pitchers start every 5 days → smaller recent windows
    if is_pitcher:
        l5 = stat_series.head(3)
        l10 = stat_series.head(7)
    else:
        l5 = stat_series.head(5)
        l10 = stat_series.head(10)

    full = stat_series
    l5_avg = l5.mean() if len(l5) >= 2 else full.mean()
    l10_avg = l10.mean() if len(l10) >= 4 else full.mean()
    full_avg = full.mean()

    if n >= 10:
        proj = 0.45 * l5_avg + 0.35 * l10_avg + 0.20 * full_avg
    elif n >= 5:
        proj = 0.55 * l5_avg + 0.45 * full_avg
    else:
        proj = full_avg

    std = stat_series.std(ddof=1) if n >= 4 else stat_series.std(ddof=0)
    if std < 0.3:
        std = 0.3

    p_over = norm.sf(line - 0.5, loc=proj, scale=std)
    p_under = 1 - p_over
    p_over = max(0.02, min(0.98, p_over))
    p_under = max(0.02, min(0.98, p_under))

    ev_over = (p_over - 0.50) * 100
    ev_under = (p_under - 0.50) * 100

    if p_over >= p_under:
        side = "Over"
        p_cal = p_over
        ev_pct = ev_over
    else:
        side = "Under"
        p_cal = p_under
        ev_pct = ev_under

    if ev_pct >= 12:
        edge_cat = "ELITE"
    elif ev_pct >= 8:
        edge_cat = "STRONG"
    elif ev_pct >= 5:
        edge_cat = "GOOD"
    elif ev_pct >= 2:
        edge_cat = "LEAN"
    else:
        edge_cat = "SKIP"

    return {
        "proj": round(proj, 2),
        "p_over": round(p_over, 4),
        "p_under": round(p_under, 4),
        "ev_over": round(ev_over, 2),
        "ev_under": round(ev_under, 2),
        "l5_avg": round(l5_avg, 2),
        "l10_avg": round(l10_avg, 2),
        "std": round(std, 2),
        "n_games": n,
        "side": side,
        "p_cal": round(p_cal, 4),
        "ev_pct": round(ev_pct, 2),
        "edge_cat": edge_cat,
    }
