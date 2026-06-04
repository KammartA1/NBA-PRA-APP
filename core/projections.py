"""
core/projections.py — Standalone NBA projection engine for background workers.
No Streamlit dependency. Uses nba_api for player game logs.

Projection method: Weighted rolling averages with recency bias,
home/away adjustment, and volatility-based probability estimation.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

log = logging.getLogger(__name__)

STAT_COLUMNS = {
    "PTS": "PTS", "REB": "REB", "AST": "AST",
    "FG3M": "FG3M", "STL": "STL", "BLK": "BLK",
    "TOV": "TOV", "FGA": "FGA", "FTM": "FTM",
}
COMBO_STATS = {
    "PRA": ["PTS", "REB", "AST"],
    "PA": ["PTS", "AST"],
    "PR": ["PTS", "REB"],
    "RA": ["REB", "AST"],
    "BLST": ["BLK", "STL"],
    "FS": None,
}


def _compute_stat(df: pd.DataFrame, market: str) -> pd.Series:
    if market in STAT_COLUMNS:
        col = STAT_COLUMNS[market]
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    if market == "PRA":
        return (pd.to_numeric(df["PTS"], errors="coerce").fillna(0) +
                pd.to_numeric(df["REB"], errors="coerce").fillna(0) +
                pd.to_numeric(df["AST"], errors="coerce").fillna(0))
    if market == "PA":
        return (pd.to_numeric(df["PTS"], errors="coerce").fillna(0) +
                pd.to_numeric(df["AST"], errors="coerce").fillna(0))
    if market == "PR":
        return (pd.to_numeric(df["PTS"], errors="coerce").fillna(0) +
                pd.to_numeric(df["REB"], errors="coerce").fillna(0))
    if market == "RA":
        return (pd.to_numeric(df["REB"], errors="coerce").fillna(0) +
                pd.to_numeric(df["AST"], errors="coerce").fillna(0))
    if market == "BLST":
        return (pd.to_numeric(df["BLK"], errors="coerce").fillna(0) +
                pd.to_numeric(df["STL"], errors="coerce").fillna(0))
    if market == "FS":
        return (pd.to_numeric(df["PTS"], errors="coerce").fillna(0) * 1.0 +
                pd.to_numeric(df["REB"], errors="coerce").fillna(0) * 1.25 +
                pd.to_numeric(df["AST"], errors="coerce").fillna(0) * 1.5 +
                pd.to_numeric(df["STL"], errors="coerce").fillna(0) * 2.0 +
                pd.to_numeric(df["BLK"], errors="coerce").fillna(0) * 2.0 +
                pd.to_numeric(df["TOV"], errors="coerce").fillna(0) * -0.5)
    return pd.Series([0] * len(df))


def fetch_player_gamelog(player_id: int, n_games: int = 20) -> Optional[pd.DataFrame]:
    try:
        from nba_api.stats.endpoints import playergamelog
        time.sleep(0.6)
        gl = playergamelog.PlayerGameLog(
            player_id=player_id,
            season="2025-26",
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
        if df.empty:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season="2024-25",
                season_type_all_star="Regular Season",
            )
            df = gl.get_data_frames()[0]
        return df.head(n_games) if not df.empty else None
    except Exception as e:
        log.warning("Failed to fetch gamelog for player_id=%s: %s", player_id, e)
        return None


def resolve_player_id(name: str) -> Optional[int]:
    try:
        from nba_api.stats.static import players as nba_players
        matches = nba_players.find_players_by_full_name(name)
        if matches:
            return matches[0]["id"]
        parts = name.split()
        if len(parts) >= 2:
            last = parts[-1]
            candidates = nba_players.find_players_by_last_name(last)
            active = [p for p in candidates if p.get("is_active")]
            if len(active) == 1:
                return active[0]["id"]
            first_initial = parts[0][0].lower()
            for p in active:
                if p["full_name"].lower().startswith(first_initial):
                    return p["id"]
        return None
    except Exception as e:
        log.warning("resolve_player_id(%s) failed: %s", name, e)
        return None


def actual_stat_for_date(
    gamelog_df: pd.DataFrame,
    market: str,
    game_date,
) -> Optional[float]:
    """Return the player's ACTUAL value for `market` on `game_date`.

    `game_date` is a datetime.date (the ET date the game was played).
    Returns None if the player has no game row on that date (DNP / no game),
    which the grader treats as a void.
    """
    if gamelog_df is None or gamelog_df.empty or "GAME_DATE" not in gamelog_df.columns:
        return None
    try:
        dates = pd.to_datetime(
            gamelog_df["GAME_DATE"], format="%b %d, %Y", errors="coerce"
        ).dt.date
    except Exception:
        return None
    mask = dates == game_date
    if not mask.any():
        return None
    row = gamelog_df[mask]
    stat_series = _compute_stat(row, market)
    if stat_series is None or len(stat_series) == 0:
        return None
    try:
        return float(stat_series.iloc[0])
    except Exception:
        return None


def project_player_prop(
    gamelog_df: pd.DataFrame,
    market: str,
    line: float,
) -> dict:
    """
    Calculate projection and probability for a player prop.

    Returns dict with: proj, p_over, p_under, ev_over, ev_under, l5_avg, l10_avg,
                       std, n_games, side, p_cal, ev_pct, edge_cat
    """
    stat_series = _compute_stat(gamelog_df, market)
    n = len(stat_series)
    if n < 3:
        return {"error": "insufficient_games", "n_games": n}

    # Weighted averages: L5 gets 45%, L10 gets 35%, rest gets 20%
    l5 = stat_series.head(5)
    l10 = stat_series.head(10)
    full = stat_series

    l5_avg = l5.mean() if len(l5) >= 3 else full.mean()
    l10_avg = l10.mean() if len(l10) >= 5 else full.mean()
    full_avg = full.mean()

    if n >= 10:
        proj = 0.45 * l5_avg + 0.35 * l10_avg + 0.20 * full_avg
    elif n >= 5:
        proj = 0.55 * l5_avg + 0.45 * full_avg
    else:
        proj = full_avg

    # Volatility
    std = stat_series.std(ddof=1) if n >= 4 else stat_series.std(ddof=0)
    if std < 0.5:
        std = 0.5

    # P(over) via normal CDF
    # Use continuity correction: P(X > line) ≈ P(X > line - 0.5) for discrete stats
    z = (proj - line) / std
    p_over = norm.sf(line - 0.5, loc=proj, scale=std)
    p_under = 1 - p_over

    # Clamp
    p_over = max(0.02, min(0.98, p_over))
    p_under = max(0.02, min(0.98, p_under))

    # EV calculation (PrizePicks is flat 50/50 payout = 1:1)
    # EV = p * 1 - (1-p) * 1 = 2p - 1
    ev_over = (p_over - 0.50) * 100
    ev_under = (p_under - 0.50) * 100

    # Pick the better side
    if p_over >= p_under:
        side = "Over"
        p_cal = p_over
        ev_pct = ev_over
    else:
        side = "Under"
        p_cal = p_under
        ev_pct = ev_under

    # Edge category
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
