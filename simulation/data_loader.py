"""
simulation/data_loader.py
=========================
Converts real player/team data from the app's existing data sources
(nba_api calls, gamelog fetches, TEAM_CTX) into the format expected
by the PossessionSimulator / GameEngine.

This module does NOT create new API connections.  It wraps the functions
already defined in ``nba_engine`` and transforms their output into
``PlayerProfile`` objects and team-context dicts for the simulator.

Usage::

    from simulation.data_loader import SimulationDataLoader
    loader = SimulationDataLoader()
    profiles = loader.build_player_profiles(player_name, player_id, gamelog_df, ...)
    engine = GameEngine(config, home_profiles, away_profiles, ...)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation.config import CoachArchetype, SimulationConfig, DEFAULT_CONFIG
from simulation.player_state import PlayerProfile


# ---------------------------------------------------------------------------
# Stat extraction helpers (pure functions, no API calls)
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    """Safely coerce a value to float."""
    try:
        if val is None:
            return default
        if isinstance(val, str) and ":" in val:
            return float(val.split(":")[0])
        return float(val)
    except (ValueError, TypeError):
        return default


def _gamelog_mean(df: pd.DataFrame, col: str, n: int = 10, default: float = 0.0) -> float:
    """Mean of a gamelog column over the last *n* games."""
    if df is None or df.empty or col not in df.columns:
        return default
    vals = pd.to_numeric(df[col].head(n), errors="coerce").dropna()
    return float(vals.mean()) if len(vals) >= 1 else default


def _gamelog_rate(df: pd.DataFrame, num_col: str, denom_col: str,
                  n: int = 10, default: float = 0.0) -> float:
    """Ratio of means for two gamelog columns (e.g., FGM/FGA -> FG%)."""
    num = _gamelog_mean(df, num_col, n, 0.0)
    den = _gamelog_mean(df, denom_col, n, 1.0)
    return float(num / den) if den > 0 else default


# ---------------------------------------------------------------------------
# Position mapper
# ---------------------------------------------------------------------------

_POS_STANDARD = {
    "PG": "PG", "SG": "SG", "SF": "SF", "PF": "PF", "C": "C",
    "G": "PG", "F": "SF", "G-F": "SG", "F-G": "SF", "F-C": "PF",
    "C-F": "C", "GUARD": "PG", "FORWARD": "SF", "CENTER": "C",
}


def _standardize_position(pos: str) -> str:
    """Map free-form position strings to standard 5 positions."""
    if not pos:
        return "SF"
    # Take first listed position (e.g., "SG-SF" -> "SG")
    primary = pos.strip().upper().split("-")[0].split("/")[0]
    return _POS_STANDARD.get(primary, _POS_STANDARD.get(pos.strip().upper(), "SF"))


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class SimulationDataLoader:
    """Converts existing app data into simulation-ready PlayerProfile objects.

    All data sources are passed in from the caller (nba_engine already
    fetches gamelogs, team context, positions, etc.).  This class purely
    *transforms* that data into the shape the simulator expects.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Build a single PlayerProfile from gamelog + metadata
    # ------------------------------------------------------------------

    def build_player_profile(
        self,
        player_name: str,
        player_id: str | int,
        gamelog_df: pd.DataFrame,
        position: str = "",
        is_starter: bool = True,
        rotation_order: int = 0,
        rest_days: int = 1,
        n_games: int = 10,
        age: int = 27,
    ) -> PlayerProfile:
        """Create a PlayerProfile from a player's NBA gamelog DataFrame.

        Parameters
        ----------
        player_name : display name
        player_id : NBA player ID (str or int)
        gamelog_df : DataFrame from ``fetch_player_gamelog``
        position : raw position string (e.g. "SG", "PF-C")
        is_starter : whether this player is in the starting 5
        rotation_order : 0-4 for starters, 5+ for bench
        rest_days : days since last game
        n_games : number of recent games to use for rate estimation
        age : player age in years
        """
        df = gamelog_df.head(n_games) if gamelog_df is not None and not gamelog_df.empty else pd.DataFrame()
        pos = _standardize_position(position)

        # --- Minutes ---
        avg_minutes = _gamelog_mean(df, "MIN", n_games, 28.0 if is_starter else 16.0)

        # --- Shooting percentages ---
        two_pt_pct = 0.525
        three_pt_pct = 0.362
        ft_pct = 0.775
        if not df.empty:
            fgm = _gamelog_mean(df, "FGM", n_games, 0)
            fga = _gamelog_mean(df, "FGA", n_games, 1)
            fg3m = _gamelog_mean(df, "FG3M", n_games, 0)
            fg3a = _gamelog_mean(df, "FG3A", n_games, 1)
            ftm = _gamelog_mean(df, "FTM", n_games, 0)
            fta = _gamelog_mean(df, "FTA", n_games, 1)

            if fga > 0 and fg3a >= 0:
                twop_a = fga - fg3a
                twop_m = fgm - fg3m
                if twop_a > 0:
                    two_pt_pct = float(np.clip(twop_m / twop_a, 0.30, 0.75))
            if fg3a > 0:
                three_pt_pct = float(np.clip(fg3m / fg3a, 0.20, 0.50))
            if fta > 0:
                ft_pct = float(np.clip(ftm / fta, 0.40, 0.95))

        # --- Per-possession rates ---
        # Usage rate approximation: (FGA + 0.44*FTA + TOV) / (team possessions per player)
        # We normalize to 0-1 range representing share of possessions used.
        usage_raw = 0.0
        if not df.empty:
            fga_m = _gamelog_mean(df, "FGA", n_games, 0)
            fta_m = _gamelog_mean(df, "FTA", n_games, 0)
            tov_m = _gamelog_mean(df, "TOV", n_games, 0)
            usage_raw = fga_m + 0.44 * fta_m + tov_m
        # Estimate team possessions per game from context or default ~100
        est_team_poss = 100.0
        usage_rate = float(np.clip(usage_raw / max(est_team_poss, 50), 0.05, 0.40)) if usage_raw > 0 else (
            0.20 if is_starter else 0.12
        )

        # Assist rate: AST / minutes-weighted teammates-FGM estimate
        ast_avg = _gamelog_mean(df, "AST", n_games, 0)
        # Approximate: assist on ~60% of teammate FGs → normalize
        assist_rate = float(np.clip(ast_avg / max(avg_minutes, 10) * 2.5, 0.05, 0.45)) if ast_avg > 0 else 0.12

        # Rebound rate
        reb_avg = _gamelog_mean(df, "REB", n_games, 0)
        rebound_rate = float(np.clip(reb_avg / max(avg_minutes, 10) * 2.0, 0.03, 0.30)) if reb_avg > 0 else 0.10

        # Steal rate (per possession)
        stl_avg = _gamelog_mean(df, "STL", n_games, 0)
        steal_rate = float(np.clip(stl_avg / max(avg_minutes, 10) * 0.8, 0.005, 0.04)) if stl_avg > 0 else 0.015

        # Block rate
        blk_avg = _gamelog_mean(df, "BLK", n_games, 0)
        block_rate = float(np.clip(blk_avg / max(avg_minutes, 10) * 0.8, 0.002, 0.06)) if blk_avg > 0 else 0.010

        # Turnover rate (per usage)
        tov_avg = _gamelog_mean(df, "TOV", n_games, 0)
        turnover_rate = float(np.clip(tov_avg / max(usage_raw, 5) if usage_raw > 0 else 0.12, 0.05, 0.25))

        # Foul rates
        pf_avg = _gamelog_mean(df, "PF", n_games, 0)
        foul_rate = float(np.clip(pf_avg / max(avg_minutes, 10) * 0.5, 0.01, 0.06)) if pf_avg > 0 else 0.030
        fta_avg = _gamelog_mean(df, "FTA", n_games, 0)
        foul_draw_rate = float(np.clip(fta_avg / max(avg_minutes, 10) * 0.3, 0.01, 0.06)) if fta_avg > 0 else 0.030

        # Height heuristic from position
        _height_by_pos = {"PG": 74, "SG": 76, "SF": 79, "PF": 81, "C": 83}

        return PlayerProfile(
            name=player_name,
            player_id=str(player_id),
            position=pos,
            age=age,
            height_inches=_height_by_pos.get(pos, 78),
            rest_days=rest_days,
            two_pt_pct=two_pt_pct,
            three_pt_pct=three_pt_pct,
            ft_pct=ft_pct,
            usage_rate=usage_rate,
            assist_rate=assist_rate,
            rebound_rate=rebound_rate,
            steal_rate=steal_rate,
            block_rate=block_rate,
            turnover_rate=turnover_rate,
            foul_rate=foul_rate,
            foul_draw_rate=foul_draw_rate,
            is_starter=is_starter,
            rotation_order=rotation_order,
        )

    # ------------------------------------------------------------------
    # Build full roster profiles for a team
    # ------------------------------------------------------------------

    def build_team_profiles(
        self,
        roster_data: List[Dict[str, Any]],
        n_games: int = 10,
    ) -> List[PlayerProfile]:
        """Build PlayerProfile list for an entire team roster.

        Parameters
        ----------
        roster_data : list of dicts, each containing:
            - "player_name": str
            - "player_id": str or int
            - "gamelog_df": pd.DataFrame (from fetch_player_gamelog)
            - "position": str (optional)
            - "is_starter": bool (optional, default True for first 5)
            - "rest_days": int (optional, default 1)
            - "age": int (optional, default 27)
        n_games : number of recent games for rate estimation
        """
        profiles = []
        for i, rd in enumerate(roster_data):
            is_starter = rd.get("is_starter", i < 5)
            profiles.append(self.build_player_profile(
                player_name=rd["player_name"],
                player_id=rd["player_id"],
                gamelog_df=rd.get("gamelog_df", pd.DataFrame()),
                position=rd.get("position", ""),
                is_starter=is_starter,
                rotation_order=i,
                rest_days=rd.get("rest_days", 1),
                n_games=n_games,
                age=rd.get("age", 27),
            ))
        return profiles

    # ------------------------------------------------------------------
    # Extract team context for GameEngine from TEAM_CTX
    # ------------------------------------------------------------------

    @staticmethod
    def extract_team_context(
        team_ctx: Dict[str, Dict[str, float]],
        team_abbr: str,
        opp_abbr: str,
    ) -> Dict[str, float]:
        """Extract pace and ratings from the app's TEAM_CTX global.

        Returns a dict with keys: home_pace, away_pace, spread_estimate.
        """
        default_pace = 100.0
        t = team_ctx.get(str(team_abbr).upper(), {})
        o = team_ctx.get(str(opp_abbr).upper(), {})
        home_pace = float(t.get("PACE", default_pace))
        away_pace = float(o.get("PACE", default_pace))
        # Spread estimate from net ratings (positive = home favored)
        home_net = float(t.get("NET_RATING", 0))
        away_net = float(o.get("NET_RATING", 0))
        spread_est = -(home_net - away_net) / 2.5  # rough conversion to spread points
        return {
            "home_pace": home_pace,
            "away_pace": away_pace,
            "spread_estimate": float(np.clip(spread_est, -20, 20)),
        }

    # ------------------------------------------------------------------
    # Map market names to simulation stat keys
    # ------------------------------------------------------------------

    # Maps the app's market names (e.g., "Points", "H1 Rebounds") to
    # the keys used in GameEngine STAT_KEYS / PlayerDistribution
    MARKET_TO_SIM_STAT = {
        "Points": "points",
        "Rebounds": "rebounds",
        "Assists": "assists",
        "PRA": "pra",
        "PR": "pr",
        "PA": "pa",
        "RA": "ra",
        "Blocks": "blocks",
        "Steals": "steals",
        "Turnovers": "turnovers",
        "3PM": "three_pm",
        "FGM": "fgm",
        "FGA": "fga",
        "3PA": "three_pa",
        "FTM": "ftm",
        "FTA": "fta",
        # Half-game markets → h1_/h2_ sim stat keys
        "H1 Points": "h1_points",
        "H1 Rebounds": "h1_rebounds",
        "H1 Assists": "h1_assists",
        "H1 3PM": "h1_three_pm",
        "H1 PRA": "h1_pra",
        "H1 PR": "h1_pr",
        "H1 PA": "h1_pa",
        "H1 RA": "h1_ra",
        "H1 FGM": "h1_fgm",
        "H1 FGA": "h1_fga",
        "H1 FTM": "h1_ftm",
        "H1 FTA": "h1_fta",
        "H2 Points": "h2_points",
        "H2 Rebounds": "h2_rebounds",
        "H2 Assists": "h2_assists",
        "H2 3PM": "h2_three_pm",
        "H2 PRA": "h2_pra",
        "H2 PR": "h2_pr",
        "H2 PA": "h2_pa",
        "H2 RA": "h2_ra",
        "H2 FGM": "h2_fgm",
        "H2 FGA": "h2_fga",
        "H2 FTM": "h2_ftm",
        "H2 FTA": "h2_fta",
    }

    @classmethod
    def market_to_sim_key(cls, market_name: str) -> Optional[str]:
        """Convert an app market name to the corresponding simulation stat key.

        Returns None if the market has no direct simulation equivalent
        (e.g., Double Double, Fantasy Score, Stocks).
        """
        return cls.MARKET_TO_SIM_STAT.get(market_name)

    # ------------------------------------------------------------------
    # Determine coach archetype from team context
    # ------------------------------------------------------------------

    @staticmethod
    def infer_coach_archetype(pace: float) -> CoachArchetype:
        """Heuristic: faster teams tend toward starter-heavy rotations,
        slower teams use deeper benches."""
        if pace >= 103.0:
            return CoachArchetype.STARTER_HEAVY
        elif pace <= 97.0:
            return CoachArchetype.DEEP_BENCH
        return CoachArchetype.BALANCED
