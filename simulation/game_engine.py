"""
simulation/game_engine.py
=========================
The core game engine.  Orchestrates a full NBA game simulation at the
possession level: alternating possessions, lineup management, fatigue,
fouls, game script, blowout detection, and stat accumulation.

Main entry point: ``GameEngine.run_simulation(n)`` which runs N
independent game simulations and returns full stat distributions.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from simulation.config import (
    CoachArchetype,
    SimulationConfig,
    DEFAULT_CONFIG,
)
from simulation.player_state import PlayerProfile, PlayerState
from simulation.team_state import TeamState
from simulation.fatigue_model import FatigueModel
from simulation.foul_model import FoulModel
from simulation.game_script import GameScriptModel
from simulation.blowout_model import BlowoutModel
from simulation.lineup_manager import LineupManager
from simulation.possession import PossessionEngine, PossessionResult


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PlayerDistribution:
    """Statistical summary of a player's simulated stat distributions."""
    player_name: str
    player_id: str
    stat_name: str                     # "points", "rebounds", "assists", "pra", etc.
    n_sims: int
    values: np.ndarray                 # raw array of simulated values

    # Descriptive stats (populated by compute())
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    p5: float = 0.0
    p10: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0

    def compute(self) -> None:
        """Populate descriptive statistics from the raw values array."""
        v = self.values.astype(np.float64)
        self.mean = float(np.mean(v))
        self.median = float(np.median(v))
        self.std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        self.skew = float(sp_stats.skew(v)) if len(v) > 2 else 0.0
        self.kurtosis = float(sp_stats.kurtosis(v)) if len(v) > 3 else 0.0
        self.p5 = float(np.percentile(v, 5))
        self.p10 = float(np.percentile(v, 10))
        self.p25 = float(np.percentile(v, 25))
        self.p75 = float(np.percentile(v, 75))
        self.p90 = float(np.percentile(v, 90))
        self.p95 = float(np.percentile(v, 95))
        self.min_val = float(np.min(v))
        self.max_val = float(np.max(v))

    def prob_over(self, line: float) -> float:
        """P(stat > line) from the empirical distribution."""
        return float(np.mean(self.values > line))

    def prob_under(self, line: float) -> float:
        """P(stat < line) from the empirical distribution."""
        return float(np.mean(self.values < line))

    def to_dict(self) -> dict:
        return {
            "player": self.player_name,
            "stat": self.stat_name,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "std": round(self.std, 2),
            "skew": round(self.skew, 3),
            "kurtosis": round(self.kurtosis, 3),
            "p5": round(self.p5, 1),
            "p10": round(self.p10, 1),
            "p25": round(self.p25, 1),
            "p75": round(self.p75, 1),
            "p90": round(self.p90, 1),
            "p95": round(self.p95, 1),
            "min": round(self.min_val, 1),
            "max": round(self.max_val, 1),
        }


@dataclass
class SimulationOutput:
    """Full output from run_simulation()."""
    n_simulations: int
    home_team: str
    away_team: str
    distributions: Dict[str, Dict[str, PlayerDistribution]]
    # distributions[player_id][stat_name] -> PlayerDistribution
    game_results: List[Dict[str, Any]]   # per-sim summary (scores, blowout, etc.)

    def get_player_dist(
        self, player_id: str, stat: str,
    ) -> Optional[PlayerDistribution]:
        return self.distributions.get(player_id, {}).get(stat)

    def prob_over(self, player_id: str, stat: str, line: float) -> Optional[float]:
        dist = self.get_player_dist(player_id, stat)
        return dist.prob_over(line) if dist else None

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten distributions to a summary DataFrame."""
        rows = []
        for pid, stat_map in self.distributions.items():
            for stat_name, dist in stat_map.items():
                rows.append(dist.to_dict())
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Game Engine
# ---------------------------------------------------------------------------

class GameEngine:
    """Possession-level NBA game simulator.

    Usage::

        engine = GameEngine(config, home_profiles, away_profiles)
        output = engine.run_simulation(n=10000)
        p_over = output.prob_over("player_123", "points", 24.5)
    """

    STAT_KEYS = [
        "points", "rebounds", "assists", "steals", "blocks",
        "turnovers", "minutes", "pra", "pr", "pa", "ra",
        "fgm", "fga", "three_pm", "three_pa", "ftm", "fta",
        # Half-game stats (H1 = Q1+Q2, H2 = Q3+Q4)
        "h1_points", "h1_rebounds", "h1_assists", "h1_steals", "h1_blocks",
        "h1_turnovers", "h1_minutes", "h1_pra", "h1_pr", "h1_pa", "h1_ra",
        "h1_fgm", "h1_fga", "h1_three_pm", "h1_three_pa", "h1_ftm", "h1_fta",
        "h2_points", "h2_rebounds", "h2_assists", "h2_steals", "h2_blocks",
        "h2_turnovers", "h2_minutes", "h2_pra", "h2_pr", "h2_pa", "h2_ra",
        "h2_fgm", "h2_fga", "h2_three_pm", "h2_three_pa", "h2_ftm", "h2_fta",
    ]

    def __init__(
        self,
        config: SimulationConfig | None = None,
        home_profiles: List[PlayerProfile] | None = None,
        away_profiles: List[PlayerProfile] | None = None,
        home_name: str = "Home",
        away_name: str = "Away",
        home_pace: float = 100.0,
        away_pace: float = 100.0,
        pre_game_spread: float = 0.0,
        home_archetype: CoachArchetype | None = None,
        away_archetype: CoachArchetype | None = None,
    ) -> None:
        self.cfg = config or DEFAULT_CONFIG
        self.home_profiles = home_profiles or self._default_roster(True)
        self.away_profiles = away_profiles or self._default_roster(False)
        self.home_name = home_name
        self.away_name = away_name
        self.home_pace = home_pace
        self.away_pace = away_pace
        self.pre_game_spread = pre_game_spread
        self.home_archetype = home_archetype
        self.away_archetype = away_archetype

        # Sub-models
        self.fatigue_model = FatigueModel(self.cfg)
        self.foul_model = FoulModel(self.cfg)
        self.possession_engine = PossessionEngine(self.cfg)
        self.blowout_model = BlowoutModel(self.cfg)

    # ------------------------------------------------------------------
    # Default roster generation (for testing / when no data available)
    # ------------------------------------------------------------------

    @staticmethod
    def _default_roster(is_home: bool) -> List[PlayerProfile]:
        """Generate a realistic 12-man roster with default attributes."""
        positions = ["PG", "SG", "SF", "PF", "C",
                      "PG", "SG", "SF", "PF", "C", "SF", "PF"]
        prefix = "H" if is_home else "A"
        profiles = []
        for i, pos in enumerate(positions):
            is_starter = i < 5
            usage = 0.22 - i * 0.012 if is_starter else 0.12
            profiles.append(PlayerProfile(
                name=f"{prefix}_Player_{i+1}",
                player_id=f"{prefix.lower()}_{i+1}",
                position=pos,
                age=25 + (i % 8),
                height_inches=72 + (i % 5) * 2 + (3 if pos == "C" else 0),
                rest_days=1,
                two_pt_pct=0.54 + (0.02 if is_starter else -0.01),
                three_pt_pct=0.37 + (0.02 if is_starter else -0.01),
                ft_pct=0.78 + (0.02 if is_starter else -0.02),
                usage_rate=max(usage, 0.08),
                assist_rate=0.22 if pos == "PG" else 0.12,
                rebound_rate=0.15 if pos in ("C", "PF") else 0.07,
                steal_rate=0.018 if pos in ("PG", "SG") else 0.012,
                block_rate=0.025 if pos == "C" else 0.008,
                turnover_rate=0.12,
                foul_rate=0.035 if pos in ("C", "PF") else 0.025,
                foul_draw_rate=0.03,
                is_starter=is_starter,
                rotation_order=i,
            ))
        return profiles

    # ------------------------------------------------------------------
    # Build fresh game state
    # ------------------------------------------------------------------

    def _build_team(
        self,
        profiles: List[PlayerProfile],
        team_name: str,
        pace: float,
    ) -> TeamState:
        players = [PlayerState(profile=copy.deepcopy(p)) for p in profiles]
        team = TeamState(team_name=team_name, players=players, pace_factor=pace)
        team.initialize_lineup()
        team.update_dynamic_usage()
        return team

    # ------------------------------------------------------------------
    # Simulate a single game
    # ------------------------------------------------------------------

    def simulate_game(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Run one full game simulation.  Returns a dict with player stat
        lines and game metadata.
        """
        cfg = self.cfg

        # Build fresh state
        home = self._build_team(self.home_profiles, self.home_name, self.home_pace)
        away = self._build_team(self.away_profiles, self.away_name, self.away_pace)

        # Sub-models (per-game instances for stateful models)
        home_lineup_mgr = LineupManager(cfg, self.home_archetype)
        away_lineup_mgr = LineupManager(cfg, self.away_archetype)
        home_script = GameScriptModel(cfg)
        away_script = GameScriptModel(cfg)
        home_lineup_mgr.initialize(home)
        away_lineup_mgr.initialize(away)

        # Game pace = average of both teams
        game_pace = (self.home_pace + self.away_pace) / 2.0
        total_poss = int(round(cfg.possessions_per_game * game_pace / cfg.league_avg_pace))
        min_per_poss = 48.0 / max(total_poss, 1)

        offense_is_home = True
        blowout_detected = False
        blowout_possession = -1

        possession_num = 0
        while possession_num < total_poss:
            quarter = min(possession_num // (total_poss // 4) + 1, 4)
            # --- Track which half we're in (H1 = Q1-Q2, H2 = Q3-Q4) ---
            current_half = 1 if quarter <= 2 else 2
            for p in home.players + away.players:
                p._current_half = current_half

            # Reset quarter fouls at quarter boundaries
            if possession_num > 0 and possession_num % (total_poss // 4) == 0:
                home.reset_quarter_fouls()
                away.reset_quarter_fouls()

            offense = home if offense_is_home else away
            defense = away if offense_is_home else home
            off_lineup_mgr = home_lineup_mgr if offense_is_home else away_lineup_mgr
            def_lineup_mgr = away_lineup_mgr if offense_is_home else home_lineup_mgr
            off_script_model = home_script if offense_is_home else away_script
            def_script_model = away_script if offense_is_home else home_script

            # --- Fatigue update for all players ---
            for p in home.players:
                self.fatigue_model.update_player_fatigue(p, game_pace, min_per_poss)
            for p in away.players:
                self.fatigue_model.update_player_fatigue(p, game_pace, min_per_poss)

            # --- Game script evaluation ---
            off_decision = off_script_model.evaluate(
                offense.score, defense.score,
                possession_num, total_poss,
                offense.unanswered_opponent_points,
                offense.timeouts_remaining,
            )
            def_decision = def_script_model.evaluate(
                defense.score, offense.score,
                possession_num, total_poss,
                defense.unanswered_opponent_points,
                defense.timeouts_remaining,
            )

            # --- Blowout check ---
            margin = home.score - away.score
            blowout_assessment = self.blowout_model.evaluate(
                margin, possession_num, total_poss, self.pre_game_spread,
            )
            if blowout_assessment.is_blowout and not blowout_detected:
                blowout_detected = True
                blowout_possession = possession_num

            # --- Timeout ---
            if off_decision.call_timeout:
                offense.call_timeout()
            if def_decision.call_timeout:
                defense.call_timeout()

            # --- Substitutions ---
            off_lineup_mgr.process_substitutions(
                offense, possession_num, total_poss, quarter,
                off_decision, blowout_assessment.pull_starters and offense_is_home == (margin > 0),
            )
            def_lineup_mgr.process_substitutions(
                defense, possession_num, total_poss, quarter,
                def_decision, blowout_assessment.pull_starters and offense_is_home != (margin > 0),
            )

            # --- Foul adjudication ---
            off_players = offense.get_on_court_players()
            off_indices = list(offense.current_lineup)
            def_players = defense.get_on_court_players()
            def_indices = list(defense.current_lineup)

            foul_event = self.foul_model.evaluate_possession(
                off_players, off_indices,
                def_players, def_indices,
                quarter, rng,
                defense_in_bonus=defense.in_bonus(),
            )

            if foul_event.foul_occurred and not foul_event.is_offensive:
                defense.add_team_foul()

            # --- Possession resolution ---
            if foul_event.foul_occurred and foul_event.free_throws_awarded > 0:
                # Free-throw possession
                poss_result = self.possession_engine.resolve(
                    off_players, off_indices,
                    def_players, def_indices,
                    rng,
                    free_throw_possession=True,
                    ft_shooter_idx=foul_event.fouled_player_idx,
                    ft_count=foul_event.free_throws_awarded,
                )
            elif foul_event.foul_occurred and foul_event.is_offensive:
                # Offensive foul → turnover, change possession
                poss_result = PossessionResult(
                    turnover=True, change_possession=True,
                    turnover_player_idx=foul_event.fouling_player_idx,
                )
            else:
                # Normal possession
                poss_result = self.possession_engine.resolve(
                    off_players, off_indices,
                    def_players, def_indices,
                    rng,
                )

            # --- Score updates ---
            if poss_result.points_scored > 0:
                offense.add_points(poss_result.points_scored)
                defense.opponent_scored(poss_result.points_scored)

            offense.advance_possession()

            # --- Possession change ---
            if poss_result.change_possession:
                offense_is_home = not offense_is_home

            possession_num += 1

        # --- Compile results ---
        all_stat_lines = {}
        for p in home.players + away.players:
            all_stat_lines[p.profile.player_id] = p.to_stat_line()

        return {
            "home_score": home.score,
            "away_score": away.score,
            "margin": home.score - away.score,
            "total_possessions": possession_num,
            "blowout": blowout_detected,
            "blowout_possession": blowout_possession,
            "stat_lines": all_stat_lines,
        }

    # ------------------------------------------------------------------
    # Run N simulations
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        n: int | None = None,
        seed: int | None = None,
    ) -> SimulationOutput:
        """Run *n* independent game simulations and return distributions.

        Parameters
        ----------
        n : number of simulations (default from config)
        seed : random seed for reproducibility
        """
        n = n or self.cfg.default_simulations
        seed = seed if seed is not None else self.cfg.random_seed
        rng = np.random.default_rng(seed)

        # Collect raw stat arrays
        all_results: List[Dict[str, Any]] = []
        # player_id -> stat_name -> list of values
        raw_stats: Dict[str, Dict[str, List[float]]] = {}

        for sim_idx in range(n):
            # Each sim gets a child RNG for independence
            child_rng = np.random.default_rng(rng.integers(0, 2**63))
            game_result = self.simulate_game(child_rng)
            all_results.append({
                "sim": sim_idx,
                "home_score": game_result["home_score"],
                "away_score": game_result["away_score"],
                "margin": game_result["margin"],
                "blowout": game_result["blowout"],
            })

            for pid, stats in game_result["stat_lines"].items():
                if pid not in raw_stats:
                    raw_stats[pid] = {k: [] for k in self.STAT_KEYS}
                for key in self.STAT_KEYS:
                    raw_stats[pid][key].append(stats.get(key, 0))

        # Build PlayerDistribution objects
        distributions: Dict[str, Dict[str, PlayerDistribution]] = {}
        all_profiles = self.home_profiles + self.away_profiles
        profile_map = {p.player_id: p for p in all_profiles}

        for pid, stat_map in raw_stats.items():
            distributions[pid] = {}
            pname = profile_map[pid].name if pid in profile_map else pid
            for stat_name, values in stat_map.items():
                arr = np.array(values, dtype=np.float64)
                dist = PlayerDistribution(
                    player_name=pname,
                    player_id=pid,
                    stat_name=stat_name,
                    n_sims=n,
                    values=arr,
                )
                dist.compute()
                distributions[pid][stat_name] = dist

        return SimulationOutput(
            n_simulations=n,
            home_team=self.home_name,
            away_team=self.away_name,
            distributions=distributions,
            game_results=all_results,
        )

    # ------------------------------------------------------------------
    # Convenience: P(over) calculator
    # ------------------------------------------------------------------

    @staticmethod
    def prob_over_line(
        output: SimulationOutput,
        player_id: str,
        stat: str,
        line: float,
    ) -> Optional[float]:
        """Calculate P(stat > line) for a given player from simulation output."""
        return output.prob_over(player_id, stat, line)
