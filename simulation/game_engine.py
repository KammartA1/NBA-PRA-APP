"""
simulation/game_engine.py
=========================
The core game engine.  Orchestrates a full NBA game simulation at the
possession level: alternating possessions, lineup management, fatigue,
fouls, game script, blowout detection, transition play, dynamic pace,
hot/cold streaks, context brain integration, and stat accumulation.

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
    SeasonPhase,
    SimulationConfig,
    DEFAULT_CONFIG,
)
from simulation.player_state import PlayerProfile, PlayerState
from simulation.team_state import TeamState
from simulation.fatigue_model import FatigueModel
from simulation.foul_model import FoulModel
from simulation.game_script import GameScriptModel, GamePhase
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
    stat_name: str
    n_sims: int
    values: np.ndarray

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
        return float(np.mean(self.values > line))

    def prob_under(self, line: float) -> float:
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
    game_results: List[Dict[str, Any]]

    def get_player_dist(
        self, player_id: str, stat: str,
    ) -> Optional[PlayerDistribution]:
        return self.distributions.get(player_id, {}).get(stat)

    def prob_over(self, player_id: str, stat: str, line: float) -> Optional[float]:
        dist = self.get_player_dist(player_id, stat)
        return dist.prob_over(line) if dist else None

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for pid, stat_map in self.distributions.items():
            for stat_name, dist in stat_map.items():
                rows.append(dist.to_dict())
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Game Engine
# ---------------------------------------------------------------------------

class GameEngine:
    """Possession-level NBA game simulator with Context Brain integration.

    Usage::

        engine = GameEngine(config, home_profiles, away_profiles,
                           game_context=my_context)
        output = engine.run_simulation(n=10000)
        p_over = output.prob_over("player_123", "points", 24.5)
    """

    STAT_KEYS = [
        "points", "rebounds", "assists", "steals", "blocks",
        "turnovers", "minutes", "pra", "pr", "pa", "ra",
        "fgm", "fga", "three_pm", "three_pa", "ftm", "fta",
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
        game_context: Any | None = None,
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
        self.game_context = game_context

        self.fatigue_model = FatigueModel(self.cfg)
        self.foul_model = FoulModel(self.cfg)
        self.possession_engine = PossessionEngine(self.cfg)
        self.blowout_model = BlowoutModel(self.cfg)

        self._context_adjustments = None
        if self.game_context is not None:
            try:
                from simulation.context_brain import ContextBrain
                brain = ContextBrain(self.cfg)
                self._context_adjustments = brain.compute_adjustments(self.game_context)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Default roster generation
    # ------------------------------------------------------------------

    @staticmethod
    def _default_roster(is_home: bool) -> List[PlayerProfile]:
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

    def _apply_context_to_profiles(
        self,
        profiles: List[PlayerProfile],
        is_perspective_team: bool,
    ) -> List[PlayerProfile]:
        """Apply Context Brain adjustments to player profiles pre-game."""
        adj = self._context_adjustments
        if adj is None:
            return profiles

        try:
            from simulation.context_brain import ContextBrain
            brain = ContextBrain(self.cfg)
            adjusted = []
            for p in profiles:
                is_star = p.usage_rate >= 0.25 and p.is_starter
                if is_perspective_team:
                    adjusted.append(brain.apply_to_player_profile(p, adj, is_star))
                else:
                    adjusted.append(p)
            return adjusted
        except Exception:
            return profiles

    def _apply_starting_fatigue(
        self, team: TeamState, is_perspective_team: bool,
    ) -> None:
        """Set pre-game fatigue for all players based on context."""
        adj = self._context_adjustments
        if adj is None or not is_perspective_team:
            return
        starting_fatigue = adj.fatigue_starting_level
        if starting_fatigue <= 0:
            return
        for p in team.players:
            p.fatigue_level = self.fatigue_model.compute_starting_fatigue(
                p, context_fatigue_start=starting_fatigue,
            )
            p.update_efficiency()

    # ------------------------------------------------------------------
    # Previous play type determination
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_previous_play_type(poss_result: PossessionResult) -> str:
        if poss_result.turnover or poss_result.steal:
            return "turnover"
        if poss_result.shot_made:
            return "made_shot"
        if poss_result.offensive_rebound:
            return "offensive_rebound"
        return "defensive_rebound"

    # ------------------------------------------------------------------
    # Simulate a single game
    # ------------------------------------------------------------------

    def simulate_game(self, rng: np.random.Generator) -> Dict[str, Any]:
        cfg = self.cfg
        adj = self._context_adjustments

        context_pace_mult = adj.pace_multiplier if adj else 1.0
        context_defense_adj = adj.defense_adjustment if adj else 0.0
        context_3pt_mod = adj.three_pt_rate_modifier if adj else 0.0
        context_2pt_mod = adj.two_pt_rate_modifier if adj else 0.0
        context_fatigue_rate_mult = adj.fatigue_rate_multiplier if adj else 1.0
        altitude_factor = 0.0
        if adj and self.game_context:
            city = getattr(self.game_context, "altitude_city", "")
            if city in ("DEN", "UTA", "SLC"):
                altitude_factor = cfg.fatigue_altitude_factor

        home_profiles = self._apply_context_to_profiles(self.home_profiles, True)
        away_profiles = self._apply_context_to_profiles(self.away_profiles, False)

        home = self._build_team(home_profiles, self.home_name, self.home_pace)
        away = self._build_team(away_profiles, self.away_name, self.away_pace)

        self._apply_starting_fatigue(home, True)
        self._apply_starting_fatigue(away, False)

        home_lineup_mgr = LineupManager(cfg, self.home_archetype)
        away_lineup_mgr = LineupManager(cfg, self.away_archetype)
        home_script = GameScriptModel(cfg)
        away_script = GameScriptModel(cfg)
        home_lineup_mgr.initialize(home)
        away_lineup_mgr.initialize(away)

        base_game_pace = (self.home_pace + self.away_pace) / 2.0 * context_pace_mult
        base_total_poss = int(round(cfg.possessions_per_game * base_game_pace / cfg.league_avg_pace))
        base_min_per_poss = 48.0 / max(base_total_poss, 1)

        offense_is_home = True
        blowout_detected = False
        blowout_possession = -1
        transition_possessions = 0

        previous_play_type = "made_shot"
        hot_streaks: Dict[int, int] = {}

        elapsed_minutes = 0.0
        possession_num = 0
        max_possessions = base_total_poss + 40

        while elapsed_minutes < 48.0 and possession_num < max_possessions:
            poss_per_quarter = max(base_total_poss // 4, 1)
            quarter = min(int(elapsed_minutes / 12.0) + 1, 4)
            current_half = 1 if quarter <= 2 else 2
            for p in home.players + away.players:
                p._current_half = current_half

            if possession_num > 0 and quarter > 1:
                prev_quarter = min(int((elapsed_minutes - base_min_per_poss) / 12.0) + 1, 4)
                if prev_quarter < quarter:
                    home.reset_quarter_fouls()
                    away.reset_quarter_fouls()

            offense = home if offense_is_home else away
            defense = away if offense_is_home else home
            off_lineup_mgr = home_lineup_mgr if offense_is_home else away_lineup_mgr
            def_lineup_mgr = away_lineup_mgr if offense_is_home else home_lineup_mgr
            off_script_model = home_script if offense_is_home else away_script
            def_script_model = away_script if offense_is_home else home_script

            off_decision = off_script_model.evaluate(
                offense.score, defense.score,
                possession_num, base_total_poss,
                offense.unanswered_opponent_points,
                offense.timeouts_remaining,
            )
            def_decision = def_script_model.evaluate(
                defense.score, offense.score,
                possession_num, base_total_poss,
                defense.unanswered_opponent_points,
                defense.timeouts_remaining,
            )

            script_pace_mult = getattr(off_decision, 'pace_multiplier', 1.0)
            effective_pace_mult = context_pace_mult * script_pace_mult
            min_per_poss = base_min_per_poss / max(effective_pace_mult, 0.5)
            min_per_poss = max(0.15, min(min_per_poss, 0.6))

            for p in home.players:
                self.fatigue_model.update_player_fatigue(
                    p, base_game_pace, min_per_poss,
                    altitude_factor=altitude_factor if not offense_is_home else 0.0,
                    schedule_density_factor=(context_fatigue_rate_mult - 1.0),
                )
            for p in away.players:
                self.fatigue_model.update_player_fatigue(
                    p, base_game_pace, min_per_poss,
                    altitude_factor=altitude_factor if offense_is_home else 0.0,
                    schedule_density_factor=(context_fatigue_rate_mult - 1.0),
                )

            margin = home.score - away.score
            blowout_assessment = self.blowout_model.evaluate(
                margin, possession_num, base_total_poss, self.pre_game_spread,
            )
            if blowout_assessment.is_blowout and not blowout_detected:
                blowout_detected = True
                blowout_possession = possession_num

            if off_decision.call_timeout:
                offense.call_timeout()
            if def_decision.call_timeout:
                defense.call_timeout()

            off_lineup_mgr.process_substitutions(
                offense, possession_num, base_total_poss, quarter,
                off_decision,
                blowout_assessment.pull_starters and offense_is_home == (margin > 0),
            )
            def_lineup_mgr.process_substitutions(
                defense, possession_num, base_total_poss, quarter,
                def_decision,
                blowout_assessment.pull_starters and offense_is_home != (margin > 0),
            )

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

            is_clutch = getattr(off_decision, 'phase', None) == GamePhase.CLUTCH
            script_3pt_mod = getattr(off_decision, 'three_pt_rate_modifier', 0.0)
            script_2pt_mod = getattr(off_decision, 'two_pt_rate_modifier', 0.0)
            script_ft_mod = getattr(off_decision, 'ft_draw_modifier', 0.0)
            two_for_one = getattr(off_decision, 'two_for_one', False)
            end_quarter_heave = getattr(off_decision, 'end_quarter_heave', False)

            combined_3pt_mod = context_3pt_mod + script_3pt_mod
            combined_2pt_mod = context_2pt_mod + script_2pt_mod

            if foul_event.foul_occurred and foul_event.free_throws_awarded > 0:
                poss_result = self.possession_engine.resolve(
                    off_players, off_indices,
                    def_players, def_indices,
                    rng,
                    free_throw_possession=True,
                    ft_shooter_idx=foul_event.fouled_player_idx,
                    ft_count=foul_event.free_throws_awarded,
                )
            elif foul_event.foul_occurred and foul_event.is_offensive:
                poss_result = PossessionResult(
                    turnover=True, change_possession=True,
                    turnover_player_idx=foul_event.fouling_player_idx,
                )
            else:
                poss_result = self.possession_engine.resolve(
                    off_players, off_indices,
                    def_players, def_indices,
                    rng,
                    previous_play_type=previous_play_type,
                    three_pt_rate_modifier=combined_3pt_mod,
                    two_pt_rate_modifier=combined_2pt_mod,
                    ft_draw_modifier=script_ft_mod,
                    defense_adjustment=context_defense_adj,
                    is_clutch=is_clutch,
                    player_hot_streaks=hot_streaks,
                    two_for_one=two_for_one,
                    end_quarter_heave=end_quarter_heave,
                )

            if getattr(poss_result, 'is_transition', False):
                transition_possessions += 1

            if poss_result.shot_made and poss_result.shooter_idx is not None:
                hot_streaks[poss_result.shooter_idx] = hot_streaks.get(poss_result.shooter_idx, 0) + 1
                _shooter_pid = str(poss_result.shooter_idx)
                if hasattr(off_script_model, 'record_player_make'):
                    off_script_model.record_player_make(_shooter_pid)
            elif poss_result.shot_attempted and poss_result.shooter_idx is not None:
                hot_streaks[poss_result.shooter_idx] = 0
                _shooter_pid = str(poss_result.shooter_idx)
                if hasattr(off_script_model, 'record_player_miss'):
                    off_script_model.record_player_miss(_shooter_pid)

            if poss_result.points_scored > 0:
                offense.add_points(poss_result.points_scored)
                defense.opponent_scored(poss_result.points_scored)
                if hasattr(off_script_model, 'record_own_score'):
                    off_script_model.record_own_score(poss_result.points_scored)
                if hasattr(def_script_model, 'record_opponent_score'):
                    def_script_model.record_opponent_score(poss_result.points_scored)
            else:
                if hasattr(off_script_model, 'record_no_score'):
                    off_script_model.record_no_score()

            offense.advance_possession()

            previous_play_type = self._determine_previous_play_type(poss_result)

            if poss_result.change_possession:
                offense_is_home = not offense_is_home

            elapsed_minutes += min_per_poss
            possession_num += 1

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
            "transition_possessions": transition_possessions,
            "elapsed_minutes": round(elapsed_minutes, 1),
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
        n = n or self.cfg.default_simulations
        seed = seed if seed is not None else self.cfg.random_seed
        rng = np.random.default_rng(seed)

        all_results: List[Dict[str, Any]] = []
        raw_stats: Dict[str, Dict[str, List[float]]] = {}

        for sim_idx in range(n):
            child_rng = np.random.default_rng(rng.integers(0, 2**63))
            game_result = self.simulate_game(child_rng)
            all_results.append({
                "sim": sim_idx,
                "home_score": game_result["home_score"],
                "away_score": game_result["away_score"],
                "margin": game_result["margin"],
                "blowout": game_result["blowout"],
                "total_possessions": game_result["total_possessions"],
                "transition_possessions": game_result.get("transition_possessions", 0),
            })

            for pid, stats in game_result["stat_lines"].items():
                if pid not in raw_stats:
                    raw_stats[pid] = {k: [] for k in self.STAT_KEYS}
                for key in self.STAT_KEYS:
                    raw_stats[pid][key].append(stats.get(key, 0))

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
        return output.prob_over(player_id, stat, line)
