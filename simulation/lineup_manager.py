"""
simulation/lineup_manager.py
============================
Dynamic lineup and substitution modelling.  Manages rotation patterns,
substitution triggers (fatigue, foul trouble, game script, scheduled
rest), usage redistribution when a star sits, and coach-specific
rotation depths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation.config import CoachArchetype, SimulationConfig, DEFAULT_CONFIG
from simulation.fatigue_model import FatigueModel
from simulation.foul_model import FoulModel
from simulation.game_script import ScriptDecision, GamePhase
from simulation.player_state import PlayerState
from simulation.team_state import TeamState


@dataclass
class SubstitutionEvent:
    """Record of a single substitution."""
    possession: int
    player_out_idx: int
    player_in_idx: int
    reason: str   # "rotation", "fatigue", "foul_trouble", "blowout", "game_script"


class LineupManager:
    """Orchestrate substitutions and rotation patterns for one team."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        archetype: CoachArchetype | None = None,
    ) -> None:
        self.cfg = config or DEFAULT_CONFIG
        self.archetype = archetype or self.cfg.coach_archetype
        self.rotation_depth = self.cfg.rotation_depth[self.archetype]
        self.starter_target = self.cfg.starter_target_minutes[self.archetype]
        self.bench_target = self.cfg.bench_target_minutes[self.archetype]
        self.fatigue_model = FatigueModel(self.cfg)
        self.foul_model = FoulModel(self.cfg)
        self.sub_log: List[SubstitutionEvent] = []

        # Track stint lengths (possessions since last sub for each player)
        self._stint_start: Dict[int, int] = {}

    def initialize(self, team: TeamState) -> None:
        """Called at game start to set up stint tracking."""
        for idx in team.current_lineup:
            self._stint_start[idx] = 0

    def _minutes_per_possession(self, total_possessions: int) -> float:
        """Approximate game-minutes per possession."""
        return 48.0 / max(total_possessions, 1)

    def _should_rotate(
        self,
        player: PlayerState,
        player_idx: int,
        possession: int,
        total_possessions: int,
        quarter: int,
    ) -> bool:
        """Check if a player's scheduled rotation calls for a sub."""
        min_per_poss = self._minutes_per_possession(total_possessions)
        stint_start = self._stint_start.get(player_idx, 0)
        stint_possessions = possession - stint_start
        stint_minutes = stint_possessions * min_per_poss

        if player.profile.is_starter:
            # Starters play ~8 min blocks, then rest ~4 min
            return stint_minutes >= self.cfg.starter_block_minutes
        else:
            # Bench players play shorter stints
            return stint_minutes >= self.cfg.bench_block_minutes

    def _find_best_replacement(
        self,
        team: TeamState,
        player_out_idx: int,
        exclude: set | None = None,
    ) -> Optional[int]:
        """Find the best bench player to replace the outgoing player.

        Prefers same position, lowest fatigue, not fouled out, within
        rotation depth.
        """
        exclude = exclude or set()
        out_pos = team.players[player_out_idx].profile.position
        out_order = team.players[player_out_idx].profile.rotation_order

        candidates = []
        for idx in team.bench:
            p = team.players[idx]
            if p.is_fouled_out or idx in exclude:
                continue
            if p.profile.rotation_order >= self.rotation_depth:
                continue  # outside rotation
            # Score: prefer same position, lower fatigue, lower rotation order
            pos_match = 1.0 if p.profile.position == out_pos else 0.0
            fatigue_score = 1.0 - p.fatigue_level
            order_score = 1.0 / (1.0 + p.profile.rotation_order)
            score = pos_match * 3.0 + fatigue_score * 2.0 + order_score
            candidates.append((idx, score))

        if not candidates:
            # Expand to anyone not fouled out
            for idx in team.bench:
                p = team.players[idx]
                if not p.is_fouled_out and idx not in exclude:
                    candidates.append((idx, 1.0 - p.fatigue_level))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def process_substitutions(
        self,
        team: TeamState,
        possession: int,
        total_possessions: int,
        quarter: int,
        script_decision: ScriptDecision,
        is_blowout: bool = False,
    ) -> List[SubstitutionEvent]:
        """Evaluate and execute substitutions for one possession.

        Returns list of SubstitutionEvent that occurred.
        """
        events: List[SubstitutionEvent] = []
        subs_to_make: List[Tuple[int, str]] = []  # (player_out_idx, reason)
        exclude_in: set = set()

        # --- 1. Fouled-out players must be replaced immediately ---
        for idx in list(team.current_lineup):
            p = team.players[idx]
            if p.is_fouled_out:
                subs_to_make.append((idx, "fouled_out"))

        # --- 2. Foul trouble (not fouled out, but coach pulls) ---
        for idx in list(team.current_lineup):
            p = team.players[idx]
            if not p.is_fouled_out and self.foul_model.should_bench_for_fouls(p, quarter):
                if idx not in [s[0] for s in subs_to_make]:
                    subs_to_make.append((idx, "foul_trouble"))

        # --- 3. Blowout: pull starters ---
        if is_blowout or script_decision.pull_starters:
            for idx in list(team.current_lineup):
                p = team.players[idx]
                if p.profile.is_starter and idx not in [s[0] for s in subs_to_make]:
                    subs_to_make.append((idx, "blowout"))

        # --- 4. Fatigue threshold ---
        fatigue_threshold = 0.65
        for idx in list(team.current_lineup):
            p = team.players[idx]
            if p.fatigue_level >= fatigue_threshold:
                if idx not in [s[0] for s in subs_to_make]:
                    subs_to_make.append((idx, "fatigue"))

        # --- 5. Scheduled rotation (only at natural break points) ---
        is_quarter_break = possession > 0 and (
            possession % (total_possessions // 4) == 0
        )
        is_sub_window = possession > 0 and possession % 8 == 0  # check every ~8 poss

        if is_sub_window and not subs_to_make:
            for idx in list(team.current_lineup):
                p = team.players[idx]
                if self._should_rotate(p, idx, possession, total_possessions, quarter):
                    if idx not in [s[0] for s in subs_to_make]:
                        subs_to_make.append((idx, "rotation"))
                        break  # one rotation sub at a time normally

        # --- 6. Game script triggered substitution ---
        if script_decision.sub_trigger and not subs_to_make:
            # Bring back rested starters
            for idx in team.bench:
                p = team.players[idx]
                if p.profile.is_starter and not p.is_fouled_out and p.fatigue_level < 0.4:
                    # Find a non-starter on court to pull
                    for court_idx in team.current_lineup:
                        cp = team.players[court_idx]
                        if not cp.profile.is_starter:
                            subs_to_make.append((court_idx, "game_script"))
                            break
                    break

        # --- Execute subs ---
        for out_idx, reason in subs_to_make:
            if out_idx not in team.current_lineup:
                continue
            in_idx = self._find_best_replacement(team, out_idx, exclude_in)
            if in_idx is None:
                continue
            team.substitute(out_idx, in_idx)
            exclude_in.add(in_idx)
            self._stint_start[in_idx] = possession
            event = SubstitutionEvent(
                possession=possession,
                player_out_idx=out_idx,
                player_in_idx=in_idx,
                reason=reason,
            )
            events.append(event)
            self.sub_log.append(event)

        # Update dynamic usage rates after subs
        if events:
            team.update_dynamic_usage()

        return events

    def redistribute_usage(
        self,
        team: TeamState,
        absent_star_idx: Optional[int] = None,
    ) -> None:
        """When a high-usage player sits, redistribute their usage to
        on-court teammates proportionally."""
        if absent_star_idx is None:
            team.update_dynamic_usage()
            return

        star_usage = team.players[absent_star_idx].profile.usage_rate
        redistributable = star_usage * self.cfg.usage_redistribution_factor

        on_court = team.current_lineup
        remaining_usage = [team.players[i].profile.usage_rate for i in on_court]
        total_remaining = sum(remaining_usage)

        for i, idx in enumerate(on_court):
            base = team.players[idx].profile.usage_rate
            share = (base / total_remaining) if total_remaining > 0 else 0.2
            team.players[idx].current_usage_rate = base + redistributable * share

        # Re-normalize to sum = 1
        total = sum(team.players[i].current_usage_rate for i in on_court)
        if total > 0:
            for idx in on_court:
                team.players[idx].current_usage_rate /= total

    def reset(self) -> None:
        """Reset between simulations."""
        self.sub_log.clear()
        self._stint_start.clear()
