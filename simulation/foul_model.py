"""
simulation/foul_model.py
========================
Foul accumulation model.  Per-possession foul probability varies by
player tendency, fatigue, and position.  Tracks offensive vs defensive
fouls, foul-drawing ability, and coach-initiated benchings at 5 fouls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from simulation.config import SimulationConfig, DEFAULT_CONFIG
from simulation.player_state import PlayerState


@dataclass
class FoulEvent:
    """Result of foul adjudication on a single possession."""
    foul_occurred: bool
    fouling_player_idx: Optional[int] = None
    fouled_player_idx: Optional[int] = None   # player who drew the foul
    is_offensive: bool = False
    free_throws_awarded: int = 0
    fouled_out: bool = False                  # did the fouling player foul out?
    in_bonus: bool = False


class FoulModel:
    """Determine foul events each possession and manage foul trouble."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

    def _base_foul_probability(self, player: PlayerState) -> float:
        """Per-possession probability that *player* commits a foul."""
        base = player.profile.foul_rate
        # Fatigue increases foul probability
        fatigue_boost = player.fatigue_level * self.cfg.foul_fatigue_multiplier
        return min(base + fatigue_boost * base, 0.25)  # cap at 25%

    def _draw_foul_weight(self, player: PlayerState) -> float:
        """How likely *player* is to draw a foul (higher = draws more)."""
        return player.profile.foul_draw_rate

    def evaluate_possession(
        self,
        offense_players: list[PlayerState],
        offense_indices: list[int],
        defense_players: list[PlayerState],
        defense_indices: list[int],
        quarter: int,
        rng: np.random.Generator,
        defense_in_bonus: bool = False,
    ) -> FoulEvent:
        """Run foul adjudication for one possession.

        Parameters
        ----------
        offense_players / defense_players : on-court PlayerState lists
        offense_indices / defense_indices : roster indices for those players
        quarter : current quarter (0-indexed internally, passed as 1-based)
        rng : numpy random generator
        defense_in_bonus : whether the offensive team has put the defense in bonus
                           (actually: whether the *defense's* team foul count
                           means the offense shoots FTs on non-shooting fouls)

        Returns
        -------
        FoulEvent
        """
        # --- Check for a defensive foul first (most common) ---
        for i, d_player in enumerate(defense_players):
            prob = self._base_foul_probability(d_player)
            if rng.random() < prob:
                # Defensive foul occurred — determine who drew it
                draw_weights = np.array(
                    [self._draw_foul_weight(op) for op in offense_players],
                    dtype=np.float64,
                )
                total_w = draw_weights.sum()
                if total_w > 0:
                    draw_weights /= total_w
                else:
                    draw_weights = np.full(len(offense_players), 0.2)
                drawer_local = rng.choice(len(offense_players), p=draw_weights)
                fouled_player_idx = offense_indices[drawer_local]

                # Record foul on defensive player
                d_idx = defense_indices[i]
                fouled_out = d_player.record_foul(quarter)

                # Free throws?
                fts = 0
                if defense_in_bonus:
                    fts = 2
                elif rng.random() < 0.35:
                    # Shooting foul
                    fts = 2 if rng.random() < 0.70 else 3  # 2-pt vs 3-pt shooting foul

                return FoulEvent(
                    foul_occurred=True,
                    fouling_player_idx=d_idx,
                    fouled_player_idx=fouled_player_idx,
                    is_offensive=False,
                    free_throws_awarded=fts,
                    fouled_out=fouled_out,
                    in_bonus=defense_in_bonus,
                )

        # --- Check for an offensive foul (less common) ---
        if rng.random() < self.cfg.offensive_foul_share * self.cfg.foul_rate_per_possession:
            # Pick an offensive player (weighted by usage — higher usage = more ball handling)
            usage = np.array(
                [op.current_usage_rate for op in offense_players], dtype=np.float64,
            )
            total_u = usage.sum()
            if total_u > 0:
                usage /= total_u
            else:
                usage = np.full(len(offense_players), 0.2)
            off_local = rng.choice(len(offense_players), p=usage)
            off_idx = offense_indices[off_local]
            fouled_out = offense_players[off_local].record_foul(quarter)

            return FoulEvent(
                foul_occurred=True,
                fouling_player_idx=off_idx,
                fouled_player_idx=None,
                is_offensive=True,
                free_throws_awarded=0,
                fouled_out=fouled_out,
            )

        return FoulEvent(foul_occurred=False)

    def should_bench_for_fouls(self, player: PlayerState, quarter: int) -> bool:
        """Return True if the coach should pull this player due to foul trouble.

        Logic:
        - 3+ fouls in 1st half → sit until 2nd half
        - 4 fouls before 4th quarter → sit until 4th quarter
        - 5 fouls any time → sit (unless <3 min left)
        """
        fouls = player.personal_fouls
        if fouls >= self.cfg.coach_pull_foul_count:
            return True
        if quarter <= 2 and fouls >= 3:
            return True
        if quarter == 3 and fouls >= 4:
            return True
        return False
