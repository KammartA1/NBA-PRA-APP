"""
simulation/game_script.py
=========================
Score-differential-driven coaching decisions.  The game script model
determines rotation aggressiveness, usage adjustments, timeout calls,
and end-of-game intentional fouling based on the current margin and
time remaining.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from simulation.config import SimulationConfig, DEFAULT_CONFIG


class GamePhase(enum.Enum):
    CLOSE = "close"            # within 5 points
    MODERATE_LEAD = "moderate_lead"
    MODERATE_DEFICIT = "moderate_deficit"
    BLOWOUT_LEAD = "blowout_lead"
    BLOWOUT_DEFICIT = "blowout_deficit"
    COMEBACK = "comeback"


@dataclass
class ScriptDecision:
    """Coaching decisions driven by the current game script."""
    phase: GamePhase
    starter_minutes_modifier: float    # multiplier on starter target minutes
    usage_boost: float                 # added to star usage rate (0 = normal)
    pull_starters: bool                # True → send starters to bench
    call_timeout: bool                 # True → coach should call TO
    intentional_foul: bool             # True → foul on defense to stop clock
    sub_trigger: bool                  # True → make a substitution now


class GameScriptModel:
    """Evaluate the current score differential and time remaining to
    produce coaching decisions that shape player minutes and usage."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG
        self._prev_deficit: Optional[int] = None
        self._max_deficit: int = 0

    def evaluate(
        self,
        own_score: int,
        opp_score: int,
        possession_number: int,
        total_possessions: int,
        unanswered_opponent_points: int,
        timeouts_remaining: int,
    ) -> ScriptDecision:
        """Produce a ``ScriptDecision`` given the current game state.

        Parameters
        ----------
        own_score, opp_score : current scores
        possession_number : 0-based current possession
        total_possessions : expected total possessions in the game
        unanswered_opponent_points : consecutive opponent points
        timeouts_remaining : team's remaining timeouts
        """
        margin = own_score - opp_score   # positive = leading
        abs_margin = abs(margin)
        game_progress = possession_number / max(total_possessions, 1)
        is_fourth_quarter = game_progress >= 0.75
        is_late_game = game_progress >= 0.90
        cfg = self.cfg

        # Track comeback
        deficit = -margin if margin < 0 else 0
        if deficit > self._max_deficit:
            self._max_deficit = deficit
        comeback = (
            self._max_deficit >= cfg.comeback_shrink_threshold
            and deficit < self._max_deficit - cfg.comeback_shrink_threshold
            and deficit > 0
        )
        self._prev_deficit = deficit

        # --- Determine phase ---
        if comeback and is_fourth_quarter:
            phase = GamePhase.COMEBACK
        elif abs_margin <= cfg.close_game_margin:
            phase = GamePhase.CLOSE
        elif margin > cfg.moderate_margin:
            phase = GamePhase.BLOWOUT_LEAD
        elif margin < -cfg.moderate_margin:
            phase = GamePhase.BLOWOUT_DEFICIT
        elif margin > 0:
            phase = GamePhase.MODERATE_LEAD
        else:
            phase = GamePhase.MODERATE_DEFICIT

        # --- Defaults ---
        starter_mod = 1.0
        usage_boost = 0.0
        pull_starters = False
        call_timeout = False
        intentional_foul = False
        sub_trigger = False

        # --- Phase-specific adjustments ---
        if phase == GamePhase.CLOSE:
            # Starters stay in longer, higher usage for stars
            starter_mod = 1.10
            if is_late_game:
                usage_boost = 0.05
                starter_mod = 1.20

        elif phase == GamePhase.MODERATE_LEAD:
            starter_mod = 1.0  # normal rotation

        elif phase == GamePhase.MODERATE_DEFICIT:
            starter_mod = 1.05
            if is_fourth_quarter:
                usage_boost = 0.03
                starter_mod = 1.10

        elif phase == GamePhase.BLOWOUT_LEAD:
            if is_fourth_quarter:
                pull_starters = True
                starter_mod = 0.80
            else:
                starter_mod = 0.95

        elif phase == GamePhase.BLOWOUT_DEFICIT:
            if is_fourth_quarter:
                pull_starters = True
                starter_mod = 0.80
            else:
                starter_mod = 1.0

        elif phase == GamePhase.COMEBACK:
            # Starters back in, high usage
            starter_mod = 1.15
            usage_boost = 0.05
            sub_trigger = True

        # --- Timeout logic ---
        if (
            unanswered_opponent_points >= cfg.timeout_run_threshold
            and timeouts_remaining > 0
        ):
            call_timeout = True

        # --- Intentional fouling (trailing, very late) ---
        if is_late_game and margin < 0 and abs_margin <= 6 and game_progress >= 0.95:
            intentional_foul = True

        return ScriptDecision(
            phase=phase,
            starter_minutes_modifier=starter_mod,
            usage_boost=usage_boost,
            pull_starters=pull_starters,
            call_timeout=call_timeout,
            intentional_foul=intentional_foul,
            sub_trigger=sub_trigger,
        )

    def reset(self) -> None:
        """Reset state between simulations."""
        self._prev_deficit = None
        self._max_deficit = 0
