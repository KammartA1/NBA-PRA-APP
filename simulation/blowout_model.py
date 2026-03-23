"""
simulation/blowout_model.py
===========================
Garbage-time detection.  Estimates the probability that the current game
is a blowout based on score differential, time remaining, and pre-game
spread.  When a blowout is detected starters are pulled and bench
players get extended minutes, truncating starter stat distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simulation.config import SimulationConfig, DEFAULT_CONFIG


@dataclass
class BlowoutAssessment:
    """Output of blowout evaluation at a single point in the game."""
    blowout_probability: float         # P(final margin >= blowout threshold)
    is_blowout: bool                   # True when probability crosses decision threshold
    pull_starters: bool                # coaching action
    garbage_time: bool                 # True = full garbage time (bench only)
    expected_final_margin: float       # estimated final point differential


class BlowoutModel:
    """Estimate blowout probability and drive garbage-time decisions.

    The core idea: given a current lead *L* with *R* possessions
    remaining, the probability that the trailing team closes the gap is
    modelled via a random-walk approximation (each remaining possession
    has an expected point differential drawn from the spread).
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

    def evaluate(
        self,
        margin: int,
        possession_number: int,
        total_possessions: int,
        pre_game_spread: float = 0.0,
    ) -> BlowoutAssessment:
        """Assess blowout status.

        Parameters
        ----------
        margin : int
            Current point differential (positive = team is leading).
        possession_number : int
            0-based possession index.
        total_possessions : int
            Expected total possessions in the game.
        pre_game_spread : float
            Vegas spread for the leading team (negative = favored).
            E.g., -7.5 means team is favored by 7.5.
        """
        remaining = max(total_possessions - possession_number, 1)
        abs_margin = abs(margin)
        game_fraction_remaining = remaining / max(total_possessions, 1)
        cfg = self.cfg

        # --- Random-walk model ---
        # Each remaining possession: expected differential ~ spread / total_poss
        # Variance per possession ~ 4 points^2 (empirical NBA)
        per_poss_mean = -pre_game_spread / max(total_possessions, 1)  # positive if leading team favored
        per_poss_var = 4.0

        # Expected final margin given current margin
        expected_swing = per_poss_mean * remaining
        expected_final_margin = margin + expected_swing

        # Std dev of the swing
        swing_std = math.sqrt(per_poss_var * remaining)

        # P(final margin >= blowout_threshold) using normal CDF approximation
        if swing_std > 0:
            z_blow = (abs_margin + expected_swing - cfg.blowout_threshold) / swing_std
            blowout_prob = _normal_cdf(z_blow)
        else:
            blowout_prob = 1.0 if abs_margin >= cfg.blowout_threshold else 0.0

        # If the team is trailing, the sign flips
        if margin < 0:
            # For trailing team, their "blowout" means they are getting blown out
            z_blow_trail = (abs_margin - expected_swing - cfg.blowout_threshold) / max(swing_std, 0.01)
            blowout_prob = _normal_cdf(z_blow_trail)

        # --- Decision thresholds ---
        # In the 4th quarter, declare blowout if P > 0.80
        is_fourth_q = (1.0 - game_fraction_remaining) >= 0.75
        past_check_possession = possession_number >= cfg.blowout_start_check_possession

        blowout_declared = (
            blowout_prob > 0.80
            and past_check_possession
            and abs_margin >= cfg.blowout_threshold
        )

        garbage_time = (
            blowout_prob > 0.90
            and is_fourth_q
            and abs_margin >= cfg.blowout_threshold + 5
        )

        pull_starters = blowout_declared or garbage_time

        return BlowoutAssessment(
            blowout_probability=blowout_prob,
            is_blowout=blowout_declared,
            pull_starters=pull_starters,
            garbage_time=garbage_time,
            expected_final_margin=expected_final_margin,
        )


def _normal_cdf(z: float) -> float:
    """Fast approximation of the standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
