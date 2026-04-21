"""
simulation/fatigue_model.py
===========================
Continuous fatigue model.  Fatigue is a function of minutes played,
rest days, age, and game pace.  It reduces shooting efficiency, rebound
rate, and assist rate.  Recovery during bench time is partial and
non-linear: fatigue accelerates sharply after 30+ minutes of play.

Extended with:
- Pre-game fatigue from back-to-back, travel, and schedule density
- Altitude effects (Denver, Utah) that compound with heavy minutes
- Travel fatigue (jet lag, flight tiredness)
- Schedule density penalties (4-in-5, 3-in-4)
- Non-linear bench recovery (sqrt curve, altitude-dampened)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from simulation.config import SimulationConfig, DEFAULT_CONFIG
from simulation.player_state import PlayerState


@dataclass
class FatigueResult:
    """Output of a fatigue computation for one player on one possession."""
    fatigue_level: float               # 0.0 – 1.0
    efficiency_multiplier: float       # 1.0 = fresh, ~0.70 at max fatigue
    rebound_multiplier: float
    assist_multiplier: float
    pre_game_fatigue: float = 0.0      # starting fatigue from B2B / travel / etc.


class FatigueModel:
    """Compute and apply fatigue effects to player state."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Pre-game fatigue initialization
    # ------------------------------------------------------------------

    def compute_starting_fatigue(
        self,
        player: PlayerState,
        context_fatigue_start: float = 0.0,
    ) -> float:
        """Compute pre-game fatigue for *player* based on context brain
        inputs (back-to-back, travel, schedule density, etc.).

        Older players (age > 27) feel pre-game fatigue more acutely.
        The result is capped at 0.35 so a player never starts a game
        in an unrecoverable fatigue hole.

        Parameters
        ----------
        player : PlayerState
            The player whose starting fatigue is being computed.
        context_fatigue_start : float
            Pre-computed starting fatigue from the context brain
            (aggregates B2B penalty, travel penalty, etc.).  Defaults
            to 0.0 (fully rested).

        Returns
        -------
        float
            Starting fatigue level, clamped to [0.0, 0.35].
        """
        if context_fatigue_start <= 0.0:
            return 0.0

        age = player.profile.age
        age_penalty = max(0.0, (age - 27)) * 0.008 * context_fatigue_start

        return min(context_fatigue_start + age_penalty, 0.35)

    # ------------------------------------------------------------------
    # Core fatigue calculation
    # ------------------------------------------------------------------

    def compute_fatigue(
        self,
        player: PlayerState,
        team_pace: float,
        *,
        altitude_factor: float = 0.0,
        travel_fatigue: float = 0.0,
        schedule_density_factor: float = 0.0,
    ) -> float:
        """Return the updated fatigue level for *player* given current
        minutes played, rest days, age, team pace, and contextual
        factors.

        The function is non-linear: after ``fatigue_acceleration_threshold``
        minutes the fatigue rate doubles (configurable).

        Parameters
        ----------
        player : PlayerState
            The player to evaluate.
        team_pace : float
            Team's pace factor (possessions per 48 min).
        altitude_factor : float
            Venue altitude penalty (e.g. Denver=0.20, Utah=0.12).
            Increases fatigue accumulation rate and compounds with
            heavy minutes past the acceleration threshold.
        travel_fatigue : float
            Flat fatigue bump from travel (jet lag, flight tiredness).
        schedule_density_factor : float
            Multiplicative penalty for compressed schedules (4-in-5,
            3-in-4).  Applied to the base fatigue rate.
        """
        minutes = player.minutes_played
        age = player.profile.age
        rest = player.profile.rest_days
        cfg = self.cfg

        # Effective base rate adjusted for schedule density
        effective_base_rate = cfg.fatigue_base_rate * (1.0 + schedule_density_factor)

        # Base fatigue from minutes (with altitude compounding)
        if minutes <= cfg.fatigue_acceleration_threshold:
            # Pre-threshold: altitude applies uniformly
            base = minutes * effective_base_rate * (1.0 + altitude_factor)
        else:
            # Pre-threshold portion
            base = (
                cfg.fatigue_acceleration_threshold
                * effective_base_rate
                * (1.0 + altitude_factor)
            )
            # Post-threshold portion (accelerated)
            # Altitude effect is STRONGER after the threshold
            excess = minutes - cfg.fatigue_acceleration_threshold
            altitude_post_threshold = altitude_factor * 1.5
            base += (
                excess
                * effective_base_rate
                * cfg.fatigue_acceleration_factor
                * (1.0 + altitude_post_threshold)
            )

        # Age adjustment (older players tire faster)
        age_adj = max(0.0, (age - 25)) * cfg.fatigue_age_factor * minutes

        # Pace adjustment (faster pace = more fatigue)
        pace_adj = (
            max(0.0, (team_pace - 100.0))
            * cfg.fatigue_pace_factor
            * (minutes / 48.0)
        )

        # Rest day bonus (more rest = less fatigue, up to 3 days)
        rest_bonus = min(rest, 3) * cfg.fatigue_rest_day_bonus

        # Travel fatigue: flat bump added to base calculation
        fatigue = base + age_adj + pace_adj + travel_fatigue - rest_bonus
        return max(0.0, min(fatigue, cfg.fatigue_max))

    # ------------------------------------------------------------------
    # Bench recovery
    # ------------------------------------------------------------------

    def apply_bench_recovery(
        self,
        player: PlayerState,
        *,
        altitude_factor: float = 0.0,
    ) -> float:
        """Reduce fatigue for a player currently on the bench.

        Recovery is **non-linear**: the first few minutes on the bench
        recover more than later minutes (diminishing returns).  Uses a
        power-law curve: ``bench_minutes^0.7 * recovery_rate``.

        At altitude, recovery is dampened because the thinner air makes
        it harder to catch your breath on the bench.

        Parameters
        ----------
        player : PlayerState
            The benched player.
        altitude_factor : float
            Venue altitude factor; reduces recovery effectiveness.

        Returns
        -------
        float
            New fatigue level after bench recovery.
        """
        bench_minutes = player.bench_time_accumulated
        if bench_minutes <= 0.0:
            return player.fatigue_level

        # Non-linear recovery: sqrt-ish curve (exponent 0.7)
        # 2 min → 2^0.7 ≈ 1.62, 4 min → 4^0.7 ≈ 2.64  →  ratio ≈ 0.61
        # So 2 minutes gives ~61% of what 4 minutes would give.
        recovery = math.pow(bench_minutes, 0.7) * self.cfg.fatigue_bench_recovery_rate

        # Altitude dampens recovery (thinner air = harder to recover)
        if altitude_factor > 0.0:
            recovery *= (1.0 - altitude_factor * 0.3)

        new_fatigue = max(0.0, player.fatigue_level - recovery)
        return new_fatigue

    # ------------------------------------------------------------------
    # Efficiency multipliers
    # ------------------------------------------------------------------

    def get_efficiency_multiplier(self, fatigue: float) -> float:
        """Shooting / scoring efficiency multiplier.
        1.0 at fatigue=0, 0.70 at fatigue=1.0 (30% reduction).
        """
        return 1.0 - 0.30 * fatigue

    def get_rebound_multiplier(self, fatigue: float) -> float:
        """Rebounding drops less sharply: 1.0 → 0.80 at max fatigue."""
        return 1.0 - 0.20 * fatigue

    def get_assist_multiplier(self, fatigue: float) -> float:
        """Assist rate drops moderately: 1.0 → 0.85 at max fatigue."""
        return 1.0 - 0.15 * fatigue

    # ------------------------------------------------------------------
    # Full update
    # ------------------------------------------------------------------

    def update_player_fatigue(
        self,
        player: PlayerState,
        team_pace: float,
        minutes_this_possession: float,
        *,
        altitude_factor: float = 0.0,
        travel_fatigue: float = 0.0,
        schedule_density_factor: float = 0.0,
        pre_game_fatigue: float = 0.0,
    ) -> FatigueResult:
        """Compute fatigue, update player state, and return multipliers.

        Parameters
        ----------
        player : PlayerState
            The player to update.
        team_pace : float
            Team's pace factor (possessions per 48 min).
        minutes_this_possession : float
            Clock time consumed by this possession.
        altitude_factor : float
            Venue altitude penalty passed from the context brain.
        travel_fatigue : float
            Flat fatigue bump from travel / timezone crossing.
        schedule_density_factor : float
            Multiplicative penalty for compressed schedules.
        pre_game_fatigue : float
            Starting fatigue computed by ``compute_starting_fatigue``.
            Tracked for reporting; the actual pre-game fatigue should
            already be set on ``player.fatigue_level`` before the game
            loop starts.
        """
        if player.is_on_court:
            player.add_minutes(minutes_this_possession)
            fatigue = self.compute_fatigue(
                player,
                team_pace,
                altitude_factor=altitude_factor,
                travel_fatigue=travel_fatigue,
                schedule_density_factor=schedule_density_factor,
            )
        else:
            player.add_bench_time(minutes_this_possession)
            fatigue = self.apply_bench_recovery(
                player,
                altitude_factor=altitude_factor,
            )

        player.fatigue_level = fatigue
        player.update_efficiency()

        return FatigueResult(
            fatigue_level=fatigue,
            efficiency_multiplier=self.get_efficiency_multiplier(fatigue),
            rebound_multiplier=self.get_rebound_multiplier(fatigue),
            assist_multiplier=self.get_assist_multiplier(fatigue),
            pre_game_fatigue=pre_game_fatigue,
        )
