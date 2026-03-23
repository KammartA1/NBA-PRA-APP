"""
simulation/fatigue_model.py
===========================
Continuous fatigue model.  Fatigue is a function of minutes played,
rest days, age, and game pace.  It reduces shooting efficiency, rebound
rate, and assist rate.  Recovery during bench time is partial and
non-linear: fatigue accelerates sharply after 30+ minutes of play.
"""

from __future__ import annotations

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


class FatigueModel:
    """Compute and apply fatigue effects to player state."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.cfg = config or DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Core fatigue calculation
    # ------------------------------------------------------------------

    def compute_fatigue(self, player: PlayerState, team_pace: float) -> float:
        """Return the updated fatigue level for *player* given current
        minutes played, rest days, age, and team pace.

        The function is non-linear: after ``fatigue_acceleration_threshold``
        minutes the fatigue rate doubles (configurable).
        """
        minutes = player.minutes_played
        age = player.profile.age
        rest = player.profile.rest_days
        cfg = self.cfg

        # Base fatigue from minutes
        if minutes <= cfg.fatigue_acceleration_threshold:
            base = minutes * cfg.fatigue_base_rate
        else:
            # Pre-threshold portion
            base = cfg.fatigue_acceleration_threshold * cfg.fatigue_base_rate
            # Post-threshold portion (accelerated)
            excess = minutes - cfg.fatigue_acceleration_threshold
            base += excess * cfg.fatigue_base_rate * cfg.fatigue_acceleration_factor

        # Age adjustment (older players tire faster)
        age_adj = max(0.0, (age - 25)) * cfg.fatigue_age_factor * minutes

        # Pace adjustment (faster pace = more fatigue)
        pace_adj = max(0.0, (team_pace - 100.0)) * cfg.fatigue_pace_factor * (minutes / 48.0)

        # Rest day bonus (more rest = less fatigue, up to 3 days)
        rest_bonus = min(rest, 3) * cfg.fatigue_rest_day_bonus

        fatigue = base + age_adj + pace_adj - rest_bonus
        return max(0.0, min(fatigue, cfg.fatigue_max))

    # ------------------------------------------------------------------
    # Bench recovery
    # ------------------------------------------------------------------

    def apply_bench_recovery(self, player: PlayerState) -> float:
        """Reduce fatigue for a player currently on the bench.
        Returns the new fatigue level.
        """
        bench_minutes = player.bench_time_accumulated
        recovery = bench_minutes * self.cfg.fatigue_bench_recovery_rate
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
        """
        if player.is_on_court:
            player.add_minutes(minutes_this_possession)
            fatigue = self.compute_fatigue(player, team_pace)
        else:
            player.add_bench_time(minutes_this_possession)
            fatigue = self.apply_bench_recovery(player)

        player.fatigue_level = fatigue
        player.update_efficiency()

        return FatigueResult(
            fatigue_level=fatigue,
            efficiency_multiplier=self.get_efficiency_multiplier(fatigue),
            rebound_multiplier=self.get_rebound_multiplier(fatigue),
            assist_multiplier=self.get_assist_multiplier(fatigue),
        )
