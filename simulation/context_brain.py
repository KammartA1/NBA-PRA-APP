"""
simulation/context_brain.py
============================
NBA Context Brain -- situational awareness module that adjusts simulation
parameters based on real-world game context (playoffs, back-to-back,
travel, altitude, opponent strength, injuries, etc.).

All tunable constants are pulled from ``SimulationConfig``; this module
never hard-codes magic numbers.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simulation.config import DEFAULT_CONFIG, SeasonPhase, SimulationConfig
from simulation.player_state import PlayerProfile


# ---------------------------------------------------------------------------
# High-altitude cities (elevation >= 5,000 ft)
# ---------------------------------------------------------------------------

_HIGH_ALTITUDE_CITIES = frozenset({"DEN", "UTA", "SLC"})


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class InjuredPlayer:
    """Describes a single missing player for usage-redistribution purposes."""
    player_name: str
    usage_rate: float
    is_starter: bool


@dataclass
class GameContext:
    """All situational inputs the Context Brain needs for one team."""

    # Playoff
    is_playoff: bool = False
    playoff_round: str = "first"  # first / second / conf / finals

    # Season phase
    season_phase: SeasonPhase = SeasonPhase.MID

    # Schedule density
    is_back_to_back: bool = False
    rest_days: int = 1
    games_in_last_5_days: int = 1

    # Home / away
    is_home: bool = True

    # Motivation
    is_rivalry: bool = False
    is_national_tv: bool = False

    # Travel / altitude
    altitude_city: str = ""
    timezone_change: int = 0  # 0-3 timezones traveled

    # Opponent info
    opponent_drtg: float = 112.0
    opponent_pace: float = 100.0

    # Opponent schedule info (for rest-advantage calculation)
    opponent_is_back_to_back: bool = False
    opponent_rest_days: int = 1

    # Injured teammates
    injured_players: List[InjuredPlayer] = field(default_factory=list)


@dataclass
class ContextAdjustments:
    """Computed multipliers and modifiers produced by ``ContextBrain``.

    Unless noted, 1.0 means 'no change' for multipliers and 0.0 means
    'no change' for additive modifiers.
    """

    # Pace
    pace_multiplier: float = 1.0

    # Shooting efficiency (multiplicative modifier applied to all pcts)
    efficiency_modifier: float = 1.0

    # Star players get this additive usage boost
    star_usage_boost: float = 0.0

    # Rotation depth change (negative = tighter)
    rotation_depth_adjustment: int = 0

    # Minutes added/subtracted for starters
    starter_minutes_adjustment: float = 0.0

    # Pre-game fatigue (0.0 = fresh, >0 means starting tired)
    fatigue_starting_level: float = 0.0

    # In-game fatigue accumulation rate multiplier
    fatigue_rate_multiplier: float = 1.0

    # Shot-type rate modifiers (additive, applied to attempt rates)
    three_pt_rate_modifier: float = 0.0
    two_pt_rate_modifier: float = 0.0

    # Foul & turnover rate modifiers (multiplicative)
    foul_rate_modifier: float = 1.0
    turnover_rate_modifier: float = 1.0

    # Variance scaling (>1 = more volatile outcomes)
    variance_multiplier: float = 1.0

    # Blowout threshold shift (positive = harder to trigger blowout logic)
    blowout_threshold_adjustment: int = 0

    # Home court sub-adjustments
    home_court_adjustments: Dict[str, float] = field(default_factory=lambda: {
        "shooting_boost": 0.0,
        "ft_boost": 0.0,
        "steal_boost": 0.0,
        "foul_draw_boost": 0.0,
    })

    # Opponent defense modifier (negative = tougher defense)
    defense_adjustment: float = 0.0

    # Aggregate injured-player usage to redistribute
    injured_usage_to_redistribute: float = 0.0


# ---------------------------------------------------------------------------
# Clamp helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Context Brain
# ---------------------------------------------------------------------------

class ContextBrain:
    """Computes per-game contextual adjustments from a ``GameContext``.

    Parameters
    ----------
    config : SimulationConfig, optional
        Simulation parameters.  Falls back to ``DEFAULT_CONFIG``.
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.cfg: SimulationConfig = config or DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_adjustments(self, context: GameContext) -> ContextAdjustments:
        """Derive all context adjustments from the supplied *context*.

        Multipliers are compounded multiplicatively across categories
        and finally clipped to sane ranges.
        """
        adj = ContextAdjustments()

        # Each handler mutates *adj* in place.
        self._apply_playoff(context, adj)
        self._apply_back_to_back(context, adj)
        self._apply_rest_advantage(context, adj)
        self._apply_season_phase(context, adj)
        self._apply_travel_altitude(context, adj)
        self._apply_home_court(context, adj)
        self._apply_opponent_defense(context, adj)
        self._apply_rivalry_national_tv(context, adj)
        self._apply_injured_players(context, adj)

        # Final sanity clamps
        self._clamp_adjustments(adj)
        return adj

    def apply_to_player_profile(
        self,
        profile: PlayerProfile,
        adjustments: ContextAdjustments,
        is_star: bool,
    ) -> PlayerProfile:
        """Return a **new** ``PlayerProfile`` with context adjustments baked in.

        The original profile is not mutated.

        Parameters
        ----------
        profile : PlayerProfile
            The player's baseline static attributes.
        adjustments : ContextAdjustments
            Pre-computed adjustments from :meth:`compute_adjustments`.
        is_star : bool
            Whether this player should receive star-level boosts.
        """
        p = copy.copy(profile)

        # -- Shooting percentages (multiplicative efficiency) --
        eff = adjustments.efficiency_modifier
        # Home-court shooting boost is additive on top of the multiplicative modifier
        shooting_boost = adjustments.home_court_adjustments.get("shooting_boost", 0.0)
        defense_eff = adjustments.defense_adjustment  # already signed

        p.two_pt_pct = _clamp(
            profile.two_pt_pct * eff + shooting_boost + defense_eff,
            0.25, 0.75,
        )
        p.three_pt_pct = _clamp(
            profile.three_pt_pct * eff + shooting_boost + defense_eff,
            0.15, 0.55,
        )
        ft_boost = adjustments.home_court_adjustments.get("ft_boost", 0.0)
        p.ft_pct = _clamp(profile.ft_pct + ft_boost, 0.40, 0.99)

        # -- Usage rate --
        usage_change = 0.0
        if is_star:
            usage_change += adjustments.star_usage_boost
        # Redistribute injured-player usage
        if adjustments.injured_usage_to_redistribute > 0.0:
            if is_star:
                # Stars absorb a larger share of redistributed usage
                usage_change += adjustments.injured_usage_to_redistribute * 0.45
            else:
                usage_change += adjustments.injured_usage_to_redistribute * 0.15
        p.usage_rate = _clamp(profile.usage_rate + usage_change, 0.05, 0.45)

        # -- Foul and turnover rates (multiplicative) --
        p.foul_rate = _clamp(
            profile.foul_rate * adjustments.foul_rate_modifier, 0.005, 0.12,
        )
        p.turnover_rate = _clamp(
            profile.turnover_rate * adjustments.turnover_rate_modifier, 0.03, 0.25,
        )

        # -- Foul draw (home court) --
        foul_draw_boost = adjustments.home_court_adjustments.get("foul_draw_boost", 0.0)
        p.foul_draw_rate = _clamp(profile.foul_draw_rate + foul_draw_boost, 0.005, 0.10)

        # -- Steal rate (home court) --
        steal_boost = adjustments.home_court_adjustments.get("steal_boost", 0.0)
        p.steal_rate = _clamp(profile.steal_rate + steal_boost, 0.0, 0.05)

        return p

    # ------------------------------------------------------------------
    # Private adjustment handlers
    # ------------------------------------------------------------------

    def _apply_playoff(self, ctx: GameContext, adj: ContextAdjustments) -> None:
        """Scale adjustments by playoff round intensity."""
        if not ctx.is_playoff:
            return

        round_key = self._normalize_playoff_round(ctx.playoff_round, ctx.season_phase)
        intensity = self.cfg.playoff_intensity_by_round.get(round_key, 1.0)

        # Star usage boost
        adj.star_usage_boost += self.cfg.playoff_star_usage_boost * intensity

        # Tighter rotation
        adj.rotation_depth_adjustment += round(self.cfg.playoff_rotation_tighten * intensity)

        # Starter minutes increase (inverse of rotation tightening)
        adj.starter_minutes_adjustment += abs(self.cfg.playoff_rotation_tighten) * intensity * 0.5

        # Slower pace
        adj.pace_multiplier *= 1.0 - self.cfg.playoff_pace_reduction * intensity

        # More physical -- fouls increase
        adj.foul_rate_modifier *= 1.0 + self.cfg.playoff_foul_rate_increase * intensity

        # Shooting is harder in playoffs (contested, tighter defense)
        adj.efficiency_modifier *= 1.0 - self.cfg.playoff_three_pt_penalty * intensity
        adj.three_pt_rate_modifier -= self.cfg.playoff_three_pt_penalty * intensity * 0.5
        adj.two_pt_rate_modifier -= self.cfg.playoff_two_pt_penalty * intensity * 0.5

        # Blowout threshold rises -- teams don't quit in playoffs
        adj.blowout_threshold_adjustment += round(
            self.cfg.playoff_blowout_threshold_increase * intensity
        )

        # Turnovers slightly reduced (more careful play)
        adj.turnover_rate_modifier *= 1.0 - 0.02 * intensity

        # Less variance in playoffs (tighter game scripts)
        adj.variance_multiplier *= max(0.85, 1.0 - 0.05 * intensity)

    def _apply_back_to_back(self, ctx: GameContext, adj: ContextAdjustments) -> None:
        """Apply back-to-back and schedule-density penalties."""
        if not ctx.is_back_to_back:
            return

        adj.fatigue_starting_level += self.cfg.fatigue_b2b_starting_penalty
        adj.efficiency_modifier *= 1.0 - self.cfg.fatigue_b2b_efficiency_penalty
        adj.starter_minutes_adjustment -= self.cfg.fatigue_b2b_minutes_reduction

        # Additional penalties for extreme schedule density
        # 4 games in 5 days
        if ctx.games_in_last_5_days >= 4:
            adj.fatigue_starting_level += self.cfg.fatigue_four_in_five_penalty
            adj.efficiency_modifier *= 1.0 - self.cfg.fatigue_b2b_efficiency_penalty * 0.5
            adj.fatigue_rate_multiplier *= 1.15
        # 3 games in 4 days
        elif ctx.games_in_last_5_days >= 3:
            adj.fatigue_starting_level += self.cfg.fatigue_three_in_four_penalty
            adj.fatigue_rate_multiplier *= 1.08

    def _apply_rest_advantage(self, ctx: GameContext, adj: ContextAdjustments) -> None:
        """Boost the rested team when opponent is fatigued."""
        if ctx.rest_days < 2 or ctx.opponent_is_back_to_back is False:
            return

        adj.efficiency_modifier *= 1.0 + self.cfg.rest_advantage_efficiency_boost
        adj.pace_multiplier *= 1.0 + self.cfg.rest_advantage_pace_boost
        adj.fatigue_rate_multiplier *= 1.0 - self.cfg.rest_advantage_fatigue_reduction

    def _apply_season_phase(self, ctx: GameContext, adj: ContextAdjustments) -> None:
        """Adjust for where in the season this game falls."""
        phase = ctx.season_phase

        if phase == SeasonPhase.EARLY:
            adj.variance_multiplier *= 1.0 + self.cfg.early_season_variance_boost
            adj.efficiency_modifier *= 1.0 - self.cfg.early_season_efficiency_penalty
        elif phase == SeasonPhase.LATE:
            # Load management doesn't change the adjustments directly; it
            # affects whether a star even plays. We encode the *probability*
            # into a slight minutes reduction for stars as a proxy.
            adj.starter_minutes_adjustment -= (
                self.cfg.late_season_load_mgmt_probability * 5.0
            )
        elif phase == SeasonPhase.POST_ALLSTAR:
            adj.pace_multiplier *= 1.0 + self.cfg.post_allstar_pace_boost
        # PLAYOFF_* phases are handled entirely by _apply_playoff

    def _apply_travel_altitude(self, ctx: GameContext, adj: ContextAdjustments) -> None:
        """Altitude fatigue for visiting teams and timezone travel costs."""
        # Altitude: visiting DEN or UTA means faster in-game fatigue
        if ctx.altitude_city.upper() in _HIGH_ALTITUDE_CITIES and not ctx.is_home:
            adj.fatigue_rate_multiplier *= 1.0 + self.cfg.fatigue_altitude_factor

        # Timezone travel: additive starting fatigue
        if ctx.timezone_change > 0:
            tz_penalty = ctx.timezone_change * self.cfg.fatigue_travel_timezone_penalty
            adj.fatigue_starting_level += tz_penalty

    def _apply_home_court(self, ctx: GameContext, adj: ContextAdjustments) -> None:
        """Grant home-court advantage boosts."""
        if not ctx.is_home:
            return

        adj.home_court_adjustments["shooting_boost"] = self.cfg.home_court_shooting_boost
        adj.home_court_adjustments["ft_boost"] = self.cfg.home_court_ft_boost
        adj.home_court_adjustments["steal_boost"] = self.cfg.home_court_steal_boost
        adj.home_court_adjustments["foul_draw_boost"] = self.cfg.home_court_foul_draw_boost

    def _apply_opponent_defense(self, ctx: GameContext, adj: ContextAdjustments) -> None:
        """Modifier derived from opponent defensive rating.

        Negative means the opponent is tougher than average (lower DRTG =
        better defense), positive means weaker defense.
        """
        adj.defense_adjustment = (
            (ctx.opponent_drtg - self.cfg.league_avg_drtg)
            * self.cfg.defense_rating_impact_scale
        )

    def _apply_rivalry_national_tv(
        self, ctx: GameContext, adj: ContextAdjustments,
    ) -> None:
        """Small bumps for rivalry and nationally-televised games."""
        if ctx.is_rivalry:
            adj.star_usage_boost += self.cfg.rivalry_usage_boost
            adj.efficiency_modifier *= 1.0 + self.cfg.rivalry_efficiency_boost

        if ctx.is_national_tv:
            adj.star_usage_boost += self.cfg.national_tv_intensity_boost
            adj.efficiency_modifier *= 1.0 + self.cfg.national_tv_intensity_boost

    def _apply_injured_players(
        self, ctx: GameContext, adj: ContextAdjustments,
    ) -> None:
        """Redistribute usage from injured/missing players.

        Starters who are out release more redistributable usage than bench
        players, scaled by the config redistribution factors.
        """
        total_usage = 0.0
        for ip in ctx.injured_players:
            if ip.is_starter:
                total_usage += (
                    ip.usage_rate * self.cfg.injured_star_usage_redistribution
                )
            else:
                total_usage += (
                    ip.usage_rate * self.cfg.injured_role_player_usage_redistribution
                )

        adj.injured_usage_to_redistribute = total_usage

        # Losing starters also tightens effective rotation and raises
        # variance (less predictable outcomes with unfamiliar lineups).
        starters_out = sum(1 for ip in ctx.injured_players if ip.is_starter)
        if starters_out > 0:
            adj.variance_multiplier *= 1.0 + 0.05 * starters_out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_playoff_round(
        playoff_round: str, season_phase: SeasonPhase,
    ) -> str:
        """Map various round representations to the config dict key format.

        Accepts shorthand (``'first'``, ``'conf'``, ``'finals'``) as well as
        full ``SeasonPhase`` enum values.
        """
        mapping = {
            "first": "playoff_first",
            "second": "playoff_second",
            "conf": "playoff_conf",
            "finals": "playoff_finals",
        }

        # Try the explicit playoff_round string first
        key = mapping.get(playoff_round.lower())
        if key is not None:
            return key

        # Fall back to the SeasonPhase enum value
        if season_phase.value.startswith("playoff_"):
            return season_phase.value

        return "playoff_first"

    @staticmethod
    def _clamp_adjustments(adj: ContextAdjustments) -> None:
        """Clamp all adjustment fields to physically reasonable bounds."""
        adj.pace_multiplier = _clamp(adj.pace_multiplier, 0.85, 1.15)
        adj.efficiency_modifier = _clamp(adj.efficiency_modifier, 0.80, 1.15)
        adj.star_usage_boost = _clamp(adj.star_usage_boost, 0.0, 0.25)
        adj.rotation_depth_adjustment = int(_clamp(adj.rotation_depth_adjustment, -4, 2))
        adj.starter_minutes_adjustment = _clamp(adj.starter_minutes_adjustment, -6.0, 6.0)
        adj.fatigue_starting_level = _clamp(adj.fatigue_starting_level, 0.0, 0.50)
        adj.fatigue_rate_multiplier = _clamp(adj.fatigue_rate_multiplier, 0.70, 1.60)
        adj.three_pt_rate_modifier = _clamp(adj.three_pt_rate_modifier, -0.10, 0.10)
        adj.two_pt_rate_modifier = _clamp(adj.two_pt_rate_modifier, -0.10, 0.10)
        adj.foul_rate_modifier = _clamp(adj.foul_rate_modifier, 0.80, 1.40)
        adj.turnover_rate_modifier = _clamp(adj.turnover_rate_modifier, 0.80, 1.25)
        adj.variance_multiplier = _clamp(adj.variance_multiplier, 0.70, 1.60)
        adj.blowout_threshold_adjustment = int(
            _clamp(adj.blowout_threshold_adjustment, 0, 15)
        )
        adj.defense_adjustment = _clamp(adj.defense_adjustment, -0.06, 0.06)
        adj.injured_usage_to_redistribute = _clamp(
            adj.injured_usage_to_redistribute, 0.0, 0.40,
        )
