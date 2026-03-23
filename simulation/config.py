"""
simulation/config.py
====================
All simulation parameters as frozen dataclasses.  Every tunable constant
lives here so the rest of the package never hard-codes magic numbers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Coaching archetypes
# ---------------------------------------------------------------------------

class CoachArchetype(enum.Enum):
    STARTER_HEAVY = "starter_heavy"   # 7-man rotation, starters 36+ min
    BALANCED = "balanced"             # 8-9 man rotation, starters 32-34 min
    DEEP_BENCH = "deep_bench"         # 10-man rotation, starters 28-30 min


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    """Central parameter store for the possession-level simulator."""

    # --- Core ---
    possessions_per_game: int = 220
    default_simulations: int = 10_000
    random_seed: int = 42

    # --- Quarter / clock ---
    quarter_length_minutes: float = 12.0
    overtime_length_minutes: float = 5.0
    max_overtimes: int = 4
    num_quarters: int = 4

    # --- Timeouts ---
    timeouts_per_half: int = 4         # Each team gets 4 full + extras
    mandatory_timeout_quarter: int = 2  # TV timeout threshold
    timeout_run_threshold: int = 8     # Call TO after opponent scores 8+ unanswered

    # --- Pace ---
    league_avg_pace: float = 100.0     # possessions per 48 minutes (team)
    pace_std: float = 4.0
    min_pace: float = 90.0
    max_pace: float = 112.0

    # --- Fatigue ---
    fatigue_base_rate: float = 0.015            # per minute played
    fatigue_acceleration_threshold: float = 30.0 # minutes after which fatigue accelerates
    fatigue_acceleration_factor: float = 2.0     # multiplier on rate after threshold
    fatigue_age_factor: float = 0.003            # additional per year above 25
    fatigue_pace_factor: float = 0.001           # additional per pace unit above 100
    fatigue_bench_recovery_rate: float = 0.03    # recovery per minute on bench
    fatigue_max: float = 1.0
    fatigue_rest_day_bonus: float = 0.02         # less base fatigue per rest day (up to 3)

    # --- Shooting / possession outcomes (league averages) ---
    # These rates must sum to ~1.0 (turnover + 2pt + 3pt + ft)
    two_pt_attempt_rate: float = 0.42    # of all possessions
    three_pt_attempt_rate: float = 0.34
    free_throw_rate: float = 0.12        # possessions ending in FTs (non-and-one)
    turnover_rate: float = 0.12          # per possession (NBA ~12-13%)

    league_avg_two_pt_pct: float = 0.545
    league_avg_three_pt_pct: float = 0.367
    league_avg_ft_pct: float = 0.780

    and_one_probability: float = 0.025   # given a made 2pt/3pt
    offensive_rebound_rate: float = 0.27

    # --- Assist / steal / block league averages (per possession) ---
    assist_rate_on_made_fg: float = 0.60
    steal_rate_per_possession: float = 0.04
    block_rate_per_possession: float = 0.025

    # --- Rebounds ---
    rebound_per_missed_shot: float = 1.0   # always 1 rebound per miss
    defensive_rebound_share: float = 0.73

    # --- Fouls ---
    foul_rate_per_possession: float = 0.10
    foul_fatigue_multiplier: float = 0.3   # extra foul rate per unit fatigue
    foul_limit: int = 6
    coach_pull_foul_count: int = 5         # sit player at this many fouls
    offensive_foul_share: float = 0.15     # fraction of fouls that are offensive
    fts_per_foul: float = 1.8             # average free throws awarded per foul

    # --- Blowout thresholds ---
    blowout_threshold: int = 20
    blowout_quarter_trigger: int = 4       # check blowout in 4th quarter
    blowout_start_check_possession: int = 150  # earliest possession to check
    historical_blowout_rate: float = 0.15

    # --- Game script thresholds ---
    close_game_margin: int = 5
    moderate_margin: int = 15
    comeback_shrink_threshold: int = 8     # deficit shrank by this much → starters back

    # --- Usage redistribution ---
    star_usage_rate: float = 0.30
    role_player_usage_rate: float = 0.15
    bench_player_usage_rate: float = 0.10
    usage_redistribution_factor: float = 0.6  # fraction of star usage that redistributes

    # --- Coaching rotation defaults ---
    coach_archetype: CoachArchetype = CoachArchetype.BALANCED
    rotation_depth: Dict[CoachArchetype, int] = field(default_factory=lambda: {
        CoachArchetype.STARTER_HEAVY: 7,
        CoachArchetype.BALANCED: 9,
        CoachArchetype.DEEP_BENCH: 10,
    })
    starter_block_minutes: float = 8.0     # starters play ~8 min blocks
    bench_block_minutes: float = 4.0       # then sit ~4 min
    starter_target_minutes: Dict[CoachArchetype, float] = field(default_factory=lambda: {
        CoachArchetype.STARTER_HEAVY: 36.0,
        CoachArchetype.BALANCED: 33.0,
        CoachArchetype.DEEP_BENCH: 29.0,
    })
    bench_target_minutes: Dict[CoachArchetype, float] = field(default_factory=lambda: {
        CoachArchetype.STARTER_HEAVY: 12.0,
        CoachArchetype.BALANCED: 18.0,
        CoachArchetype.DEEP_BENCH: 22.0,
    })

    # --- Position indices ---
    positions: List[str] = field(default_factory=lambda: [
        "PG", "SG", "SF", "PF", "C",
    ])


# Singleton default config
DEFAULT_CONFIG = SimulationConfig()
