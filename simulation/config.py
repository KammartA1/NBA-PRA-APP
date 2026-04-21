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
# Season phase enum
# ---------------------------------------------------------------------------

class SeasonPhase(enum.Enum):
    EARLY = "early"           # Games 1-20: rust, new lineups, higher variance
    MID = "mid"               # Games 21-60: baseline
    LATE = "late"             # Games 61-72: load management, tanking risk
    POST_ALLSTAR = "post_allstar"  # Post ASB: historically elevated pace
    PLAYOFF_FIRST = "playoff_first"    # 1st round
    PLAYOFF_SECOND = "playoff_second"  # 2nd round / conf semis
    PLAYOFF_CONF = "playoff_conf"      # Conference finals
    PLAYOFF_FINALS = "playoff_finals"  # NBA Finals


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
    timeouts_per_half: int = 4
    mandatory_timeout_quarter: int = 2
    timeout_run_threshold: int = 8

    # --- Pace ---
    league_avg_pace: float = 100.0
    pace_std: float = 4.0
    min_pace: float = 90.0
    max_pace: float = 112.0

    # --- Fatigue ---
    fatigue_base_rate: float = 0.015
    fatigue_acceleration_threshold: float = 30.0
    fatigue_acceleration_factor: float = 2.0
    fatigue_age_factor: float = 0.003
    fatigue_pace_factor: float = 0.001
    fatigue_bench_recovery_rate: float = 0.03
    fatigue_max: float = 1.0
    fatigue_rest_day_bonus: float = 0.02
    fatigue_sub_threshold: float = 0.65

    # --- Fatigue: back-to-back and travel ---
    fatigue_b2b_starting_penalty: float = 0.15
    fatigue_b2b_efficiency_penalty: float = 0.04
    fatigue_b2b_minutes_reduction: float = 2.5
    fatigue_altitude_factor: float = 0.20
    fatigue_travel_timezone_penalty: float = 0.01
    fatigue_four_in_five_penalty: float = 0.10
    fatigue_three_in_four_penalty: float = 0.06

    # --- Shooting / possession outcomes (league averages) ---
    two_pt_attempt_rate: float = 0.42
    three_pt_attempt_rate: float = 0.34
    free_throw_rate: float = 0.12
    turnover_rate: float = 0.12

    league_avg_two_pt_pct: float = 0.545
    league_avg_three_pt_pct: float = 0.367
    league_avg_ft_pct: float = 0.780

    and_one_probability: float = 0.025
    offensive_rebound_rate: float = 0.27

    # --- Assist / steal / block league averages (per possession) ---
    assist_rate_on_made_fg: float = 0.60
    steal_rate_per_possession: float = 0.04
    block_rate_per_possession: float = 0.025

    # --- Rebounds ---
    rebound_per_missed_shot: float = 1.0
    defensive_rebound_share: float = 0.73

    # --- Fouls ---
    foul_rate_per_possession: float = 0.10
    foul_fatigue_multiplier: float = 0.3
    foul_limit: int = 6
    coach_pull_foul_count: int = 5
    offensive_foul_share: float = 0.15
    fts_per_foul: float = 1.8

    # --- Blowout thresholds ---
    blowout_threshold: int = 20
    blowout_quarter_trigger: int = 4
    blowout_start_check_possession: int = 150
    historical_blowout_rate: float = 0.15

    # --- Game script thresholds ---
    close_game_margin: int = 5
    moderate_margin: int = 15
    comeback_shrink_threshold: int = 8

    # --- Game script: clutch mode ---
    clutch_game_progress: float = 0.90
    clutch_margin: int = 5
    clutch_star_usage_boost: float = 0.12
    clutch_role_player_usage_penalty: float = 0.04
    clutch_efficiency_variance: float = 0.03

    # --- Game script: dynamic pace ---
    trailing_pace_boost_per_point: float = 0.003
    trailing_pace_boost_max: float = 0.08
    leading_pace_slow_per_point: float = 0.002
    leading_pace_slow_max: float = 0.06
    q4_close_pace_boost: float = 0.03

    # --- Game script: shot selection shift ---
    trailing_three_pt_boost_per_point: float = 0.008
    trailing_three_pt_boost_max: float = 0.15
    leading_paint_boost: float = 0.06
    leading_ft_draw_boost: float = 0.03

    # --- Game script: two-for-one ---
    two_for_one_window_seconds: float = 35.0
    two_for_one_probability: float = 0.70
    two_for_one_three_pt_bias: float = 0.15

    # --- Game script: end of quarter heave ---
    end_quarter_heave_probability: float = 0.10
    heave_three_pt_pct: float = 0.03

    # --- Game script: momentum / hot hand ---
    hot_hand_streak_threshold: int = 3
    hot_hand_usage_boost: float = 0.04
    hot_hand_efficiency_boost: float = 0.025
    cold_streak_threshold: int = 4
    cold_streak_usage_penalty: float = 0.03
    momentum_run_threshold: int = 8
    momentum_timeout_trigger: int = 10

    # --- Game script: garbage time gradient ---
    garbage_time_margin_per_minute: float = 3.5
    garbage_time_bench_scale_start: float = 0.4
    garbage_time_bench_scale_full: float = 0.85

    # --- Transition play ---
    transition_rate_after_turnover: float = 0.55
    transition_rate_after_made_shot: float = 0.08
    transition_rate_after_defensive_rebound: float = 0.18
    transition_efg: float = 0.58
    halfcourt_efg: float = 0.50
    transition_and_one_boost: float = 0.015

    # --- Opponent defense ---
    defense_rating_impact_scale: float = 0.003
    league_avg_drtg: float = 112.0
    position_defense_weight: float = 0.35
    team_defense_weight: float = 0.65

    # --- Home court advantage ---
    home_court_shooting_boost: float = 0.015
    home_court_ft_boost: float = 0.02
    home_court_steal_boost: float = 0.005
    home_court_foul_draw_boost: float = 0.005
    home_court_expected_margin: float = 3.5

    # --- Usage redistribution ---
    star_usage_rate: float = 0.30
    role_player_usage_rate: float = 0.15
    bench_player_usage_rate: float = 0.10
    usage_redistribution_factor: float = 0.6

    # --- Coaching rotation defaults ---
    coach_archetype: CoachArchetype = CoachArchetype.BALANCED
    rotation_depth: Dict[CoachArchetype, int] = field(default_factory=lambda: {
        CoachArchetype.STARTER_HEAVY: 7,
        CoachArchetype.BALANCED: 9,
        CoachArchetype.DEEP_BENCH: 10,
    })
    starter_block_minutes: float = 8.0
    bench_block_minutes: float = 4.0
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

    # --- Context Brain: playoff adjustments ---
    playoff_star_usage_boost: float = 0.10
    playoff_rotation_tighten: int = -2
    playoff_pace_reduction: float = 0.03
    playoff_foul_rate_increase: float = 0.15
    playoff_three_pt_penalty: float = 0.02
    playoff_two_pt_penalty: float = 0.01
    playoff_blowout_threshold_increase: int = 5
    playoff_intensity_by_round: Dict[str, float] = field(default_factory=lambda: {
        "playoff_first": 1.0,
        "playoff_second": 1.15,
        "playoff_conf": 1.30,
        "playoff_finals": 1.50,
    })

    # --- Context Brain: season phase adjustments ---
    early_season_variance_boost: float = 0.15
    early_season_efficiency_penalty: float = 0.02
    late_season_load_mgmt_probability: float = 0.08
    post_allstar_pace_boost: float = 0.015

    # --- Context Brain: rest advantage ---
    rest_advantage_efficiency_boost: float = 0.03
    rest_advantage_pace_boost: float = 0.02
    rest_advantage_fatigue_reduction: float = 0.15

    # --- Context Brain: rivalry / motivation ---
    rivalry_usage_boost: float = 0.02
    rivalry_efficiency_boost: float = 0.01
    national_tv_intensity_boost: float = 0.01

    # --- Player variance ---
    default_variance_multiplier: float = 1.0
    boom_bust_variance_multiplier: float = 1.35
    consistent_variance_multiplier: float = 0.75

    # --- Injury teammate impact ---
    injured_star_usage_redistribution: float = 0.70
    injured_role_player_usage_redistribution: float = 0.50

    # --- Minutes restrictions ---
    minutes_restriction_detection_std: float = 2.0
    minutes_restriction_cap_buffer: float = 1.5

    # --- Position indices ---
    positions: List[str] = field(default_factory=lambda: [
        "PG", "SG", "SF", "PF", "C",
    ])


# Singleton default config
DEFAULT_CONFIG = SimulationConfig()
