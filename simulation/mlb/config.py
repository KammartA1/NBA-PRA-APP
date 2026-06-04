"""
simulation/mlb/config.py — League constants, park factors, and engine config.

All baseline rates are real MLB league averages (2023-2024 seasons, per
plate appearance) used as the denominator in the log5 / odds-ratio matchup
method. Park factors are published 3-year (2022-2024) Statcast/FanGraphs
park factors normalized to 100 = neutral.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── League-average PA outcome rates (per plate appearance) ───────────
# Source: MLB league totals 2023-2024. These sum to ~1.0 and form the
# categorical PA-outcome distribution baseline for the odds-ratio method.
LEAGUE_PA_RATES = {
    "K":   0.2250,   # strikeout
    "BB":  0.0820,   # walk (unintentional + intentional)
    "HBP": 0.0110,   # hit by pitch
    "HR":  0.0320,   # home run
    "3B":  0.0045,   # triple
    "2B":  0.0470,   # double
    "1B":  0.1420,   # single
    # out_in_play (non-K outs: groundouts, flyouts, lineouts, etc.) is the
    # remainder so the categorical distribution always normalizes to 1.0.
}
LEAGUE_OUT_IN_PLAY = 1.0 - sum(LEAGUE_PA_RATES.values())  # ~0.4565

# League batting reference rates used for SB modeling and contact outs.
LEAGUE_SB_ATTEMPT_RATE = 0.085   # SB attempts per time-on-1B opportunity
LEAGUE_SB_SUCCESS_RATE = 0.785   # leaguewide SB success %

# Average pitches per plate appearance (for starter pitch-count proxy).
PITCHES_PER_PA = 3.9

# ── Park factors (100 = neutral). Index 0 = runs PF, 1 = HR PF. ──────
# Published multi-year park factors. >100 = hitter-friendly.
PARK_FACTORS = {
    "Coors Field":            (112, 116),  # COL — altitude
    "Great American Ball Park": (108, 121),# CIN
    "Fenway Park":            (108, 97),    # BOS
    "Globe Life Field":       (103, 104),  # TEX
    "Yankee Stadium":         (103, 120),  # NYY — short RF porch
    "Citizens Bank Park":     (103, 113),  # PHI
    "Wrigley Field":          (103, 104),  # CHC
    "Chase Field":            (102, 102),  # ARI
    "Camden Yards":           (101, 104),  # BAL
    "Truist Park":            (101, 103),  # ATL
    "Nationals Park":         (101, 101),  # WSH
    "Dodger Stadium":         (100, 110),  # LAD
    "Angel Stadium":          (100, 103),  # LAA
    "Rogers Centre":          (100, 102),  # TOR
    "Minute Maid Park":       (100, 101),  # HOU
    "Target Field":           ( 99, 100),  # MIN
    "PNC Park":               ( 99,  92),  # PIT
    "Progressive Field":      ( 99,  98),  # CLE
    "Busch Stadium":          ( 98,  92),  # STL
    "Citi Field":             ( 98,  97),  # NYM
    "Guaranteed Rate Field":  ( 98, 105),  # CWS
    "Kauffman Stadium":       ( 98,  90),  # KC — big outfield
    "American Family Field":  ( 98, 105),  # MIL
    "loanDepot park":         ( 97,  95),  # MIA
    "Comerica Park":          ( 97,  91),  # DET — deep CF
    "Petco Park":             ( 96,  96),  # SD
    "Oracle Park":            ( 95,  85),  # SF — deep RF, marine air
    "T-Mobile Park":          ( 95,  92),  # SEA — marine layer
    "Oakland Coliseum":       ( 93,  88),  # OAK — foul territory
    "Tropicana Field":        ( 96,  95),  # TB — dome
    "Sutter Health Park":     (101, 103),  # ATH (Sacramento, 2025)
    "George M. Steinbrenner Field": (102, 110),  # TB temp (2025)
}
NEUTRAL_PARK = (100, 100)


def park_factor(park_name: str) -> tuple[float, float]:
    """Return (runs_pf, hr_pf) as multipliers around 1.0 for a park name."""
    if not park_name:
        return (1.0, 1.0)
    runs, hr = PARK_FACTORS.get(park_name, NEUTRAL_PARK)
    return (runs / 100.0, hr / 100.0)


def weather_hr_factor(temp_f: float | None, wind_mph: float | None,
                      wind_out: bool | None) -> float:
    """HR multiplier from weather. Warm air + wind blowing out → more HR.

    ~1% HR distance-carry per degree above 70F is the established rule of
    thumb (Alan Nathan's ballistics work). Wind blowing out adds carry.
    Capped to a sane range so weather never dominates the projection.
    """
    factor = 1.0
    if temp_f is not None:
        factor *= 1.0 + (temp_f - 70.0) * 0.006
    if wind_mph is not None and wind_out is not None:
        if wind_out:
            factor *= 1.0 + min(wind_mph, 20.0) * 0.010
        else:
            factor *= 1.0 - min(wind_mph, 20.0) * 0.007
    return max(0.80, min(1.25, factor))


@dataclass
class MLBSimConfig:
    """Configuration for the MLB Monte Carlo game engine."""
    n_sims: int = 20000
    random_seed: int = 42
    # Starter removal: pull when EITHER threshold is exceeded.
    starter_max_pitches: float = 100.0
    starter_max_batters_faced: int = 27
    # Third-time-through-order penalty: K% drops / hit% rises each time a
    # starter faces the lineup. Applied as a small multiplicative drift.
    tto_penalty_per_pass: float = 0.04
    # Extra-innings cap to bound runtime.
    max_innings: int = 12
    # Reliever league-average rates (slightly better K, worse BB than SP avg).
    reliever_k_mult: float = 1.08
    reliever_bb_mult: float = 1.05
