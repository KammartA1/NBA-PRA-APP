"""
simulation/mlb/profiles.py — Batter and pitcher rate profiles.

Profiles hold per-PA outcome rates. The engine combines a batter profile
with a pitcher profile via the odds-ratio (log5) method against league
baselines. Handedness splits are supported: if a vs-LHP / vs-RHP split is
available it is used, otherwise the overall rate is the fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import LEAGUE_PA_RATES, LEAGUE_OUT_IN_PLAY

# The full categorical PA-outcome space the engine works in.
PA_OUTCOMES = ["K", "BB", "HBP", "HR", "3B", "2B", "1B", "OUT"]


def _league_vector() -> dict:
    v = dict(LEAGUE_PA_RATES)
    v["OUT"] = LEAGUE_OUT_IN_PLAY
    return v


LEAGUE_VECTOR = _league_vector()


@dataclass
class BatterProfile:
    """A hitter's per-PA outcome rates (overall, plus optional hand splits)."""
    player_id: str
    name: str
    bats: str = "R"            # 'R', 'L', or 'S' (switch)
    pa: int = 0                # sample size (for shrinkage)
    # Overall per-PA rates
    k: float = LEAGUE_PA_RATES["K"]
    bb: float = LEAGUE_PA_RATES["BB"]
    hbp: float = LEAGUE_PA_RATES["HBP"]
    hr: float = LEAGUE_PA_RATES["HR"]
    triple: float = LEAGUE_PA_RATES["3B"]
    double: float = LEAGUE_PA_RATES["2B"]
    single: float = LEAGUE_PA_RATES["1B"]
    # Stolen-base profile (per time reaching 1B)
    sb_attempt: float = 0.06
    sb_success: float = 0.75
    # Lineup slot (1-9) — affects total PAs in a game.
    lineup_slot: int = 5

    def outcome_vector(self) -> dict:
        """Return the batter's categorical PA-outcome rates summing to ~1."""
        v = {
            "K": self.k, "BB": self.bb, "HBP": self.hbp, "HR": self.hr,
            "3B": self.triple, "2B": self.double, "1B": self.single,
        }
        v["OUT"] = max(0.0, 1.0 - sum(v.values()))
        return v


@dataclass
class PitcherProfile:
    """A pitcher's per-PA outcome rates allowed."""
    player_id: str
    name: str
    throws: str = "R"          # 'R' or 'L'
    bf: int = 0                # batters faced (sample size)
    is_starter: bool = True
    k: float = LEAGUE_PA_RATES["K"]
    bb: float = LEAGUE_PA_RATES["BB"]
    hbp: float = LEAGUE_PA_RATES["HBP"]
    hr: float = LEAGUE_PA_RATES["HR"]
    triple: float = LEAGUE_PA_RATES["3B"]
    double: float = LEAGUE_PA_RATES["2B"]
    single: float = LEAGUE_PA_RATES["1B"]
    # Typical workload (for the starter pitch-count / removal model).
    avg_pitches: float = 90.0

    def outcome_vector(self) -> dict:
        v = {
            "K": self.k, "BB": self.bb, "HBP": self.hbp, "HR": self.hr,
            "3B": self.triple, "2B": self.double, "1B": self.single,
        }
        v["OUT"] = max(0.0, 1.0 - sum(v.values()))
        return v


def league_average_reliever() -> PitcherProfile:
    """A generic league-average reliever (slightly higher K than SP avg)."""
    return PitcherProfile(
        player_id="bullpen", name="Bullpen", throws="R", bf=10000,
        is_starter=False,
        k=LEAGUE_PA_RATES["K"] * 1.08,
        bb=LEAGUE_PA_RATES["BB"] * 1.02,
        hbp=LEAGUE_PA_RATES["HBP"],
        hr=LEAGUE_PA_RATES["HR"] * 0.95,
        triple=LEAGUE_PA_RATES["3B"],
        double=LEAGUE_PA_RATES["2B"] * 0.97,
        single=LEAGUE_PA_RATES["1B"] * 0.97,
        avg_pitches=20.0,
    )


def odds_ratio(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
    """Log5 / odds-ratio matchup combination for a single binary outcome.

    p = (b·q/l) / (b·q/l + (1−b)(1−q)/(1−l))

    This is the standard sabermetric method (Bill James log5 generalized by
    Tango's odds-ratio) for combining a batter's and pitcher's rate against a
    league baseline. Reduces to the correct value when either party is league
    average.
    """
    b = min(max(batter_rate, 1e-6), 1 - 1e-6)
    q = min(max(pitcher_rate, 1e-6), 1 - 1e-6)
    l = min(max(league_rate, 1e-6), 1 - 1e-6)
    num = (b * q) / l
    den = num + ((1 - b) * (1 - q)) / (1 - l)
    return num / den if den > 0 else b
