"""
simulation/mlb/profiles.py — Batter and pitcher rate profiles.

Profiles hold per-PA outcome rates. The engine combines a batter profile
with a pitcher profile via the odds-ratio (log5) method against league
baselines. Platoon-aware: if vs-LHP / vs-RHP splits are available, the
engine uses the correct split based on the opposing pitcher/batter's hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import LEAGUE_PA_RATES, LEAGUE_OUT_IN_PLAY

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
    bats: str = "R"
    pa: int = 0
    k: float = LEAGUE_PA_RATES["K"]
    bb: float = LEAGUE_PA_RATES["BB"]
    hbp: float = LEAGUE_PA_RATES["HBP"]
    hr: float = LEAGUE_PA_RATES["HR"]
    triple: float = LEAGUE_PA_RATES["3B"]
    double: float = LEAGUE_PA_RATES["2B"]
    single: float = LEAGUE_PA_RATES["1B"]
    sb_attempt: float = 0.06
    sb_success: float = 0.75
    lineup_slot: int = 5
    # Platoon splits: dict with keys K/BB/HBP/HR/3B/2B/1B, or None.
    vs_l: Optional[dict] = None
    vs_r: Optional[dict] = None

    def outcome_vector(self) -> dict:
        v = {
            "K": self.k, "BB": self.bb, "HBP": self.hbp, "HR": self.hr,
            "3B": self.triple, "2B": self.double, "1B": self.single,
        }
        v["OUT"] = max(0.0, 1.0 - sum(v.values()))
        return v

    def outcome_vector_vs(self, pitcher_hand: str) -> dict:
        """Platoon-aware outcome vector. Uses the correct split if available.

        pitcher_hand: 'L' or 'R'. Switch hitters ('S') use the opposite-hand
        split (vs-R when facing RHP is actually the weaker matchup for switch
        hitters, matching real platoon behavior).
        """
        split = None
        if pitcher_hand == "L" and self.vs_l:
            split = self.vs_l
        elif pitcher_hand == "R" and self.vs_r:
            split = self.vs_r
        if not split:
            return self.outcome_vector()
        v = {
            "K": split.get("K", self.k),
            "BB": split.get("BB", self.bb),
            "HBP": split.get("HBP", self.hbp),
            "HR": split.get("HR", self.hr),
            "3B": split.get("3B", self.triple),
            "2B": split.get("2B", self.double),
            "1B": split.get("1B", self.single),
        }
        v["OUT"] = max(0.0, 1.0 - sum(v.values()))
        return v


@dataclass
class PitcherProfile:
    """A pitcher's per-PA outcome rates allowed."""
    player_id: str
    name: str
    throws: str = "R"
    bf: int = 0
    is_starter: bool = True
    k: float = LEAGUE_PA_RATES["K"]
    bb: float = LEAGUE_PA_RATES["BB"]
    hbp: float = LEAGUE_PA_RATES["HBP"]
    hr: float = LEAGUE_PA_RATES["HR"]
    triple: float = LEAGUE_PA_RATES["3B"]
    double: float = LEAGUE_PA_RATES["2B"]
    single: float = LEAGUE_PA_RATES["1B"]
    avg_pitches: float = 90.0
    # Platoon splits: rates allowed vs LHB and RHB.
    vs_lhb: Optional[dict] = None
    vs_rhb: Optional[dict] = None

    def outcome_vector(self) -> dict:
        v = {
            "K": self.k, "BB": self.bb, "HBP": self.hbp, "HR": self.hr,
            "3B": self.triple, "2B": self.double, "1B": self.single,
        }
        v["OUT"] = max(0.0, 1.0 - sum(v.values()))
        return v

    def outcome_vector_vs(self, batter_hand: str) -> dict:
        """Platoon-aware outcome vector vs a batter of the given handedness.

        batter_hand: 'L', 'R', or 'S'. Switch hitters are treated as the
        opposite hand (they bat from the side opposite the pitcher).
        """
        split = None
        effective_hand = batter_hand
        if batter_hand == "S":
            effective_hand = "R" if self.throws == "L" else "L"
        if effective_hand == "L" and self.vs_lhb:
            split = self.vs_lhb
        elif effective_hand == "R" and self.vs_rhb:
            split = self.vs_rhb
        if not split:
            return self.outcome_vector()
        v = {
            "K": split.get("K", self.k),
            "BB": split.get("BB", self.bb),
            "HBP": split.get("HBP", self.hbp),
            "HR": split.get("HR", self.hr),
            "3B": split.get("3B", self.triple),
            "2B": split.get("2B", self.double),
            "1B": split.get("1B", self.single),
        }
        v["OUT"] = max(0.0, 1.0 - sum(v.values()))
        return v


def league_average_reliever() -> PitcherProfile:
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
    """Log5 / odds-ratio matchup combination.

    p = (b*q/l) / (b*q/l + (1-b)(1-q)/(1-l))

    Standard sabermetric method (Bill James log5 / Tango odds-ratio) for
    combining a batter's and pitcher's rate against a league baseline.
    Equivalent to the Bradley-Terry paired comparison model.
    """
    b = min(max(batter_rate, 1e-6), 1 - 1e-6)
    q = min(max(pitcher_rate, 1e-6), 1 - 1e-6)
    l = min(max(league_rate, 1e-6), 1 - 1e-6)
    num = (b * q) / l
    den = num + ((1 - b) * (1 - q)) / (1 - l)
    return num / den if den > 0 else b
