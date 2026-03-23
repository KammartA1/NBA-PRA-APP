"""
simulation/player_state.py
==========================
Per-player mutable state tracked through an entire simulated game.
Every stat category, fatigue, foul count, and dynamic efficiency are
maintained here and updated after each possession.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PlayerProfile:
    """Static attributes for a player (unchanged within a game)."""
    name: str
    player_id: str
    position: str                     # PG / SG / SF / PF / C
    age: int = 27
    height_inches: int = 78           # 6'6" default
    rest_days: int = 1                # days since last game

    # Individual shooting talent (0-1 scale relative to league avg)
    two_pt_pct: float = 0.525
    three_pt_pct: float = 0.362
    ft_pct: float = 0.775

    # Per-possession rates (personal tendencies)
    usage_rate: float = 0.20          # share of team possessions used
    assist_rate: float = 0.15         # P(assist | teammate makes FG while on court)
    rebound_rate: float = 0.10        # share of available rebounds grabbed
    steal_rate: float = 0.015         # P(steal) per defensive possession
    block_rate: float = 0.010         # P(block) per defensive possession
    turnover_rate: float = 0.12       # P(turnover) when using possession
    foul_rate: float = 0.03           # P(committing a foul) per possession
    foul_draw_rate: float = 0.03      # P(drawing a foul) per possession

    # Flags
    is_starter: bool = True
    rotation_order: int = 0           # 0-4 starters, 5-9+ bench


@dataclass
class PlayerState:
    """Mutable game state for a single player.  Reset at game start."""
    profile: PlayerProfile

    # --- Counting stats ---
    minutes_played: float = 0.0
    points: int = 0
    rebounds: int = 0
    offensive_rebounds: int = 0
    assists: int = 0
    steals: int = 0
    blocks: int = 0
    turnovers: int = 0
    personal_fouls: int = 0

    # Shot tracking
    fga: int = 0
    fgm: int = 0
    three_pa: int = 0
    three_pm: int = 0
    fta: int = 0
    ftm: int = 0

    # --- Dynamic state ---
    fatigue_level: float = 0.0        # 0.0 (fresh) → 1.0 (exhausted)
    is_on_court: bool = False
    is_fouled_out: bool = False
    current_efficiency_modifier: float = 1.0  # multiplier on shooting %
    current_usage_rate: float = 0.0   # dynamic usage (changes with lineup)

    # --- Per-quarter foul tracking ---
    fouls_by_quarter: list = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])

    # --- Bench tracking ---
    bench_time_accumulated: float = 0.0  # minutes on bench this stint

    @property
    def pra(self) -> int:
        """Points + Rebounds + Assists."""
        return self.points + self.rebounds + self.assists

    @property
    def pr(self) -> int:
        return self.points + self.rebounds

    @property
    def pa(self) -> int:
        return self.points + self.assists

    @property
    def ra(self) -> int:
        return self.rebounds + self.assists

    def update_efficiency(self) -> None:
        """Recalculate the efficiency modifier from current fatigue."""
        # Linear decay: at fatigue=0 → 1.0, at fatigue=1.0 → 0.70
        self.current_efficiency_modifier = 1.0 - 0.30 * self.fatigue_level

    def add_minutes(self, minutes: float) -> None:
        """Add minutes played and reset bench accumulator."""
        self.minutes_played += minutes
        self.bench_time_accumulated = 0.0

    def add_bench_time(self, minutes: float) -> None:
        """Track bench rest time."""
        self.bench_time_accumulated += minutes

    def record_two_pt_attempt(self, made: bool) -> None:
        self.fga += 1
        if made:
            self.fgm += 1
            self.points += 2

    def record_three_pt_attempt(self, made: bool) -> None:
        self.fga += 1
        self.three_pa += 1
        if made:
            self.fgm += 1
            self.three_pm += 1
            self.points += 3

    def record_free_throws(self, attempts: int, makes: int) -> None:
        self.fta += attempts
        self.ftm += makes
        self.points += makes

    def record_rebound(self, offensive: bool = False) -> None:
        self.rebounds += 1
        if offensive:
            self.offensive_rebounds += 1

    def record_assist(self) -> None:
        self.assists += 1

    def record_steal(self) -> None:
        self.steals += 1

    def record_block(self) -> None:
        self.blocks += 1

    def record_turnover(self) -> None:
        self.turnovers += 1

    def record_foul(self, quarter: int) -> bool:
        """Record a personal foul.  Returns True if player fouled out."""
        self.personal_fouls += 1
        q_idx = min(quarter, len(self.fouls_by_quarter) - 1)
        self.fouls_by_quarter[q_idx] += 1
        if self.personal_fouls >= 6:
            self.is_fouled_out = True
        return self.is_fouled_out

    def to_stat_line(self) -> dict:
        """Return a dictionary of final stats."""
        return {
            "player": self.profile.name,
            "player_id": self.profile.player_id,
            "position": self.profile.position,
            "minutes": round(self.minutes_played, 1),
            "points": self.points,
            "rebounds": self.rebounds,
            "assists": self.assists,
            "steals": self.steals,
            "blocks": self.blocks,
            "turnovers": self.turnovers,
            "fouls": self.personal_fouls,
            "fgm": self.fgm,
            "fga": self.fga,
            "three_pm": self.three_pm,
            "three_pa": self.three_pa,
            "ftm": self.ftm,
            "fta": self.fta,
            "pra": self.pra,
            "pr": self.pr,
            "pa": self.pa,
            "ra": self.ra,
            "fatigue_final": round(self.fatigue_level, 3),
            "fouled_out": self.is_fouled_out,
        }
