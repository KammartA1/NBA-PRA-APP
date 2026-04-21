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

    # Clutch performance
    clutch_rating: float = 0.0        # -1.0 (chokes) → 0 (neutral) → +1.0 (ice cold closer)

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

    # --- Half-game stat tracking (H1 = Q1+Q2, H2 = Q3+Q4) ---
    h1_points: int = 0
    h1_rebounds: int = 0
    h1_assists: int = 0
    h1_steals: int = 0
    h1_blocks: int = 0
    h1_turnovers: int = 0
    h1_fgm: int = 0
    h1_fga: int = 0
    h1_three_pm: int = 0
    h1_three_pa: int = 0
    h1_ftm: int = 0
    h1_fta: int = 0
    h1_minutes: float = 0.0
    h2_points: int = 0
    h2_rebounds: int = 0
    h2_assists: int = 0
    h2_steals: int = 0
    h2_blocks: int = 0
    h2_turnovers: int = 0
    h2_fgm: int = 0
    h2_fga: int = 0
    h2_three_pm: int = 0
    h2_three_pa: int = 0
    h2_ftm: int = 0
    h2_fta: int = 0
    h2_minutes: float = 0.0

    # --- Current half tracking (set externally by GameEngine) ---
    _current_half: int = 1   # 1 or 2; updated each possession by the engine

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

    # --- Half-game composite properties ---
    @property
    def h1_pra(self) -> int:
        return self.h1_points + self.h1_rebounds + self.h1_assists

    @property
    def h1_pr(self) -> int:
        return self.h1_points + self.h1_rebounds

    @property
    def h1_pa(self) -> int:
        return self.h1_points + self.h1_assists

    @property
    def h1_ra(self) -> int:
        return self.h1_rebounds + self.h1_assists

    @property
    def h2_pra(self) -> int:
        return self.h2_points + self.h2_rebounds + self.h2_assists

    @property
    def h2_pr(self) -> int:
        return self.h2_points + self.h2_rebounds

    @property
    def h2_pa(self) -> int:
        return self.h2_points + self.h2_assists

    @property
    def h2_ra(self) -> int:
        return self.h2_rebounds + self.h2_assists

    def update_efficiency(self) -> None:
        """Recalculate the efficiency modifier from current fatigue."""
        # Linear decay: at fatigue=0 → 1.0, at fatigue=1.0 → 0.70
        self.current_efficiency_modifier = 1.0 - 0.30 * self.fatigue_level

    def add_minutes(self, minutes: float) -> None:
        """Add minutes played and reset bench accumulator."""
        self.minutes_played += minutes
        if self._current_half == 1:
            self.h1_minutes += minutes
        else:
            self.h2_minutes += minutes
        self.bench_time_accumulated = 0.0

    def add_bench_time(self, minutes: float) -> None:
        """Track bench rest time."""
        self.bench_time_accumulated += minutes

    def record_two_pt_attempt(self, made: bool) -> None:
        self.fga += 1
        if self._current_half == 1:
            self.h1_fga += 1
        else:
            self.h2_fga += 1
        if made:
            self.fgm += 1
            self.points += 2
            if self._current_half == 1:
                self.h1_fgm += 1
                self.h1_points += 2
            else:
                self.h2_fgm += 1
                self.h2_points += 2

    def record_three_pt_attempt(self, made: bool) -> None:
        self.fga += 1
        self.three_pa += 1
        if self._current_half == 1:
            self.h1_fga += 1
            self.h1_three_pa += 1
        else:
            self.h2_fga += 1
            self.h2_three_pa += 1
        if made:
            self.fgm += 1
            self.three_pm += 1
            self.points += 3
            if self._current_half == 1:
                self.h1_fgm += 1
                self.h1_three_pm += 1
                self.h1_points += 3
            else:
                self.h2_fgm += 1
                self.h2_three_pm += 1
                self.h2_points += 3

    def record_free_throws(self, attempts: int, makes: int) -> None:
        self.fta += attempts
        self.ftm += makes
        self.points += makes
        if self._current_half == 1:
            self.h1_fta += attempts
            self.h1_ftm += makes
            self.h1_points += makes
        else:
            self.h2_fta += attempts
            self.h2_ftm += makes
            self.h2_points += makes

    def record_rebound(self, offensive: bool = False) -> None:
        self.rebounds += 1
        if offensive:
            self.offensive_rebounds += 1
        if self._current_half == 1:
            self.h1_rebounds += 1
        else:
            self.h2_rebounds += 1

    def record_assist(self) -> None:
        self.assists += 1
        if self._current_half == 1:
            self.h1_assists += 1
        else:
            self.h2_assists += 1

    def record_steal(self) -> None:
        self.steals += 1
        if self._current_half == 1:
            self.h1_steals += 1
        else:
            self.h2_steals += 1

    def record_block(self) -> None:
        self.blocks += 1
        if self._current_half == 1:
            self.h1_blocks += 1
        else:
            self.h2_blocks += 1

    def record_turnover(self) -> None:
        self.turnovers += 1
        if self._current_half == 1:
            self.h1_turnovers += 1
        else:
            self.h2_turnovers += 1

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
            # Half-game stats (H1 = Q1+Q2, H2 = Q3+Q4)
            "h1_points": self.h1_points,
            "h1_rebounds": self.h1_rebounds,
            "h1_assists": self.h1_assists,
            "h1_steals": self.h1_steals,
            "h1_blocks": self.h1_blocks,
            "h1_turnovers": self.h1_turnovers,
            "h1_fgm": self.h1_fgm,
            "h1_fga": self.h1_fga,
            "h1_three_pm": self.h1_three_pm,
            "h1_three_pa": self.h1_three_pa,
            "h1_ftm": self.h1_ftm,
            "h1_fta": self.h1_fta,
            "h1_minutes": round(self.h1_minutes, 1),
            "h1_pra": self.h1_pra,
            "h1_pr": self.h1_pr,
            "h1_pa": self.h1_pa,
            "h1_ra": self.h1_ra,
            "h2_points": self.h2_points,
            "h2_rebounds": self.h2_rebounds,
            "h2_assists": self.h2_assists,
            "h2_steals": self.h2_steals,
            "h2_blocks": self.h2_blocks,
            "h2_turnovers": self.h2_turnovers,
            "h2_fgm": self.h2_fgm,
            "h2_fga": self.h2_fga,
            "h2_three_pm": self.h2_three_pm,
            "h2_three_pa": self.h2_three_pa,
            "h2_ftm": self.h2_ftm,
            "h2_fta": self.h2_fta,
            "h2_minutes": round(self.h2_minutes, 1),
            "h2_pra": self.h2_pra,
            "h2_pr": self.h2_pr,
            "h2_pa": self.h2_pa,
            "h2_ra": self.h2_ra,
        }
