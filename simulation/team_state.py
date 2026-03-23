"""
simulation/team_state.py
========================
Per-team mutable state for a simulated game: score, fouls, timeouts,
current lineup, bench, and possession tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from simulation.config import SimulationConfig, DEFAULT_CONFIG
from simulation.player_state import PlayerState


@dataclass
class TeamState:
    """Full mutable state for one team during a simulated game."""

    team_name: str
    players: List[PlayerState]         # full roster (index 0-N)
    pace_factor: float = 1.0           # team-specific pace multiplier

    # --- Score / game state ---
    score: int = 0
    possession_count: int = 0

    # --- Fouls & timeouts ---
    team_fouls_quarter: int = 0
    timeouts_remaining: int = 7        # full-game total (NBA rules)
    timeouts_used: int = 0
    last_timeout_score_diff: Optional[int] = None

    # --- Lineup ---
    current_lineup: List[int] = field(default_factory=list)   # indices into players
    bench: List[int] = field(default_factory=list)            # indices into players

    # --- Run tracking (for timeout decisions) ---
    unanswered_opponent_points: int = 0
    unanswered_own_points: int = 0

    def initialize_lineup(self) -> None:
        """Set starters on court and bench players off court."""
        starters = [
            i for i, p in enumerate(self.players) if p.profile.is_starter
        ]
        bench = [
            i for i, p in enumerate(self.players) if not p.profile.is_starter
        ]
        # Ensure exactly 5 starters; if fewer, fill from bench by rotation_order
        if len(starters) < 5:
            bench_sorted = sorted(bench, key=lambda i: self.players[i].profile.rotation_order)
            while len(starters) < 5 and bench_sorted:
                starters.append(bench_sorted.pop(0))
            bench = bench_sorted
        elif len(starters) > 5:
            starters_sorted = sorted(starters, key=lambda i: self.players[i].profile.rotation_order)
            starters = starters_sorted[:5]
            bench = [i for i in range(len(self.players)) if i not in starters]

        self.current_lineup = starters
        self.bench = bench

        for i in starters:
            self.players[i].is_on_court = True
        for i in bench:
            self.players[i].is_on_court = False

    def advance_possession(self) -> None:
        """Increment possession counter."""
        self.possession_count += 1

    def add_points(self, pts: int) -> None:
        """Add points and update run tracking."""
        self.score += pts
        self.unanswered_own_points += pts
        # Opponent's unanswered run resets on the defensive side (handled externally)

    def opponent_scored(self, pts: int) -> None:
        """Track opponent scoring run."""
        self.unanswered_opponent_points += pts
        self.unanswered_own_points = 0

    def call_timeout(self) -> bool:
        """Attempt to call a timeout.  Returns True if successful."""
        if self.timeouts_remaining > 0:
            self.timeouts_remaining -= 1
            self.timeouts_used += 1
            self.unanswered_opponent_points = 0
            return True
        return False

    def reset_quarter_fouls(self) -> None:
        """Reset team fouls at the start of a new quarter."""
        self.team_fouls_quarter = 0

    def add_team_foul(self) -> None:
        """Add one team foul for the current quarter."""
        self.team_fouls_quarter += 1

    def in_bonus(self) -> bool:
        """Return True if team is in the bonus (opponent has 5+ fouls)."""
        return self.team_fouls_quarter >= 5

    def substitute(self, out_idx: int, in_idx: int) -> None:
        """Swap a player out of the lineup and bring one in from bench."""
        if out_idx in self.current_lineup and in_idx in self.bench:
            self.current_lineup.remove(out_idx)
            self.current_lineup.append(in_idx)
            self.bench.remove(in_idx)
            self.bench.append(out_idx)
            self.players[out_idx].is_on_court = False
            self.players[in_idx].is_on_court = True
            self.players[in_idx].bench_time_accumulated = 0.0

    def get_on_court_players(self) -> List[PlayerState]:
        """Return list of PlayerState objects currently on court."""
        return [self.players[i] for i in self.current_lineup]

    def get_on_court_usage_rates(self) -> List[float]:
        """Return normalized usage rates for the current 5 on-court players."""
        raw = [self.players[i].profile.usage_rate for i in self.current_lineup]
        total = sum(raw)
        if total == 0:
            return [0.2] * 5
        return [r / total for r in raw]

    def update_dynamic_usage(self) -> None:
        """Set current_usage_rate on each on-court player (normalized to sum=1)."""
        rates = self.get_on_court_usage_rates()
        for idx, lineup_idx in enumerate(self.current_lineup):
            self.players[lineup_idx].current_usage_rate = rates[idx]
