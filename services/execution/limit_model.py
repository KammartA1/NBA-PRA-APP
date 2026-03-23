"""
services/execution/limit_model.py
==================================
Models sportsbook limiting behavior.

Sportsbooks limit winning bettors. After N winning bets on a book, max bet
size decreases. This module simulates the impact: if limited to $50/bet on
sharp books, what happens to expected profit?
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BookLimitState:
    """Current limiting status for a single sportsbook."""
    book_name: str
    current_max_bet: float          # Current max bet allowed ($)
    original_max_bet: float         # Original max bet before limits ($)
    limit_pct: float                # current / original (1.0 = unlimited)
    total_bets_placed: int
    total_wins: int
    win_rate: float
    net_pnl: float                  # Lifetime P&L on this book
    first_bet_date: Optional[datetime] = None
    last_bet_date: Optional[datetime] = None
    is_limited: bool = False
    estimated_bets_to_limit: int = 0  # Remaining bets before next limit tier


@dataclass
class LimitImpact:
    """Impact of limiting on a single bet."""
    desired_stake: float
    allowed_stake: float            # After limit cap
    stake_reduction_pct: float      # How much stake was cut
    profit_impact: float            # Lost expected profit due to limit
    book_name: str
    is_capped: bool


@dataclass
class LimitProfile:
    """Aggregate limit status across all sportsbooks."""
    n_books_tracked: int
    n_books_limited: int
    n_books_unlimited: int
    book_states: List[BookLimitState]
    total_desired_action: float     # What we WANT to bet
    total_allowed_action: float     # What we CAN bet
    effective_capacity_pct: float   # allowed / desired
    estimated_monthly_profit_loss: float  # $ lost to limits per month
    worst_limited_book: str
    best_unlimited_book: str


class LimitModel:
    """Models sportsbook limiting and its impact on bankroll growth.

    Core insight: winning bettors get limited. The question is not IF but WHEN
    and HOW MUCH. This model tracks each book's limit trajectory and computes
    the profit erosion from reduced bet sizing.
    """

    # Default limit tiers: after N wins on a book, max bet drops
    # (cumulative_wins_threshold, max_bet_multiplier)
    DEFAULT_LIMIT_TIERS = [
        (0, 1.00),      # 0-9 wins: full limits
        (10, 0.80),     # 10-24 wins: 80% of max
        (25, 0.50),     # 25-49 wins: 50% of max
        (50, 0.25),     # 50-99 wins: 25% of max
        (100, 0.10),    # 100-199 wins: 10% of max
        (200, 0.05),    # 200+ wins: 5% of max (effectively limited out)
    ]

    # Default max bet by book tier
    DEFAULT_MAX_BETS = {
        "pinnacle": 5000.0,
        "betrivers": 500.0,
        "draftkings": 1000.0,
        "fanduel": 1000.0,
        "mgm": 500.0,
        "caesars": 500.0,
        "pointsbet": 250.0,
        "prizepicks": 100.0,
        "underdog": 100.0,
        "default": 500.0,
    }

    def __init__(
        self,
        sport: str = "nba",
        limit_tiers: Optional[list] = None,
        max_bets: Optional[dict] = None,
    ):
        self.sport = sport
        self.limit_tiers = limit_tiers or self.DEFAULT_LIMIT_TIERS
        self.max_bets = max_bets or self.DEFAULT_MAX_BETS
        # Track state per book: {book_name: {wins, losses, bets, pnl, ...}}
        self._book_history: Dict[str, dict] = {}
        # Manual overrides: {book_name: max_bet}
        self._manual_limits: Dict[str, float] = {}

    def set_manual_limit(self, book_name: str, max_bet: float) -> None:
        """Manually set a limit for a specific book (e.g., from known limiting)."""
        self._manual_limits[book_name.lower()] = max_bet
        log.info("Manual limit set: %s -> $%.2f", book_name, max_bet)

    def load_bet_history(self, bets: list) -> None:
        """Load bet history to build per-book state."""
        self._book_history.clear()
        for b in bets:
            if hasattr(b, "sportsbook"):
                book = getattr(b, "sportsbook", "unknown").lower()
                won = getattr(b, "won", None)
                pnl = getattr(b, "pnl", 0)
                stake = getattr(b, "stake", 0)
                ts = getattr(b, "timestamp", None)
            elif isinstance(b, dict):
                book = b.get("sportsbook", "unknown").lower()
                won = b.get("won")
                pnl = b.get("pnl", 0)
                stake = b.get("stake", 0)
                ts = b.get("timestamp")
            else:
                continue

            if book not in self._book_history:
                self._book_history[book] = {
                    "total_bets": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0.0,
                    "total_staked": 0.0,
                    "first_bet": ts,
                    "last_bet": ts,
                }

            state = self._book_history[book]
            state["total_bets"] += 1
            state["pnl"] += pnl
            state["total_staked"] += stake
            if won is True:
                state["wins"] += 1
            elif won is False:
                state["losses"] += 1
            if ts:
                if state["first_bet"] is None or ts < state["first_bet"]:
                    state["first_bet"] = ts
                if state["last_bet"] is None or ts > state["last_bet"]:
                    state["last_bet"] = ts

        log.info("LimitModel loaded history for %d books", len(self._book_history))

    def _get_original_max(self, book: str) -> float:
        """Get the original (pre-limit) max bet for a book."""
        bk = book.lower()
        for key, val in self.max_bets.items():
            if key in bk:
                return val
        return self.max_bets.get("default", 500.0)

    def _compute_limit_multiplier(self, wins: int) -> float:
        """Compute the limit multiplier based on cumulative wins."""
        multiplier = 1.0
        for threshold, mult in reversed(self.limit_tiers):
            if wins >= threshold:
                multiplier = mult
                break
        return multiplier

    def _next_limit_threshold(self, wins: int) -> int:
        """How many more wins until the next limit tier kicks in."""
        for threshold, _ in self.limit_tiers:
            if wins < threshold:
                return threshold - wins
        return 0  # Already at max limit

    def get_book_state(self, book_name: str) -> BookLimitState:
        """Get the current limit state for a specific book."""
        bk = book_name.lower()
        history = self._book_history.get(bk, {})
        wins = history.get("wins", 0)
        total = history.get("total_bets", 0)
        pnl = history.get("pnl", 0.0)

        original_max = self._get_original_max(bk)

        # Manual override takes precedence
        if bk in self._manual_limits:
            current_max = self._manual_limits[bk]
        else:
            multiplier = self._compute_limit_multiplier(wins)
            current_max = original_max * multiplier

        limit_pct = current_max / original_max if original_max > 0 else 1.0
        win_rate = wins / total if total > 0 else 0.0

        return BookLimitState(
            book_name=book_name,
            current_max_bet=current_max,
            original_max_bet=original_max,
            limit_pct=limit_pct,
            total_bets_placed=total,
            total_wins=wins,
            win_rate=win_rate,
            net_pnl=pnl,
            first_bet_date=history.get("first_bet"),
            last_bet_date=history.get("last_bet"),
            is_limited=limit_pct < 0.99,
            estimated_bets_to_limit=self._next_limit_threshold(wins),
        )

    def apply_limit(self, book_name: str, desired_stake: float) -> LimitImpact:
        """Apply limiting to a desired stake for a specific book."""
        state = self.get_book_state(book_name)
        allowed = min(desired_stake, state.current_max_bet)
        reduction = 1.0 - (allowed / desired_stake) if desired_stake > 0 else 0.0

        return LimitImpact(
            desired_stake=desired_stake,
            allowed_stake=allowed,
            stake_reduction_pct=reduction * 100,
            profit_impact=(desired_stake - allowed),
            book_name=book_name,
            is_capped=allowed < desired_stake,
        )

    def simulate_future_limits(
        self,
        book_name: str,
        bets_per_month: int = 30,
        win_rate: float = 0.55,
        months: int = 12,
    ) -> List[dict]:
        """Simulate future limiting trajectory for a book."""
        state = self.get_book_state(book_name)
        current_wins = state.total_wins
        original_max = state.original_max_bet
        trajectory = []

        for month in range(1, months + 1):
            new_wins = int(bets_per_month * win_rate)
            current_wins += new_wins
            multiplier = self._compute_limit_multiplier(current_wins)
            max_bet = original_max * multiplier

            trajectory.append({
                "month": month,
                "cumulative_wins": current_wins,
                "max_bet": max_bet,
                "limit_pct": multiplier * 100,
                "effective_action": max_bet * bets_per_month,
            })

        return trajectory

    def profile(self) -> LimitProfile:
        """Generate aggregate limit profile across all tracked books."""
        states = []
        for book in self._book_history:
            states.append(self.get_book_state(book))

        if not states:
            return LimitProfile(
                n_books_tracked=0,
                n_books_limited=0,
                n_books_unlimited=0,
                book_states=[],
                total_desired_action=0,
                total_allowed_action=0,
                effective_capacity_pct=100.0,
                estimated_monthly_profit_loss=0.0,
                worst_limited_book="none",
                best_unlimited_book="none",
            )

        limited = [s for s in states if s.is_limited]
        unlimited = [s for s in states if not s.is_limited]

        total_original = sum(s.original_max_bet for s in states)
        total_current = sum(s.current_max_bet for s in states)
        capacity = (total_current / total_original * 100) if total_original > 0 else 100.0

        worst = min(states, key=lambda s: s.limit_pct)
        best = max(states, key=lambda s: s.limit_pct)

        # Estimate monthly profit loss from limits
        monthly_loss = sum(
            (s.original_max_bet - s.current_max_bet) * 30 * 0.03  # 3% edge * 30 bets
            for s in limited
        )

        return LimitProfile(
            n_books_tracked=len(states),
            n_books_limited=len(limited),
            n_books_unlimited=len(unlimited),
            book_states=states,
            total_desired_action=total_original,
            total_allowed_action=total_current,
            effective_capacity_pct=capacity,
            estimated_monthly_profit_loss=monthly_loss,
            worst_limited_book=worst.book_name,
            best_unlimited_book=best.book_name,
        )
