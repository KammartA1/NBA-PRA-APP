"""Execution Realism Layer — Models what price you ACTUALLY get.

Most bettors ignore slippage, limits, and latency. This module models
the real-world execution costs that destroy most theoretical edges.

Usage:
    from quant_system.market.execution_realism import ExecutionRealismEngine

    engine = ExecutionRealismEngine()
    result = engine.simulate_execution(
        edge_pct=0.03, stake=100, book="draftkings"
    )
    if not result["edge_survives"]:
        print("Edge destroyed by execution costs!")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


class ExecutionRealismEngine:
    """Models what price you ACTUALLY get after slippage, limits, and latency.

    Most bettors ignore this. It destroys most edges.
    """

    def __init__(self):
        # Typical slippage by book (fraction of edge lost per bet)
        self.slippage_model = {
            "pinnacle": 0.001,       # 0.1% — very tight, sharp book
            "circa": 0.002,          # 0.2%
            "draftkings": 0.005,     # 0.5% — retail spread
            "fanduel": 0.005,        # 0.5%
            "betmgm": 0.008,         # 0.8%
            "caesars": 0.008,        # 0.8%
            "prizepicks": 0.01,      # 1.0% — DFS spread
        }

        # Typical max bet limits by book (dollars)
        self.limit_model = {
            "pinnacle": 5000,
            "circa": 2000,
            "draftkings": 500,       # Props often capped at $500
            "fanduel": 500,
            "betmgm": 250,
            "caesars": 250,
            "prizepicks": 100,       # PrizePicks cap
        }

        # Latency model: seconds from signal to execution
        self.latency_model = {
            "pinnacle": 2.0,         # API bet, fast
            "circa": 5.0,            # Kiosk / app
            "draftkings": 3.0,       # App
            "fanduel": 3.0,
            "betmgm": 4.0,
            "caesars": 4.0,
            "prizepicks": 5.0,       # Manual entry
        }

        # Edge decay per second of latency (for liquid markets)
        self.edge_decay_per_second = 0.001  # 0.1% per second

        # Sharp profile limit reduction schedule
        # After N winning bets, limits get cut to X% of original
        self.sharp_limit_schedule = [
            (20, 0.80),    # After 20 wins: 80% of limits
            (50, 0.50),    # After 50 wins: 50%
            (100, 0.25),   # After 100 wins: 25%
            (200, 0.10),   # After 200 wins: 10%
            (500, 0.02),   # After 500 wins: 2% (effectively limited out)
        ]

    # ── Core Execution Simulation ──────────────────────────────────────

    def simulate_execution(
        self,
        edge_pct: float,
        stake: float,
        book: str,
        is_sharp_profile: bool = False,
        latency_seconds: Optional[float] = None,
    ) -> dict:
        """Given theoretical edge, compute ACTUAL expected edge after costs.

        Args:
            edge_pct: Theoretical edge as decimal (e.g., 0.03 = 3%)
            stake: Desired stake in dollars
            book: Sportsbook name (lowercase)
            is_sharp_profile: Whether account is flagged as sharp
            latency_seconds: Override latency; uses default if None

        Returns:
            {
                "theoretical_edge": float,
                "actual_edge": float,
                "slippage_cost": float,
                "latency_cost": float,
                "max_executable_stake": float,
                "limit_adjusted_ev": float,
                "edge_survives": bool,
                "breakdown": dict,
            }
        """
        book_lower = book.lower()

        # Get book-specific parameters (default to conservative estimates)
        slippage = self.slippage_model.get(book_lower, 0.01)
        max_limit = self.limit_model.get(book_lower, 250)
        latency = latency_seconds or self.latency_model.get(book_lower, 5.0)

        # 1. Slippage cost: line moves between signal and execution
        slippage_cost = slippage

        # 2. Latency cost: edge decays while you place the bet
        latency_cost = latency * self.edge_decay_per_second

        # 3. Sharp profile penalty: reduced limits
        if is_sharp_profile:
            max_limit = int(max_limit * 0.25)  # Sharp profiles get 25% limits
            slippage_cost *= 1.5  # Lines move faster against known sharps

        # 4. Compute actual edge
        total_cost = slippage_cost + latency_cost
        actual_edge = edge_pct - total_cost

        # 5. Limit-adjusted stake
        executable_stake = min(stake, max_limit)

        # 6. Expected value
        theoretical_ev = edge_pct * stake
        actual_ev = actual_edge * executable_stake

        edge_survives = actual_edge > 0 and executable_stake > 0

        return {
            "theoretical_edge": round(edge_pct * 100, 3),
            "actual_edge": round(actual_edge * 100, 3),
            "slippage_cost": round(slippage_cost * 100, 3),
            "latency_cost": round(latency_cost * 100, 3),
            "total_execution_cost": round(total_cost * 100, 3),
            "desired_stake": round(stake, 2),
            "max_executable_stake": round(executable_stake, 2),
            "theoretical_ev": round(theoretical_ev, 2),
            "limit_adjusted_ev": round(actual_ev, 2),
            "edge_survives": edge_survives,
            "book": book_lower,
            "is_sharp_profile": is_sharp_profile,
            "breakdown": {
                "slippage_pct": round(slippage_cost * 100, 3),
                "latency_seconds": latency,
                "latency_pct": round(latency_cost * 100, 3),
                "limit_cap": max_limit,
                "stake_reduction": round(stake - executable_stake, 2),
                "ev_loss_from_costs": round(theoretical_ev - actual_ev, 2),
            },
        }

    # ── Market Reaction Model ──────────────────────────────────────────

    def model_market_reaction(
        self,
        win_rate: float,
        n_bets: int,
        avg_stake: float,
        book: str = "draftkings",
    ) -> dict:
        """Simulate how sportsbooks react to winning behavior.

        Models the reality that consistent winners get limited,
        and edge decays due to the bettor's own success.

        Args:
            win_rate: Historical win rate (e.g., 0.55)
            n_bets: Number of bets placed so far
            avg_stake: Average stake per bet

        Returns:
            {
                "months_until_limited": int,
                "limit_trajectory": list,
                "edge_after_limits": float,
                "system_survives_12_months": bool,
            }
        """
        book_lower = book.lower()
        initial_limit = self.limit_model.get(book_lower, 250)

        # Estimate how quickly the book catches on
        # Higher win rate = faster detection
        # Higher stakes = faster detection
        # More bets = more data for the book

        # Books typically review after ~50 settled bets
        # Sharp detection score (0-1): higher = faster limiting
        excess_win_rate = max(0, win_rate - 0.52)  # 52% is break-even at -110
        sharp_score = (
            excess_win_rate * 5.0 +                  # Win rate signal
            min(avg_stake / 500.0, 1.0) * 0.3 +     # Stake size signal
            min(n_bets / 200.0, 1.0) * 0.2           # Volume signal
        )
        sharp_score = min(sharp_score, 1.0)

        # Months until first limit reduction
        if sharp_score < 0.1:
            months_first_limit = 24  # Very low risk
        elif sharp_score < 0.3:
            months_first_limit = 12
        elif sharp_score < 0.5:
            months_first_limit = 6
        elif sharp_score < 0.7:
            months_first_limit = 3
        else:
            months_first_limit = 1

        # Build 12-month limit trajectory
        limit_trajectory = []
        current_limit = initial_limit

        for month in range(1, 13):
            if month >= months_first_limit:
                # Exponential decay after first limit
                months_since = month - months_first_limit
                decay = math.exp(-0.3 * sharp_score * months_since)
                current_limit = max(
                    initial_limit * decay,
                    initial_limit * 0.02  # Floor: 2% of original
                )
            limit_trajectory.append(round(current_limit, 2))

        # Edge after limits: can we still execute meaningful bets?
        final_limit = limit_trajectory[-1]
        edge_after_limits = (win_rate - 0.52) * (final_limit / max(avg_stake, 1.0))

        # Bets per month (estimate ~30 bets/month for active bettors)
        bets_per_month = 30
        monthly_ev_after_limits = edge_after_limits * final_limit * bets_per_month

        # System survives if we can still make >$50/month after 12 months
        system_survives = final_limit >= avg_stake * 0.25 and monthly_ev_after_limits > 50

        return {
            "win_rate": win_rate,
            "sharp_score": round(sharp_score, 3),
            "months_until_limited": months_first_limit,
            "initial_limit": initial_limit,
            "limit_trajectory": limit_trajectory,
            "final_limit_month_12": round(final_limit, 2),
            "limit_reduction_pct": round((1 - final_limit / initial_limit) * 100, 1),
            "edge_after_limits": round(edge_after_limits, 4),
            "monthly_ev_after_limits": round(monthly_ev_after_limits, 2),
            "system_survives_12_months": system_survives,
            "book": book_lower,
            "recommendation": (
                "SUSTAINABLE" if system_survives and months_first_limit >= 6
                else "SHORT_LIVED" if system_survives
                else "UNSUSTAINABLE"
            ),
        }

    # ── Portfolio Execution Cost ───────────────────────────────────────

    def portfolio_execution_cost(self, bets: list) -> dict:
        """For a full portfolio of bets, compute total execution drag.

        Each bet should be a dict with at minimum:
        - edge_pct (float): theoretical edge
        - stake (float): desired stake
        - book (str): sportsbook name

        Returns:
            {
                "n_bets": int,
                "total_theoretical_ev": float,
                "total_actual_ev": float,
                "total_slippage_cost": float,
                "total_latency_cost": float,
                "total_limit_reduction": float,
                "execution_drag_pct": float,
                "bets_where_edge_destroyed": int,
                "per_book_summary": dict,
            }
        """
        if not bets:
            return {
                "n_bets": 0,
                "total_theoretical_ev": 0.0,
                "total_actual_ev": 0.0,
                "total_slippage_cost": 0.0,
                "total_latency_cost": 0.0,
                "total_limit_reduction": 0.0,
                "execution_drag_pct": 0.0,
                "bets_where_edge_destroyed": 0,
                "per_book_summary": {},
            }

        total_theoretical_ev = 0.0
        total_actual_ev = 0.0
        total_slippage = 0.0
        total_latency = 0.0
        total_limit_reduction = 0.0
        edge_destroyed_count = 0
        per_book = {}

        for bet in bets:
            if isinstance(bet, dict):
                edge = bet.get("edge_pct", 0.0)
                stake = bet.get("stake", 0.0)
                book = bet.get("book", "unknown")
                is_sharp = bet.get("is_sharp_profile", False)
            else:
                edge = getattr(bet, "edge_pct", 0.0)
                stake = getattr(bet, "stake", 0.0)
                book = getattr(bet, "book", "unknown")
                is_sharp = getattr(bet, "is_sharp_profile", False)

            result = self.simulate_execution(edge, stake, book, is_sharp)

            theo_ev = result["theoretical_ev"]
            actual_ev = result["limit_adjusted_ev"]
            slip = result["slippage_cost"] / 100.0 * stake
            lat = result["latency_cost"] / 100.0 * stake
            limit_red = stake - result["max_executable_stake"]

            total_theoretical_ev += theo_ev
            total_actual_ev += actual_ev
            total_slippage += slip
            total_latency += lat
            total_limit_reduction += limit_red

            if not result["edge_survives"]:
                edge_destroyed_count += 1

            # Per-book aggregation
            book_lower = book.lower()
            if book_lower not in per_book:
                per_book[book_lower] = {
                    "n_bets": 0,
                    "theoretical_ev": 0.0,
                    "actual_ev": 0.0,
                    "slippage_total": 0.0,
                    "edge_destroyed_count": 0,
                    "avg_limit": self.limit_model.get(book_lower, 250),
                }
            per_book[book_lower]["n_bets"] += 1
            per_book[book_lower]["theoretical_ev"] += theo_ev
            per_book[book_lower]["actual_ev"] += actual_ev
            per_book[book_lower]["slippage_total"] += slip
            if not result["edge_survives"]:
                per_book[book_lower]["edge_destroyed_count"] += 1

        # Round per-book values
        for bk in per_book:
            for key in ("theoretical_ev", "actual_ev", "slippage_total"):
                per_book[bk][key] = round(per_book[bk][key], 2)

        # Execution drag percentage
        if total_theoretical_ev > 0:
            drag_pct = (total_theoretical_ev - total_actual_ev) / total_theoretical_ev * 100
        else:
            drag_pct = 0.0

        # Find books where we'll be limited fastest
        books_by_risk = sorted(
            per_book.items(),
            key=lambda x: x[1]["edge_destroyed_count"] / max(x[1]["n_bets"], 1),
            reverse=True,
        )

        return {
            "n_bets": len(bets),
            "total_theoretical_ev": round(total_theoretical_ev, 2),
            "total_actual_ev": round(total_actual_ev, 2),
            "total_slippage_cost": round(total_slippage, 2),
            "total_latency_cost": round(total_latency, 2),
            "total_limit_reduction": round(total_limit_reduction, 2),
            "execution_drag_pct": round(drag_pct, 1),
            "bets_where_edge_destroyed": edge_destroyed_count,
            "pct_bets_edge_destroyed": round(
                edge_destroyed_count / len(bets) * 100, 1
            ),
            "per_book_summary": per_book,
            "highest_risk_books": [b[0] for b in books_by_risk[:3]],
        }
