"""
tests/adversarial/best_bet_removal.py
=======================================
Remove the top 10% most profitable bets and check if the remaining
bets are still profitable.

If your system's profitability depends entirely on a few lucky bets,
you don't have systematic edge — you have variance disguised as skill.

Tests:
  - Remove top 10% bets: remaining must be profitable (critical)
  - Remove top 20% bets: remaining should still be close to breakeven
  - Remove top 5% bets: should barely affect overall ROI
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from database.connection import session_scope
from database.models import Bet

log = logging.getLogger(__name__)


@dataclass
class RemovalResult:
    """Result of removing top N% profitable bets."""
    removal_pct: float
    n_removed: int
    n_remaining: int
    original_roi_pct: float
    remaining_roi_pct: float
    original_total_pnl: float
    remaining_total_pnl: float
    removed_pnl: float          # How much profit was in removed bets
    removed_pct_of_total_pnl: float
    remaining_profitable: bool
    passed: bool
    threshold: str


class BestBetRemovalTest:
    """Remove the best bets and check if the system still works."""

    REMOVAL_LEVELS = [
        {"pct": 5.0, "must_profit": True, "label": "Remove top 5%"},
        {"pct": 10.0, "must_profit": True, "label": "Remove top 10% (critical)"},
        {"pct": 20.0, "must_profit": False, "label": "Remove top 20% (stress)"},
    ]

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def run(self) -> Dict[str, Any]:
        """Run the best-bet removal test."""
        bets = self._load_bets()
        if len(bets) < 30:
            return {
                "status": "insufficient_data",
                "n_bets": len(bets),
                "results": [],
                "overall_pass": False,
            }

        results = []
        for level in self.REMOVAL_LEVELS:
            result = self._test_removal(bets, level["pct"], level["must_profit"])
            results.append(result)

        # Critical test: top 10% removal must still be profitable
        critical_pass = all(r.passed for r in results if r.removal_pct == 10.0)

        # Check concentration: what % of profit comes from top 10%?
        top10 = next((r for r in results if r.removal_pct == 10.0), None)
        concentration_warning = False
        if top10 and top10.removed_pct_of_total_pnl > 50:
            concentration_warning = True

        return {
            "status": "completed",
            "n_bets": len(bets),
            "results": [self._to_dict(r) for r in results],
            "overall_pass": critical_pass,
            "verdict": "PASS" if critical_pass else "FAIL",
            "concentration_warning": concentration_warning,
            "interpretation": self._interpret(results, critical_pass, concentration_warning),
        }

    def _test_removal(
        self,
        bets: List[Dict],
        removal_pct: float,
        must_profit: bool,
    ) -> RemovalResult:
        """Test a single removal level."""
        profits = np.array([b.get("profit", 0) or 0 for b in bets])
        stakes = np.array([max(b.get("stake", 1.0), 0.01) for b in bets])

        n = len(bets)
        n_remove = max(int(n * removal_pct / 100), 1)

        # Sort by profit (descending) and remove top N
        sorted_indices = np.argsort(profits)[::-1]
        removed_indices = set(sorted_indices[:n_remove])
        remaining_indices = [i for i in range(n) if i not in removed_indices]

        # Original metrics
        orig_total_pnl = float(np.sum(profits))
        orig_total_stake = float(np.sum(stakes))
        orig_roi = (orig_total_pnl / max(orig_total_stake, 1)) * 100

        # Remaining metrics
        remaining_profits = profits[remaining_indices]
        remaining_stakes = stakes[remaining_indices]
        remaining_pnl = float(np.sum(remaining_profits))
        remaining_stake = float(np.sum(remaining_stakes))
        remaining_roi = (remaining_pnl / max(remaining_stake, 1)) * 100

        # Removed metrics
        removed_pnl = orig_total_pnl - remaining_pnl
        removed_pct = (removed_pnl / max(abs(orig_total_pnl), 0.01)) * 100

        still_profitable = remaining_roi > 0

        if must_profit:
            passed = still_profitable
            threshold = "remaining bets must be profitable"
        else:
            passed = remaining_roi > -10.0
            threshold = "remaining ROI must be > -10%"

        return RemovalResult(
            removal_pct=removal_pct,
            n_removed=n_remove,
            n_remaining=len(remaining_indices),
            original_roi_pct=round(orig_roi, 2),
            remaining_roi_pct=round(remaining_roi, 2),
            original_total_pnl=round(orig_total_pnl, 2),
            remaining_total_pnl=round(remaining_pnl, 2),
            removed_pnl=round(removed_pnl, 2),
            removed_pct_of_total_pnl=round(removed_pct, 1),
            remaining_profitable=still_profitable,
            passed=passed,
            threshold=threshold,
        )

    def _interpret(
        self, results: List[RemovalResult], passed: bool, concentration: bool
    ) -> str:
        if passed and not concentration:
            return "Edge is SYSTEMATIC — not dependent on a few lucky bets"
        elif passed and concentration:
            return "Edge survives but >50% of profit comes from top 10% of bets. Monitor for concentration risk."
        else:
            return "OVERFIT WARNING: Removing top bets destroys profitability. Edge may be variance."

    def _to_dict(self, r: RemovalResult) -> Dict[str, Any]:
        return {
            "removal_pct": r.removal_pct,
            "n_removed": r.n_removed,
            "n_remaining": r.n_remaining,
            "original_roi_pct": r.original_roi_pct,
            "remaining_roi_pct": r.remaining_roi_pct,
            "removed_pnl": r.removed_pnl,
            "removed_pct_of_total_pnl": r.removed_pct_of_total_pnl,
            "remaining_profitable": r.remaining_profitable,
            "passed": r.passed,
            "threshold": r.threshold,
        }

    def _load_bets(self) -> List[Dict]:
        try:
            with session_scope() as session:
                bets = (
                    session.query(Bet)
                    .filter(Bet.sport == self.sport, Bet.status == "settled")
                    .order_by(Bet.timestamp.asc())
                    .all()
                )
                return [b.to_dict() for b in bets]
        except Exception as e:
            log.warning("Failed to load bets: %s", e)
            return []
