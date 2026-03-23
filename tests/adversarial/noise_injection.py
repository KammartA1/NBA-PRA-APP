"""
tests/adversarial/noise_injection.py
======================================
Add noise to model inputs and find the breakpoint where the system fails.

Systematically increase noise in all input features until profitability
collapses. The noise level at which the system breaks reveals how much
"margin of safety" exists in the edge.

Tests at: 1%, 5%, 10%, 20% noise levels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from database.connection import session_scope
from database.models import Bet

log = logging.getLogger(__name__)


@dataclass
class NoiseResult:
    """Result of a single noise injection level."""
    noise_pct: float
    n_bets: int
    original_roi_pct: float
    noisy_roi_pct: float
    roi_degradation_pct: float
    original_brier: float
    noisy_brier: float
    brier_degradation: float
    still_profitable: bool
    is_breakpoint: bool
    n_simulations: int


class NoiseInjectionTest:
    """Add noise to all inputs and find the breakpoint."""

    NOISE_LEVELS = [1.0, 5.0, 10.0, 20.0]

    def __init__(self, sport: str = "NBA", n_simulations: int = 50):
        self.sport = sport
        self.n_simulations = n_simulations

    def run(self) -> Dict[str, Any]:
        """Run noise injection across all levels."""
        bets = self._load_bets()
        if len(bets) < 30:
            return {
                "status": "insufficient_data",
                "n_bets": len(bets),
                "results": [],
                "breakpoint_pct": None,
                "overall_pass": False,
            }

        # Original metrics
        profits = np.array([b.get("profit", 0) or 0 for b in bets])
        stakes = np.array([max(b.get("stake", 1.0), 0.01) for b in bets])
        pred_probs = np.array([b.get("predicted_prob", 0.5) for b in bets])
        outcomes = np.array([1.0 if p > 0 else 0.0 for p in profits])

        orig_roi = float(np.sum(profits) / np.sum(stakes) * 100)
        orig_brier = float(np.mean((pred_probs - outcomes) ** 2))

        results = []
        breakpoint_found = False
        breakpoint_pct = None

        for noise_pct in self.NOISE_LEVELS:
            result = self._test_noise(
                bets, profits, stakes, pred_probs, outcomes,
                noise_pct, orig_roi, orig_brier,
            )
            results.append(result)

            if result.is_breakpoint and not breakpoint_found:
                breakpoint_found = True
                breakpoint_pct = noise_pct

        # Pass if system survives 5% noise
        survived_5pct = any(
            r.still_profitable for r in results if r.noise_pct == 5.0
        )

        return {
            "status": "completed",
            "n_bets": len(bets),
            "original_roi_pct": round(orig_roi, 2),
            "original_brier": round(orig_brier, 6),
            "results": [self._to_dict(r) for r in results],
            "breakpoint_pct": breakpoint_pct,
            "overall_pass": survived_5pct,
            "verdict": "PASS" if survived_5pct else "FAIL",
            "margin_of_safety": (
                f"System breaks at {breakpoint_pct}% noise" if breakpoint_pct
                else "System survived all noise levels tested"
            ),
        }

    def _test_noise(
        self,
        bets: List[Dict],
        profits: np.ndarray,
        stakes: np.ndarray,
        pred_probs: np.ndarray,
        outcomes: np.ndarray,
        noise_pct: float,
        orig_roi: float,
        orig_brier: float,
    ) -> NoiseResult:
        """Test a single noise injection level."""
        n = len(bets)
        rng = np.random.default_rng(42)

        noisy_rois = []
        noisy_briers = []

        for _ in range(self.n_simulations):
            # Add noise to features (which affect predictions)
            noise = rng.normal(0, noise_pct / 100.0, n)
            noisy_probs = np.clip(pred_probs + noise, 0.01, 0.99)

            # Brier with noisy predictions
            noisy_brier = float(np.mean((noisy_probs - outcomes) ** 2))
            noisy_briers.append(noisy_brier)

            # Re-evaluate bet decisions
            odds_decimal = np.array([
                b.get("odds_decimal", 1.91) or 1.91 for b in bets
            ])
            implied = 1.0 / np.maximum(odds_decimal, 1.01)
            edge = noisy_probs - implied
            would_bet = edge > 0.02  # Minimum 2% edge filter

            if np.sum(would_bet) == 0:
                noisy_rois.append(-100.0)
                continue

            noisy_profit = float(np.sum(profits[would_bet]))
            noisy_stake = float(np.sum(stakes[would_bet]))
            noisy_roi = (noisy_profit / max(noisy_stake, 1)) * 100
            noisy_rois.append(noisy_roi)

        avg_roi = float(np.mean(noisy_rois))
        avg_brier = float(np.mean(noisy_briers))
        still_profitable = avg_roi > 0
        is_breakpoint = not still_profitable and orig_roi > 0

        return NoiseResult(
            noise_pct=noise_pct,
            n_bets=n,
            original_roi_pct=round(orig_roi, 2),
            noisy_roi_pct=round(avg_roi, 2),
            roi_degradation_pct=round(orig_roi - avg_roi, 2),
            original_brier=round(orig_brier, 6),
            noisy_brier=round(avg_brier, 6),
            brier_degradation=round(avg_brier - orig_brier, 6),
            still_profitable=still_profitable,
            is_breakpoint=is_breakpoint,
            n_simulations=self.n_simulations,
        )

    def _to_dict(self, r: NoiseResult) -> Dict[str, Any]:
        return {
            "noise_pct": r.noise_pct,
            "original_roi_pct": r.original_roi_pct,
            "noisy_roi_pct": r.noisy_roi_pct,
            "roi_degradation_pct": r.roi_degradation_pct,
            "noisy_brier": r.noisy_brier,
            "brier_degradation": r.brier_degradation,
            "still_profitable": r.still_profitable,
            "is_breakpoint": r.is_breakpoint,
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
