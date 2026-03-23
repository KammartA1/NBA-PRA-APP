"""
tests/adversarial/probability_perturbation.py
===============================================
Add noise to probability estimates and check if system remains profitable.

If adding 2% noise to probabilities destroys all profit, the system is
fragile and likely overfit. Real edge should survive moderate perturbation.

Tests:
  - +/-2% noise: Must remain profitable (basic robustness)
  - +/-5% noise: Should remain positive ROI (moderate robustness)
  - +/-10% noise: Stress test (may lose money, but should not collapse)
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
class PerturbationResult:
    """Result of a single perturbation level test."""
    noise_pct: float           # Noise level (e.g., 2, 5, 10)
    n_bets: int
    original_roi_pct: float
    perturbed_roi_pct: float
    roi_degradation_pct: float
    original_win_rate: float
    perturbed_win_rate: float
    still_profitable: bool
    n_simulations: int
    roi_std: float             # Std of perturbed ROI across simulations
    passed: bool               # Did this noise level pass?
    threshold: str             # What the pass condition was


class ProbabilityPerturbationTest:
    """Adversarial test: perturb probability estimates."""

    NOISE_LEVELS = [
        {"pct": 2.0, "must_profit": True, "label": "2% noise (basic)"},
        {"pct": 5.0, "must_profit": True, "label": "5% noise (moderate)"},
        {"pct": 10.0, "must_profit": False, "label": "10% noise (stress)"},
    ]

    def __init__(self, sport: str = "NBA", n_simulations: int = 100):
        self.sport = sport
        self.n_simulations = n_simulations

    def run(self) -> Dict[str, Any]:
        """Run all perturbation tests."""
        bets = self._load_bets()
        if len(bets) < 30:
            return {
                "status": "insufficient_data",
                "n_bets": len(bets),
                "results": [],
                "overall_pass": False,
            }

        results = []
        for level in self.NOISE_LEVELS:
            result = self._test_noise_level(
                bets, level["pct"], level["must_profit"], level["label"]
            )
            results.append(result)

        # Overall pass: 2% and 5% must pass
        critical_pass = all(r.passed for r in results if r.noise_pct <= 5.0)

        return {
            "status": "completed",
            "n_bets": len(bets),
            "results": [self._result_to_dict(r) for r in results],
            "overall_pass": critical_pass,
            "verdict": "PASS" if critical_pass else "FAIL",
            "interpretation": (
                "System is robust to probability noise" if critical_pass
                else "System is FRAGILE — small probability errors destroy profitability"
            ),
        }

    def _test_noise_level(
        self,
        bets: List[Dict],
        noise_pct: float,
        must_profit: bool,
        label: str,
    ) -> PerturbationResult:
        """Test a single noise level."""
        n = len(bets)

        # Original performance
        orig_profits = np.array([b.get("profit", 0) or 0 for b in bets])
        orig_stakes = np.array([max(b.get("stake", 1.0), 0.01) for b in bets])
        orig_roi = float(np.sum(orig_profits) / np.sum(orig_stakes) * 100)
        orig_wr = float(np.mean(orig_profits > 0))

        pred_probs = np.array([b.get("predicted_prob", 0.5) for b in bets])
        odds_decimal = np.array([b.get("odds_decimal", 1.91) or 1.91 for b in bets])

        rng = np.random.default_rng(42)
        perturbed_rois = []

        for _ in range(self.n_simulations):
            # Add noise to probabilities
            noise = rng.normal(0, noise_pct / 100.0, n)
            noisy_probs = np.clip(pred_probs + noise, 0.01, 0.99)

            # Re-simulate bet decisions
            # A bet is "good" if prob > implied_prob
            implied_probs = 1.0 / np.maximum(odds_decimal, 1.01)
            edge = noisy_probs - implied_probs

            # Only bet when edge > 0 (after noise)
            would_bet = edge > 0

            if np.sum(would_bet) == 0:
                perturbed_rois.append(-100.0)
                continue

            # Profits for bets we'd still take
            perturbed_profit = np.sum(orig_profits[would_bet])
            perturbed_stake = np.sum(orig_stakes[would_bet])
            perturbed_roi = (perturbed_profit / max(perturbed_stake, 1)) * 100
            perturbed_rois.append(perturbed_roi)

        avg_perturbed_roi = float(np.mean(perturbed_rois))
        std_roi = float(np.std(perturbed_rois, ddof=1)) if len(perturbed_rois) > 1 else 0.0
        perturbed_wr = float(np.mean(np.array(perturbed_rois) > 0))

        roi_degradation = orig_roi - avg_perturbed_roi
        still_profitable = avg_perturbed_roi > 0

        if must_profit:
            passed = still_profitable
            threshold = "must be profitable"
        else:
            # For 10% noise, pass if ROI doesn't collapse to < -20%
            passed = avg_perturbed_roi > -20.0
            threshold = "ROI must be > -20%"

        return PerturbationResult(
            noise_pct=noise_pct,
            n_bets=n,
            original_roi_pct=round(orig_roi, 2),
            perturbed_roi_pct=round(avg_perturbed_roi, 2),
            roi_degradation_pct=round(roi_degradation, 2),
            original_win_rate=round(orig_wr, 4),
            perturbed_win_rate=round(perturbed_wr, 4),
            still_profitable=still_profitable,
            n_simulations=self.n_simulations,
            roi_std=round(std_roi, 2),
            passed=passed,
            threshold=threshold,
        )

    def _result_to_dict(self, r: PerturbationResult) -> Dict[str, Any]:
        return {
            "noise_pct": r.noise_pct,
            "n_bets": r.n_bets,
            "original_roi_pct": r.original_roi_pct,
            "perturbed_roi_pct": r.perturbed_roi_pct,
            "roi_degradation_pct": r.roi_degradation_pct,
            "still_profitable": r.still_profitable,
            "roi_std": r.roi_std,
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
