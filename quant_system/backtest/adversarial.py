"""Adversarial Destruction Tests — Detects overfitting through perturbation.

If the model collapses under noise, outlier removal, or time splitting,
it is overfit and the perceived edge is fake.

Usage:
    from quant_system.backtest.adversarial import AdversarialTestSuite

    suite = AdversarialTestSuite()
    report = suite.full_destruction_report(bets)
    if report["verdict"] == "OVERFIT":
        print("MODEL IS OVERFIT. DO NOT DEPLOY.")
"""

from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BetRecord:
    """Minimal bet record for adversarial testing.

    Accepts dicts with at minimum: model_prob, market_prob, odds_decimal,
    stake, pnl, status, timestamp.  Also supports raw dict access.
    """
    model_prob: float
    market_prob: float
    odds_decimal: float
    stake: float
    pnl: float
    status: str          # "won" or "lost"
    timestamp: str = ""  # ISO string or sortable date string
    features: dict | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "BetRecord":
        return cls(
            model_prob=d.get("model_prob", 0.5),
            market_prob=d.get("market_prob", 0.5),
            odds_decimal=d.get("odds_decimal", 2.0),
            stake=d.get("stake", 1.0),
            pnl=d.get("pnl", 0.0),
            status=d.get("status", "lost"),
            timestamp=str(d.get("timestamp", d.get("settled_at", ""))),
            features=d.get("features", d.get("features_snapshot", None)),
        )


class AdversarialTestSuite:
    """Runs destruction tests against the model to detect overfitting.

    If model collapses under perturbation, it is overfit.
    """

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_bets(bets: list) -> List[BetRecord]:
        """Convert list of dicts or BetRecords to BetRecord list."""
        result = []
        for b in bets:
            if isinstance(b, BetRecord):
                result.append(b)
            elif isinstance(b, dict):
                result.append(BetRecord.from_dict(b))
            else:
                # Assume object with attributes
                result.append(BetRecord(
                    model_prob=getattr(b, "model_prob", 0.5),
                    market_prob=getattr(b, "market_prob", 0.5),
                    odds_decimal=getattr(b, "odds_decimal", 2.0),
                    stake=getattr(b, "stake", 1.0),
                    pnl=getattr(b, "pnl", 0.0),
                    status=getattr(b, "status", "lost"),
                    timestamp=str(getattr(b, "timestamp", "")),
                    features=getattr(b, "features", None),
                ))
        return result

    @staticmethod
    def _compute_roi(bets: List[BetRecord]) -> float:
        """Compute ROI = total_pnl / total_staked."""
        total_pnl = sum(b.pnl for b in bets)
        total_staked = sum(b.stake for b in bets)
        if total_staked == 0:
            return 0.0
        return total_pnl / total_staked

    @staticmethod
    def _compute_win_rate(bets: List[BetRecord]) -> float:
        """Compute win rate."""
        if not bets:
            return 0.0
        wins = sum(1 for b in bets if b.status == "won")
        return wins / len(bets)

    @staticmethod
    def _coefficient_of_variation(values: list) -> float:
        """CV = std / |mean|. Measures relative dispersion."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        if abs(mean) < 1e-10:
            return float("inf")
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance)
        return std / abs(mean)

    # ── Test 1: Probability Perturbation ───────────────────────────────

    def probability_perturbation(
        self,
        bets: list,
        perturbation_range: Tuple[float, float] = (0.02, 0.10),
        n_simulations: int = 200,
    ) -> dict:
        """Add random noise to model probabilities and re-evaluate.

        If performance collapses: the model is overfit to precise probability
        estimates. A robust model should tolerate small noise.

        Returns:
            {
                "original_roi": float,
                "perturbed_roi_mean": float,
                "perturbed_roi_std": float,
                "roi_drop_pct": float,
                "collapse_detected": bool,
                "perturbation_range": tuple,
            }
        """
        records = self._normalize_bets(bets)
        if not records:
            return {"error": "No bets provided", "collapse_detected": False}

        original_roi = self._compute_roi(records)

        perturbed_rois = []
        for _ in range(n_simulations):
            perturbed = deepcopy(records)
            for bet in perturbed:
                noise = random.uniform(*perturbation_range) * random.choice([-1, 1])
                new_prob = max(0.01, min(0.99, bet.model_prob + noise))
                # Re-evaluate: would we still have bet?
                # If perturbed prob < market_prob, the bet was a mistake
                if new_prob < bet.market_prob:
                    # Would not have bet — zero out PnL
                    bet.pnl = 0.0
                    bet.stake = 0.0
            perturbed_rois.append(self._compute_roi(perturbed))

        mean_perturbed = sum(perturbed_rois) / len(perturbed_rois)
        variance = sum((r - mean_perturbed) ** 2 for r in perturbed_rois) / max(len(perturbed_rois) - 1, 1)
        std_perturbed = math.sqrt(variance)

        # Collapse = ROI drops more than 50% on average
        if original_roi > 0:
            roi_drop_pct = (original_roi - mean_perturbed) / original_roi * 100
        else:
            roi_drop_pct = 0.0

        collapse = roi_drop_pct > 50.0

        return {
            "original_roi": round(original_roi * 100, 2),
            "perturbed_roi_mean": round(mean_perturbed * 100, 2),
            "perturbed_roi_std": round(std_perturbed * 100, 2),
            "roi_drop_pct": round(roi_drop_pct, 1),
            "collapse_detected": collapse,
            "perturbation_range": perturbation_range,
            "n_simulations": n_simulations,
        }

    # ── Test 2: Remove Best Bets ───────────────────────────────────────

    def remove_best_bets(
        self,
        bets: list,
        pct_to_remove: float = 0.10,
    ) -> dict:
        """Remove top N% most profitable bets.

        If remaining bets are unprofitable, the edge is concentrated in
        a few outlier wins. This is fragile and possibly lucky.

        Returns:
            {
                "original_roi": float,
                "trimmed_roi": float,
                "n_removed": int,
                "removed_pnl": float,
                "concentration_risk": bool,
            }
        """
        records = self._normalize_bets(bets)
        if not records:
            return {"error": "No bets provided", "concentration_risk": False}

        original_roi = self._compute_roi(records)
        total_original_pnl = sum(b.pnl for b in records)

        # Sort by PnL descending, remove top N%
        sorted_bets = sorted(records, key=lambda b: b.pnl, reverse=True)
        n_remove = max(1, int(len(sorted_bets) * pct_to_remove))
        removed = sorted_bets[:n_remove]
        remaining = sorted_bets[n_remove:]

        removed_pnl = sum(b.pnl for b in removed)
        trimmed_roi = self._compute_roi(remaining)
        remaining_pnl = sum(b.pnl for b in remaining)

        # Concentration risk: remaining bets are unprofitable
        concentration_risk = remaining_pnl <= 0 and total_original_pnl > 0

        return {
            "original_roi": round(original_roi * 100, 2),
            "trimmed_roi": round(trimmed_roi * 100, 2),
            "original_pnl": round(total_original_pnl, 2),
            "remaining_pnl": round(remaining_pnl, 2),
            "n_total": len(records),
            "n_removed": n_remove,
            "removed_pnl": round(removed_pnl, 2),
            "pct_pnl_from_top_bets": round(
                removed_pnl / total_original_pnl * 100, 1
            ) if total_original_pnl > 0 else 0.0,
            "concentration_risk": concentration_risk,
        }

    # ── Test 3: Input Noise Test ───────────────────────────────────────

    def input_noise_test(
        self,
        bets: list,
        noise_std: float = 0.05,
        n_simulations: int = 200,
    ) -> dict:
        """Add Gaussian noise to all input features.

        If performance drops >50%: overfitting to precise feature values.
        A robust model should handle measurement noise gracefully.

        Returns:
            {
                "original_roi": float,
                "noisy_roi_mean": float,
                "performance_drop_pct": float,
                "overfit_to_features": bool,
            }
        """
        records = self._normalize_bets(bets)
        if not records:
            return {"error": "No bets provided", "overfit_to_features": False}

        original_roi = self._compute_roi(records)

        noisy_rois = []
        for _ in range(n_simulations):
            noisy = deepcopy(records)
            for bet in noisy:
                # Add noise to model_prob (the primary derived feature)
                noise = random.gauss(0, noise_std)
                new_prob = max(0.01, min(0.99, bet.model_prob + noise))

                # If features dict exists, perturb those too
                if bet.features and isinstance(bet.features, dict):
                    for key in bet.features:
                        val = bet.features[key]
                        if isinstance(val, (int, float)):
                            feat_noise = random.gauss(0, noise_std * abs(val)) if val != 0 else random.gauss(0, noise_std)
                            bet.features[key] = val + feat_noise

                # Re-evaluate: did this noise change the bet decision?
                if new_prob < bet.market_prob:
                    bet.pnl = 0.0
                    bet.stake = 0.0
                else:
                    # Adjust PnL proportionally to probability shift
                    prob_ratio = new_prob / max(bet.model_prob, 0.01)
                    # Edge changed — adjust expected PnL
                    old_edge = bet.model_prob - bet.market_prob
                    new_edge = new_prob - bet.market_prob
                    if old_edge > 0 and new_edge > 0:
                        edge_ratio = new_edge / old_edge
                        # Scale PnL by edge change (approximate)
                        bet.pnl = bet.pnl * min(edge_ratio, 2.0)  # cap to avoid explosion

            noisy_rois.append(self._compute_roi(noisy))

        mean_noisy = sum(noisy_rois) / len(noisy_rois)

        if original_roi > 0:
            performance_drop = (original_roi - mean_noisy) / original_roi * 100
        else:
            performance_drop = 0.0

        overfit = performance_drop > 50.0

        return {
            "original_roi": round(original_roi * 100, 2),
            "noisy_roi_mean": round(mean_noisy * 100, 2),
            "noise_std": noise_std,
            "performance_drop_pct": round(performance_drop, 1),
            "overfit_to_features": overfit,
            "n_simulations": n_simulations,
        }

    # ── Test 4: Time Stability Test ────────────────────────────────────

    def time_stability_test(
        self,
        bets: list,
        n_windows: int = 5,
    ) -> dict:
        """Split bets into time windows and test edge stability.

        If edge exists in only 1-2 windows out of 5, it is
        regime-dependent and not stable enough to deploy.

        Returns:
            {
                "per_window_roi": list,
                "per_window_n_bets": list,
                "profitable_windows": int,
                "stable": bool,
                "cv": float,  # coefficient of variation of ROI across windows
            }
        """
        records = self._normalize_bets(bets)
        if not records:
            return {"error": "No bets provided", "stable": False, "cv": 0.0}

        # Sort by timestamp
        records.sort(key=lambda b: b.timestamp)

        # Split into equal-sized windows
        window_size = max(1, len(records) // n_windows)
        windows = []
        for i in range(n_windows):
            start = i * window_size
            if i == n_windows - 1:
                # Last window gets remaining bets
                end = len(records)
            else:
                end = start + window_size
            windows.append(records[start:end])

        # Filter out empty windows
        windows = [w for w in windows if len(w) > 0]

        per_window_roi = []
        per_window_n = []
        per_window_pnl = []
        profitable_count = 0

        for window in windows:
            roi = self._compute_roi(window)
            pnl = sum(b.pnl for b in window)
            per_window_roi.append(round(roi * 100, 2))
            per_window_n.append(len(window))
            per_window_pnl.append(round(pnl, 2))
            if pnl > 0:
                profitable_count += 1

        # Coefficient of variation of ROI across windows
        cv = self._coefficient_of_variation(per_window_roi)

        # Stable = profitable in at least 60% of windows AND low CV
        n_actual = len(windows)
        stable = (
            profitable_count >= max(n_actual * 0.6, 3)
            and cv < 1.5
        )

        return {
            "per_window_roi": per_window_roi,
            "per_window_n_bets": per_window_n,
            "per_window_pnl": per_window_pnl,
            "n_windows": n_actual,
            "profitable_windows": profitable_count,
            "unprofitable_windows": n_actual - profitable_count,
            "stable": stable,
            "cv": round(cv, 3),
        }

    # ── Full Destruction Report ────────────────────────────────────────

    def full_destruction_report(self, bets: list) -> dict:
        """Run ALL destruction tests and return combined verdict.

        Verdicts:
        - "ROBUST": survives all tests
        - "FRAGILE": fails 1-2 tests
        - "OVERFIT": fails 3+ tests
        """
        if not bets:
            return {
                "verdict": "INSUFFICIENT_DATA",
                "tests_failed": 0,
                "tests_run": 0,
                "details": {},
            }

        # Run all four tests
        prob_test = self.probability_perturbation(bets)
        outlier_test = self.remove_best_bets(bets)
        noise_test = self.input_noise_test(bets)
        time_test = self.time_stability_test(bets)

        # Count failures
        failures = []
        tests_run = 4

        if prob_test.get("collapse_detected", False):
            failures.append("probability_perturbation")

        if outlier_test.get("concentration_risk", False):
            failures.append("outlier_concentration")

        if noise_test.get("overfit_to_features", False):
            failures.append("input_noise_sensitivity")

        if not time_test.get("stable", False):
            failures.append("time_instability")

        n_failures = len(failures)

        if n_failures == 0:
            verdict = "ROBUST"
        elif n_failures <= 2:
            verdict = "FRAGILE"
        else:
            verdict = "OVERFIT"

        return {
            "verdict": verdict,
            "tests_run": tests_run,
            "tests_failed": n_failures,
            "tests_passed": tests_run - n_failures,
            "failures": failures,
            "details": {
                "probability_perturbation": prob_test,
                "remove_best_bets": outlier_test,
                "input_noise": noise_test,
                "time_stability": time_test,
            },
            "recommendation": {
                "ROBUST": "Model appears sound. Monitor ongoing.",
                "FRAGILE": "Model has weaknesses. Investigate failed tests before increasing stake.",
                "OVERFIT": "MODEL IS OVERFIT. Do NOT deploy with real capital. Retrain required.",
            }.get(verdict, "Unknown"),
        }
