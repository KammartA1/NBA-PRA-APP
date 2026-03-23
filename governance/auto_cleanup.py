"""
governance/auto_cleanup.py
===========================
Automated feature cleanup — removes degrading features by evaluating
them on a holdout set and removing those that don't contribute.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from database.connection import session_scope
from database.models import Bet

log = logging.getLogger(__name__)


class AutoCleanup:
    """Automated feature cleanup system.

    For each feature, evaluates whether it contributes to model
    performance. Features that don't help (or actively hurt) are
    flagged for removal.

    Process:
      1. Split data into training and holdout sets
      2. For each feature, compute performance with and without
      3. If removing the feature doesn't degrade performance
         (or improves it), mark for deletion
    """

    def __init__(
        self,
        sport: str = "NBA",
        holdout_fraction: float = 0.3,
        min_samples: int = 50,
        degradation_threshold: float = 0.005,  # Brier change threshold
    ):
        self.sport = sport
        self.holdout_fraction = holdout_fraction
        self.min_samples = min_samples
        self.degradation_threshold = degradation_threshold

    def evaluate_features(self) -> Dict[str, Any]:
        """Evaluate all features and recommend which to remove.

        Returns a report with keep/delete/uncertain decisions for each feature.
        """
        bets = self._load_bets()
        if len(bets) < self.min_samples * 2:
            return {
                "status": "insufficient_data",
                "n_bets": len(bets),
                "features": [],
            }

        # Split into holdout
        n = len(bets)
        split_idx = int(n * (1 - self.holdout_fraction))
        holdout = bets[split_idx:]

        # Extract features
        feature_data = self._extract_features(holdout)
        if not feature_data["feature_names"]:
            return {
                "status": "no_features_found",
                "n_bets": len(holdout),
                "features": [],
            }

        # Baseline performance on holdout
        baseline_brier = feature_data["baseline_brier"]
        baseline_roi = feature_data["baseline_roi"]

        # Evaluate each feature
        results = []
        for feat_name in feature_data["feature_names"]:
            eval_result = self._evaluate_single_feature(
                feat_name, holdout, baseline_brier, baseline_roi
            )
            results.append(eval_result)

        # Sort by recommendation (delete first, then uncertain, then keep)
        priority_map = {"delete": 0, "uncertain": 1, "keep": 2}
        results.sort(key=lambda x: priority_map.get(x["decision"], 3))

        keep_count = sum(1 for r in results if r["decision"] == "keep")
        delete_count = sum(1 for r in results if r["decision"] == "delete")
        uncertain_count = sum(1 for r in results if r["decision"] == "uncertain")

        return {
            "status": "evaluated",
            "n_bets": len(holdout),
            "n_features": len(results),
            "keep_count": keep_count,
            "delete_count": delete_count,
            "uncertain_count": uncertain_count,
            "baseline_brier": round(baseline_brier, 6),
            "baseline_roi": round(baseline_roi, 2),
            "features": results,
            "recommended_deletions": [r["feature"] for r in results if r["decision"] == "delete"],
        }

    def auto_clean(self, dry_run: bool = True) -> Dict[str, Any]:
        """Run automatic cleanup. If dry_run=True, just report what would happen."""
        evaluation = self.evaluate_features()

        if evaluation["status"] != "evaluated":
            return {"action": "none", "reason": evaluation["status"]}

        to_delete = evaluation["recommended_deletions"]

        if dry_run:
            return {
                "action": "dry_run",
                "would_delete": to_delete,
                "n_deletions": len(to_delete),
                "details": evaluation,
            }

        # In production, this would modify the model configuration
        # For now, we log the recommendation
        log.info("Auto-cleanup recommends deleting %d features: %s", len(to_delete), to_delete)

        return {
            "action": "recommended",
            "features_to_delete": to_delete,
            "n_deletions": len(to_delete),
            "details": evaluation,
        }

    def _evaluate_single_feature(
        self,
        feat_name: str,
        holdout: List[Dict],
        baseline_brier: float,
        baseline_roi: float,
    ) -> Dict[str, Any]:
        """Evaluate a single feature's contribution on the holdout set."""
        # Separate bets where feature is active vs not
        with_feat = []
        without_feat = []

        for b in holdout:
            snap = b.get("features_snapshot_json", "{}")
            try:
                features = json.loads(snap) if isinstance(snap, str) else snap
            except (json.JSONDecodeError, TypeError):
                features = {}

            if feat_name in features and isinstance(features[feat_name], (int, float)):
                if abs(features[feat_name]) > 1e-10:
                    with_feat.append(b)
                else:
                    without_feat.append(b)
            else:
                without_feat.append(b)

        if len(with_feat) < 10:
            return {
                "feature": feat_name,
                "decision": "uncertain",
                "reason": f"only {len(with_feat)} bets with this feature",
                "n_with": len(with_feat),
                "n_without": len(without_feat),
                "brier_with": None,
                "brier_without": None,
                "p_value": None,
            }

        # Compute Brier for each group
        brier_with = self._compute_brier(with_feat)
        brier_without = self._compute_brier(without_feat) if len(without_feat) >= 10 else baseline_brier

        # Compute profit for each group
        profit_with = float(np.mean([b.get("profit", 0) or 0 for b in with_feat]))
        profit_without = float(np.mean([b.get("profit", 0) or 0 for b in without_feat])) if without_feat else 0

        # Statistical test: is the feature's impact significant?
        profits_with = np.array([b.get("profit", 0) or 0 for b in with_feat])
        profits_without = np.array([b.get("profit", 0) or 0 for b in without_feat]) if without_feat else np.array([0])

        if len(profits_with) >= 10 and len(profits_without) >= 10:
            t_stat, p_value = sp_stats.ttest_ind(profits_with, profits_without)
        else:
            p_value = 1.0

        # Decision logic
        brier_diff = brier_with - brier_without  # Positive = feature makes Brier worse

        if brier_diff > self.degradation_threshold:
            # Feature makes things worse
            decision = "delete"
            reason = f"Feature hurts performance: Brier +{brier_diff:.4f}"
        elif brier_diff < -self.degradation_threshold and p_value < 0.05:
            # Feature significantly helps
            decision = "keep"
            reason = f"Feature helps (p={p_value:.3f}): Brier {brier_diff:.4f}"
        elif abs(brier_diff) <= self.degradation_threshold:
            # Feature has no effect — delete for simplicity
            decision = "delete"
            reason = f"No significant effect (Brier diff={brier_diff:.4f})"
        else:
            decision = "uncertain"
            reason = f"Ambiguous: Brier diff={brier_diff:.4f}, p={p_value:.3f}"

        return {
            "feature": feat_name,
            "decision": decision,
            "reason": reason,
            "n_with": len(with_feat),
            "n_without": len(without_feat),
            "brier_with": round(brier_with, 6),
            "brier_without": round(brier_without, 6),
            "brier_diff": round(brier_diff, 6),
            "profit_with": round(profit_with, 2),
            "profit_without": round(profit_without, 2),
            "p_value": round(p_value, 4),
        }

    def _extract_features(self, bets: List[Dict]) -> Dict[str, Any]:
        """Extract feature names and compute baseline metrics."""
        feature_names: set = set()
        pred_probs = []
        outcomes = []
        profits = []
        stakes = []

        for b in bets:
            snap = b.get("features_snapshot_json", "{}")
            try:
                features = json.loads(snap) if isinstance(snap, str) else snap
            except (json.JSONDecodeError, TypeError):
                features = {}

            for k, v in features.items():
                if isinstance(v, (int, float)):
                    feature_names.add(k)

            pred_probs.append(b.get("predicted_prob", 0.5))
            outcomes.append(1.0 if (b.get("profit", 0) or 0) > 0 else 0.0)
            profits.append(b.get("profit", 0) or 0)
            stakes.append(max(b.get("stake", 1.0), 0.01))

        pred_arr = np.array(pred_probs)
        out_arr = np.array(outcomes)
        baseline_brier = float(np.mean((pred_arr - out_arr) ** 2))

        total_stake = sum(stakes)
        baseline_roi = (sum(profits) / max(total_stake, 1)) * 100

        return {
            "feature_names": sorted(feature_names),
            "baseline_brier": baseline_brier,
            "baseline_roi": baseline_roi,
        }

    def _compute_brier(self, bets: List[Dict]) -> float:
        """Compute Brier score for a set of bets."""
        if not bets:
            return 0.25
        pred = np.array([b.get("predicted_prob", 0.5) for b in bets])
        actual = np.array([1.0 if (b.get("profit", 0) or 0) > 0 else 0.0 for b in bets])
        return float(np.mean((pred - actual) ** 2))

    def _load_bets(self) -> List[Dict]:
        """Load all settled bets."""
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
