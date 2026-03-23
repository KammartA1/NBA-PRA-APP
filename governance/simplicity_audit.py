"""
governance/simplicity_audit.py
===============================
For EACH feature: remove it, re-run on holdout, compute Brier/ROI
with vs without, and p-value.

Decision:
  - KEEP (p < 0.05 and removing it hurts performance)
  - DELETE (no effect or removing it improves performance)

The philosophy: simpler models are better models. Every feature must
justify its existence with statistical evidence. The burden of proof
is on the feature to PROVE it helps, not on us to prove it doesn't.
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


class SimplicityAuditor:
    """Audit every feature for statistical justification.

    The simplicity principle: if removing a feature doesn't significantly
    hurt performance (p < 0.05), remove it. Complexity without proof = risk.
    """

    def __init__(
        self,
        sport: str = "NBA",
        significance_level: float = 0.05,
        holdout_fraction: float = 0.3,
    ):
        self.sport = sport
        self.alpha = significance_level
        self.holdout_fraction = holdout_fraction

    def audit(self) -> Dict[str, Any]:
        """Run the full simplicity audit on all features.

        For each feature:
          1. Split data into with-feature and without-feature groups
          2. Compute Brier score and ROI for each group
          3. Statistical test for significant difference
          4. Decision: KEEP or DELETE
        """
        bets = self._load_bets()

        if len(bets) < 100:
            return {
                "status": "insufficient_data",
                "n_bets": len(bets),
                "results": [],
                "summary": {
                    "keep": 0,
                    "delete": 0,
                    "total": 0,
                    "complexity_score": 0,
                },
            }

        # Use holdout portion
        n = len(bets)
        split = int(n * (1 - self.holdout_fraction))
        holdout = bets[split:]

        # Extract all features
        all_features = self._collect_feature_names(holdout)

        if not all_features:
            return {
                "status": "no_features",
                "n_bets": len(holdout),
                "results": [],
                "summary": {
                    "keep": 0,
                    "delete": 0,
                    "total": 0,
                    "complexity_score": 0,
                },
            }

        # Audit each feature
        results = []
        for feat_name in all_features:
            result = self._audit_feature(feat_name, holdout)
            results.append(result)

        # Sort: DELETE first (features that should be removed)
        results.sort(key=lambda x: (0 if x["decision"] == "DELETE" else 1, -x.get("p_value", 1.0)))

        keep_count = sum(1 for r in results if r["decision"] == "KEEP")
        delete_count = sum(1 for r in results if r["decision"] == "DELETE")

        # Complexity score: 0 = minimal (good), 100 = maximally complex (bad)
        total = len(results)
        complexity = (total - keep_count) / max(total, 1) * 100  # % of features that should go

        return {
            "status": "completed",
            "n_bets": len(holdout),
            "n_holdout_bets": len(holdout),
            "significance_level": self.alpha,
            "results": results,
            "summary": {
                "keep": keep_count,
                "delete": delete_count,
                "total": total,
                "complexity_score": round(complexity, 1),
                "features_to_delete": [r["feature"] for r in results if r["decision"] == "DELETE"],
                "features_to_keep": [r["feature"] for r in results if r["decision"] == "KEEP"],
            },
        }

    def _audit_feature(self, feat_name: str, holdout: List[Dict]) -> Dict[str, Any]:
        """Audit a single feature."""
        # Split holdout into feature-active and feature-inactive groups
        active_bets = []
        inactive_bets = []

        for b in holdout:
            snap = b.get("features_snapshot_json", "{}")
            try:
                features = json.loads(snap) if isinstance(snap, str) else snap
            except (json.JSONDecodeError, TypeError):
                features = {}

            val = features.get(feat_name, 0)
            if isinstance(val, (int, float)) and abs(val) > 1e-10:
                active_bets.append(b)
            else:
                inactive_bets.append(b)

        n_active = len(active_bets)
        n_inactive = len(inactive_bets)

        # Not enough data in one group
        if n_active < 15 or n_inactive < 15:
            return {
                "feature": feat_name,
                "decision": "DELETE",
                "reason": f"Insufficient split: {n_active} active, {n_inactive} inactive",
                "p_value": 1.0,
                "brier_with": None,
                "brier_without": None,
                "roi_with": None,
                "roi_without": None,
                "n_active": n_active,
                "n_inactive": n_inactive,
                "evidence_strength": "none",
            }

        # Brier scores
        brier_with = self._brier(active_bets)
        brier_without = self._brier(inactive_bets)

        # ROI
        roi_with = self._roi(active_bets)
        roi_without = self._roi(inactive_bets)

        # Profit arrays for statistical testing
        profits_with = np.array([b.get("profit", 0) or 0 for b in active_bets])
        profits_without = np.array([b.get("profit", 0) or 0 for b in inactive_bets])

        # Two-sample t-test: is there a significant difference?
        if np.std(profits_with, ddof=1) > 0 and np.std(profits_without, ddof=1) > 0:
            t_stat, p_value = sp_stats.ttest_ind(profits_with, profits_without)
        else:
            t_stat, p_value = 0.0, 1.0

        # Also test Brier difference
        brier_diff = brier_with - brier_without  # Negative = feature group has better Brier

        # Decision logic:
        # KEEP if: removing the feature significantly hurts (p < alpha AND feature group performs better)
        # DELETE if: no significant difference, or removing the feature improves things

        feature_helps = (brier_with < brier_without) or (
            float(np.mean(profits_with)) > float(np.mean(profits_without))
        )

        if p_value < self.alpha and feature_helps:
            decision = "KEEP"
            reason = f"Removing hurts: p={p_value:.4f}, Brier diff={brier_diff:.4f}"
            evidence = "strong" if p_value < 0.01 else "moderate"
        elif p_value < self.alpha and not feature_helps:
            decision = "DELETE"
            reason = f"Feature active group performs WORSE: p={p_value:.4f}"
            evidence = "strong_negative"
        elif p_value >= self.alpha:
            decision = "DELETE"
            reason = f"No significant effect: p={p_value:.4f} >= {self.alpha}"
            evidence = "none"
        else:
            decision = "DELETE"
            reason = "No evidence of contribution"
            evidence = "none"

        return {
            "feature": feat_name,
            "decision": decision,
            "reason": reason,
            "p_value": round(float(p_value), 6),
            "t_statistic": round(float(t_stat), 4),
            "brier_with": round(brier_with, 6),
            "brier_without": round(brier_without, 6),
            "brier_diff": round(brier_diff, 6),
            "roi_with": round(roi_with, 2),
            "roi_without": round(roi_without, 2),
            "avg_profit_with": round(float(np.mean(profits_with)), 2),
            "avg_profit_without": round(float(np.mean(profits_without)), 2),
            "n_active": n_active,
            "n_inactive": n_inactive,
            "evidence_strength": evidence,
        }

    def _collect_feature_names(self, bets: List[Dict]) -> List[str]:
        """Collect all feature names from bet snapshots."""
        names: set = set()
        for b in bets:
            snap = b.get("features_snapshot_json", "{}")
            try:
                features = json.loads(snap) if isinstance(snap, str) else snap
            except (json.JSONDecodeError, TypeError):
                features = {}
            for k, v in features.items():
                if isinstance(v, (int, float)):
                    names.add(k)
        return sorted(names)

    def _brier(self, bets: List[Dict]) -> float:
        """Compute Brier score."""
        if not bets:
            return 0.25
        pred = np.array([b.get("predicted_prob", 0.5) for b in bets])
        actual = np.array([1.0 if (b.get("profit", 0) or 0) > 0 else 0.0 for b in bets])
        return float(np.mean((pred - actual) ** 2))

    def _roi(self, bets: List[Dict]) -> float:
        """Compute ROI %."""
        if not bets:
            return 0.0
        profits = sum(b.get("profit", 0) or 0 for b in bets)
        stakes = sum(max(b.get("stake", 1.0), 0.01) for b in bets)
        return (profits / max(stakes, 1)) * 100

    def _load_bets(self) -> List[Dict]:
        """Load settled bets from database."""
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
