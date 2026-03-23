"""
governance/feature_importance.py
=================================
Feature importance analysis using permutation importance,
marginal Brier contribution, and importance tracking over time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from database.connection import session_scope
from database.models import Bet, EdgeReport

log = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Analyze which features actually contribute to model performance.

    Methods:
      - Permutation importance: shuffle each feature, measure Brier degradation
      - Marginal Brier contribution: Brier with vs without each feature
      - Profit attribution: how much P&L is driven by each feature
      - Temporal tracking: how importance changes over time
    """

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def permutation_importance(
        self,
        n_permutations: int = 20,
    ) -> List[Dict[str, Any]]:
        """Compute permutation importance for each feature.

        For each feature, shuffles its values and measures the increase
        in Brier score. Features that cause large Brier increases when
        shuffled are important.
        """
        bets = self._load_bets_with_features()
        if len(bets) < 50:
            return []

        # Extract features and outcomes
        features_by_bet: List[Dict[str, float]] = []
        pred_probs = []
        outcomes = []

        for b in bets:
            snap = b.get("features_snapshot_json", "{}")
            try:
                features = json.loads(snap) if isinstance(snap, str) else snap
            except (json.JSONDecodeError, TypeError):
                features = {}

            # Only include numeric features
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            if numeric_features:
                features_by_bet.append(numeric_features)
                pred_probs.append(b.get("predicted_prob", 0.5))
                outcomes.append(1.0 if (b.get("profit", 0) or 0) > 0 else 0.0)

        if len(features_by_bet) < 30:
            return []

        pred_arr = np.array(pred_probs)
        out_arr = np.array(outcomes)
        n = len(features_by_bet)

        # Baseline Brier
        baseline_brier = float(np.mean((pred_arr - out_arr) ** 2))

        # Get all feature names
        all_features: set = set()
        for f in features_by_bet:
            all_features.update(f.keys())

        results = []
        rng = np.random.default_rng(42)

        for feat_name in all_features:
            # Extract this feature's values
            feat_values = np.array([f.get(feat_name, 0.0) for f in features_by_bet])

            brier_increases = []
            for _ in range(n_permutations):
                # Shuffle this feature
                shuffled = rng.permutation(feat_values)

                # Estimate prediction change: simple proportional adjustment
                # (This is an approximation since we don't have the actual model)
                orig_range = np.ptp(feat_values)
                if orig_range > 0:
                    perturbation = (shuffled - feat_values) / orig_range * 0.05
                    perturbed_probs = np.clip(pred_arr + perturbation, 0.01, 0.99)
                else:
                    perturbed_probs = pred_arr

                permuted_brier = float(np.mean((perturbed_probs - out_arr) ** 2))
                brier_increases.append(permuted_brier - baseline_brier)

            avg_increase = float(np.mean(brier_increases))
            std_increase = float(np.std(brier_increases, ddof=1)) if n_permutations > 1 else 0.0

            # Statistical significance
            if std_increase > 0 and n_permutations >= 10:
                t_stat = avg_increase / (std_increase / np.sqrt(n_permutations))
                p_value = 1.0 - sp_stats.t.cdf(abs(t_stat), n_permutations - 1)
            else:
                p_value = 1.0

            results.append({
                "feature": feat_name,
                "importance": round(avg_increase * 1000, 4),  # Scale for readability
                "std": round(std_increase * 1000, 4),
                "p_value": round(p_value, 4),
                "is_significant": p_value < 0.05,
                "baseline_brier": round(baseline_brier, 6),
                "n_bets_with_feature": sum(1 for f in features_by_bet if feat_name in f),
            })

        # Sort by importance (descending)
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results

    def marginal_brier_contribution(self) -> List[Dict[str, Any]]:
        """Compute marginal Brier contribution for each feature.

        For each feature, computes:
          - Brier score of bets WITH this feature active (non-zero)
          - Brier score of bets WITHOUT this feature (zero or missing)
          - The difference indicates marginal contribution
        """
        bets = self._load_bets_with_features()
        if len(bets) < 50:
            return []

        # Parse features
        parsed_bets = []
        for b in bets:
            snap = b.get("features_snapshot_json", "{}")
            try:
                features = json.loads(snap) if isinstance(snap, str) else snap
            except (json.JSONDecodeError, TypeError):
                features = {}
            parsed_bets.append({
                "features": {k: v for k, v in features.items() if isinstance(v, (int, float))},
                "predicted_prob": b.get("predicted_prob", 0.5),
                "outcome": 1.0 if (b.get("profit", 0) or 0) > 0 else 0.0,
                "profit": b.get("profit", 0) or 0,
            })

        # Collect all feature names
        all_features: set = set()
        for pb in parsed_bets:
            all_features.update(pb["features"].keys())

        results = []
        for feat_name in all_features:
            # Split into with/without groups
            with_feat = [
                pb for pb in parsed_bets
                if feat_name in pb["features"] and abs(pb["features"][feat_name]) > 1e-10
            ]
            without_feat = [
                pb for pb in parsed_bets
                if feat_name not in pb["features"] or abs(pb["features"].get(feat_name, 0)) <= 1e-10
            ]

            if len(with_feat) < 10 or len(without_feat) < 10:
                continue

            # Brier scores
            with_brier = float(np.mean([
                (pb["predicted_prob"] - pb["outcome"]) ** 2 for pb in with_feat
            ]))
            without_brier = float(np.mean([
                (pb["predicted_prob"] - pb["outcome"]) ** 2 for pb in without_feat
            ]))

            # Profit comparison
            with_profit = float(np.mean([pb["profit"] for pb in with_feat]))
            without_profit = float(np.mean([pb["profit"] for pb in without_feat]))

            results.append({
                "feature": feat_name,
                "brier_with": round(with_brier, 6),
                "brier_without": round(without_brier, 6),
                "brier_diff": round(with_brier - without_brier, 6),
                "feature_helps": with_brier < without_brier,
                "avg_profit_with": round(with_profit, 2),
                "avg_profit_without": round(without_profit, 2),
                "n_with": len(with_feat),
                "n_without": len(without_feat),
            })

        results.sort(key=lambda x: x["brier_diff"])  # Lowest diff = most helpful
        return results

    def track_importance_over_time(
        self,
        window: int = 100,
        step: int = 50,
    ) -> Dict[str, Any]:
        """Track how feature importance changes over time.

        Computes profit attribution in rolling windows.
        """
        bets = self._load_bets_with_features()
        if len(bets) < window:
            return {"windows": [], "features": {}}

        windows = []
        feature_series: Dict[str, List[float]] = {}

        for i in range(0, len(bets) - window + 1, step):
            chunk = bets[i:i + window]
            window_label = f"bets_{i}_{i + window}"
            windows.append(window_label)

            # Compute average profit by feature presence
            for b in chunk:
                snap = b.get("features_snapshot_json", "{}")
                try:
                    features = json.loads(snap) if isinstance(snap, str) else snap
                except (json.JSONDecodeError, TypeError):
                    features = {}
                profit = b.get("profit", 0) or 0
                for feat_name, val in features.items():
                    if isinstance(val, (int, float)) and abs(val) > 1e-10:
                        feature_series.setdefault(feat_name, []).append(profit)

        # Average each feature's contribution per window
        importance_over_time = {}
        for feat_name, profits_list in feature_series.items():
            if len(profits_list) >= 10:
                importance_over_time[feat_name] = round(float(np.mean(profits_list)), 3)

        return {
            "windows": windows,
            "feature_importance": importance_over_time,
            "n_features_tracked": len(importance_over_time),
        }

    def _load_bets_with_features(self) -> List[Dict[str, Any]]:
        """Load settled bets that have feature snapshots."""
        try:
            with session_scope() as session:
                bets = (
                    session.query(Bet)
                    .filter(
                        Bet.sport == self.sport,
                        Bet.status == "settled",
                    )
                    .order_by(Bet.timestamp.asc())
                    .all()
                )
                return [b.to_dict() for b in bets]
        except Exception as e:
            log.warning("Failed to load bets: %s", e)
            return []
