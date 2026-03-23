"""Feature Ablation Testing — Which features actually matter?

For each feature in the model:
1. Run walk-forward with all features
2. Run walk-forward WITHOUT that feature
3. Compare: if performance drops, feature is valuable
4. If performance improves, feature is HURTING the model

This answers: "Is this 27-factor NBA adjustment actually helping,
or is it just adding noise?"
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class FeatureAblation:
    """Test feature importance by systematic removal."""

    def run_ablation(
        self,
        data: list[dict],
        features: list[str],
        model_fn: Callable,
        baseline_metric: str = "roi_pct",
    ) -> dict:
        """Run ablation study on each feature.

        Args:
            data: Historical data with all features
            features: List of feature names to test
            model_fn: fn(data, excluded_features) -> BacktestResult
                      The model function must accept a list of features to exclude
            baseline_metric: Metric to compare (roi_pct, sharpe_ratio, etc.)

        Returns:
            {
                "baseline": {metric: value},
                "ablation_results": {
                    feature_name: {
                        "metric_without": float,
                        "metric_change": float,    # negative = feature helps
                        "is_valuable": bool,
                        "importance_rank": int,
                    }
                },
                "features_to_remove": [str],       # Features that hurt performance
                "features_ranked": [str],           # Best to worst
            }
        """
        # Run baseline (all features)
        baseline_result = model_fn(data, excluded_features=[])
        baseline_value = getattr(baseline_result, baseline_metric, 0.0)

        results = {}
        for feature in features:
            try:
                ablated_result = model_fn(data, excluded_features=[feature])
                ablated_value = getattr(ablated_result, baseline_metric, 0.0)
                change = ablated_value - baseline_value

                results[feature] = {
                    "metric_without": round(ablated_value, 4),
                    "metric_change": round(change, 4),
                    "is_valuable": change < 0,  # Performance dropped = feature was helping
                }
            except Exception as e:
                logger.warning("Ablation failed for feature %s: %s", feature, e)
                results[feature] = {
                    "metric_without": 0.0,
                    "metric_change": 0.0,
                    "is_valuable": False,
                    "error": str(e),
                }

        # Rank features (most negative change = most important)
        ranked = sorted(results.items(), key=lambda x: x[1]["metric_change"])
        for i, (name, _) in enumerate(ranked):
            results[name]["importance_rank"] = i + 1

        # Features that hurt the model (positive change when removed)
        to_remove = [
            name for name, r in results.items()
            if r["metric_change"] > 0.5  # Removing it improved ROI by > 0.5%
        ]

        return {
            "baseline": {baseline_metric: baseline_value},
            "ablation_results": results,
            "features_to_remove": to_remove,
            "features_ranked": [name for name, _ in ranked],
        }
