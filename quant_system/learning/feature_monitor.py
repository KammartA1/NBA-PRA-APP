"""Feature Monitor — Tracks which features are performing over time.

For every feature in the model, we track:
1. Correlation with actual outcomes (is it predictive?)
2. Stability of correlation (is it consistently predictive?)
3. Directional accuracy (does it point the right way?)

Features that degrade get flagged for review or automatic removal.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
from scipy import stats as sp_stats

from ..db.schema import BetLog, FeatureLog, get_session
from ..core.types import Sport

logger = logging.getLogger(__name__)


class FeatureMonitor:
    """Monitors feature importance and stability over time."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def evaluate_features(self, window: int = 200) -> dict:
        """Evaluate all tracked features against actual outcomes.

        Requires bets to have features_snapshot populated.

        Returns:
            {
                "features": {
                    "feature_name": {
                        "correlation": float,
                        "p_value": float,
                        "directional_accuracy": float,
                        "is_significant": bool,
                        "is_degraded": bool,
                    }
                },
                "degraded_features": [str],
                "top_features": [str],
            }
        """
        session = self._session()
        try:
            bets = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .filter(BetLog.features_snapshot != "")
                .filter(BetLog.features_snapshot != "{}")
                .order_by(BetLog.timestamp.desc())
                .limit(window)
                .all()
            )

            if len(bets) < 30:
                return {"features": {}, "degraded_features": [], "top_features": [],
                        "message": "Insufficient data with feature snapshots"}

            # Parse feature snapshots
            outcomes = []
            feature_values: dict[str, list[float]] = {}

            for bet in bets:
                try:
                    features = json.loads(bet.features_snapshot)
                except (json.JSONDecodeError, TypeError):
                    continue

                outcome = 1.0 if bet.status == "won" else 0.0
                outcomes.append(outcome)

                for fname, fval in features.items():
                    if isinstance(fval, (int, float)):
                        feature_values.setdefault(fname, []).append(float(fval))

            outcomes_arr = np.array(outcomes)
            results = {}

            for fname, vals in feature_values.items():
                if len(vals) != len(outcomes_arr) or len(vals) < 20:
                    continue

                vals_arr = np.array(vals)

                # Correlation with outcome
                if np.std(vals_arr) > 0:
                    corr, p_val = sp_stats.pearsonr(vals_arr, outcomes_arr[:len(vals_arr)])
                else:
                    corr, p_val = 0.0, 1.0

                # Directional accuracy
                # For positive feature values, did the bet win more often?
                median_val = np.median(vals_arr)
                above_median = outcomes_arr[:len(vals_arr)][vals_arr > median_val]
                below_median = outcomes_arr[:len(vals_arr)][vals_arr <= median_val]

                if len(above_median) > 0 and len(below_median) > 0:
                    dir_accuracy = abs(float(np.mean(above_median) - np.mean(below_median)))
                else:
                    dir_accuracy = 0.0

                is_sig = abs(corr) > 0.05 and p_val < 0.10
                is_degraded = abs(corr) < 0.02 or p_val > 0.50

                results[fname] = {
                    "correlation": round(float(corr), 4),
                    "p_value": round(float(p_val), 4),
                    "directional_accuracy": round(dir_accuracy, 4),
                    "is_significant": is_sig,
                    "is_degraded": is_degraded,
                    "n_samples": len(vals),
                }

            # Rank features
            ranked = sorted(results.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True)
            degraded = [name for name, r in results.items() if r["is_degraded"]]
            top = [name for name, _ in ranked[:10]]

            return {
                "features": results,
                "degraded_features": degraded,
                "top_features": top,
                "n_features_evaluated": len(results),
            }
        finally:
            session.close()

    def save_feature_report(self) -> None:
        """Save feature evaluation to the log table."""
        evaluation = self.evaluate_features()
        if not evaluation["features"]:
            return

        session = self._session()
        try:
            now = datetime.utcnow()
            for fname, metrics in evaluation["features"].items():
                row = FeatureLog(
                    sport=self.sport.value,
                    report_date=now,
                    feature_name=fname,
                    importance_score=metrics["correlation"],
                    directional_accuracy=metrics["directional_accuracy"],
                    n_samples=metrics["n_samples"],
                    is_degraded=metrics["is_degraded"],
                )
                session.add(row)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to save feature report")
        finally:
            session.close()
