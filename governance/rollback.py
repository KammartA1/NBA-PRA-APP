"""
governance/rollback.py
=======================
Auto-rollback system that reverts to a previous model version when
the current model underperforms the previous one by more than a
configurable threshold over N bets.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from database.connection import session_scope
from database.models import EdgeReport, ModelVersion
from governance.performance_tracker import PerformanceTracker
from governance.version_control import ModelVersionManager

log = logging.getLogger(__name__)


class RollbackManager:
    """Automatic model rollback when performance degrades.

    Monitors current model performance against the previous version.
    If the current version underperforms by more than `threshold_pct`
    over `min_bets` samples, automatically rolls back.
    """

    def __init__(
        self,
        sport: str = "NBA",
        threshold_pct: float = 10.0,
        min_bets: int = 100,
    ):
        """
        Args:
            sport: Sport to monitor.
            threshold_pct: Performance degradation % that triggers rollback.
            min_bets: Minimum bets before rollback can trigger.
        """
        self.sport = sport
        self.threshold_pct = threshold_pct
        self.min_bets = min_bets
        self.tracker = PerformanceTracker(sport)
        self.version_mgr = ModelVersionManager(sport)

    def check_and_rollback(self) -> Dict[str, Any]:
        """Check if rollback is needed and execute if so.

        Returns:
            Dict with rollback decision and details.
        """
        # Get version history
        history = self.version_mgr.get_version_history(limit=5)

        if len(history) < 2:
            return {
                "rollback_needed": False,
                "reason": "fewer than 2 versions available",
                "action_taken": None,
            }

        current = history[0]
        previous = history[1]

        current_id = current["version"]
        previous_id = previous["version"]

        # Get performance for both
        current_perf = self.tracker.track_version_performance(current_id)
        previous_perf = self.tracker.track_version_performance(previous_id)

        # Need minimum samples
        if current_perf["n_bets"] < self.min_bets:
            return {
                "rollback_needed": False,
                "reason": f"current version has {current_perf['n_bets']} bets, need {self.min_bets}",
                "current_version": current_id,
                "current_performance": current_perf,
                "action_taken": None,
            }

        # Compare key metrics
        degradations = []

        # Brier score (lower is better)
        if previous_perf["brier_score"] > 0:
            brier_change = (
                (current_perf["brier_score"] - previous_perf["brier_score"])
                / previous_perf["brier_score"] * 100
            )
            if brier_change > self.threshold_pct:
                degradations.append(f"Brier degraded by {brier_change:.1f}%")

        # CLV (higher is better)
        if previous_perf["avg_clv"] > 0:
            clv_change = (
                (previous_perf["avg_clv"] - current_perf["avg_clv"])
                / previous_perf["avg_clv"] * 100
            )
            if clv_change > self.threshold_pct:
                degradations.append(f"CLV degraded by {clv_change:.1f}%")

        # ROI (higher is better)
        if previous_perf["roi_pct"] > 0:
            roi_change = (
                (previous_perf["roi_pct"] - current_perf["roi_pct"])
                / max(abs(previous_perf["roi_pct"]), 0.01) * 100
            )
            if roi_change > self.threshold_pct:
                degradations.append(f"ROI degraded by {roi_change:.1f}%")

        rollback_needed = len(degradations) >= 2  # Need at least 2 metrics degraded

        result = {
            "rollback_needed": rollback_needed,
            "current_version": current_id,
            "previous_version": previous_id,
            "current_performance": current_perf,
            "previous_performance": previous_perf,
            "degradations": degradations,
            "threshold_pct": self.threshold_pct,
            "action_taken": None,
        }

        if rollback_needed:
            success = self.version_mgr.activate_version(previous_id)
            if success:
                result["action_taken"] = f"Rolled back from {current_id} to {previous_id}"
                self._log_rollback(current_id, previous_id, degradations)
                log.warning("AUTO-ROLLBACK: %s → %s (%s)", current_id, previous_id, "; ".join(degradations))
            else:
                result["action_taken"] = "Rollback attempted but failed"

        return result

    def get_rollback_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get history of rollback events."""
        try:
            with session_scope() as session:
                reports = (
                    session.query(EdgeReport)
                    .filter(
                        EdgeReport.report_type == "model_rollback",
                        EdgeReport.sport == self.sport,
                    )
                    .order_by(EdgeReport.generated_at.desc())
                    .limit(limit)
                    .all()
                )
                return [json.loads(r.report_json or "{}") for r in reports]
        except Exception as e:
            log.warning("Failed to load rollback history: %s", e)
            return []

    def _log_rollback(
        self,
        from_version: str,
        to_version: str,
        reasons: List[str],
    ) -> None:
        """Log a rollback event to the database."""
        try:
            data = {
                "from_version": from_version,
                "to_version": to_version,
                "reasons": reasons,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "threshold_pct": self.threshold_pct,
            }
            with session_scope() as session:
                session.add(EdgeReport(
                    report_type="model_rollback",
                    sport=self.sport,
                    report_json=json.dumps(data),
                ))
        except Exception as e:
            log.warning("Failed to log rollback: %s", e)
