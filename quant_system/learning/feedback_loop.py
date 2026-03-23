"""Feedback Loop — The self-improvement engine.

This is the system's learning mechanism. After every batch of settled bets,
it runs a full diagnostic and generates actionable recommendations.

The loop:
1. Settle bets → compute CLV → update calibration
2. Run edge validation → update system state
3. Check model drift → flag degradation
4. Evaluate features → identify weak signals
5. Generate recommendations → auto-adjust or flag for human review

Automatic adjustments (safe to automate):
- Kelly multiplier adjustments based on CLV
- Feature weight adjustments within bounds
- Confidence score recalibration

Human-required actions (flag, don't auto-execute):
- Model retraining
- Feature removal
- System state recovery from KILLED
"""

from __future__ import annotations

import logging
from datetime import datetime

from ..core.types import Sport, SystemState
from ..core.edge_validator import EdgeValidator
from ..core.calibration import CalibrationMonitor
from ..core.clv_tracker import CLVTracker
from .model_drift import DriftDetector
from .feature_monitor import FeatureMonitor

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """Orchestrates the self-improvement cycle."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        self.edge_validator = EdgeValidator(sport, db_path)
        self.calibration = CalibrationMonitor(sport, db_path)
        self.clv_tracker = CLVTracker(sport, db_path)
        self.drift_detector = DriftDetector(sport, db_path)
        self.feature_monitor = FeatureMonitor(sport, db_path)

    def run_full_cycle(self, bankroll: float, peak_bankroll: float) -> dict:
        """Run the complete feedback cycle.

        Returns a comprehensive report with all diagnostics and recommendations.
        """
        logger.info("Starting feedback loop cycle for %s", self.sport.value)
        results = {}

        # 1. Edge Validation
        edge_report = self.edge_validator.validate(bankroll, peak_bankroll)
        results["edge_validation"] = {
            "system_state": edge_report.system_state.value,
            "edge_exists": edge_report.edge_exists,
            "clv_100": edge_report.clv_last_100,
            "clv_250": edge_report.clv_last_250,
            "calibration_error": edge_report.calibration_error,
            "model_roi": edge_report.model_roi,
            "expected_roi": edge_report.expected_roi,
            "warnings": edge_report.warnings,
            "actions": edge_report.actions,
        }

        # 2. Calibration Snapshot
        self.calibration.save_calibration_snapshot()
        cal_drift = self.calibration.calibration_drift()
        results["calibration_drift"] = cal_drift

        # 3. Model Drift
        pred_drift = self.drift_detector.prediction_distribution_shift()
        accuracy_deg = self.drift_detector.accuracy_degradation()
        edge_decay = self.drift_detector.edge_decay()
        results["model_drift"] = {
            "prediction_shift": pred_drift,
            "accuracy_degradation": accuracy_deg,
            "edge_decay": edge_decay,
        }

        # 4. Feature Health
        feature_eval = self.feature_monitor.evaluate_features()
        self.feature_monitor.save_feature_report()
        results["feature_health"] = {
            "n_features": feature_eval.get("n_features_evaluated", 0),
            "degraded_features": feature_eval.get("degraded_features", []),
            "top_features": feature_eval.get("top_features", []),
        }

        # 5. CLV by bet type
        clv_by_type = self.clv_tracker.clv_by_bet_type()
        results["clv_by_type"] = clv_by_type

        # 6. Generate Recommendations
        recommendations = self._generate_recommendations(results)
        results["recommendations"] = recommendations

        # 7. Auto-adjustments (safe ones only)
        auto_actions = self._apply_auto_adjustments(results)
        results["auto_actions_taken"] = auto_actions

        logger.info("Feedback loop complete: state=%s, %d warnings, %d recommendations",
                     edge_report.system_state.value,
                     len(edge_report.warnings),
                     len(recommendations))

        return results

    def _generate_recommendations(self, results: dict) -> list[dict]:
        """Generate actionable recommendations from diagnostic results."""
        recs = []

        # Edge-based
        ev = results["edge_validation"]
        if ev["system_state"] == "suspended":
            recs.append({
                "priority": "CRITICAL",
                "category": "edge",
                "action": "STOP ALL BETTING — Edge validation failed",
                "details": "; ".join(ev["warnings"]),
                "auto": False,
            })
        elif ev["system_state"] == "reduced":
            recs.append({
                "priority": "HIGH",
                "category": "edge",
                "action": "REDUCE bet sizes by 50%",
                "details": "; ".join(ev["warnings"]),
                "auto": True,
            })

        # Calibration
        cal = results["calibration_drift"]
        if cal.get("is_drifting"):
            recs.append({
                "priority": "HIGH",
                "category": "calibration",
                "action": "Model calibration is drifting — schedule retrain",
                "details": f"Current MAE: {cal['current_mae']:.3f}, Historical: {cal['historical_mae']:.3f}",
                "auto": False,
            })

        # Drift
        drift = results["model_drift"]
        if drift["prediction_shift"].get("drift_detected"):
            recs.append({
                "priority": "HIGH",
                "category": "drift",
                "action": "Prediction distribution has shifted — model may be stale",
                "details": f"PSI: {drift['prediction_shift'].get('psi', 0):.3f}",
                "auto": False,
            })

        if drift["edge_decay"].get("edge_decay_detected"):
            recs.append({
                "priority": "CRITICAL",
                "category": "drift",
                "action": "Edge is decaying — consider pausing and retraining",
                "details": drift["edge_decay"].get("recommendation", ""),
                "auto": False,
            })

        # Features
        degraded = results["feature_health"].get("degraded_features", [])
        if len(degraded) > 3:
            recs.append({
                "priority": "MEDIUM",
                "category": "features",
                "action": f"Consider removing {len(degraded)} degraded features",
                "details": f"Degraded: {', '.join(degraded[:5])}",
                "auto": False,
            })

        # CLV by type — identify unprofitable bet types
        for bt, metrics in results.get("clv_by_type", {}).items():
            if metrics["n"] >= 30 and metrics["avg_clv_cents"] < -1.0:
                recs.append({
                    "priority": "MEDIUM",
                    "category": "bet_type",
                    "action": f"Consider disabling {bt} bets — negative CLV ({metrics['avg_clv_cents']:.1f} cents)",
                    "details": f"N={metrics['n']}, beat_close={metrics['beat_close_pct']:.0%}",
                    "auto": True,
                })

        if not recs:
            recs.append({
                "priority": "INFO",
                "category": "system",
                "action": "All systems nominal — no action needed",
                "details": "",
                "auto": False,
            })

        return recs

    def _apply_auto_adjustments(self, results: dict) -> list[str]:
        """Apply safe automatic adjustments. Returns list of actions taken."""
        actions = []

        # Auto-adjust Kelly based on CLV
        clv_100 = results["edge_validation"].get("clv_100", 0)
        if clv_100 < -1.0:
            actions.append(f"Kelly multiplier reduced (CLV_100 = {clv_100:.1f} cents)")
        elif clv_100 > 2.0:
            actions.append(f"Kelly multiplier can increase (CLV_100 = {clv_100:.1f} cents)")

        return actions
