"""Quant Dashboard & Reporting — Single pane of glass for system health.

Generates structured reports for display in Streamlit or CLI.
This module does NOT render UI — it produces data structures that
the sport-specific dashboard consumes.

Key metrics displayed:
1. CLV (most important)
2. ROI vs Expected ROI
3. Win rate vs predicted probability
4. Drawdown curve
5. Feature performance
6. Model vs market accuracy
7. System state & circuit breaker status
"""

from __future__ import annotations

import logging
from datetime import datetime

from ..core.types import Sport, SystemState
from ..core.bet_logger import BetLogger
from ..core.clv_tracker import CLVTracker
from ..core.calibration import CalibrationMonitor
from ..core.edge_validator import EdgeValidator
from ..risk.bankroll_manager import BankrollManager
from ..risk.failure_protection import FailureProtection
from ..learning.model_drift import DriftDetector
from ..learning.feature_monitor import FeatureMonitor
from ..backtest.mc_bankroll import BankrollSimulator

logger = logging.getLogger(__name__)


class QuantDashboard:
    """Generates comprehensive dashboard data."""

    def __init__(
        self,
        sport: Sport,
        initial_bankroll: float = 1000.0,
        db_path: str | None = None,
    ):
        self.sport = sport
        self._db_path = db_path
        self.bet_logger = BetLogger(sport, db_path)
        self.clv_tracker = CLVTracker(sport, db_path)
        self.calibration = CalibrationMonitor(sport, db_path)
        self.edge_validator = EdgeValidator(sport, db_path)
        self.bankroll_mgr = BankrollManager(sport, initial_bankroll, db_path=db_path)
        self.failure_protection = FailureProtection(sport, db_path=db_path)
        self.drift_detector = DriftDetector(sport, db_path)
        self.feature_monitor = FeatureMonitor(sport, db_path)
        self.mc_simulator = BankrollSimulator()

    def full_report(self) -> dict:
        """Generate the complete dashboard report.

        Returns a nested dict suitable for Streamlit rendering.
        """
        risk_state = self.bankroll_mgr.get_risk_state()

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "sport": self.sport.value,
            "system_state": risk_state.system_state.value,
        }

        # 1. Bankroll Overview
        report["bankroll"] = {
            "current": risk_state.bankroll,
            "peak": risk_state.peak_bankroll,
            "drawdown_pct": round(risk_state.current_drawdown_pct * 100, 1),
            "max_drawdown_pct": round(risk_state.max_drawdown_pct * 100, 1),
            "daily_pnl": risk_state.daily_pnl,
            "daily_bets": risk_state.daily_bet_count,
            "total_exposure": risk_state.total_exposure,
            "daily_loss_remaining": risk_state.daily_loss_remaining,
        }

        # 2. CLV (MOST IMPORTANT)
        report["clv"] = self.clv_tracker.clv_summary()
        report["clv_by_type"] = self.clv_tracker.clv_by_bet_type()

        # 3. Calibration
        report["calibration"] = self.calibration.compute_calibration()
        report["calibration_drift"] = self.calibration.calibration_drift()

        # 4. Edge Validation
        edge_report = self.edge_validator.validate(risk_state.bankroll, risk_state.peak_bankroll)
        report["edge"] = {
            "exists": edge_report.edge_exists,
            "roi_actual": edge_report.model_roi,
            "roi_expected": edge_report.expected_roi,
            "warnings": edge_report.warnings,
            "actions": edge_report.actions,
        }

        # 5. Circuit Breakers
        breakers = self.failure_protection.check_all(risk_state)
        report["circuit_breakers"] = {
            "triggered": breakers["breakers_triggered"],
            "clear": breakers["breakers_clear"],
            "details": {k: v.get("message", "") for k, v in breakers["details"].items()},
        }

        # 6. P&L Curve
        report["pnl_curve"] = self.bankroll_mgr.pnl_curve()

        # 7. Bet Summary
        all_bets = self.bet_logger.get_all_bets(limit=500)
        settled = [b for b in all_bets if b["status"] in ("won", "lost")]
        report["bet_summary"] = {
            "total_bets": len(all_bets),
            "settled_bets": len(settled),
            "pending_bets": len([b for b in all_bets if b["status"] == "pending"]),
            "win_rate": round(
                sum(1 for b in settled if b["status"] == "won") / max(len(settled), 1), 4
            ),
            "total_staked": round(sum(b["stake"] for b in settled), 2),
            "total_pnl": round(sum(b["pnl"] for b in settled), 2),
            "avg_edge": round(
                sum(b["edge"] for b in settled) / max(len(settled), 1), 4
            ),
            "avg_stake": round(
                sum(b["stake"] for b in settled) / max(len(settled), 1), 2
            ),
        }

        # 8. Model Drift
        report["drift"] = {
            "prediction_shift": self.drift_detector.prediction_distribution_shift(),
            "accuracy_degradation": self.drift_detector.accuracy_degradation(),
            "edge_decay": self.drift_detector.edge_decay(),
        }

        # 9. Feature Health
        report["features"] = self.feature_monitor.evaluate_features()

        # 10. Monte Carlo Future Projection
        if len(settled) >= 30:
            avg_edge = report["bet_summary"]["avg_edge"]
            avg_stake_pct = report["bet_summary"]["avg_stake"] / max(risk_state.bankroll, 1)
            win_rate = report["bet_summary"]["win_rate"]

            report["mc_projection"] = self.mc_simulator.simulate(
                avg_edge=avg_edge,
                avg_odds_decimal=1.91,  # Average -110 odds
                avg_stake_pct=min(avg_stake_pct, 0.05),
                win_rate=win_rate,
            )

            # Also run bootstrap from actual history
            history = [
                {"pnl": b["pnl"], "bankroll_after": risk_state.bankroll}
                for b in settled[-200:]
            ]
            report["mc_bootstrap"] = self.mc_simulator.simulate_from_history(history)
        else:
            report["mc_projection"] = {"message": "Need 30+ settled bets"}
            report["mc_bootstrap"] = {"message": "Need 30+ settled bets"}

        return report

    def health_summary(self) -> str:
        """One-line health summary for quick status checks."""
        risk_state = self.bankroll_mgr.get_risk_state()
        clv = self.clv_tracker.rolling_clv(100)

        state = risk_state.system_state.value.upper()
        bankroll = risk_state.bankroll
        dd = risk_state.current_drawdown_pct * 100
        clv_cents = clv.get("avg_clv_cents", 0)
        beat_close = clv.get("beat_close_pct", 0) * 100

        return (
            f"[{state}] Bankroll: ${bankroll:,.0f} | "
            f"Drawdown: {dd:.1f}% | "
            f"CLV: {clv_cents:+.1f} cents | "
            f"Beat Close: {beat_close:.0f}%"
        )
