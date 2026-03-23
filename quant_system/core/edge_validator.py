"""Edge Validation Engine — Answers daily: "Do I actually have edge right now?"

This is the brain of the system. It aggregates CLV, calibration, P&L, and
model performance into a single verdict: ACTIVE, REDUCED, SUSPENDED, or KILLED.

Thresholds (configurable):
    CLV < 0 over 100 bets  → REDUCED (halve bet sizes)
    CLV < 0 over 250 bets  → SUSPENDED (stop betting)
    Calibration error > 8% → trigger retrain
    Drawdown > 20%         → REDUCED
    Drawdown > 35%         → SUSPENDED
    Drawdown > 50%         → KILLED
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from ..db.schema import BetLog, SystemStateLog, get_session
from .calibration import CalibrationMonitor
from .clv_tracker import CLVTracker
from .types import EdgeReport, Sport, SystemState

logger = logging.getLogger(__name__)


# ── Configurable Thresholds ───────────────────────────────────────────

THRESHOLDS = {
    # CLV thresholds (in cents per dollar)
    "clv_reduce_window": 100,       # Window for REDUCED trigger
    "clv_suspend_window": 250,      # Window for SUSPENDED trigger
    "clv_reduce_threshold": 0.0,    # CLV below this → REDUCED
    "clv_suspend_threshold": 0.0,   # CLV below this over 250 → SUSPENDED

    # Calibration
    "calibration_error_retrain": 0.08,  # MAE > 8% → retrain
    "calibration_error_suspend": 0.15,  # MAE > 15% → suspend

    # Drawdown (% of peak bankroll)
    "drawdown_reduce": 0.20,        # 20% drawdown → REDUCED
    "drawdown_suspend": 0.35,       # 35% drawdown → SUSPENDED
    "drawdown_kill": 0.50,          # 50% drawdown → KILLED

    # Win rate deviation
    "win_rate_deviation_warn": 0.06,  # Actual win rate 6% below expected → warn
    "win_rate_deviation_suspend": 0.12,  # 12% below → suspend

    # Minimum sample sizes
    "min_bets_for_clv": 30,
    "min_bets_for_calibration": 50,
}


class EdgeValidator:
    """Daily edge validation engine."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        self.clv_tracker = CLVTracker(sport, db_path)
        self.calibration_monitor = CalibrationMonitor(sport, db_path)

    def _session(self):
        return get_session(self._db_path)

    def validate(self, bankroll: float, peak_bankroll: float) -> EdgeReport:
        """Run full edge validation. Returns EdgeReport with system state recommendation."""
        warnings = []
        actions = []
        recommended_state = SystemState.ACTIVE

        # ── 1. CLV Analysis ───────────────────────────────────────
        clv_summary = self.clv_tracker.clv_summary()

        clv_100 = clv_summary["clv_100"]
        clv_250 = clv_summary["clv_250"]
        clv_50 = clv_summary["clv_50"]
        clv_500 = clv_summary["clv_500"]

        # CLV-based state transitions
        if clv_100["n_bets"] >= THRESHOLDS["min_bets_for_clv"]:
            if clv_100["avg_clv_cents"] < THRESHOLDS["clv_reduce_threshold"]:
                warnings.append(f"CLV negative over last {clv_100['n_bets']} bets: {clv_100['avg_clv_cents']:.2f} cents")
                recommended_state = max(recommended_state, SystemState.REDUCED, key=lambda s: list(SystemState).index(s))
                actions.append("REDUCE bet sizes by 50%")

        if clv_250["n_bets"] >= THRESHOLDS["clv_suspend_window"] * 0.8:
            if clv_250["avg_clv_cents"] < THRESHOLDS["clv_suspend_threshold"]:
                warnings.append(f"CLV negative over last {clv_250['n_bets']} bets: {clv_250['avg_clv_cents']:.2f} cents — EDGE LIKELY GONE")
                recommended_state = max(recommended_state, SystemState.SUSPENDED, key=lambda s: list(SystemState).index(s))
                actions.append("STOP all betting until model is retrained and CLV recovers")

        # CLV trend
        if clv_100.get("trend") == "declining":
            warnings.append("CLV trend is DECLINING — edge may be eroding")

        # ── 2. Calibration Analysis ───────────────────────────────
        cal_report = self.calibration_monitor.compute_calibration()
        cal_error = cal_report.get("mean_absolute_error", 0.0)

        if cal_report.get("n_total", 0) >= THRESHOLDS["min_bets_for_calibration"]:
            if cal_error > THRESHOLDS["calibration_error_suspend"]:
                warnings.append(f"Calibration MAE = {cal_error:.3f} — model is severely miscalibrated")
                recommended_state = max(recommended_state, SystemState.SUSPENDED, key=lambda s: list(SystemState).index(s))
                actions.append("RETRAIN model immediately — calibration is broken")
            elif cal_error > THRESHOLDS["calibration_error_retrain"]:
                warnings.append(f"Calibration MAE = {cal_error:.3f} — model needs retraining")
                actions.append("Schedule model retrain within 48 hours")

            # Check for systematic overconfidence
            if cal_report.get("overconfidence_ratio", 0) > 0.7:
                warnings.append("Model is systematically OVERCONFIDENT in 70%+ of buckets")
                actions.append("Apply probability shrinkage or recalibrate")

        # ── 3. Drawdown Analysis ──────────────────────────────────
        drawdown_pct = 0.0
        if peak_bankroll > 0:
            drawdown_pct = (peak_bankroll - bankroll) / peak_bankroll

        if drawdown_pct > THRESHOLDS["drawdown_kill"]:
            warnings.append(f"CRITICAL DRAWDOWN: {drawdown_pct:.1%} — system should be killed")
            recommended_state = SystemState.KILLED
            actions.append("KILL system — drawdown exceeds survival threshold")
        elif drawdown_pct > THRESHOLDS["drawdown_suspend"]:
            warnings.append(f"Severe drawdown: {drawdown_pct:.1%}")
            recommended_state = max(recommended_state, SystemState.SUSPENDED, key=lambda s: list(SystemState).index(s))
            actions.append("SUSPEND betting until drawdown recovers below 25%")
        elif drawdown_pct > THRESHOLDS["drawdown_reduce"]:
            warnings.append(f"Elevated drawdown: {drawdown_pct:.1%}")
            recommended_state = max(recommended_state, SystemState.REDUCED, key=lambda s: list(SystemState).index(s))
            actions.append("REDUCE bet sizes proportional to drawdown")

        # ── 4. Win Rate vs Expected ───────────────────────────────
        win_rate_report = self._check_win_rate()
        if win_rate_report.get("deviation", 0) > THRESHOLDS["win_rate_deviation_suspend"]:
            warnings.append(f"Win rate {win_rate_report['deviation']:.1%} below expected — model may be broken")
            recommended_state = max(recommended_state, SystemState.SUSPENDED, key=lambda s: list(SystemState).index(s))
        elif win_rate_report.get("deviation", 0) > THRESHOLDS["win_rate_deviation_warn"]:
            warnings.append(f"Win rate {win_rate_report['deviation']:.1%} below expected")

        # ── 5. Model vs Market Comparison ─────────────────────────
        model_roi = self._compute_roi()
        expected_roi = self._compute_expected_roi()

        # ── Build Report ──────────────────────────────────────────
        edge_exists = (
            clv_100.get("clv_positive", False)
            and cal_error < THRESHOLDS["calibration_error_retrain"]
            and drawdown_pct < THRESHOLDS["drawdown_suspend"]
        )

        report = EdgeReport(
            report_date=datetime.utcnow(),
            sport=self.sport,
            clv_last_50=clv_50.get("avg_clv_cents", 0.0),
            clv_last_100=clv_100.get("avg_clv_cents", 0.0),
            clv_last_250=clv_250.get("avg_clv_cents", 0.0),
            clv_last_500=clv_500.get("avg_clv_cents", 0.0),
            calibration_error=cal_error,
            calibration_buckets=cal_report.get("buckets", []),
            model_roi=model_roi,
            expected_roi=expected_roi,
            edge_exists=edge_exists,
            system_state=recommended_state,
            warnings=warnings,
            actions=actions,
        )

        # Log state change if different from current
        self._log_state_change(recommended_state, warnings, bankroll, drawdown_pct,
                               clv_100.get("avg_clv_cents", 0.0))

        level = logging.WARNING if not edge_exists else logging.INFO
        logger.log(level, "Edge validation: state=%s | edge_exists=%s | CLV_100=%.2f | cal_error=%.3f | drawdown=%.1f%% | warnings=%d",
                   recommended_state.value, edge_exists,
                   clv_100.get("avg_clv_cents", 0.0), cal_error,
                   drawdown_pct * 100, len(warnings))

        return report

    def _check_win_rate(self) -> dict:
        """Compare actual vs expected win rate over last 200 bets."""
        session = self._session()
        try:
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.desc())
                .limit(200)
                .all()
            )
            if len(rows) < 30:
                return {"actual": 0.0, "expected": 0.0, "deviation": 0.0, "n": len(rows)}

            actual_wins = sum(1 for r in rows if r.status == "won")
            actual_rate = actual_wins / len(rows)
            expected_rate = float(np.mean([r.model_prob for r in rows]))
            deviation = max(0, expected_rate - actual_rate)

            return {
                "actual": round(actual_rate, 4),
                "expected": round(expected_rate, 4),
                "deviation": round(deviation, 4),
                "n": len(rows),
            }
        finally:
            session.close()

    def _compute_roi(self) -> float:
        """Actual ROI over all settled bets."""
        session = self._session()
        try:
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .all()
            )
            if not rows:
                return 0.0
            total_staked = sum(r.stake for r in rows)
            total_pnl = sum(r.pnl for r in rows)
            return round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0
        finally:
            session.close()

    def _compute_expected_roi(self) -> float:
        """Expected ROI based on model probabilities and odds."""
        session = self._session()
        try:
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .all()
            )
            if not rows:
                return 0.0
            total_staked = sum(r.stake for r in rows)
            expected_pnl = sum(
                r.stake * (r.model_prob * (r.odds_decimal - 1.0) - (1.0 - r.model_prob))
                for r in rows
            )
            return round(expected_pnl / total_staked, 4) if total_staked > 0 else 0.0
        finally:
            session.close()

    def _log_state_change(
        self,
        new_state: SystemState,
        warnings: list,
        bankroll: float,
        drawdown: float,
        clv: float,
    ) -> None:
        """Log state transition to audit trail."""
        session = self._session()
        try:
            # Get current state
            last_log = (
                session.query(SystemStateLog)
                .filter_by(sport=self.sport.value)
                .order_by(SystemStateLog.timestamp.desc())
                .first()
            )
            prev_state = last_log.new_state if last_log else SystemState.ACTIVE.value

            if prev_state != new_state.value:
                log = SystemStateLog(
                    sport=self.sport.value,
                    timestamp=datetime.utcnow(),
                    previous_state=prev_state,
                    new_state=new_state.value,
                    reason="; ".join(warnings[:3]) if warnings else "Routine validation",
                    clv_at_change=clv,
                    bankroll_at_change=bankroll,
                    drawdown_at_change=drawdown,
                )
                session.add(log)
                session.commit()
                logger.warning("SYSTEM STATE CHANGE: %s → %s | Reason: %s",
                               prev_state, new_state.value, warnings[:1])
        except Exception:
            session.rollback()
            logger.exception("Failed to log state change")
        finally:
            session.close()

    def get_current_state(self) -> SystemState:
        """Get the current system state."""
        session = self._session()
        try:
            last_log = (
                session.query(SystemStateLog)
                .filter_by(sport=self.sport.value)
                .order_by(SystemStateLog.timestamp.desc())
                .first()
            )
            if last_log:
                return SystemState(last_log.new_state)
            return SystemState.ACTIVE
        finally:
            session.close()
