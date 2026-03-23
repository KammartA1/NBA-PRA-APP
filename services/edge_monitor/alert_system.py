"""
services/edge_monitor/alert_system.py
======================================
Daily verdict and alert system for edge monitoring.

Generates a daily EDGE = YES / NO verdict based on all available metrics.
If NO for 3 consecutive days, triggers a CRITICAL alert.

Alert levels:
  - INFO: Metric noteworthy but not concerning
  - WARNING: Metric approaching danger zone
  - CRITICAL: Immediate action required
  - SYSTEM_HALT: Kill switch territory
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from database.connection import session_scope
from database.models import EdgeReport
from services.edge_monitor.daily_metrics import DailyEdgeMetrics, EdgeMetricsSnapshot
from services.edge_monitor.trend_detection import TrendDetector

log = logging.getLogger(__name__)


@dataclass
class Alert:
    """A single monitoring alert."""
    level: str           # INFO, WARNING, CRITICAL, SYSTEM_HALT
    category: str        # clv, calibration, roi, variance, drawdown
    message: str
    metric_name: str
    current_value: float
    threshold: float
    recommendation: str


@dataclass
class DailyVerdict:
    """Daily edge verdict."""
    date: str
    has_edge: bool                   # YES or NO
    confidence: float                # 0-1
    edge_score: float                # Composite score 0-100
    alerts: List[Alert]
    metrics: EdgeMetricsSnapshot
    regime: str
    trend: str
    consecutive_no_edge_days: int
    is_critical: bool                # True if 3+ consecutive NO days
    recommended_action: str          # CONTINUE, REDUCE, SUSPEND, HALT


class EdgeAlertSystem:
    """Daily edge verdict and alert generation system.

    Combines metrics from DailyEdgeMetrics and trend detection from
    TrendDetector to produce a single daily verdict.
    """

    # Thresholds for alert generation
    CLV_WARNING = 0.0          # CLV below this → WARNING
    CLV_CRITICAL = -0.5        # CLV below this → CRITICAL
    BRIER_WARNING = 0.26       # Brier above this → WARNING
    BRIER_CRITICAL = 0.28      # Brier above this → CRITICAL
    ROI_WARNING = -2.0         # ROI below this → WARNING
    ROI_CRITICAL = -5.0        # ROI below this → CRITICAL
    VARIANCE_WARNING = 2.0     # Variance ratio above this → WARNING
    VARIANCE_CRITICAL = 3.0    # Variance ratio above this → CRITICAL
    DRAWDOWN_WARNING = 15.0    # Drawdown above this → WARNING
    DRAWDOWN_CRITICAL = 25.0   # Drawdown above this → CRITICAL
    CONSECUTIVE_NO_EDGE_CRITICAL = 3

    def __init__(self, sport: str = "NBA"):
        self.sport = sport
        self.metrics_engine = DailyEdgeMetrics(sport=sport)
        self.trend_detector = TrendDetector()

    def generate_daily_verdict(self) -> DailyVerdict:
        """Generate today's edge verdict."""
        metrics = self.metrics_engine.compute()
        alerts = self._generate_alerts(metrics)

        # Compute edge score (0-100)
        edge_score = self._compute_edge_score(metrics)

        # Determine verdict
        has_edge = edge_score >= 40  # Below 40 = no edge

        # Get trend info
        rolling = self.metrics_engine.compute_rolling_series(window=50)
        if rolling["clv"] and len(rolling["clv"]) >= 10:
            clv_arr = np.array(rolling["clv"])
            trend_result = self.trend_detector.detect_edge_decay(clv_arr)
            regime = trend_result["regime"]
            trend = trend_result["trend_direction"]
        else:
            regime = "insufficient_data"
            trend = "unknown"

        # Check consecutive no-edge days
        consecutive = self._get_consecutive_no_edge_days(has_edge)

        is_critical = consecutive >= self.CONSECUTIVE_NO_EDGE_CRITICAL

        # Recommended action
        if is_critical:
            action = "HALT"
        elif any(a.level == "SYSTEM_HALT" for a in alerts):
            action = "HALT"
        elif any(a.level == "CRITICAL" for a in alerts):
            action = "SUSPEND"
        elif not has_edge:
            action = "REDUCE"
        elif any(a.level == "WARNING" for a in alerts):
            action = "CONTINUE"  # Continue but monitor
        else:
            action = "CONTINUE"

        verdict = DailyVerdict(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            has_edge=has_edge,
            confidence=round(edge_score / 100.0, 3),
            edge_score=round(edge_score, 1),
            alerts=alerts,
            metrics=metrics,
            regime=regime,
            trend=trend,
            consecutive_no_edge_days=consecutive,
            is_critical=is_critical,
            recommended_action=action,
        )

        # Persist verdict
        self._save_verdict(verdict)

        return verdict

    def _generate_alerts(self, metrics: EdgeMetricsSnapshot) -> List[Alert]:
        """Generate alerts from current metrics."""
        alerts: List[Alert] = []

        # CLV alerts
        if metrics.clv_50_avg < self.CLV_CRITICAL:
            alerts.append(Alert(
                level="CRITICAL",
                category="clv",
                message=f"50-bet CLV is {metrics.clv_50_avg:.2f} cents — below critical threshold",
                metric_name="clv_50_avg",
                current_value=metrics.clv_50_avg,
                threshold=self.CLV_CRITICAL,
                recommendation="Reduce bet sizing immediately. Investigate edge sources.",
            ))
        elif metrics.clv_50_avg < self.CLV_WARNING:
            alerts.append(Alert(
                level="WARNING",
                category="clv",
                message=f"50-bet CLV is {metrics.clv_50_avg:.2f} cents — approaching zero",
                metric_name="clv_50_avg",
                current_value=metrics.clv_50_avg,
                threshold=self.CLV_WARNING,
                recommendation="Monitor CLV closely. Consider reducing exposure.",
            ))

        if metrics.settled_bets >= 200 and metrics.clv_200_avg < self.CLV_CRITICAL:
            alerts.append(Alert(
                level="SYSTEM_HALT",
                category="clv",
                message=f"200-bet CLV is {metrics.clv_200_avg:.2f} cents — sustained negative CLV",
                metric_name="clv_200_avg",
                current_value=metrics.clv_200_avg,
                threshold=self.CLV_CRITICAL,
                recommendation="HALT betting. CLV has been negative over a significant sample.",
            ))

        # Brier score alerts
        if metrics.brier_score > self.BRIER_CRITICAL:
            alerts.append(Alert(
                level="CRITICAL",
                category="calibration",
                message=f"Brier score {metrics.brier_score:.4f} exceeds critical threshold",
                metric_name="brier_score",
                current_value=metrics.brier_score,
                threshold=self.BRIER_CRITICAL,
                recommendation="Model calibration has degraded significantly. Retrain or recalibrate.",
            ))
        elif metrics.brier_score > self.BRIER_WARNING:
            alerts.append(Alert(
                level="WARNING",
                category="calibration",
                message=f"Brier score {metrics.brier_score:.4f} above warning threshold",
                metric_name="brier_score",
                current_value=metrics.brier_score,
                threshold=self.BRIER_WARNING,
                recommendation="Monitor calibration. Consider recalibration.",
            ))

        # Brier vs market
        if metrics.brier_advantage > 0.01 and metrics.settled_bets >= 100:
            alerts.append(Alert(
                level="CRITICAL",
                category="calibration",
                message=f"Model Brier ({metrics.brier_score:.4f}) worse than market ({metrics.market_brier_score:.4f})",
                metric_name="brier_advantage",
                current_value=metrics.brier_advantage,
                threshold=0.0,
                recommendation="Model is WORSE than market implied probabilities. Investigate immediately.",
            ))

        # ROI alerts
        if metrics.roi_50_pct < self.ROI_CRITICAL:
            alerts.append(Alert(
                level="CRITICAL",
                category="roi",
                message=f"50-bet ROI is {metrics.roi_50_pct:.1f}% — significant losses",
                metric_name="roi_50_pct",
                current_value=metrics.roi_50_pct,
                threshold=self.ROI_CRITICAL,
                recommendation="Reduce stake sizes. Review recent bet quality.",
            ))
        elif metrics.roi_50_pct < self.ROI_WARNING:
            alerts.append(Alert(
                level="WARNING",
                category="roi",
                message=f"50-bet ROI is {metrics.roi_50_pct:.1f}%",
                metric_name="roi_50_pct",
                current_value=metrics.roi_50_pct,
                threshold=self.ROI_WARNING,
                recommendation="ROI trending negative. Monitor closely.",
            ))

        # Variance ratio alerts
        if metrics.variance_ratio > self.VARIANCE_CRITICAL:
            alerts.append(Alert(
                level="CRITICAL",
                category="variance",
                message=f"Variance ratio {metrics.variance_ratio:.1f}x — actual variance far exceeds expected",
                metric_name="variance_ratio",
                current_value=metrics.variance_ratio,
                threshold=self.VARIANCE_CRITICAL,
                recommendation="Reduce to 50% Kelly. Variance explosion indicates model misspecification.",
            ))
        elif metrics.variance_ratio > self.VARIANCE_WARNING:
            alerts.append(Alert(
                level="WARNING",
                category="variance",
                message=f"Variance ratio {metrics.variance_ratio:.1f}x above expected",
                metric_name="variance_ratio",
                current_value=metrics.variance_ratio,
                threshold=self.VARIANCE_WARNING,
                recommendation="Consider reducing bet sizes.",
            ))

        # Drawdown alerts
        if metrics.current_drawdown_pct > self.DRAWDOWN_CRITICAL:
            alerts.append(Alert(
                level="SYSTEM_HALT",
                category="drawdown",
                message=f"Drawdown {metrics.current_drawdown_pct:.1f}% exceeds critical limit",
                metric_name="current_drawdown_pct",
                current_value=metrics.current_drawdown_pct,
                threshold=self.DRAWDOWN_CRITICAL,
                recommendation="HALT all betting. Drawdown is at dangerous levels.",
            ))
        elif metrics.current_drawdown_pct > self.DRAWDOWN_WARNING:
            alerts.append(Alert(
                level="WARNING",
                category="drawdown",
                message=f"Drawdown at {metrics.current_drawdown_pct:.1f}%",
                metric_name="current_drawdown_pct",
                current_value=metrics.current_drawdown_pct,
                threshold=self.DRAWDOWN_WARNING,
                recommendation="Reduce exposure. Monitor bankroll closely.",
            ))

        return alerts

    def _compute_edge_score(self, metrics: EdgeMetricsSnapshot) -> float:
        """Compute composite edge score (0-100).

        Components:
          - CLV contribution (0-35)
          - Calibration contribution (0-25)
          - ROI contribution (0-25)
          - Win rate contribution (0-15)
        """
        score = 0.0

        # CLV (0-35)
        if metrics.clv_50_avg > 2.0:
            score += 35.0
        elif metrics.clv_50_avg > 1.0:
            score += 25.0 + (metrics.clv_50_avg - 1.0) * 10.0
        elif metrics.clv_50_avg > 0.0:
            score += 15.0 + metrics.clv_50_avg * 10.0
        elif metrics.clv_50_avg > -1.0:
            score += max(15.0 + metrics.clv_50_avg * 15.0, 0.0)

        # Calibration (0-25)
        if metrics.brier_advantage < -0.02:
            score += 25.0  # Much better than market
        elif metrics.brier_advantage < 0.0:
            score += 15.0 + abs(metrics.brier_advantage) * 500
        elif metrics.brier_advantage < 0.02:
            score += 10.0
        else:
            score += max(10.0 - metrics.brier_advantage * 200, 0.0)

        # ROI (0-25)
        if metrics.roi_50_pct > 10:
            score += 25.0
        elif metrics.roi_50_pct > 5:
            score += 20.0
        elif metrics.roi_50_pct > 0:
            score += 10.0 + metrics.roi_50_pct * 2.0
        elif metrics.roi_50_pct > -5:
            score += max(10.0 + metrics.roi_50_pct * 2.0, 0.0)

        # Win rate (0-15)
        excess_wr = metrics.win_rate_50 - 0.50
        if excess_wr > 0.08:
            score += 15.0
        elif excess_wr > 0.04:
            score += 10.0
        elif excess_wr > 0:
            score += excess_wr * 125
        else:
            score += max(5.0 + excess_wr * 50, 0.0)

        return min(max(score, 0.0), 100.0)

    def _get_consecutive_no_edge_days(self, today_has_edge: bool) -> int:
        """Get count of consecutive days without edge (including today)."""
        if today_has_edge:
            return 0

        consecutive = 1
        try:
            with session_scope() as session:
                recent = (
                    session.query(EdgeReport)
                    .filter(EdgeReport.report_type == "daily_verdict")
                    .filter(EdgeReport.sport == self.sport)
                    .order_by(EdgeReport.generated_at.desc())
                    .limit(10)
                    .all()
                )
                for report in recent:
                    data = json.loads(report.report_json or "{}")
                    if not data.get("has_edge", True):
                        consecutive += 1
                    else:
                        break
        except Exception as e:
            log.warning("Failed to check consecutive verdicts: %s", e)

        return consecutive

    def _save_verdict(self, verdict: DailyVerdict) -> None:
        """Persist verdict to database."""
        try:
            report_data = {
                "date": verdict.date,
                "has_edge": verdict.has_edge,
                "edge_score": verdict.edge_score,
                "confidence": verdict.confidence,
                "regime": verdict.regime,
                "trend": verdict.trend,
                "consecutive_no_edge_days": verdict.consecutive_no_edge_days,
                "is_critical": verdict.is_critical,
                "recommended_action": verdict.recommended_action,
                "n_alerts": len(verdict.alerts),
                "alert_levels": [a.level for a in verdict.alerts],
                "clv_50": verdict.metrics.clv_50_avg,
                "brier": verdict.metrics.brier_score,
                "roi_50": verdict.metrics.roi_50_pct,
                "win_rate": verdict.metrics.win_rate,
            }
            with session_scope() as session:
                session.add(EdgeReport(
                    report_type="daily_verdict",
                    sport=self.sport,
                    report_json=json.dumps(report_data),
                ))
        except Exception as e:
            log.warning("Failed to save verdict: %s", e)

    def get_verdict_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Retrieve recent verdict history."""
        try:
            with session_scope() as session:
                reports = (
                    session.query(EdgeReport)
                    .filter(EdgeReport.report_type == "daily_verdict")
                    .filter(EdgeReport.sport == self.sport)
                    .order_by(EdgeReport.generated_at.desc())
                    .limit(days)
                    .all()
                )
                return [json.loads(r.report_json or "{}") for r in reports]
        except Exception as e:
            log.warning("Failed to load verdict history: %s", e)
            return []
