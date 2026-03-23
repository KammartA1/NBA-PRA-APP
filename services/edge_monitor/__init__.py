"""
services/edge_monitor/
=======================
Continuous edge monitoring system that answers the daily question:
"Do I still have edge?"

Computes rolling metrics, detects trend changes via CUSUM, and
generates daily verdicts with configurable alert thresholds.
"""

from services.edge_monitor.daily_metrics import DailyEdgeMetrics
from services.edge_monitor.trend_detection import TrendDetector
from services.edge_monitor.alert_system import EdgeAlertSystem

__all__ = [
    "DailyEdgeMetrics",
    "TrendDetector",
    "EdgeAlertSystem",
]
