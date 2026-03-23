"""
services/edge_monitor/trend_detection.py
=========================================
CUSUM (Cumulative Sum) change-point detection for edge decay
and regime changes.

CUSUM detects shifts in the mean of a process. When CLV starts
declining, CUSUM will flag it before a simple rolling average notices.

Also implements:
  - Page's CUSUM for detecting both upward and downward shifts
  - Bayesian online change-point detection (simplified)
  - Regime classification (strong/normal/weak/dead edge)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

log = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    """A detected change point in the data."""
    index: int
    timestamp: str
    direction: str        # "up" or "down"
    magnitude: float      # Size of the shift
    confidence: float     # Confidence in the detection (0-1)
    metric_name: str      # Which metric changed


@dataclass
class RegimeState:
    """Current regime classification."""
    regime: str               # strong_edge, normal_edge, weak_edge, no_edge
    confidence: float
    entered_at_bet: int       # Bet index when this regime started
    duration_bets: int
    trend_direction: str      # improving, stable, declining
    change_points: List[ChangePoint]


class TrendDetector:
    """CUSUM-based trend detection for edge monitoring metrics.

    Uses Page's CUSUM algorithm which maintains two accumulators:
      S_high: detects upward shifts
      S_low:  detects downward shifts

    A signal fires when either accumulator exceeds a threshold h.
    """

    def __init__(
        self,
        target_mean: float = 0.0,
        allowable_slack: float = 0.5,
        threshold: float = 4.0,
    ):
        """
        Args:
            target_mean: Expected mean of the process (e.g., 0 for CLV in cents).
            allowable_slack: Slack parameter k — minimum shift to detect.
            threshold: Decision threshold h — higher = fewer false alarms.
        """
        self.target_mean = target_mean
        self.k = allowable_slack
        self.h = threshold

    def cusum(
        self,
        data: np.ndarray,
        target_mean: float | None = None,
        k: float | None = None,
        h: float | None = None,
    ) -> Dict[str, Any]:
        """Run CUSUM on a data series.

        Returns:
            Dict with S_high, S_low arrays, detected change points, and alarms.
        """
        mu = target_mean if target_mean is not None else self.target_mean
        slack = k if k is not None else self.k
        threshold = h if h is not None else self.h

        n = len(data)
        s_high = np.zeros(n)
        s_low = np.zeros(n)
        alarms_high: List[int] = []
        alarms_low: List[int] = []

        for i in range(1, n):
            s_high[i] = max(0, s_high[i - 1] + (data[i] - mu) - slack)
            s_low[i] = max(0, s_low[i - 1] - (data[i] - mu) - slack)

            if s_high[i] > threshold:
                alarms_high.append(i)
                s_high[i] = 0  # Reset after alarm
            if s_low[i] > threshold:
                alarms_low.append(i)
                s_low[i] = 0  # Reset after alarm

        return {
            "s_high": s_high,
            "s_low": s_low,
            "alarms_high": alarms_high,
            "alarms_low": alarms_low,
            "n_upshift_alarms": len(alarms_high),
            "n_downshift_alarms": len(alarms_low),
            "current_s_high": float(s_high[-1]) if n > 0 else 0.0,
            "current_s_low": float(s_low[-1]) if n > 0 else 0.0,
        }

    def detect_edge_decay(
        self,
        clv_series: np.ndarray,
        timestamps: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Detect edge decay in a CLV series.

        Specifically looks for:
          - Downward shift in mean CLV (edge degradation)
          - Upward shift in mean CLV (edge improvement)
          - Trend significance via linear regression
        """
        if len(clv_series) < 20:
            return {
                "is_decaying": False,
                "change_points": [],
                "trend_slope": 0.0,
                "trend_p_value": 1.0,
                "regime": "insufficient_data",
                "confidence": 0.0,
            }

        # CUSUM detection
        # For CLV, we care about downward shifts (edge decay)
        # Set target to the first-quarter mean as baseline
        baseline_n = max(len(clv_series) // 4, 10)
        baseline_mean = float(np.mean(clv_series[:baseline_n]))

        # Scale slack by data standard deviation
        data_std = float(np.std(clv_series, ddof=1))
        adaptive_k = max(data_std * 0.5, 0.1)
        adaptive_h = max(data_std * 3.0, 1.0)

        cusum_result = self.cusum(clv_series, target_mean=baseline_mean, k=adaptive_k, h=adaptive_h)

        # Linear regression for overall trend
        x = np.arange(len(clv_series))
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, clv_series)

        # Build change points
        change_points: List[ChangePoint] = []
        ts = timestamps or [str(i) for i in range(len(clv_series))]

        for idx in cusum_result["alarms_low"]:
            if idx < len(ts):
                # Estimate shift magnitude
                pre_mean = float(np.mean(clv_series[max(0, idx - 20):idx]))
                post_end = min(idx + 20, len(clv_series))
                post_mean = float(np.mean(clv_series[idx:post_end]))
                change_points.append(ChangePoint(
                    index=idx,
                    timestamp=ts[idx],
                    direction="down",
                    magnitude=round(pre_mean - post_mean, 3),
                    confidence=min(abs(pre_mean - post_mean) / max(data_std, 0.01), 1.0),
                    metric_name="clv",
                ))

        for idx in cusum_result["alarms_high"]:
            if idx < len(ts):
                pre_mean = float(np.mean(clv_series[max(0, idx - 20):idx]))
                post_end = min(idx + 20, len(clv_series))
                post_mean = float(np.mean(clv_series[idx:post_end]))
                change_points.append(ChangePoint(
                    index=idx,
                    timestamp=ts[idx],
                    direction="up",
                    magnitude=round(post_mean - pre_mean, 3),
                    confidence=min(abs(post_mean - pre_mean) / max(data_std, 0.01), 1.0),
                    metric_name="clv",
                ))

        # Regime classification
        recent_clv = float(np.mean(clv_series[-50:])) if len(clv_series) >= 50 else float(np.mean(clv_series))
        if recent_clv > 1.5:
            regime = "strong_edge"
        elif recent_clv > 0.5:
            regime = "normal_edge"
        elif recent_clv > -0.5:
            regime = "weak_edge"
        else:
            regime = "no_edge"

        # Is it decaying?
        is_decaying = slope < 0 and p_value < 0.05

        # Trend direction
        if slope > 0 and p_value < 0.10:
            trend = "improving"
        elif slope < 0 and p_value < 0.10:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "is_decaying": is_decaying,
            "change_points": change_points,
            "trend_slope": round(float(slope), 6),
            "trend_p_value": round(float(p_value), 6),
            "trend_r_squared": round(float(r_value ** 2), 4),
            "regime": regime,
            "trend_direction": trend,
            "confidence": round(1.0 - float(p_value), 4),
            "baseline_mean": round(baseline_mean, 3),
            "current_mean": round(recent_clv, 3),
            "cusum_s_high": round(cusum_result["current_s_high"], 3),
            "cusum_s_low": round(cusum_result["current_s_low"], 3),
        }

    def detect_metric_shifts(
        self,
        metric_series: np.ndarray,
        metric_name: str = "generic",
        target_mean: float | None = None,
    ) -> Dict[str, Any]:
        """Generic shift detection for any metric series."""
        if len(metric_series) < 10:
            return {
                "metric": metric_name,
                "shifts_detected": 0,
                "change_points": [],
                "current_mean": 0.0,
                "trend": "insufficient_data",
            }

        # Auto-detect target mean if not provided
        if target_mean is None:
            target_mean = float(np.mean(metric_series[:max(len(metric_series) // 4, 5)]))

        data_std = float(np.std(metric_series, ddof=1))
        adaptive_k = max(data_std * 0.5, 0.001)
        adaptive_h = max(data_std * 3.0, 0.01)

        cusum_result = self.cusum(metric_series, target_mean, adaptive_k, adaptive_h)

        total_shifts = cusum_result["n_upshift_alarms"] + cusum_result["n_downshift_alarms"]

        # Recent trend
        x = np.arange(len(metric_series))
        slope, _, _, p_val, _ = sp_stats.linregress(x, metric_series)

        if slope > 0 and p_val < 0.10:
            trend = "increasing"
        elif slope < 0 and p_val < 0.10:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "metric": metric_name,
            "shifts_detected": total_shifts,
            "upshift_count": cusum_result["n_upshift_alarms"],
            "downshift_count": cusum_result["n_downshift_alarms"],
            "current_mean": round(float(np.mean(metric_series[-20:])), 4),
            "baseline_mean": round(target_mean, 4),
            "trend": trend,
            "trend_slope": round(float(slope), 6),
            "trend_p_value": round(float(p_val), 6),
        }

    def classify_regime(
        self,
        clv_series: np.ndarray,
        brier_series: np.ndarray | None = None,
        roi_series: np.ndarray | None = None,
    ) -> RegimeState:
        """Classify the current edge regime using multiple signals."""
        if len(clv_series) < 10:
            return RegimeState(
                regime="insufficient_data",
                confidence=0.0,
                entered_at_bet=0,
                duration_bets=0,
                trend_direction="unknown",
                change_points=[],
            )

        decay_result = self.detect_edge_decay(clv_series)
        regime = decay_result["regime"]
        trend = decay_result["trend_direction"]

        # Refine with Brier score if available
        if brier_series is not None and len(brier_series) >= 20:
            recent_brier = float(np.mean(brier_series[-20:]))
            if recent_brier > 0.28:  # Poor calibration
                if regime == "strong_edge":
                    regime = "normal_edge"
                elif regime == "normal_edge":
                    regime = "weak_edge"

        # Refine with ROI if available
        if roi_series is not None and len(roi_series) >= 20:
            recent_roi = float(np.mean(roi_series[-20:]))
            if recent_roi < -5.0:  # Losing money
                regime = "no_edge"

        # Estimate when this regime started
        n = len(clv_series)
        regime_start = 0
        window = 20
        for i in range(n - window, -1, -window):
            segment_mean = float(np.mean(clv_series[max(0, i):i + window]))
            if regime == "strong_edge" and segment_mean <= 1.5:
                regime_start = min(i + window, n - 1)
                break
            elif regime == "no_edge" and segment_mean > -0.5:
                regime_start = min(i + window, n - 1)
                break
            elif regime == "normal_edge" and (segment_mean <= 0.5 or segment_mean > 1.5):
                regime_start = min(i + window, n - 1)
                break
            elif regime == "weak_edge" and (segment_mean <= -0.5 or segment_mean > 0.5):
                regime_start = min(i + window, n - 1)
                break

        return RegimeState(
            regime=regime,
            confidence=decay_result["confidence"],
            entered_at_bet=regime_start,
            duration_bets=n - regime_start,
            trend_direction=trend,
            change_points=decay_result["change_points"],
        )
