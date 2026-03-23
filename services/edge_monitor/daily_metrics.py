"""
services/edge_monitor/daily_metrics.py
=======================================
Computes all daily edge monitoring metrics:
  - CLV trend (50-bet and 200-bet rolling averages)
  - Brier score vs market
  - Log loss
  - Calibration drift
  - ROI trend
  - Win rate
  - Variance ratio (actual vs expected)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from database.connection import session_scope
from database.models import Bet

log = logging.getLogger(__name__)


@dataclass
class EdgeMetricsSnapshot:
    """Point-in-time snapshot of all edge metrics."""
    timestamp: datetime

    # CLV metrics
    clv_50_avg: float          # 50-bet rolling avg CLV (cents)
    clv_200_avg: float         # 200-bet rolling avg CLV (cents)
    clv_total_avg: float       # Lifetime avg CLV
    clv_beat_rate: float       # % of bets that beat close

    # Calibration metrics
    brier_score: float         # Our Brier score
    market_brier_score: float  # Market Brier score (benchmark)
    brier_advantage: float     # Our Brier - market Brier (negative = better)
    log_loss: float            # Log loss of predictions
    calibration_error: float   # Mean absolute calibration error

    # Performance metrics
    roi_50_pct: float          # 50-bet rolling ROI %
    roi_200_pct: float         # 200-bet rolling ROI %
    roi_total_pct: float       # Lifetime ROI %
    win_rate: float            # Overall win rate
    win_rate_50: float         # 50-bet rolling win rate

    # Risk metrics
    variance_ratio: float      # Actual / expected variance
    current_drawdown_pct: float
    max_drawdown_pct: float

    # Sample sizes
    total_bets: int
    settled_bets: int


class DailyEdgeMetrics:
    """Compute daily edge monitoring metrics from bet history.

    Pulls data from the database and computes all metrics needed
    for the edge monitoring dashboard and alert system.
    """

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def compute(self) -> EdgeMetricsSnapshot:
        """Compute current edge metrics from all available bet data."""
        bets = self._load_bets()
        if not bets:
            return self._empty_snapshot()

        settled = [b for b in bets if b.get("status") == "settled" and b.get("profit") is not None]
        if len(settled) < 5:
            return self._empty_snapshot()

        # Sort by timestamp
        settled.sort(key=lambda x: x.get("timestamp", ""))

        # Extract arrays
        profits = np.array([b["profit"] for b in settled])
        stakes = np.array([max(b.get("stake", 1.0), 0.01) for b in settled])
        pred_probs = np.array([b.get("predicted_prob", 0.5) for b in settled])
        outcomes = np.array([1.0 if b.get("profit", 0) > 0 else 0.0 for b in settled])

        # CLV
        clv_values = self._compute_clv_series(settled)
        clv_50 = float(np.mean(clv_values[-50:])) if len(clv_values) >= 50 else float(np.mean(clv_values))
        clv_200 = float(np.mean(clv_values[-200:])) if len(clv_values) >= 200 else float(np.mean(clv_values))
        clv_total = float(np.mean(clv_values)) if clv_values.size > 0 else 0.0
        clv_beat_rate = float(np.mean(clv_values > 0)) if clv_values.size > 0 else 0.5

        # Brier score
        brier = self._brier_score(pred_probs, outcomes)
        market_brier = self._market_brier_score(settled)
        brier_advantage = brier - market_brier

        # Log loss
        log_loss = self._log_loss(pred_probs, outcomes)

        # Calibration error
        cal_error = self._calibration_error(pred_probs, outcomes)

        # ROI
        roi_50 = self._rolling_roi(profits[-50:], stakes[-50:])
        roi_200 = self._rolling_roi(profits[-200:], stakes[-200:])
        roi_total = self._rolling_roi(profits, stakes)

        # Win rate
        win_rate = float(np.mean(outcomes))
        win_rate_50 = float(np.mean(outcomes[-50:])) if len(outcomes) >= 50 else win_rate

        # Variance ratio
        variance_ratio = self._variance_ratio(profits, pred_probs)

        # Drawdown
        equity = np.cumsum(profits)
        peak = np.maximum.accumulate(equity)
        drawdowns = np.where(peak > 0, (peak - equity) / peak * 100, 0.0)
        current_dd = float(drawdowns[-1]) if len(drawdowns) > 0 else 0.0
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return EdgeMetricsSnapshot(
            timestamp=datetime.now(timezone.utc),
            clv_50_avg=round(clv_50, 3),
            clv_200_avg=round(clv_200, 3),
            clv_total_avg=round(clv_total, 3),
            clv_beat_rate=round(clv_beat_rate, 4),
            brier_score=round(brier, 6),
            market_brier_score=round(market_brier, 6),
            brier_advantage=round(brier_advantage, 6),
            log_loss=round(log_loss, 6),
            calibration_error=round(cal_error, 4),
            roi_50_pct=round(roi_50, 2),
            roi_200_pct=round(roi_200, 2),
            roi_total_pct=round(roi_total, 2),
            win_rate=round(win_rate, 4),
            win_rate_50=round(win_rate_50, 4),
            variance_ratio=round(variance_ratio, 3),
            current_drawdown_pct=round(current_dd, 2),
            max_drawdown_pct=round(max_dd, 2),
            total_bets=len(bets),
            settled_bets=len(settled),
        )

    def compute_rolling_series(self, window: int = 50) -> Dict[str, Any]:
        """Compute rolling metric series for charting."""
        bets = self._load_bets()
        settled = [b for b in bets if b.get("status") == "settled" and b.get("profit") is not None]
        settled.sort(key=lambda x: x.get("timestamp", ""))

        if len(settled) < window:
            return {"timestamps": [], "clv": [], "roi": [], "brier": [], "win_rate": []}

        profits = np.array([b["profit"] for b in settled])
        stakes = np.array([max(b.get("stake", 1.0), 0.01) for b in settled])
        pred_probs = np.array([b.get("predicted_prob", 0.5) for b in settled])
        outcomes = np.array([1.0 if b.get("profit", 0) > 0 else 0.0 for b in settled])
        clv_vals = self._compute_clv_series(settled)
        timestamps = [b.get("timestamp", "") for b in settled]

        n = len(settled)
        rolling_clv = []
        rolling_roi = []
        rolling_brier = []
        rolling_wr = []
        rolling_ts = []

        for i in range(window - 1, n):
            start = i - window + 1
            rolling_clv.append(float(np.mean(clv_vals[start:i + 1])))
            rolling_roi.append(self._rolling_roi(profits[start:i + 1], stakes[start:i + 1]))
            rolling_brier.append(self._brier_score(pred_probs[start:i + 1], outcomes[start:i + 1]))
            rolling_wr.append(float(np.mean(outcomes[start:i + 1])))
            rolling_ts.append(timestamps[i])

        return {
            "timestamps": rolling_ts,
            "clv": rolling_clv,
            "roi": rolling_roi,
            "brier": rolling_brier,
            "win_rate": rolling_wr,
        }

    def _load_bets(self) -> List[Dict[str, Any]]:
        """Load all bets from database."""
        try:
            with session_scope() as session:
                bets = session.query(Bet).filter(
                    Bet.sport == self.sport
                ).order_by(Bet.timestamp.asc()).all()
                return [b.to_dict() for b in bets]
        except Exception as e:
            log.warning("Failed to load bets: %s", e)
            return []

    def _compute_clv_series(self, settled: List[Dict]) -> np.ndarray:
        """Compute CLV for each settled bet."""
        clv_values = []
        for b in settled:
            bl = b.get("bet_line")
            cl = b.get("closing_line")
            direction = b.get("direction", "over")
            if bl is not None and cl is not None:
                if direction.lower() == "over":
                    clv = cl - bl
                else:
                    clv = bl - cl
                clv_values.append(clv)
            else:
                clv_values.append(0.0)
        return np.array(clv_values)

    def _brier_score(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Compute Brier score (lower is better)."""
        if len(predicted) == 0:
            return 1.0
        return float(np.mean((predicted - actual) ** 2))

    def _market_brier_score(self, settled: List[Dict]) -> float:
        """Compute market's Brier score using implied probabilities."""
        market_probs = []
        outcomes = []
        for b in settled:
            odds = b.get("odds_decimal")
            if odds and odds > 0:
                implied = 1.0 / odds
                market_probs.append(implied)
                outcomes.append(1.0 if (b.get("profit", 0) or 0) > 0 else 0.0)
        if not market_probs:
            return 0.25  # Default
        return float(np.mean((np.array(market_probs) - np.array(outcomes)) ** 2))

    def _log_loss(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Compute log loss (lower is better)."""
        if len(predicted) == 0:
            return 1.0
        p = np.clip(predicted, 1e-7, 1 - 1e-7)
        return float(-np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))

    def _calibration_error(self, predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """Compute mean absolute calibration error using binning."""
        if len(predicted) < n_bins:
            return 0.0

        bin_edges = np.linspace(0, 1, n_bins + 1)
        total_error = 0.0
        n_nonempty = 0

        for i in range(n_bins):
            mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
            if np.any(mask):
                bin_pred = np.mean(predicted[mask])
                bin_actual = np.mean(actual[mask])
                total_error += abs(bin_pred - bin_actual)
                n_nonempty += 1

        return total_error / max(n_nonempty, 1)

    def _rolling_roi(self, profits: np.ndarray, stakes: np.ndarray) -> float:
        """Compute ROI percentage."""
        total_stake = np.sum(stakes)
        if total_stake <= 0:
            return 0.0
        return float(np.sum(profits) / total_stake * 100)

    def _variance_ratio(self, profits: np.ndarray, pred_probs: np.ndarray) -> float:
        """Compute variance ratio: actual variance / expected variance."""
        if len(profits) < 10:
            return 1.0
        actual_var = float(np.var(profits, ddof=1))
        # Expected variance under the model
        avg_stake = float(np.mean(np.abs(profits)))
        avg_prob = float(np.mean(pred_probs))
        expected_var = avg_prob * (1 - avg_prob) * (avg_stake ** 2)
        return actual_var / max(expected_var, 1e-10)

    def _empty_snapshot(self) -> EdgeMetricsSnapshot:
        return EdgeMetricsSnapshot(
            timestamp=datetime.now(timezone.utc),
            clv_50_avg=0.0, clv_200_avg=0.0, clv_total_avg=0.0, clv_beat_rate=0.5,
            brier_score=0.25, market_brier_score=0.25, brier_advantage=0.0,
            log_loss=0.693, calibration_error=0.0,
            roi_50_pct=0.0, roi_200_pct=0.0, roi_total_pct=0.0,
            win_rate=0.5, win_rate_50=0.5,
            variance_ratio=1.0, current_drawdown_pct=0.0, max_drawdown_pct=0.0,
            total_bets=0, settled_bets=0,
        )
