"""Model Drift Detection — Detects when the model's assumptions no longer hold.

Models degrade over time because:
1. Market adapts to your edge (if you're big enough)
2. Underlying data distributions shift (rule changes, player behavior)
3. Feature relationships change (what predicted well before doesn't now)
4. Sample characteristics shift (different player pool, different venues)

Detection methods:
- Population Stability Index (PSI) on feature distributions
- Prediction distribution shift (KS test)
- Rolling accuracy degradation
- Feature importance drift
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
from scipy import stats as sp_stats

from ..db.schema import BetLog, FeatureLog, get_session
from ..core.types import Sport

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects model drift across multiple dimensions."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def prediction_distribution_shift(self, window_recent: int = 100, window_baseline: int = 300) -> dict:
        """Compare recent prediction distribution to baseline using KS test.

        If the distribution of model probabilities has shifted, the model
        is likely miscalibrated or the data has changed.
        """
        session = self._session()
        try:
            all_bets = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.desc())
                .limit(window_recent + window_baseline)
                .all()
            )

            if len(all_bets) < window_recent + 50:
                return {"drift_detected": False, "message": "Insufficient data"}

            recent_probs = [r.model_prob for r in all_bets[:window_recent]]
            baseline_probs = [r.model_prob for r in all_bets[window_recent:window_recent + window_baseline]]

            # KS test: are these from the same distribution?
            ks_stat, p_value = sp_stats.ks_2samp(recent_probs, baseline_probs)

            # PSI (Population Stability Index)
            psi = self._compute_psi(baseline_probs, recent_probs)

            drift_detected = p_value < 0.05 or psi > 0.20

            return {
                "drift_detected": drift_detected,
                "ks_statistic": round(float(ks_stat), 4),
                "ks_p_value": round(float(p_value), 4),
                "psi": round(psi, 4),
                "psi_interpretation": (
                    "No drift" if psi < 0.10 else
                    "Moderate drift" if psi < 0.25 else
                    "Significant drift"
                ),
                "recent_mean_prob": round(float(np.mean(recent_probs)), 4),
                "baseline_mean_prob": round(float(np.mean(baseline_probs)), 4),
                "recent_std": round(float(np.std(recent_probs)), 4),
                "baseline_std": round(float(np.std(baseline_probs)), 4),
            }
        finally:
            session.close()

    def accuracy_degradation(self, window: int = 50, stride: int = 25) -> dict:
        """Track rolling accuracy over time to detect degradation.

        Computes win rate in sliding windows and checks for downward trend.
        """
        session = self._session()
        try:
            bets = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.asc())
                .all()
            )

            if len(bets) < window * 2:
                return {"degradation_detected": False, "message": "Insufficient data"}

            # Sliding window accuracy
            windows = []
            for i in range(0, len(bets) - window, stride):
                chunk = bets[i:i + window]
                win_rate = sum(1 for b in chunk if b.status == "won") / len(chunk)
                avg_edge = float(np.mean([b.edge for b in chunk]))
                windows.append({
                    "start_idx": i,
                    "win_rate": round(win_rate, 4),
                    "avg_edge": round(avg_edge, 4),
                    "n_bets": len(chunk),
                })

            if len(windows) < 3:
                return {"degradation_detected": False, "message": "Not enough windows"}

            # Linear regression on win rates
            x = np.arange(len(windows))
            y = np.array([w["win_rate"] for w in windows])
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

            degradation = slope < -0.005 and p_value < 0.10

            return {
                "degradation_detected": degradation,
                "slope": round(float(slope), 6),
                "r_squared": round(float(r_value ** 2), 4),
                "p_value": round(float(p_value), 4),
                "current_win_rate": windows[-1]["win_rate"],
                "initial_win_rate": windows[0]["win_rate"],
                "n_windows": len(windows),
                "windows": windows,
            }
        finally:
            session.close()

    def edge_decay(self) -> dict:
        """Detect if the model's edge is decaying over time.

        Edge decay manifests as:
        - Decreasing average edge per bet
        - Decreasing CLV
        - Increasing calibration error
        """
        session = self._session()
        try:
            bets = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.asc())
                .all()
            )

            if len(bets) < 100:
                return {"edge_decay_detected": False, "message": "Insufficient data"}

            # Compare first half vs second half
            mid = len(bets) // 2
            first_half = bets[:mid]
            second_half = bets[mid:]

            first_avg_edge = float(np.mean([b.edge for b in first_half]))
            second_avg_edge = float(np.mean([b.edge for b in second_half]))

            first_win_rate = sum(1 for b in first_half if b.status == "won") / len(first_half)
            second_win_rate = sum(1 for b in second_half if b.status == "won") / len(second_half)

            edge_change = second_avg_edge - first_avg_edge
            wr_change = second_win_rate - first_win_rate

            decay_detected = edge_change < -0.02 or wr_change < -0.04

            return {
                "edge_decay_detected": decay_detected,
                "first_half_edge": round(first_avg_edge, 4),
                "second_half_edge": round(second_avg_edge, 4),
                "edge_change": round(edge_change, 4),
                "first_half_win_rate": round(first_win_rate, 4),
                "second_half_win_rate": round(second_win_rate, 4),
                "win_rate_change": round(wr_change, 4),
                "recommendation": (
                    "RETRAIN required — edge is decaying" if decay_detected
                    else "Edge stable — no action needed"
                ),
            }
        finally:
            session.close()

    @staticmethod
    def _compute_psi(baseline: list[float], recent: list[float], n_bins: int = 10) -> float:
        """Population Stability Index.

        PSI < 0.10: No significant drift
        PSI 0.10-0.25: Moderate drift
        PSI > 0.25: Significant drift
        """
        baseline_arr = np.array(baseline)
        recent_arr = np.array(recent)

        # Create bins from baseline
        bins = np.percentile(baseline_arr, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        baseline_counts = np.histogram(baseline_arr, bins=bins)[0]
        recent_counts = np.histogram(recent_arr, bins=bins)[0]

        # Normalize
        baseline_pct = baseline_counts / max(len(baseline_arr), 1) + 1e-6
        recent_pct = recent_counts / max(len(recent_arr), 1) + 1e-6

        psi = float(np.sum((recent_pct - baseline_pct) * np.log(recent_pct / baseline_pct)))
        return max(psi, 0.0)
