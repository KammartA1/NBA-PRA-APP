"""Calibration Monitor — Tracks predicted vs actual win rates.

A well-calibrated model means: when it says 60% probability, the bet wins ~60%
of the time. Miscalibration means the model's probability outputs cannot be
trusted, which invalidates edge calculations and Kelly sizing.

We bin bets into probability buckets (50-55%, 55-60%, etc.) and compare
predicted probability to actual hit rate."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np

from ..db.schema import BetLog, CalibrationLog, get_session
from .types import CalibrationBucket, Sport

logger = logging.getLogger(__name__)


class CalibrationMonitor:
    """Monitors model calibration over time."""

    # Probability buckets for calibration curve
    BUCKETS = [
        (0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
        (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
        (0.90, 1.00),
    ]

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def compute_calibration(self, window: int = 500) -> dict:
        """Compute calibration curve over last N settled bets.

        Returns:
            {
                "buckets": [CalibrationBucket, ...],
                "mean_absolute_error": float,
                "max_error": float,
                "overconfidence_ratio": float,  # % of buckets where pred > actual
                "n_total": int,
                "brier_score": float,           # Brier score (lower = better)
            }
        """
        session = self._session()
        try:
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.desc())
                .limit(window)
                .all()
            )

            if len(rows) < 20:
                return {
                    "buckets": [],
                    "mean_absolute_error": 0.0,
                    "max_error": 0.0,
                    "overconfidence_ratio": 0.0,
                    "n_total": len(rows),
                    "brier_score": 0.0,
                }

            # Build calibration buckets
            buckets: list[CalibrationBucket] = []
            errors = []
            overconf_count = 0

            for lower, upper in self.BUCKETS:
                bucket_bets = [
                    r for r in rows
                    if lower <= r.model_prob < upper
                ]
                if len(bucket_bets) < 3:
                    continue

                predicted_avg = float(np.mean([r.model_prob for r in bucket_bets]))
                actual_wins = sum(1 for r in bucket_bets if r.status == "won")
                actual_rate = actual_wins / len(bucket_bets)
                cal_error = abs(predicted_avg - actual_rate)
                is_overconf = predicted_avg > actual_rate

                bucket = CalibrationBucket(
                    prob_lower=lower,
                    prob_upper=upper,
                    predicted_avg=round(predicted_avg, 4),
                    actual_rate=round(actual_rate, 4),
                    n_bets=len(bucket_bets),
                    calibration_error=round(cal_error, 4),
                    is_overconfident=is_overconf,
                )
                buckets.append(bucket)
                errors.append(cal_error)
                if is_overconf:
                    overconf_count += 1

            # Brier score
            brier = float(np.mean([
                (r.model_prob - (1.0 if r.status == "won" else 0.0)) ** 2
                for r in rows
            ]))

            mae = float(np.mean(errors)) if errors else 0.0
            max_err = float(max(errors)) if errors else 0.0
            overconf_ratio = overconf_count / len(buckets) if buckets else 0.0

            return {
                "buckets": buckets,
                "mean_absolute_error": round(mae, 4),
                "max_error": round(max_err, 4),
                "overconfidence_ratio": round(overconf_ratio, 3),
                "n_total": len(rows),
                "brier_score": round(brier, 4),
            }
        finally:
            session.close()

    def save_calibration_snapshot(self) -> None:
        """Save current calibration to the log table for historical tracking."""
        cal = self.compute_calibration()
        if not cal["buckets"]:
            return

        session = self._session()
        try:
            now = datetime.utcnow()
            for bucket in cal["buckets"]:
                row = CalibrationLog(
                    sport=self.sport.value,
                    report_date=now,
                    bucket_label=f"{int(bucket.prob_lower*100)}-{int(bucket.prob_upper*100)}%",
                    prob_lower=bucket.prob_lower,
                    prob_upper=bucket.prob_upper,
                    predicted_avg=bucket.predicted_avg,
                    actual_rate=bucket.actual_rate,
                    n_bets=bucket.n_bets,
                    calibration_error=bucket.calibration_error,
                    is_overconfident=bucket.is_overconfident,
                )
                session.add(row)
            session.commit()
            logger.info("Calibration snapshot saved: %d buckets, MAE=%.3f",
                        len(cal["buckets"]), cal["mean_absolute_error"])
        except Exception:
            session.rollback()
            logger.exception("Failed to save calibration snapshot")
        finally:
            session.close()

    def calibration_drift(self) -> dict:
        """Compare current calibration to historical average.

        Returns:
            {
                "current_mae": float,
                "historical_mae": float,
                "drift": float,              # current - historical (positive = worse)
                "is_drifting": bool,          # drift > 0.03
                "direction": str,             # "worsening", "stable", "improving"
            }
        """
        session = self._session()
        try:
            # Current
            current = self.compute_calibration(window=100)
            current_mae = current["mean_absolute_error"]

            # Historical (older bets: offset by 100, take 400)
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost"]))
                .order_by(BetLog.timestamp.desc())
                .offset(100)
                .limit(400)
                .all()
            )

            if len(rows) < 50:
                return {
                    "current_mae": current_mae,
                    "historical_mae": 0.0,
                    "drift": 0.0,
                    "is_drifting": False,
                    "direction": "insufficient_data",
                }

            # Compute historical calibration manually
            hist_errors = []
            for lower, upper in self.BUCKETS:
                bucket_bets = [r for r in rows if lower <= r.model_prob < upper]
                if len(bucket_bets) < 3:
                    continue
                pred_avg = float(np.mean([r.model_prob for r in bucket_bets]))
                actual_rate = sum(1 for r in bucket_bets if r.status == "won") / len(bucket_bets)
                hist_errors.append(abs(pred_avg - actual_rate))

            hist_mae = float(np.mean(hist_errors)) if hist_errors else 0.0
            drift = current_mae - hist_mae

            if drift > 0.03:
                direction = "worsening"
            elif drift < -0.02:
                direction = "improving"
            else:
                direction = "stable"

            return {
                "current_mae": round(current_mae, 4),
                "historical_mae": round(hist_mae, 4),
                "drift": round(drift, 4),
                "is_drifting": drift > 0.03,
                "direction": direction,
            }
        finally:
            session.close()
