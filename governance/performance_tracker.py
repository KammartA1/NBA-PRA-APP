"""
governance/performance_tracker.py
==================================
Track prediction vs actual per model version, compare versions,
and detect performance degradation via rolling window Brier/CLV decline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from database.connection import session_scope
from database.models import Bet, ModelVersion

log = logging.getLogger(__name__)


class PerformanceTracker:
    """Track and compare model performance across versions.

    Monitors rolling Brier score, CLV, ROI, and win rate per model version
    and triggers alerts when degradation is detected.
    """

    def __init__(self, sport: str = "NBA"):
        self.sport = sport

    def track_version_performance(self, version_id: str | None = None) -> Dict[str, Any]:
        """Compute performance metrics for a specific model version.

        If version_id is None, uses the active version.
        """
        try:
            with session_scope() as session:
                query = session.query(Bet).filter(
                    Bet.sport == self.sport,
                    Bet.status == "settled",
                )
                if version_id:
                    query = query.filter(Bet.model_version == version_id)

                bets = query.order_by(Bet.timestamp.asc()).all()
                bet_dicts = [b.to_dict() for b in bets]
        except Exception as e:
            log.warning("Failed to load bets: %s", e)
            bet_dicts = []

        if not bet_dicts:
            return self._empty_performance(version_id)

        # Extract arrays
        pred_probs = np.array([b.get("predicted_prob", 0.5) for b in bet_dicts])
        profits = np.array([b.get("profit", 0) or 0 for b in bet_dicts])
        stakes = np.array([max(b.get("stake", 1.0), 0.01) for b in bet_dicts])
        outcomes = np.array([1.0 if p > 0 else 0.0 for p in profits])

        # CLV
        clv_vals = []
        for b in bet_dicts:
            bl, cl = b.get("bet_line"), b.get("closing_line")
            d = b.get("direction", "over")
            if bl is not None and cl is not None:
                clv_vals.append((cl - bl) if d.lower() == "over" else (bl - cl))
        clv_arr = np.array(clv_vals) if clv_vals else np.array([0.0])

        # Brier score
        brier = float(np.mean((pred_probs - outcomes) ** 2))

        # Log loss
        p = np.clip(pred_probs, 1e-7, 1 - 1e-7)
        log_loss = float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))

        # ROI
        total_profit = float(np.sum(profits))
        total_stake = float(np.sum(stakes))
        roi = (total_profit / max(total_stake, 1)) * 100

        # Win rate
        win_rate = float(np.mean(outcomes))

        # Timestamps for range
        timestamps = [b.get("timestamp", "") for b in bet_dicts]

        return {
            "version_id": version_id or "all",
            "n_bets": len(bet_dicts),
            "brier_score": round(brier, 6),
            "log_loss": round(log_loss, 6),
            "avg_clv": round(float(np.mean(clv_arr)), 3),
            "clv_beat_rate": round(float(np.mean(clv_arr > 0)), 4) if len(clv_arr) > 0 else 0.5,
            "roi_pct": round(roi, 2),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_profit, 2),
            "avg_stake": round(float(np.mean(stakes)), 2),
            "first_bet": timestamps[0] if timestamps else None,
            "last_bet": timestamps[-1] if timestamps else None,
        }

    def compare_version_performance(
        self,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """Compare performance between two model versions."""
        perf_a = self.track_version_performance(version_a)
        perf_b = self.track_version_performance(version_b)

        comparison = {}
        for metric in ["brier_score", "log_loss", "avg_clv", "roi_pct", "win_rate"]:
            va = perf_a.get(metric, 0)
            vb = perf_b.get(metric, 0)
            # For Brier and log_loss, lower is better
            if metric in ("brier_score", "log_loss"):
                improved = vb < va
            else:
                improved = vb > va

            comparison[metric] = {
                "version_a": va,
                "version_b": vb,
                "diff": round(vb - va, 6),
                "improved": improved,
            }

        return {
            "version_a": version_a,
            "version_b": version_b,
            "performance_a": perf_a,
            "performance_b": perf_b,
            "comparison": comparison,
            "overall_better": version_b if sum(
                1 for v in comparison.values() if v["improved"]
            ) > len(comparison) / 2 else version_a,
        }

    def detect_degradation(
        self,
        window: int = 100,
        threshold_pct: float = 10.0,
    ) -> Dict[str, Any]:
        """Detect performance degradation using rolling window analysis.

        Compares recent performance (last `window` bets) against the
        first `window` bets to detect degradation.
        """
        try:
            with session_scope() as session:
                bets = (
                    session.query(Bet)
                    .filter(Bet.sport == self.sport, Bet.status == "settled")
                    .order_by(Bet.timestamp.asc())
                    .all()
                )
                bet_dicts = [b.to_dict() for b in bets]
        except Exception as e:
            log.warning("Failed to load bets: %s", e)
            return {"degradation_detected": False, "reason": "data_load_failure"}

        if len(bet_dicts) < window * 2:
            return {
                "degradation_detected": False,
                "reason": f"insufficient_data (need {window * 2}, have {len(bet_dicts)})",
            }

        early = bet_dicts[:window]
        recent = bet_dicts[-window:]

        # Compute metrics for each window
        early_metrics = self._window_metrics(early)
        recent_metrics = self._window_metrics(recent)

        # Check degradation
        degradation_flags = {}

        # Brier degradation (higher = worse)
        if early_metrics["brier"] > 0:
            brier_change = (recent_metrics["brier"] - early_metrics["brier"]) / early_metrics["brier"] * 100
            degradation_flags["brier"] = {
                "early": early_metrics["brier"],
                "recent": recent_metrics["brier"],
                "change_pct": round(brier_change, 1),
                "degraded": brier_change > threshold_pct,
            }

        # CLV degradation (lower = worse)
        clv_change = recent_metrics["avg_clv"] - early_metrics["avg_clv"]
        degradation_flags["clv"] = {
            "early": early_metrics["avg_clv"],
            "recent": recent_metrics["avg_clv"],
            "change": round(clv_change, 3),
            "degraded": recent_metrics["avg_clv"] < early_metrics["avg_clv"] * (1 - threshold_pct / 100),
        }

        # ROI degradation
        roi_change = recent_metrics["roi"] - early_metrics["roi"]
        degradation_flags["roi"] = {
            "early": early_metrics["roi"],
            "recent": recent_metrics["roi"],
            "change": round(roi_change, 2),
            "degraded": recent_metrics["roi"] < early_metrics["roi"] - threshold_pct,
        }

        any_degraded = any(v.get("degraded", False) for v in degradation_flags.values())

        return {
            "degradation_detected": any_degraded,
            "window_size": window,
            "threshold_pct": threshold_pct,
            "flags": degradation_flags,
            "early_period": f"bets 1-{window}",
            "recent_period": f"bets {len(bet_dicts) - window}-{len(bet_dicts)}",
            "recommendation": "Consider rollback" if any_degraded else "Performance is stable",
        }

    def rolling_performance(self, window: int = 50) -> Dict[str, Any]:
        """Compute rolling performance metrics for charting."""
        try:
            with session_scope() as session:
                bets = (
                    session.query(Bet)
                    .filter(Bet.sport == self.sport, Bet.status == "settled")
                    .order_by(Bet.timestamp.asc())
                    .all()
                )
                bet_dicts = [b.to_dict() for b in bets]
        except Exception:
            return {"timestamps": [], "brier": [], "clv": [], "roi": []}

        if len(bet_dicts) < window:
            return {"timestamps": [], "brier": [], "clv": [], "roi": []}

        timestamps = []
        brier_series = []
        clv_series = []
        roi_series = []

        for i in range(window - 1, len(bet_dicts)):
            chunk = bet_dicts[i - window + 1:i + 1]
            metrics = self._window_metrics(chunk)
            timestamps.append(chunk[-1].get("timestamp", ""))
            brier_series.append(metrics["brier"])
            clv_series.append(metrics["avg_clv"])
            roi_series.append(metrics["roi"])

        return {
            "timestamps": timestamps,
            "brier": brier_series,
            "clv": clv_series,
            "roi": roi_series,
        }

    def _window_metrics(self, bets: List[Dict]) -> Dict[str, float]:
        """Compute metrics for a window of bets."""
        if not bets:
            return {"brier": 0.25, "avg_clv": 0.0, "roi": 0.0, "win_rate": 0.5}

        pred_probs = np.array([b.get("predicted_prob", 0.5) for b in bets])
        profits = np.array([b.get("profit", 0) or 0 for b in bets])
        stakes = np.array([max(b.get("stake", 1.0), 0.01) for b in bets])
        outcomes = np.array([1.0 if p > 0 else 0.0 for p in profits])

        brier = float(np.mean((pred_probs - outcomes) ** 2))

        clv_vals = []
        for b in bets:
            bl, cl = b.get("bet_line"), b.get("closing_line")
            d = b.get("direction", "over")
            if bl is not None and cl is not None:
                clv_vals.append((cl - bl) if d.lower() == "over" else (bl - cl))
        avg_clv = float(np.mean(clv_vals)) if clv_vals else 0.0

        total_stake = float(np.sum(stakes))
        roi = float(np.sum(profits)) / max(total_stake, 1) * 100

        return {
            "brier": round(brier, 6),
            "avg_clv": round(avg_clv, 3),
            "roi": round(roi, 2),
            "win_rate": round(float(np.mean(outcomes)), 4),
        }

    def _empty_performance(self, version_id: str | None) -> Dict[str, Any]:
        return {
            "version_id": version_id or "none",
            "n_bets": 0,
            "brier_score": 0.25,
            "log_loss": 0.693,
            "avg_clv": 0.0,
            "clv_beat_rate": 0.5,
            "roi_pct": 0.0,
            "win_rate": 0.5,
            "total_pnl": 0.0,
            "avg_stake": 0.0,
            "first_bet": None,
            "last_bet": None,
        }
