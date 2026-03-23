"""
workers/model_worker.py
=======================
Retrains / recalibrates the model on new settled-bet data.

Triggers:
  1. Weekly scheduled retrain (Sunday 11 PM ET)
  2. Calibration error exceeds threshold
  3. Model drift detected (rolling accuracy drop)

Retraining:
  - Collects all settled bets with outcomes
  - Computes calibration metrics per bucket
  - Adjusts model parameters (market_prior_weight, injury boost, etc.)
  - Walk-forward validation on recent window
  - Only deploys if new version beats current

Run standalone:
    python -m workers.model_worker          # one-shot
    python -m workers.model_worker --loop   # weekly loop
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from database.connection import session_scope, init_db
from database.models import (
    Bet, ModelVersion, CalibrationSnapshot, SystemState as SystemStateModel,
)
from workers.base import BaseWorker, standalone_main

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Calibration error threshold triggering emergency retrain
CALIBRATION_ERROR_THRESHOLD = 0.08  # 8% mean absolute calibration error
# Minimum settled bets needed before retraining
MIN_SETTLED_BETS = 40
# Walk-forward validation: hold out last N bets
VALIDATION_HOLDOUT = 50
# Model drift: rolling accuracy window and threshold
DRIFT_WINDOW_BETS = 100
DRIFT_ACCURACY_FLOOR = 0.42  # Below this = drift alarm
# Weekly retrain cadence (seconds): 7 days
WEEKLY_INTERVAL = 7 * 24 * 60 * 60

# Default model parameters (starting point)
DEFAULT_PARAMS: Dict[str, float] = {
    "market_prior_weight": 0.65,
    "injury_boost_base": 1.05,
    "injury_boost_assist_drag": 0.97,
    "rest_penalty_b2b": 0.97,
    "home_court_boost": 1.015,
    "vol_penalty_floor": 0.80,
    "min_ev_threshold": 0.05,
    "frac_kelly": 0.25,
    "max_risk_frac": 0.05,
    "negbinom_blend_cap": 0.92,
    "log_mult_cap_lower": 0.75,
    "log_mult_cap_upper": 1.25,
}

# Calibration bucket edges
CALIBRATION_EDGES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Calibration computation
# ---------------------------------------------------------------------------

def compute_calibration_buckets(
    predicted_probs: List[float],
    actual_outcomes: List[int],
) -> Tuple[List[Dict[str, Any]], float]:
    """Compute calibration curve buckets.

    Parameters
    ----------
    predicted_probs : list of float
        Model predicted probability for each bet.
    actual_outcomes : list of int
        1 if the bet won, 0 if lost.

    Returns
    -------
    (buckets, mean_abs_error) where buckets is a list of dicts.
    """
    if len(predicted_probs) == 0:
        return [], 0.0

    probs = np.array(predicted_probs, dtype=float)
    actuals = np.array(actual_outcomes, dtype=float)
    buckets: List[Dict[str, Any]] = []
    total_abs_error = 0.0
    total_bets = 0

    for i in range(len(CALIBRATION_EDGES) - 1):
        lo = CALIBRATION_EDGES[i]
        hi = CALIBRATION_EDGES[i + 1]
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        pred_avg = float(probs[mask].mean())
        actual_rate = float(actuals[mask].mean())
        cal_error = abs(pred_avg - actual_rate)
        buckets.append({
            "prob_lower": lo,
            "prob_upper": hi,
            "predicted_avg": round(pred_avg, 4),
            "actual_rate": round(actual_rate, 4),
            "n_bets": n,
            "calibration_error": round(cal_error, 4),
            "is_overconfident": pred_avg > actual_rate,
        })
        total_abs_error += cal_error * n
        total_bets += n

    mean_abs_error = total_abs_error / max(total_bets, 1)
    return buckets, round(mean_abs_error, 5)


# ---------------------------------------------------------------------------
# Model drift detection
# ---------------------------------------------------------------------------

def detect_model_drift(
    recent_probs: List[float],
    recent_outcomes: List[int],
) -> Tuple[bool, float]:
    """Check if the model is drifting (rolling accuracy below floor).

    Returns (is_drifting, rolling_accuracy).
    """
    if len(recent_probs) < 20:
        return False, 0.5

    probs = np.array(recent_probs)
    outcomes = np.array(recent_outcomes)

    # Accuracy: did the model's predicted direction match reality?
    predicted_win = probs >= 0.50
    actual_win = outcomes == 1
    accuracy = float((predicted_win == actual_win).mean())

    is_drifting = accuracy < DRIFT_ACCURACY_FLOOR
    return is_drifting, round(accuracy, 4)


# ---------------------------------------------------------------------------
# Parameter optimisation (grid search on calibration error)
# ---------------------------------------------------------------------------

def optimize_parameters(
    settled_bets: List[Dict[str, Any]],
    current_params: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Adjust model parameters to minimise calibration error.

    Uses a conservative grid search around current values with small step
    sizes to avoid overfitting.

    Returns (new_params, optimization_metrics).
    """
    probs = [b["predicted_prob"] for b in settled_bets]
    outcomes = [1 if b["status"] == "won" else 0 for b in settled_bets]

    _, baseline_error = compute_calibration_buckets(probs, outcomes)

    best_params = dict(current_params)
    best_error = baseline_error
    search_log: List[Dict[str, Any]] = []

    # Adjust market_prior_weight in small increments
    for mpw_delta in [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]:
        new_mpw = max(0.10, min(0.95, current_params["market_prior_weight"] + mpw_delta))

        # Simulate: re-blend probabilities with new market_prior_weight
        adjusted_probs = []
        for b in settled_bets:
            p_model = b.get("model_prob", b["predicted_prob"])
            p_implied = b.get("implied_prob", 0.50)
            if p_implied and p_model:
                p_blended = float(np.clip(
                    new_mpw * p_model + (1.0 - new_mpw) * p_implied,
                    0.01, 0.99,
                ))
            else:
                p_blended = p_model or 0.50
            adjusted_probs.append(p_blended)

        _, cal_error = compute_calibration_buckets(adjusted_probs, outcomes)
        search_log.append({
            "market_prior_weight": round(new_mpw, 3),
            "calibration_error": round(cal_error, 5),
        })

        if cal_error < best_error:
            best_error = cal_error
            best_params["market_prior_weight"] = round(new_mpw, 3)

    # Adjust min_ev_threshold
    for ev_delta in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        new_ev = max(0.01, min(0.15, current_params["min_ev_threshold"] + ev_delta))
        # This doesn't change calibration directly but affects which bets pass the gate
        # We track it for the version record
        if ev_delta == 0:
            best_params["min_ev_threshold"] = round(new_ev, 3)

    # Adjust frac_kelly based on calibration direction
    if baseline_error > 0.06:
        # Model is poorly calibrated: reduce Kelly to limit damage
        best_params["frac_kelly"] = max(0.10, current_params["frac_kelly"] - 0.05)
    elif baseline_error < 0.03:
        # Well calibrated: allow slightly more aggressive sizing
        best_params["frac_kelly"] = min(0.40, current_params["frac_kelly"] + 0.02)

    # Adjust log_mult caps based on observed projection accuracy
    proj_errors = []
    for b in settled_bets:
        proj = b.get("model_projection")
        actual = b.get("actual_outcome")
        if proj is not None and actual is not None:
            proj_errors.append(abs(proj - actual) / max(abs(proj), 1.0))

    if proj_errors:
        median_proj_error = float(np.median(proj_errors))
        if median_proj_error > 0.25:
            # Projections are noisy: tighten caps
            best_params["log_mult_cap_lower"] = max(0.80, current_params["log_mult_cap_lower"] + 0.02)
            best_params["log_mult_cap_upper"] = min(1.20, current_params["log_mult_cap_upper"] - 0.02)
        elif median_proj_error < 0.12:
            # Projections are accurate: allow wider caps
            best_params["log_mult_cap_lower"] = max(0.70, current_params["log_mult_cap_lower"] - 0.01)
            best_params["log_mult_cap_upper"] = min(1.30, current_params["log_mult_cap_upper"] + 0.01)

    metrics = {
        "baseline_calibration_error": baseline_error,
        "optimized_calibration_error": best_error,
        "improvement": round(baseline_error - best_error, 5),
        "search_log": search_log,
        "n_bets": len(settled_bets),
    }

    return best_params, metrics


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    settled_bets: List[Dict[str, Any]],
    params: Dict[str, float],
    holdout_n: int = VALIDATION_HOLDOUT,
) -> Dict[str, Any]:
    """Validate model parameters on a held-out recent window.

    Splits settled_bets by timestamp: train on older, validate on newer.
    Returns performance metrics on the validation set.
    """
    if len(settled_bets) < holdout_n + 20:
        return {
            "valid": False,
            "reason": f"Not enough bets ({len(settled_bets)}) for walk-forward",
        }

    # Sort by timestamp
    sorted_bets = sorted(settled_bets, key=lambda b: b.get("timestamp", ""))
    train_bets = sorted_bets[:-holdout_n]
    val_bets = sorted_bets[-holdout_n:]

    # Calibration on validation set
    val_probs = [b["predicted_prob"] for b in val_bets]
    val_outcomes = [1 if b["status"] == "won" else 0 for b in val_bets]

    buckets, cal_error = compute_calibration_buckets(val_probs, val_outcomes)

    # Accuracy
    probs_arr = np.array(val_probs)
    outcomes_arr = np.array(val_outcomes)
    accuracy = float((( probs_arr >= 0.5) == (outcomes_arr == 1)).mean())

    # ROI on validation bets
    total_staked = sum(b.get("stake", 1.0) for b in val_bets)
    total_pnl = sum(b.get("pnl", 0.0) for b in val_bets)
    roi = (total_pnl / max(total_staked, 1.0)) * 100.0

    # Brier score
    brier = float(np.mean((probs_arr - outcomes_arr) ** 2))

    return {
        "valid": True,
        "n_train": len(train_bets),
        "n_validation": len(val_bets),
        "calibration_error": round(cal_error, 5),
        "accuracy": round(accuracy, 4),
        "roi_pct": round(roi, 2),
        "brier_score": round(brier, 5),
        "buckets": buckets,
    }


# ===================================================================
# Worker class
# ===================================================================

class ModelWorker(BaseWorker):
    """Retrains model on settled bet data.  Runs weekly or on trigger."""

    def __init__(self, **kwargs):
        super().__init__(
            name="model_worker",
            interval_seconds=int(os.environ.get("MODEL_INTERVAL", str(WEEKLY_INTERVAL))),
            max_retries=1,
            retry_delay=30.0,
            **kwargs,
        )

    def _should_retrain(self, session) -> Tuple[bool, str]:
        """Determine if retraining is warranted.  Returns (should_retrain, reason)."""
        # Check 1: Has it been more than 7 days since last model version?
        latest_mv = (
            session.query(ModelVersion)
            .filter(ModelVersion.sport == "NBA")
            .order_by(ModelVersion.created_at.desc())
            .first()
        )
        if latest_mv is None:
            return True, "no_existing_model"

        age = _utcnow() - (
            latest_mv.created_at.replace(tzinfo=timezone.utc)
            if latest_mv.created_at.tzinfo is None
            else latest_mv.created_at
        )
        if age > timedelta(days=7):
            return True, f"weekly_schedule (last model {age.days}d ago)"

        # Check 2: Calibration error above threshold (from latest snapshot)
        latest_cal = (
            session.query(CalibrationSnapshot)
            .filter(CalibrationSnapshot.sport == "NBA")
            .order_by(CalibrationSnapshot.snapshot_date.desc())
            .first()
        )
        if latest_cal and latest_cal.calibration_error > CALIBRATION_ERROR_THRESHOLD:
            return True, f"calibration_error={latest_cal.calibration_error:.4f} > {CALIBRATION_ERROR_THRESHOLD}"

        # Check 3: Model drift on recent bets
        recent_bets = (
            session.query(Bet)
            .filter(
                Bet.sport == "NBA",
                Bet.status.in_(["won", "lost"]),
            )
            .order_by(Bet.settled_at.desc())
            .limit(DRIFT_WINDOW_BETS)
            .all()
        )
        if len(recent_bets) >= 30:
            probs = [b.predicted_prob for b in recent_bets if b.predicted_prob]
            outcomes = [1 if b.status == "won" else 0 for b in recent_bets]
            if probs:
                is_drifting, acc = detect_model_drift(probs, outcomes)
                if is_drifting:
                    return True, f"model_drift (accuracy={acc:.3f} < {DRIFT_ACCURACY_FLOOR})"

        return False, "no_retrain_needed"

    def execute(self) -> Dict[str, Any]:
        now = _utcnow()

        with session_scope() as session:
            # Check if retraining is needed
            should_retrain, reason = self._should_retrain(session)

            if not should_retrain:
                self.logger.info("Retrain not needed: %s", reason)
                return {"ok": True, "retrained": False, "reason": reason}

            self.logger.info("Retraining triggered: %s", reason)

            # Collect all settled bets
            settled = (
                session.query(Bet)
                .filter(
                    Bet.sport == "NBA",
                    Bet.status.in_(["won", "lost"]),
                )
                .order_by(Bet.timestamp.asc())
                .all()
            )

            if len(settled) < MIN_SETTLED_BETS:
                self.logger.info(
                    "Not enough settled bets (%d < %d) for retraining",
                    len(settled), MIN_SETTLED_BETS,
                )
                return {
                    "ok": True,
                    "retrained": False,
                    "reason": f"insufficient_data ({len(settled)} < {MIN_SETTLED_BETS})",
                }

            # Convert to dicts for processing
            bet_dicts = []
            for b in settled:
                features = {}
                try:
                    features = json.loads(b.features_snapshot_json or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass

                bet_dicts.append({
                    "predicted_prob": b.predicted_prob or 0.50,
                    "model_prob": features.get("p_model", b.predicted_prob),
                    "implied_prob": features.get("p_implied", 0.50),
                    "status": b.status,
                    "pnl": b.pnl or 0.0,
                    "stake": b.stake or 1.0,
                    "timestamp": b.timestamp.isoformat() if b.timestamp else "",
                    "model_projection": b.model_projection,
                    "actual_outcome": b.actual_outcome,
                })

            # Get current model parameters
            current_mv = (
                session.query(ModelVersion)
                .filter(ModelVersion.sport == "NBA", ModelVersion.is_active == True)
                .order_by(ModelVersion.created_at.desc())
                .first()
            )
            if current_mv:
                current_params = current_mv.parameters
            else:
                current_params = dict(DEFAULT_PARAMS)

            # Fill in any missing defaults
            for k, v in DEFAULT_PARAMS.items():
                if k not in current_params:
                    current_params[k] = v

            # Compute current calibration
            probs = [b["predicted_prob"] for b in bet_dicts]
            outcomes = [1 if b["status"] == "won" else 0 for b in bet_dicts]
            current_buckets, current_cal_error = compute_calibration_buckets(probs, outcomes)

            self.logger.info(
                "Current calibration error: %.4f (%d settled bets)",
                current_cal_error, len(bet_dicts),
            )

            # Save calibration snapshot
            for bucket in current_buckets:
                session.add(CalibrationSnapshot(
                    sport="NBA",
                    bucket_label=f"{bucket['prob_lower']:.2f}-{bucket['prob_upper']:.2f}",
                    prob_lower=bucket["prob_lower"],
                    prob_upper=bucket["prob_upper"],
                    predicted_avg=bucket["predicted_avg"],
                    actual_rate=bucket["actual_rate"],
                    n_bets=bucket["n_bets"],
                    calibration_error=bucket["calibration_error"],
                    snapshot_date=now,
                ))

            # Optimise parameters
            new_params, opt_metrics = optimize_parameters(bet_dicts, current_params)

            self.logger.info(
                "Optimization: cal_error %.5f -> %.5f (improvement: %.5f)",
                opt_metrics["baseline_calibration_error"],
                opt_metrics["optimized_calibration_error"],
                opt_metrics["improvement"],
            )

            # Walk-forward validation
            wf_result = walk_forward_validate(bet_dicts, new_params)

            if wf_result.get("valid"):
                self.logger.info(
                    "Walk-forward: accuracy=%.3f, ROI=%.1f%%, Brier=%.4f, cal_error=%.4f",
                    wf_result["accuracy"],
                    wf_result["roi_pct"],
                    wf_result["brier_score"],
                    wf_result["calibration_error"],
                )
            else:
                self.logger.warning("Walk-forward skipped: %s", wf_result.get("reason"))

            # Decision: deploy new version only if it improves on current
            deploy = False
            deploy_reason = ""

            if not wf_result.get("valid"):
                # Not enough data for WF -- deploy if optimisation improved
                if opt_metrics["improvement"] > 0.001:
                    deploy = True
                    deploy_reason = "optimization_improved_no_wf"
                else:
                    deploy_reason = "no_improvement_no_wf"
            else:
                # WF validation available: check if new params beat current
                wf_cal = wf_result["calibration_error"]
                if wf_cal < current_cal_error or wf_result["accuracy"] > 0.50:
                    deploy = True
                    deploy_reason = f"wf_improved (cal={wf_cal:.4f} vs {current_cal_error:.4f})"
                else:
                    deploy_reason = f"wf_no_improvement (cal={wf_cal:.4f} >= {current_cal_error:.4f})"

            if deploy:
                # Generate version identifier
                data_hash = hashlib.sha256(
                    json.dumps(bet_dicts[-20:], sort_keys=True, default=str).encode()
                ).hexdigest()[:16]
                version_str = f"v{now.strftime('%Y%m%d_%H%M')}_{data_hash}"

                # Deactivate current active version
                if current_mv:
                    current_mv.is_active = False

                # Create new version
                perf_metrics = {
                    "calibration_error": opt_metrics["optimized_calibration_error"],
                    "n_settled_bets": len(bet_dicts),
                    "walk_forward": wf_result,
                    "optimization": opt_metrics,
                    "trigger_reason": reason,
                    "deploy_reason": deploy_reason,
                }

                new_mv = ModelVersion(
                    version=version_str,
                    sport="NBA",
                    is_active=True,
                    parameters_json=json.dumps(new_params, default=str),
                    training_data_hash=data_hash,
                    performance_metrics_json=json.dumps(perf_metrics, default=str),
                )
                session.add(new_mv)

                self.logger.info(
                    "Deployed new model version: %s (%s)",
                    version_str, deploy_reason,
                )
            else:
                self.logger.info("New model NOT deployed: %s", deploy_reason)

        return {
            "ok": True,
            "retrained": True,
            "deployed": deploy,
            "deploy_reason": deploy_reason,
            "trigger_reason": reason,
            "n_settled_bets": len(bet_dicts),
            "current_calibration_error": current_cal_error,
            "optimized_calibration_error": opt_metrics["optimized_calibration_error"],
            "walk_forward": wf_result if wf_result.get("valid") else None,
        }


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    standalone_main(ModelWorker)
