"""
edge_analysis/predictive.py
============================
Component 1: PREDICTIVE EDGE — How accurate are probability outputs vs actuals?

Computes:
  - Brier score (model vs market baseline)
  - Log loss (model vs market baseline)
  - Calibration curve (predicted vs actual probability buckets)
  - Predictive edge attribution as % of total ROI
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from edge_analysis.schemas import BetRecord, CalibrationPoint, EdgeComponentResult

log = logging.getLogger(__name__)

# Calibration bucket boundaries
_BUCKET_EDGES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]


def _brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(np.mean((probs - outcomes) ** 2))


def _log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy loss."""
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1.0 - outcomes) * np.log(1.0 - p)))


def _build_calibration_curve(
    probs: np.ndarray, outcomes: np.ndarray,
) -> List[CalibrationPoint]:
    """Bucket predicted probabilities and compute actual win rates."""
    points = []
    for i in range(len(_BUCKET_EDGES) - 1):
        lo, hi = _BUCKET_EDGES[i], _BUCKET_EDGES[i + 1]
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n < 3:
            continue
        pred_avg = float(np.mean(probs[mask]))
        actual_rate = float(np.mean(outcomes[mask]))
        cal_err = abs(pred_avg - actual_rate)
        points.append(CalibrationPoint(
            bucket_lower=lo,
            bucket_upper=hi,
            predicted_avg=round(pred_avg, 4),
            actual_rate=round(actual_rate, 4),
            n_bets=n,
            calibration_error=round(cal_err, 4),
        ))
    return points


def _skill_score(model_score: float, baseline_score: float) -> float:
    """Compute skill score: 1 - (model / baseline).  >0 means model beats baseline."""
    if baseline_score <= 0:
        return 0.0
    return 1.0 - (model_score / baseline_score)


def compute_predictive_edge(
    bets: List[BetRecord],
    total_roi: float,
) -> EdgeComponentResult:
    """Analyze the predictive accuracy component of edge.

    Takes every settled bet, computes Brier & log loss vs market baseline,
    builds calibration curve, and attributes a % of total ROI to predictive skill.

    Args:
        bets: List of settled BetRecord objects with outcomes.
        total_roi: The system's total ROI (for attribution).

    Returns:
        EdgeComponentResult with full predictive edge analysis.
    """
    # Filter to settled bets with valid outcomes
    settled = [b for b in bets if b.won is not None and b.predicted_prob > 0]
    if len(settled) < 20:
        return EdgeComponentResult(
            name="predictive",
            edge_pct_of_roi=0.0,
            absolute_value=0.0,
            p_value=1.0,
            is_significant=False,
            is_positive=False,
            sample_size=len(settled),
            verdict="Insufficient data for predictive edge analysis (need 20+ settled bets)",
        )

    # Build arrays
    model_probs = np.array([b.predicted_prob for b in settled])
    market_probs = np.array([b.market_prob_at_bet for b in settled])
    outcomes = np.array([1.0 if b.won else 0.0 for b in settled])

    # Brier scores
    brier_model = _brier_score(model_probs, outcomes)
    brier_market = _brier_score(market_probs, outcomes)
    brier_skill = _skill_score(brier_model, brier_market)

    # Log loss
    logloss_model = _log_loss(model_probs, outcomes)
    logloss_market = _log_loss(market_probs, outcomes)
    logloss_skill = _skill_score(logloss_model, logloss_market)

    # Calibration curve
    cal_curve = _build_calibration_curve(model_probs, outcomes)
    mean_cal_error = float(np.mean([p.calibration_error for p in cal_curve])) if cal_curve else 0.0

    # Significance testing — paired t-test on squared errors
    model_sq_errors = (model_probs - outcomes) ** 2
    market_sq_errors = (market_probs - outcomes) ** 2
    diff = market_sq_errors - model_sq_errors  # positive = model better

    t_stat = 0.0
    p_value = 1.0
    if np.std(diff) > 0:
        t, p = sp_stats.ttest_1samp(diff, 0.0)
        t_stat = float(t)
        p_value = float(p / 2 if t > 0 else 1.0 - p / 2)  # one-sided

    # Attribution: predictive edge share of ROI
    # Brier skill score represents the fraction of total forecasting improvement
    # attributable to better probability estimation vs the market
    is_positive = brier_skill > 0 and logloss_skill > 0
    is_significant = p_value < 0.05

    # Predictive edge contribution: scale by how much of total ROI comes from
    # having better probabilities. Use average of Brier & logloss skill as proxy.
    avg_skill = (brier_skill + logloss_skill) / 2.0
    # Clamp to reasonable range — predictive edge is typically 30-70% of total edge
    predictive_pct = max(0.0, min(1.0, avg_skill)) * 100.0

    # Overconfidence check
    n_overconfident = sum(1 for p in cal_curve if p.predicted_avg > p.actual_rate)
    n_underconfident = sum(1 for p in cal_curve if p.predicted_avg < p.actual_rate)
    confidence_bias = "overconfident" if n_overconfident > n_underconfident else "underconfident"
    if n_overconfident == n_underconfident:
        confidence_bias = "balanced"

    verdict_parts = []
    if is_positive and is_significant:
        verdict_parts.append(
            f"REAL predictive edge detected. Model Brier {brier_model:.4f} vs "
            f"market {brier_market:.4f} (skill={brier_skill:.1%}). p={p_value:.4f}."
        )
    elif is_positive and not is_significant:
        verdict_parts.append(
            f"Possible predictive edge but NOT statistically significant (p={p_value:.4f}). "
            f"Model Brier {brier_model:.4f} vs market {brier_market:.4f}."
        )
    else:
        verdict_parts.append(
            f"NO predictive edge. Model Brier {brier_model:.4f} vs market {brier_market:.4f} "
            f"(skill={brier_skill:.1%}). Market probabilities are more accurate."
        )
    verdict_parts.append(f"Calibration: {confidence_bias}, mean error {mean_cal_error:.1%}.")

    return EdgeComponentResult(
        name="predictive",
        edge_pct_of_roi=round(predictive_pct, 2),
        absolute_value=round(brier_skill, 4),
        p_value=round(p_value, 4),
        is_significant=is_significant,
        is_positive=is_positive,
        sample_size=len(settled),
        details={
            "brier_model": round(brier_model, 6),
            "brier_market": round(brier_market, 6),
            "brier_skill": round(brier_skill, 4),
            "logloss_model": round(logloss_model, 6),
            "logloss_market": round(logloss_market, 6),
            "logloss_skill": round(logloss_skill, 4),
            "mean_calibration_error": round(mean_cal_error, 4),
            "confidence_bias": confidence_bias,
            "n_overconfident_buckets": n_overconfident,
            "n_underconfident_buckets": n_underconfident,
            "calibration_curve": [
                {
                    "bucket": f"{p.bucket_lower:.0%}-{p.bucket_upper:.0%}",
                    "predicted": p.predicted_avg,
                    "actual": p.actual_rate,
                    "n": p.n_bets,
                    "error": p.calibration_error,
                }
                for p in cal_curve
            ],
        },
        verdict=" ".join(verdict_parts),
    )
