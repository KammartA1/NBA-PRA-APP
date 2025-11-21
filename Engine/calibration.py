# =========================================================
#  TIER C MODULE 11 — MODEL CALIBRATION ENGINE
# =========================================================

import numpy as np
import pandas as pd

def reliability_curve(pred_probs, outcomes, bins=10):
    pred_probs = np.array(pred_probs, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    edges = np.linspace(0,1,bins+1)
    curve = []
    for i in range(bins):
        low, high = edges[i], edges[i+1]
        mask = (pred_probs>=low)&(pred_probs<high)
        if mask.sum()==0:
            curve.append((low,high,np.nan,np.nan))
        else:
            obs = outcomes[mask].mean()
            avg = pred_probs[mask].mean()
            curve.append((low,high,avg,obs))
    return curve

def brier_score(pred_probs, outcomes):
    pred_probs = np.array(pred_probs, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    return float(np.mean((pred_probs-outcomes)**2))

def calibration_bias(pred_probs, outcomes):
    pred_probs = np.array(pred_probs, dtype=float)
    outcomes = np.array(outcomes, dtype=float)
    return float(np.mean(pred_probs-outcomes))

def calibration_engine(pred_probs, outcomes):
    """
    Produce calibration metrics for adjusting ensemble confidence.
    """
    curve = reliability_curve(pred_probs, outcomes)
    bias = calibration_bias(pred_probs, outcomes)
    brier = brier_score(pred_probs, outcomes)

    confidence_adj = float(np.clip(1 - abs(bias), 0.75, 1.25))
    spread_adj = float(np.clip(1 - brier, 0.70, 1.10))

    return {
        "bias": bias,
        "spread_adj": spread_adj,
        "calibration_curve": curve,
        "confidence_adj": confidence_adj
    }
