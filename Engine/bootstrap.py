# =========================================================
#  TIER C MODULE 3 — WEIGHTED BOOTSTRAP ENGINE
# =========================================================

import numpy as np
import pandas as pd

def weighted_bootstrap(samples, n_draws=8000, trim_outliers=True):
    """Runs a weighted empirical bootstrap simulation.

    Args:
        samples (array-like): recent performance samples (float values)
        n_draws (int): number of bootstrap draws
        trim_outliers (bool): optional robust trimming

    Returns:
        dict: {
            'mean': float,
            'sd': float,
            'p_over_line': function(line)->prob,
            'samples_adj': np.array
        }
    """

    samples = np.array(samples, dtype=float)
    samples = samples[~np.isnan(samples)]

    if len(samples) == 0:
        return {
            "mean": 0.0,
            "sd": 1.0,
            "p_over_line": lambda l: 0.5,
            "samples_adj": np.array([0.0])
        }

    # Optional outlier trimming (IQR-based)
    if trim_outliers and len(samples) >= 6:
        q1, q3 = np.percentile(samples, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        samples = samples[(samples >= lower) & (samples <= upper)]
        if len(samples) == 0:
            samples = np.array([0.0])

    # Weight more recent games heavier
    weights = np.linspace(0.5, 1.5, len(samples))
    weights = weights / weights.sum()

    # Draw bootstrap samples
    draws = np.random.choice(samples, size=n_draws, replace=True, p=weights)
    mu = float(draws.mean())
    sd = float(draws.std(ddof=1)) if draws.std() > 0 else 1.0

    def p_over(line):
        return float(np.mean(draws > line))

    return {
        "mean": mu,
        "sd": sd,
        "p_over_line": p_over,
        "samples_adj": draws
    }
