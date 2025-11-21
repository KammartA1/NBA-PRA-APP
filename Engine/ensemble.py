# =========================================================
#  TIER C MODULE 4 — ENSEMBLE PROJECTION ENGINE
# =========================================================

import numpy as np
import pandas as pd
from scipy.stats import norm
from .bootstrap import weighted_bootstrap

def normal_model(samples, line):
    samples = np.array(samples, dtype=float)
    if len(samples) == 0:
        return 0.5, 0.0, 1.0
    mu = samples.mean()
    sd = samples.std() if samples.std() > 0 else 1.0
    p = 1 - norm.cdf(line, mu, sd)
    return float(p), float(mu), float(sd)

def regression_model(samples, line):
    # simple ridge-style linear projection
    if len(samples) < 3:
        return 0.5
    x = np.arange(len(samples))
    y = np.array(samples)
    X = np.vstack([x, np.ones(len(x))]).T
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = coef[0] * (len(samples)) + coef[1]
    return float(1 - norm.cdf(line, pred, max(y.std(),1)))

def defensive_adjust(prob, ctx_mult):
    return float(np.clip(prob * ctx_mult, 0.01, 0.99))

def ensemble_projection(samples, line, ctx_mult=1.0, game_script_prob=0.5):
    bs = weighted_bootstrap(samples)
    p_bs = bs["p_over_line"](line)
    p_norm, mu_norm, sd_norm = normal_model(samples, line)
    p_reg = regression_model(samples, line)
    p_def = defensive_adjust(p_norm, ctx_mult)

    # blend 5 components
    p_final = (
        0.25*p_bs +
        0.20*p_norm +
        0.20*p_reg +
        0.20*p_def +
        0.15*game_script_prob
    )
    return float(np.clip(p_final, 0.01, 0.99))
