# =========================================================
#  TIER C MODULE 6 — COVARIANCE-BASED JOINT MONTE CARLO
# =========================================================

import numpy as np

def estimate_correlation(leg1, leg2):
    """
    Estimate correlation between two prop legs using contextual factors.
    """
    corr = 0.0
    if leg1.get("team") and leg1.get("team") == leg2.get("team"):
        corr += 0.15
    if leg1.get("market") == "Points" and leg2.get("market") == "Points":
        corr += 0.10
    if (leg1.get("market") == "Assists" and leg2.get("market") == "Points") or        (leg2.get("market") == "Assists" and leg1.get("market") == "Points"):
        corr -= 0.10
    if leg1.get("ctx_mult",1) > 1.05 and leg2.get("ctx_mult",1) > 1.05:
        corr += 0.05

    return float(np.clip(corr, -0.25, 0.40))


def joint_mc(mu1, sd1, line1, mu2, sd2, line2, corr, n_sims=15000):
    """
    Multivariate normal Monte Carlo simulation for joint hit probability.
    """
    corr = float(np.clip(corr, -0.99, 0.99))
    cov = [
        [sd1**2, corr * sd1 * sd2],
        [corr * sd1 * sd2, sd2**2]
    ]
    mean = [mu1, mu2]

    draws = np.random.multivariate_normal(mean, cov, size=n_sims)
    hits = (draws[:, 0] > line1) & (draws[:, 1] > line2)

    return {
        "joint_prob": float(np.mean(hits)),
        "correlation": float(corr),
        "cov_matrix": cov,
        "sd1": float(sd1),
        "sd2": float(sd2)
    }
