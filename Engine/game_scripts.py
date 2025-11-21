# =========================================================
#  TIER C MODULE 5 — GAME SCRIPT SIMULATION ENGINE
# =========================================================

import numpy as np

def simulate_game_script(samples, line, pace_mult=1.0, minutes_mult=1.0,
                         blowout_risk=0.0, n_sims=5000):
    """Simulates multiple game-script scenarios:

    Args:
        samples (array-like): empirical production samples
        line (float): target line
        pace_mult (float): pace adjustment factor
        minutes_mult (float): minute-based usage multiplier
        blowout_risk (float): 0.0 - 1.0
        n_sims (int): number of game scripts

    Returns:
        dict: {
            'p_over': float,
            'mu_adj': float,
            'sd_adj': float,
            'scenario_probs': {...}
        }
    """

    samples = np.array(samples, dtype=float)
    if len(samples) == 0:
        samples = np.array([0.0])

    # Scenario weights
    w_competitive = (1.0 - blowout_risk) * 0.70
    w_mild_blowout = (blowout_risk * 0.20)
    w_severe_blowout = (blowout_risk * 0.10)

    # Generate scenario draws
    draws = []
    scenario_probs = {
        "competitive": w_competitive,
        "mild_blowout": w_mild_blowout,
        "severe_blowout": w_severe_blowout
    }

    for _ in range(n_sims):
        r = np.random.rand()

        if r <= w_competitive:
            mult = pace_mult * minutes_mult
        elif r <= w_competitive + w_mild_blowout:
            mult = pace_mult * max(minutes_mult * 0.90, 0.70)
        else:
            mult = pace_mult * max(minutes_mult * 0.75, 0.50)

        val = np.random.choice(samples) * mult
        draws.append(val)

    draws = np.array(draws)
    mu_adj = float(draws.mean())
    sd_adj = float(draws.std(ddof=1)) if draws.std() > 0 else 1.0
    p_over = float(np.mean(draws > line))

    return {
        "p_over": p_over,
        "mu_adj": mu_adj,
        "sd_adj": sd_adj,
        "scenario_probs": scenario_probs
    }
