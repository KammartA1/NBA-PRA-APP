# =========================================================
#  TIER C MODULE 9 — LINEUP & ROLE-SHIFT ENGINE
# =========================================================

import numpy as np

def expected_minutes(base_min, blowout_risk, foul_rate=0.08):
    """
    Estimate expected minutes with volatility.
    """
    foul_adjust = np.random.normal(1 - foul_rate*0.5, 0.03)
    blowout_adjust = 1 - blowout_risk*0.25
    return float(base_min * foul_adjust * blowout_adjust)

def role_shift_multiplier(position, injuries):
    """
    Calculate role shifts depending on who is out.
    """
    shift = 1.0
    for p in injuries:
        pos = p.get("position","")
        if position == "PG" and pos == "SG":
            shift += 0.10
        if position == "SG" and pos == "SF":
            shift += 0.05
        if position == "PF" and pos in ["C","PF"]:
            shift += 0.08
        if position == "C" and pos in ["PF","C"]:
            shift += 0.10
    return float(shift)

def usage_multiplier(position, injuries):
    um = 1.0
    for p in injuries:
        pos = p.get("position","")
        if pos in ["PG","SG"]:
            um += 0.12
    return float(um)

def rebound_multiplier(position, injuries):
    rm = 1.0
    for p in injuries:
        pos = p.get("position","")
        if pos in ["PF","C"]:
            rm += 0.15
    return float(rm)

def assist_multiplier(position, injuries):
    am = 1.0
    for p in injuries:
        pos = p.get("position","")
        if pos == "PG":
            am += 0.10
    return float(am)

def lineup_context(position, injuries, base_minutes=32, blowout_risk=0.1):
    mins = expected_minutes(base_minutes, blowout_risk)
    role_mult = role_shift_multiplier(position, injuries)
    usage_mult_val = usage_multiplier(position, injuries)
    reb_mult_val = rebound_multiplier(position, injuries)
    ast_mult_val = assist_multiplier(position, injuries)
    return {
        "minutes_curve": np.random.normal(mins, 2.5, size=2000),
        "role_shift_mult": role_mult,
        "usage_mult": usage_mult_val,
        "reb_mult": reb_mult_val,
        "ast_mult": ast_mult_val
    }
