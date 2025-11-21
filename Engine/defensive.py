# =========================================================
#  TIER C MODULE 8 — DEFENSIVE MATCHUP ENGINE
# =========================================================

import numpy as np

# Simplified defensive database (Tier C foundation)
DEF_DB = {
    "BOS": {"drtg": 109, "pace": 97, "reb_rate": 0.52, "ast_rate": 0.57,
            "paint_def": 0.88, "three_def": 0.92},
    "MIL": {"drtg": 112, "pace": 99, "reb_rate": 0.50, "ast_rate": 0.55,
            "paint_def": 0.95, "three_def": 0.90},
    "LAL": {"drtg": 114, "pace": 101,"reb_rate": 0.51, "ast_rate": 0.58,
            "paint_def": 0.93, "three_def": 0.95},
    # teams default below
}

DEFAULT_DEF = {"drtg": 113, "pace": 98, "reb_rate": 0.50, "ast_rate": 0.55,
               "paint_def": 0.95, "three_def": 0.95}

def get_team_def(team):
    return DEF_DB.get(team, DEFAULT_DEF)

def position_matchup_multiplier(position, team_profile):
    """
    Position-based defensive multiplier.
    """
    if position == "PG":
        return float(team_profile["ast_rate"] * 1.05)
    if position == "SG":
        return float(team_profile["three_def"] * 1.02)
    if position == "SF":
        return float(team_profile["drtg"] / 113)
    if position == "PF":
        return float(team_profile["paint_def"] * 1.03)
    if position == "C":
        return float(team_profile["reb_rate"] * 1.05)
    return 1.0

def defensive_context(player_position, opp_team):
    """
    Returns defensive multiplier and contextual components.
    """
    prof = get_team_def(opp_team)
    pos_mult = position_matchup_multiplier(player_position, prof)
    def_mult = float(np.clip(pos_mult, 0.85, 1.20))
    return {
        "def_mult": def_mult,
        "team_profile": prof,
        "pos_context": player_position
    }
