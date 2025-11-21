# =========================================================
#  TIER C MODULE 7 — INJURY & ROTATION ENGINE
# =========================================================

import requests
import numpy as np

NBA_API_BASE = "https://cdn.nba.com/static/json/liveData"

def fetch_injury_report():
    """
    Fetch basic NBA injury report data from public endpoints.
    Returns a dict keyed by team with status lists.
    """
    try:
        url = f"{NBA_API_BASE}/inGameBook/inGameBook.json"
        r = requests.get(url, timeout=10)
        data = r.json()
        report = {}
        for g in data.get("teams", []):
            team = g.get("triCode")
            players = g.get("players", [])
            injured = [p for p in players if p.get("status") != "ACTIVE"]
            report[team] = injured
        return report
    except Exception:
        return {}

def compute_usage_multiplier(player, team, injury_report):
    """
    Dynamic usage redistribution when key teammates are out.
    """
    if team not in injury_report:
        return 1.0
    injured = injury_report[team]
    key_out = 0
    for p in injured:
        pos = p.get("position", "")
        # weight missing ballhandlers / scorers higher
        if pos in ["PG","SG"]:
            key_out += 1
        if pos in ["PF","C"]:
            key_out += 0.5
    return float(1.0 + min(0.15 * key_out, 0.40))

def compute_minutes_multiplier(blowout_risk):
    """
    Minutes volatility based on blowout probability.
    """
    return float(max(0.7, 1.0 - blowout_risk*0.25))

def compute_blowout_risk(spread):
    """
    Crude blowout approximation from betting spread.
    """
    if spread is None:
        return 0.05
    spread = abs(spread)
    if spread < 5: return 0.05
    if spread < 8: return 0.10
    if spread < 12: return 0.18
    return 0.28

def injury_context(player, team, spread=None):
    """
    Returns dict of context multipliers for integration with ensemble + MC.
    """
    injuries = fetch_injury_report()
    um = compute_usage_multiplier(player, team, injuries)
    br = compute_blowout_risk(spread)
    mm = compute_minutes_multiplier(br)
    return {
        "ctx_mult": um,
        "minutes_mult": mm,
        "blowout_risk": br,
        "injury_report": injuries.get(team, [])
    }
