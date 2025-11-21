# =========================================================
#  TIER C MODULE 1 — ODDS API CLIENT
# =========================================================

import requests
import time
from datetime import datetime, timedelta

ODDS_API_KEY = "621ec92ab709da9fce59cf2e513af55"
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

RATE_LOG = "/tmp/odds_api_rate_log.txt"
MAX_DAILY = 20

def _load_ts():
    try:
        with open(RATE_LOG, "r") as f:
            ts = [float(x) for x in f.read().split() if x.strip()]
        return ts
    except:
        return []

def _save_ts(ts):
    with open(RATE_LOG, "w") as f:
        f.write("\n".join(str(x) for x in ts))

def _check_rate():
    ts = _load_ts()
    now = time.time()
    ts = [t for t in ts if now - t < 86400]
    if len(ts) >= MAX_DAILY:
        raise Exception("Daily Odds API request limit reached.")
    ts.append(now)
    _save_ts(ts)

def fetch_odds(markets=None):
    if markets is None:
        markets = [
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_points_rebounds_assists"
        ]
    _check_rate()
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "decimal"
    }
    r = requests.get(BASE_URL, params=params, timeout=10)
    if r.status_code != 200:
        raise Exception(f"Odds API Error {r.status_code}: {r.text}")
    data = r.json()
    return data
