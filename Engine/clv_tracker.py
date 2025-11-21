# =========================================================
#  TIER C MODULE 10 — CLV TRACKER
# =========================================================

import json, os, time
from datetime import datetime

CLV_FILE = "/tmp/nba_quant_clv.json"

def _load_clv():
    if not os.path.exists(CLV_FILE):
        return {}
    try:
        with open(CLV_FILE,'r') as f:
            return json.load(f)
    except:
        return {}

def _save_clv(data):
    with open(CLV_FILE,'w') as f:
        json.dump(data,f,indent=2)

def record_am_line(player, market, line, model_prob, timestamp=None):
    data = _load_clv()
    key = f"{player}_{market}"
    if timestamp is None:
        timestamp = time.time()
    data[key] = {
        "am_line": line,
        "model_am": model_prob,
        "timestamp": timestamp
    }
    _save_clv(data)

def compute_clv(player, market, current_line, model_now):
    data = _load_clv()
    key = f"{player}_{market}"
    if key not in data:
        return None
    entry = data[key]
    am_line = entry["am_line"]
    model_am = entry["model_am"]
    clv_value = (model_am - model_now) * 100.0
    return {
        "clv_value": clv_value,
        "am_line": am_line,
        "current_line": current_line,
        "model_AM": model_am,
        "model_now": model_now
    }
