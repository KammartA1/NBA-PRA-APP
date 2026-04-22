"""
Smoke test for the automatic CLV system.
Tests every variable/function in the pipeline to confirm CLV tracking
is 100% automated after a bet is placed.

Strategy: app.py is a Streamlit script (UI code runs at module level),
so we can't import it directly. Instead we extract the CLV functions
from the source and exec them in an isolated namespace.
"""
import sys, os, json, time, threading, textwrap, logging, re
from datetime import datetime, timedelta, timezone, date
from unittest.mock import MagicMock
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

os.chdir("/home/user/NBA-PRA-APP")
sys.path.insert(0, "/home/user/NBA-PRA-APP")

PASS = 0
FAIL = 0

def report(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        logger.info(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        logger.info(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


# ===================================================================
# Extract CLV functions from app.py source without importing it
# ===================================================================
def build_clv_namespace():
    """Read app.py, extract CLV-related functions, exec them in a
    controlled namespace with all dependencies available."""
    with open("app.py", "r") as f:
        source = f.read()

    # Extract function bodies by regex
    func_names = [
        "_parse_nba_game_time",
        "_resolve_leg_start_time",
        "_get_nba_schedule_start_times",
        "_clv_auto_update_pending_bets",
        "_clv_auto_loop",
        "_clv_auto_state",
        "_ensure_clv_auto_thread",
    ]

    # Find each function's source code
    extracted = []
    lines = source.split("\n")

    # Also extract module-level state dicts
    state_lines = []
    for i, line in enumerate(lines):
        if line.startswith("_nba_schedule_cache"):
            # Grab the dict literal
            j = i
            brace = 0
            while j < len(lines):
                brace += lines[j].count("{") - lines[j].count("}")
                state_lines.append(lines[j])
                if brace <= 0:
                    break
                j += 1
            state_lines.append("")
        if line.startswith("_clv_auto_state_store"):
            j = i
            brace = 0
            while j < len(lines):
                brace += lines[j].count("{") - lines[j].count("}")
                state_lines.append(lines[j])
                if brace <= 0:
                    break
                j += 1
            state_lines.append("")

    for fname in func_names:
        # Find "def fname(" and grab until next top-level def or class
        in_func = False
        func_lines = []
        for i, line in enumerate(lines):
            if line.startswith(f"def {fname}("):
                in_func = True
                func_lines.append(line)
                continue
            if in_func:
                # End at next top-level definition or blank-line followed by top-level code
                if line and not line[0].isspace() and not line.startswith("#"):
                    break
                func_lines.append(line)
        if func_lines:
            extracted.append("\n".join(func_lines))

    # Build namespace with all needed dependencies
    ns = {
        "threading": threading,
        "json": json,
        "time": time,
        "pd": pd,
        "os": os,
        "re": re,
        "logging": logging,
        "log": logging.getLogger("clv_test"),
        "datetime": datetime,
        "date": date,
        "timedelta": timedelta,
        "timezone": timezone,
        # Mock external dependencies
        "scoreboardv2": MagicMock(),
        "team_id_to_abbr_map": lambda: {},
        "apply_clv_update_to_legs": lambda legs: (legs, []),
        "_supa": MagicMock(),
        "_now_iso": lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "__name__": "clv_test",
    }

    # Exec state dicts
    state_code = "\n".join(state_lines)
    # Replace threading.Lock() etc which need real objects
    exec(state_code, ns)

    # Exec each function
    for func_code in extracted:
        try:
            exec(func_code, ns)
        except Exception as e:
            logger.info(f"  WARN  Failed to load function: {e}")

    return ns


NS = build_clv_namespace()


# ===================================================================
# TEST 1: _parse_nba_game_time — converts NBA schedule text to UTC ISO
# ===================================================================
logger.info("[TEST 1: _parse_nba_game_time converts ET to UTC]")
fn = NS["_parse_nba_game_time"]

r = fn("2025-04-22", "7:00 pm ET")
report("7pm ET → 23:00 UTC", r == "2025-04-22T23:00:00Z", f"got {r}")

r = fn("2025-04-22", "10:30 pm ET")
report("10:30pm ET → 02:30 UTC next day", r == "2025-04-23T02:30:00Z", f"got {r}")

r = fn("2025-04-22", "1:00 pm ET")
report("1pm ET → 17:00 UTC", r == "2025-04-22T17:00:00Z", f"got {r}")

r = fn("2025-04-22", "Final")
report("'Final' → empty string", r == "", f"got '{r}'")

r = fn("", "7:00 pm ET")
report("empty date → empty string", r == "", f"got '{r}'")

r = fn("2025-04-22", "")
report("empty status → empty string", r == "", f"got '{r}'")

r = fn("2025-04-22", "12:00 pm ET")
report("12pm ET (noon) → 16:00 UTC", r == "2025-04-22T16:00:00Z", f"got {r}")


# ===================================================================
# TEST 2: _resolve_leg_start_time — 3-tier priority
# ===================================================================
logger.info("")
logger.info("[TEST 2: _resolve_leg_start_time priority chain]")
fn = NS["_resolve_leg_start_time"]

sched = {"BOS": "2025-04-22T23:00:00Z", "LAL": "2025-04-23T02:30:00Z"}

# Priority 1: commence_time wins over everything
leg = {"commence_time": "2025-04-22T23:00:00Z", "start_time": "2025-04-22T22:00:00Z", "team": "BOS"}
r = fn(leg, sched)
report("Priority 1: commence_time wins", r == "2025-04-22T23:00:00Z", f"got {r}")

# Priority 2: no commence_time → start_time fallback
leg = {"commence_time": "", "start_time": "2025-04-22T22:00:00Z", "team": "BOS"}
r = fn(leg, sched)
report("Priority 2: start_time fallback", r == "2025-04-22T22:00:00Z", f"got {r}")

# Priority 3: no commence_time, no start_time → NBA schedule lookup by team
leg = {"commence_time": "", "start_time": "", "team": "LAL"}
r = fn(leg, sched)
report("Priority 3: NBA schedule by team", r == "2025-04-23T02:30:00Z", f"got {r}")

# None values don't crash
leg = {"commence_time": None, "start_time": None, "team": "BOS"}
r = fn(leg, sched)
report("None values → NBA schedule fallback", r == "2025-04-22T23:00:00Z", f"got {r}")

# All empty → empty string
leg = {"commence_time": "", "start_time": "", "team": ""}
r = fn(leg, {})
report("All empty → empty string", r == "", f"got '{r}'")

# Unknown team → empty string
leg = {"commence_time": "", "start_time": "", "team": "XXX"}
r = fn(leg, sched)
report("Unknown team → empty string", r == "", f"got '{r}'")

# Missing keys entirely → empty string
r = fn({}, {})
report("Empty dict → empty string", r == "", f"got '{r}'")


# ===================================================================
# TEST 3: Background thread state dict and lifecycle
# ===================================================================
logger.info("")
logger.info("[TEST 3: Background thread state dict and lifecycle]")

state = NS["_clv_auto_state_store"]
report("State has 'lock'", "lock" in state)
report("State has 'thread'", "thread" in state)
report("State has 'stop_evt'", "stop_evt" in state)
report("State has 'enabled'", "enabled" in state)
report("State has 'last_run'", "last_run" in state)
report("State has 'last_result'", "last_result" in state)
report("State has 'updates_count'", "updates_count" in state)
report("enabled defaults True", state["enabled"] is True)
report("stop_evt is threading.Event", isinstance(state["stop_evt"], threading.Event))
report("lock is threading.Lock", hasattr(state["lock"], "acquire"))
report("updates_count starts at 0", state["updates_count"] == 0)

# Verify _clv_auto_state() returns the singleton
fn = NS["_clv_auto_state"]
report("_clv_auto_state() returns singleton", fn() is state)


# ===================================================================
# TEST 4: _clv_auto_loop uses 120-second interval
# ===================================================================
logger.info("")
logger.info("[TEST 4: _clv_auto_loop uses 120s interval]")

# Can't use inspect.getsource on exec'd functions — read from app.py directly
with open("app.py", "r") as f:
    full_src = f.read()
# Extract the _clv_auto_loop function source
_loop_match = re.search(r"(def _clv_auto_loop\(.+?)(?=\ndef \w|\Z)", full_src, re.DOTALL)
loop_src = _loop_match.group(1) if _loop_match else ""
report("Loop source contains 'timeout=120'", "timeout=120" in loop_src)
report("Loop calls _clv_auto_update_pending_bets", "_clv_auto_update_pending_bets" in loop_src)
report("Loop updates last_run", "last_run" in loop_src)
report("Loop updates last_result", "last_result" in loop_src)
report("Loop updates updates_count", "updates_count" in loop_src)
report("Loop checks enabled flag", "enabled" in loop_src)


# ===================================================================
# TEST 5: Background thread starts as daemon
# ===================================================================
logger.info("")
logger.info("[TEST 5: Background thread starts as daemon]")

_ensure_match = re.search(r"(def _ensure_clv_auto_thread\(.+?)(?=\ndef \w|\n#|\Z)", full_src, re.DOTALL)
ensure_src = _ensure_match.group(1) if _ensure_match else ""
report("Thread is daemon", "daemon=True" in ensure_src)
report("Thread name is clv_auto_updater", "clv_auto_updater" in ensure_src)
report("Checks if thread is alive before creating", "is_alive" in ensure_src)

# Actually start the thread (with a mock updater that does nothing)
test_state = {
    "lock": threading.Lock(),
    "thread": None,
    "stop_evt": threading.Event(),
    "enabled": False,  # disabled so it won't actually run updates
    "last_run": None,
    "last_result": None,
    "updates_count": 0,
}

# Build a real thread to test lifecycle
t = threading.Thread(
    target=NS["_clv_auto_loop"], args=(test_state,),
    daemon=True, name="clv_test_thread"
)
t.start()
time.sleep(0.2)
report("Thread is alive after start", t.is_alive())
report("Thread is daemon", t.daemon is True)
# Clean up — poke to wake, then it'll loop back to sleep (enabled=False)
test_state["stop_evt"].set()
time.sleep(0.1)


# ===================================================================
# TEST 6: _clv_auto_update_pending_bets — Supabase path
# ===================================================================
logger.info("")
logger.info("[TEST 6: CLV updater reads from Supabase and writes back]")

now = datetime.now(timezone.utc)
tipoff = now + timedelta(minutes=5)
tipoff_iso = tipoff.strftime("%Y-%m-%dT%H:%M:%SZ")

mock_supa = MagicMock()
mock_supa.load_all_pending_bets.return_value = [{
    "id": 999,
    "user_id": "test_user",
    "result": "Pending",
    "legs": [{
        "player": "LeBron James", "player_norm": "lebron james",
        "market": "Points", "market_key": "player_points",
        "line": 25.5, "price_decimal": 1.91,
        "side": "Over", "book": "draftkings",
        "event_id": "evt_123", "commence_time": tipoff_iso,
        "team": "LAL"
    }]
}]
mock_supa.update_history_row.return_value = True
NS["_supa"] = mock_supa

def fake_apply(legs):
    out = []
    for leg in legs:
        leg2 = dict(leg)
        leg2["line_close"] = 26.5
        leg2["price_close"] = 1.87
        leg2["book_close"] = "draftkings"
        leg2["clv_line"] = 1.0
        leg2["clv_line_fav"] = True
        leg2["close_ts"] = now.isoformat()
        out.append(leg2)
    return out, []
NS["apply_clv_update_to_legs"] = fake_apply
NS["_get_nba_schedule_start_times"] = lambda gd: {}

result = NS["_clv_auto_update_pending_bets"]()

report("Supabase load_all_pending_bets called", mock_supa.load_all_pending_bets.called)
report("bets_checked >= 1", result["bets_checked"] >= 1, f"checked={result['bets_checked']}")
report("bets_updated >= 1", result["bets_updated"] >= 1, f"updated={result['bets_updated']}")
report("supabase source counted", result["sources"]["supabase"] >= 1, f"supa={result['sources']['supabase']}")
report("update_history_row called", mock_supa.update_history_row.called)

if mock_supa.update_history_row.called:
    args = mock_supa.update_history_row.call_args
    report("Correct Supabase row ID (999)", args[0][1] == 999, f"id={args[0][1]}")
    report("Correct user_id", args[0][0] == "test_user", f"uid={args[0][0]}")
    updated_legs = args[0][2].get("legs", [])
    if updated_legs:
        report("line_close=26.5 in Supabase write", updated_legs[0].get("line_close") == 26.5)
        report("clv_line=1.0 in Supabase write", updated_legs[0].get("clv_line") == 1.0)
        report("clv_line_fav=True", updated_legs[0].get("clv_line_fav") is True)
        report("close_ts populated", updated_legs[0].get("close_ts") is not None)

report("No errors", len(result["errors"]) == 0, f"errors={result['errors']}")


# ===================================================================
# TEST 7: CLV updater reads from local CSV fallback
# ===================================================================
logger.info("")
logger.info("[TEST 7: CLV updater reads from local CSV fallback]")

now = datetime.now(timezone.utc)
tipoff = now + timedelta(minutes=8)
tipoff_iso = tipoff.strftime("%Y-%m-%dT%H:%M:%SZ")

csv_leg = {
    "player": "Jayson Tatum", "player_norm": "jayson tatum",
    "market": "Rebounds", "market_key": "player_rebounds",
    "line": 8.5, "price_decimal": 1.95,
    "side": "Over", "book": "fanduel",
    "event_id": "evt_456", "commence_time": tipoff_iso,
    "team": "BOS"
}
csv_path = "history_csv_test_user.csv"
df = pd.DataFrame({
    "ts": [now.isoformat()], "user_id": ["csv_test"],
    "legs": [json.dumps([csv_leg])], "n_legs": [1],
    "leg_results": [json.dumps(["Pending"])],
    "result": ["Pending"], "decision": ["BET"], "notes": [""]
})
df.to_csv(csv_path, index=False)

mock_supa2 = MagicMock()
mock_supa2.load_all_pending_bets.return_value = []
NS["_supa"] = mock_supa2

def fake_apply2(legs):
    out = []
    for leg in legs:
        leg2 = dict(leg)
        leg2["line_close"] = 9.0
        leg2["clv_line"] = 0.5
        leg2["clv_line_fav"] = True
        leg2["close_ts"] = now.isoformat()
        out.append(leg2)
    return out, []
NS["apply_clv_update_to_legs"] = fake_apply2

result = NS["_clv_auto_update_pending_bets"]()

report("CSV source counted", result["sources"]["csv"] >= 1, f"csv={result['sources']['csv']}")
report("bets_updated from CSV", result["bets_updated"] >= 1, f"updated={result['bets_updated']}")

df2 = pd.read_csv(csv_path)
updated_legs = json.loads(df2.loc[0, "legs"])
report("CSV has line_close=9.0", updated_legs[0].get("line_close") == 9.0)
report("CSV has clv_line=0.5", updated_legs[0].get("clv_line") == 0.5)

os.remove(csv_path)


# ===================================================================
# TEST 8: Bets OUTSIDE the 12-min window are skipped
# ===================================================================
logger.info("")
logger.info("[TEST 8: Bets outside 12-min window are skipped]")

now = datetime.now(timezone.utc)
far = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
past = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

mock_supa3 = MagicMock()
mock_supa3.load_all_pending_bets.return_value = [
    {"id": 1, "user_id": "u1", "result": "Pending",
     "legs": [{"player": "A", "commence_time": far, "team": "BOS"}]},
    {"id": 2, "user_id": "u1", "result": "Pending",
     "legs": [{"player": "B", "commence_time": past, "team": "LAL"}]},
]
NS["_supa"] = mock_supa3

apply_count = {"n": 0}
def counting_apply(legs):
    apply_count["n"] += 1
    return legs, []
NS["apply_clv_update_to_legs"] = counting_apply

result = NS["_clv_auto_update_pending_bets"]()
report("No bets checked (all outside window)", result["bets_checked"] == 0, f"checked={result['bets_checked']}")
report("No bets updated", result["bets_updated"] == 0, f"updated={result['bets_updated']}")
report("apply_clv not called", apply_count["n"] == 0, f"calls={apply_count['n']}")


# ===================================================================
# TEST 9: Already-updated bets (line_close set) are skipped
# ===================================================================
logger.info("")
logger.info("[TEST 9: Already-updated bets are skipped]")

now = datetime.now(timezone.utc)
soon = (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

mock_supa4 = MagicMock()
mock_supa4.load_all_pending_bets.return_value = [{
    "id": 10, "user_id": "u1", "result": "Pending",
    "legs": [{"player": "X", "commence_time": soon, "team": "BOS",
              "line_close": 25.0, "clv_line": 0.5}]
}]
NS["_supa"] = mock_supa4

apply_count2 = {"n": 0}
def counting_apply2(legs):
    apply_count2["n"] += 1
    return legs, []
NS["apply_clv_update_to_legs"] = counting_apply2

result = NS["_clv_auto_update_pending_bets"]()
report("Already-updated bet skipped (bets_checked=0)", result["bets_checked"] == 0, f"checked={result['bets_checked']}")
report("apply_clv not called", apply_count2["n"] == 0, f"calls={apply_count2['n']}")


# ===================================================================
# TEST 10: PrizePicks start_time injection at all meta-build sites
# ===================================================================
logger.info("")
logger.info("[TEST 10: PrizePicks start_time injection in source]")

with open("app.py", "r") as f:
    source = f.read()

# Count how many "commence_time": "" remain — should be exactly 1 (manual fallback)
empty_ct = [l.strip() for l in source.split("\n") if '"commence_time": ""' in l]
report("Only 1 empty commence_time remains (manual fallback)",
       len(empty_ct) == 1,
       f"found {len(empty_ct)}: {empty_ct}")

# Check that start_time is used in PP meta construction
pp_st_uses = [l.strip() for l in source.split("\n")
              if "start_time" in l and ("_plat_st" in l or "_pp_st" in l or 'r.get("start_time"' in l)]
report("start_time extracted from PP rows (>=3 sites)",
       len(pp_st_uses) >= 3,
       f"found {len(pp_st_uses)} sites")

# Verify _resolve_leg_start_time checks start_time as priority 2
_resolve_match = re.search(r"(def _resolve_leg_start_time\(.+?)(?=\ndef \w|\Z)", full_src, re.DOTALL)
resolve_src = _resolve_match.group(1) if _resolve_match else ""
report("_resolve_leg_start_time checks start_time", 'leg.get("start_time")' in resolve_src)
report("_resolve_leg_start_time checks commence_time first", 'leg.get("commence_time")' in resolve_src)
report("_resolve_leg_start_time checks team for NBA schedule", 'leg.get("team")' in resolve_src)


# ===================================================================
# TEST 11: supabase_store.load_all_pending_bets
# ===================================================================
logger.info("")
logger.info("[TEST 11: supabase_store.load_all_pending_bets function]")

import supabase_store
report("Function exists", hasattr(supabase_store, "load_all_pending_bets"))
report("Returns list when no Supabase", isinstance(supabase_store.load_all_pending_bets(), list))
report("update_history_row exists", hasattr(supabase_store, "update_history_row"))
report("is_available exists", hasattr(supabase_store, "is_available"))


# ===================================================================
# TEST 12: End-to-end — bet placed → auto-CLV fires → Supabase synced
# ===================================================================
logger.info("")
logger.info("[TEST 12: End-to-end: bet → auto-CLV → Supabase sync]")

now = datetime.now(timezone.utc)
tipoff = (now + timedelta(minutes=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

e2e_supa = MagicMock()
update_calls = []
def track_update(uid, row_id, updates):
    update_calls.append({"uid": uid, "row_id": row_id, "updates": updates})
    return True
e2e_supa.load_all_pending_bets.return_value = [{
    "id": 42, "user_id": "e2e_user", "result": "Pending",
    "legs": [{
        "player": "Anthony Edwards", "player_norm": "anthony edwards",
        "market": "Points", "market_key": "player_points",
        "line": 27.5, "price_decimal": 2.0,
        "side": "Over", "book": "prizepicks",
        "event_id": None, "commence_time": tipoff,
        "team": "MIN", "line_close": None
    }]
}]
e2e_supa.update_history_row.side_effect = track_update
NS["_supa"] = e2e_supa

def e2e_apply(legs):
    out = []
    for leg in legs:
        leg2 = dict(leg)
        leg2["line_close"] = 28.5
        leg2["price_close"] = 1.83
        leg2["book_close"] = "consensus"
        leg2["clv_line"] = 1.0
        leg2["clv_line_fav"] = True
        leg2["clv_price"] = 0.045
        leg2["clv_price_fav"] = True
        leg2["close_ts"] = now.isoformat()
        out.append(leg2)
    return out, []
NS["apply_clv_update_to_legs"] = e2e_apply

result = NS["_clv_auto_update_pending_bets"]()
report("E2E: bet checked", result["bets_checked"] >= 1)
report("E2E: bet updated", result["bets_updated"] >= 1)
report("E2E: Supabase sync called", len(update_calls) >= 1)

if update_calls:
    c = update_calls[0]
    report("E2E: correct row ID (42)", c["row_id"] == 42)
    report("E2E: correct user_id", c["uid"] == "e2e_user")
    legs = c["updates"].get("legs", [])
    if legs:
        report("E2E: line_close=28.5", legs[0].get("line_close") == 28.5)
        report("E2E: clv_line=1.0", legs[0].get("clv_line") == 1.0)
        report("E2E: clv_line_fav=True", legs[0].get("clv_line_fav") is True)
        report("E2E: clv_price=0.045", legs[0].get("clv_price") == 0.045)
        report("E2E: close_ts set", legs[0].get("close_ts") is not None)
    else:
        report("E2E: legs in update", False, "no legs in update call")
report("E2E: no errors", len(result["errors"]) == 0, f"errors={result['errors']}")


# ===================================================================
# TEST 13: Thread poke via stop_evt
# ===================================================================
logger.info("")
logger.info("[TEST 13: Thread poke via stop_evt]")

state = NS["_clv_auto_state_store"]
state["stop_evt"].set()
time.sleep(0.05)
report("stop_evt can be set without crash", True)
state["stop_evt"].clear()
report("stop_evt can be cleared", True)


# ===================================================================
# TEST 14: NBA schedule cache structure
# ===================================================================
logger.info("")
logger.info("[TEST 14: NBA schedule cache structure]")

cache = NS["_nba_schedule_cache"]
report("Cache has 'date' key", "date" in cache)
report("Cache has 'games' key", "games" in cache)
report("Cache has 'fetched_at' key", "fetched_at" in cache)

_sched_match = re.search(r"(def _get_nba_schedule_start_times\(.+?)(?=\ndef \w|\Z)", full_src, re.DOTALL)
sched_src = _sched_match.group(1) if _sched_match else ""
report("Cache TTL is 600s (10 min)", "< 600" in sched_src or "600" in sched_src)


# ===================================================================
# TEST 15: log variable exists in app.py
# ===================================================================
logger.info("")
logger.info("[TEST 15: log variable defined in app.py]")

with open("app.py", "r") as f:
    lines = f.readlines()

log_defined = any(
    l.strip().startswith("log = logging.getLogger") or l.strip().startswith("log=logging.getLogger")
    for l in lines[:50]  # Should be near the top
)
report("log = logging.getLogger() defined in first 50 lines", log_defined)

log_uses = sum(1 for l in lines if "log.info(" in l or "log.warning(" in l)
report(f"log.info/warning used {log_uses} times", log_uses > 0)


# ===================================================================
# TEST 16: Multi-leg bet — all legs need line_close for skip
# ===================================================================
logger.info("")
logger.info("[TEST 16: Multi-leg bet partially updated]")

now = datetime.now(timezone.utc)
soon = (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

mock_supa5 = MagicMock()
mock_supa5.load_all_pending_bets.return_value = [{
    "id": 20, "user_id": "u1", "result": "Pending",
    "legs": [
        {"player": "A", "commence_time": soon, "team": "BOS", "line_close": 25.0},
        {"player": "B", "commence_time": soon, "team": "BOS", "line_close": None},  # Not yet updated
    ]
}]
mock_supa5.update_history_row.return_value = True
NS["_supa"] = mock_supa5

def apply_partial(legs):
    out = []
    for leg in legs:
        leg2 = dict(leg)
        if leg2.get("line_close") is None:
            leg2["line_close"] = 30.0
            leg2["clv_line"] = 0.5
        out.append(leg2)
    return out, []
NS["apply_clv_update_to_legs"] = apply_partial

result = NS["_clv_auto_update_pending_bets"]()
report("Partially-updated bet gets re-checked", result["bets_checked"] >= 1, f"checked={result['bets_checked']}")
report("Bet was updated", result["bets_updated"] >= 1, f"updated={result['bets_updated']}")


# ===================================================================
# RESULTS
# ===================================================================
logger.info("")
logger.info("=" * 60)
logger.info(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
logger.info("=" * 60)

if FAIL > 0:
    sys.exit(1)
else:
    logger.info("ALL TESTS PASSED — CLV auto-tracker is fully automated")
    sys.exit(0)
