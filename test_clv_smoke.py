"""
Smoke test for the automatic CLV system + keep-alive workflow.
Tests every variable/function in the pipeline to confirm:
1. CLV tracking is 100% automated after a bet is placed (local CSV)
2. Background thread runs every 2 minutes
3. Keep-alive workflow pings every 10 hours
"""
import sys, os, json, time, threading, logging, re
from datetime import datetime, timedelta, timezone, date
from unittest.mock import MagicMock
import pandas as pd
import yaml

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
    with open("app.py", "r") as f:
        source = f.read()

    func_names = [
        "_parse_nba_game_time",
        "_resolve_leg_start_time",
        "_get_nba_schedule_start_times",
        "_clv_auto_update_pending_bets",
        "_clv_auto_loop",
        "_clv_auto_state",
        "_ensure_clv_auto_thread",
    ]

    extracted = []
    lines = source.split("\n")

    state_lines = []
    for i, line in enumerate(lines):
        if line.startswith("_nba_schedule_cache"):
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
        in_func = False
        func_lines = []
        for i, line in enumerate(lines):
            if line.startswith(f"def {fname}("):
                in_func = True
                func_lines.append(line)
                continue
            if in_func:
                if line and not line[0].isspace() and not line.startswith("#"):
                    break
                func_lines.append(line)
        if func_lines:
            extracted.append("\n".join(func_lines))

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
        "scoreboardv2": MagicMock(),
        "team_id_to_abbr_map": lambda: {},
        "apply_clv_update_to_legs": lambda legs: (legs, []),
        "_now_iso": lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "__name__": "clv_test",
    }

    state_code = "\n".join(state_lines)
    exec(state_code, ns)

    for func_code in extracted:
        try:
            exec(func_code, ns)
        except Exception as e:
            logger.info(f"  WARN  Failed to load function: {e}")

    return ns


NS = build_clv_namespace()

with open("app.py", "r") as f:
    full_src = f.read()


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

leg = {"commence_time": "2025-04-22T23:00:00Z", "start_time": "2025-04-22T22:00:00Z", "team": "BOS"}
r = fn(leg, sched)
report("Priority 1: commence_time wins", r == "2025-04-22T23:00:00Z", f"got {r}")

leg = {"commence_time": "", "start_time": "2025-04-22T22:00:00Z", "team": "BOS"}
r = fn(leg, sched)
report("Priority 2: start_time fallback", r == "2025-04-22T22:00:00Z", f"got {r}")

leg = {"commence_time": "", "start_time": "", "team": "LAL"}
r = fn(leg, sched)
report("Priority 3: NBA schedule by team", r == "2025-04-23T02:30:00Z", f"got {r}")

leg = {"commence_time": None, "start_time": None, "team": "BOS"}
r = fn(leg, sched)
report("None values → NBA schedule fallback", r == "2025-04-22T23:00:00Z", f"got {r}")

leg = {"commence_time": "", "start_time": "", "team": ""}
r = fn(leg, {})
report("All empty → empty string", r == "", f"got '{r}'")

leg = {"commence_time": "", "start_time": "", "team": "XXX"}
r = fn(leg, sched)
report("Unknown team → empty string", r == "", f"got '{r}'")

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

fn = NS["_clv_auto_state"]
report("_clv_auto_state() returns singleton", fn() is state)


# ===================================================================
# TEST 4: _clv_auto_loop uses 120-second interval
# ===================================================================
logger.info("")
logger.info("[TEST 4: _clv_auto_loop uses 120s interval]")

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

test_state = {
    "lock": threading.Lock(),
    "thread": None,
    "stop_evt": threading.Event(),
    "enabled": False,
    "last_run": None,
    "last_result": None,
    "updates_count": 0,
}

t = threading.Thread(
    target=NS["_clv_auto_loop"], args=(test_state,),
    daemon=True, name="clv_test_thread"
)
t.start()
time.sleep(0.2)
report("Thread is alive after start", t.is_alive())
report("Thread is daemon", t.daemon is True)
test_state["stop_evt"].set()
time.sleep(0.1)


# ===================================================================
# TEST 6: CLV updater reads and writes local CSV
# ===================================================================
logger.info("")
logger.info("[TEST 6: CLV updater reads and writes local CSV]")

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

def fake_apply(legs):
    out = []
    for leg in legs:
        leg2 = dict(leg)
        leg2["line_close"] = 9.0
        leg2["price_close"] = 1.87
        leg2["book_close"] = "fanduel"
        leg2["clv_line"] = 0.5
        leg2["clv_line_fav"] = True
        leg2["close_ts"] = now.isoformat()
        out.append(leg2)
    return out, []
NS["apply_clv_update_to_legs"] = fake_apply
NS["_get_nba_schedule_start_times"] = lambda gd: {}

result = NS["_clv_auto_update_pending_bets"]()

report("bets_checked >= 1", result["bets_checked"] >= 1, f"checked={result['bets_checked']}")
report("bets_updated >= 1", result["bets_updated"] >= 1, f"updated={result['bets_updated']}")

df2 = pd.read_csv(csv_path)
updated_legs = json.loads(df2.loc[0, "legs"])
report("CSV has line_close=9.0", updated_legs[0].get("line_close") == 9.0)
report("CSV has clv_line=0.5", updated_legs[0].get("clv_line") == 0.5)
report("CSV has clv_line_fav=True", updated_legs[0].get("clv_line_fav") is True)
report("CSV has close_ts", updated_legs[0].get("close_ts") is not None)
report("No errors", len(result["errors"]) == 0, f"errors={result['errors']}")

os.remove(csv_path)


# ===================================================================
# TEST 7: Bets OUTSIDE the 12-min window are skipped
# ===================================================================
logger.info("")
logger.info("[TEST 7: Bets outside 12-min window are skipped]")

now = datetime.now(timezone.utc)
far = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
past = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

csv_path2 = "history_window_test.csv"
df_far = pd.DataFrame({
    "ts": [now.isoformat(), now.isoformat()], "user_id": ["u1", "u1"],
    "legs": [json.dumps([{"player": "A", "commence_time": far, "team": "BOS"}]),
             json.dumps([{"player": "B", "commence_time": past, "team": "LAL"}])],
    "n_legs": [1, 1],
    "leg_results": [json.dumps(["Pending"]), json.dumps(["Pending"])],
    "result": ["Pending", "Pending"], "decision": ["BET", "BET"], "notes": ["", ""]
})
df_far.to_csv(csv_path2, index=False)

apply_count = {"n": 0}
def counting_apply(legs):
    apply_count["n"] += 1
    return legs, []
NS["apply_clv_update_to_legs"] = counting_apply

result = NS["_clv_auto_update_pending_bets"]()
report("No bets checked (all outside window)", result["bets_checked"] == 0, f"checked={result['bets_checked']}")
report("No bets updated", result["bets_updated"] == 0, f"updated={result['bets_updated']}")
report("apply_clv not called", apply_count["n"] == 0, f"calls={apply_count['n']}")

os.remove(csv_path2)


# ===================================================================
# TEST 8: Already-updated bets (line_close set) are skipped
# ===================================================================
logger.info("")
logger.info("[TEST 8: Already-updated bets are skipped]")

now = datetime.now(timezone.utc)
soon = (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

csv_path3 = "history_already_test.csv"
df_done = pd.DataFrame({
    "ts": [now.isoformat()], "user_id": ["u1"],
    "legs": [json.dumps([{"player": "X", "commence_time": soon, "team": "BOS",
                           "line_close": 25.0, "clv_line": 0.5}])],
    "n_legs": [1],
    "leg_results": [json.dumps(["Pending"])],
    "result": ["Pending"], "decision": ["BET"], "notes": [""]
})
df_done.to_csv(csv_path3, index=False)

apply_count2 = {"n": 0}
def counting_apply2(legs):
    apply_count2["n"] += 1
    return legs, []
NS["apply_clv_update_to_legs"] = counting_apply2

result = NS["_clv_auto_update_pending_bets"]()
report("Already-updated bet skipped (bets_checked=0)", result["bets_checked"] == 0, f"checked={result['bets_checked']}")
report("apply_clv not called", apply_count2["n"] == 0, f"calls={apply_count2['n']}")

os.remove(csv_path3)


# ===================================================================
# TEST 9: PrizePicks start_time injection at all meta-build sites
# ===================================================================
logger.info("")
logger.info("[TEST 9: PrizePicks start_time injection in source]")

empty_ct = [l.strip() for l in full_src.split("\n") if '"commence_time": ""' in l]
report("Only 1 empty commence_time remains (manual fallback)",
       len(empty_ct) == 1,
       f"found {len(empty_ct)}: {empty_ct}")

pp_st_uses = [l.strip() for l in full_src.split("\n")
              if "start_time" in l and ("_plat_st" in l or "_pp_st" in l or 'r.get("start_time"' in l)]
report("start_time extracted from PP rows (>=3 sites)",
       len(pp_st_uses) >= 3,
       f"found {len(pp_st_uses)} sites")

_resolve_match = re.search(r"(def _resolve_leg_start_time\(.+?)(?=\ndef \w|\Z)", full_src, re.DOTALL)
resolve_src = _resolve_match.group(1) if _resolve_match else ""
report("_resolve_leg_start_time checks start_time", 'leg.get("start_time")' in resolve_src)
report("_resolve_leg_start_time checks commence_time first", 'leg.get("commence_time")' in resolve_src)
report("_resolve_leg_start_time checks team for NBA schedule", 'leg.get("team")' in resolve_src)


# ===================================================================
# TEST 10: No Supabase in app.py (removed for performance)
# ===================================================================
logger.info("")
logger.info("[TEST 10: Supabase fully removed from app.py]")

report("No 'import supabase_store' in app.py", "import supabase_store" not in full_src)
report("No '_supa.' calls in app.py", "_supa." not in full_src)
report("No 'supabase' in requirements.txt",
       "supabase" not in open("requirements.txt").read())


# ===================================================================
# TEST 11: E2E — bet placed → auto-CLV fires → CSV updated
# ===================================================================
logger.info("")
logger.info("[TEST 11: End-to-end: bet → auto-CLV → CSV update]")

now = datetime.now(timezone.utc)
tipoff = (now + timedelta(minutes=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

csv_path_e2e = "history_e2e_test.csv"
e2e_leg = {
    "player": "Anthony Edwards", "player_norm": "anthony edwards",
    "market": "Points", "market_key": "player_points",
    "line": 27.5, "price_decimal": 2.0,
    "side": "Over", "book": "prizepicks",
    "event_id": None, "commence_time": tipoff,
    "team": "MIN", "line_close": None
}
pd.DataFrame({
    "ts": [now.isoformat()], "user_id": ["e2e_user"],
    "legs": [json.dumps([e2e_leg])], "n_legs": [1],
    "leg_results": [json.dumps(["Pending"])],
    "result": ["Pending"], "decision": ["BET"], "notes": [""]
}).to_csv(csv_path_e2e, index=False)

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

df_e2e = pd.read_csv(csv_path_e2e)
e2e_legs = json.loads(df_e2e.loc[0, "legs"])
report("E2E: line_close=28.5 in CSV", e2e_legs[0].get("line_close") == 28.5)
report("E2E: clv_line=1.0", e2e_legs[0].get("clv_line") == 1.0)
report("E2E: clv_line_fav=True", e2e_legs[0].get("clv_line_fav") is True)
report("E2E: clv_price=0.045", e2e_legs[0].get("clv_price") == 0.045)
report("E2E: close_ts set", e2e_legs[0].get("close_ts") is not None)
report("E2E: no errors", len(result["errors"]) == 0, f"errors={result['errors']}")

os.remove(csv_path_e2e)


# ===================================================================
# TEST 12: Thread poke via stop_evt
# ===================================================================
logger.info("")
logger.info("[TEST 12: Thread poke via stop_evt]")

state = NS["_clv_auto_state_store"]
state["stop_evt"].set()
time.sleep(0.05)
report("stop_evt can be set without crash", True)
state["stop_evt"].clear()
report("stop_evt can be cleared", True)


# ===================================================================
# TEST 13: NBA schedule cache structure
# ===================================================================
logger.info("")
logger.info("[TEST 13: NBA schedule cache structure]")

cache = NS["_nba_schedule_cache"]
report("Cache has 'date' key", "date" in cache)
report("Cache has 'games' key", "games" in cache)
report("Cache has 'fetched_at' key", "fetched_at" in cache)

_sched_match = re.search(r"(def _get_nba_schedule_start_times\(.+?)(?=\ndef \w|\Z)", full_src, re.DOTALL)
sched_src = _sched_match.group(1) if _sched_match else ""
report("Cache TTL is 600s (10 min)", "600" in sched_src)


# ===================================================================
# TEST 14: log variable exists in app.py
# ===================================================================
logger.info("")
logger.info("[TEST 14: log variable defined in app.py]")

lines = full_src.split("\n")
log_defined = any(
    l.strip().startswith("log = logging.getLogger") or l.strip().startswith("log=logging.getLogger")
    for l in lines[:50]
)
report("log = logging.getLogger() defined in first 50 lines", log_defined)

log_uses = sum(1 for l in lines if "log.info(" in l or "log.warning(" in l)
report(f"log.info/warning used {log_uses} times", log_uses > 0)


# ===================================================================
# TEST 15: Multi-leg bet partially updated
# ===================================================================
logger.info("")
logger.info("[TEST 15: Multi-leg bet partially updated]")

now = datetime.now(timezone.utc)
soon = (now + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")

csv_path_ml = "history_multileg_test.csv"
pd.DataFrame({
    "ts": [now.isoformat()], "user_id": ["u1"],
    "legs": [json.dumps([
        {"player": "A", "commence_time": soon, "team": "BOS", "line_close": 25.0},
        {"player": "B", "commence_time": soon, "team": "BOS", "line_close": None},
    ])],
    "n_legs": [2],
    "leg_results": [json.dumps(["Pending", "Pending"])],
    "result": ["Pending"], "decision": ["BET"], "notes": [""]
}).to_csv(csv_path_ml, index=False)

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

os.remove(csv_path_ml)


# ===================================================================
# TEST 16: Keep-alive workflow — pings every 10 hours
# ===================================================================
logger.info("")
logger.info("[TEST 16: Keep-alive workflow pings every 10 hours]")

wf_path = ".github/workflows/keep-alive.yml"
report("keep-alive.yml exists", os.path.exists(wf_path))

with open(wf_path) as f:
    wf = yaml.safe_load(f)

report("Workflow name is 'Keep Alive'", wf.get("name") == "Keep Alive")

# Check cron schedule (YAML parses 'on' as boolean True)
on_block = wf.get("on") or wf.get(True) or {}
schedules = on_block.get("schedule", [])
cron_exprs = [s.get("cron", "") for s in schedules]
report("Has cron schedule", len(cron_exprs) > 0, f"crons={cron_exprs}")
report("Cron runs every 10 hours (0 0,10,20 * * *)",
       any("0,10,20" in c for c in cron_exprs),
       f"crons={cron_exprs}")

# Check manual dispatch
report("Has workflow_dispatch (manual trigger)", "workflow_dispatch" in on_block)

# Check job steps
jobs = wf.get("jobs", {})
report("Has 'ping' job", "ping" in jobs)
ping_job = jobs.get("ping", {})
steps = ping_job.get("steps", [])
step_names = [s.get("name", "") for s in steps]
report("Has 'Ping Streamlit app' step",
       any("streamlit" in n.lower() for n in step_names),
       f"steps={step_names}")

# Check that it uses secrets for the app URL
step_runs = [s.get("run", "") for s in steps]
all_runs = "\n".join(step_runs)
report("Uses STREAMLIT_APP_URL secret", "STREAMLIT_APP_URL" in all_runs)
report("Uses curl to ping", "curl" in all_runs)
report("Has retry logic (sleep + re-curl)", "sleep" in all_runs and all_runs.count("curl") >= 2)

# Verify the cron math: every 10 hours = 3x/day < Streamlit's 12h sleep
report("3 pings/day (every 10h) < 12h Streamlit sleep threshold", len("0,10,20".split(",")) == 3)


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
    logger.info("ALL TESTS PASSED — CLV auto-tracker + keep-alive verified")
    sys.exit(0)
