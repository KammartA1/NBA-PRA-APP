#!/usr/bin/env python3
"""
core/scanner_worker.py — Background scanner that runs in GitHub Actions.
Fetches PrizePicks lines → projects each prop → writes edges to Supabase → sends Telegram alerts.

Usage:
    python -m core.scanner_worker              # full scan
    python -m core.scanner_worker --test-notify # test Telegram only
    python -m core.scanner_worker --test-db     # test Supabase only
    python -m core.scanner_worker --test-pp     # test PrizePicks fetch only
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import db, notify, pp_fetcher, projections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("scanner_worker")

MIN_EV_THRESHOLD = 3.0
MIN_GAMES = 5
ALERT_EV_THRESHOLD = 6.0
MAX_WORKERS = 4
MAX_PROPS = 80          # hard cap to stay well within the 12-min time budget
TIME_BUDGET_SECS = 660  # 11 minutes — leaves 4 min for install + cleanup


def _process_one_prop(pp_line: dict) -> dict | None:
    player = pp_line["player"]
    market = pp_line.get("market")
    if not market:
        return None

    player_id = projections.resolve_player_id(player)
    if not player_id:
        log.debug("Could not resolve player: %s", player)
        return None

    gamelog = projections.fetch_player_gamelog(player_id, n_games=20)
    if gamelog is None or gamelog.empty:
        return None

    result = projections.project_player_prop(gamelog, market, pp_line["line"])
    if "error" in result:
        return None
    if result["n_games"] < MIN_GAMES:
        return None
    if result["ev_pct"] < MIN_EV_THRESHOLD:
        return None

    return {
        "player": player,
        "team": pp_line.get("team", ""),
        "opp": "",
        "market": market,
        "stat_type": pp_line.get("stat_type", ""),
        "line": pp_line["line"],
        "side": result["side"],
        "proj": result["proj"],
        "p_cal": result["p_cal"],
        "ev_pct": result["ev_pct"],
        "edge_cat": result["edge_cat"],
        "l5_avg": result["l5_avg"],
        "l10_avg": result["l10_avg"],
        "std": result["std"],
        "n_games": result["n_games"],
        "src": "PrizePicks",
    }


def run_scan() -> dict:
    scan_id = str(uuid.uuid4())[:8]
    start = time.time()
    log.info("=== SCAN START (id=%s) ===", scan_id)

    # 1. Fetch PrizePicks lines
    pp_lines, err = pp_fetcher.fetch_prizepicks_nba()
    if err:
        log.error("PrizePicks fetch failed: %s", err)
        db.log_worker_run("scanner", "error", {"error": err})
        notify.send_worker_status("ERROR", f"PrizePicks fetch failed: {err}")
        return {"ok": False, "error": err}

    log.info("Fetched %d PrizePicks NBA lines", len(pp_lines))

    # Store all PP lines for CLV tracking
    db.append_pp_lines(pp_lines)

    # 2. Filter to props we can project (known markets)
    projectable = [p for p in pp_lines if p.get("market")]
    log.info("%d lines have mappable markets", len(projectable))

    # 3. Deduplicate by player+market, then cap at MAX_PROPS
    seen_keys: dict[tuple, bool] = {}
    deduped = []
    for p in projectable:
        key = (p["player"], p["market"])
        if key not in seen_keys:
            seen_keys[key] = True
            deduped.append(p)
    if len(deduped) > MAX_PROPS:
        log.info("Capping from %d to %d props", len(deduped), MAX_PROPS)
        deduped = deduped[:MAX_PROPS]

    log.info("Processing %d unique player-market combos (from %d lines)", len(deduped), len(projectable))

    # 4. Project each prop sequentially with time-budget guard
    edges = []
    processed = 0
    skipped = 0
    timed_out = False

    for p in deduped:
        if time.time() - start > TIME_BUDGET_SECS:
            log.warning("Time budget (%.0fs) exceeded after %d props — stopping early", TIME_BUDGET_SECS, processed)
            timed_out = True
            break
        processed += 1
        try:
            result = _process_one_prop(p)
            if result:
                edges.append(result)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            log.debug("Prop processing error: %s", e)

        if processed % 20 == 0:
            elapsed = time.time() - start
            log.info("Progress: %d/%d processed, %d edges found (%.0fs elapsed)", processed, len(deduped), len(edges), elapsed)

    # Sort by EV descending
    edges.sort(key=lambda x: x.get("ev_pct", 0), reverse=True)

    elapsed = time.time() - start
    log.info(
        "=== SCAN COMPLETE (id=%s) === %d edges from %d props in %.0fs",
        scan_id, len(edges), processed, elapsed,
    )

    # 4. Write to Supabase
    if edges:
        db.deactivate_old_scans("NBA", scan_id)
        written = db.write_scan_results(scan_id, "NBA", edges)
        log.info("Wrote %d edges to Supabase", written)

    # 5. Send Telegram alerts for high-EV edges
    alert_edges = [e for e in edges if e.get("ev_pct", 0) >= ALERT_EV_THRESHOLD]
    if alert_edges:
        notify.send_edge_alert(alert_edges, sport="NBA")
        for e in alert_edges:
            db.log_notification(
                "high_ev_alert",
                f"{e['player']} {e['market']} {e['side']} {e['line']} EV:{e['ev_pct']:+.1f}%",
            )

    # 6. Log the run
    status = "partial" if timed_out else "success"
    details = {
        "scan_id": scan_id,
        "pp_lines": len(pp_lines),
        "processed": processed,
        "total_deduped": len(deduped),
        "edges_found": len(edges),
        "alerts_sent": len(alert_edges),
        "elapsed_seconds": round(elapsed, 1),
        "timed_out": timed_out,
        "top_edge": edges[0] if edges else None,
    }
    db.log_worker_run("scanner", status, details)

    return {"ok": True, **details}


# --- Test utilities ---

def test_pp():
    log.info("Testing PrizePicks fetch...")
    lines, err = pp_fetcher.fetch_prizepicks_nba()
    if err:
        log.error("FAIL: %s", err)
        return False
    log.info("SUCCESS: %d NBA lines fetched", len(lines))
    for line in lines[:5]:
        log.info("  %s | %s | %s", line["player"], line["stat_type"], line["line"])
    return True


def test_db():
    log.info("Testing Supabase connection...")
    ok = db.log_worker_run("test", "test_ping", {"test": True})
    if ok:
        log.info("SUCCESS: Supabase write OK")
    else:
        log.error("FAIL: Could not write to Supabase")
    return ok


def test_notify():
    log.info("Testing Telegram notification...")
    ok = notify.test_connection()
    if ok:
        log.info("SUCCESS: Telegram message sent")
    else:
        log.error("FAIL: Could not send Telegram message")
    return ok


def main():
    if "--test-notify" in sys.argv:
        sys.exit(0 if test_notify() else 1)
    if "--test-db" in sys.argv:
        sys.exit(0 if test_db() else 1)
    if "--test-pp" in sys.argv:
        sys.exit(0 if test_pp() else 1)
    if "--test-all" in sys.argv:
        pp_ok = test_pp()
        db_ok = test_db()
        notify_ok = test_notify()
        log.info("Results: PP=%s DB=%s Notify=%s", pp_ok, db_ok, notify_ok)
        sys.exit(0 if all([pp_ok, db_ok, notify_ok]) else 1)

    result = run_scan()
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
