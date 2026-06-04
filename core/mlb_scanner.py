#!/usr/bin/env python3
"""
core/mlb_scanner.py — Background MLB scanner (runs in GitHub Actions).
Fetches PrizePicks MLB lines → projects each prop → writes edges to Supabase → Telegram alerts.

Usage:
    python -m core.mlb_scanner              # full scan
    python -m core.mlb_scanner --test-pp    # test PrizePicks fetch only
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core import db, notify, pp_fetcher
from core import mlb_projections as projections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("mlb_scanner")

MIN_EV_THRESHOLD = 3.0
MIN_GAMES = 5
ALERT_EV_THRESHOLD = 6.0
MAX_PROPS = 100
TIME_BUDGET_SECS = 600


def _process_one_prop(pp_line: dict) -> dict | None:
    player = pp_line["player"]
    stat_type = pp_line.get("stat_type", "")
    market = projections.MLB_MARKET_MAP.get(stat_type)
    if not market:
        return None

    player_id = projections.resolve_player_id(player)
    if not player_id:
        log.debug("Could not resolve MLB player: %s", player)
        return None

    ptype = projections.get_player_type(player_id)
    gamelog = projections.fetch_player_gamelog(player_id, n_games=20, player_type=ptype)
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
        "stat_type": stat_type,
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


def _fetch_mlb_lines() -> tuple[list[dict], str | None]:
    """Fetch PrizePicks lines filtered to MLB only."""
    all_rows, err = pp_fetcher.fetch_prizepicks_all_sports()
    if err:
        return [], err
    mlb = [r for r in all_rows if str(r.get("league", "")).upper().startswith("MLB")]
    log.info("Filtered %d MLB lines from %d total", len(mlb), len(all_rows))
    return mlb, None


def run_scan() -> dict:
    scan_id = str(uuid.uuid4())[:8]
    start = time.time()
    log.info("=== MLB SCAN START (id=%s) ===", scan_id)

    pp_lines, err = _fetch_mlb_lines()
    if err:
        log.warning("First PP fetch failed (%s), waiting 30s for retry…", err)
        time.sleep(30)
        pp_lines, err = _fetch_mlb_lines()
    if err:
        log.error("PrizePicks fetch failed after retry: %s", err)
        db.log_worker_run("mlb_scanner", "error", {"error": err})
        notify.send_worker_status("ERROR", f"MLB PrizePicks fetch failed: {err}")
        return {"ok": False, "error": err}

    log.info("Fetched %d PrizePicks MLB lines", len(pp_lines))

    projectable = [p for p in pp_lines if projections.MLB_MARKET_MAP.get(p.get("stat_type", ""))]
    log.info("%d MLB lines have mappable markets", len(projectable))

    seen_keys: dict[tuple, bool] = {}
    deduped = []
    for p in projectable:
        key = (p["player"], projections.MLB_MARKET_MAP.get(p["stat_type"]))
        if key not in seen_keys:
            seen_keys[key] = True
            deduped.append(p)
    if len(deduped) > MAX_PROPS:
        log.info("Capping from %d to %d props", len(deduped), MAX_PROPS)
        deduped = deduped[:MAX_PROPS]

    log.info("Processing %d unique MLB player-market combos", len(deduped))

    edges = []
    processed = 0
    skipped = 0
    timed_out = False

    for p in deduped:
        if time.time() - start > TIME_BUDGET_SECS:
            log.warning("Time budget exceeded after %d props", processed)
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
            log.debug("MLB prop error: %s", e)

        if processed % 20 == 0:
            elapsed = time.time() - start
            log.info("Progress: %d/%d, %d edges (%.0fs)", processed, len(deduped), len(edges), elapsed)

    edges.sort(key=lambda x: x.get("ev_pct", 0), reverse=True)
    elapsed = time.time() - start
    log.info("=== MLB SCAN COMPLETE (id=%s) === %d edges from %d props in %.0fs",
             scan_id, len(edges), processed, elapsed)

    if edges:
        db.deactivate_old_scans("MLB", scan_id)
        written = db.write_scan_results(scan_id, "MLB", edges)
        log.info("Wrote %d MLB edges to Supabase", written)

    alert_edges = [e for e in edges if e.get("ev_pct", 0) >= ALERT_EV_THRESHOLD]
    if alert_edges:
        notify.send_edge_alert(alert_edges, sport="MLB")

    status = "partial" if timed_out else "success"
    details = {
        "scan_id": scan_id,
        "pp_lines": len(pp_lines),
        "processed": processed,
        "edges_found": len(edges),
        "alerts_sent": len(alert_edges),
        "elapsed_seconds": round(elapsed, 1),
        "timed_out": timed_out,
    }
    db.log_worker_run("mlb_scanner", status, details)

    return {"ok": True, **details}


def main():
    if "--test-pp" in sys.argv:
        log.info("Testing MLB PrizePicks fetch...")
        lines, err = _fetch_mlb_lines()
        if err:
            log.error("FAIL: %s", err)
            sys.exit(1)
        log.info("SUCCESS: %d MLB lines", len(lines))
        for ln in lines[:5]:
            log.info("  %s | %s | %s", ln["player"], ln.get("stat_type"), ln.get("line"))
        sys.exit(0)

    try:
        result = run_scan()
    except Exception as e:
        import traceback
        log.error("MLB scan crashed: %s\n%s", e, traceback.format_exc())
        try:
            db.log_worker_run("mlb_scanner", "error", {"error": str(e)})
            notify.send_worker_status("CRASHED", f"MLB scanner crashed: {e}")
        except Exception:
            pass
        sys.exit(1)

    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
