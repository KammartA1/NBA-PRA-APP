#!/usr/bin/env python3
"""
core/mlb_grader.py — Grades MLB scanner props from actual box scores.

Same pattern as core/results_grader.py (NBA). Pulls ungraded MLB scan_results,
fetches game logs, compares actual vs line, writes to bet_results.

Usage:
    python -m core.mlb_grader              # grade MLB props
    python -m core.mlb_grader --test-grade # dry run
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = None

from core import db, notify
from core import mlb_projections as projections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("mlb_grader")

TIME_BUDGET_SECS = 660
WIN_PROFIT = 100 / 110
LOSS_PROFIT = -1.0


def _today_et():
    now = datetime.now(timezone.utc)
    return now.astimezone(_ET).date() if _ET else now.date()


def _et_date(iso_utc: str):
    try:
        dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_ET).date() if _ET else dt.date()
    except Exception:
        return None


def run_grade(dry_run: bool = False) -> dict:
    start = time.time()
    today_et = _today_et()
    yesterday = today_et - timedelta(days=1)
    log.info("=== MLB GRADE START (today_ET=%s, dry_run=%s) ===", today_et, dry_run)

    ungraded = db.load_ungraded_results(sport="MLB", lookback_hours=72)
    log.info("Loaded %d ungraded MLB scan_results", len(ungraded))
    if not ungraded:
        db.log_worker_run("mlb_grader", "success", {"decided": 0, "note": "nothing to grade"})
        return {"ok": True, "decided": 0}

    # Dedup by (player, market, line, side) — grade the latest of each group
    seen: dict[tuple, dict] = {}
    all_ids = []
    for r in ungraded:
        key = (r.get("player", ""), r.get("market"), r.get("line"), str(r.get("side", "Over")))
        all_ids.append(r["id"])
        if key not in seen:
            seen[key] = r
        else:
            prev_dt = seen[key].get("scanned_at", "")
            curr_dt = r.get("scanned_at", "")
            if curr_dt > prev_dt:
                seen[key] = r

    log.info("Deduped to %d unique MLB props", len(seen))

    gamelog_cache: dict = {}
    graded_rows: list[dict] = []
    n_decided = 0
    n_skipped = 0

    for key, rep in seen.items():
        if time.time() - start > TIME_BUDGET_SECS:
            log.warning("Time budget exceeded — stopping early")
            break

        player = rep.get("player", "")
        market = rep.get("market", "")
        line = rep.get("line")
        side = str(rep.get("side", "Over"))
        scanned_date = _et_date(rep.get("scanned_at", ""))

        if not player or line is None or not market:
            n_skipped += 1
            continue

        if scanned_date and scanned_date >= today_et:
            n_skipped += 1
            continue

        game_date = scanned_date or yesterday

        player_id = projections.resolve_player_id(player)
        if not player_id:
            n_skipped += 1
            continue

        ptype = projections.get_player_type(player_id)
        if player_id not in gamelog_cache:
            gamelog_cache[player_id] = projections.fetch_player_gamelog(
                player_id, n_games=30, player_type=ptype
            )
        gl = gamelog_cache[player_id]
        if gl is None or gl.empty:
            n_skipped += 1
            continue

        actual = projections.actual_stat_for_date(gl, market, game_date)
        if actual is None:
            actual = projections.actual_stat_for_date(gl, market, game_date + timedelta(days=1))
        if actual is None:
            actual = projections.actual_stat_for_date(gl, market, game_date - timedelta(days=1))
        if actual is None:
            n_skipped += 1
            continue

        line_f = float(line)
        is_over = side.lower().startswith("o")

        if actual == line_f:
            result = "push"
            profit = 0.0
        elif (actual > line_f) == is_over:
            result = "win"
            profit = WIN_PROFIT
        else:
            result = "loss"
            profit = LOSS_PROFIT

        graded_rows.append({
            "sport": "MLB",
            "game_date": game_date.isoformat(),
            "player": player,
            "market": market,
            "line": line_f,
            "side": side,
            "proj": rep.get("proj"),
            "p_cal": rep.get("p_cal"),
            "ev_pct": rep.get("ev_adj_pct") or rep.get("ev_pct"),
            "edge_cat": rep.get("edge_cat", ""),
            "actual_value": actual,
            "result": result,
            "profit_units": round(profit, 4),
        })
        n_decided += 1

        log.info("  %s %s %s %.1f → actual=%.1f → %s",
                 player, market, side, line_f, actual, result)

    elapsed = time.time() - start
    log.info("=== MLB GRADE COMPLETE === %d decided, %d skipped, %.0fs", n_decided, n_skipped, elapsed)

    if not dry_run:
        if all_ids:
            db.mark_results_graded(all_ids)
        if graded_rows:
            db.upsert_bet_results(graded_rows)

    details = {"decided": n_decided, "skipped": n_skipped, "elapsed_seconds": round(elapsed, 1)}
    if not dry_run:
        db.log_worker_run("mlb_grader", "success", details)
        if graded_rows:
            _send_summary(graded_rows)

    return {"ok": True, **details}


def _send_summary(graded: list[dict]):
    if not graded:
        return
    try:
        wins = sum(1 for g in graded if g["result"] == "win")
        losses = sum(1 for g in graded if g["result"] == "loss")
        pushes = sum(1 for g in graded if g["result"] == "push")
        total_profit = sum(g["profit_units"] for g in graded)
        roi = (total_profit / len(graded) * 100) if graded else 0
        msg = (
            f"⚾ <b>MLB Results</b>\n"
            f"• Record: {wins}W-{losses}L" + (f" ({pushes} push)" if pushes else "") + "\n"
            f"• P/L: {total_profit:+.2f}u | ROI: {roi:+.1f}%"
        )
        notify.send_message(msg)
        db.log_notification("mlb_results", msg)
    except Exception as e:
        log.warning("MLB summary failed: %s", e)


def main():
    dry = "--test-grade" in sys.argv
    try:
        result = run_grade(dry_run=dry)
    except Exception as e:
        import traceback
        log.error("MLB grader crashed: %s\n%s", e, traceback.format_exc())
        try:
            db.log_worker_run("mlb_grader", "error", {"error": str(e)})
            notify.send_worker_status("CRASHED", f"MLB grader crashed: {e}")
        except Exception:
            pass
        sys.exit(1)
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
