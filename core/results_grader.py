#!/usr/bin/env python3
"""
core/results_grader.py — Background results grader that runs in GitHub Actions.

Pulls ungraded scan_results from Supabase → fetches each player's actual box
score via nba_api → grades W/L/push/void against the line+side → computes ROI
→ writes bet_results → sends a Telegram daily summary.

Designed to run each morning (~10am ET) after all games have settled.

Usage:
    python -m core.results_grader              # grade everything pending
    python -m core.results_grader --test-grade # dry run, no DB writes
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

from core import db, notify, projections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("results_grader")

# Flat 1-unit stake at standard -110 sportsbook pricing.
# (Win rate vs PrizePicks break-even is shown in the UI; this gives a clean
#  apples-to-apples ROI baseline.)
WIN_PROFIT = 100.0 / 110.0   # +0.909 units on a win
LOSS_PROFIT = -1.0           # -1.000 units on a loss

TIME_BUDGET_SECS = 660       # 11 min — leaves headroom inside the 15-min job
MAX_AHEAD_DAYS = 2           # a prop's game must be within N days of the scan date


def _et_date(iso_utc: str):
    """ET calendar date for a UTC ISO timestamp string."""
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_ET).date() if _ET else dt.date()
    except Exception:
        return None


def _today_et():
    now = datetime.now(timezone.utc)
    return now.astimezone(_ET).date() if _ET else now.date()


def _grade_side(side: str, actual: float, line: float) -> tuple[str, bool | None, float]:
    """Return (result, hit, profit_units) for a settled prop."""
    is_over = str(side or "Over").strip().lower().startswith("o")
    if actual == line:
        return "push", None, 0.0
    won = (actual > line) if is_over else (actual < line)
    if won:
        return "win", True, WIN_PROFIT
    return "loss", False, LOSS_PROFIT


def _resolve_game(gamelog_df, scan_et_date, today_et):
    """Find the date of the player's game this prop refers to.

    Props are listed for games that start AFTER the scan, so the game is the
    player's first game on-or-after the scan date. Returns one of:
        ("graded", game_date)  — game played, found in the log
        ("skip", None)         — game hasn't been played yet (don't grade)
        ("void", scan_et_date) — game is in the past but player didn't play (DNP)
    """
    import pandas as pd
    if gamelog_df is None or gamelog_df.empty or "GAME_DATE" not in gamelog_df.columns:
        # No log at all — if the slate is in the past, it's a void; else skip.
        return ("void", scan_et_date) if scan_et_date < today_et else ("skip", None)

    dates = pd.to_datetime(
        gamelog_df["GAME_DATE"], format="%b %d, %Y", errors="coerce"
    ).dt.date.dropna()
    future = sorted(d for d in dates if d >= scan_et_date)
    if future:
        game_d = future[0]
        if (game_d - scan_et_date).days <= MAX_AHEAD_DAYS:
            return ("graded", game_d)
    # No qualifying game in the log
    if scan_et_date >= today_et:
        return ("skip", None)
    return ("void", scan_et_date)


def run_grade(dry_run: bool = False) -> dict:
    grade_id = str(uuid.uuid4())[:8]
    start = time.time()
    today_et = _today_et()
    log.info("=== GRADE START (id=%s, today_ET=%s, dry_run=%s) ===", grade_id, today_et, dry_run)

    rows = db.load_ungraded_results("NBA", lookback_hours=48)
    log.info("Loaded %d ungraded scan_results rows", len(rows))
    if not rows:
        db.log_worker_run("grader", "success", {"grade_id": grade_id, "graded": 0, "note": "nothing pending"})
        return {"ok": True, "graded": 0, "skipped": 0, "void": 0}

    # Group by logical prop (player, market, line, side); keep latest scan + all ids.
    groups: dict[tuple, dict] = {}
    for r in rows:
        if not r.get("market"):
            continue
        key = (r.get("player", ""), r.get("market"), r.get("line"), str(r.get("side", "Over")))
        g = groups.get(key)
        scanned_at = r.get("scanned_at", "")
        if g is None:
            groups[key] = {"latest": r, "ids": [r["id"]], "latest_ts": scanned_at}
        else:
            g["ids"].append(r["id"])
            if scanned_at > g["latest_ts"]:
                g["latest"] = r
                g["latest_ts"] = scanned_at
    log.info("Deduped to %d unique props", len(groups))

    # Cache one gamelog per player across all their props.
    gamelog_cache: dict[str, object] = {}

    graded_rows = []
    ids_to_mark: list[int] = []
    n_win = n_loss = n_push = n_void = n_skip = 0

    for key, g in groups.items():
        if time.time() - start > TIME_BUDGET_SECS:
            log.warning("Time budget exceeded — stopping early (processed before timeout)")
            break

        rep = g["latest"]
        player = rep.get("player", "")
        market = rep.get("market")
        line = rep.get("line")
        side = rep.get("side", "Over")
        scan_et = _et_date(rep.get("scanned_at", ""))
        if line is None or scan_et is None or not player:
            continue

        # Fetch (and cache) the player's gamelog.
        if player not in gamelog_cache:
            pid = projections.resolve_player_id(player)
            if pid:
                gamelog_cache[player] = projections.fetch_player_gamelog(pid, n_games=30)
            else:
                gamelog_cache[player] = None
        gl = gamelog_cache[player]

        status, game_d = _resolve_game(gl, scan_et, today_et)
        if status == "skip":
            n_skip += 1
            continue  # leave ungraded; a later run will pick it up

        if status == "void":
            result, hit, profit, actual = "void", None, 0.0, None
        else:  # graded
            actual = projections.actual_stat_for_date(gl, market, game_d)
            if actual is None:
                result, hit, profit = "void", None, 0.0
            else:
                result, hit, profit = _grade_side(side, actual, line)

        if result == "win":
            n_win += 1
        elif result == "loss":
            n_loss += 1
        elif result == "push":
            n_push += 1
        else:
            n_void += 1

        graded_rows.append({
            "scan_result_id": rep.get("id"),
            "scan_id": rep.get("scan_id", ""),
            "sport": "NBA",
            "graded_at": datetime.now(timezone.utc).isoformat(),
            "game_date": game_d.isoformat() if game_d else None,
            "player": player,
            "team": rep.get("team", ""),
            "opp": rep.get("opp", ""),
            "market": market,
            "line": line,
            "side": side,
            "proj": rep.get("proj"),
            "p_cal": rep.get("p_cal"),
            "ev_pct": rep.get("ev_pct"),
            "edge_cat": rep.get("edge_cat", ""),
            "actual_value": actual,
            "result": result,
            "hit": hit,
            "profit_units": round(profit, 4),
            "src": rep.get("src", "PrizePicks"),
        })
        ids_to_mark.extend(g["ids"])

    elapsed = time.time() - start
    decided = n_win + n_loss + n_push + n_void
    log.info(
        "=== GRADE COMPLETE (id=%s) === %d decided (%dW-%dL, %d push, %d void), %d skipped, %.0fs",
        grade_id, decided, n_win, n_loss, n_push, n_void, n_skip, elapsed,
    )

    if dry_run:
        for gr in graded_rows[:15]:
            log.info("  %s %s %s %s | actual=%s → %s",
                     gr["player"], gr["market"], gr["side"], gr["line"],
                     gr["actual_value"], gr["result"].upper())
        return {"ok": True, "graded": decided, "skipped": n_skip,
                "win": n_win, "loss": n_loss, "push": n_push, "void": n_void,
                "dry_run": True}

    # Persist
    written = db.upsert_bet_results(graded_rows) if graded_rows else 0
    marked = db.mark_results_graded(ids_to_mark) if ids_to_mark else 0
    log.info("Wrote %d bet_results, marked %d scan_results graded", written, marked)

    # ROI on decided W/L bets
    decisive = n_win + n_loss
    units = n_win * WIN_PROFIT + n_loss * LOSS_PROFIT
    roi = (units / decisive * 100) if decisive else 0.0
    win_rate = (n_win / decisive * 100) if decisive else 0.0

    details = {
        "grade_id": grade_id,
        "graded": decided, "win": n_win, "loss": n_loss,
        "push": n_push, "void": n_void, "skipped": n_skip,
        "win_rate": round(win_rate, 1), "units": round(units, 2),
        "roi_pct": round(roi, 1), "elapsed_seconds": round(elapsed, 1),
    }
    db.log_worker_run("grader", "success", details)

    _send_daily_summary(n_win, n_loss, n_push, n_void, win_rate, units, roi, graded_rows)

    return {"ok": True, **details}


def _send_daily_summary(n_win, n_loss, n_push, n_void, win_rate, units, roi, graded_rows):
    """One Telegram message with yesterday's record + best/worst."""
    decisive = n_win + n_loss
    if decisive == 0 and n_void == 0:
        return
    try:
        emoji = "📈" if roi >= 0 else "📉"
        lines = [
            f"{emoji} <b>Results graded</b>",
            f"• Record: {n_win}-{n_loss}" + (f" ({win_rate:.0f}%)" if decisive else ""),
            f"• Units: {units:+.2f}u  |  ROI: {roi:+.1f}% (flat -110)",
        ]
        if n_push or n_void:
            lines.append(f"• Push: {n_push}  |  Void/DNP: {n_void}")
        wins = [g for g in graded_rows if g["result"] == "win"]
        losses = [g for g in graded_rows if g["result"] == "loss"]
        if wins:
            b = max(wins, key=lambda g: g.get("ev_pct") or 0)
            lines.append(f"• ✅ Best: {b['player']} {b['market']} {b['side']} {b['line']} "
                         f"(hit {b['actual_value']})")
        if losses:
            w = max(losses, key=lambda g: g.get("ev_pct") or 0)
            lines.append(f"• ❌ Worst miss: {w['player']} {w['market']} {w['side']} {w['line']} "
                         f"(got {w['actual_value']})")
        notify.send_message("\n".join(lines))
        db.log_notification("results_summary", "\n".join(lines))
    except Exception as e:
        log.warning("daily results summary failed: %s", e)


def main():
    dry = "--test-grade" in sys.argv
    try:
        result = run_grade(dry_run=dry)
    except Exception as e:
        import traceback
        log.error("Grader crashed: %s\n%s", e, traceback.format_exc())
        try:
            db.log_worker_run("grader", "error", {"error": str(e)})
            notify.send_worker_status("CRASHED", f"Results grader crashed: {e}")
        except Exception:
            pass
        sys.exit(1)

    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
