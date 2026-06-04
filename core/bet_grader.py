#!/usr/bin/env python3
"""
core/bet_grader.py — Grades the user's LOGGED bets (not just scanner props).

Shared grading logic used by BOTH:
  - the Streamlit app (when you open the HISTORY tab), and
  - the background worker (each morning via GitHub Actions, even while the
    app is asleep).

A logged bet lives in the Supabase `logged_bets` table. Each bet has N legs;
each leg is graded HIT/MISS/PUSH from the player's actual box score, then the
parlay-level result is derived.

Usage:
    python -m core.bet_grader              # grade pending logged bets
    python -m core.bet_grader --test-grade # dry run, no DB writes
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

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
log = logging.getLogger("bet_grader")

TIME_BUDGET_SECS = 600

# Map every market display/code variant to an nba_api stat code.
MARKET_STAT_MAP = {
    "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
    "PRA": "PRA", "Pts+Rebs+Asts": "PRA", "PA": "PA", "Pts+Asts": "PA",
    "PR": "PR", "Pts+Rebs": "PR", "RA": "RA", "Rebs+Asts": "RA",
    "3PM": "FG3M", "3-Pt Made": "FG3M", "Steals": "STL", "Blocks": "BLK",
    "Blocked Shots": "BLK", "Turnovers": "TOV", "Fantasy Score": "FS",
    "FG Attempted": "FGA", "FT Made": "FTM", "Blks+Stls": "BLST",
    "PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "FG3M",
    "STL": "STL", "BLK": "BLK", "TOV": "TOV", "FS": "FS", "FGA": "FGA",
    "FTM": "FTM", "BLST": "BLST",
}


def _stat_code(market: str) -> str:
    if not market:
        return ""
    return MARKET_STAT_MAP.get(market) or MARKET_STAT_MAP.get(market.replace(" ", ""), "")


def _candidate_dates(game_date_et):
    """Dates to search for a player's box score.

    The bet's game_date is inferred from when it was logged, which is fragile:
    a bet logged just after midnight ET (e.g. 1:26am) for the prior evening's
    game gets stamped with the next calendar day. So we search the inferred
    date, the day before (post-midnight logs), and the day after (late games
    that settle on the next ET date) and use whichever has a real box score.
    """
    return [
        game_date_et,
        game_date_et - timedelta(days=1),
        game_date_et + timedelta(days=1),
    ]


def _player_actual(player: str, player_id, stat: str, game_date_et, gamelog_cache: dict):
    """Resolve one player's actual stat near `game_date_et`, or None."""
    if not player:
        return None
    if not player_id:
        player_id = projections.resolve_player_id(player)
    else:
        try:
            player_id = int(player_id)
        except (TypeError, ValueError):
            player_id = projections.resolve_player_id(player)
    if not player_id:
        return None

    if player_id not in gamelog_cache:
        gamelog_cache[player_id] = projections.fetch_player_gamelog(player_id, n_games=30)
    gl = gamelog_cache[player_id]
    if gl is None or gl.empty:
        return None

    for d in _candidate_dates(game_date_et):
        actual = projections.actual_stat_for_date(gl, stat, d)
        if actual is not None:
            return actual
    return None


def _split_combo(player: str) -> list[str]:
    """Split a combo prop name ('A + B') into individual player names."""
    if not player:
        return []
    parts = [p.strip() for p in str(player).split("+")]
    return [p for p in parts if p]


def grade_legs(legs: list[dict], game_date_et, today_et, gamelog_cache: dict | None = None):
    """Grade each leg of a bet. Returns (leg_results, all_resolved).

    leg_results is a list parallel to `legs` with values HIT/MISS/PUSH/VOID/Pending.
    VOID means the box score was never found and the game is >2 days old (the data
    will never arrive). VOID legs are treated like PUSH in parlay roll-up.
    `gamelog_cache` (keyed by player_id) is reused across bets for efficiency.
    Handles combo props ('Player A + Player B') by summing each player's actual.
    """
    if gamelog_cache is None:
        gamelog_cache = {}

    leg_results = []
    all_resolved = True

    for leg in legs:
        if not isinstance(leg, dict):
            leg_results.append("Pending")
            all_resolved = False
            continue

        player = leg.get("player", "")
        player_id = leg.get("player_id")
        market = leg.get("market", "")
        line = leg.get("line")
        side = str(leg.get("side", "over")).strip().lower()
        stat = _stat_code(market)

        if not player or line is None or not stat:
            leg_results.append("Pending")
            all_resolved = False
            continue

        sub_players = _split_combo(player)
        is_combo = len(sub_players) > 1

        actual = 0.0
        resolved = True
        for sp in sub_players:
            # A stored player_id only applies to a single-player leg.
            pid = None if is_combo else player_id
            val = _player_actual(sp, pid, stat, game_date_et, gamelog_cache)
            if val is None:
                resolved = False
                break
            actual += val

        if not resolved:
            # No box score found. If the game date is more than 2 days in the
            # past the data will never arrive — mark the leg VOID so it doesn't
            # stay Pending forever. VOID legs are treated like PUSH (they don't
            # count for or against the parlay).
            if game_date_et and today_et and (today_et - game_date_et).days > 2:
                leg_results.append("VOID")
            else:
                leg_results.append("Pending")
                all_resolved = False
            continue

        line_f = float(line)
        is_over = side.startswith("o")
        if actual == line_f:
            leg_results.append("PUSH")
        elif (actual > line_f) == is_over:
            leg_results.append("HIT")
        else:
            leg_results.append("MISS")

    return leg_results, all_resolved


def derive_parlay_result(leg_results: list[str], current: str = "Pending") -> str:
    """Roll up per-leg results into a parlay-level result.

    VOID legs are treated like PUSH — they don't count for or against.
    """
    decided = [r for r in leg_results if r in ("HIT", "MISS", "PUSH", "VOID")]
    if not decided:
        return current
    if any(r == "MISS" for r in leg_results):
        return "MISS"
    if all(r == "HIT" for r in leg_results):
        return "HIT"
    if all(r in ("HIT", "PUSH", "VOID") for r in leg_results):
        # At least one real leg must have resolved to HIT for this to be a win;
        # if every leg is PUSH/VOID, the whole bet is PUSH.
        if any(r == "HIT" for r in leg_results):
            return "HIT"
        return "PUSH"
    # Some legs still pending and none missed yet → still live
    return current


def _et_date(iso_utc: str):
    try:
        dt = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_ET).date() if _ET else dt.date()
    except Exception:
        return None


def _today_et():
    now = datetime.now(timezone.utc)
    return now.astimezone(_ET).date() if _ET else now.date()


def run_grade_logged_bets(dry_run: bool = False) -> dict:
    start = time.time()
    today_et = _today_et()
    log.info("=== BET GRADE START (today_ET=%s, dry_run=%s) ===", today_et, dry_run)

    pending = db.load_pending_logged_bets()
    log.info("Loaded %d pending logged bets", len(pending))
    if not pending:
        db.log_worker_run("bet_grader", "success", {"graded": 0, "note": "nothing pending"})
        return {"ok": True, "graded": 0}

    gamelog_cache: dict = {}
    n_graded = 0
    n_skipped = 0
    fully = 0
    summaries = []

    for bet in pending:
        if time.time() - start > TIME_BUDGET_SECS:
            log.warning("Time budget exceeded — stopping early")
            break

        legs = bet.get("legs") or []
        if isinstance(legs, str):
            import json
            try:
                legs = json.loads(legs)
            except Exception:
                legs = []
        if not legs:
            n_skipped += 1
            continue

        game_date_et = None
        if bet.get("game_date"):
            try:
                game_date_et = datetime.fromisoformat(str(bet["game_date"])).date()
            except Exception:
                game_date_et = None
        if game_date_et is None:
            game_date_et = _et_date(bet.get("logged_at", ""))
        if game_date_et is None:
            n_skipped += 1
            continue

        # Don't attempt clearly-future slates. Today's games ARE graded — if the
        # box score isn't final yet, grade_legs returns Pending and we retry next
        # run (self-correcting), rather than waiting a full extra day.
        if game_date_et > today_et:
            n_skipped += 1
            continue

        leg_results, all_resolved = grade_legs(legs, game_date_et, today_et, gamelog_cache)
        if not any(r in ("HIT", "MISS", "PUSH") for r in leg_results):
            n_skipped += 1
            continue

        new_result = derive_parlay_result(leg_results, bet.get("result", "Pending"))
        is_final = all(r in ("HIT", "MISS", "PUSH", "VOID") for r in leg_results)

        if is_final:
            fully += 1
        n_graded += 1
        hits = sum(1 for r in leg_results if r == "HIT")
        misses = sum(1 for r in leg_results if r == "MISS")
        summaries.append({
            "result": new_result, "hits": hits, "misses": misses,
            "n_legs": len(legs), "final": is_final,
        })

        if not dry_run:
            db.update_logged_bet(
                bet_id=bet["bet_id"],
                leg_results=leg_results,
                result=new_result,
                graded=is_final,
            )

        log.info("Bet %s: %s (%dH/%dM, %s)", bet.get("bet_id", "?")[:8],
                 new_result, hits, misses, "final" if is_final else "partial")

    elapsed = time.time() - start
    log.info("=== BET GRADE COMPLETE === %d graded (%d final), %d skipped, %.0fs",
             n_graded, fully, n_skipped, elapsed)

    details = {"graded": n_graded, "final": fully, "skipped": n_skipped,
               "elapsed_seconds": round(elapsed, 1)}
    if not dry_run:
        db.log_worker_run("bet_grader", "success", details)
        _send_summary(summaries)

    return {"ok": True, **details}


def _send_summary(summaries: list[dict]):
    finals = [s for s in summaries if s["final"]]
    if not finals:
        return
    try:
        wins = sum(1 for s in finals if s["result"] == "HIT")
        losses = sum(1 for s in finals if s["result"] == "MISS")
        pushes = sum(1 for s in finals if s["result"] == "PUSH")
        leg_hits = sum(s["hits"] for s in finals)
        leg_total = sum(s["hits"] + s["misses"] for s in finals)
        leg_rate = (leg_hits / leg_total * 100) if leg_total else 0
        msg = (
            f"🎯 <b>Your bets graded</b>\n"
            f"• Parlays: {wins}W-{losses}L" + (f" ({pushes} push)" if pushes else "") + "\n"
            f"• Legs: {leg_hits}/{leg_total} hit ({leg_rate:.0f}%)"
        )
        notify.send_message(msg)
        db.log_notification("bet_results", msg)
    except Exception as e:
        log.warning("bet summary failed: %s", e)


def main():
    dry = "--test-grade" in sys.argv
    try:
        result = run_grade_logged_bets(dry_run=dry)
    except Exception as e:
        import traceback
        log.error("Bet grader crashed: %s\n%s", e, traceback.format_exc())
        try:
            db.log_worker_run("bet_grader", "error", {"error": str(e)})
            notify.send_worker_status("CRASHED", f"Bet grader crashed: {e}")
        except Exception:
            pass
        sys.exit(1)
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
