"""
workers/closing_worker.py
=========================
Captures closing lines at event start time -- the most important data point
for CLV (Closing Line Value) tracking.

Every 5 minutes this worker:
  1. Checks the events table for games starting within the next 15 minutes
  2. For each starting event, fetches final lines from all books
  3. Marks those line_movements as ``is_closing=True``
  4. Updates any open bets with their closing_line value
  5. Calculates CLV for bets that now have closing lines

Run standalone:
    python -m workers.closing_worker          # one-shot
    python -m workers.closing_worker --loop   # 5-min loop
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from database.connection import session_scope, init_db
from database.models import Event, LineMovement, Bet
from workers.base import BaseWorker, standalone_main

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY_NBA = "basketball_nba"
REGION_US = "us"

# Window: events starting within the next 15 minutes
CLOSING_WINDOW_MINUTES = 15

# How far back to look for the "closing" line snapshot (the latest recorded
# line within this window before game start counts as "closing")
CLOSING_LOOKBACK_MINUTES = 60

ODDS_API_MARKET_KEYS = [
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_points_rebounds_assists",
    "player_points_rebounds", "player_points_assists",
    "player_rebounds_assists", "player_blocks", "player_steals",
    "player_turnovers",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _odds_api_key() -> str:
    return os.environ.get("ODDS_API_KEY", "")


# ---------------------------------------------------------------------------
# Fetch final lines from Odds API
# ---------------------------------------------------------------------------

def _fetch_closing_lines_from_api(event_id: str) -> List[Dict[str, Any]]:
    """Fetch the latest player prop lines for a specific event from The Odds API."""
    key = _odds_api_key()
    if not key:
        return []

    market_str = ",".join(ODDS_API_MARKET_KEYS)
    try:
        r = requests.get(
            f"{ODDS_BASE}/sports/{SPORT_KEY_NBA}/events/{event_id}/odds",
            params={
                "apiKey": key,
                "regions": REGION_US,
                "markets": market_str,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            timeout=20,
        )
        if not r.ok:
            log.warning("Odds API closing line fetch HTTP %d for event %s", r.status_code, event_id)
            return []

        data = r.json()
        rows = []
        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker.get("key", "unknown")
            for market_obj in bookmaker.get("markets", []):
                market_key = market_obj.get("key", "")
                for outcome in market_obj.get("outcomes", []):
                    desc = outcome.get("description", "")
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if desc and point is not None:
                        rows.append({
                            "player": desc,
                            "market": market_key,
                            "line": float(point),
                            "book": book_key,
                            "odds_decimal": float(price) if price else None,
                        })
        return rows
    except Exception as exc:
        log.warning("Failed to fetch closing lines for event %s: %s", event_id, exc)
        return []


# ---------------------------------------------------------------------------
# CLV computation
# ---------------------------------------------------------------------------

def compute_clv(bet_line: float, closing_line: float, direction: str) -> float:
    """Compute Closing Line Value in cents-per-dollar.

    Positive CLV = we got a better line than the market closed at.

    For OVER bets: CLV = closing_line - bet_line
       (If closing line moves UP after we bet OVER, we got value.)

    For UNDER bets: CLV = bet_line - closing_line
       (If closing line moves DOWN after we bet UNDER, we got value.)

    Normalised to the bet line magnitude so result is in "cents per dollar".
    """
    if bet_line == 0:
        return 0.0

    if direction.lower() in ("over", "o"):
        raw_clv = closing_line - bet_line
    else:
        raw_clv = bet_line - closing_line

    # Express as percentage of the line
    clv_pct = (raw_clv / abs(bet_line)) * 100.0
    return round(clv_pct, 3)


# ===================================================================
# Worker class
# ===================================================================

class ClosingWorker(BaseWorker):
    """Captures closing lines for events about to start and calculates CLV."""

    def __init__(self, **kwargs):
        super().__init__(
            name="closing_worker",
            interval_seconds=int(os.environ.get("CLOSING_INTERVAL", "300")),
            max_retries=2,
            retry_delay=10.0,
            **kwargs,
        )

    def execute(self) -> Dict[str, Any]:
        now = _utcnow()
        window_end = now + timedelta(minutes=CLOSING_WINDOW_MINUTES)
        lookback_start = now - timedelta(minutes=CLOSING_LOOKBACK_MINUTES)

        events_processed = 0
        lines_marked_closing = 0
        bets_updated = 0
        clv_computed = 0
        api_lines_fetched = 0

        with session_scope() as session:
            # 1. Find events starting within the next 15 minutes
            starting_events = (
                session.query(Event)
                .filter(
                    Event.sport == "NBA",
                    Event.status == "scheduled",
                    Event.start_time >= now,
                    Event.start_time <= window_end,
                )
                .all()
            )

            if not starting_events:
                self.logger.info(
                    "No events starting in the next %d minutes",
                    CLOSING_WINDOW_MINUTES,
                )
                return {"ok": True, "events_processed": 0, "reason": "no_starting_events"}

            self.logger.info(
                "Found %d events starting within %d minutes",
                len(starting_events),
                CLOSING_WINDOW_MINUTES,
            )

            for ev in starting_events:
                events_processed += 1
                ev_name = ev.event_name

                # 2. Try fetching fresh closing lines from The Odds API
                ev_meta = {}
                try:
                    ev_meta = json.loads(ev.metadata_json or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass

                api_event_id = ev_meta.get("event_id", "")
                if api_event_id:
                    api_closing_rows = _fetch_closing_lines_from_api(api_event_id)
                    if api_closing_rows:
                        api_lines_fetched += len(api_closing_rows)
                        # Store them as closing line movements
                        for row in api_closing_rows:
                            lm = LineMovement(
                                sport="NBA",
                                event=ev_name,
                                market=row.get("market", ""),
                                book=row.get("book", ""),
                                player=row.get("player", ""),
                                line=row["line"],
                                odds=None,
                                timestamp=now,
                                is_opening=False,
                                is_closing=True,
                            )
                            session.add(lm)
                            lines_marked_closing += 1

                # 3. Mark the most recent existing lines as closing
                #    For each unique (player, market, book) combination,
                #    find the latest line_movement within the lookback window
                recent_lines = (
                    session.query(LineMovement)
                    .filter(
                        LineMovement.event == ev_name,
                        LineMovement.timestamp >= lookback_start,
                        LineMovement.is_closing == False,
                    )
                    .order_by(LineMovement.timestamp.desc())
                    .all()
                )

                # Deduplicate: keep only the latest per (player, market, book)
                seen_keys = set()
                for lm in recent_lines:
                    key = (lm.player, lm.market, lm.book)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        lm.is_closing = True
                        lines_marked_closing += 1

                # 4. Update open bets with closing lines
                #    Match bets by player + market for this event
                open_bets = (
                    session.query(Bet)
                    .filter(
                        Bet.sport == "NBA",
                        Bet.status == "pending",
                        Bet.event == ev_name,
                        Bet.closing_line == None,
                    )
                    .all()
                )

                for bet in open_bets:
                    # Find the closing line for this bet's player + market
                    closing_lm = (
                        session.query(LineMovement)
                        .filter(
                            LineMovement.event == ev_name,
                            LineMovement.player == bet.player,
                            LineMovement.market == bet.market,
                            LineMovement.is_closing == True,
                        )
                        .order_by(LineMovement.timestamp.desc())
                        .first()
                    )

                    if closing_lm is not None:
                        bet.closing_line = closing_lm.line
                        bets_updated += 1

                        # 5. Calculate CLV
                        clv = compute_clv(
                            bet_line=bet.bet_line,
                            closing_line=closing_lm.line,
                            direction=bet.direction or "over",
                        )

                        # Store CLV in notes or features_snapshot
                        try:
                            features = json.loads(bet.features_snapshot_json or "{}")
                        except (json.JSONDecodeError, TypeError):
                            features = {}
                        features["closing_line"] = closing_lm.line
                        features["clv_cents"] = clv
                        features["clv_computed_at"] = now.isoformat()
                        features["closing_book"] = closing_lm.book
                        bet.features_snapshot_json = json.dumps(features, default=str)
                        clv_computed += 1

                        self.logger.info(
                            "CLV for %s %s %s: bet=%.1f close=%.1f CLV=%.2f%%",
                            bet.player, bet.market, bet.direction,
                            bet.bet_line, closing_lm.line, clv,
                        )

                # Mark event as in-progress
                ev.status = "in_progress"

        self.logger.info(
            "Processed %d events: %d lines marked closing, %d bets updated, %d CLV computed",
            events_processed, lines_marked_closing, bets_updated, clv_computed,
        )

        return {
            "ok": True,
            "events_processed": events_processed,
            "lines_marked_closing": lines_marked_closing,
            "api_lines_fetched": api_lines_fetched,
            "bets_updated": bets_updated,
            "clv_computed": clv_computed,
        }


# ===================================================================
# Standalone entry point
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    standalone_main(ClosingWorker)
