"""
services/clv_system/closing_capture.py
========================================
Automatic closing line capture at event start.

"Closing line" = last line available before an event starts.
  - NBA: capture lines at tip-off time
  - Golf: capture lines at first tee time of the round

This module monitors event start times and automatically captures
closing lines, then matches them to bet records for CLV calculation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func as sa_func, and_

from quant_system.db.schema import get_engine, get_session, BetLog, CLVLog
from services.clv_system.models import (
    CLVLineMovement,
    CLVClosingLine,
    Base,
)

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClosingLineCapture:
    """Automatic closing line capture and bet-matching service.

    Monitors event start times and captures the last available line
    before each event begins.  Then matches closing lines to bet records
    so CLV can be calculated.
    """

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    # ── Capture closing lines for an event ───────────────────────────

    def capture_closing_lines(
        self,
        event_id: str,
        event_start_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Capture closing lines for all player/market combos in an event.

        Takes the last recorded line before the event_start_time for each
        player/market/book combination and stores them as closing lines.

        Args:
            event_id: Event identifier (game name, tournament round, etc.)
            event_start_time: When the event starts. Defaults to now.

        Returns:
            Summary dict with counts.
        """
        if event_start_time is None:
            event_start_time = _utcnow()

        session = self._session()
        captured = 0

        try:
            # Find the last line for each player/market/book before event start
            subq = (
                session.query(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                    sa_func.max(CLVLineMovement.timestamp).label("max_ts"),
                )
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.event_id == event_id,
                    CLVLineMovement.timestamp <= event_start_time,
                )
                .group_by(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                )
                .subquery()
            )

            closing_rows = (
                session.query(CLVLineMovement)
                .join(
                    subq,
                    and_(
                        CLVLineMovement.player == subq.c.player,
                        CLVLineMovement.market_type == subq.c.market_type,
                        CLVLineMovement.book == subq.c.book,
                        CLVLineMovement.timestamp == subq.c.max_ts,
                    ),
                )
                .filter(CLVLineMovement.event_id == event_id)
                .all()
            )

            # Also mark them as closing in the line_movements table
            for row in closing_rows:
                row.is_closing = True

                # Check if we already have this closing line
                existing = (
                    session.query(CLVClosingLine)
                    .filter(
                        CLVClosingLine.event_id == event_id,
                        CLVClosingLine.player == row.player,
                        CLVClosingLine.market_type == row.market_type,
                        CLVClosingLine.book == row.book,
                    )
                    .first()
                )
                if existing:
                    existing.closing_line = row.line
                    existing.closing_odds_american = row.odds_american
                    existing.closing_odds_decimal = row.odds_decimal
                    existing.closing_implied_prob = row.implied_prob
                    existing.captured_at = _utcnow()
                else:
                    cl = CLVClosingLine(
                        sport=self.sport,
                        event_id=event_id,
                        player=row.player,
                        market_type=row.market_type,
                        book=row.book,
                        closing_line=row.line,
                        closing_odds_american=row.odds_american,
                        closing_odds_decimal=row.odds_decimal,
                        closing_implied_prob=row.implied_prob,
                        event_start_time=event_start_time,
                        captured_at=_utcnow(),
                        is_consensus=False,
                    )
                    session.add(cl)
                captured += 1

            # Also create consensus closing lines
            self._create_consensus_closings(session, event_id, event_start_time)

            session.commit()
            log.info(
                "Captured %d closing lines for event %s",
                captured, event_id,
            )

            return {
                "event_id": event_id,
                "closing_lines_captured": captured,
                "event_start_time": event_start_time.isoformat(),
            }

        except Exception:
            session.rollback()
            log.exception("Failed to capture closing lines for %s", event_id)
            raise
        finally:
            session.close()

    def _create_consensus_closings(
        self,
        session,
        event_id: str,
        event_start_time: datetime,
    ) -> None:
        """Create consensus closing lines (median across books) for each player/market."""
        closings = (
            session.query(CLVClosingLine)
            .filter(
                CLVClosingLine.event_id == event_id,
                CLVClosingLine.sport == self.sport,
                CLVClosingLine.is_consensus == False,  # noqa: E712
            )
            .all()
        )

        # Group by player/market
        groups: Dict[tuple, List[float]] = {}
        for cl in closings:
            key = (cl.player, cl.market_type)
            groups.setdefault(key, []).append(cl.closing_line)

        for (player, market_type), lines in groups.items():
            if not lines:
                continue
            sorted_lines = sorted(lines)
            n = len(sorted_lines)
            if n % 2 == 1:
                consensus = sorted_lines[n // 2]
            else:
                consensus = (sorted_lines[n // 2 - 1] + sorted_lines[n // 2]) / 2.0

            # Check if consensus already exists
            existing = (
                session.query(CLVClosingLine)
                .filter(
                    CLVClosingLine.event_id == event_id,
                    CLVClosingLine.player == player,
                    CLVClosingLine.market_type == market_type,
                    CLVClosingLine.is_consensus == True,  # noqa: E712
                )
                .first()
            )
            if existing:
                existing.closing_line = consensus
                existing.captured_at = _utcnow()
            else:
                session.add(CLVClosingLine(
                    sport=self.sport,
                    event_id=event_id,
                    player=player,
                    market_type=market_type,
                    book="consensus",
                    closing_line=consensus,
                    event_start_time=event_start_time,
                    captured_at=_utcnow(),
                    is_consensus=True,
                ))

    # ── Match closing lines to bets ──────────────────────────────────

    def match_closing_to_bets(self) -> Dict[str, Any]:
        """Match closing lines to pending/settled bets that are missing them.

        Finds bets without closing_line set and attempts to match them
        to captured closing lines based on player, market, and event.

        Returns:
            Summary of matched bets.
        """
        session = self._session()
        matched = 0
        try:
            # Find bets missing closing lines
            bets = (
                session.query(BetLog)
                .filter(
                    BetLog.sport == self.sport.lower(),
                    BetLog.closing_line == None,  # noqa: E711
                    BetLog.status.in_(["pending", "won", "lost", "push"]),
                )
                .all()
            )

            for bet in bets:
                # Find consensus closing line for this bet's player/market
                closing = (
                    session.query(CLVClosingLine)
                    .filter(
                        CLVClosingLine.sport == self.sport,
                        CLVClosingLine.player == bet.player,
                        CLVClosingLine.market_type == bet.stat_type,
                        CLVClosingLine.is_consensus == True,  # noqa: E712
                    )
                    .order_by(CLVClosingLine.captured_at.desc())
                    .first()
                )

                if closing:
                    bet.closing_line = closing.closing_line
                    bet.closing_odds = closing.closing_odds_american
                    matched += 1
                    log.debug(
                        "Matched closing line to bet %s: %.1f",
                        bet.bet_id, closing.closing_line,
                    )

            session.commit()
            log.info("Matched %d closing lines to bets", matched)

            return {"bets_matched": matched}

        except Exception:
            session.rollback()
            log.exception("Failed to match closing lines to bets")
            raise
        finally:
            session.close()

    # ── Auto-capture for NBA (tip-off based) ─────────────────────────

    def auto_capture_nba(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Automatically capture closing lines for NBA events at tip-off.

        Checks each event's start_time against current time.  If an event
        has started (or is about to start within 2 minutes), capture
        closing lines.

        Args:
            events: List of event dicts with 'event_id', 'event_name',
                    'start_time' (ISO string or datetime).

        Returns:
            Summary of events processed.
        """
        now = _utcnow()
        capture_window = timedelta(minutes=2)
        processed = 0
        results = []

        for ev in events:
            event_id = ev.get("event_id") or ev.get("event_name", "")
            start_time = ev.get("start_time")

            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(
                        start_time.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    continue

            if start_time is None:
                continue

            # Make timezone-aware if needed
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)

            # Check if event is starting within the capture window
            time_to_start = start_time - now
            if timedelta(0) <= time_to_start <= capture_window:
                result = self.capture_closing_lines(event_id, start_time)
                results.append(result)
                processed += 1
            elif time_to_start < timedelta(0) and time_to_start > -capture_window:
                # Just started — still capture
                result = self.capture_closing_lines(event_id, start_time)
                results.append(result)
                processed += 1

        return {
            "events_processed": processed,
            "results": results,
        }

    # ── Auto-capture for Golf (first tee time based) ─────────────────

    def auto_capture_golf(
        self,
        event_id: str,
        first_tee_time: datetime,
    ) -> Dict[str, Any]:
        """Capture closing lines for a golf round at first tee time.

        Args:
            event_id: Tournament/round identifier.
            first_tee_time: Time of first tee-off.

        Returns:
            Capture result.
        """
        return self.capture_closing_lines(event_id, first_tee_time)

    # ── Query closing lines ──────────────────────────────────────────

    def get_closing_line(
        self,
        player: str,
        market_type: str,
        event_id: Optional[str] = None,
        consensus_only: bool = True,
    ) -> Optional[float]:
        """Get the closing line for a player/market.

        Args:
            player: Player name.
            market_type: Market type.
            event_id: Event filter (optional).
            consensus_only: If True, only return consensus closing line.

        Returns:
            Closing line value or None.
        """
        session = self._session()
        try:
            q = (
                session.query(CLVClosingLine)
                .filter(
                    CLVClosingLine.sport == self.sport,
                    CLVClosingLine.player == player,
                    CLVClosingLine.market_type == market_type,
                )
            )
            if event_id:
                q = q.filter(CLVClosingLine.event_id == event_id)
            if consensus_only:
                q = q.filter(CLVClosingLine.is_consensus == True)  # noqa: E712

            row = q.order_by(CLVClosingLine.captured_at.desc()).first()
            return row.closing_line if row else None
        finally:
            session.close()

    def get_all_closing_lines(
        self, event_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all closing lines for an event."""
        session = self._session()
        try:
            rows = (
                session.query(CLVClosingLine)
                .filter(
                    CLVClosingLine.sport == self.sport,
                    CLVClosingLine.event_id == event_id,
                )
                .order_by(CLVClosingLine.player, CLVClosingLine.market_type)
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def closing_line_count(self) -> int:
        """Total number of closing line records."""
        session = self._session()
        try:
            return (
                session.query(sa_func.count(CLVClosingLine.id))
                .filter(CLVClosingLine.sport == self.sport)
                .scalar()
            ) or 0
        finally:
            session.close()
