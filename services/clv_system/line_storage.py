"""
services/clv_system/line_storage.py
====================================
Time-series storage manager for all line movements.

Responsibilities:
  - Query interface for line movement data
  - Retention policy: keep 90 days detailed, aggregate older data hourly
  - Opening line detection (first observed line for a player/market/event)
  - Statistics computation (volatility, movement patterns)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func as sa_func, and_

from quant_system.db.schema import get_engine, get_session
from services.clv_system.models import (
    CLVLineMovement,
    CLVLineMovementArchive,
    Base,
)

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LineStorage:
    """Time-series storage and query engine for line movements.

    Manages the clv_line_movements table with automatic retention policy
    and provides rich query capabilities for CLV analysis.
    """

    RETENTION_DAYS = 90

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    # ── Query: Line history for a player/market ──────────────────────

    def get_line_history(
        self,
        player: str,
        market_type: str,
        hours: int = 48,
        book: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get time-series of line movements for a player/market.

        Args:
            player: Player name.
            market_type: Market type (e.g., "Points", "PRA").
            hours: How many hours back to look.
            book: Optional book filter.

        Returns:
            List of dicts sorted by timestamp ascending.
        """
        cutoff = _utcnow() - timedelta(hours=hours)
        session = self._session()
        try:
            q = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                    CLVLineMovement.timestamp >= cutoff,
                )
            )
            if book:
                q = q.filter(CLVLineMovement.book == book)

            rows = q.order_by(CLVLineMovement.timestamp.asc()).all()
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    # ── Query: Opening line ──────────────────────────────────────────

    def get_opening_line(
        self,
        player: str,
        market_type: str,
        event_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the first recorded line for a player/market (opening line).

        Checks is_opening flag first, then falls back to earliest timestamp.
        """
        session = self._session()
        try:
            # Try explicit opening flag
            q = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                    CLVLineMovement.is_opening == True,  # noqa: E712
                )
            )
            if event_id:
                q = q.filter(CLVLineMovement.event_id == event_id)

            row = q.order_by(CLVLineMovement.timestamp.asc()).first()
            if row:
                return row.to_dict()

            # Fallback: earliest observation
            q2 = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                )
            )
            if event_id:
                q2 = q2.filter(CLVLineMovement.event_id == event_id)

            row2 = q2.order_by(CLVLineMovement.timestamp.asc()).first()
            return row2.to_dict() if row2 else None
        finally:
            session.close()

    # ── Query: Latest line across all books ───────────────────────────

    def get_current_lines(
        self, player: str, market_type: str,
    ) -> List[Dict[str, Any]]:
        """Get the most recent line from each book for a player/market."""
        session = self._session()
        try:
            subq = (
                session.query(
                    CLVLineMovement.book,
                    sa_func.max(CLVLineMovement.timestamp).label("max_ts"),
                )
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                )
                .group_by(CLVLineMovement.book)
                .subquery()
            )

            rows = (
                session.query(CLVLineMovement)
                .join(
                    subq,
                    and_(
                        CLVLineMovement.book == subq.c.book,
                        CLVLineMovement.timestamp == subq.c.max_ts,
                    ),
                )
                .filter(
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                )
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    # ── Query: Consensus line ────────────────────────────────────────

    def get_consensus_line(
        self, player: str, market_type: str,
    ) -> Optional[float]:
        """Calculate the consensus (median) line across all books."""
        current = self.get_current_lines(player, market_type)
        if not current:
            return None

        lines = sorted(r["line"] for r in current if r.get("line") is not None)
        if not lines:
            return None

        n = len(lines)
        if n % 2 == 1:
            return lines[n // 2]
        return (lines[n // 2 - 1] + lines[n // 2]) / 2.0

    # ── Query: Best available line ───────────────────────────────────

    def get_best_line(
        self,
        player: str,
        market_type: str,
        direction: str = "over",
    ) -> Optional[Dict[str, Any]]:
        """Get the best available line for a direction.

        For over bets: lowest line is best.
        For under bets: highest line is best.
        """
        current = self.get_current_lines(player, market_type)
        if not current:
            return None

        valid = [r for r in current if r.get("line") is not None]
        if not valid:
            return None

        if direction.lower() == "over":
            return min(valid, key=lambda r: r["line"])
        else:
            return max(valid, key=lambda r: r["line"])

    # ── Line movement statistics ─────────────────────────────────────

    def compute_movement_stats(
        self, player: str, market_type: str, hours: int = 24,
    ) -> Dict[str, Any]:
        """Compute movement statistics for a player/market.

        Returns:
            {
                "n_observations": int,
                "opening_line": float,
                "current_line": float,
                "total_movement": float,
                "volatility": float,
                "direction": str,  # "up", "down", "stable"
                "n_books": int,
            }
        """
        history = self.get_line_history(player, market_type, hours=hours)
        if not history:
            return {
                "n_observations": 0,
                "opening_line": None,
                "current_line": None,
                "total_movement": 0.0,
                "volatility": 0.0,
                "direction": "unknown",
                "n_books": 0,
            }

        lines = [h["line"] for h in history if h.get("line") is not None]
        books = set(h.get("book", "") for h in history)

        if not lines:
            return {
                "n_observations": 0,
                "opening_line": None,
                "current_line": None,
                "total_movement": 0.0,
                "volatility": 0.0,
                "direction": "unknown",
                "n_books": 0,
            }

        opening = lines[0]
        current = lines[-1]
        total_movement = current - opening

        # Volatility: std dev of line changes
        if len(lines) > 1:
            changes = [lines[i] - lines[i - 1] for i in range(1, len(lines))]
            import numpy as np
            volatility = float(np.std(changes)) if changes else 0.0
        else:
            volatility = 0.0

        if total_movement > 0.25:
            direction = "up"
        elif total_movement < -0.25:
            direction = "down"
        else:
            direction = "stable"

        return {
            "n_observations": len(lines),
            "opening_line": opening,
            "current_line": current,
            "total_movement": round(total_movement, 3),
            "volatility": round(volatility, 4),
            "direction": direction,
            "n_books": len(books),
        }

    # ── Retention policy ─────────────────────────────────────────────

    def run_retention_policy(self) -> Dict[str, int]:
        """Archive detailed data older than RETENTION_DAYS into hourly aggregates.

        Returns counts of archived and deleted rows.
        """
        cutoff = _utcnow() - timedelta(days=self.RETENTION_DAYS)
        session = self._session()
        archived = 0
        deleted = 0

        try:
            # Find distinct player/market/book/hour combos older than cutoff
            old_rows = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp < cutoff,
                )
                .order_by(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                    CLVLineMovement.timestamp,
                )
                .all()
            )

            if not old_rows:
                return {"archived": 0, "deleted": 0}

            # Group by (player, market_type, book, hour_bucket)
            buckets: Dict[tuple, List[CLVLineMovement]] = {}
            for row in old_rows:
                ts = row.timestamp
                hour_bucket = ts.replace(minute=0, second=0, microsecond=0)
                key = (row.player, row.market_type, row.book, hour_bucket)
                buckets.setdefault(key, []).append(row)

            # Create archive records
            for (player, market_type, book, hour_bucket), rows in buckets.items():
                lines = [r.line for r in rows]
                archive = CLVLineMovementArchive(
                    sport=self.sport,
                    event_id=rows[0].event_id,
                    market_type=market_type,
                    book=book,
                    player=player,
                    hour_bucket=hour_bucket,
                    line_open=lines[0],
                    line_close=lines[-1],
                    line_high=max(lines),
                    line_low=min(lines),
                    n_observations=len(lines),
                )
                session.add(archive)
                archived += 1

            # Delete the old detailed rows
            deleted = (
                session.query(CLVLineMovement)
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.timestamp < cutoff,
                )
                .delete(synchronize_session="fetch")
            )

            session.commit()
            log.info(
                "Retention policy: archived %d hour-buckets, deleted %d rows",
                archived, deleted,
            )
        except Exception:
            session.rollback()
            log.exception("Retention policy failed")
            raise
        finally:
            session.close()

        return {"archived": archived, "deleted": deleted}

    # ── Mark opening lines ───────────────────────────────────────────

    def mark_opening_lines(self, event_id: str) -> int:
        """Mark the first observation for each player/market/book as opening.

        Called when lines are first seen for a new event.
        Returns count of rows marked.
        """
        session = self._session()
        marked = 0
        try:
            subq = (
                session.query(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                    sa_func.min(CLVLineMovement.timestamp).label("min_ts"),
                )
                .filter(
                    CLVLineMovement.sport == self.sport,
                    CLVLineMovement.event_id == event_id,
                )
                .group_by(
                    CLVLineMovement.player,
                    CLVLineMovement.market_type,
                    CLVLineMovement.book,
                )
                .subquery()
            )

            rows = (
                session.query(CLVLineMovement)
                .join(
                    subq,
                    and_(
                        CLVLineMovement.player == subq.c.player,
                        CLVLineMovement.market_type == subq.c.market_type,
                        CLVLineMovement.book == subq.c.book,
                        CLVLineMovement.timestamp == subq.c.min_ts,
                    ),
                )
                .filter(CLVLineMovement.event_id == event_id)
                .all()
            )

            for row in rows:
                if not row.is_opening:
                    row.is_opening = True
                    marked += 1

            session.commit()
            log.info("Marked %d opening lines for event %s", marked, event_id)
        except Exception:
            session.rollback()
            log.exception("Failed to mark opening lines")
        finally:
            session.close()

        return marked

    # ── Table size info ──────────────────────────────────────────────

    def get_table_stats(self) -> Dict[str, Any]:
        """Return statistics about the line movements table."""
        session = self._session()
        try:
            total = (
                session.query(sa_func.count(CLVLineMovement.id))
                .filter(CLVLineMovement.sport == self.sport)
                .scalar()
            )
            oldest = (
                session.query(sa_func.min(CLVLineMovement.timestamp))
                .filter(CLVLineMovement.sport == self.sport)
                .scalar()
            )
            newest = (
                session.query(sa_func.max(CLVLineMovement.timestamp))
                .filter(CLVLineMovement.sport == self.sport)
                .scalar()
            )
            n_books = (
                session.query(sa_func.count(sa_func.distinct(CLVLineMovement.book)))
                .filter(CLVLineMovement.sport == self.sport)
                .scalar()
            )
            n_players = (
                session.query(sa_func.count(sa_func.distinct(CLVLineMovement.player)))
                .filter(CLVLineMovement.sport == self.sport)
                .scalar()
            )
            archive_count = (
                session.query(sa_func.count(CLVLineMovementArchive.id))
                .filter(CLVLineMovementArchive.sport == self.sport)
                .scalar()
            )

            return {
                "total_rows": total or 0,
                "oldest_record": oldest.isoformat() if oldest else None,
                "newest_record": newest.isoformat() if newest else None,
                "n_books": n_books or 0,
                "n_players": n_players or 0,
                "archive_rows": archive_count or 0,
            }
        finally:
            session.close()
