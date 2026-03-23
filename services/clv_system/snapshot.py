"""
services/clv_system/snapshot.py
================================
Automatic bet-time price snapshot service.

When the system generates a bet signal, this module automatically captures:
  - Current lines across all tracked books
  - Best available line
  - Market consensus line
  - Timestamp of signal generation

This snapshot is stored in clv_bet_snapshots and used later to measure
how the line the bettor got compares to what the market offered.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from quant_system.db.schema import get_engine, get_session
from services.clv_system.models import CLVBetSnapshot, CLVLineMovement, Base

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BetTimeSnapshotService:
    """Captures and stores full market snapshots at bet-signal time.

    Automatically triggered when the system generates a betting signal.
    Stores all available lines so we can later verify the price we got
    was the best available.
    """

    def __init__(self, sport: str = "NBA", db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path
        engine = get_engine(db_path)
        Base.metadata.create_all(engine)

    def _session(self):
        return get_session(self._db_path)

    def capture_snapshot(
        self,
        bet_id: str,
        player: str,
        market_type: str,
        event_id: str = "",
    ) -> Dict[str, Any]:
        """Capture a full market snapshot for a bet signal.

        Queries all recent lines for the player/market across all books,
        computes best line and consensus, and stores the snapshot.

        Args:
            bet_id: Unique bet identifier.
            player: Player name.
            market_type: Market type (e.g., "Points").
            event_id: Event identifier (optional).

        Returns:
            Dict with snapshot details.
        """
        session = self._session()
        try:
            # Get the most recent line from each book
            from sqlalchemy import func as sa_func

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

            recent_lines = (
                session.query(CLVLineMovement)
                .join(
                    subq,
                    (CLVLineMovement.book == subq.c.book)
                    & (CLVLineMovement.timestamp == subq.c.max_ts),
                )
                .filter(
                    CLVLineMovement.player == player,
                    CLVLineMovement.market_type == market_type,
                )
                .all()
            )

            # Build lines dict
            lines_data = {}
            line_values = []
            for lm in recent_lines:
                lines_data[lm.book] = {
                    "line": lm.line,
                    "odds_american": lm.odds_american,
                    "odds_decimal": lm.odds_decimal,
                    "implied_prob": lm.implied_prob,
                    "timestamp": lm.timestamp.isoformat() if lm.timestamp else None,
                }
                if lm.line is not None:
                    line_values.append((lm.book, lm.line))

            # Compute best and consensus
            best_line = None
            best_line_book = None
            consensus_line = None

            if line_values:
                # Best line (lowest for over bets — we store the generic best)
                best_entry = min(line_values, key=lambda x: x[1])
                best_line = best_entry[1]
                best_line_book = best_entry[0]

                # Consensus (median)
                sorted_vals = sorted(v[1] for v in line_values)
                n = len(sorted_vals)
                if n % 2 == 1:
                    consensus_line = sorted_vals[n // 2]
                else:
                    consensus_line = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

            # Store snapshot
            snapshot = CLVBetSnapshot(
                bet_id=bet_id,
                sport=self.sport,
                player=player,
                market_type=market_type,
                event_id=event_id,
                best_line=best_line,
                best_line_book=best_line_book,
                consensus_line=consensus_line,
                lines_json=json.dumps(lines_data, default=str),
                signal_timestamp=_utcnow(),
                n_books_captured=len(lines_data),
            )
            session.add(snapshot)
            session.commit()

            result = snapshot.to_dict()
            log.info(
                "Bet snapshot captured: bet=%s player=%s best=%.1f consensus=%.1f books=%d",
                bet_id, player,
                best_line or 0, consensus_line or 0,
                len(lines_data),
            )
            return result

        except Exception:
            session.rollback()
            log.exception("Failed to capture bet snapshot for %s", bet_id)
            raise
        finally:
            session.close()

    def get_snapshot(self, bet_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the bet-time snapshot for a given bet."""
        session = self._session()
        try:
            row = (
                session.query(CLVBetSnapshot)
                .filter(CLVBetSnapshot.bet_id == bet_id)
                .order_by(CLVBetSnapshot.signal_timestamp.desc())
                .first()
            )
            return row.to_dict() if row else None
        finally:
            session.close()

    def get_snapshot_lines(self, bet_id: str) -> Dict[str, Any]:
        """Get the detailed lines from a bet-time snapshot.

        Returns dict of {book: {line, odds_american, ...}}.
        """
        snapshot = self.get_snapshot(bet_id)
        if not snapshot:
            return {}
        try:
            return json.loads(snapshot.get("lines_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_snapshots_for_player(
        self, player: str, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get all bet-time snapshots for a player, most recent first."""
        session = self._session()
        try:
            rows = (
                session.query(CLVBetSnapshot)
                .filter(
                    CLVBetSnapshot.sport == self.sport,
                    CLVBetSnapshot.player == player,
                )
                .order_by(CLVBetSnapshot.signal_timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def get_recent_snapshots(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent bet-time snapshots across all players."""
        session = self._session()
        try:
            rows = (
                session.query(CLVBetSnapshot)
                .filter(CLVBetSnapshot.sport == self.sport)
                .order_by(CLVBetSnapshot.signal_timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in rows]
        finally:
            session.close()

    def snapshot_count(self) -> int:
        """Total number of bet-time snapshots."""
        from sqlalchemy import func as sa_func
        session = self._session()
        try:
            return (
                session.query(sa_func.count(CLVBetSnapshot.id))
                .filter(CLVBetSnapshot.sport == self.sport)
                .scalar()
            ) or 0
        finally:
            session.close()
