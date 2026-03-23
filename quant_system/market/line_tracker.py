"""Line Movement Tracker — Records every line change for CLV calculation.

Lines move because sharp bettors (professionals) bet into them. Tracking
line movement tells you:
1. Where sharp money is going
2. Whether you got a good price (CLV)
3. Whether the market agrees with your model

Implementation:
- Poll odds APIs every 5-15 minutes during active periods
- Store every snapshot with timestamp
- Flag opening and closing lines
- Calculate movement velocity and direction
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from ..db.schema import LineSnapshot, get_session
from ..core.types import Sport

logger = logging.getLogger(__name__)


class LineTracker:
    """Tracks line movements across all sources."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def record_snapshot(
        self,
        player: str,
        stat_type: str,
        source: str,
        line: float,
        odds_american: Optional[int] = None,
        is_opening: bool = False,
        is_closing: bool = False,
    ) -> None:
        """Record a line snapshot."""
        session = self._session()
        try:
            snap = LineSnapshot(
                sport=self.sport.value,
                player=player,
                stat_type=stat_type,
                source=source,
                line=line,
                odds_american=odds_american,
                captured_at=datetime.utcnow(),
                is_opening=is_opening,
                is_closing=is_closing,
            )
            session.add(snap)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to record line snapshot")
        finally:
            session.close()

    def get_line_history(
        self,
        player: str,
        stat_type: str,
        source: str | None = None,
        hours: int = 48,
    ) -> list[dict]:
        """Get line history for a player/stat over last N hours."""
        session = self._session()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            query = (
                session.query(LineSnapshot)
                .filter_by(sport=self.sport.value, player=player, stat_type=stat_type)
                .filter(LineSnapshot.captured_at >= cutoff)
            )
            if source:
                query = query.filter_by(source=source)

            rows = query.order_by(LineSnapshot.captured_at.asc()).all()
            return [{
                "line": r.line,
                "odds_american": r.odds_american,
                "source": r.source,
                "captured_at": r.captured_at.isoformat(),
                "is_opening": r.is_opening,
                "is_closing": r.is_closing,
            } for r in rows]
        finally:
            session.close()

    def get_opening_line(self, player: str, stat_type: str) -> Optional[float]:
        """Get the opening line for a player/stat."""
        session = self._session()
        try:
            snap = (
                session.query(LineSnapshot)
                .filter_by(
                    sport=self.sport.value,
                    player=player,
                    stat_type=stat_type,
                    is_opening=True,
                )
                .order_by(LineSnapshot.captured_at.desc())
                .first()
            )
            return snap.line if snap else None
        finally:
            session.close()

    def get_closing_line(self, player: str, stat_type: str) -> Optional[float]:
        """Get the closing line (or latest line if game hasn't started)."""
        session = self._session()
        try:
            # First try explicit closing line
            snap = (
                session.query(LineSnapshot)
                .filter_by(
                    sport=self.sport.value,
                    player=player,
                    stat_type=stat_type,
                    is_closing=True,
                )
                .order_by(LineSnapshot.captured_at.desc())
                .first()
            )
            if snap:
                return snap.line

            # Fall back to most recent snapshot
            snap = (
                session.query(LineSnapshot)
                .filter_by(
                    sport=self.sport.value,
                    player=player,
                    stat_type=stat_type,
                )
                .order_by(LineSnapshot.captured_at.desc())
                .first()
            )
            return snap.line if snap else None
        finally:
            session.close()

    def line_movement_analysis(self, player: str, stat_type: str) -> dict:
        """Analyze line movement for a specific prop.

        Returns:
            {
                "opening_line": float,
                "current_line": float,
                "total_movement": float,
                "movement_direction": str,   # "up", "down", "stable"
                "velocity": float,           # Points per hour
                "n_changes": int,
                "steam_detected": bool,      # Rapid movement in one direction
            }
        """
        history = self.get_line_history(player, stat_type, hours=48)
        if len(history) < 2:
            return {
                "opening_line": history[0]["line"] if history else None,
                "current_line": history[-1]["line"] if history else None,
                "total_movement": 0.0,
                "movement_direction": "stable",
                "velocity": 0.0,
                "n_changes": 0,
                "steam_detected": False,
            }

        opening = history[0]["line"]
        current = history[-1]["line"]
        total_move = current - opening

        # Count actual line changes
        changes = 0
        for i in range(1, len(history)):
            if history[i]["line"] != history[i-1]["line"]:
                changes += 1

        # Movement velocity
        first_time = datetime.fromisoformat(history[0]["captured_at"])
        last_time = datetime.fromisoformat(history[-1]["captured_at"])
        hours_elapsed = max((last_time - first_time).total_seconds() / 3600, 0.01)
        velocity = total_move / hours_elapsed

        # Steam detection: > 1 point move in < 30 minutes
        steam = False
        for i in range(1, len(history)):
            t1 = datetime.fromisoformat(history[i-1]["captured_at"])
            t2 = datetime.fromisoformat(history[i]["captured_at"])
            dt_minutes = (t2 - t1).total_seconds() / 60
            dl = abs(history[i]["line"] - history[i-1]["line"])
            if dt_minutes < 30 and dl >= 1.0:
                steam = True
                break

        direction = "up" if total_move > 0.25 else ("down" if total_move < -0.25 else "stable")

        return {
            "opening_line": opening,
            "current_line": current,
            "total_movement": round(total_move, 2),
            "movement_direction": direction,
            "velocity": round(velocity, 4),
            "n_changes": changes,
            "steam_detected": steam,
        }
