"""Auto-logging of every bet with full context snapshot.

Every bet placed by the system gets logged here BEFORE execution.
This creates the immutable audit trail that feeds CLV tracking,
calibration, and edge validation."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from ..db.schema import BetLog, get_session
from .types import BetRecord, BetStatus, BetType, Sport

logger = logging.getLogger(__name__)


class BetLogger:
    """Logs every bet to the database with full context."""

    def __init__(self, sport: Sport, db_path: str | None = None):
        self.sport = sport
        self._db_path = db_path

    def _session(self):
        return get_session(self._db_path)

    def log_bet(
        self,
        player: str,
        bet_type: BetType,
        stat_type: str,
        line: float,
        direction: str,
        model_prob: float,
        market_prob: float,
        stake: float,
        kelly_fraction: float,
        odds_american: int,
        model_projection: float,
        model_std: float,
        confidence_score: float = 0.0,
        engine_agreement: float = 0.0,
        features_snapshot: dict | None = None,
        model_version: str = "1.0",
        notes: str = "",
    ) -> BetRecord:
        """Log a new bet. Returns the BetRecord with generated bet_id."""
        bet_id = f"{self.sport.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        edge = round(model_prob - market_prob, 6)

        if odds_american > 0:
            odds_decimal = round(1.0 + odds_american / 100.0, 4)
        elif odds_american < 0:
            odds_decimal = round(1.0 + 100.0 / abs(odds_american), 4)
        else:
            odds_decimal = 2.0  # Even money for PrizePicks

        record = BetRecord(
            bet_id=bet_id,
            sport=self.sport,
            timestamp=datetime.utcnow(),
            player=player,
            bet_type=bet_type,
            stat_type=stat_type,
            line=line,
            direction=direction,
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            stake=stake,
            kelly_fraction=kelly_fraction,
            odds_american=odds_american,
            odds_decimal=odds_decimal,
            model_projection=model_projection,
            model_std=model_std,
            confidence_score=confidence_score,
            engine_agreement=engine_agreement,
            model_version=model_version,
            features_used=json.dumps(features_snapshot or {}),
        )

        # Persist
        session = self._session()
        try:
            row = BetLog(
                bet_id=record.bet_id,
                sport=self.sport.value,
                timestamp=record.timestamp,
                player=player,
                bet_type=bet_type.value,
                stat_type=stat_type,
                line=line,
                direction=direction,
                model_prob=model_prob,
                market_prob=market_prob,
                edge=edge,
                stake=stake,
                kelly_fraction=kelly_fraction,
                odds_american=odds_american,
                odds_decimal=odds_decimal,
                model_projection=model_projection,
                model_std=model_std,
                confidence_score=confidence_score,
                engine_agreement=engine_agreement,
                model_version=model_version,
                features_snapshot=json.dumps(features_snapshot or {}),
                notes=notes,
            )
            session.add(row)
            session.commit()
            logger.info("Bet logged: %s | %s %s %s @ %.2f | edge=%.3f | stake=$%.2f",
                        bet_id, player, direction, stat_type, line, edge, stake)
        except Exception:
            session.rollback()
            logger.exception("Failed to log bet %s", bet_id)
            raise
        finally:
            session.close()

        return record

    def settle_bet(
        self,
        bet_id: str,
        status: BetStatus,
        actual_result: float,
        closing_line: Optional[float] = None,
        closing_odds: Optional[int] = None,
    ) -> float:
        """Settle a bet and return P&L."""
        session = self._session()
        try:
            row = session.query(BetLog).filter_by(bet_id=bet_id).first()
            if row is None:
                raise ValueError(f"Bet {bet_id} not found")

            row.status = status.value
            row.actual_result = actual_result
            row.settled_at = datetime.utcnow()

            if closing_line is not None:
                row.closing_line = closing_line
            if closing_odds is not None:
                row.closing_odds = closing_odds

            # Calculate P&L
            if status == BetStatus.WON:
                row.pnl = round(row.stake * (row.odds_decimal - 1.0), 2)
            elif status == BetStatus.LOST:
                row.pnl = -row.stake
            else:
                row.pnl = 0.0  # push/void

            session.commit()
            logger.info("Bet settled: %s → %s | P&L=$%.2f", bet_id, status.value, row.pnl)
            return row.pnl
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_pending_bets(self) -> list[dict]:
        """Get all unsettled bets."""
        session = self._session()
        try:
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value, status="pending")
                .order_by(BetLog.timestamp.desc())
                .all()
            )
            return [self._row_to_dict(r) for r in rows]
        finally:
            session.close()

    def get_settled_bets(self, limit: int = 500) -> list[dict]:
        """Get last N settled bets."""
        session = self._session()
        try:
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .filter(BetLog.status.in_(["won", "lost", "push"]))
                .order_by(BetLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]
        finally:
            session.close()

    def get_all_bets(self, limit: int = 1000) -> list[dict]:
        """Get all bets for this sport."""
        session = self._session()
        try:
            rows = (
                session.query(BetLog)
                .filter_by(sport=self.sport.value)
                .order_by(BetLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]
        finally:
            session.close()

    @staticmethod
    def _row_to_dict(row: BetLog) -> dict:
        return {
            "bet_id": row.bet_id,
            "sport": row.sport,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "player": row.player,
            "bet_type": row.bet_type,
            "stat_type": row.stat_type,
            "line": row.line,
            "direction": row.direction,
            "model_prob": row.model_prob,
            "market_prob": row.market_prob,
            "edge": row.edge,
            "stake": row.stake,
            "kelly_fraction": row.kelly_fraction,
            "odds_american": row.odds_american,
            "odds_decimal": row.odds_decimal,
            "model_projection": row.model_projection,
            "model_std": row.model_std,
            "confidence_score": row.confidence_score,
            "engine_agreement": row.engine_agreement,
            "status": row.status,
            "actual_result": row.actual_result,
            "closing_line": row.closing_line,
            "closing_odds": row.closing_odds,
            "settled_at": row.settled_at.isoformat() if row.settled_at else None,
            "pnl": row.pnl,
            "model_version": row.model_version,
        }
