"""
services/clv_system/models.py
==============================
Database models for the autonomous CLV tracking system.

These extend the existing quant_system schema with CLV-specific tables:
  - clv_line_movements   — High-resolution time-series of every line change
  - clv_bet_snapshots    — Full market snapshot at bet-signal time
  - clv_closing_lines    — Captured closing lines matched to events
  - clv_integrity_reports — Periodic data-quality audit results

Uses the same Base and engine from quant_system.db.schema so all tables
live in the same quant_system.db file.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Text,
    Index,
)
from quant_system.db.schema import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════
# 1. High-resolution line movements (millisecond timestamps)
# ═══════════════════════════════════════════════════════════════════════════

class CLVLineMovement(Base):
    """Every line movement captured across all books and markets.

    This is the master time-series table for CLV tracking.  Grows large;
    a retention policy archives rows older than 90 days into aggregated form.
    """
    __tablename__ = "clv_line_movements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(16), nullable=False, index=True)
    event_id = Column(String(256), nullable=True)
    market_type = Column(String(64), nullable=False)
    book = Column(String(64), nullable=False)
    player = Column(String(128), nullable=True)
    line = Column(Float, nullable=False)
    odds_american = Column(Integer, nullable=True)
    odds_decimal = Column(Float, nullable=True)
    implied_prob = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=False, default=_utcnow, index=True)
    is_opening = Column(Boolean, nullable=False, default=False)
    is_closing = Column(Boolean, nullable=False, default=False)
    source_raw = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_clvm_event_market_book_ts", "event_id", "market_type", "book", "timestamp"),
        Index("ix_clvm_player_market_ts", "player", "market_type", "timestamp"),
        Index("ix_clvm_sport_ts", "sport", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<CLVLineMovement id={self.id} player={self.player!r} "
            f"book={self.book!r} line={self.line} ts={self.timestamp}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sport": self.sport,
            "event_id": self.event_id,
            "market_type": self.market_type,
            "book": self.book,
            "player": self.player,
            "line": self.line,
            "odds_american": self.odds_american,
            "odds_decimal": self.odds_decimal,
            "implied_prob": self.implied_prob,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_opening": self.is_opening,
            "is_closing": self.is_closing,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Bet-time snapshots (captured when a signal fires)
# ═══════════════════════════════════════════════════════════════════════════

class CLVBetSnapshot(Base):
    """Full market snapshot at the moment a bet signal is generated.

    Captures lines across all tracked books so we can later compare
    the price we got vs. what was available.
    """
    __tablename__ = "clv_bet_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_id = Column(String(64), nullable=False, index=True)
    sport = Column(String(16), nullable=False)
    player = Column(String(128), nullable=False)
    market_type = Column(String(64), nullable=False)
    event_id = Column(String(256), nullable=True)
    # Snapshot data
    best_line = Column(Float, nullable=True)
    best_line_book = Column(String(64), nullable=True)
    consensus_line = Column(Float, nullable=True)
    lines_json = Column(Text, nullable=False, default="{}")
    # Metadata
    signal_timestamp = Column(DateTime, nullable=False, default=_utcnow, index=True)
    n_books_captured = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_clvbs_bet_id", "bet_id"),
        Index("ix_clvbs_player_market", "player", "market_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<CLVBetSnapshot bet_id={self.bet_id!r} player={self.player!r} "
            f"best={self.best_line} consensus={self.consensus_line}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bet_id": self.bet_id,
            "sport": self.sport,
            "player": self.player,
            "market_type": self.market_type,
            "event_id": self.event_id,
            "best_line": self.best_line,
            "best_line_book": self.best_line_book,
            "consensus_line": self.consensus_line,
            "lines_json": self.lines_json,
            "signal_timestamp": self.signal_timestamp.isoformat() if self.signal_timestamp else None,
            "n_books_captured": self.n_books_captured,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Closing line records (last line before event starts)
# ═══════════════════════════════════════════════════════════════════════════

class CLVClosingLine(Base):
    """The definitive closing line for each player/market/book combination.

    NBA:  captured at tip-off.
    Golf: captured at first tee time of the round.
    """
    __tablename__ = "clv_closing_lines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(16), nullable=False)
    event_id = Column(String(256), nullable=False)
    player = Column(String(128), nullable=False)
    market_type = Column(String(64), nullable=False)
    book = Column(String(64), nullable=False)
    closing_line = Column(Float, nullable=False)
    closing_odds_american = Column(Integer, nullable=True)
    closing_odds_decimal = Column(Float, nullable=True)
    closing_implied_prob = Column(Float, nullable=True)
    event_start_time = Column(DateTime, nullable=True)
    captured_at = Column(DateTime, nullable=False, default=_utcnow, index=True)
    is_consensus = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_clvcl_event_player_market", "event_id", "player", "market_type"),
        Index("ix_clvcl_sport_captured", "sport", "captured_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<CLVClosingLine event={self.event_id!r} player={self.player!r} "
            f"market={self.market_type!r} close={self.closing_line}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sport": self.sport,
            "event_id": self.event_id,
            "player": self.player,
            "market_type": self.market_type,
            "book": self.book,
            "closing_line": self.closing_line,
            "closing_odds_american": self.closing_odds_american,
            "closing_implied_prob": self.closing_implied_prob,
            "event_start_time": self.event_start_time.isoformat() if self.event_start_time else None,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "is_consensus": self.is_consensus,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integrity reports (periodic data quality audits)
# ═══════════════════════════════════════════════════════════════════════════

class CLVIntegrityReport(Base):
    """Stores periodic data-quality audit results.

    If integrity_score < 80, CLV data cannot be trusted and system edge
    is unverifiable.
    """
    __tablename__ = "clv_integrity_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(16), nullable=False)
    generated_at = Column(DateTime, nullable=False, default=_utcnow, index=True)
    total_bets = Column(Integer, nullable=False, default=0)
    missing_opening_lines = Column(Integer, nullable=False, default=0)
    missing_closing_lines = Column(Integer, nullable=False, default=0)
    missing_bet_snapshots = Column(Integer, nullable=False, default=0)
    suspected_data_errors = Column(Integer, nullable=False, default=0)
    integrity_score = Column(Float, nullable=False, default=100.0)
    report_text = Column(Text, nullable=False, default="")
    details_json = Column(Text, nullable=False, default="{}")

    __table_args__ = (
        Index("ix_clvir_sport_generated", "sport", "generated_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<CLVIntegrityReport sport={self.sport!r} "
            f"score={self.integrity_score:.1f} bets={self.total_bets}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sport": self.sport,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "total_bets": self.total_bets,
            "missing_opening_lines": self.missing_opening_lines,
            "missing_closing_lines": self.missing_closing_lines,
            "missing_bet_snapshots": self.missing_bet_snapshots,
            "suspected_data_errors": self.suspected_data_errors,
            "integrity_score": self.integrity_score,
            "report_text": self.report_text,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Aggregated line movement archive (for retention policy)
# ═══════════════════════════════════════════════════════════════════════════

class CLVLineMovementArchive(Base):
    """Aggregated (hourly) archive of old line movements.

    After 90 days, detailed per-poll data is aggregated into hourly summaries
    to manage database size while preserving trend information.
    """
    __tablename__ = "clv_line_movement_archive"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(16), nullable=False)
    event_id = Column(String(256), nullable=True)
    market_type = Column(String(64), nullable=False)
    book = Column(String(64), nullable=False)
    player = Column(String(128), nullable=True)
    hour_bucket = Column(DateTime, nullable=False, index=True)
    line_open = Column(Float, nullable=False)
    line_close = Column(Float, nullable=False)
    line_high = Column(Float, nullable=False)
    line_low = Column(Float, nullable=False)
    n_observations = Column(Integer, nullable=False, default=1)

    __table_args__ = (
        Index("ix_clvma_player_market_hour", "player", "market_type", "hour_bucket"),
        Index("ix_clvma_sport_hour", "sport", "hour_bucket"),
    )

    def __repr__(self) -> str:
        return (
            f"<CLVLineMovementArchive player={self.player!r} "
            f"hour={self.hour_bucket} open={self.line_open} close={self.line_close}>"
        )


# Convenience: all CLV models for create_all
CLV_MODELS = [
    CLVLineMovement,
    CLVBetSnapshot,
    CLVClosingLine,
    CLVIntegrityReport,
    CLVLineMovementArchive,
]
