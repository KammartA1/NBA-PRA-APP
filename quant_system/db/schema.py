"""SQLAlchemy schema for the quant system. Single source of truth for all bet
tracking, CLV, calibration, and system state data."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text, Index,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class BetLog(Base):
    """Every bet ever placed. Immutable after creation (except settlement)."""
    __tablename__ = "bet_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_id = Column(String(64), unique=True, nullable=False, index=True)
    sport = Column(String(10), nullable=False)                  # "nba" or "golf"
    timestamp = Column(DateTime, nullable=False, index=True)
    player = Column(String(128), nullable=False, index=True)
    bet_type = Column(String(32), nullable=False)               # "over", "under", "outright", etc.
    stat_type = Column(String(64), nullable=False)              # "points", "birdies", etc.
    line = Column(Float, nullable=False)
    direction = Column(String(10), nullable=False)
    model_prob = Column(Float, nullable=False)
    market_prob = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)
    stake = Column(Float, nullable=False)
    kelly_fraction = Column(Float, nullable=False)
    odds_american = Column(Integer, nullable=False)
    odds_decimal = Column(Float, nullable=False)
    model_projection = Column(Float, nullable=False)
    model_std = Column(Float, nullable=False)
    confidence_score = Column(Float, default=0.0)
    engine_agreement = Column(Float, default=0.0)
    # Settlement
    status = Column(String(16), default="pending", index=True)  # pending/won/lost/push/void
    actual_result = Column(Float, nullable=True)
    closing_line = Column(Float, nullable=True)
    closing_odds = Column(Integer, nullable=True)
    settled_at = Column(DateTime, nullable=True)
    pnl = Column(Float, default=0.0)
    # Metadata
    model_version = Column(String(16), default="1.0")
    features_snapshot = Column(Text, default="")                # JSON blob
    notes = Column(Text, default="")

    __table_args__ = (
        Index("ix_bet_sport_status", "sport", "status"),
        Index("ix_bet_sport_timestamp", "sport", "timestamp"),
    )


class LineSnapshot(Base):
    """Time-series of line movements for every tracked prop/market."""
    __tablename__ = "line_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    player = Column(String(128), nullable=False)
    stat_type = Column(String(64), nullable=False)
    source = Column(String(32), nullable=False)                 # "prizepicks", "draftkings", etc.
    line = Column(Float, nullable=False)
    odds_american = Column(Integer, nullable=True)
    over_prob_implied = Column(Float, nullable=True)
    under_prob_implied = Column(Float, nullable=True)
    captured_at = Column(DateTime, nullable=False, index=True)
    is_opening = Column(Boolean, default=False)
    is_closing = Column(Boolean, default=False)

    __table_args__ = (
        Index("ix_line_player_stat", "player", "stat_type", "captured_at"),
        Index("ix_line_sport_source", "sport", "source", "captured_at"),
    )


class CLVLog(Base):
    """CLV measurement for every settled bet."""
    __tablename__ = "clv_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_id = Column(String(64), nullable=False, index=True)
    sport = Column(String(10), nullable=False)
    opening_line = Column(Float, nullable=False)
    bet_line = Column(Float, nullable=False)
    closing_line = Column(Float, nullable=False)
    line_movement = Column(Float, nullable=False)
    clv_raw = Column(Float, nullable=False)                     # closing_prob - bet_prob
    clv_cents = Column(Float, nullable=False)                   # CLV in cents/dollar
    beat_close = Column(Boolean, nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_clv_sport", "sport", "calculated_at"),
    )


class CalibrationLog(Base):
    """Periodic calibration snapshots."""
    __tablename__ = "calibration_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    report_date = Column(DateTime, nullable=False)
    bucket_label = Column(String(16), nullable=False)           # "50-55%", "55-60%", etc.
    prob_lower = Column(Float, nullable=False)
    prob_upper = Column(Float, nullable=False)
    predicted_avg = Column(Float, nullable=False)
    actual_rate = Column(Float, nullable=False)
    n_bets = Column(Integer, nullable=False)
    calibration_error = Column(Float, nullable=False)
    is_overconfident = Column(Boolean, nullable=False)

    __table_args__ = (
        Index("ix_cal_sport_date", "sport", "report_date"),
    )


class SystemStateLog(Base):
    """Audit trail of system state changes."""
    __tablename__ = "system_state_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    previous_state = Column(String(16), nullable=False)
    new_state = Column(String(16), nullable=False)
    reason = Column(Text, nullable=False)
    clv_at_change = Column(Float, nullable=True)
    bankroll_at_change = Column(Float, nullable=True)
    drawdown_at_change = Column(Float, nullable=True)


class FeatureLog(Base):
    """Track feature importance over time for drift detection."""
    __tablename__ = "feature_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sport = Column(String(10), nullable=False)
    report_date = Column(DateTime, nullable=False)
    feature_name = Column(String(64), nullable=False)
    importance_score = Column(Float, nullable=False)            # Correlation with outcomes
    directional_accuracy = Column(Float, nullable=True)         # % correct direction
    n_samples = Column(Integer, nullable=False)
    is_degraded = Column(Boolean, default=False)                # Below threshold

    __table_args__ = (
        Index("ix_feature_sport_date", "sport", "report_date"),
    )


# ── Engine & Session Factory ──────────────────────────────────────────────

_engine = None
_Session = None


def get_engine(db_path: str | None = None):
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        if db_path is None:
            db_path = os.environ.get(
                "QUANT_DB_PATH",
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "quant_system.db"),
            )
        db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for concurrent reads
        from sqlalchemy import text
        with _engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
        Base.metadata.create_all(_engine)
    return _engine


def get_session(db_path: str | None = None):
    """Get a new SQLAlchemy session."""
    global _Session
    if _Session is None:
        engine = get_engine(db_path)
        _Session = sessionmaker(bind=engine)
    return _Session()
