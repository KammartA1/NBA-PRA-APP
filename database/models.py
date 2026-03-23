"""
database/models.py
==================
Canonical SQLAlchemy ORM models for every table in the NBA-PRA-APP.
This is the SINGLE source of truth for the database schema.

All models use SQLAlchemy 2.0+ declarative style with:
  - Proper composite/single-column indexes for common query patterns
  - __repr__ for debugging
  - to_dict() for easy serialization
  - created_at / updated_at audit columns where applicable
  - Relationships where appropriate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    Text,
    Index,
    ForeignKey,
    event,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Application-wide declarative base."""
    pass


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Helper mixin
# ---------------------------------------------------------------------------

class _DictMixin:
    """Adds ``to_dict()`` to every model that inherits it."""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for col in self.__table__.columns:
            val = getattr(self, col.name)
            if isinstance(val, datetime):
                val = val.isoformat()
            result[col.name] = val
        return result


# ===================================================================
# 1. bets
# ===================================================================

class Bet(_DictMixin, Base):
    """Every bet placed through the system.  Immutable after creation except
    for settlement fields (status, settled_at, pnl, actual_outcome,
    closing_line)."""

    __tablename__ = "bets"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    sport                 = Column(String(16), nullable=False, default="NBA")
    event                 = Column(String(256), nullable=True)
    market                = Column(String(64), nullable=True)
    player                = Column(String(128), nullable=False, index=True)
    signal_line           = Column(Float, nullable=True)
    bet_line              = Column(Float, nullable=False)
    closing_line          = Column(Float, nullable=True)
    predicted_prob        = Column(Float, nullable=False)
    actual_outcome        = Column(Float, nullable=True)
    stake                 = Column(Float, nullable=False, default=0.0)
    profit                = Column(Float, nullable=True)
    timestamp             = Column(DateTime, nullable=False, default=_utcnow, index=True)
    model_version         = Column(String(32), nullable=True)
    direction             = Column(String(16), nullable=False)          # over / under
    odds_american         = Column(Integer, nullable=True)
    odds_decimal          = Column(Float, nullable=True)
    model_projection      = Column(Float, nullable=True)
    model_std             = Column(Float, nullable=True)
    confidence_score      = Column(Float, nullable=True, default=0.0)
    status                = Column(String(16), nullable=False, default="pending", index=True)
    settled_at            = Column(DateTime, nullable=True)
    pnl                   = Column(Float, nullable=True, default=0.0)
    features_snapshot_json = Column(Text, nullable=True, default="{}")
    notes                 = Column(Text, nullable=True, default="")
    created_at            = Column(DateTime, nullable=False, default=_utcnow)
    updated_at            = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    # -- relationships --
    signals = relationship("Signal", back_populates="bet", lazy="select")

    __table_args__ = (
        Index("ix_bets_sport_status", "sport", "status"),
        Index("ix_bets_sport_timestamp", "sport", "timestamp"),
        Index("ix_bets_player_market", "player", "market"),
        Index("ix_bets_model_version", "model_version"),
    )

    def __repr__(self) -> str:
        return (
            f"<Bet(id={self.id}, player={self.player!r}, direction={self.direction!r}, "
            f"status={self.status!r}, pnl={self.pnl})>"
        )


# ===================================================================
# 2. line_movements
# ===================================================================

class LineMovement(_DictMixin, Base):
    """Time-series of line / odds snapshots for a given prop market."""

    __tablename__ = "line_movements"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    sport      = Column(String(16), nullable=False, default="NBA")
    event      = Column(String(256), nullable=True)
    market     = Column(String(64), nullable=True)
    book       = Column(String(64), nullable=True)
    player     = Column(String(128), nullable=False, index=True)
    line       = Column(Float, nullable=False)
    odds       = Column(Integer, nullable=True)
    timestamp  = Column(DateTime, nullable=False, default=_utcnow, index=True)
    is_opening = Column(Boolean, nullable=False, default=False)
    is_closing = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_lm_player_market_ts", "player", "market", "timestamp"),
        Index("ix_lm_sport_book_ts", "sport", "book", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<LineMovement(id={self.id}, player={self.player!r}, "
            f"line={self.line}, book={self.book!r})>"
        )


# ===================================================================
# 3. model_versions
# ===================================================================

class ModelVersion(_DictMixin, Base):
    """Registry of every model version deployed."""

    __tablename__ = "model_versions"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    version                 = Column(String(32), unique=True, nullable=False, index=True)
    created_at              = Column(DateTime, nullable=False, default=_utcnow)
    parameters_json         = Column(Text, nullable=True, default="{}")
    training_data_hash      = Column(String(128), nullable=True)
    performance_metrics_json = Column(Text, nullable=True, default="{}")
    sport                   = Column(String(16), nullable=False, default="NBA")
    is_active               = Column(Boolean, nullable=False, default=True)

    __table_args__ = (
        Index("ix_mv_sport_active", "sport", "is_active"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelVersion(id={self.id}, version={self.version!r}, "
            f"sport={self.sport!r}, active={self.is_active})>"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return json.loads(self.parameters_json or "{}")

    @property
    def performance_metrics(self) -> Dict[str, Any]:
        return json.loads(self.performance_metrics_json or "{}")


# ===================================================================
# 4. edge_reports
# ===================================================================

class EdgeReport(_DictMixin, Base):
    """Periodic or on-demand edge analysis reports (JSON blobs)."""

    __tablename__ = "edge_reports"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    report_type  = Column(String(64), nullable=False, index=True)
    generated_at = Column(DateTime, nullable=False, default=_utcnow, index=True)
    report_json  = Column(Text, nullable=False, default="{}")
    sport        = Column(String(16), nullable=False, default="NBA")

    __table_args__ = (
        Index("ix_er_sport_type", "sport", "report_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<EdgeReport(id={self.id}, type={self.report_type!r}, "
            f"sport={self.sport!r})>"
        )

    @property
    def report(self) -> Dict[str, Any]:
        return json.loads(self.report_json or "{}")


# ===================================================================
# 5. signals
# ===================================================================

class Signal(_DictMixin, Base):
    """Generated trading signals from the model pipeline."""

    __tablename__ = "signals"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    sport         = Column(String(16), nullable=False, default="NBA")
    event         = Column(String(256), nullable=True)
    market        = Column(String(64), nullable=True)
    player        = Column(String(128), nullable=False, index=True)
    signal_value  = Column(Float, nullable=False)
    confidence    = Column(Float, nullable=True)
    generated_at  = Column(DateTime, nullable=False, default=_utcnow, index=True)
    model_version = Column(String(32), nullable=True)
    direction     = Column(String(16), nullable=True)
    edge_pct      = Column(Float, nullable=True)
    kelly_stake   = Column(Float, nullable=True)
    bet_id        = Column(Integer, ForeignKey("bets.id"), nullable=True, index=True)

    # -- relationships --
    bet = relationship("Bet", back_populates="signals", lazy="select")

    __table_args__ = (
        Index("ix_sig_sport_gen", "sport", "generated_at"),
        Index("ix_sig_player_market", "player", "market"),
    )

    def __repr__(self) -> str:
        return (
            f"<Signal(id={self.id}, player={self.player!r}, "
            f"edge_pct={self.edge_pct}, direction={self.direction!r})>"
        )


# ===================================================================
# 6. players
# ===================================================================

class Player(_DictMixin, Base):
    """NBA player roster with metadata."""

    __tablename__ = "players"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String(128), nullable=False, index=True)
    team          = Column(String(64), nullable=True)
    sport         = Column(String(16), nullable=False, default="NBA")
    active        = Column(Boolean, nullable=False, default=True)
    metadata_json = Column(Text, nullable=True, default="{}")
    position      = Column(String(16), nullable=True)
    last_updated  = Column(DateTime, nullable=True, default=_utcnow, onupdate=_utcnow)
    created_at    = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_player_team_sport", "team", "sport"),
        Index("ix_player_name_sport", "name", "sport"),
    )

    def __repr__(self) -> str:
        return (
            f"<Player(id={self.id}, name={self.name!r}, "
            f"team={self.team!r}, active={self.active})>"
        )

    @property
    def extra_data(self) -> Dict[str, Any]:
        return json.loads(self.metadata_json or "{}")


# ===================================================================
# 7. events
# ===================================================================

class Event(_DictMixin, Base):
    """NBA games / events."""

    __tablename__ = "events"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    sport         = Column(String(16), nullable=False, default="NBA")
    event_name    = Column(String(256), nullable=False)
    start_time    = Column(DateTime, nullable=True, index=True)
    status        = Column(String(32), nullable=False, default="scheduled", index=True)
    metadata_json = Column(Text, nullable=True, default="{}")
    season        = Column(String(16), nullable=True)
    venue         = Column(String(128), nullable=True)
    created_at    = Column(DateTime, nullable=False, default=_utcnow)
    updated_at    = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("ix_event_sport_status", "sport", "status"),
        Index("ix_event_sport_start", "sport", "start_time"),
        Index("ix_event_season", "season"),
    )

    def __repr__(self) -> str:
        return (
            f"<Event(id={self.id}, name={self.event_name!r}, "
            f"status={self.status!r})>"
        )

    @property
    def extra_data(self) -> Dict[str, Any]:
        return json.loads(self.metadata_json or "{}")


# ===================================================================
# 8. user_settings
# ===================================================================

class UserSetting(_DictMixin, Base):
    """Persistent key-value settings (replaces session_state for durability)."""

    __tablename__ = "user_settings"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(String(64), nullable=False, default="default", index=True)
    setting_key   = Column(String(128), nullable=False)
    setting_value = Column(Text, nullable=True)
    updated_at    = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    __table_args__ = (
        Index("ix_us_user_key", "user_id", "setting_key", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<UserSetting(user={self.user_id!r}, key={self.setting_key!r})>"
        )


# ===================================================================
# 9. worker_status
# ===================================================================

class WorkerStatus(_DictMixin, Base):
    """Health / schedule tracking for background workers and scrapers."""

    __tablename__ = "worker_status"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    worker_name        = Column(String(128), unique=True, nullable=False, index=True)
    last_run           = Column(DateTime, nullable=True)
    last_success       = Column(DateTime, nullable=True)
    last_error         = Column(Text, nullable=True)
    next_scheduled_run = Column(DateTime, nullable=True)
    status             = Column(String(32), nullable=False, default="idle", index=True)
    metadata_json      = Column(Text, nullable=True, default="{}")
    created_at         = Column(DateTime, nullable=False, default=_utcnow)
    updated_at         = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<WorkerStatus(name={self.worker_name!r}, status={self.status!r})>"
        )

    @property
    def extra_data(self) -> Dict[str, Any]:
        return json.loads(self.metadata_json or "{}")


# ===================================================================
# 10. calibration_snapshots
# ===================================================================

class CalibrationSnapshot(_DictMixin, Base):
    """Point-in-time calibration bucket data for model monitoring."""

    __tablename__ = "calibration_snapshots"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    sport             = Column(String(16), nullable=False, default="NBA")
    bucket_label      = Column(String(32), nullable=False)
    prob_lower        = Column(Float, nullable=False)
    prob_upper        = Column(Float, nullable=False)
    predicted_avg     = Column(Float, nullable=False)
    actual_rate       = Column(Float, nullable=False)
    n_bets            = Column(Integer, nullable=False)
    calibration_error = Column(Float, nullable=False)
    snapshot_date     = Column(DateTime, nullable=False, default=_utcnow, index=True)

    __table_args__ = (
        Index("ix_cal_sport_date", "sport", "snapshot_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<CalibrationSnapshot(id={self.id}, bucket={self.bucket_label!r}, "
            f"error={self.calibration_error:.4f})>"
        )


# ===================================================================
# 11. system_state
# ===================================================================

class SystemState(_DictMixin, Base):
    """Audit trail of system operating-state transitions."""

    __tablename__ = "system_state"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    sport               = Column(String(16), nullable=False, default="NBA")
    state               = Column(String(32), nullable=False, default="ACTIVE", index=True)
    reason              = Column(Text, nullable=True)
    changed_at          = Column(DateTime, nullable=False, default=_utcnow, index=True)
    clv_at_change       = Column(Float, nullable=True)
    bankroll_at_change  = Column(Float, nullable=True)
    drawdown_at_change  = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_ss_sport_changed", "sport", "changed_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<SystemState(id={self.id}, sport={self.sport!r}, "
            f"state={self.state!r})>"
        )


# ===================================================================
# 12. schema_versions  (used by migrations.py)
# ===================================================================

class SchemaVersion(_DictMixin, Base):
    """Tracks the current schema migration version."""

    __tablename__ = "schema_versions"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    version    = Column(Integer, nullable=False, unique=True)
    applied_at = Column(DateTime, nullable=False, default=_utcnow)
    description = Column(String(256), nullable=True)

    def __repr__(self) -> str:
        return f"<SchemaVersion(version={self.version}, applied={self.applied_at})>"


# ---------------------------------------------------------------------------
# Convenience: collect every model for ``create_all`` / introspection
# ---------------------------------------------------------------------------

ALL_MODELS = [
    Bet,
    LineMovement,
    ModelVersion,
    EdgeReport,
    Signal,
    Player,
    Event,
    UserSetting,
    WorkerStatus,
    CalibrationSnapshot,
    SystemState,
    SchemaVersion,
]
