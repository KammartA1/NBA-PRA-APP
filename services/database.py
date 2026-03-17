"""
NBA PrizePicks — Database Layer
SQLite storage for scraped PrizePicks NBA lines and scraper monitoring.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    Boolean, DateTime, Text, Index, event,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

log = logging.getLogger(__name__)
Base = declarative_base()

DB_DIR = Path(__file__).resolve().parent.parent
DB_PATH = DB_DIR / "data" / "nba_prizepicks.db"


class NbaPrizePicksLine(Base):
    """A single NBA PrizePicks prop line."""
    __tablename__ = "nba_prizepicks_lines"
    __table_args__ = (
        Index("ix_nba_pp_player_stat", "player_name", "stat_type"),
        Index("ix_nba_pp_fetched", "fetched_at"),
    )
    id              = Column(Integer, primary_key=True)
    player_name     = Column(String, nullable=False)
    stat_type       = Column(String, nullable=False)
    line_score      = Column(Float, nullable=False)
    start_time      = Column(String)
    odds_type       = Column(String, default="standard")
    league          = Column(String, default="NBA")
    fetched_at      = Column(DateTime, default=datetime.utcnow, index=True)
    is_latest       = Column(Boolean, default=True, index=True)


class ScraperStatus(Base):
    """Monitoring table — one row per scraper, updated each run."""
    __tablename__ = "scraper_status"
    id              = Column(Integer, primary_key=True)
    scraper_name    = Column(String, unique=True, nullable=False)
    last_success    = Column(DateTime)
    last_attempt    = Column(DateTime)
    last_error      = Column(Text)
    lines_fetched   = Column(Integer, default=0)
    total_runs      = Column(Integer, default=0)
    total_errors    = Column(Integer, default=0)


class AuditLog(Base):
    """Scraper audit events."""
    __tablename__ = "scraper_audit_logs"
    id          = Column(Integer, primary_key=True)
    timestamp   = Column(DateTime, default=datetime.utcnow, index=True)
    event_type  = Column(String)
    description = Column(Text)
    data_json   = Column(Text)


# ── Engine ─────────────────────────────────────────────────────────────────
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            echo=False,
        )

        @event.listens_for(_engine, "connect")
        def set_wal(dbapi_conn, connection_record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(_engine)
        log.info(f"NBA PrizePicks DB initialized at {DB_PATH}")
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=True, autocommit=False)
    return _SessionLocal()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine
