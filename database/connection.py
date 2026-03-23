"""
database/connection.py
======================
SQLite connection manager (upgradeable to PostgreSQL) with:
  - Singleton engine + session factory
  - WAL journal mode & foreign keys enabled on every SQLite connection
  - SQLAlchemy 2.0 connection pooling
  - Health-check helper
  - ``init_db()`` to create all tables from models.py
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DB_NAME = "nba_quant.db"


def _resolve_db_url() -> str:
    """Build the database URL from the environment or fall back to the default
    SQLite path ``data/nba_quant.db``."""
    # Full URL override (allows postgres, mysql, etc.)
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    # Path-only override for SQLite
    db_path = os.environ.get(
        "NBA_DB_PATH",
        str(_DEFAULT_DB_DIR / _DEFAULT_DB_NAME),
    )
    db_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return f"sqlite:///{db_path}"


# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def get_engine() -> Engine:
    """Return the singleton SQLAlchemy ``Engine``, creating it on first call.

    For SQLite back-ends the engine is configured with:
      - ``check_same_thread=False`` (required by Streamlit / threads)
      - WAL journal mode
      - Foreign-key enforcement
      - A static pool size of 5 with 10 overflow connections
    """
    global _engine
    if _engine is not None:
        return _engine

    url = _resolve_db_url()
    is_sqlite = url.startswith("sqlite")

    connect_args = {}
    pool_kwargs = {}
    if is_sqlite:
        connect_args["check_same_thread"] = False
        # SQLite uses a NullPool by default; override with a static pool so
        # connections are reusable across Streamlit re-runs.
        pool_kwargs.update(pool_size=5, max_overflow=10, pool_pre_ping=True)
    else:
        pool_kwargs.update(pool_size=5, max_overflow=10, pool_pre_ping=True)

    _engine = create_engine(
        url,
        echo=False,
        connect_args=connect_args,
        **pool_kwargs,
    )

    # SQLite-specific PRAGMAs on every new raw DBAPI connection
    if is_sqlite:
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, connection_record):  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    log.info("Database engine created  url=%s", url.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the singleton ``sessionmaker`` bound to :func:`get_engine`."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autoflush=True,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def get_session() -> Session:
    """Open and return a new ``Session``.  Caller is responsible for calling
    ``session.close()`` (or use :func:`session_scope` for auto-close)."""
    factory = get_session_factory()
    return factory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager that yields a ``Session``, auto-committing on success
    and rolling back on exception.

    Usage::

        with session_scope() as s:
            s.add(Bet(...))
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def init_db() -> Engine:
    """Create every table defined in ``database.models`` (idempotent).

    Returns the engine for convenience.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    log.info("All tables created / verified.")
    return engine


def health_check() -> dict:
    """Run a lightweight connectivity + integrity check.

    Returns a dict with ``ok`` (bool) and diagnostic keys.
    """
    result: dict = {"ok": False, "tables": [], "error": None}
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Basic connectivity
            conn.execute(text("SELECT 1"))
            # List tables
            if engine.url.drivername.startswith("sqlite"):
                rows = conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                ).fetchall()
                result["tables"] = [r[0] for r in rows]
                # WAL check
                wal = conn.execute(text("PRAGMA journal_mode")).scalar()
                result["journal_mode"] = wal
            else:
                # PostgreSQL / other: use inspector
                from sqlalchemy import inspect as sa_inspect
                inspector = sa_inspect(engine)
                result["tables"] = sorted(inspector.get_table_names())
            result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
        log.exception("Database health-check failed")
    return result


# ---------------------------------------------------------------------------
# Teardown (useful in tests)
# ---------------------------------------------------------------------------

def dispose_engine() -> None:
    """Dispose the singleton engine and reset module-level state."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None
    log.info("Database engine disposed.")
