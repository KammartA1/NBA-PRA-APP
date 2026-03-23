"""
database/migrations.py
======================
Lightweight, forward-compatible migration framework.

Each migration is registered with :func:`register` and consists of an ``up``
callable (apply) and a ``down`` callable (rollback).  Migrations are numbered
sequentially starting at 1.

Key API
-------
- ``auto_migrate()``       — apply all pending migrations
- ``rollback(to_version)`` — roll back to (but not including) *to_version*
- ``get_current_version()``
- ``set_version(v)``

The ``schema_versions`` table (defined in ``database.models``) persists the
applied-version watermark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from database.connection import get_engine, get_session, session_scope
from database.models import (
    Base,
    SchemaVersion,
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
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Migration dataclass + registry
# ---------------------------------------------------------------------------

@dataclass
class Migration:
    version: int
    description: str
    up: Callable[[Session], None]
    down: Callable[[Session], None]


_registry: Dict[int, Migration] = {}


def register(version: int, description: str,
             up: Callable[[Session], None],
             down: Callable[[Session], None]) -> None:
    """Register a migration.  ``up`` and ``down`` receive a *Session* and
    should execute DDL / DML through it."""
    if version in _registry:
        raise ValueError(f"Migration version {version} already registered")
    _registry[version] = Migration(
        version=version,
        description=description,
        up=up,
        down=down,
    )


def get_registered_migrations() -> List[Migration]:
    """Return all registered migrations sorted by version ascending."""
    return [_registry[v] for v in sorted(_registry)]


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def _ensure_version_table() -> None:
    """Create the ``schema_versions`` table if it doesn't exist yet."""
    engine = get_engine()
    SchemaVersion.__table__.create(engine, checkfirst=True)


def get_current_version() -> int:
    """Return the highest applied migration version, or 0 if none."""
    _ensure_version_table()
    session = get_session()
    try:
        row = (
            session.query(SchemaVersion)
            .order_by(SchemaVersion.version.desc())
            .first()
        )
        return row.version if row else 0
    finally:
        session.close()


def set_version(session: Session, version: int, description: str = "") -> None:
    """Record that *version* has been applied."""
    sv = SchemaVersion(
        version=version,
        applied_at=datetime.now(timezone.utc),
        description=description,
    )
    session.add(sv)
    session.flush()


def _remove_version(session: Session, version: int) -> None:
    """Remove the record for *version* (used during rollback)."""
    session.query(SchemaVersion).filter(SchemaVersion.version == version).delete()
    session.flush()


# ---------------------------------------------------------------------------
# Migration execution
# ---------------------------------------------------------------------------

def auto_migrate() -> int:
    """Apply every registered migration whose version is greater than the
    current DB version.  Returns the number of migrations applied."""
    _ensure_version_table()
    current = get_current_version()
    pending = [m for m in get_registered_migrations() if m.version > current]
    if not pending:
        log.info("Database is up-to-date at version %d.", current)
        return 0

    applied = 0
    for mig in pending:
        log.info("Applying migration v%d: %s ...", mig.version, mig.description)
        with session_scope() as session:
            mig.up(session)
            set_version(session, mig.version, mig.description)
        applied += 1
        log.info("Migration v%d applied.", mig.version)

    log.info("Applied %d migration(s). Current version: %d.", applied, mig.version)
    return applied


def rollback(to_version: int = 0) -> int:
    """Roll back migrations down to (but NOT including) *to_version*.

    For example ``rollback(0)`` undoes every migration.
    Returns the number of migrations rolled back.
    """
    _ensure_version_table()
    current = get_current_version()
    if current <= to_version:
        log.info("Nothing to roll back (current=%d, target=%d).", current, to_version)
        return 0

    # Walk backwards
    versions_desc = sorted(_registry.keys(), reverse=True)
    rolled = 0
    for v in versions_desc:
        if v <= to_version:
            break
        if v > current:
            continue
        mig = _registry[v]
        log.info("Rolling back migration v%d: %s ...", v, mig.description)
        with session_scope() as session:
            mig.down(session)
            _remove_version(session, v)
        rolled += 1
        log.info("Migration v%d rolled back.", v)

    log.info("Rolled back %d migration(s). Current version: %d.", rolled, to_version)
    return rolled


# ===================================================================
# Migration v1 — Initial schema (creates all tables)
# ===================================================================

def _v1_up(session: Session) -> None:
    """Create every table defined in database.models using ``create_all``."""
    engine = session.get_bind()
    Base.metadata.create_all(engine)
    log.info("v1 up: All tables created.")


def _v1_down(session: Session) -> None:
    """Drop every application table EXCEPT schema_versions (so we can
    still track that we rolled back)."""
    engine = session.get_bind()
    # Drop tables in reverse dependency order; exclude schema_versions
    tables_to_drop = [
        t for t in reversed(Base.metadata.sorted_tables)
        if t.name != "schema_versions"
    ]
    for table in tables_to_drop:
        table.drop(engine, checkfirst=True)
    log.info("v1 down: All application tables dropped.")


register(
    version=1,
    description="Initial schema — all core tables",
    up=_v1_up,
    down=_v1_down,
)


# ===================================================================
# Convenience: one-call bootstrap
# ===================================================================

def bootstrap_db() -> None:
    """Ensure the database exists, the schema_versions table is present,
    and all pending migrations have been applied.  Safe to call on every
    app startup.
    """
    _ensure_version_table()
    auto_migrate()
    log.info("Database bootstrap complete (version %d).", get_current_version())
