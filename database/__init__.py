"""
database — Canonical data layer for NBA-PRA-APP
================================================
Single source of truth for all persistent storage.

Quick start::

    from database import init_db, get_session, session_scope, Bet, Signal

    init_db()                       # create tables (idempotent)
    with session_scope() as s:
        s.add(Bet(player="LeBron James", ...))
"""

# -- connection management --------------------------------------------------
from database.connection import (
    get_engine,
    get_session,
    get_session_factory,
    session_scope,
    init_db,
    health_check,
    dispose_engine,
)

# -- ORM models -------------------------------------------------------------
from database.models import (
    Base,
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
    ALL_MODELS,
)

# -- migrations --------------------------------------------------------------
from database.migrations import (
    auto_migrate,
    rollback,
    get_current_version,
    set_version,
    bootstrap_db,
    register as register_migration,
    get_registered_migrations,
)

__all__ = [
    # connection
    "get_engine",
    "get_session",
    "get_session_factory",
    "session_scope",
    "init_db",
    "health_check",
    "dispose_engine",
    # models
    "Base",
    "Bet",
    "LineMovement",
    "ModelVersion",
    "EdgeReport",
    "Signal",
    "Player",
    "Event",
    "UserSetting",
    "WorkerStatus",
    "CalibrationSnapshot",
    "SystemState",
    "SchemaVersion",
    "ALL_MODELS",
    # migrations
    "auto_migrate",
    "rollback",
    "get_current_version",
    "set_version",
    "bootstrap_db",
    "register_migration",
    "get_registered_migrations",
]
