"""
SQLAlchemy engine and session factory for DemandIQ.

Supports both SQLite (default, zero-config) and PostgreSQL (via DATABASE_URL env var).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from demandiq.config import settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    """Create or return the cached SQLAlchemy engine."""
    global _engine
    if _engine is None:
        kwargs: dict = {}
        if settings.DATABASE_URL.startswith("sqlite"):
            # SQLite: enable WAL mode and enforce foreign keys
            kwargs["connect_args"] = {"check_same_thread": False}

        _engine = create_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
            **kwargs,
        )

        if settings.DATABASE_URL.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragmas(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info("Database engine created: %s", settings.DATABASE_URL.split("@")[-1])
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the cached session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager that provides a transactional database session."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """
    Create all tables if they don't exist (idempotent).

    Safe to call on every application start — does not drop existing data.
    """
    from demandiq.database.models import Base  # noqa: F401 — import triggers metadata registration

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database schema initialized (tables created or already exist).")


def check_connection() -> bool:
    """Test database connectivity. Returns True if connection succeeds."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)
        return False
