"""Database service for VisionSuite backend."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from server.core.config import settings

logger = logging.getLogger("database")

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    try:
        _engine = create_engine(
            settings.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=settings.DEBUG,
            future=True,
        )
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False, expire_on_commit=False)
        logger.info("PostgreSQL engine initialized")
        return _engine
    except Exception as exc:
        logger.error("Failed to create database engine: %s", exc)
        return None


def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        get_engine()
    if _SessionLocal is None:
        raise RuntimeError("Database not available")
    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    yield from get_session()


def is_available() -> bool:
    try:
        engine = get_engine()
        if engine is None:
            return False
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def init_db() -> bool:
    try:
        engine = get_engine()
        if engine is None:
            return False
        from server.models.orm_models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized")
        return True
    except Exception as exc:
        logger.error("Failed to initialize database: %s", exc)
        return False
