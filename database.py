"""VisionSuite — Local PostgreSQL persistence layer.

Central engine + session management for the application's local PostgreSQL
database. Uses SQLAlchemy ORM (2.x) with connection pooling and psycopg.

Design rules:
  * The DB is LOCAL to the device (localhost:5432) — never a cloud service.
  * Credentials come from environment variables (via ``.env`` / python-dotenv),
    never hard-coded in Python source.
  * If PostgreSQL is unavailable, nothing here raises out of the app: callers
    use ``get_pg_status()`` / ``is_available()`` and degrade gracefully.
  * Slow queries are never issued on the Qt GUI thread — repositories are
    called from worker threads / queued jobs.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

log = logging.getLogger("database")

# Load .env if present (project root), without overriding existing env vars.
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Env-derived configuration.
# ---------------------------------------------------------------------------
def get_db_config() -> dict[str, Any]:
    """Return the PostgreSQL connection config from environment variables."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "db": os.getenv("POSTGRES_DB", "visionsuite"),
        "user": os.getenv("POSTGRES_USER", "visionsuite_user"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
        "pool_size": int(os.getenv("POSTGRES_POOL_SIZE", "5")),
        "max_overflow": int(os.getenv("POSTGRES_MAX_OVERFLOW", "10")),
    }


def build_url(cfg: Optional[dict[str, Any]] = None) -> str:
    """Build a SQLAlchemy database URL from config (no password logged)."""
    c = cfg or get_db_config()
    return (
        f"postgresql+psycopg://{c['user']}:{c['password']}"
        f"@{c['host']}:{c['port']}/{c['db']}"
    )


# ---------------------------------------------------------------------------
# ORM base.
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Declarative base for all VisionSuite ORM models."""


# ---------------------------------------------------------------------------
# Engine / session management (module-level singletons, thread-safe).
# ---------------------------------------------------------------------------
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None
_engine_lock = threading.Lock()
_status: dict[str, Any] = {"connected": False, "error": "", "last_check": 0.0}


def make_engine() -> Engine:
    """Create a new SQLAlchemy engine with connection pooling for PostgreSQL."""
    cfg = get_db_config()
    if not cfg["password"]:
        log.warning("POSTGRES_PASSWORD is not set — connection will likely fail.")
    url = build_url(cfg)
    engine = create_engine(
        url,
        pool_size=cfg["pool_size"],
        max_overflow=cfg["max_overflow"],
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=False,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_connection, connection_record):  # pragma: no cover
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("SET search_path TO public")
            cursor.close()
        except Exception:
            pass

    return engine


def get_engine() -> Optional[Engine]:
    """Return the shared engine, creating it once (thread-safe)."""
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            try:
                _engine = make_engine()
                _SessionLocal = sessionmaker(
                    bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
                )
            except Exception as exc:  # pragma: no cover
                log.exception("Failed to build PostgreSQL engine")
                _engine = None
                _SessionLocal = None
    return _engine


def get_session() -> Session:
    """Return a new ORM session bound to the shared engine."""
    factory = _SessionLocal
    if factory is None:
        engine = get_engine()
        if engine is None:
            raise RuntimeError("PostgreSQL engine is not available.")
        factory = _SessionLocal  # type: ignore[assignment]
    return factory()


def is_available() -> bool:
    """Return True if PostgreSQL is reachable right now."""
    return bool(probe_connection().get("connected"))


# ---------------------------------------------------------------------------
# Connection probe / status.
# ---------------------------------------------------------------------------
def probe_connection() -> dict[str, Any]:
    """Test the connection and return a status dict (never raises)."""
    global _status
    engine = get_engine()
    if engine is None:
        _status = {"connected": False, "error": "PostgreSQL engine not initialised", "last_check": 0.0}
        return dict(_status)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _status = {"connected": True, "error": "", "last_check": __import__("time").time()}
    except Exception as exc:  # pragma: no cover - depends on server
        _status = {"connected": False, "error": str(exc), "last_check": __import__("time").time()}
        log.debug("PostgreSQL probe failed: %s", exc)
    return dict(_status)


def get_pg_status() -> dict[str, Any]:
    """Return a cached/current status dict suitable for the UI."""
    global _status
    # Refresh if we haven't checked recently (or never).
    now = __import__("time").time()
    if not _status.get("last_check") or (now - _status["last_check"]) > 5.0:
        return probe_connection()
    return dict(_status)


# ---------------------------------------------------------------------------
# Schema creation / simple migrations.
# ---------------------------------------------------------------------------
def init_db(create_all: bool = True) -> bool:
    """Create all tables on the local PostgreSQL database.

    Returns True on success, False if PostgreSQL is unavailable. This is
    idempotent and safe (``CREATE TABLE IF NOT EXISTS`` semantics via
    ``Base.metadata.create_all``), so it never deletes existing data.
    """
    engine = get_engine()
    if engine is None:
        return False
    try:
        # Import models so metadata is fully populated.
        import models  # noqa: F401

        if create_all:
            Base.metadata.create_all(engine)
        log.info("PostgreSQL schema ready (tables: %s)", ", ".join(sorted(Base.metadata.tables)))
        return True
    except Exception as exc:  # pragma: no cover - server unavailable / schema error
        log.warning("PostgreSQL schema init failed: %s", exc)
        return False


def dispose_engine() -> None:
    """Dispose the connection pool (call on app shutdown)."""
    global _engine, _SessionLocal
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.dispose()
            except Exception:  # pragma: no cover
                pass
        _engine = None
        _SessionLocal = None


def connection_status_label() -> str:
    """Human-readable status label for the UI."""
    st = get_pg_status()
    if st.get("connected"):
        return "● PostgreSQL Connected"
    if st.get("error"):
        return "● PostgreSQL Disconnected"
    return "● Reconnecting..."


def list_tables() -> list[str]:
    """Return the list of public table names (empty when unavailable)."""
    engine = get_engine()
    if engine is None:
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
            ).fetchall()
            return [r[0] for r in rows]
    except Exception as exc:  # pragma: no cover - server unavailable
        log.debug("list_tables failed: %s", exc)
        return []


def backup_postgres(destination: Path, env: Optional[dict[str, str]] = None) -> str:
    """Backup the database via pg_dump into ``destination``.

    Returns an empty string on success or an error message on failure.
    """
    try:
        import subprocess
        cfg = get_db_config()
        cmd = [
            "pg_dump",
            "-h", cfg["host"],
            "-p", str(cfg["port"]),
            "-U", cfg["user"],
            "-Fc",  # custom format for portable restore
            "-f", str(destination),
            "-d", cfg["db"],
        ]
        env2 = dict(env) if env else dict(os.environ)
        env2.setdefault("PGPASSWORD", cfg["password"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env2)
        if result.returncode != 0:
            return result.stderr.strip() or "pg_dump failed"
        return ""
    except FileNotFoundError:
        return "pg_dump not found — install PostgreSQL command-line tools."
    except Exception as exc:  # pragma: no cover
        return str(exc)


def restore_postgres(source: Path, env: Optional[dict[str, str]] = None) -> str:
    """Restore the database from a pg_dump file via pg_restore.

    Returns an empty string on success or an error message on failure.
    """
    try:
        import subprocess
        cfg = get_db_config()
        cmd = [
            "pg_restore",
            "-h", cfg["host"],
            "-p", str(cfg["port"]),
            "-U", cfg["user"],
            "-d", cfg["db"],
            "--clean",
            "--if-exists",
            "--no-owner",
            str(source),
        ]
        env2 = dict(env) if env else dict(os.environ)
        env2.setdefault("PGPASSWORD", cfg["password"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env2)
        # pg_restore can return non-zero for non-fatal warnings; treat errors only.
        if result.returncode != 0 and "error:" in result.stderr.lower():
            return result.stderr.strip()[:4000]
        return ""
    except FileNotFoundError:
        return "pg_restore not found — install PostgreSQL command-line tools."
    except Exception as exc:  # pragma: no cover
        return str(exc)
