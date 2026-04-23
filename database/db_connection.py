"""
PostgreSQL connection pool using SQLAlchemy.
All other modules import `get_engine` and `get_session` from here.
"""

import os
from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


def build_database_url() -> str:
    """Construct the PostgreSQL DSN from environment variables."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "healthcare_db")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


# Module-level engine — created once, reused everywhere
_engine = None


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        url = build_database_url()
        _engine = create_engine(
            url,
            pool_pre_ping=True,   # drop stale connections automatically
            pool_size=5,
            max_overflow=10,
            echo=False,           # set True to log all SQL (debugging)
        )
    return _engine


def get_session_factory():
    """Return a bound sessionmaker."""
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that yields a SQLAlchemy Session and handles
    commit / rollback / close automatically.

    Usage:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    """
    SessionLocal = get_session_factory()
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_connection() -> bool:
    """Return True if the database is reachable, False otherwise."""
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        print(f"[DB] Connection test failed: {exc}")
        return False