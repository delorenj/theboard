"""Database connection and session management."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from theboard.config import settings
from theboard.models.base import Base

# Sync engine (for migrations and CLI operations)
sync_engine = create_engine(
    settings.database_url_str,
    echo=settings.debug,
    pool_pre_ping=True,
)

# Async engine (for application use)
async_engine = create_async_engine(
    settings.database_url_str.replace("postgresql+psycopg", "postgresql+psycopg_async"),
    echo=settings.debug,
    poolclass=NullPool,
    pool_pre_ping=True,
)

# Session factories
SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """Get synchronous database session context manager."""
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get asynchronous database session context manager."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def init_db() -> None:
    """Initialize database tables (for development only)."""
    Base.metadata.create_all(bind=sync_engine)


async def init_db_async() -> None:
    """Initialize database tables asynchronously (for development only)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await async_engine.dispose()
