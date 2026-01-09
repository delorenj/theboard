"""FastAPI service layer for TheBoard 33GOD integration."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, status
from pydantic import BaseModel, Field

from theboard.config import settings
from theboard.database import get_sync_db
from theboard.events.emitter import get_event_emitter
from theboard.utils.redis_manager import RedisManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TheBoard Service API",
    description="Multi-agent brainstorming simulation system",
    version="2.1.0",
)

# Track service start time for uptime calculation
SERVICE_START_TIME = datetime.now(timezone.utc)


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Service health status"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp",
    )
    version: str = Field(default="2.1.0", description="Service version")
    database: Literal["connected", "disconnected", "error"] = Field(
        description="Database connectivity status"
    )
    redis: Literal["connected", "disconnected", "error"] = Field(
        description="Redis connectivity status"
    )
    bloodbank: Literal["connected", "disconnected", "disabled"] = Field(
        description="Bloodbank event bus status"
    )
    details: dict[str, str] | None = Field(
        default=None, description="Additional diagnostic details"
    )


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for service monitoring.

    Returns service health status and connectivity checks for:
    - PostgreSQL database
    - Redis cache
    - Bloodbank event bus

    Status codes:
    - healthy: All systems operational
    - degraded: One non-critical system unavailable
    - unhealthy: Critical system unavailable
    """
    details = {}

    # Check database connectivity
    db_status: Literal["connected", "disconnected", "error"] = "disconnected"
    try:
        from sqlalchemy import text
        with get_sync_db() as db:
            db.execute(text("SELECT 1"))
            db_status = "connected"
            details["database"] = f"Connected to {settings.postgres_db}"
    except Exception as e:
        db_status = "error"
        details["database_error"] = str(e)
        logger.error(f"Database health check failed: {e}")

    # Check Redis connectivity
    redis_status: Literal["connected", "disconnected", "error"] = "disconnected"
    try:
        redis_manager = RedisManager()
        redis_manager.client.ping()
        redis_status = "connected"
        details["redis"] = f"Connected to {settings.redis_host}:{settings.redis_port}"
    except Exception as e:
        redis_status = "error"
        details["redis_error"] = str(e)
        logger.error(f"Redis health check failed: {e}")

    # Check Bloodbank connectivity
    bloodbank_status: Literal["connected", "disconnected", "disabled"] = "disabled"
    if settings.event_emitter == "rabbitmq":
        try:
            emitter = get_event_emitter()
            # If emitter is not NullEmitter, consider it connected
            if emitter.__class__.__name__ != "NullEventEmitter":
                bloodbank_status = "connected"
                details["bloodbank"] = f"Connected to {settings.rabbitmq_url}"
            else:
                bloodbank_status = "disconnected"
        except Exception as e:
            bloodbank_status = "disconnected"
            details["bloodbank_error"] = str(e)
            logger.error(f"Bloodbank health check failed: {e}")
    else:
        details["bloodbank"] = f"Event emitter disabled (mode: {settings.event_emitter})"

    # Determine overall health status
    if db_status == "connected" and redis_status == "connected":
        if bloodbank_status in ("connected", "disabled"):
            overall_status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        else:
            overall_status = "degraded"  # Bloodbank optional
    elif db_status == "connected" or redis_status == "connected":
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return HealthCheckResponse(
        status=overall_status,
        database=db_status,
        redis=redis_status,
        bloodbank=bloodbank_status,
        details=details,
    )


@app.get("/", tags=["Info"])
async def root() -> dict[str, str]:
    """Root endpoint with service information."""
    return {
        "service": "TheBoard",
        "version": "2.1.0",
        "description": "Multi-agent brainstorming simulation system",
        "health": "/health",
        "docs": "/docs",
    }


@app.on_event("startup")
async def startup_event():
    """Emit service.registered event when FastAPI starts."""
    logger.info("TheBoard service starting up...")

    emitter = get_event_emitter()

    try:
        await emitter.emit_service_registered(
            service_id="theboard-producer",
            service_name="TheBoard",
            version="2.1.0",
            capabilities=[
                "multi-agent-brainstorming",
                "context-compression",
                "convergence-detection",
            ],
            endpoints={
                "health": "http://localhost:8000/health",
                "docs": "http://localhost:8000/docs",
            },
        )
        logger.info("Service registration event emitted")
    except Exception as e:
        logger.error(f"Failed to emit service.registered event: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Handle graceful shutdown."""
    logger.info("TheBoard service shutting down...")
    # Future: Emit service.shutdown event


# Health check background task (optional enhancement)
async def periodic_health_check():
    """Emit health events every 60 seconds."""
    while True:
        await asyncio.sleep(60)

        try:
            # Call health endpoint and emit event
            health_response = await health_check()
            emitter = get_event_emitter()

            uptime = (datetime.now(timezone.utc) - SERVICE_START_TIME).total_seconds()

            await emitter.emit_service_health(
                service_id="theboard-producer",
                status=health_response.status,
                database=health_response.database,
                redis=health_response.redis,
                bloodbank=health_response.bloodbank,
                uptime_seconds=int(uptime),
                details=health_response.details,
            )
        except Exception as e:
            logger.error(f"Health check emission failed: {e}")


# Uncomment to enable periodic health events
# @app.on_event("startup")
# async def start_health_checker():
#     asyncio.create_task(periodic_health_check())
