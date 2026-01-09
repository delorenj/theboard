# TheBoard 33GOD Integration - Phase 1 Technical Specification

**Version**: 1.0
**Status**: Implementation Ready
**Effort**: XS (S for development, XS for review)
**Target**: Week 1 completion

## Executive Summary

Phase 1 establishes TheBoard as a registered, discoverable service in the 33GOD ecosystem. This foundational integration enables health monitoring, service discovery via Candybar, and proper event lifecycle management without disrupting existing CLI functionality.

## Goals

**Primary Objectives:**
1. Register TheBoard in 33GOD service registry
2. Implement health check endpoint for monitoring
3. Emit service lifecycle events to Bloodbank
4. Enable Candybar visualization of TheBoard node

**Success Criteria:**
- TheBoard appears in Candybar service graph
- Health endpoint returns 200 OK with service status
- Service registration event captured by Bloodbank
- Existing CLI commands function unchanged

## Architecture Changes

### Component Diagram

```mermaid
graph TD
    CLI[TheBoard CLI] --> API[FastAPI Service]
    API --> Health[/health Endpoint]
    API --> Registry[Service Registration]
    Registry --> BB[Bloodbank Event Bus]
    BB --> CB[Candybar Dashboard]

    Health -.->|Health Check| Monitor[Monitoring System]
    Registry -.->|service.registered| BB
```

### System Context

**Before Phase 1:**
- TheBoard: Standalone CLI application
- Bloodbank: Receives meeting events but TheBoard not registered as service
- Candybar: No visibility into TheBoard status

**After Phase 1:**
- TheBoard: Registered service with health endpoint
- Bloodbank: Tracks TheBoard lifecycle (startup, shutdown, health)
- Candybar: Displays TheBoard node in service graph with live status

## Technical Requirements

### REQ-001: Service Registry Entry

**File**: `/home/delorenj/code/33GOD/services/registry.yaml`

**Changes Required:**

1. Add TheBoard producer entry:
```yaml
theboard-producer:
  name: "theboard-producer"
  description: "Multi-agent brainstorming simulation system (event producer)"
  type: "event-producer"
  status: "active"
  owner: "33GOD"
  tags: ["theboard", "brainstorming", "multi-agent"]
  produces:
    - "theboard.service.registered"
    - "theboard.service.health"
    - "theboard.meeting.created"
    - "theboard.meeting.started"
    - "theboard.meeting.round_completed"
    - "theboard.meeting.comment_extracted"
    - "theboard.meeting.converged"
    - "theboard.meeting.completed"
    - "theboard.meeting.failed"
```

2. Update existing `theboard-sync` consumer entry:
```yaml
theboard-sync:
  name: "theboard-sync"
  description: "Syncs TheBoard meeting events for visualization"
  type: "event-consumer"
  queue_name: "theboard_sync_queue"
  routing_keys:
    - "theboard.meeting.#"
    - "theboard.service.#"
  status: "planned"  # Will be implemented in Phase 4
  owner: "33GOD"
  tags: ["theboard", "sync", "observability"]
```

3. Add event subscription mappings:
```yaml
event_subscriptions:
  theboard.service.registered:
    - "service-registry-monitor"  # Future service
  theboard.service.health:
    - "service-registry-monitor"
  theboard.meeting.created:
    - "theboard-sync"
  theboard.meeting.started:
    - "theboard-sync"
  # ... (other meeting events)
```

4. Add to topology:
```yaml
topology:
  event_producers:
    - "theboard-producer"

  event_consumers:
    - "theboard-sync"
```

**Validation:**
- YAML parses without errors
- Candybar can load and visualize topology
- No duplicate service IDs

---

### REQ-002: Health Check Endpoint

**File**: `/home/delorenj/code/33GOD/theboard/trunk-main/src/theboard/api.py` (NEW)

**Implementation:**

```python
"""FastAPI service layer for TheBoard 33GOD integration."""

import logging
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, status
from pydantic import BaseModel, Field

from theboard.config import settings
from theboard.database import get_sync_db
from theboard.utils.redis_manager import RedisManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TheBoard Service API",
    description="Multi-agent brainstorming simulation system",
    version="2.1.0",
)


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
        with get_sync_db() as db:
            db.execute("SELECT 1")
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
            from theboard.events.emitter import get_event_emitter
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
```

**Deployment:**

Add to `compose.yml`:
```yaml
services:
  theboard:
    # ... existing config ...
    ports:
      - "8000:8000"  # Expose health endpoint
    command: >
      sh -c "
        alembic upgrade head &&
        uvicorn theboard.api:app --host 0.0.0.0 --port 8000 --reload
      "
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

**Testing:**

```bash
# Local testing
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2026-01-09T12:00:00Z",
  "version": "2.1.0",
  "database": "connected",
  "redis": "connected",
  "bloodbank": "connected",
  "details": {
    "database": "Connected to theboard",
    "redis": "Connected to localhost:6380",
    "bloodbank": "Connected to amqp://theboard:***@localhost:5673/"
  }
}
```

**Validation:**
- Health endpoint returns 200 OK when all systems healthy
- Returns 200 with "degraded" status if Redis/Bloodbank unavailable
- Returns 200 with "unhealthy" if database unavailable
- Docker healthcheck passes after service startup

---

### REQ-003: Service Lifecycle Events

**File**: `/home/delorenj/code/33GOD/theboard/trunk-main/src/theboard/events/schemas.py`

**Add New Event Schemas:**

```python
"""Event schemas for TheBoard Bloodbank integration."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Service Lifecycle Events (NEW - Phase 1)
# ============================================================================

class ServiceRegisteredPayload(BaseModel):
    """Payload for service.registered event."""

    service_id: str = Field(description="Service identifier")
    service_name: str = Field(description="Human-readable service name")
    version: str = Field(description="Service version")
    capabilities: list[str] = Field(
        description="List of capabilities this service provides"
    )
    endpoints: dict[str, str] = Field(
        description="Exposed endpoints (e.g., {'health': 'http://host:port/health'})"
    )


class ServiceHealthPayload(BaseModel):
    """Payload for service.health event."""

    service_id: str = Field(description="Service identifier")
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall health status"
    )
    database: Literal["connected", "disconnected", "error"]
    redis: Literal["connected", "disconnected", "error"]
    bloodbank: Literal["connected", "disconnected", "disabled"]
    uptime_seconds: int = Field(description="Service uptime in seconds")
    details: dict[str, str] | None = None


# Existing meeting events remain unchanged
# ...
```

**File**: `/home/delorenj/code/33GOD/theboard/trunk-main/src/theboard/events/emitter.py`

**Add Lifecycle Event Methods:**

```python
"""Event emitter interface for Bloodbank integration."""

import logging
from datetime import datetime, timezone

from theboard.events.schemas import (
    ServiceRegisteredPayload,
    ServiceHealthPayload,
)

logger = logging.getLogger(__name__)


class EventEmitter:
    """Base event emitter interface."""

    async def emit_service_registered(
        self,
        service_id: str,
        service_name: str,
        version: str,
        capabilities: list[str],
        endpoints: dict[str, str],
    ) -> None:
        """
        Emit service.registered event when TheBoard starts up.

        Args:
            service_id: Unique service identifier (e.g., "theboard-producer")
            service_name: Human-readable name (e.g., "TheBoard")
            version: Service version (e.g., "2.1.0")
            capabilities: List of capabilities (e.g., ["multi-agent-brainstorming"])
            endpoints: Exposed endpoints (e.g., {"health": "http://localhost:8000/health"})
        """
        payload = ServiceRegisteredPayload(
            service_id=service_id,
            service_name=service_name,
            version=version,
            capabilities=capabilities,
            endpoints=endpoints,
        )

        await self._emit_event(
            event_type="theboard.service.registered",
            payload=payload.model_dump(),
        )

    async def emit_service_health(
        self,
        service_id: str,
        status: str,
        database: str,
        redis: str,
        bloodbank: str,
        uptime_seconds: int,
        details: dict[str, str] | None = None,
    ) -> None:
        """
        Emit service.health event for periodic health monitoring.

        Args:
            service_id: Service identifier
            status: Overall health status
            database: Database connectivity status
            redis: Redis connectivity status
            bloodbank: Bloodbank connectivity status
            uptime_seconds: Service uptime in seconds
            details: Optional diagnostic details
        """
        payload = ServiceHealthPayload(
            service_id=service_id,
            status=status,
            database=database,
            redis=redis,
            bloodbank=bloodbank,
            uptime_seconds=uptime_seconds,
            details=details,
        )

        await self._emit_event(
            event_type="theboard.service.health",
            payload=payload.model_dump(),
        )

    async def _emit_event(self, event_type: str, payload: dict) -> None:
        """Internal method to emit events (override in subclasses)."""
        raise NotImplementedError


# RabbitMQEventEmitter and NullEventEmitter implementations remain...
```

**File**: `/home/delorenj/code/33GOD/theboard/trunk-main/src/theboard/api.py`

**Add Startup Event Handler:**

```python
"""FastAPI service layer - startup/shutdown lifecycle."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import FastAPI

from theboard.events.emitter import get_event_emitter

logger = logging.getLogger(__name__)

app = FastAPI(...)

# Track service start time for uptime calculation
SERVICE_START_TIME = datetime.now(timezone.utc)


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
                "comment-extraction",
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
```

**Validation:**
- Service emits `theboard.service.registered` on startup
- Event captured by Bloodbank
- Candybar receives and displays event
- CLI commands still work (FastAPI runs alongside)

---

## Implementation Plan

### Step 1: Update Service Registry (30 min)

```bash
cd /home/delorenj/code/33GOD
nano services/registry.yaml
# Add theboard-producer entry
# Update theboard-sync entry
# Add event_subscriptions mappings
# Add topology entries

# Validate YAML
python -c "import yaml; yaml.safe_load(open('services/registry.yaml'))"
```

### Step 2: Create FastAPI Layer (2 hours)

```bash
cd /home/delorenj/code/33GOD/theboard/trunk-main

# Create API module
touch src/theboard/api.py

# Add dependencies to pyproject.toml
uv add "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0"

# Implement health endpoint
# Implement lifecycle event handlers
# Add startup/shutdown hooks

# Test locally
uv run uvicorn theboard.api:app --reload
curl http://localhost:8000/health
```

### Step 3: Update Event Schemas (1 hour)

```bash
# Edit event schemas
nano src/theboard/events/schemas.py
# Add ServiceRegisteredPayload
# Add ServiceHealthPayload

# Edit event emitter
nano src/theboard/events/emitter.py
# Add emit_service_registered()
# Add emit_service_health()

# Test event emission
uv run python -c "
from theboard.events.emitter import get_event_emitter
import asyncio

async def test():
    emitter = get_event_emitter()
    await emitter.emit_service_registered(
        service_id='test',
        service_name='Test',
        version='1.0',
        capabilities=[],
        endpoints={}
    )

asyncio.run(test())
"
```

### Step 4: Update Docker Compose (30 min)

```bash
# Edit compose.yml
nano compose.yml
# Add port mapping (8000:8000)
# Update command to run uvicorn
# Add healthcheck configuration

# Rebuild and test
docker compose build theboard
docker compose up -d theboard
docker compose logs -f theboard

# Verify health endpoint
curl http://localhost:8000/health
```

### Step 5: Integration Testing (1 hour)

```bash
# Start full stack
docker compose up -d

# Verify Bloodbank receives events
cd /home/delorenj/code/33GOD/bloodbank/trunk-main
uv run python -m event_producers.watch

# In another terminal, restart TheBoard
docker compose restart theboard

# Should see: theboard.service.registered event in watch output

# Verify health checks
watch -n 5 'curl -s http://localhost:8000/health | jq'

# Verify CLI still works
docker compose exec theboard uv run board create --topic "test" --max-rounds 1
```

---

## Testing Strategy

### Unit Tests

**File**: `/home/delorenj/code/33GOD/theboard/trunk-main/tests/unit/test_api_health.py`

```python
"""Unit tests for health check endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from theboard.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_all_systems_healthy(client, monkeypatch):
    """Test health check when all systems are healthy."""
    # Mock database connection
    mock_db = MagicMock()
    mock_db.execute.return_value = None

    # Mock Redis connection
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    with patch('theboard.api.get_sync_db', return_value=mock_db):
        with patch('theboard.api.RedisManager') as mock_redis_manager:
            mock_redis_manager.return_value.client = mock_redis

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["database"] == "connected"
            assert data["redis"] == "connected"


def test_health_check_database_unavailable(client):
    """Test health check when database is unavailable."""
    with patch('theboard.api.get_sync_db', side_effect=Exception("DB error")):
        response = client.get("/health")

        assert response.status_code == 200  # Still returns 200, but status unhealthy
        data = response.json()
        assert data["status"] in ("degraded", "unhealthy")
        assert data["database"] == "error"
```

### Integration Tests

**File**: `/home/delorenj/code/33GOD/theboard/trunk-main/tests/integration/test_service_lifecycle.py`

```python
"""Integration tests for service lifecycle events."""

import pytest
import asyncio
from datetime import datetime

from theboard.events.emitter import get_event_emitter, reset_event_emitter


@pytest.mark.asyncio
async def test_service_registered_event_emission():
    """Test that service.registered event is emitted correctly."""
    emitter = get_event_emitter()

    # This test assumes Bloodbank is running
    await emitter.emit_service_registered(
        service_id="theboard-test",
        service_name="TheBoard Test",
        version="2.1.0",
        capabilities=["test"],
        endpoints={"health": "http://localhost:8000/health"},
    )

    # If no exception, emission succeeded
    assert True


@pytest.fixture(autouse=True)
def reset_emitter():
    """Reset event emitter after each test."""
    yield
    reset_event_emitter()
```

---

## Rollback Strategy

If Phase 1 integration causes issues:

1. **Revert Registry Changes:**
   ```bash
   cd /home/delorenj/code/33GOD
   git checkout services/registry.yaml
   ```

2. **Disable FastAPI Service:**
   ```yaml
   # In compose.yml, revert to original command:
   command: tail -f /dev/null
   ```

3. **Remove Port Mapping:**
   ```yaml
   # Remove from compose.yml:
   ports:
     - "8000:8000"
   ```

4. **Keep Event Schemas:**
   - New event schemas don't break existing functionality
   - Can remain in codebase for future use

---

## Success Metrics

**Phase 1 Complete When:**

1. ✅ TheBoard entry exists in `services/registry.yaml`
2. ✅ Health endpoint responds at `http://localhost:8000/health`
3. ✅ `theboard.service.registered` event appears in Bloodbank watch
4. ✅ Candybar displays TheBoard node (once Candybar supports registry visualization)
5. ✅ All existing CLI commands function unchanged
6. ✅ Docker healthcheck passes
7. ✅ Integration tests pass

**Quality Gates:**

- Health endpoint returns <500ms response time
- Service starts successfully 99% of the time
- No breaking changes to existing CLI interface
- All unit tests pass (70% coverage target)

---

## Next Steps (Phase 2)

After Phase 1 completion:

1. Implement event consumer for meeting triggers (`meeting.trigger` → auto-create meeting)
2. Add granular agent-level events (`agent.selected`, `agent.responded`)
3. Add compression event with metrics
4. Add per-round cost tracking events

See `33GOD_INTEGRATION_PHASE2.md` for details.

---

## References

- TheBoard README: `/home/delorenj/code/33GOD/theboard/trunk-main/README.md`
- Bloodbank Integration: `/home/delorenj/code/33GOD/theboard/trunk-main/BLOODBANK_INTEGRATION.md`
- Service Registry: `/home/delorenj/code/33GOD/services/registry.yaml`
- 33GOD Architecture: `/home/delorenj/code/33GOD/docs/ARCHITECTURE.md`
- Services Guide: `/home/delorenj/code/33GOD/docs/SERVICES_GUIDE.md`
