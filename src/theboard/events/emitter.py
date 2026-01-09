"""Event emitter abstraction for TheBoard.

Provides pluggable event emission with support for:
- RabbitMQ (production)
- In-memory (testing)
- Null emitter (disabled)

Design Philosophy:
- Abstract interface (EventEmitter protocol)
- Lazy initialization (factory pattern)
- Environment-driven configuration
- Graceful degradation (falls back to null emitter)
"""

import logging
from typing import Protocol

from theboard.config import get_settings
from theboard.events.schemas import (
    BaseEvent,
    ServiceHealthPayload,
    ServiceRegisteredPayload,
)

logger = logging.getLogger(__name__)


class EventEmitter(Protocol):
    """Protocol for event emission.

    Implementations must provide emit() method accepting BaseEvent subclasses.
    """

    def emit(self, event: BaseEvent) -> None:
        """Emit an event to the configured backend.

        Args:
            event: Pydantic event instance (subclass of BaseEvent)

        Raises:
            RuntimeError: If emission fails critically
        """
        ...

    async def emit_service_registered(
        self,
        service_id: str,
        service_name: str,
        version: str,
        capabilities: list[str],
        endpoints: dict[str, str],
    ) -> None:
        """Emit service.registered lifecycle event.

        Args:
            service_id: Unique service identifier
            service_name: Human-readable name
            version: Service version
            capabilities: List of service capabilities
            endpoints: Exposed endpoints
        """
        ...

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
        """Emit service.health lifecycle event.

        Args:
            service_id: Service identifier
            status: Health status
            database: Database connectivity
            redis: Redis connectivity
            bloodbank: Bloodbank connectivity
            uptime_seconds: Service uptime
            details: Optional diagnostic details
        """
        ...


class NullEventEmitter:
    """No-op event emitter for testing or disabled events.

    Logs events at DEBUG level but does not transmit them.
    """

    def emit(self, event: BaseEvent) -> None:
        """Log event but do not transmit.

        Args:
            event: Event to log
        """
        logger.debug("NullEventEmitter: %s - %s", event.event_type, event.model_dump())

    async def emit_service_registered(
        self,
        service_id: str,
        service_name: str,
        version: str,
        capabilities: list[str],
        endpoints: dict[str, str],
    ) -> None:
        """Log service.registered event but do not transmit."""
        logger.debug(
            "NullEventEmitter: service.registered - %s v%s", service_name, version
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
        """Log service.health event but do not transmit."""
        logger.debug(
            "NullEventEmitter: service.health - %s (status=%s)", service_id, status
        )


class InMemoryEventEmitter:
    """In-memory event emitter for testing.

    Stores events in a list for assertion in tests.
    """

    def __init__(self) -> None:
        """Initialize in-memory event store."""
        self.events: list[BaseEvent] = []
        self.lifecycle_events: list[dict] = []

    def emit(self, event: BaseEvent) -> None:
        """Store event in memory.

        Args:
            event: Event to store
        """
        self.events.append(event)
        logger.debug("InMemoryEventEmitter: %s", event.event_type)

    async def emit_service_registered(
        self,
        service_id: str,
        service_name: str,
        version: str,
        capabilities: list[str],
        endpoints: dict[str, str],
    ) -> None:
        """Store service.registered event in memory."""
        self.lifecycle_events.append({
            "type": "service.registered",
            "service_id": service_id,
            "service_name": service_name,
            "version": version,
            "capabilities": capabilities,
            "endpoints": endpoints,
        })
        logger.debug("InMemoryEventEmitter: service.registered")

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
        """Store service.health event in memory."""
        self.lifecycle_events.append({
            "type": "service.health",
            "service_id": service_id,
            "status": status,
            "database": database,
            "redis": redis,
            "bloodbank": bloodbank,
            "uptime_seconds": uptime_seconds,
            "details": details,
        })
        logger.debug("InMemoryEventEmitter: service.health")

    def clear(self) -> None:
        """Clear all stored events (for test isolation)."""
        self.events.clear()
        self.lifecycle_events.clear()

    def get_events(self, event_type: str | None = None) -> list[BaseEvent]:
        """Retrieve stored events, optionally filtered by type.

        Args:
            event_type: Event type to filter (e.g., "meeting.created")

        Returns:
            List of matching events
        """
        if event_type is None:
            return self.events.copy()
        return [e for e in self.events if e.event_type == event_type]


# RabbitMQEventEmitter moved to bloodbank_emitter.py
# Import here for backward compatibility
try:
    from theboard.events.bloodbank_emitter import RabbitMQEventEmitter
except ImportError:
    # Bloodbank not available, provide stub
    class RabbitMQEventEmitter:
        """RabbitMQ emitter stub (Bloodbank not available)."""

        def __init__(self, connection_url: str, exchange: str = "theboard.events") -> None:
            raise NotImplementedError(
                "RabbitMQEventEmitter requires Bloodbank. "
                "Ensure bloodbank repository is cloned and accessible."
            )


# Global emitter instance (lazy initialization)
_emitter: EventEmitter | None = None


def get_event_emitter() -> EventEmitter:
    """Get or create the global event emitter.

    Emitter selection priority:
    1. TESTING mode: InMemoryEventEmitter
    2. EVENT_EMITTER=rabbitmq: RabbitMQEventEmitter (stub in Sprint 2.5)
    3. EVENT_EMITTER=null or unset: NullEventEmitter (default)

    Returns:
        EventEmitter instance (singleton per process)
    """
    global _emitter

    if _emitter is not None:
        return _emitter

    config = get_settings()

    # Testing mode: always use in-memory emitter
    if config.testing:
        logger.info("Event emitter: InMemoryEventEmitter (testing mode)")
        _emitter = InMemoryEventEmitter()
        return _emitter

    # Production mode: check event_emitter setting
    emitter_type = config.event_emitter

    if emitter_type == "rabbitmq":
        try:
            # Use Bloodbank RabbitMQ emitter
            _emitter = RabbitMQEventEmitter(
                connection_url=config.rabbitmq_url,
                exchange="events"  # Bloodbank uses 'events' exchange
            )
            logger.info("Event emitter: RabbitMQEventEmitter (Bloodbank integration)")
        except (NotImplementedError, RuntimeError) as e:
            logger.warning(f"RabbitMQEventEmitter unavailable: {e}")
            logger.warning("Falling back to NullEventEmitter")
            _emitter = NullEventEmitter()
    else:
        logger.info("Event emitter: NullEventEmitter (events disabled)")
        _emitter = NullEventEmitter()

    return _emitter


def reset_event_emitter() -> None:
    """Reset global emitter (for testing isolation).

    Forces lazy reinitialization on next get_event_emitter() call.
    """
    global _emitter
    _emitter = None
