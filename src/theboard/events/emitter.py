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
from theboard.events.schemas import BaseEvent

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


class InMemoryEventEmitter:
    """In-memory event emitter for testing.

    Stores events in a list for assertion in tests.
    """

    def __init__(self) -> None:
        """Initialize in-memory event store."""
        self.events: list[BaseEvent] = []

    def emit(self, event: BaseEvent) -> None:
        """Store event in memory.

        Args:
            event: Event to store
        """
        self.events.append(event)
        logger.debug("InMemoryEventEmitter: %s", event.event_type)

    def clear(self) -> None:
        """Clear all stored events (for test isolation)."""
        self.events.clear()

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
