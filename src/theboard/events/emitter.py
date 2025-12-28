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

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Protocol

import aiormq
from aiormq.abc import DeliveredMessage

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


class RabbitMQEventEmitter:
    """RabbitMQ event emitter for production event streaming.

    Features (Sprint 4 Story 12):
    - aiormq connection management with lazy initialization
    - Topic exchange declaration (theboard.events)
    - Event publishing with routing keys: meeting.{event_type}
    - Graceful degradation (logs errors but doesn't crash)
    - JSON serialization with Pydantic model_dump

    Connection Management:
    - Lazy connection on first emit() call
    - Auto-reconnect on connection failures
    - No connection pooling (single persistent connection per emitter)
    """

    def __init__(self, connection_url: str, exchange: str = "theboard.events") -> None:
        """Initialize RabbitMQ emitter with lazy connection.

        Args:
            connection_url: AMQP connection URL (e.g., amqp://user:pass@localhost:5672/)
            exchange: Exchange name for event routing (default: theboard.events)
        """
        self.connection_url = connection_url
        self.exchange = exchange
        self._connection: aiormq.Connection | None = None
        self._channel: aiormq.Channel | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        logger.info(
            "RabbitMQEventEmitter initialized: exchange=%s, url=%s",
            self.exchange,
            connection_url.split("@")[-1],  # Hide credentials
        )

    async def _ensure_connected(self) -> aiormq.Channel:
        """Ensure RabbitMQ connection and channel are ready.

        Lazy initialization: Connects on first emit() call.
        Reconnects automatically if connection is lost.

        Returns:
            aiormq.Channel for publishing messages

        Raises:
            RuntimeError: If connection fails after retries
        """
        # Check if already connected
        if self._connection and not self._connection.is_closed and self._channel:
            return self._channel

        try:
            # Establish connection
            logger.info("Connecting to RabbitMQ: %s", self.connection_url.split("@")[-1])
            self._connection = await aiormq.connect(self.connection_url)
            self._channel = await self._connection.channel()

            # Declare topic exchange (idempotent)
            await self._channel.exchange_declare(
                exchange=self.exchange,
                exchange_type="topic",
                durable=True,
            )

            logger.info("RabbitMQ connected: exchange=%s declared", self.exchange)
            return self._channel

        except Exception as e:
            logger.error("RabbitMQ connection failed: %s", e)
            raise RuntimeError(f"Failed to connect to RabbitMQ: {e}") from e

    def emit(self, event: BaseEvent) -> None:
        """Emit event to RabbitMQ topic exchange.

        Routing key pattern: meeting.{event_type}
        Example: meeting.round_completed, meeting.human.input.needed

        This is a synchronous wrapper around async _async_emit().
        Creates event loop if needed (for CLI/synchronous contexts).

        Args:
            event: Pydantic event instance to publish

        Raises:
            RuntimeError: If RabbitMQ publish fails critically
        """
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running - create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async emit in event loop
        try:
            if loop.is_running():
                # If loop is running, schedule as task
                asyncio.create_task(self._async_emit(event))
            else:
                # If loop is not running, run until complete
                loop.run_until_complete(self._async_emit(event))
        except Exception as e:
            logger.error("Failed to emit event %s: %s", event.event_type, e)
            # Graceful degradation: log error but don't crash
            # In production, events are best-effort

    async def _async_emit(self, event: BaseEvent) -> None:
        """Async event emission to RabbitMQ.

        Args:
            event: Pydantic event instance to publish
        """
        try:
            # Ensure connection
            channel = await self._ensure_connected()

            # Generate routing key: meeting.{event_type}
            # Examples: meeting.round_completed, meeting.human.input.needed
            routing_key = event.event_type

            # Serialize event to JSON
            message_body = event.model_dump_json().encode("utf-8")

            # Publish to exchange
            await channel.basic_publish(
                body=message_body,
                exchange=self.exchange,
                routing_key=routing_key,
                properties=aiormq.spec.Basic.Properties(
                    content_type="application/json",
                    delivery_mode=2,  # Persistent
                ),
            )

            logger.debug(
                "Event published: %s -> %s.%s", event.event_type, self.exchange, routing_key
            )

        except Exception as e:
            logger.error("Failed to publish event %s: %s", event.event_type, e)
            # Reset connection on failure (will reconnect on next emit)
            self._connection = None
            self._channel = None

    async def close(self) -> None:
        """Close RabbitMQ connection gracefully.

        Should be called on application shutdown.
        """
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("RabbitMQ connection closed")
            self._connection = None
            self._channel = None


# Global emitter instance (lazy initialization)
_emitter: EventEmitter | None = None


def get_event_emitter() -> EventEmitter:
    """Get or create the global event emitter.

    Emitter selection priority:
    1. TESTING mode: InMemoryEventEmitter
    2. EVENT_EMITTER=rabbitmq: RabbitMQEventEmitter (Sprint 4 Story 12)
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
        logger.info("Event emitter: RabbitMQEventEmitter (production)")
        _emitter = RabbitMQEventEmitter(
            connection_url=config.rabbitmq_url,
            exchange="theboard.events",
        )
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
