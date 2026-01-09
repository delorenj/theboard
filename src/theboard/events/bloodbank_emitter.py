"""RabbitMQ event emitter using Bloodbank Publisher.

This module implements the EventEmitter protocol using Bloodbank's Publisher
class to publish events to the RabbitMQ event bus.

Architecture:
- Wraps theboard events in Bloodbank's EventEnvelope format
- Maps theboard.events.schemas events to bloodbank routing keys
- Handles async publication within sync contexts using asyncio
- Provides lifecycle management for RabbitMQ connections
"""

import asyncio
import logging
import socket
from pathlib import Path
from typing import Any

from theboard.config import get_settings
from theboard.events.schemas import (
    BaseEvent,
    ServiceHealthPayload,
    ServiceRegisteredPayload,
)

# Lazy import bloodbank components
_bloodbank_available = False
_Publisher = None
_create_envelope = None
_TriggerType = None
_Source = None

try:
    # Add bloodbank to path if needed
    import sys
    bloodbank_path = Path.home() / "code" / "bloodbank" / "trunk-main"
    if bloodbank_path.exists() and str(bloodbank_path) not in sys.path:
        sys.path.insert(0, str(bloodbank_path))

    from event_producers.rabbit import Publisher as _Publisher_class
    from event_producers.events.envelope import create_envelope as _create_envelope_func
    from event_producers.events.base import TriggerType as _TriggerType_enum, Source as _Source_class

    _Publisher = _Publisher_class
    _create_envelope = _create_envelope_func
    _TriggerType = _TriggerType_enum
    _Source = _Source_class
    _bloodbank_available = True
except ImportError as e:
    logging.getLogger(__name__).warning(
        f"Bloodbank components not available: {e}. "
        "RabbitMQEventEmitter will not be functional."
    )

logger = logging.getLogger(__name__)


def _event_to_routing_key(event: BaseEvent) -> str:
    """Convert theboard event to bloodbank routing key.

    Maps:
    - meeting.created -> theboard.meeting.created
    - meeting.started -> theboard.meeting.started
    - etc.

    Args:
        event: theboard event instance

    Returns:
        Bloodbank-compatible routing key with 'theboard.' prefix
    """
    return f"theboard.{event.event_type}"


def _create_bloodbank_envelope(event: BaseEvent) -> dict[str, Any]:
    """Create Bloodbank EventEnvelope from theboard event.

    Args:
        event: theboard event instance

    Returns:
        EventEnvelope as dict ready for publishing
    """
    if not _bloodbank_available:
        raise RuntimeError("Bloodbank components not available")

    # Create source metadata
    source = _Source(
        host=socket.gethostname(),
        type=_TriggerType.MANUAL,  # Service lifecycle events
        app="theboard",
        meta={"version": "1.0.0"}
    )

    # Create envelope with theboard event as payload
    envelope = _create_envelope(
        event_type=_event_to_routing_key(event),
        payload=event.model_dump(),  # Serialize Pydantic event
        source=source
    )

    return envelope.model_dump()


class RabbitMQEventEmitter:
    """RabbitMQ event emitter using Bloodbank Publisher.

    Integrates with Bloodbank's event bus to publish theboard events.
    Handles async operations within sync contexts using asyncio event loop.

    Lifecycle:
    - Lazy connection on first emit()
    - Connection pooled for efficiency
    - Graceful shutdown on close()
    """

    def __init__(self, connection_url: str, exchange: str = "events") -> None:
        """Initialize RabbitMQ emitter.

        Args:
            connection_url: AMQP connection URL (from settings)
            exchange: Exchange name (bloodbank uses 'events')
        """
        if not _bloodbank_available:
            raise RuntimeError(
                "Bloodbank components not available. "
                "Ensure bloodbank is installed and accessible."
            )

        self.connection_url = connection_url
        self.exchange = exchange
        self._publisher: _Publisher | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False

        logger.info(
            f"RabbitMQEventEmitter: Initialized (exchange={exchange})"
        )

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for async operations.

        Returns:
            Running event loop
        """
        try:
            # Try to get current running loop
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop, create new one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            return self._loop

    async def _start_async(self) -> None:
        """Start publisher asynchronously (internal)."""
        if self._started:
            return

        self._publisher = _Publisher(enable_correlation_tracking=False)
        await self._publisher.start()
        self._started = True

        logger.info("RabbitMQEventEmitter: Connected to Bloodbank")

    def _start(self) -> None:
        """Start publisher (sync wrapper)."""
        if self._started:
            return

        loop = self._get_or_create_loop()

        # Run start coroutine
        if loop.is_running():
            # Already in async context, schedule task
            asyncio.create_task(self._start_async())
        else:
            # Sync context, run until complete
            loop.run_until_complete(self._start_async())

    async def _emit_async(self, event: BaseEvent) -> None:
        """Emit event asynchronously (internal).

        Args:
            event: theboard event to publish
        """
        if not self._started:
            await self._start_async()

        # Create Bloodbank envelope
        envelope = _create_bloodbank_envelope(event)
        routing_key = _event_to_routing_key(event)

        # Publish to Bloodbank
        await self._publisher.publish(
            routing_key=routing_key,
            body=envelope
        )

        logger.debug(
            f"RabbitMQEventEmitter: Published {routing_key} (meeting_id={event.meeting_id})"
        )

    def emit(self, event: BaseEvent) -> None:
        """Emit event to Bloodbank.

        Args:
            event: theboard event to publish

        Raises:
            RuntimeError: If emission fails
        """
        try:
            loop = self._get_or_create_loop()

            if loop.is_running():
                # Already in async context, schedule task
                asyncio.create_task(self._emit_async(event))
            else:
                # Sync context, run until complete
                loop.run_until_complete(self._emit_async(event))

        except Exception as e:
            logger.error(f"RabbitMQEventEmitter: Failed to emit {event.event_type}: {e}")
            raise RuntimeError(f"Event emission failed: {e}") from e

    async def _close_async(self) -> None:
        """Close publisher asynchronously (internal)."""
        if self._publisher:
            await self._publisher.close()
            self._publisher = None
        self._started = False

        logger.info("RabbitMQEventEmitter: Closed connection")

    def close(self) -> None:
        """Close publisher and clean up resources (sync wrapper)."""
        if not self._started:
            return

        loop = self._get_or_create_loop()

        if loop.is_running():
            asyncio.create_task(self._close_async())
        else:
            loop.run_until_complete(self._close_async())

        # Clean up event loop if we created it
        if self._loop and not self._loop.is_running():
            self._loop.close()
            self._loop = None

    async def emit_service_registered(
        self,
        service_id: str,
        service_name: str,
        version: str,
        capabilities: list[str],
        endpoints: dict[str, str],
    ) -> None:
        """Emit service.registered lifecycle event to Bloodbank.

        Args:
            service_id: Unique service identifier
            service_name: Human-readable name
            version: Service version
            capabilities: List of service capabilities
            endpoints: Exposed endpoints
        """
        if not self._started:
            await self._start_async()

        payload = ServiceRegisteredPayload(
            service_id=service_id,
            service_name=service_name,
            version=version,
            capabilities=capabilities,
            endpoints=endpoints,
        )

        # Create Bloodbank envelope
        source = _Source(
            host=socket.gethostname(),
            type=_TriggerType.MANUAL,
            app="theboard",
            meta={"version": version}
        )

        envelope = _create_envelope(
            event_type="theboard.service.registered",
            payload=payload.model_dump(),
            source=source
        )

        # Publish to Bloodbank
        await self._publisher.publish(
            routing_key="theboard.service.registered",
            body=envelope.model_dump()
        )

        logger.info(f"RabbitMQEventEmitter: Published service.registered ({service_id})")

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
        """Emit service.health lifecycle event to Bloodbank.

        Args:
            service_id: Service identifier
            status: Health status
            database: Database connectivity
            redis: Redis connectivity
            bloodbank: Bloodbank connectivity
            uptime_seconds: Service uptime
            details: Optional diagnostic details
        """
        if not self._started:
            await self._start_async()

        payload = ServiceHealthPayload(
            service_id=service_id,
            status=status,
            database=database,
            redis=redis,
            bloodbank=bloodbank,
            uptime_seconds=uptime_seconds,
            details=details,
        )

        # Create Bloodbank envelope
        source = _Source(
            host=socket.gethostname(),
            type=_TriggerType.MANUAL,
            app="theboard",
            meta={"uptime_seconds": uptime_seconds}
        )

        envelope = _create_envelope(
            event_type="theboard.service.health",
            payload=payload.model_dump(),
            source=source
        )

        # Publish to Bloodbank
        await self._publisher.publish(
            routing_key="theboard.service.health",
            body=envelope.model_dump()
        )

        logger.debug(f"RabbitMQEventEmitter: Published service.health ({service_id}, {status})")
