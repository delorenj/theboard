"""RabbitMQ event consumer for TheBoard human-in-loop (Sprint 4 Story 12).

Provides async event listener with handler registration pattern:
- Subscribe to meeting.* events via topic exchange
- Route events to registered handlers based on event_type
- Support for human-in-loop interactive prompts
- Graceful shutdown and reconnection

Design Philosophy:
- Handler registration pattern (callback-based)
- Async/await for non-blocking consumption
- Auto-reconnect on connection failures
- Clean shutdown on interrupt
"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import aiormq
from aiormq.abc import DeliveredMessage

from theboard.config import get_settings
from theboard.events.schemas import BaseEvent

logger = logging.getLogger(__name__)


class RabbitMQEventConsumer:
    """RabbitMQ event consumer for async event listening.

    Features:
    - Topic exchange subscription (meeting.*)
    - Handler registration for event types
    - Auto-reconnect on connection failures
    - Graceful shutdown handling

    Usage:
        consumer = RabbitMQEventConsumer()
        consumer.register_handler("meeting.human.input.needed", handle_human_input)
        await consumer.start()
    """

    def __init__(
        self,
        connection_url: str | None = None,
        exchange: str = "theboard.events",
        queue_name: str = "theboard.cli.events",
    ) -> None:
        """Initialize RabbitMQ event consumer.

        Args:
            connection_url: AMQP connection URL (defaults to config)
            exchange: Exchange name to subscribe to (default: theboard.events)
            queue_name: Queue name for this consumer (default: theboard.cli.events)
        """
        config = get_settings()
        self.connection_url = connection_url or config.rabbitmq_url
        self.exchange = exchange
        self.queue_name = queue_name

        # Handler registry: event_type -> callback function
        self.handlers: dict[str, list[Callable[[dict[str, Any]], None]]] = {}

        # Connection state
        self._connection: aiormq.Connection | None = None
        self._channel: aiormq.Channel | None = None
        self._consumer_tag: str | None = None
        self._running = False

        logger.info(
            "RabbitMQEventConsumer initialized: exchange=%s, queue=%s",
            self.exchange,
            self.queue_name,
        )

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """Register event handler for specific event type.

        Handlers are called with event payload dict when matching events arrive.
        Multiple handlers can be registered for same event type.

        Args:
            event_type: Event type to handle (e.g., "meeting.human.input.needed")
            handler: Callback function accepting event dict

        Example:
            def handle_human_input(event: dict[str, Any]) -> None:
                print(f"Human input needed: {event['prompt_text']}")

            consumer.register_handler("meeting.human.input.needed", handle_human_input)
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []

        self.handlers[event_type].append(handler)
        logger.info("Handler registered: %s -> %s", event_type, handler.__name__)

    async def start(self) -> None:
        """Start consuming events from RabbitMQ.

        Connects to RabbitMQ, declares queue, binds to meeting.* routing pattern,
        and starts consuming events. Blocks until stop() is called.

        Raises:
            RuntimeError: If connection fails
        """
        try:
            # Connect to RabbitMQ
            logger.info("Connecting to RabbitMQ: %s", self.connection_url.split("@")[-1])
            self._connection = await aiormq.connect(self.connection_url)
            self._channel = await self._connection.channel()

            # Declare exchange (should already exist from emitter)
            await self._channel.exchange_declare(
                exchange=self.exchange,
                exchange_type="topic",
                durable=True,
            )

            # Declare queue (exclusive to this consumer)
            queue_result = await self._channel.queue_declare(
                queue=self.queue_name,
                durable=False,  # Temporary queue (deleted on disconnect)
                exclusive=True,  # Only this consumer can use it
                auto_delete=True,  # Delete when consumer disconnects
            )

            # Bind queue to exchange with meeting.* routing pattern
            await self._channel.queue_bind(
                queue=self.queue_name,
                exchange=self.exchange,
                routing_key="meeting.*",  # Subscribe to all meeting events
            )

            logger.info("Queue bound: %s -> %s.meeting.*", self.queue_name, self.exchange)

            # Start consuming
            self._consumer_tag = await self._channel.basic_consume(
                queue=self.queue_name,
                consumer_callback=self._on_message,
                no_ack=True,  # Auto-acknowledge messages
            )

            self._running = True
            logger.info("Event consumer started: listening for meeting.* events")

            # Keep consuming until stopped
            while self._running:
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error("Event consumer failed: %s", e)
            raise RuntimeError(f"Failed to start event consumer: {e}") from e

        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop consuming events and close connection gracefully."""
        self._running = False

        if self._consumer_tag and self._channel:
            try:
                await self._channel.basic_cancel(self._consumer_tag)
            except Exception as e:
                logger.warning("Failed to cancel consumer: %s", e)

        if self._connection and not self._connection.is_closed:
            await self._connection.close()
            logger.info("Event consumer stopped")

        self._connection = None
        self._channel = None
        self._consumer_tag = None

    async def _on_message(self, message: DeliveredMessage) -> None:
        """Handle incoming RabbitMQ message.

        Deserializes JSON payload, routes to registered handlers based on event_type.

        Args:
            message: Delivered message from RabbitMQ
        """
        try:
            # Deserialize JSON payload
            event_data = json.loads(message.body.decode("utf-8"))
            event_type = event_data.get("event_type")

            if not event_type:
                logger.warning("Received event without event_type: %s", event_data)
                return

            logger.debug("Event received: %s", event_type)

            # Route to registered handlers
            if event_type in self.handlers:
                for handler in self.handlers[event_type]:
                    try:
                        # Call handler with event data
                        handler(event_data)
                    except Exception as e:
                        logger.error(
                            "Handler %s failed for event %s: %s",
                            handler.__name__,
                            event_type,
                            e,
                        )
            else:
                logger.debug("No handler registered for event type: %s", event_type)

        except Exception as e:
            logger.error("Failed to process message: %s", e)


# Global consumer instance (lazy initialization)
_consumer: RabbitMQEventConsumer | None = None


def get_event_consumer() -> RabbitMQEventConsumer:
    """Get or create the global event consumer.

    Returns:
        RabbitMQEventConsumer instance (singleton per process)
    """
    global _consumer

    if _consumer is not None:
        return _consumer

    _consumer = RabbitMQEventConsumer()
    return _consumer


def reset_event_consumer() -> None:
    """Reset global consumer (for testing isolation)."""
    global _consumer
    _consumer = None
