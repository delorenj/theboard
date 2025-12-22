"""Redis connection and caching manager."""

import json
import logging
from typing import Any

from redis import Redis
from redis.exceptions import ConnectionError, RedisError

from theboard.config import settings

logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis connections and provides caching utilities."""

    def __init__(self) -> None:
        """Initialize Redis connection."""
        self._client: Redis | None = None
        self._connected: bool = False

    @property
    def client(self) -> Redis:
        """Get Redis client, initializing if needed."""
        if self._client is None:
            self._connect()
        if self._client is None:
            raise ConnectionError("Failed to establish Redis connection")
        return self._client

    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._client = Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(
                "Redis connected: %s:%d", settings.redis_host, settings.redis_port
            )
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to connect to Redis: %s", e)
            self._connected = False
            raise

    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except (ConnectionError, RedisError):
            self._connected = False
            return False

    def set_meeting_state(
        self, meeting_id: str, state: dict[str, Any], ttl: int = 604800
    ) -> bool:
        """Set meeting state in Redis.

        Args:
            meeting_id: Meeting UUID
            state: Meeting state dictionary
            ttl: Time to live in seconds (default: 7 days)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"meeting:{meeting_id}:state"
            serialized = json.dumps(state)
            self.client.setex(key, ttl, serialized)
            return True
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to set meeting state: %s", e)
            return False

    def get_meeting_state(self, meeting_id: str) -> dict[str, Any] | None:
        """Get meeting state from Redis.

        Args:
            meeting_id: Meeting UUID

        Returns:
            Meeting state dictionary or None if not found
        """
        try:
            key = f"meeting:{meeting_id}:state"
            data = self.client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to get meeting state: %s", e)
            return None

    def set_context(
        self, meeting_id: str, context: str, ttl: int = 604800
    ) -> bool:
        """Set meeting context in Redis.

        Args:
            meeting_id: Meeting UUID
            context: Context string
            ttl: Time to live in seconds (default: 7 days)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"meeting:{meeting_id}:context"
            self.client.setex(key, ttl, context)
            return True
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to set context: %s", e)
            return False

    def get_context(self, meeting_id: str) -> str | None:
        """Get meeting context from Redis.

        Args:
            meeting_id: Meeting UUID

        Returns:
            Context string or None if not found
        """
        try:
            key = f"meeting:{meeting_id}:context"
            data = self.client.get(key)
            if data is None:
                return None
            return data.decode("utf-8")
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to get context: %s", e)
            return None

    def set_compression_metrics(
        self, meeting_id: str, round_num: int, metrics: dict[str, Any], ttl: int = 604800
    ) -> bool:
        """Set compression metrics for a meeting round.

        Args:
            meeting_id: Meeting UUID
            round_num: Round number
            metrics: Compression metrics dictionary
            ttl: Time to live in seconds (default: 7 days)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"meeting:{meeting_id}:compression:{round_num}"
            serialized = json.dumps(metrics)
            self.client.setex(key, ttl, serialized)
            return True
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to set compression metrics: %s", e)
            return False

    def get_compression_metrics(
        self, meeting_id: str, round_num: int
    ) -> dict[str, Any] | None:
        """Get compression metrics for a meeting round.

        Args:
            meeting_id: Meeting UUID
            round_num: Round number

        Returns:
            Compression metrics dictionary or None if not found
        """
        try:
            key = f"meeting:{meeting_id}:compression:{round_num}"
            data = self.client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to get compression metrics: %s", e)
            return None

    def delete_meeting_data(self, meeting_id: str) -> bool:
        """Delete all meeting data from Redis.

        Args:
            meeting_id: Meeting UUID

        Returns:
            True if successful, False otherwise
        """
        try:
            pattern = f"meeting:{meeting_id}:*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to delete meeting data: %s", e)
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            try:
                self._client.close()
                self._connected = False
                logger.info("Redis connection closed")
            except (ConnectionError, RedisError) as e:
                logger.error("Error closing Redis connection: %s", e)


# Global Redis manager instance
_redis_manager: RedisManager | None = None


def get_redis_manager() -> RedisManager:
    """Get global Redis manager instance."""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager
