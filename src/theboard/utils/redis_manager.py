"""Redis connection and caching manager."""

import json
import logging
from typing import Any

from redis import Redis
from redis.exceptions import ConnectionError, RedisError

from theboard.config import settings

logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis connections and provides caching utilities.

    Sprint 5 Story 16: Optimized TTL values for different cache types:
    - Meeting state: 7 days (long-lived, needed for history)
    - Context: 2 days (working data, regenerable)
    - Metrics: 7 days (historical data)
    """

    # Sprint 5 Story 16: Configurable TTL values (seconds)
    TTL_MEETING_STATE = 604800  # 7 days - long-lived state
    TTL_CONTEXT = 172800  # 2 days - shorter for working context
    TTL_METRICS = 604800  # 7 days - historical metrics
    TTL_PAUSE_STATE = 86400  # 1 day - Sprint 4 Story 12: pause state (24 hours max)

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
        self, meeting_id: str, state: dict[str, Any], ttl: int | None = None
    ) -> bool:
        """Set meeting state in Redis.

        Args:
            meeting_id: Meeting UUID
            state: Meeting state dictionary
            ttl: Time to live in seconds (default: TTL_MEETING_STATE)

        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.TTL_MEETING_STATE
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
        self, meeting_id: str, context: str, ttl: int | None = None
    ) -> bool:
        """Set meeting context in Redis.

        Args:
            meeting_id: Meeting UUID
            context: Context string
            ttl: Time to live in seconds (default: TTL_CONTEXT - Sprint 5 optimized to 2 days)

        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.TTL_CONTEXT  # Sprint 5: Shorter TTL for regenerable context
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
        self, meeting_id: str, round_num: int, metrics: dict[str, Any], ttl: int | None = None
    ) -> bool:
        """Set compression metrics for a meeting round.

        Args:
            meeting_id: Meeting UUID
            round_num: Round number
            metrics: Compression metrics dictionary
            ttl: Time to live in seconds (default: TTL_METRICS)

        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.TTL_METRICS
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

    # =========================================================================
    # Sprint 4 Story 12: Human-in-the-Loop Pause State Management
    # =========================================================================

    def set_pause_state(
        self,
        meeting_id: str,
        round_num: int,
        steering_context: str | None = None,
        timeout_seconds: int = 300,
    ) -> bool:
        """Set meeting pause state in Redis.

        Sprint 4 Story 12: Store pause state for human-in-the-loop.

        Args:
            meeting_id: Meeting UUID
            round_num: Current round when paused
            steering_context: Optional human steering text
            timeout_seconds: Auto-continue timeout (default 5 min)

        Returns:
            True if successful, False otherwise
        """
        try:
            import time

            key = f"meeting:{meeting_id}:pause"
            state = {
                "paused_at": time.time(),
                "round_num": round_num,
                "steering_context": steering_context,
                "timeout_seconds": timeout_seconds,
                "auto_continue_at": time.time() + timeout_seconds,
            }
            serialized = json.dumps(state)
            self.client.setex(key, self.TTL_PAUSE_STATE, serialized)
            logger.info("Set pause state for meeting %s at round %d", meeting_id, round_num)
            return True
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to set pause state: %s", e)
            return False

    def get_pause_state(self, meeting_id: str) -> dict[str, Any] | None:
        """Get meeting pause state from Redis.

        Sprint 4 Story 12: Retrieve pause state for resume.

        Args:
            meeting_id: Meeting UUID

        Returns:
            Pause state dictionary or None if not paused
        """
        try:
            key = f"meeting:{meeting_id}:pause"
            data = self.client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to get pause state: %s", e)
            return None

    def clear_pause_state(self, meeting_id: str) -> bool:
        """Clear meeting pause state from Redis.

        Sprint 4 Story 12: Clear pause state on resume.

        Args:
            meeting_id: Meeting UUID

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"meeting:{meeting_id}:pause"
            self.client.delete(key)
            logger.info("Cleared pause state for meeting %s", meeting_id)
            return True
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to clear pause state: %s", e)
            return False

    def is_meeting_paused(self, meeting_id: str) -> bool:
        """Check if meeting is currently paused.

        Sprint 4 Story 12: Quick check for pause status.

        Args:
            meeting_id: Meeting UUID

        Returns:
            True if paused, False otherwise
        """
        return self.get_pause_state(meeting_id) is not None

    def should_auto_continue(self, meeting_id: str) -> bool:
        """Check if paused meeting should auto-continue due to timeout.

        Sprint 4 Story 12: Check if timeout has elapsed.

        Args:
            meeting_id: Meeting UUID

        Returns:
            True if timeout elapsed and should auto-continue
        """
        import time

        state = self.get_pause_state(meeting_id)
        if state is None:
            return False
        auto_continue_at = state.get("auto_continue_at", 0)
        return time.time() >= auto_continue_at

    def update_steering_context(
        self, meeting_id: str, steering_context: str
    ) -> bool:
        """Update steering context for a paused meeting.

        Sprint 4 Story 12: Allow human to add steering before resume.

        Args:
            meeting_id: Meeting UUID
            steering_context: Human steering text to add

        Returns:
            True if successful, False otherwise
        """
        state = self.get_pause_state(meeting_id)
        if state is None:
            logger.warning("Cannot update steering: meeting %s not paused", meeting_id)
            return False

        state["steering_context"] = steering_context
        try:
            key = f"meeting:{meeting_id}:pause"
            serialized = json.dumps(state)
            self.client.setex(key, self.TTL_PAUSE_STATE, serialized)
            logger.info("Updated steering context for meeting %s", meeting_id)
            return True
        except (ConnectionError, RedisError) as e:
            logger.error("Failed to update steering context: %s", e)
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
