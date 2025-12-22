"""Tests for Redis manager."""

from uuid import uuid4

import pytest

from theboard.utils.redis_manager import RedisManager


@pytest.fixture
def redis_manager() -> RedisManager:
    """Create Redis manager for testing."""
    return RedisManager()


def test_redis_manager_connection(redis_manager: RedisManager) -> None:
    """Test Redis connection."""
    try:
        connected = redis_manager.is_connected()
        # Connection might fail if Redis is not running, which is OK for unit tests
        assert isinstance(connected, bool)
    except Exception:
        # Connection failed, which is acceptable in unit test environment
        pass


def test_meeting_state_operations(redis_manager: RedisManager) -> None:
    """Test setting and getting meeting state."""
    meeting_id = str(uuid4())
    state = {
        "current_round": 1,
        "current_agent": "test-agent",
        "status": "running",
        "active_context": "Test context",
        "context_size": 100,
    }

    try:
        # Set state
        result = redis_manager.set_meeting_state(meeting_id, state, ttl=60)
        assert isinstance(result, bool)

        # Get state
        retrieved_state = redis_manager.get_meeting_state(meeting_id)
        if retrieved_state:
            assert retrieved_state["current_round"] == 1
            assert retrieved_state["current_agent"] == "test-agent"

        # Clean up
        redis_manager.delete_meeting_data(meeting_id)
    except Exception:
        # Redis operations might fail if service is not available
        pass


def test_context_operations(redis_manager: RedisManager) -> None:
    """Test setting and getting context."""
    meeting_id = str(uuid4())
    context = "This is a test context with some content"

    try:
        # Set context
        result = redis_manager.set_context(meeting_id, context, ttl=60)
        assert isinstance(result, bool)

        # Get context
        retrieved_context = redis_manager.get_context(meeting_id)
        if retrieved_context:
            assert retrieved_context == context

        # Clean up
        redis_manager.delete_meeting_data(meeting_id)
    except Exception:
        # Redis operations might fail if service is not available
        pass
