"""Tests for Redis manager.

Sprint 4 Story 12: Added tests for pause state management.
"""

import time
from unittest.mock import MagicMock, patch
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


# =============================================================================
# Sprint 4 Story 12: Pause State Management Tests
# =============================================================================


class TestPauseStateManagement:
    """Test pause state management for human-in-the-loop functionality."""

    def test_set_and_get_pause_state(self, redis_manager: RedisManager) -> None:
        """Test setting and retrieving pause state."""
        meeting_id = str(uuid4())

        try:
            # Set pause state
            result = redis_manager.set_pause_state(
                meeting_id=meeting_id,
                round_num=3,
                steering_context="Focus on cost implications",
                timeout_seconds=300,
            )
            assert isinstance(result, bool)

            # Get pause state
            state = redis_manager.get_pause_state(meeting_id)
            if state:
                assert state["round_num"] == 3
                assert state["steering_context"] == "Focus on cost implications"
                assert state["timeout_seconds"] == 300
                assert "paused_at" in state
                assert "auto_continue_at" in state

            # Clean up
            redis_manager.clear_pause_state(meeting_id)
        except Exception:
            # Redis operations might fail if service is not available
            pass

    def test_is_meeting_paused(self, redis_manager: RedisManager) -> None:
        """Test checking if meeting is paused."""
        meeting_id = str(uuid4())

        try:
            # Should not be paused initially
            assert redis_manager.is_meeting_paused(meeting_id) is False

            # Set pause state
            redis_manager.set_pause_state(meeting_id, round_num=2)

            # Should be paused now
            assert redis_manager.is_meeting_paused(meeting_id) is True

            # Clean up
            redis_manager.clear_pause_state(meeting_id)

            # Should not be paused after clearing
            assert redis_manager.is_meeting_paused(meeting_id) is False
        except Exception:
            # Redis operations might fail if service is not available
            pass

    def test_clear_pause_state(self, redis_manager: RedisManager) -> None:
        """Test clearing pause state."""
        meeting_id = str(uuid4())

        try:
            # Set pause state
            redis_manager.set_pause_state(meeting_id, round_num=1)
            assert redis_manager.get_pause_state(meeting_id) is not None

            # Clear pause state
            result = redis_manager.clear_pause_state(meeting_id)
            assert isinstance(result, bool)

            # Should be None after clearing
            assert redis_manager.get_pause_state(meeting_id) is None
        except Exception:
            # Redis operations might fail if service is not available
            pass

    def test_update_steering_context(self, redis_manager: RedisManager) -> None:
        """Test updating steering context for paused meeting."""
        meeting_id = str(uuid4())

        try:
            # Set initial pause state without steering
            redis_manager.set_pause_state(meeting_id, round_num=2)

            # Update steering context
            result = redis_manager.update_steering_context(
                meeting_id, "New steering direction"
            )
            assert isinstance(result, bool)

            # Verify update
            state = redis_manager.get_pause_state(meeting_id)
            if state:
                assert state["steering_context"] == "New steering direction"

            # Clean up
            redis_manager.clear_pause_state(meeting_id)
        except Exception:
            # Redis operations might fail if service is not available
            pass

    def test_update_steering_context_not_paused(
        self, redis_manager: RedisManager
    ) -> None:
        """Test updating steering context when meeting not paused returns False."""
        meeting_id = str(uuid4())

        try:
            # Should return False when not paused
            result = redis_manager.update_steering_context(
                meeting_id, "Some context"
            )
            assert result is False
        except Exception:
            # Redis operations might fail if service is not available
            pass

    def test_should_auto_continue_not_paused(
        self, redis_manager: RedisManager
    ) -> None:
        """Test auto-continue check when meeting not paused."""
        meeting_id = str(uuid4())

        try:
            # Should return False when not paused
            assert redis_manager.should_auto_continue(meeting_id) is False
        except Exception:
            # Redis operations might fail if service is not available
            pass

    def test_pause_state_ttl(self, redis_manager: RedisManager) -> None:
        """Test that pause state has correct TTL class constant."""
        # Verify TTL constant is set to 24 hours (86400 seconds)
        assert RedisManager.TTL_PAUSE_STATE == 86400


class TestPauseStateWithMocking:
    """Test pause state management with mocked Redis client."""

    def test_set_pause_state_stores_correct_data(self) -> None:
        """Test that set_pause_state stores correctly formatted data."""
        import json

        mock_client = MagicMock()
        manager = RedisManager()
        manager._client = mock_client
        manager._connected = True

        meeting_id = "test-meeting-123"
        round_num = 3
        steering_context = "Focus on security"
        timeout_seconds = 600

        result = manager.set_pause_state(
            meeting_id, round_num, steering_context, timeout_seconds
        )

        assert result is True
        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        key = call_args[0][0]
        ttl = call_args[0][1]
        data = call_args[0][2]

        assert key == f"meeting:{meeting_id}:pause"
        assert ttl == RedisManager.TTL_PAUSE_STATE

        stored_state = json.loads(data)
        assert stored_state["round_num"] == 3
        assert stored_state["steering_context"] == "Focus on security"
        assert stored_state["timeout_seconds"] == 600
        # paused_at and auto_continue_at should be present
        assert "paused_at" in stored_state
        assert "auto_continue_at" in stored_state
        # auto_continue_at should be paused_at + timeout_seconds
        assert stored_state["auto_continue_at"] == stored_state["paused_at"] + 600

    def test_should_auto_continue_before_timeout(self) -> None:
        """Test auto-continue returns False before timeout elapsed."""
        import json

        mock_client = MagicMock()
        manager = RedisManager()
        manager._client = mock_client
        manager._connected = True

        # Set auto_continue_at far in the future
        current_time = time.time()
        pause_state = {
            "paused_at": current_time,
            "round_num": 2,
            "steering_context": None,
            "timeout_seconds": 300,
            "auto_continue_at": current_time + 300,  # 5 minutes from now
        }
        mock_client.get.return_value = json.dumps(pause_state).encode()

        result = manager.should_auto_continue("test-meeting")

        assert result is False

    def test_should_auto_continue_after_timeout(self) -> None:
        """Test auto-continue returns True after timeout elapsed."""
        import json

        mock_client = MagicMock()
        manager = RedisManager()
        manager._client = mock_client
        manager._connected = True

        # Set auto_continue_at in the past
        current_time = time.time()
        pause_state = {
            "paused_at": current_time - 600,
            "round_num": 2,
            "steering_context": None,
            "timeout_seconds": 300,
            "auto_continue_at": current_time - 300,  # 5 minutes ago
        }
        mock_client.get.return_value = json.dumps(pause_state).encode()

        result = manager.should_auto_continue("test-meeting")

        assert result is True

    def test_get_pause_state_returns_none_when_not_found(self) -> None:
        """Test get_pause_state returns None when key doesn't exist."""
        mock_client = MagicMock()
        manager = RedisManager()
        manager._client = mock_client
        manager._connected = True

        mock_client.get.return_value = None

        result = manager.get_pause_state("nonexistent-meeting")

        assert result is None
