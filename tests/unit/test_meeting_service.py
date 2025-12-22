"""Unit tests for meeting service layer."""

from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from theboard.models.meeting import Meeting
from theboard.schemas import MeetingStatus, StrategyType
from theboard.services.meeting_service import (
    create_meeting,
    delete_meeting,
    get_meeting_status,
)


class TestCreateMeeting:
    """Test meeting creation functionality."""

    def test_create_meeting_success(self):
        """Test successful meeting creation."""
        from datetime import datetime

        with patch("theboard.services.meeting_service.get_sync_db") as mock_db:
            mock_session = MagicMock(spec=Session)
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock the created meeting with all required fields
            now = datetime.utcnow()
            mock_meeting = Meeting(
                id=uuid4(),
                topic="Test topic for discussion",
                strategy="sequential",
                max_rounds=5,
                current_round=0,
                status="created",
                convergence_detected=False,
                context_size=0,
                total_comments=0,
                total_cost=0.0,
                created_at=now,
                updated_at=now,
            )

            def mock_refresh(obj):
                for key, value in vars(mock_meeting).items():
                    if not key.startswith('_'):
                        setattr(obj, key, value)

            mock_session.add = MagicMock()
            mock_session.commit = MagicMock()
            mock_session.refresh = MagicMock(side_effect=mock_refresh)

            # Create meeting
            result = create_meeting(
                topic="Test topic for discussion",
                strategy=StrategyType.SEQUENTIAL,
                max_rounds=5,
                agent_count=5,
                auto_select=True,
            )

            # Verify
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
            assert result.topic == "Test topic for discussion"

    def test_create_meeting_invalid_topic_too_short(self):
        """Test meeting creation fails with too short topic."""
        with pytest.raises(ValueError, match="Topic must be between 10 and 500 characters"):
            create_meeting(
                topic="Short",
                strategy=StrategyType.SEQUENTIAL,
                max_rounds=5,
                agent_count=5,
                auto_select=True,
            )

    def test_create_meeting_invalid_topic_too_long(self):
        """Test meeting creation fails with too long topic."""
        with pytest.raises(ValueError, match="Topic must be between 10 and 500 characters"):
            create_meeting(
                topic="x" * 501,
                strategy=StrategyType.SEQUENTIAL,
                max_rounds=5,
                agent_count=5,
                auto_select=True,
            )

    def test_create_meeting_invalid_max_rounds_too_low(self):
        """Test meeting creation fails with max_rounds < 1."""
        with pytest.raises(ValueError, match="Max rounds must be between 1 and 10"):
            create_meeting(
                topic="Valid topic here",
                strategy=StrategyType.SEQUENTIAL,
                max_rounds=0,
                agent_count=5,
                auto_select=True,
            )

    def test_create_meeting_invalid_max_rounds_too_high(self):
        """Test meeting creation fails with max_rounds > 10."""
        with pytest.raises(ValueError, match="Max rounds must be between 1 and 10"):
            create_meeting(
                topic="Valid topic here",
                strategy=StrategyType.SEQUENTIAL,
                max_rounds=11,
                agent_count=5,
                auto_select=True,
            )

    def test_create_meeting_database_error(self):
        """Test meeting creation handles database errors."""
        with patch("theboard.services.meeting_service.get_sync_db") as mock_db:
            mock_session = MagicMock(spec=Session)
            mock_db.return_value.__enter__.return_value = mock_session

            # Simulate database error
            mock_session.add.side_effect = Exception("Database connection failed")

            with pytest.raises(ValueError, match="Failed to create meeting"):
                create_meeting(
                    topic="Valid topic here",
                    strategy=StrategyType.SEQUENTIAL,
                    max_rounds=5,
                    agent_count=5,
                    auto_select=True,
                )

            # Verify rollback was called
            mock_session.rollback.assert_called_once()


class TestDeleteMeeting:
    """Test meeting deletion functionality."""

    def test_delete_meeting_success(self):
        """Test successful meeting deletion."""
        meeting_id = uuid4()

        with patch("theboard.services.meeting_service.get_sync_db") as mock_db, \
             patch("theboard.services.meeting_service.get_redis_manager") as mock_redis:

            mock_session = MagicMock(spec=Session)
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock meeting exists
            mock_meeting = Meeting(id=meeting_id, topic="Test topic")
            mock_session.scalars.return_value.first.return_value = mock_meeting

            # Mock Redis manager
            mock_redis_instance = MagicMock()
            mock_redis.return_value = mock_redis_instance

            # Delete meeting
            result = delete_meeting(meeting_id)

            # Verify
            assert result is True
            mock_session.delete.assert_called_once_with(mock_meeting)
            mock_session.commit.assert_called_once()
            mock_redis_instance.delete_meeting_data.assert_called_once()

    def test_delete_meeting_not_found(self):
        """Test deleting non-existent meeting."""
        meeting_id = uuid4()

        with patch("theboard.services.meeting_service.get_sync_db") as mock_db, \
             patch("theboard.services.meeting_service.get_redis_manager"):

            mock_session = MagicMock(spec=Session)
            mock_db.return_value.__enter__.return_value = mock_session

            # Meeting doesn't exist
            mock_session.scalars.return_value.first.return_value = None

            with pytest.raises(ValueError, match="Meeting not found"):
                delete_meeting(meeting_id)

    def test_delete_meeting_database_error(self):
        """Test deletion handles database errors."""
        meeting_id = uuid4()

        with patch("theboard.services.meeting_service.get_sync_db") as mock_db, \
             patch("theboard.services.meeting_service.get_redis_manager"):

            mock_session = MagicMock(spec=Session)
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock meeting exists
            mock_meeting = Meeting(id=meeting_id, topic="Test")
            mock_session.scalars.return_value.first.return_value = mock_meeting

            # Simulate database error on delete
            mock_session.delete.side_effect = Exception("Delete failed")

            with pytest.raises(ValueError, match="Failed to delete meeting"):
                delete_meeting(meeting_id)

            mock_session.rollback.assert_called_once()


class TestGetMeetingStatus:
    """Test get meeting status functionality."""

    def test_get_meeting_status_success(self):
        """Test successful status retrieval."""
        from datetime import datetime

        meeting_id = uuid4()

        with patch("theboard.services.meeting_service.get_sync_db") as mock_db:
            mock_session = MagicMock(spec=Session)
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock meeting with all required fields
            now = datetime.utcnow()
            mock_meeting = Meeting(
                id=meeting_id,
                topic="Test topic",
                strategy="sequential",
                max_rounds=5,
                current_round=1,
                status="completed",
                convergence_detected=False,
                context_size=0,
                total_comments=0,
                total_cost=0.0,
                created_at=now,
                updated_at=now,
            )

            # Mock empty collections
            mock_session.scalars.return_value.first.return_value = mock_meeting
            mock_session.scalars.return_value.all.return_value = []

            result = get_meeting_status(meeting_id)

            assert result.meeting.id == meeting_id
            assert result.meeting.topic == "Test topic"

    def test_get_meeting_status_not_found(self):
        """Test status retrieval for non-existent meeting."""
        meeting_id = uuid4()

        with patch("theboard.services.meeting_service.get_sync_db") as mock_db:
            mock_session = MagicMock(spec=Session)
            mock_db.return_value.__enter__.return_value = mock_session

            # Meeting doesn't exist
            mock_session.scalars.return_value.first.return_value = None

            with pytest.raises(ValueError, match="Meeting not found"):
                get_meeting_status(meeting_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
