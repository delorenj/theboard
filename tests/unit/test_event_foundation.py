"""Unit tests for event foundation (Sprint 2.5).

Tests cover:
- Event schema validation (Pydantic immutability, typing)
- InMemoryEventEmitter (storage, filtering, clearing)
- NullEventEmitter (logging behavior)
- get_event_emitter() factory (environment-driven selection)
- Event integration patterns
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from theboard.config import Settings
from theboard.events import (
    CommentExtractedEvent,
    MeetingCompletedEvent,
    MeetingConvergedEvent,
    MeetingCreatedEvent,
    MeetingFailedEvent,
    MeetingStartedEvent,
    RoundCompletedEvent,
    get_event_emitter,
)
from theboard.events.emitter import InMemoryEventEmitter, NullEventEmitter, reset_event_emitter


class TestEventSchemas:
    """Test event schema validation and immutability."""

    def test_base_event_immutability(self):
        """Test that events are immutable (frozen=True)."""
        meeting_id = uuid4()
        event = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="Test topic",
            strategy="sequential",
            max_rounds=5,
            agent_count=3,
        )

        # Attempt to modify frozen field should raise error
        with pytest.raises(Exception):  # Pydantic raises ValidationError or AttributeError
            event.topic = "Modified topic"

    def test_meeting_created_event_validation(self):
        """Test MeetingCreatedEvent schema validation."""
        meeting_id = uuid4()
        event = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="AI ethics discussion",
            strategy="sequential",
            max_rounds=5,
            agent_count=3,
        )

        assert event.event_type == "meeting.created"
        assert event.meeting_id == meeting_id
        assert event.topic == "AI ethics discussion"
        assert event.strategy == "sequential"
        assert event.max_rounds == 5
        assert event.agent_count == 3
        assert isinstance(event.timestamp, datetime)

    def test_meeting_started_event_validation(self):
        """Test MeetingStartedEvent schema validation."""
        meeting_id = uuid4()
        event = MeetingStartedEvent(
            meeting_id=meeting_id,
            selected_agents=["Alice", "Bob", "Charlie"],
            agent_count=3,
        )

        assert event.event_type == "meeting.started"
        assert event.meeting_id == meeting_id
        assert event.selected_agents == ["Alice", "Bob", "Charlie"]
        assert event.agent_count == 3

    def test_round_completed_event_validation(self):
        """Test RoundCompletedEvent schema validation."""
        meeting_id = uuid4()
        event = RoundCompletedEvent(
            meeting_id=meeting_id,
            round_num=1,
            agent_name="Alice",
            response_length=1500,
            comment_count=5,
            avg_novelty=0.75,
            tokens_used=250,
            cost=0.005,
        )

        assert event.event_type == "meeting.round_completed"
        assert event.round_num == 1
        assert event.agent_name == "Alice"
        assert event.response_length == 1500
        assert event.comment_count == 5
        assert event.avg_novelty == 0.75
        assert event.tokens_used == 250
        assert event.cost == 0.005

    def test_meeting_converged_event_validation(self):
        """Test MeetingConvergedEvent schema validation."""
        meeting_id = uuid4()
        event = MeetingConvergedEvent(
            meeting_id=meeting_id,
            round_num=3,
            avg_novelty=0.25,
            novelty_threshold=0.3,
            total_comments=15,
        )

        assert event.event_type == "meeting.converged"
        assert event.round_num == 3
        assert event.avg_novelty == 0.25
        assert event.novelty_threshold == 0.3
        assert event.total_comments == 15

    def test_meeting_completed_event_validation(self):
        """Test MeetingCompletedEvent schema validation."""
        meeting_id = uuid4()
        event = MeetingCompletedEvent(
            meeting_id=meeting_id,
            total_rounds=3,
            total_comments=15,
            total_cost=0.125,
            convergence_detected=True,
            stopping_reason="Converged at round 3 (novelty=0.25)",
        )

        assert event.event_type == "meeting.completed"
        assert event.total_rounds == 3
        assert event.total_comments == 15
        assert event.total_cost == 0.125
        assert event.convergence_detected is True
        assert "Converged" in event.stopping_reason

    def test_meeting_failed_event_validation(self):
        """Test MeetingFailedEvent schema validation."""
        meeting_id = uuid4()
        event = MeetingFailedEvent(
            meeting_id=meeting_id,
            error_type="RuntimeError",
            error_message="Database connection failed",
            round_num=2,
            agent_name="Bob",
        )

        assert event.event_type == "meeting.failed"
        assert event.error_type == "RuntimeError"
        assert event.error_message == "Database connection failed"
        assert event.round_num == 2
        assert event.agent_name == "Bob"

    def test_comment_extracted_event_validation(self):
        """Test CommentExtractedEvent schema validation."""
        meeting_id = uuid4()
        event = CommentExtractedEvent(
            meeting_id=meeting_id,
            round_num=1,
            agent_name="Alice",
            comment_text="We should prioritize user privacy",
            category="suggestion",
            novelty_score=0.85,
        )

        assert event.event_type == "meeting.comment_extracted"
        assert event.round_num == 1
        assert event.agent_name == "Alice"
        assert event.comment_text == "We should prioritize user privacy"
        assert event.category == "suggestion"
        assert event.novelty_score == 0.85


class TestInMemoryEventEmitter:
    """Test InMemoryEventEmitter for testing scenarios."""

    def test_emit_and_retrieve_events(self):
        """Test emitting and retrieving events from memory."""
        emitter = InMemoryEventEmitter()
        meeting_id = uuid4()

        # Emit multiple events
        event1 = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="Test topic",
            strategy="sequential",
            max_rounds=5,
        )
        event2 = MeetingStartedEvent(
            meeting_id=meeting_id,
            selected_agents=["Alice", "Bob"],
            agent_count=2,
        )

        emitter.emit(event1)
        emitter.emit(event2)

        # Retrieve all events
        events = emitter.get_events()
        assert len(events) == 2
        assert events[0] == event1
        assert events[1] == event2

    def test_filter_events_by_type(self):
        """Test filtering events by event_type."""
        emitter = InMemoryEventEmitter()
        meeting_id = uuid4()

        # Emit different event types
        created_event = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="Test topic",
            strategy="sequential",
            max_rounds=5,
        )
        started_event = MeetingStartedEvent(
            meeting_id=meeting_id,
            selected_agents=["Alice"],
            agent_count=1,
        )
        completed_event = MeetingCompletedEvent(
            meeting_id=meeting_id,
            total_rounds=3,
            total_comments=10,
            total_cost=0.05,
            convergence_detected=True,
            stopping_reason="Converged",
        )

        emitter.emit(created_event)
        emitter.emit(started_event)
        emitter.emit(completed_event)

        # Filter by event type
        started_events = emitter.get_events("meeting.started")
        assert len(started_events) == 1
        assert started_events[0].event_type == "meeting.started"

        completed_events = emitter.get_events("meeting.completed")
        assert len(completed_events) == 1
        assert completed_events[0].event_type == "meeting.completed"

    def test_clear_events(self):
        """Test clearing event store."""
        emitter = InMemoryEventEmitter()
        meeting_id = uuid4()

        # Emit events
        event = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="Test topic",
            strategy="sequential",
            max_rounds=5,
        )
        emitter.emit(event)

        assert len(emitter.get_events()) == 1

        # Clear events
        emitter.clear()
        assert len(emitter.get_events()) == 0

    def test_get_events_returns_copy(self):
        """Test that get_events() returns a copy, not original list."""
        emitter = InMemoryEventEmitter()
        meeting_id = uuid4()

        event = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="Test topic",
            strategy="sequential",
            max_rounds=5,
        )
        emitter.emit(event)

        events1 = emitter.get_events()
        events2 = emitter.get_events()

        # Should be different list objects (copies)
        assert events1 is not events2
        # But with same content
        assert events1 == events2


class TestNullEventEmitter:
    """Test NullEventEmitter for production with disabled events."""

    def test_emit_does_not_raise(self):
        """Test that NullEventEmitter accepts events without errors."""
        emitter = NullEventEmitter()
        meeting_id = uuid4()

        event = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="Test topic",
            strategy="sequential",
            max_rounds=5,
        )

        # Should not raise any exceptions
        emitter.emit(event)

    @patch("theboard.events.emitter.logger")
    def test_emit_logs_event(self, mock_logger):
        """Test that NullEventEmitter logs events at DEBUG level."""
        emitter = NullEventEmitter()
        meeting_id = uuid4()

        event = MeetingCreatedEvent(
            meeting_id=meeting_id,
            topic="Test topic",
            strategy="sequential",
            max_rounds=5,
        )

        emitter.emit(event)

        # Verify debug logging was called
        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0]
        assert "NullEventEmitter" in call_args[0]
        assert event.event_type in call_args[1]


class TestEventEmitterFactory:
    """Test get_event_emitter() factory with environment-driven configuration."""

    def teardown_method(self):
        """Reset global emitter after each test."""
        reset_event_emitter()

    @patch("theboard.events.emitter.get_settings")
    def test_factory_returns_inmemory_in_testing_mode(self, mock_get_settings):
        """Test that factory returns InMemoryEventEmitter when testing=True."""
        mock_config = MagicMock(spec=Settings)
        mock_config.testing = True
        mock_get_settings.return_value = mock_config

        emitter = get_event_emitter()

        assert isinstance(emitter, InMemoryEventEmitter)

    @patch("theboard.events.emitter.get_settings")
    def test_factory_returns_null_when_disabled(self, mock_get_settings):
        """Test that factory returns NullEventEmitter when events disabled."""
        mock_config = MagicMock(spec=Settings)
        mock_config.testing = False
        mock_config.event_emitter = "null"
        mock_get_settings.return_value = mock_config

        emitter = get_event_emitter()

        assert isinstance(emitter, NullEventEmitter)

    @patch("theboard.events.emitter.get_settings")
    def test_factory_returns_null_for_rabbitmq_stub(self, mock_get_settings):
        """Test that factory returns NullEventEmitter for RabbitMQ (Sprint 2.5 stub)."""
        mock_config = MagicMock(spec=Settings)
        mock_config.testing = False
        mock_config.event_emitter = "rabbitmq"
        mock_get_settings.return_value = mock_config

        emitter = get_event_emitter()

        # Sprint 2.5: RabbitMQ is a stub, falls back to null emitter
        assert isinstance(emitter, NullEventEmitter)

    @patch("theboard.events.emitter.get_settings")
    def test_factory_returns_singleton(self, mock_get_settings):
        """Test that factory returns singleton instance."""
        mock_config = MagicMock(spec=Settings)
        mock_config.testing = True
        mock_get_settings.return_value = mock_config

        emitter1 = get_event_emitter()
        emitter2 = get_event_emitter()

        # Should be same instance (singleton)
        assert emitter1 is emitter2

    @patch("theboard.events.emitter.get_settings")
    def test_reset_emitter_forces_reinitialization(self, mock_get_settings):
        """Test that reset_event_emitter() forces lazy reinitialization."""
        mock_config = MagicMock(spec=Settings)
        mock_config.testing = True
        mock_get_settings.return_value = mock_config

        emitter1 = get_event_emitter()
        reset_event_emitter()
        emitter2 = get_event_emitter()

        # Should be different instances after reset
        assert emitter1 is not emitter2


class TestEventIntegrationPatterns:
    """Test event integration patterns for workflow usage."""

    def test_event_emission_sequence(self):
        """Test typical event emission sequence in a meeting workflow."""
        emitter = InMemoryEventEmitter()
        meeting_id = uuid4()

        # Simulate meeting workflow event sequence
        emitter.emit(
            MeetingCreatedEvent(
                meeting_id=meeting_id,
                topic="AI safety",
                strategy="sequential",
                max_rounds=3,
                agent_count=2,
            )
        )

        emitter.emit(
            MeetingStartedEvent(
                meeting_id=meeting_id,
                selected_agents=["Alice", "Bob"],
                agent_count=2,
            )
        )

        # Round 1
        emitter.emit(
            RoundCompletedEvent(
                meeting_id=meeting_id,
                round_num=1,
                agent_name="Alice",
                response_length=1000,
                comment_count=3,
                avg_novelty=0.8,
                tokens_used=200,
                cost=0.004,
            )
        )

        # Convergence
        emitter.emit(
            MeetingConvergedEvent(
                meeting_id=meeting_id,
                round_num=2,
                avg_novelty=0.25,
                novelty_threshold=0.3,
                total_comments=8,
            )
        )

        # Completion
        emitter.emit(
            MeetingCompletedEvent(
                meeting_id=meeting_id,
                total_rounds=2,
                total_comments=8,
                total_cost=0.015,
                convergence_detected=True,
                stopping_reason="Converged at round 2",
            )
        )

        # Verify event sequence
        events = emitter.get_events()
        assert len(events) == 5
        assert events[0].event_type == "meeting.created"
        assert events[1].event_type == "meeting.started"
        assert events[2].event_type == "meeting.round_completed"
        assert events[3].event_type == "meeting.converged"
        assert events[4].event_type == "meeting.completed"

    def test_event_filtering_for_analytics(self):
        """Test filtering events for analytics use cases."""
        emitter = InMemoryEventEmitter()
        meeting_id = uuid4()

        # Emit multiple round events
        for round_num in range(1, 4):
            emitter.emit(
                RoundCompletedEvent(
                    meeting_id=meeting_id,
                    round_num=round_num,
                    agent_name="Alice",
                    response_length=1000,
                    comment_count=3,
                    avg_novelty=0.8 - (round_num * 0.2),
                    tokens_used=200,
                    cost=0.004,
                )
            )

        # Filter for round events only
        round_events = emitter.get_events("meeting.round_completed")
        assert len(round_events) == 3

        # Analytics: Calculate total cost across rounds
        total_cost = sum(event.cost for event in round_events)
        assert total_cost == pytest.approx(0.012)

        # Analytics: Track novelty progression
        novelty_scores = [event.avg_novelty for event in round_events]
        assert novelty_scores == pytest.approx([0.6, 0.4, 0.2])

    def test_failure_event_emission(self):
        """Test failure event emission in error scenarios."""
        emitter = InMemoryEventEmitter()
        meeting_id = uuid4()

        # Emit failure event
        emitter.emit(
            MeetingFailedEvent(
                meeting_id=meeting_id,
                error_type="RuntimeError",
                error_message="Database connection lost",
                round_num=2,
                agent_name="Bob",
            )
        )

        # Verify failure event
        failed_events = emitter.get_events("meeting.failed")
        assert len(failed_events) == 1
        assert failed_events[0].error_type == "RuntimeError"
        assert failed_events[0].round_num == 2
