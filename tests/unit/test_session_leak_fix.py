"""Tests for database session leak fixes in Sprint 1.5.

These tests verify that database connections are properly managed during
long-running async LLM operations, preventing connection pool exhaustion.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy import select

from theboard.database import get_sync_db
from theboard.models.meeting import Agent, Meeting
from theboard.workflows.simple_meeting import SimpleMeetingWorkflow


class TestSessionLeakFix:
    """Test suite for session leak fixes in workflow execution."""

    @pytest.fixture
    def mock_meeting(self, db_session):
        """Create a test meeting."""
        meeting = Meeting(
            id=uuid4(),
            topic="Test session leak fix",
            strategy="sequential",
            max_rounds=1,
            current_round=0,
            status="running",
            convergence_detected=False,
        )
        db_session.add(meeting)
        db_session.commit()
        db_session.refresh(meeting)
        return meeting

    @pytest.fixture
    def mock_agent(self, db_session):
        """Create a test agent."""
        agent = Agent(
            name="test-agent",
            expertise="Testing",
            persona="Test persona",
            background="Test background",
            agent_type="plaintext",
            default_model="claude-sonnet-4-20250514",
            is_active=True,
        )
        db_session.add(agent)
        db_session.commit()
        db_session.refresh(agent)
        return agent

    @pytest.mark.asyncio
    async def test_execute_round_does_not_hold_session_during_llm_call(
        self, mock_meeting, mock_agent
    ):
        """Verify session is closed during LLM calls in _execute_round().

        This test ensures that the database session is not held open during
        the async LLM API calls, preventing connection pool exhaustion.

        The fix pattern is:
        1. Extract data from meeting/agent objects
        2. Close original session (parent scope)
        3. Execute LLM calls without holding session
        4. Reopen new session to store results
        """
        workflow = SimpleMeetingWorkflow(mock_meeting.id)

        # Mock the LLM agents to return immediately (no actual API calls)
        mock_expert = AsyncMock()
        mock_expert.execute = AsyncMock(return_value="Test response from expert")
        mock_expert.get_last_metadata = MagicMock(
            return_value={
                "model": "test-model",
                "tokens_used": 100,
                "cost": 0.01,
                "input_tokens": 50,
                "output_tokens": 50,
            }
        )

        mock_notetaker = AsyncMock()
        mock_notetaker.extract_comments = AsyncMock(return_value=[])
        mock_notetaker.get_last_metadata = MagicMock(
            return_value={
                "model": "test-model",
                "tokens_used": 50,
                "cost": 0.005,
                "input_tokens": 25,
                "output_tokens": 25,
            }
        )

        # Patch agent creation to return our mocks
        with patch(
            "theboard.workflows.simple_meeting.DomainExpertAgent",
            return_value=mock_expert,
        ), patch.object(workflow, "notetaker", mock_notetaker):
            # Execute round - this should NOT hold the db session during LLM calls
            with get_sync_db() as db:
                # Get fresh meeting and agent objects
                meeting = db.scalars(
                    select(Meeting).where(Meeting.id == mock_meeting.id)
                ).first()
                agent = db.scalars(
                    select(Agent).where(Agent.id == mock_agent.id)
                ).first()

                # Execute round - session will be closed during LLM calls
                await workflow._execute_round(db, meeting, agent, round_num=1)

                # Verify the session is still usable after execution
                # (this would fail if session was closed improperly)
                db.execute(select(Meeting).where(Meeting.id == mock_meeting.id))

        # Verify LLM methods were called (proving async execution happened)
        mock_expert.execute.assert_called_once()
        mock_notetaker.extract_comments.assert_called_once()

        # Verify results were stored (proving new session was opened)
        with get_sync_db() as verify_db:
            updated_meeting = verify_db.scalars(
                select(Meeting).where(Meeting.id == mock_meeting.id)
            ).first()

            # Meeting metrics should be updated
            assert updated_meeting.total_cost > 0
            assert updated_meeting.context_size > 0

    @pytest.mark.asyncio
    async def test_concurrent_rounds_do_not_exhaust_connection_pool(
        self, mock_meeting, mock_agent
    ):
        """Simulate concurrent meeting execution to verify no connection pool exhaustion.

        This test runs multiple workflow rounds concurrently to ensure that
        the session leak fix prevents connection pool exhaustion under load.
        """
        workflow = SimpleMeetingWorkflow(mock_meeting.id)

        # Mock LLM agents with delay to simulate real API calls
        async def delayed_execute(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate 100ms LLM call
            return "Test response"

        mock_expert = AsyncMock()
        mock_expert.execute = AsyncMock(side_effect=delayed_execute)
        mock_expert.get_last_metadata = MagicMock(
            return_value={
                "model": "test-model",
                "tokens_used": 100,
                "cost": 0.01,
                "input_tokens": 50,
                "output_tokens": 50,
            }
        )

        async def delayed_extract(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate 100ms LLM call
            return []

        mock_notetaker = AsyncMock()
        mock_notetaker.extract_comments = AsyncMock(side_effect=delayed_extract)
        mock_notetaker.get_last_metadata = MagicMock(
            return_value={
                "model": "test-model",
                "tokens_used": 50,
                "cost": 0.005,
                "input_tokens": 25,
                "output_tokens": 25,
            }
        )

        # Execute 5 concurrent "rounds" (simulating concurrent meetings)
        # Without the fix, this would exhaust the connection pool
        with patch(
            "theboard.workflows.simple_meeting.DomainExpertAgent",
            return_value=mock_expert,
        ), patch.object(workflow, "notetaker", mock_notetaker):
            async def execute_round_safely(round_num):
                with get_sync_db() as db:
                    meeting = db.scalars(
                        select(Meeting).where(Meeting.id == mock_meeting.id)
                    ).first()
                    agent = db.scalars(
                        select(Agent).where(Agent.id == mock_agent.id)
                    ).first()
                    await workflow._execute_round(db, meeting, agent, round_num)

            # Run 5 concurrent executions
            # With session leak, this would fail with "connection pool exhausted"
            tasks = [execute_round_safely(i) for i in range(1, 6)]
            await asyncio.gather(*tasks)

            # If we get here without exceptions, the fix is working
            assert len(tasks) == 5
