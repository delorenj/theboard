"""Unit tests for workflow execution."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from theboard.models.meeting import Agent, Meeting
from theboard.workflows.simple_meeting import SimpleMeetingWorkflow


class TestSimpleMeetingWorkflow:
    """Test SimpleMeetingWorkflow functionality."""

    @pytest.mark.asyncio
    async def test_workflow_initialization(self):
        """Test workflow initializes correctly."""
        meeting_id = uuid4()
        workflow = SimpleMeetingWorkflow(meeting_id)

        assert workflow.meeting_id == meeting_id
        assert workflow.notetaker is not None

    @pytest.mark.asyncio
    async def test_get_or_create_test_agent_existing(self):
        """Test getting existing test agent."""
        meeting_id = uuid4()
        workflow = SimpleMeetingWorkflow(meeting_id)

        # Mock database session
        mock_db = MagicMock()
        existing_agent = Agent(
            id=uuid4(),
            name="test-architect",
            expertise="Software architecture",
        )
        mock_db.scalars.return_value.first.return_value = existing_agent

        agent = await workflow._get_or_create_test_agent(mock_db)

        assert agent.name == "test-architect"
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_test_agent_new(self):
        """Test creating new test agent."""
        meeting_id = uuid4()
        workflow = SimpleMeetingWorkflow(meeting_id)

        # Mock database session
        mock_db = MagicMock()
        mock_db.scalars.return_value.first.return_value = None  # No existing agent

        agent = await workflow._get_or_create_test_agent(mock_db)

        # Should create new agent
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow_meeting_not_found(self):
        """Test workflow fails when meeting not found."""
        meeting_id = uuid4()
        workflow = SimpleMeetingWorkflow(meeting_id)

        with patch("theboard.workflows.simple_meeting.get_sync_db") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.scalars.return_value.first.return_value = None

            with pytest.raises(RuntimeError, match="Meeting not found"):
                await workflow.execute()

    @pytest.mark.asyncio
    async def test_execute_workflow_wrong_status(self):
        """Test workflow fails when meeting has wrong status."""
        meeting_id = uuid4()
        workflow = SimpleMeetingWorkflow(meeting_id)

        with patch("theboard.workflows.simple_meeting.get_sync_db") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock meeting with wrong status
            mock_meeting = Meeting(
                id=meeting_id,
                topic="Test",
                status="completed",  # Wrong status
            )
            mock_session.scalars.return_value.first.return_value = mock_meeting

            with pytest.raises(RuntimeError, match="Meeting not in RUNNING state"):
                await workflow.execute()

    @pytest.mark.asyncio
    async def test_execute_round_success(self):
        """Test successful round execution."""
        meeting_id = uuid4()
        workflow = SimpleMeetingWorkflow(meeting_id)

        # Mock database
        mock_db = MagicMock()

        # Mock meeting and agent
        mock_meeting = Meeting(
            id=meeting_id,
            topic="Test topic",
            total_comments=0,
            total_cost=0.0,
            context_size=0,
        )

        mock_agent = Agent(
            id=uuid4(),
            name="Test Agent",
            expertise="Testing",
            persona="Tester",
            background="Test background",
            default_model="claude-sonnet-4-20250514",
        )

        # Mock agent execution
        mock_expert = MagicMock()
        mock_expert.execute = AsyncMock(return_value="Test response")
        mock_expert.get_last_metadata.return_value = {
            "tokens_used": 100,
            "cost": 0.001,
            "model": "claude-sonnet-4-20250514",
        }

        # Mock notetaker
        workflow.notetaker.extract_comments = AsyncMock(return_value=[])
        workflow.notetaker.get_last_metadata = MagicMock(return_value={
            "tokens_used": 50,
            "cost": 0.0005,
        })

        with patch("theboard.workflows.simple_meeting.DomainExpertAgent", return_value=mock_expert):
            await workflow._execute_round(mock_db, mock_meeting, mock_agent, round_num=1)

            # Verify agent was called
            mock_expert.execute.assert_called_once()

            # Verify database operations
            assert mock_db.add.call_count >= 1
            assert mock_db.commit.call_count >= 1


class TestWorkflowErrorHandling:
    """Test error handling in workflows."""

    @pytest.mark.asyncio
    async def test_workflow_database_rollback_on_error(self):
        """Test that database is rolled back on error."""
        meeting_id = uuid4()
        workflow = SimpleMeetingWorkflow(meeting_id)

        with patch("theboard.workflows.simple_meeting.get_sync_db") as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            # Simulate error during execution
            mock_session.scalars.side_effect = Exception("Database error")

            with pytest.raises(RuntimeError):
                await workflow.execute()

            # Verify rollback was called
            mock_session.rollback.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
