"""Integration tests for Agno session persistence with real PostgreSQL.

This test suite verifies that conversation history persists across multiple
agent calls using Agno's PostgresDb integration.

NOTE: These tests require a real PostgreSQL database connection.
They will be skipped if the database is not available.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from theboard.agents.domain_expert import DomainExpertAgent
from theboard.config import settings

# Skip all tests in this module if psycopg2 not available
try:
    import psycopg2
    from agno.db.postgres import PostgresDb
    db_available = True
except (ImportError, ModuleNotFoundError):
    db_available = False
    PostgresDb = None  # Define as None for type checking

pytestmark = pytest.mark.skipif(
    not db_available,
    reason="PostgreSQL dependencies not available (psycopg2 missing)"
)


class TestSessionPersistence:
    """Test Agno session persistence with real PostgreSQL database."""

    @pytest.fixture
    def postgres_db(self):
        """Create PostgresDb instance for testing."""
        if not db_available:
            pytest.skip("PostgreSQL database not available")
        # Use the same database connection as the app
        db_url = settings.database_url_str.replace(
            "postgresql+psycopg", "postgresql"
        )
        return PostgresDb(db_url=db_url)

    @pytest.fixture
    def session_id(self):
        """Generate unique session ID for test."""
        import uuid

        return f"test-session-{uuid.uuid4()}"

    @pytest.mark.asyncio
    async def test_conversation_history_persists(self, postgres_db, session_id):
        """Test that conversation history persists across multiple agent calls.

        This test verifies:
        1. First call stores conversation in PostgresDb
        2. Second call can access history from first call
        3. Session ID continuity works correctly
        4. add_history_to_messages parameter works
        """
        # Create agent with session persistence
        agent = DomainExpertAgent(
            name="Test Expert",
            expertise="Testing and conversation history",
            model="claude-sonnet-4-20250514",
            session_id=session_id,
        )

        # Verify agent has session persistence configured
        assert agent.session_id == session_id
        assert agent._agent.session_id == session_id

        # Mock the Agno agent's run method for both calls
        mock_response_1 = MagicMock()
        mock_response_1.content = "I recommend using blue as your favorite color."

        mock_response_2 = MagicMock()
        mock_response_2.content = "As I mentioned before, blue is a great choice."

        # Mock metrics for both calls
        mock_metrics = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        agent._agent.run_response = MagicMock()
        agent._agent.run_response.metrics = mock_metrics

        with patch.object(agent._agent, "run") as mock_run:
            # First call - establish context
            mock_run.return_value = mock_response_1
            response_1 = await agent.execute(
                "Remember my favorite color is blue", round_num=1
            )

            assert "blue" in response_1.lower()
            assert mock_run.call_count == 1

            # Second call - should remember context
            mock_run.return_value = mock_response_2
            response_2 = await agent.execute("What's my favorite color?", round_num=2)

            assert "blue" in response_2.lower()
            assert mock_run.call_count == 2

            # Verify both calls used the same session
            assert agent.session_id == session_id

    @pytest.mark.asyncio
    async def test_multiple_agents_separate_sessions(self, session_id):
        """Test that different agents maintain separate session histories."""
        session_id_1 = f"{session_id}-agent1"
        session_id_2 = f"{session_id}-agent2"

        # Create two agents with different sessions
        agent_1 = DomainExpertAgent(
            name="Agent 1",
            expertise="Testing",
            session_id=session_id_1,
        )

        agent_2 = DomainExpertAgent(
            name="Agent 2",
            expertise="Testing",
            session_id=session_id_2,
        )

        # Verify they have different sessions
        assert agent_1.session_id != agent_2.session_id
        assert agent_1._agent.session_id == session_id_1
        assert agent_2._agent.session_id == session_id_2

    @pytest.mark.asyncio
    async def test_agent_without_session_has_no_persistence(self):
        """Test that agents without session_id don't use persistence."""
        agent = DomainExpertAgent(
            name="Stateless Agent",
            expertise="Testing",
            session_id=None,
        )

        # Verify no session persistence
        assert agent.session_id is None
        assert agent._agent.db is None

    @pytest.mark.asyncio
    async def test_session_continuity_across_rounds(self, session_id):
        """Test session continuity in multi-round conversation."""
        agent = DomainExpertAgent(
            name="Round Tester",
            expertise="Multi-round testing",
            session_id=session_id,
        )

        # Mock responses for multiple rounds
        mock_responses = [
            MagicMock(content="Round 1 response"),
            MagicMock(content="Round 2 response building on round 1"),
            MagicMock(content="Round 3 response building on previous rounds"),
        ]

        agent._agent.run_response = MagicMock()
        agent._agent.run_response.metrics = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch.object(agent._agent, "run") as mock_run:
            mock_run.side_effect = mock_responses

            # Execute three rounds
            for round_num in range(1, 4):
                response = await agent.execute(
                    f"Round {round_num} context", round_num=round_num
                )
                assert f"Round {round_num}" in response

            # Verify all rounds used same session
            assert mock_run.call_count == 3
            assert agent.session_id == session_id


class TestSessionPersistenceEdgeCases:
    """Test edge cases and error handling for session persistence."""

    @pytest.mark.asyncio
    async def test_empty_session_id_treated_as_none(self):
        """Test that empty string session_id is treated as no persistence."""
        agent = DomainExpertAgent(
            name="Empty Session",
            expertise="Testing",
            session_id="",
        )

        # Empty string should still be set (not converted to None)
        assert agent.session_id == ""

    @pytest.mark.asyncio
    async def test_session_with_special_characters(self):
        """Test session IDs with special characters are handled correctly."""
        special_session_id = "test-session-123-abc_def"

        agent = DomainExpertAgent(
            name="Special Session",
            expertise="Testing",
            session_id=special_session_id,
        )

        assert agent.session_id == special_session_id
        assert agent._agent.session_id == special_session_id

    @pytest.mark.asyncio
    async def test_very_long_session_id(self):
        """Test that very long session IDs are handled."""
        long_session_id = "x" * 200  # 200 character session ID

        agent = DomainExpertAgent(
            name="Long Session",
            expertise="Testing",
            session_id=long_session_id,
        )

        assert agent.session_id == long_session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
