"""Integration tests for Agno framework integration.

This test suite verifies that the Agno refactoring maintains Sprint 1 functionality.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from theboard.agents.domain_expert import DomainExpertAgent
from theboard.agents.notetaker import NotetakerAgent
from theboard.schemas import Comment, CommentCategory


class TestAgnoIntegration:
    """Test Agno framework integration in TheBoard agents."""

    @pytest.mark.asyncio
    async def test_domain_expert_agent_creation(self):
        """Test that DomainExpertAgent can be created with Agno."""
        # Create agent without session_id (no persistence)
        agent = DomainExpertAgent(
            name="Test Expert",
            expertise="Software testing",
            persona="A thorough tester",
            background="10 years of QA experience",
            model="claude-sonnet-4-20250514",
            session_id=None,  # No persistence for unit test
        )

        assert agent.name == "Test Expert"
        assert agent.expertise == "Software testing"
        assert agent.persona == "A thorough tester"
        assert agent.background == "10 years of QA experience"
        assert agent._agent is not None  # Agno agent created

    @pytest.mark.asyncio
    async def test_domain_expert_with_mock_response(self):
        """Test domain expert execution with mocked Agno response."""
        agent = DomainExpertAgent(
            name="Test Expert",
            expertise="Testing",
            session_id=None,
        )

        # Mock the Agno agent's run method
        mock_response = MagicMock()
        mock_response.content = "This is a test response from the agent."

        # Mock run_response for metrics
        agent._agent.run_response = MagicMock()
        agent._agent.run_response.metrics = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch.object(agent._agent, "run", return_value=mock_response):
            result = await agent.execute("Test topic", round_num=1)

            assert result == "This is a test response from the agent."
            assert agent._last_metrics["tokens_used"] == 150
            assert agent._last_metrics["cost"] > 0

    @pytest.mark.asyncio
    async def test_notetaker_agent_creation(self):
        """Test that NotetakerAgent can be created."""
        notetaker = NotetakerAgent(model="claude-sonnet-4-20250514")

        assert notetaker.name == "Notetaker"
        assert notetaker.model == "claude-sonnet-4-20250514"
        assert notetaker._base_agent_config is not None

    @pytest.mark.asyncio
    async def test_notetaker_with_mock_structured_output(self):
        """Test notetaker extraction with mocked structured output."""
        notetaker = NotetakerAgent()

        # Create mock response with structured CommentList
        from theboard.schemas import CommentList

        mock_comment_list = CommentList(
            comments=[
                Comment(
                    text="First key insight about the topic",
                    category=CommentCategory.TECHNICAL_DECISION,
                    novelty_score=0.0,
                ),
                Comment(
                    text="Second important consideration",
                    category=CommentCategory.RISK,
                    novelty_score=0.0,
                ),
            ]
        )

        mock_response = MagicMock()
        mock_response.content = mock_comment_list

        # Mock the agent creation and execution
        with patch(
            "theboard.agents.notetaker.create_agno_agent"
        ) as mock_create_agent:
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_response
            mock_agent.run_response = MagicMock()
            mock_agent.run_response.metrics = {
                "input_tokens": 200,
                "output_tokens": 100,
            }
            mock_agent.name = "Notetaker"

            mock_create_agent.return_value = mock_agent

            comments = await notetaker.extract_comments(
                "This is a test response to extract from",
                "Test Agent",
            )

            assert len(comments) == 2
            assert comments[0].text == "First key insight about the topic"
            assert comments[0].category == CommentCategory.TECHNICAL_DECISION
            assert comments[1].text == "Second important consideration"
            assert comments[1].category == CommentCategory.RISK

    @pytest.mark.asyncio
    async def test_notetaker_fallback_on_error(self):
        """Test that notetaker has fallback behavior on extraction error."""
        notetaker = NotetakerAgent()

        # Mock agent creation to raise an exception
        with patch(
            "theboard.agents.notetaker.create_agno_agent",
            side_effect=Exception("Extraction failed"),
        ):
            response_text = "Test response that will fail to extract"
            comments = await notetaker.extract_comments(response_text, "Test Agent")

            # Should return fallback comment
            assert len(comments) == 1
            assert comments[0].category == CommentCategory.OTHER
            assert comments[0].text == response_text

    def test_metadata_extraction(self):
        """Test metadata extraction from agents."""
        agent = DomainExpertAgent(
            name="Test",
            expertise="Testing",
            session_id=None,
        )

        # Set mock metrics
        agent._last_metrics = {
            "tokens_used": 150,
            "cost": 0.0045,
            "input_tokens": 100,
            "output_tokens": 50,
        }

        metadata = agent.get_last_metadata()

        assert metadata["tokens_used"] == 150
        assert metadata["cost"] == 0.0045
        assert metadata["input_tokens"] == 100
        assert metadata["output_tokens"] == 50
        assert metadata["model"] == "claude-sonnet-4-20250514"


class TestAgnoSessionPersistence:
    """Test Agno session persistence functionality."""

    @pytest.mark.asyncio
    async def test_agent_with_session_id(self):
        """Test that agent can be created with session_id for persistence."""
        test_session_id = "test-meeting-123"

        agent = DomainExpertAgent(
            name="Test Expert",
            expertise="Testing",
            session_id=test_session_id,
        )

        assert agent.session_id == test_session_id
        # Agno agent should be configured with session persistence
        assert agent._agent.session_id == test_session_id

    @pytest.mark.asyncio
    async def test_agent_without_session_id(self):
        """Test that agent works without session_id (no persistence)."""
        agent = DomainExpertAgent(
            name="Test Expert",
            expertise="Testing",
            session_id=None,
        )

        assert agent.session_id is None
        # Agno agent should not have session persistence
        assert agent._agent.db is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
