"""Unit tests for agent implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from theboard.agents.domain_expert import DomainExpertAgent
from theboard.agents.notetaker import NotetakerAgent
from theboard.schemas import Comment, CommentCategory


class TestDomainExpertAgent:
    """Test DomainExpertAgent functionality."""

    def test_agent_initialization_with_all_params(self):
        """Test agent initialization with all parameters."""
        agent = DomainExpertAgent(
            name="Test Expert",
            expertise="Testing",
            persona="A thorough tester",
            background="10 years testing",
            model="claude-sonnet-4-20250514",
            session_id="test-session",
        )

        assert agent.name == "Test Expert"
        assert agent.expertise == "Testing"
        assert agent.persona == "A thorough tester"
        assert agent.background == "10 years testing"
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.session_id == "test-session"

    def test_agent_initialization_minimal_params(self):
        """Test agent initialization with minimal parameters."""
        agent = DomainExpertAgent(
            name="Minimal Expert",
            expertise="Testing",
        )

        assert agent.name == "Minimal Expert"
        assert agent.expertise == "Testing"
        assert agent.persona is None
        assert agent.background is None
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.session_id is None

    @pytest.mark.asyncio
    async def test_execute_first_round(self):
        """Test agent execution for first round."""
        agent = DomainExpertAgent(
            name="Test Agent",
            expertise="Testing",
        )

        mock_response = MagicMock()
        mock_response.content = "First round response"

        agent._agent.run_response = MagicMock()
        agent._agent.run_response.metrics = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        with patch.object(agent._agent, "run", return_value=mock_response):
            result = await agent.execute("Test context", round_num=1)

            assert result == "First round response"
            assert "This is the first round" in agent._agent.run.call_args[0][0]

    @pytest.mark.asyncio
    async def test_execute_subsequent_round(self):
        """Test agent execution for rounds after first."""
        agent = DomainExpertAgent(
            name="Test Agent",
            expertise="Testing",
        )

        mock_response = MagicMock()
        mock_response.content = "Second round response"

        agent._agent.run_response = MagicMock()
        agent._agent.run_response.metrics = {
            "input_tokens": 150,
            "output_tokens": 75,
        }

        with patch.object(agent._agent, "run", return_value=mock_response):
            result = await agent.execute("Test context", round_num=2)

            assert result == "Second round response"
            assert "Based on the discussion" in agent._agent.run.call_args[0][0]

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        """Test agent execution handles errors."""
        agent = DomainExpertAgent(
            name="Test Agent",
            expertise="Testing",
        )

        with patch.object(agent._agent, "run", side_effect=Exception("API error")):
            with pytest.raises(RuntimeError, match="Agent execution failed"):
                await agent.execute("Test context", round_num=1)

    def test_get_last_metadata(self):
        """Test metadata retrieval."""
        agent = DomainExpertAgent(
            name="Test Agent",
            expertise="Testing",
        )

        agent._last_metrics = {
            "tokens_used": 200,
            "cost": 0.002,
            "input_tokens": 150,
            "output_tokens": 50,
        }

        metadata = agent.get_last_metadata()

        assert metadata["tokens_used"] == 200
        assert metadata["cost"] == 0.002
        assert metadata["model"] == "claude-sonnet-4-20250514"
        assert metadata["input_tokens"] == 150
        assert metadata["output_tokens"] == 50

    def test_get_last_metadata_no_metrics(self):
        """Test metadata retrieval when no metrics available."""
        agent = DomainExpertAgent(
            name="Test Agent",
            expertise="Testing",
        )

        metadata = agent.get_last_metadata()

        assert metadata["tokens_used"] == 0
        assert metadata["cost"] == 0.0
        assert metadata["input_tokens"] == 0
        assert metadata["output_tokens"] == 0


class TestNotetakerAgent:
    """Test NotetakerAgent functionality."""

    def test_notetaker_initialization(self):
        """Test notetaker initialization."""
        notetaker = NotetakerAgent(model="claude-sonnet-4-20250514")

        assert notetaker.name == "Notetaker"
        assert notetaker.model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_extract_comments_success(self):
        """Test successful comment extraction."""
        notetaker = NotetakerAgent()

        from theboard.schemas import CommentList

        mock_comment_list = CommentList(
            comments=[
                Comment(
                    text="First insight",
                    category=CommentCategory.TECHNICAL_DECISION,
                    novelty_score=0.0,
                ),
                Comment(
                    text="Second insight",
                    category=CommentCategory.RISK,
                    novelty_score=0.0,
                ),
            ]
        )

        mock_response = MagicMock()
        mock_response.content = mock_comment_list

        with patch("theboard.agents.notetaker.create_agno_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_response
            mock_agent.run_response = MagicMock()
            mock_agent.run_response.metrics = {
                "input_tokens": 200,
                "output_tokens": 100,
            }
            mock_create.return_value = mock_agent

            comments = await notetaker.extract_comments(
                "Test response",
                "Test Agent"
            )

            assert len(comments) == 2
            assert comments[0].text == "First insight"
            assert comments[1].text == "Second insight"

    @pytest.mark.asyncio
    async def test_extract_comments_fallback_on_error(self):
        """Test fallback behavior on extraction error."""
        notetaker = NotetakerAgent()

        with patch("theboard.agents.notetaker.create_agno_agent", side_effect=Exception("Extraction failed")):
            response_text = "Test response"
            comments = await notetaker.extract_comments(response_text, "Test Agent")

            # Should return fallback comment
            assert len(comments) == 1
            assert comments[0].category == CommentCategory.OTHER
            assert comments[0].text == response_text

    @pytest.mark.asyncio
    async def test_extract_comments_empty_response(self):
        """Test extraction from empty response."""
        notetaker = NotetakerAgent()

        from theboard.schemas import CommentList

        mock_comment_list = CommentList(comments=[])
        mock_response = MagicMock()
        mock_response.content = mock_comment_list

        with patch("theboard.agents.notetaker.create_agno_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.run.return_value = mock_response
            mock_agent.run_response = MagicMock()
            mock_agent.run_response.metrics = {
                "input_tokens": 100,
                "output_tokens": 10,
            }
            mock_create.return_value = mock_agent

            comments = await notetaker.extract_comments(
                "Empty response",
                "Test Agent"
            )

            assert len(comments) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
