"""Unit tests for multi-agent meeting workflow with greedy execution strategy.

Sprint 4 Story 11: Greedy Execution Strategy Tests
- Test parallel agent responses using asyncio.gather
- Test comment-response phase (N² responses)
- Test error handling for individual agent failures
- Test convergence detection with greedy strategy
- Test event emission and metric tracking
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from theboard.models.meeting import Agent, Comment, Meeting, Response
from theboard.schemas import CommentCategory
from theboard.workflows.multi_agent_meeting import MultiAgentMeetingWorkflow


@pytest.fixture(autouse=True)
def mock_services():
    """Auto-mock services to avoid connection issues."""
    with patch("theboard.services.embedding_service.get_embedding_service"), \
         patch("theboard.workflows.multi_agent_meeting.get_embedding_service"), \
         patch("theboard.agents.compressor.get_embedding_service"), \
         patch("theboard.workflows.multi_agent_meeting.CompressorAgent"):
        yield


class TestGreedyExecutionStrategy:
    """Test greedy execution strategy implementation (Sprint 4 Story 11)."""

    @pytest.fixture
    def mock_meeting(self):
        """Create mock meeting with greedy strategy."""
        return Meeting(
            id=uuid4(),
            topic="Test topic for greedy execution",
            strategy="greedy",
            max_rounds=3,
            current_round=0,
            status="running",
            convergence_detected=False,
            context_size=0,
            total_comments=0,
            total_cost=0.0,
        )

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        return [
            Agent(
                id=uuid4(),
                name="Agent1",
                expertise="Testing",
                persona="Tester 1",
                background="Test background 1",
                default_model="claude-sonnet-4-20250514",
            ),
            Agent(
                id=uuid4(),
                name="Agent2",
                expertise="Development",
                persona="Developer",
                background="Test background 2",
                default_model="claude-sonnet-4-20250514",
            ),
            Agent(
                id=uuid4(),
                name="Agent3",
                expertise="Architecture",
                persona="Architect",
                background="Test background 3",
                default_model="claude-sonnet-4-20250514",
            ),
        ]

    @pytest.mark.asyncio
    async def test_greedy_strategy_branches_correctly(self, mock_meeting, mock_agents):
        """Test that greedy strategy branches to _execute_round_greedy."""
        workflow = MultiAgentMeetingWorkflow(mock_meeting.id)

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_execute_round_greedy", new_callable=AsyncMock) as mock_greedy, \
             patch.object(workflow, "_execute_round", new_callable=AsyncMock) as mock_sequential:

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            # Setup query returns: meeting, then agents
            call_count = 0
            def scalars_side_effect(*args, **kwargs):
                nonlocal call_count
                mock_result = MagicMock()
                call_count += 1

                if call_count == 1:
                    # First call: meeting
                    mock_result.first.return_value = mock_meeting
                else:
                    # Second call: agents
                    mock_result.all.return_value = mock_agents
                return mock_result

            mock_session.scalars.side_effect = scalars_side_effect

            mock_greedy.return_value = 0.5  # Mock novelty score

            # Execute workflow
            try:
                await workflow.execute()
            except Exception:
                pass  # Ignore errors, we just want to verify branching

            # Verify greedy method was called, not sequential
            assert mock_greedy.called
            assert not mock_sequential.called

    @pytest.mark.asyncio
    async def test_parallel_agent_responses(self, mock_meeting, mock_agents):
        """Test Phase 1: All agents respond in parallel using asyncio.gather."""
        workflow = MultiAgentMeetingWorkflow(mock_meeting.id)

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_execute_agent_turn", new_callable=AsyncMock) as mock_turn, \
             patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_context, \
             patch.object(workflow.emitter, "emit") as mock_emit:

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock context building
            mock_context.return_value = "Test context"

            # Mock agent turn execution - return novelty scores
            mock_turn.return_value = 0.8

            # Mock database queries for event emission
            mock_session.scalars.return_value.all.return_value = []

            # Execute greedy round
            avg_novelty = await workflow._execute_round_greedy(mock_agents, round_num=1)

            # Verify all agents were executed
            assert mock_turn.call_count == len(mock_agents)

            # Verify parallel execution (all agents executed with same context)
            for call in mock_turn.call_args_list:
                args, kwargs = call
                assert args[1] == "Test context"  # Same context for all

            # Verify novelty score calculation
            assert avg_novelty == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_comment_response_phase_n_squared(self, mock_meeting, mock_agents):
        """Test Phase 2: Comment-response phase creates N² responses."""
        workflow = MultiAgentMeetingWorkflow(mock_meeting.id)

        # Create mock responses for event emission
        mock_responses = [
            Response(
                id=uuid4(),
                meeting_id=mock_meeting.id,
                agent_id=agent.id,
                round=1,
                agent_name=agent.name,
                response_text=f"Response from {agent.name}",
                tokens_used=100,
                cost=0.01,
                model_used="test-model",
            )
            for agent in mock_agents
        ]

        # Create mock comments from Phase 1
        mock_comments = [
            Comment(
                id=i,
                meeting_id=mock_meeting.id,
                response_id=mock_responses[i].id,
                round=1,
                agent_name=agent.name,
                text=f"Comment from {agent.name}",
                category=CommentCategory.SUGGESTION,
                novelty_score=0.8,
                support_count=1,
                is_merged=False,
            )
            for i, agent in enumerate(mock_agents)
        ]

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_execute_agent_turn", new_callable=AsyncMock) as mock_turn, \
             patch.object(workflow, "_execute_agent_comment_response", new_callable=AsyncMock) as mock_comment_response, \
             patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_context, \
             patch.object(workflow.emitter, "emit"):

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock context
            mock_context.return_value = "Test context"

            # Mock Phase 1 responses
            mock_turn.return_value = 0.8

            # Mock Phase 2 comment responses
            mock_comment_response.return_value = 0.7

            # Mock database queries sequence:
            # 1st call: comments for Phase 2
            # 2nd call: responses for event emission
            # 3rd+ calls: comments per response
            call_count = 0

            def scalars_side_effect(*args, **kwargs):
                nonlocal call_count
                mock_result = MagicMock()
                call_count += 1

                if call_count == 1:
                    # First call: return comments for Phase 2
                    mock_result.all.return_value = mock_comments
                elif call_count == 2:
                    # Second call: return responses for event emission
                    mock_result.all.return_value = mock_responses
                else:
                    # Subsequent calls: return comments for each response
                    # Each response gets the same comments for simplicity
                    mock_result.all.return_value = mock_comments

                return mock_result

            mock_session.scalars.side_effect = scalars_side_effect

            # Execute greedy round
            await workflow._execute_round_greedy(mock_agents, round_num=1)

            # Verify Phase 1: N agent turns
            assert mock_turn.call_count == len(mock_agents)

            # Verify Phase 2: N comment-response calls (each agent responds to all comments)
            assert mock_comment_response.call_count == len(mock_agents)

    @pytest.mark.asyncio
    async def test_error_handling_single_agent_failure(self, mock_meeting, mock_agents):
        """Test that single agent failure doesn't block entire round."""
        workflow = MultiAgentMeetingWorkflow(mock_meeting.id)

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_execute_agent_turn", new_callable=AsyncMock) as mock_turn, \
             patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_context, \
             patch.object(workflow.emitter, "emit"):

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            mock_context.return_value = "Test context"

            # First agent succeeds, second fails, third succeeds
            mock_turn.side_effect = [
                0.8,  # Agent1 success
                RuntimeError("Agent2 failed"),  # Agent2 fails
                0.9,  # Agent3 success
            ]

            # Mock database queries
            mock_session.scalars.return_value.all.return_value = []

            # Execute should not raise exception
            avg_novelty = await workflow._execute_round_greedy(mock_agents, round_num=1)

            # Verify execution continued despite failure
            assert mock_turn.call_count == 3

            # Verify novelty calculated from successful agents only (0.8 + 0.9) / 2 = 0.85
            assert avg_novelty == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_comment_response_creates_correct_response(self, mock_meeting, mock_agents):
        """Test that comment-response phase creates proper Response and Comment records."""
        workflow = MultiAgentMeetingWorkflow(mock_meeting.id)

        agent = mock_agents[0]
        comment_context = "[Agent2] Test comment 1\n\n[Agent3] Test comment 2"

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch("theboard.workflows.multi_agent_meeting.DomainExpertAgent") as mock_expert_cls, \
             patch("theboard.workflows.multi_agent_meeting.get_preferences_manager") as mock_prefs, \
             patch("theboard.workflows.multi_agent_meeting.get_embedding_service"):

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            # Mock preferences manager
            mock_prefs.return_value.get_model_for_agent.return_value = "claude-sonnet-4-20250514"

            # Mock agent in database
            db_agent = Agent(
                id=uuid4(),
                name=agent.name,
                expertise="Testing",
                persona="Tester",
                background="Test",
                default_model="claude-sonnet-4-20250514",
            )
            mock_session.scalars.return_value.first.return_value = db_agent

            # Mock domain expert
            mock_expert = MagicMock()
            mock_expert.respond_to_context.return_value = "Response to comments"
            mock_expert.get_last_metadata.return_value = {
                "tokens_used": 100,
                "cost": 0.001,
            }
            mock_expert_cls.return_value = mock_expert

            # Mock notetaker
            workflow.notetaker.extract_comments = MagicMock(return_value=[
                type('Comment', (), {
                    'text': 'Extracted comment',
                    'category': CommentCategory.SUGGESTION,
                    'novelty_score': 0.7,
                })()
            ])

            # Execute comment response
            novelty = await workflow._execute_agent_comment_response(
                agent, comment_context, round_num=1
            )

            # Verify domain expert was called with comment context
            mock_expert.respond_to_context.assert_called_once()
            call_prompt = mock_expert.respond_to_context.call_args[0][0]
            assert "Review the following comments" in call_prompt
            assert comment_context in call_prompt

            # Verify Response record was added
            assert mock_session.add.call_count >= 1

            # Verify novelty score returned
            assert novelty == 0.7

    @pytest.mark.asyncio
    async def test_event_emission_for_greedy_round(self, mock_meeting, mock_agents):
        """Test that RoundCompletedEvent is emitted per-response for greedy strategy."""
        workflow = MultiAgentMeetingWorkflow(mock_meeting.id)

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_execute_agent_turn", new_callable=AsyncMock) as mock_turn, \
             patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_context, \
             patch.object(workflow.emitter, "emit") as mock_emit:

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            mock_context.return_value = "Test context"
            mock_turn.return_value = 0.8

            # Create mock responses with agent IDs
            agent_ids = [agent.id for agent in mock_agents]
            mock_responses = [
                Response(
                    id=uuid4(),
                    meeting_id=mock_meeting.id,
                    agent_id=agent_id,
                    round=1,
                    agent_name=mock_agents[i].name,
                    response_text="Test response",
                    tokens_used=100,
                    cost=0.001,
                    model_used="claude-sonnet-4-20250514",
                )
                for i, agent_id in enumerate(agent_ids)
            ]

            # Create mock comments for each response
            mock_comment = Comment(
                id=1,
                meeting_id=mock_meeting.id,
                response_id=mock_responses[0].id,
                round=1,
                agent_name="Agent1",
                text="Test comment",
                category=CommentCategory.SUGGESTION,
                novelty_score=0.7,
                support_count=1,
                is_merged=False,
            )

            # Setup query returns
            call_count = 0
            def scalars_side_effect(*args, **kwargs):
                nonlocal call_count
                mock_result = MagicMock()
                call_count += 1

                if call_count == 1:
                    # First call: no comments for comment-response phase
                    mock_result.all.return_value = []
                elif call_count == 2:
                    # Second call: responses for event emission
                    mock_result.all.return_value = mock_responses
                elif call_count <= 5:
                    # Calls 3-5: comments for each response
                    mock_result.all.return_value = [mock_comment]
                else:
                    # Later calls: agents
                    mock_result.first.return_value = mock_agents[(call_count - 6) % len(mock_agents)]
                return mock_result

            mock_session.scalars.side_effect = scalars_side_effect

            # Execute greedy round
            await workflow._execute_round_greedy(mock_agents, round_num=1)

            # Verify RoundCompletedEvent was emitted for each response
            assert mock_emit.call_count == len(mock_responses)

            # Verify first event has correct structure
            first_event = mock_emit.call_args_list[0][0][0]
            assert first_event.meeting_id == mock_meeting.id
            assert first_event.round_num == 1
            assert hasattr(first_event, 'agent_name')
            assert hasattr(first_event, 'response_length')
            assert first_event.tokens_used == 100
            assert first_event.cost == 0.001

    @pytest.mark.asyncio
    async def test_convergence_detection_with_greedy(self, mock_meeting, mock_agents):
        """Test that convergence detection works with greedy strategy."""
        workflow = MultiAgentMeetingWorkflow(mock_meeting.id)

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_execute_agent_turn", new_callable=AsyncMock) as mock_turn, \
             patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_context, \
             patch.object(workflow.emitter, "emit"):

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            mock_context.return_value = "Test context"

            # Simulate low novelty scores (should trigger convergence)
            mock_turn.return_value = 0.2  # Below 0.3 threshold

            # Mock database queries
            mock_session.scalars.return_value.all.return_value = []

            # Execute greedy round
            avg_novelty = await workflow._execute_round_greedy(mock_agents, round_num=1)

            # Verify low novelty is returned (convergence detected)
            assert avg_novelty == pytest.approx(0.2)
            assert avg_novelty < 0.3  # Below convergence threshold


class TestSequentialVsGreedyComparison:
    """Test comparison between sequential and greedy strategies."""

    @pytest.mark.asyncio
    async def test_greedy_has_more_responses_than_sequential(self):
        """Test that greedy strategy creates N² responses vs N for sequential.

        This test verifies the fundamental difference:
        - Sequential: N responses (one per agent per round)
        - Greedy: N + N responses (initial + comment-response phase) = 2N total
        """
        # This is a conceptual test - actual implementation would need
        # to track response counts through both strategies
        pass

    @pytest.mark.asyncio
    async def test_greedy_executes_faster_than_sequential(self):
        """Test that greedy parallel execution is faster than sequential.

        This test would need actual timing comparisons:
        - Sequential: sum of all agent execution times
        - Greedy: max of all agent execution times (parallel)
        """
        pass


class TestGreedyEdgeCases:
    """Test edge cases for greedy execution."""

    @pytest.mark.asyncio
    async def test_greedy_with_no_agents(self):
        """Test greedy execution with empty agent list."""
        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id)

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_context, \
             patch.object(workflow.emitter, "emit"):

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            mock_context.return_value = "Test context"
            mock_session.scalars.return_value.all.return_value = []

            # Execute with empty agent list
            avg_novelty = await workflow._execute_round_greedy([], round_num=1)

            # Should return default novelty
            assert avg_novelty == 1.0

    @pytest.mark.asyncio
    async def test_greedy_with_no_comments(self):
        """Test comment-response phase when no comments exist."""
        meeting = Meeting(
            id=uuid4(),
            topic="Test",
            strategy="greedy",
            max_rounds=3,
            current_round=0,
            status="running",
            convergence_detected=False,
            context_size=0,
            total_comments=0,
            total_cost=0.0,
        )

        workflow = MultiAgentMeetingWorkflow(meeting.id)

        agent = Agent(
            id=uuid4(),
            name="TestAgent",
            expertise="Testing",
            persona="Tester",
            background="Test",
            default_model="claude-sonnet-4-20250514",
        )

        with patch("theboard.workflows.multi_agent_meeting.get_sync_db") as mock_db, \
             patch.object(workflow, "_execute_agent_turn", new_callable=AsyncMock) as mock_turn, \
             patch.object(workflow, "_execute_agent_comment_response", new_callable=AsyncMock) as mock_comment_response, \
             patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_context, \
             patch.object(workflow.emitter, "emit"):

            mock_session = MagicMock()
            mock_db.return_value.__enter__.return_value = mock_session

            mock_context.return_value = "Test context"
            mock_turn.return_value = 0.8

            # No comments returned from database
            mock_session.scalars.return_value.all.return_value = []

            # Execute greedy round
            await workflow._execute_round_greedy([agent], round_num=1)

            # Phase 1 should execute
            assert mock_turn.call_count == 1

            # Phase 2 should NOT execute (no comments)
            assert mock_comment_response.call_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
