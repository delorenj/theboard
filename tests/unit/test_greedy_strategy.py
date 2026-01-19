"""Unit tests for Greedy Execution Strategy (Sprint 4 Story 11).

Tests cover:
- Parallel agent response execution using asyncio.gather
- Comment-response phase (N² interactions)
- Token efficiency tracking
- RoundMetrics and ExecutionBenchmark dataclasses
- Strategy selection in workflow initialization
- Convergence detection with greedy strategy
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from theboard.schemas import StrategyType
from theboard.workflows.multi_agent_meeting import (
    ExecutionBenchmark,
    MultiAgentMeetingWorkflow,
    RoundMetrics,
)


class TestRoundMetrics:
    """Test RoundMetrics dataclass."""

    def test_round_metrics_creation(self):
        """Test RoundMetrics can be created with all fields."""
        metrics = RoundMetrics(
            round_num=1,
            strategy="greedy",
            execution_time_seconds=5.5,
            agent_count=5,
            total_responses=25,
            total_comments=50,
            total_tokens=10000,
            total_cost=0.15,
            avg_novelty=0.75,
            parallel_responses=5,
            comment_response_count=20,
        )

        assert metrics.round_num == 1
        assert metrics.strategy == "greedy"
        assert metrics.execution_time_seconds == 5.5
        assert metrics.agent_count == 5
        assert metrics.total_responses == 25
        assert metrics.total_comments == 50
        assert metrics.total_tokens == 10000
        assert metrics.total_cost == 0.15
        assert metrics.avg_novelty == 0.75
        assert metrics.parallel_responses == 5
        assert metrics.comment_response_count == 20

    def test_round_metrics_defaults(self):
        """Test RoundMetrics uses correct defaults."""
        metrics = RoundMetrics(
            round_num=1,
            strategy="sequential",
            execution_time_seconds=10.0,
            agent_count=3,
            total_responses=3,
            total_comments=9,
            total_tokens=5000,
            total_cost=0.08,
            avg_novelty=0.8,
        )

        # Greedy-specific fields should default to 0
        assert metrics.parallel_responses == 0
        assert metrics.comment_response_count == 0


class TestExecutionBenchmark:
    """Test ExecutionBenchmark dataclass."""

    def test_execution_benchmark_creation(self):
        """Test ExecutionBenchmark can be created with all fields."""
        round_metrics = [
            RoundMetrics(
                round_num=1,
                strategy="greedy",
                execution_time_seconds=5.0,
                agent_count=5,
                total_responses=25,
                total_comments=50,
                total_tokens=10000,
                total_cost=0.15,
                avg_novelty=0.8,
            ),
            RoundMetrics(
                round_num=2,
                strategy="greedy",
                execution_time_seconds=4.5,
                agent_count=5,
                total_responses=25,
                total_comments=45,
                total_tokens=9000,
                total_cost=0.13,
                avg_novelty=0.5,
            ),
        ]

        benchmark = ExecutionBenchmark(
            strategy="greedy",
            total_rounds=2,
            total_execution_time_seconds=9.5,
            total_tokens=19000,
            total_cost=0.28,
            avg_tokens_per_round=9500.0,
            avg_time_per_round_seconds=4.75,
            round_metrics=round_metrics,
        )

        assert benchmark.strategy == "greedy"
        assert benchmark.total_rounds == 2
        assert benchmark.total_execution_time_seconds == 9.5
        assert benchmark.total_tokens == 19000
        assert benchmark.total_cost == 0.28
        assert len(benchmark.round_metrics) == 2


class TestWorkflowStrategyInitialization:
    """Test MultiAgentMeetingWorkflow strategy initialization."""

    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    def test_workflow_initializes_with_sequential_strategy(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding
    ):
        """Test workflow defaults to sequential strategy."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id)

        assert workflow.strategy == StrategyType.SEQUENTIAL
        assert workflow.round_metrics == []

    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    def test_workflow_initializes_with_greedy_strategy(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding
    ):
        """Test workflow can be initialized with greedy strategy."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.GREEDY)

        assert workflow.strategy == StrategyType.GREEDY
        assert workflow.round_metrics == []

    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    def test_workflow_initializes_token_tracking(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding
    ):
        """Test workflow initializes token tracking attributes."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.GREEDY)

        assert hasattr(workflow, "round_metrics")
        assert hasattr(workflow, "execution_start_time")
        assert workflow.execution_start_time is None


class TestGetExecutionBenchmark:
    """Test get_execution_benchmark method."""

    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    def test_get_benchmark_empty_metrics(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding
    ):
        """Test benchmark returns empty values when no rounds executed."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.GREEDY)

        benchmark = workflow.get_execution_benchmark()

        assert benchmark.strategy == "greedy"
        assert benchmark.total_rounds == 0
        assert benchmark.total_execution_time_seconds == 0.0
        assert benchmark.total_tokens == 0
        assert benchmark.total_cost == 0.0
        assert benchmark.avg_tokens_per_round == 0.0
        assert benchmark.avg_time_per_round_seconds == 0.0

    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    def test_get_benchmark_with_metrics(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding
    ):
        """Test benchmark aggregates metrics correctly."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.GREEDY)

        # Simulate round metrics
        workflow.round_metrics = [
            RoundMetrics(
                round_num=1,
                strategy="greedy",
                execution_time_seconds=5.0,
                agent_count=5,
                total_responses=25,
                total_comments=50,
                total_tokens=10000,
                total_cost=0.15,
                avg_novelty=0.8,
            ),
            RoundMetrics(
                round_num=2,
                strategy="greedy",
                execution_time_seconds=5.0,
                agent_count=5,
                total_responses=25,
                total_comments=40,
                total_tokens=10000,
                total_cost=0.15,
                avg_novelty=0.5,
            ),
        ]

        benchmark = workflow.get_execution_benchmark()

        assert benchmark.total_rounds == 2
        assert benchmark.total_execution_time_seconds == 10.0
        assert benchmark.total_tokens == 20000
        assert benchmark.total_cost == 0.30
        assert benchmark.avg_tokens_per_round == 10000.0
        assert benchmark.avg_time_per_round_seconds == 5.0


class TestGreedyRoundExecution:
    """Test greedy round execution logic."""

    @patch("theboard.workflows.multi_agent_meeting.get_sync_db")
    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    @pytest.mark.asyncio
    async def test_greedy_round_builds_shared_context(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding, mock_db
    ):
        """Test greedy round builds shared context for all agents."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.GREEDY)

        # Mock the context building
        with patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = "Test context"

            with patch.object(workflow, "_execute_agent_turn", new_callable=AsyncMock) as mock_turn:
                mock_turn.return_value = 0.5

                # Mock database query for comments
                mock_session = MagicMock()
                mock_db.return_value.__enter__.return_value = mock_session
                mock_session.scalars.return_value.all.return_value = []
                mock_session.scalars.return_value.first.return_value = MagicMock(
                    tokens_used=100, cost=0.01
                )

                # Create mock agents
                agents = [
                    MagicMock(id=uuid4(), name="Agent1"),
                    MagicMock(id=uuid4(), name="Agent2"),
                ]

                avg_novelty, metrics = await workflow._execute_round_greedy(agents, 1)

                # Verify shared context was built once
                mock_build.assert_called_once_with(1, agent_id=None)

    @patch("theboard.workflows.multi_agent_meeting.get_sync_db")
    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    @pytest.mark.asyncio
    async def test_greedy_round_executes_agents_in_parallel(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding, mock_db
    ):
        """Test greedy round executes all agents in parallel."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.GREEDY)

        call_times = []

        async def mock_execute_turn(agent, context, round_num):
            import asyncio

            call_times.append(asyncio.get_event_loop().time())
            return 0.5

        with patch.object(workflow, "_build_context", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = "Test context"

            with patch.object(workflow, "_execute_agent_turn", side_effect=mock_execute_turn):
                # Mock database
                mock_session = MagicMock()
                mock_db.return_value.__enter__.return_value = mock_session
                mock_session.scalars.return_value.all.return_value = []
                mock_session.scalars.return_value.first.return_value = MagicMock(
                    tokens_used=100, cost=0.01
                )

                # Create mock agents
                agents = [
                    MagicMock(id=uuid4(), name=f"Agent{i}")
                    for i in range(3)
                ]

                await workflow._execute_round_greedy(agents, 1)

                # All agents should have been called
                assert len(call_times) == 3


class TestGreedyVsSequentialComparison:
    """Test comparison between greedy and sequential strategies."""

    def test_strategy_enum_values(self):
        """Test StrategyType enum has correct values."""
        assert StrategyType.SEQUENTIAL.value == "sequential"
        assert StrategyType.GREEDY.value == "greedy"

    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    def test_both_strategies_track_metrics(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding
    ):
        """Test both strategies can track metrics."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()

        # Sequential workflow
        seq_workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.SEQUENTIAL)
        assert hasattr(seq_workflow, "round_metrics")

        # Greedy workflow
        greedy_workflow = MultiAgentMeetingWorkflow(meeting_id, strategy=StrategyType.GREEDY)
        assert hasattr(greedy_workflow, "round_metrics")


class TestConvergenceWithGreedyStrategy:
    """Test convergence detection works with greedy strategy."""

    @patch("theboard.workflows.multi_agent_meeting.get_embedding_service")
    @patch("theboard.workflows.multi_agent_meeting.get_preferences_manager")
    @patch("theboard.workflows.multi_agent_meeting.NotetakerAgent")
    @patch("theboard.workflows.multi_agent_meeting.CompressorAgent")
    def test_convergence_threshold_applies_to_greedy(
        self, mock_compressor, mock_notetaker, mock_prefs, mock_embedding
    ):
        """Test convergence threshold is respected in greedy strategy."""
        mock_prefs.return_value.get_model_for_agent.return_value = "test-model"

        meeting_id = uuid4()
        workflow = MultiAgentMeetingWorkflow(
            meeting_id,
            strategy=StrategyType.GREEDY,
            novelty_threshold=0.3,
            min_rounds=2,
        )

        assert workflow.novelty_threshold == 0.3
        assert workflow.min_rounds == 2
        assert workflow.strategy == StrategyType.GREEDY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
