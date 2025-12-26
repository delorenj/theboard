"""Unit tests for CompressorAgent (Sprint 3 Story 9).

Tests cover:
- CompressionMetrics calculation and properties
- Tier 1: Similarity-based clustering
- Tier 2: LLM semantic merging
- Tier 3: Outlier removal
- End-to-end compression workflow
- Database integration (mark merged, update support counts)
- Error handling and edge cases
"""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from agno.agent import Agent as AgnoAgent

from theboard.agents.compressor import CompressionMetrics, CompressorAgent
from theboard.models.meeting import Comment


class TestCompressionMetrics:
    """Test CompressionMetrics calculation."""

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        metrics = CompressionMetrics(
            original_count=100,
            compressed_count=40,
            clusters_formed=15,
            outliers_removed=5,
        )

        assert metrics.compression_ratio == 0.4
        assert metrics.reduction_percentage == 60.0

    def test_compression_ratio_no_compression(self):
        """Test metrics when no compression occurs."""
        metrics = CompressionMetrics(
            original_count=50,
            compressed_count=50,
            clusters_formed=0,
            outliers_removed=0,
        )

        assert metrics.compression_ratio == 1.0
        assert metrics.reduction_percentage == 0.0

    def test_compression_ratio_empty_input(self):
        """Test metrics with zero comments."""
        metrics = CompressionMetrics(
            original_count=0,
            compressed_count=0,
            clusters_formed=0,
            outliers_removed=0,
        )

        assert metrics.compression_ratio == 1.0
        assert metrics.reduction_percentage == 0.0

    def test_metrics_repr(self):
        """Test string representation."""
        metrics = CompressionMetrics(
            original_count=100,
            compressed_count=40,
            clusters_formed=15,
            outliers_removed=5,
        )

        repr_str = repr(metrics)
        assert "original=100" in repr_str
        assert "compressed=40" in repr_str
        assert "ratio=0.40" in repr_str
        assert "reduction=60.0%" in repr_str


class TestCompressorInitialization:
    """Test CompressorAgent initialization."""

    @patch("theboard.agents.compressor.get_embedding_service")
    @patch("theboard.agents.compressor.get_settings")
    def test_initialization_with_defaults(self, mock_settings, mock_embedding):
        """Test compressor initializes with default settings."""
        mock_config = MagicMock()
        mock_config.default_model = "claude-sonnet-4-5"
        mock_settings.return_value = mock_config

        compressor = CompressorAgent()

        assert compressor.model == "claude-sonnet-4-5"
        assert compressor.similarity_threshold == 0.85
        assert compressor.outlier_threshold == 2
        mock_embedding.assert_called_once()

    @patch("theboard.agents.compressor.get_embedding_service")
    def test_initialization_with_custom_parameters(self, mock_embedding):
        """Test compressor with custom parameters."""
        compressor = CompressorAgent(
            model="claude-opus-4-5",
            similarity_threshold=0.9,
            outlier_threshold=3,
        )

        assert compressor.model == "claude-opus-4-5"
        assert compressor.similarity_threshold == 0.9
        assert compressor.outlier_threshold == 3


class TestTier1Clustering:
    """Test Tier 1: Similarity-based clustering."""

    @patch("theboard.agents.compressor.get_embedding_service")
    def test_cluster_comments_forms_clusters(self, mock_embedding):
        """Test clustering groups similar comments."""
        # Mock embedding service similarity matrix
        mock_service = MagicMock()
        # Comment 1 similar to Comment 2 (cluster A)
        # Comment 3 similar to Comment 4 (cluster B)
        # Comment 5 is singleton
        mock_service.compute_similarity_matrix.return_value = {
            1: [2],
            2: [1],
            3: [4],
            4: [3],
            5: [],
        }
        mock_embedding.return_value = mock_service

        # Create test comments
        comments = [
            Comment(id=1, text="Comment 1", agent_name="Alice", novelty_score=0.8),
            Comment(id=2, text="Comment 2", agent_name="Bob", novelty_score=0.75),
            Comment(id=3, text="Comment 3", agent_name="Charlie", novelty_score=0.9),
            Comment(id=4, text="Comment 4", agent_name="Dave", novelty_score=0.85),
            Comment(id=5, text="Comment 5", agent_name="Eve", novelty_score=0.95),
        ]

        compressor = CompressorAgent()
        clusters = compressor._tier1_cluster_comments(comments)

        # Should have 3 clusters: {1,2}, {3,4}, {5}
        assert len(clusters) == 3

        # Find cluster sizes
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        assert cluster_sizes == [2, 2, 1]

    @patch("theboard.agents.compressor.get_embedding_service")
    def test_cluster_comments_handles_singletons(self, mock_embedding):
        """Test clustering handles comments with no similar matches."""
        # Mock similarity matrix with no matches
        mock_service = MagicMock()
        mock_service.compute_similarity_matrix.return_value = {
            1: [],
            2: [],
            3: [],
        }
        mock_embedding.return_value = mock_service

        comments = [
            Comment(id=1, text="Unique 1", agent_name="Alice", novelty_score=0.8),
            Comment(id=2, text="Unique 2", agent_name="Bob", novelty_score=0.75),
            Comment(id=3, text="Unique 3", agent_name="Charlie", novelty_score=0.9),
        ]

        compressor = CompressorAgent()
        clusters = compressor._tier1_cluster_comments(comments)

        # All comments should be singletons
        assert len(clusters) == 3
        assert all(len(c) == 1 for c in clusters)


class TestTier2SemanticMerge:
    """Test Tier 2: LLM semantic merging."""

    @patch("theboard.agents.compressor.get_embedding_service")
    @patch("theboard.agents.compressor.create_agno_agent")
    def test_merge_cluster_creates_merged_comment(
        self, mock_create_agent, mock_embedding
    ):
        """Test LLM merges cluster into single comment."""
        # Mock Agno agent
        mock_agent = MagicMock(spec=AgnoAgent)
        mock_agent.run.return_value = "Merged comment summarizing key points"
        mock_create_agent.return_value = mock_agent

        # Mock database session
        mock_db = MagicMock()

        # Create test cluster
        cluster = [
            Comment(
                id=1,
                text="Comment 1",
                agent_name="Alice",
                category="suggestion",
                novelty_score=0.8,
                response_id=100,
            ),
            Comment(
                id=2,
                text="Comment 2",
                agent_name="Bob",
                category="suggestion",
                novelty_score=0.75,
                response_id=100,
            ),
        ]

        meeting_id = uuid4()
        compressor = CompressorAgent()
        merged = compressor._merge_cluster_with_llm(cluster, meeting_id, 1, mock_db)

        # Verify merged comment properties
        assert merged.text == "Merged comment summarizing key points"
        assert merged.agent_name == "CompressorAgent"
        assert merged.category == "suggestion"
        assert merged.support_count == 2
        assert merged.novelty_score == 0.775  # Average of 0.8 and 0.75
        assert merged.is_merged == False  # Result of merge, not merged itself

        # Verify agent was called with merge prompt
        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args[0][0]
        assert "Merge these 2 similar comments" in call_args
        assert "Comment 1 (Alice)" in call_args
        assert "Comment 2 (Bob)" in call_args

    @patch("theboard.agents.compressor.get_embedding_service")
    def test_semantic_merge_preserves_singletons(self, mock_embedding):
        """Test semantic merge keeps singleton clusters as-is."""
        compressor = CompressorAgent()
        mock_db = MagicMock()

        # Single-comment cluster
        singleton = [
            Comment(
                id=1,
                text="Unique comment",
                agent_name="Alice",
                category="suggestion",
                novelty_score=0.9,
                support_count=1,
            )
        ]

        meeting_id = uuid4()
        clusters = [singleton]
        merged = compressor._tier2_semantic_merge(clusters, meeting_id, 1, mock_db)

        # Should preserve singleton without LLM call
        assert len(merged) == 1
        assert merged[0].text == "Unique comment"
        assert merged[0].support_count == 1
        assert compressor.agno_agent is None  # No LLM initialized for singletons

    @patch("theboard.agents.compressor.get_embedding_service")
    @patch("theboard.agents.compressor.create_agno_agent")
    def test_semantic_merge_marks_originals_as_merged(
        self, mock_create_agent, mock_embedding
    ):
        """Test semantic merge marks original comments as merged."""
        mock_agent = MagicMock(spec=AgnoAgent)
        mock_agent.run.return_value = "Merged result"
        mock_create_agent.return_value = mock_agent

        mock_db = MagicMock()

        cluster = [
            Comment(id=1, text="C1", agent_name="A", novelty_score=0.8, is_merged=False),
            Comment(id=2, text="C2", agent_name="B", novelty_score=0.75, is_merged=False),
        ]

        meeting_id = uuid4()
        compressor = CompressorAgent()
        compressor._tier2_semantic_merge([cluster], meeting_id, 1, mock_db)

        # Original comments should be marked as merged
        assert cluster[0].is_merged == True
        assert cluster[1].is_merged == True


class TestTier3OutlierRemoval:
    """Test Tier 3: Outlier removal."""

    @patch("theboard.agents.compressor.get_embedding_service")
    def test_remove_outliers_filters_low_support(self, mock_embedding):
        """Test outlier removal filters comments with support < threshold."""
        compressor = CompressorAgent(outlier_threshold=3)

        comments = [
            Comment(id=1, text="High support", support_count=5),
            Comment(id=2, text="Medium support", support_count=3),
            Comment(id=3, text="Low support", support_count=2),
            Comment(id=4, text="Very low support", support_count=1),
        ]

        filtered = compressor._tier3_remove_outliers(comments)

        # Only comments with support >= 3 should remain
        assert len(filtered) == 2
        assert filtered[0].support_count == 5
        assert filtered[1].support_count == 3

    @patch("theboard.agents.compressor.get_embedding_service")
    def test_remove_outliers_preserves_all_when_above_threshold(self, mock_embedding):
        """Test no outliers removed when all above threshold."""
        compressor = CompressorAgent(outlier_threshold=2)

        comments = [
            Comment(id=1, text="Comment 1", support_count=5),
            Comment(id=2, text="Comment 2", support_count=3),
            Comment(id=3, text="Comment 3", support_count=2),
        ]

        filtered = compressor._tier3_remove_outliers(comments)

        # All comments should remain
        assert len(filtered) == 3


class TestEndToEndCompression:
    """Test end-to-end compression workflow."""

    @patch("theboard.agents.compressor.get_sync_db")
    @patch("theboard.agents.compressor.get_embedding_service")
    @patch("theboard.agents.compressor.create_agno_agent")
    def test_compress_comments_full_workflow(
        self, mock_create_agent, mock_embedding, mock_db_context
    ):
        """Test complete compression workflow with all three tiers."""
        # Mock database
        mock_db = MagicMock()
        mock_db_context.return_value.__enter__.return_value = mock_db

        # Create test comments
        test_comments = [
            Comment(
                id=1,
                meeting_id=uuid4(),
                round=1,
                text="Similar idea A",
                agent_name="Alice",
                category="suggestion",
                novelty_score=0.8,
                support_count=1,
                is_merged=False,
            ),
            Comment(
                id=2,
                meeting_id=uuid4(),
                round=1,
                text="Similar idea B",
                agent_name="Bob",
                category="suggestion",
                novelty_score=0.75,
                support_count=1,
                is_merged=False,
            ),
            Comment(
                id=3,
                meeting_id=uuid4(),
                round=1,
                text="Unique idea",
                agent_name="Charlie",
                category="question",
                novelty_score=0.9,
                support_count=1,
                is_merged=False,
            ),
        ]

        mock_db.query.return_value.filter.return_value.all.return_value = test_comments

        # Mock embedding service (comments 1 and 2 are similar)
        mock_embedding_svc = MagicMock()
        mock_embedding_svc.compute_similarity_matrix.return_value = {
            1: [2],
            2: [1],
            3: [],
        }
        mock_embedding.return_value = mock_embedding_svc

        # Mock Agno agent for merging
        mock_agent = MagicMock(spec=AgnoAgent)
        mock_agent.run.return_value = "Merged similar ideas A and B"
        mock_create_agent.return_value = mock_agent

        # Execute compression
        meeting_id = uuid4()
        compressor = CompressorAgent(outlier_threshold=1)
        metrics = compressor.compress_comments(meeting_id, round_num=1)

        # Verify metrics
        assert metrics.original_count == 3
        assert metrics.compressed_count == 2  # Merged {1,2} + singleton {3}
        assert metrics.clusters_formed == 2  # One cluster + one singleton
        assert metrics.reduction_percentage > 0

    @patch("theboard.agents.compressor.get_sync_db")
    @patch("theboard.agents.compressor.get_embedding_service")
    def test_compress_comments_handles_empty_input(self, mock_embedding, mock_db_context):
        """Test compression handles no comments gracefully."""
        mock_db = MagicMock()
        mock_db_context.return_value.__enter__.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = []

        meeting_id = uuid4()
        compressor = CompressorAgent()
        metrics = compressor.compress_comments(meeting_id, round_num=1)

        # Should return zero metrics
        assert metrics.original_count == 0
        assert metrics.compressed_count == 0
        assert metrics.compression_ratio == 1.0


class TestMetadataTracking:
    """Test metadata tracking for compression operations."""

    @patch("theboard.agents.compressor.get_embedding_service")
    @patch("theboard.agents.compressor.create_agno_agent")
    @patch("theboard.agents.compressor.extract_agno_metrics")
    def test_get_last_metadata_after_merge(
        self, mock_extract, mock_create_agent, mock_embedding
    ):
        """Test metadata extraction from LLM merge operations."""
        mock_agent = MagicMock(spec=AgnoAgent)
        mock_agent.run.return_value = "Merged text"
        mock_create_agent.return_value = mock_agent

        mock_extract.return_value = {
            "tokens_used": 500,
            "cost": 0.025,
            "model": "claude-sonnet-4-5",
        }

        mock_db = MagicMock()
        cluster = [
            Comment(id=1, text="C1", agent_name="A", novelty_score=0.8),
            Comment(id=2, text="C2", agent_name="B", novelty_score=0.75),
        ]

        meeting_id = uuid4()
        compressor = CompressorAgent()
        compressor._merge_cluster_with_llm(cluster, meeting_id, 1, mock_db)

        metadata = compressor.get_last_metadata()
        assert metadata["tokens_used"] == 500
        assert metadata["cost"] == 0.025
        assert metadata["model"] == "claude-sonnet-4-5"

    @patch("theboard.agents.compressor.get_embedding_service")
    def test_get_last_metadata_before_any_operation(self, mock_embedding):
        """Test metadata returns defaults before any LLM operations."""
        compressor = CompressorAgent(model="claude-opus-4-5")
        metadata = compressor.get_last_metadata()

        assert metadata["tokens_used"] == 0
        assert metadata["cost"] == 0.0
        assert metadata["model"] == "claude-opus-4-5"
