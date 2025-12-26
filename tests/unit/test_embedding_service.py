"""Unit tests for embedding service (Sprint 3 Story 8).

Tests cover:
- EmbeddingService initialization and configuration
- Qdrant collection creation and management
- Embedding generation with sentence-transformers
- Comment embedding storage and retrieval
- Cosine similarity search
- Similarity matrix computation for clustering
- Factory pattern and singleton behavior
- Error handling and edge cases
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from theboard.services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
    reset_embedding_service,
)
from theboard.config import Settings


class TestEmbeddingServiceInitialization:
    """Test EmbeddingService initialization and configuration."""

    @patch("theboard.services.embedding_service.get_settings")
    @patch("theboard.services.embedding_service.SentenceTransformer")
    @patch("theboard.services.embedding_service.QdrantClient")
    def test_initialization_with_defaults(
        self, mock_qdrant_client, mock_sentence_transformer, mock_get_settings
    ):
        """Test EmbeddingService initializes with default configuration."""
        # Mock config
        mock_config = MagicMock(spec=Settings)
        mock_config.qdrant_url = "http://localhost:6333"
        mock_config.qdrant_api_key = None
        mock_config.embedding_model = "all-MiniLM-L6-v2"
        mock_config.embedding_batch_size = 32
        mock_get_settings.return_value = mock_config

        # Mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        # Initialize service
        service = EmbeddingService()

        # Verify Qdrant client initialization
        mock_qdrant_client.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
        )

        # Verify model initialization
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        assert service.embedding_dim == 384
        assert service.batch_size == 32

    @patch("theboard.services.embedding_service.get_settings")
    @patch("theboard.services.embedding_service.SentenceTransformer")
    @patch("theboard.services.embedding_service.QdrantClient")
    def test_initialization_with_api_key(
        self, mock_qdrant_client, mock_sentence_transformer, mock_get_settings
    ):
        """Test EmbeddingService initializes with API key."""
        # Mock config with API key
        mock_config = MagicMock(spec=Settings)
        mock_config.qdrant_url = "http://localhost:6333"
        mock_config.qdrant_api_key = "test-api-key"
        mock_config.embedding_model = "all-MiniLM-L6-v2"
        mock_config.embedding_batch_size = 32
        mock_get_settings.return_value = mock_config

        # Mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        # Initialize service
        service = EmbeddingService()

        # Verify Qdrant client initialization with API key
        mock_qdrant_client.assert_called_once_with(
            url="http://localhost:6333",
            api_key="test-api-key",
        )

    def test_initialization_with_custom_client_and_model(self):
        """Test EmbeddingService accepts custom client and model."""
        # Create custom mocks
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768

        # Initialize service with custom dependencies
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        assert service.qdrant_client is mock_client
        assert service.embedding_model is mock_model
        assert service.embedding_dim == 768


class TestCollectionManagement:
    """Test Qdrant collection creation and management."""

    def test_initialize_collection_creates_new_collection(self):
        """Test collection initialization creates new collection if not exists."""
        # Mock Qdrant client
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        # Mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        # Initialize collection
        service.initialize_collection()

        # Verify collection creation
        mock_client.create_collection.assert_called_once_with(
            collection_name="comments",
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    def test_initialize_collection_skips_if_exists(self):
        """Test collection initialization skips if collection exists."""
        # Mock Qdrant client with existing collection
        mock_collection = MagicMock()
        mock_collection.name = "comments"
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = [mock_collection]

        # Mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        # Initialize collection
        service.initialize_collection()

        # Verify collection creation was NOT called
        mock_client.create_collection.assert_not_called()


class TestEmbeddingGeneration:
    """Test embedding generation with sentence-transformers."""

    def test_embed_texts_generates_embeddings(self):
        """Test embed_texts generates embeddings for input texts."""
        # Mock model
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3

        # Initialize service
        service = EmbeddingService(
            qdrant_client=MagicMock(),
            embedding_model=mock_model,
        )
        service.batch_size = 32

        # Generate embeddings
        texts = ["First comment", "Second comment"]
        embeddings = service.embed_texts(texts)

        # Verify model.encode called with correct parameters
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Verify embeddings returned
        np.testing.assert_array_equal(embeddings, mock_embeddings)

    def test_embed_texts_handles_empty_list(self):
        """Test embed_texts handles empty input gracefully."""
        # Mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Initialize service
        service = EmbeddingService(
            qdrant_client=MagicMock(),
            embedding_model=mock_model,
        )

        # Generate embeddings for empty list
        embeddings = service.embed_texts([])

        # Verify empty array returned
        assert len(embeddings) == 0
        mock_model.encode.assert_not_called()


class TestCommentStorage:
    """Test comment embedding storage in Qdrant."""

    def test_store_comment_embeddings_success(self):
        """Test storing comment embeddings in Qdrant."""
        # Mock client
        mock_client = MagicMock()

        # Mock model
        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 3

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=mock_model,
        )

        # Store embeddings
        meeting_id = str(uuid4())
        service.store_comment_embeddings(
            comment_ids=[1, 2],
            texts=["Comment 1", "Comment 2"],
            meeting_id=meeting_id,
            round_num=1,
            agent_names=["Alice", "Bob"],
        )

        # Verify upsert called with correct points
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args

        assert call_args.kwargs["collection_name"] == "comments"
        points = call_args.kwargs["points"]

        assert len(points) == 2

        # Verify first point
        assert points[0].id == 1
        assert points[0].vector == [0.1, 0.2, 0.3]
        assert points[0].payload["comment_id"] == 1
        assert points[0].payload["text"] == "Comment 1"
        assert points[0].payload["meeting_id"] == meeting_id
        assert points[0].payload["round"] == 1
        assert points[0].payload["agent_name"] == "Alice"

        # Verify second point
        assert points[1].id == 2
        assert points[1].vector == [0.4, 0.5, 0.6]
        assert points[1].payload["agent_name"] == "Bob"

    def test_store_comment_embeddings_validates_input_lengths(self):
        """Test store_comment_embeddings validates input list lengths."""
        # Mock client and model
        service = EmbeddingService(
            qdrant_client=MagicMock(),
            embedding_model=MagicMock(),
        )

        # Attempt to store with mismatched lengths
        with pytest.raises(ValueError, match="Mismatched input lengths"):
            service.store_comment_embeddings(
                comment_ids=[1, 2],
                texts=["Comment 1"],  # Length mismatch
                meeting_id=str(uuid4()),
                round_num=1,
                agent_names=["Alice", "Bob"],
            )

    def test_store_comment_embeddings_handles_empty_list(self):
        """Test store_comment_embeddings handles empty input gracefully."""
        # Mock client and model
        mock_client = MagicMock()
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=MagicMock(),
        )

        # Store empty list
        service.store_comment_embeddings(
            comment_ids=[],
            texts=[],
            meeting_id=str(uuid4()),
            round_num=1,
            agent_names=[],
        )

        # Verify upsert not called
        mock_client.upsert.assert_not_called()


class TestSimilaritySearch:
    """Test cosine similarity search."""

    def test_find_similar_comments_with_numpy_array(self):
        """Test finding similar comments with numpy array input."""
        # Mock client with search results
        mock_client = MagicMock()
        mock_results = [
            ScoredPoint(id=10, score=0.95, version=1, vector=None, payload={}),
            ScoredPoint(id=15, score=0.87, version=1, vector=None, payload={}),
        ]
        mock_client.search.return_value = mock_results

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=MagicMock(),
        )

        # Search for similar comments
        query_embedding = np.array([0.1, 0.2, 0.3])
        similar = service.find_similar_comments(
            query_embedding=query_embedding,
            limit=10,
            score_threshold=0.85,
        )

        # Verify search called with correct parameters
        mock_client.search.assert_called_once_with(
            collection_name="comments",
            query_vector=[0.1, 0.2, 0.3],
            limit=10,
            score_threshold=0.85,
        )

        # Verify results
        assert similar == [(10, 0.95), (15, 0.87)]

    def test_find_similar_comments_with_list_input(self):
        """Test finding similar comments with list input."""
        # Mock client
        mock_client = MagicMock()
        mock_client.search.return_value = []

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=MagicMock(),
        )

        # Search with list input
        query_vector = [0.1, 0.2, 0.3]
        service.find_similar_comments(query_embedding=query_vector)

        # Verify search called with list (not converted)
        call_args = mock_client.search.call_args
        assert call_args.kwargs["query_vector"] == [0.1, 0.2, 0.3]


class TestSimilarityMatrix:
    """Test similarity matrix computation for clustering."""

    def test_compute_similarity_matrix(self):
        """Test computing pairwise similarity matrix."""
        # Mock client
        mock_client = MagicMock()

        # Mock retrieve calls (get embeddings for each comment)
        def mock_retrieve(collection_name, ids, with_vectors):
            comment_id = ids[0]
            if comment_id == 1:
                vector = [0.1, 0.2, 0.3]
            elif comment_id == 2:
                vector = [0.4, 0.5, 0.6]
            else:
                vector = [0.7, 0.8, 0.9]

            mock_point = MagicMock()
            mock_point.vector = vector
            return [mock_point]

        mock_client.retrieve.side_effect = mock_retrieve

        # Mock search calls (find similar comments)
        def mock_search(collection_name, query_vector, limit, score_threshold):
            # Return different similar comments based on query
            if query_vector == [0.1, 0.2, 0.3]:  # Comment 1
                return [
                    ScoredPoint(id=1, score=1.0, version=1, vector=None, payload={}),
                    ScoredPoint(id=2, score=0.88, version=1, vector=None, payload={}),
                ]
            elif query_vector == [0.4, 0.5, 0.6]:  # Comment 2
                return [
                    ScoredPoint(id=2, score=1.0, version=1, vector=None, payload={}),
                    ScoredPoint(id=1, score=0.88, version=1, vector=None, payload={}),
                    ScoredPoint(id=3, score=0.86, version=1, vector=None, payload={}),
                ]
            else:  # Comment 3
                return [
                    ScoredPoint(id=3, score=1.0, version=1, vector=None, payload={}),
                    ScoredPoint(id=2, score=0.86, version=1, vector=None, payload={}),
                ]

        mock_client.search.side_effect = mock_search

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=MagicMock(),
        )

        # Compute similarity matrix
        similarity_matrix = service.compute_similarity_matrix(
            comment_ids=[1, 2, 3],
            threshold=0.85,
        )

        # Verify similarity matrix structure
        assert 1 in similarity_matrix
        assert 2 in similarity_matrix
        assert 3 in similarity_matrix

        # Verify Comment 1 similarities (excluding self)
        assert 2 in similarity_matrix[1]
        assert 1 not in similarity_matrix[1]  # Self excluded

        # Verify Comment 2 similarities
        assert 1 in similarity_matrix[2]
        assert 3 in similarity_matrix[2]
        assert 2 not in similarity_matrix[2]  # Self excluded

        # Verify Comment 3 similarities
        assert 2 in similarity_matrix[3]
        assert 3 not in similarity_matrix[3]  # Self excluded

    def test_compute_similarity_matrix_handles_missing_comment(self):
        """Test similarity matrix handles comments not found in Qdrant."""
        # Mock client that returns empty for comment 2
        mock_client = MagicMock()

        def mock_retrieve(collection_name, ids, with_vectors):
            if ids[0] == 2:
                return []  # Comment not found
            mock_point = MagicMock()
            mock_point.vector = [0.1, 0.2, 0.3]
            return [mock_point]

        mock_client.retrieve.side_effect = mock_retrieve
        mock_client.search.return_value = []

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=MagicMock(),
        )

        # Compute similarity matrix
        similarity_matrix = service.compute_similarity_matrix(
            comment_ids=[1, 2, 3],
            threshold=0.85,
        )

        # Verify Comment 2 has empty similarity list
        assert similarity_matrix[2] == []


class TestCleanupOperations:
    """Test cleanup and deletion operations."""

    def test_delete_meeting_embeddings(self):
        """Test deleting all embeddings for a meeting."""
        # Mock client
        mock_client = MagicMock()

        # Initialize service
        service = EmbeddingService(
            qdrant_client=mock_client,
            embedding_model=MagicMock(),
        )

        # Delete embeddings
        meeting_id = str(uuid4())
        service.delete_meeting_embeddings(meeting_id)

        # Verify delete called with correct filter
        mock_client.delete.assert_called_once_with(
            collection_name="comments",
            points_selector={
                "filter": {
                    "must": [
                        {"key": "meeting_id", "match": {"value": meeting_id}},
                    ]
                }
            },
        )


class TestFactoryPattern:
    """Test factory pattern and singleton behavior."""

    def teardown_method(self):
        """Reset global service after each test."""
        reset_embedding_service()

    @patch("theboard.services.embedding_service.EmbeddingService")
    def test_get_embedding_service_returns_singleton(self, mock_service_class):
        """Test get_embedding_service returns singleton instance."""
        mock_instance = MagicMock()
        mock_service_class.return_value = mock_instance

        # Get service twice
        service1 = get_embedding_service()
        service2 = get_embedding_service()

        # Verify same instance returned
        assert service1 is service2

        # Verify EmbeddingService created only once
        assert mock_service_class.call_count == 1

        # Verify initialize_collection called
        mock_instance.initialize_collection.assert_called_once()

    @patch("theboard.services.embedding_service.EmbeddingService")
    def test_reset_embedding_service_forces_reinitialization(self, mock_service_class):
        """Test reset_embedding_service forces lazy reinitialization."""
        mock_instance1 = MagicMock()
        mock_instance2 = MagicMock()
        mock_service_class.side_effect = [mock_instance1, mock_instance2]

        # Get service, reset, get again
        service1 = get_embedding_service()
        reset_embedding_service()
        service2 = get_embedding_service()

        # Verify different instances after reset
        assert service1 is not service2

        # Verify EmbeddingService created twice
        assert mock_service_class.call_count == 2
