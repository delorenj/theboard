"""Integration tests for embedding service with real Qdrant (Sprint 3 Story 8).

These tests require Qdrant to be running on localhost:6335.
Run with: pytest tests/integration/test_embedding_integration.py -v

Tests cover:
- Real Qdrant connection and collection management
- Full embedding pipeline: generate → store → search
- Similarity matrix computation for clustering
- Cleanup operations
- Batch processing performance
"""

import os
from uuid import uuid4

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Skip all tests if Qdrant is not available
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6335")


def qdrant_available() -> bool:
    """Check if Qdrant is available."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available at {}".format(QDRANT_URL),
)


@pytest.fixture
def embedding_service():
    """Create embedding service with real Qdrant connection."""
    # Import here to avoid import errors when Qdrant is not available
    from theboard.services.embedding_service import EmbeddingService, reset_embedding_service

    # Reset singleton to ensure clean state
    reset_embedding_service()

    # Create service with test collection
    qdrant_client = QdrantClient(url=QDRANT_URL)
    service = EmbeddingService(qdrant_client=qdrant_client)
    service.collection_name = f"test_comments_{uuid4().hex[:8]}"

    yield service

    # Cleanup: delete test collection
    try:
        qdrant_client.delete_collection(service.collection_name)
    except UnexpectedResponse:
        pass  # Collection might not exist


class TestEmbeddingIntegration:
    """Integration tests for full embedding pipeline."""

    def test_full_pipeline_store_and_search(self, embedding_service):
        """Test storing embeddings and searching for similar comments."""
        # Initialize collection
        embedding_service.initialize_collection()

        # Sample brainstorming comments
        meeting_id = str(uuid4())
        comments = [
            ("Use microservices for better scalability", "alice"),
            ("Microservices will help us scale the system", "bob"),
            ("Consider a monolith for simplicity", "charlie"),
            ("Redis caching improves database performance", "david"),
            ("Use Redis to cache frequent queries", "eve"),
        ]

        comment_ids = list(range(1, len(comments) + 1))
        texts = [c[0] for c in comments]
        agent_names = [c[1] for c in comments]

        # Store embeddings
        embedding_service.store_comment_embeddings(
            comment_ids=comment_ids,
            texts=texts,
            meeting_id=meeting_id,
            round_num=1,
            agent_names=agent_names,
        )

        # Generate query embedding for "microservices scaling"
        query_embedding = embedding_service.embed_texts(
            ["microservices for scaling applications"]
        )[0]

        # Search for similar comments
        similar = embedding_service.find_similar_comments(
            query_embedding=query_embedding,
            limit=5,
            score_threshold=0.7,  # Lower threshold for integration test
        )

        # Verify we found the microservices-related comments
        assert len(similar) >= 2, "Should find at least 2 similar microservices comments"

        # The top results should be comments 1 and 2 (microservices related)
        similar_ids = [hit[0] for hit in similar]
        assert 1 in similar_ids or 2 in similar_ids, "Should find microservices comments"

    def test_similarity_matrix_for_clustering(self, embedding_service):
        """Test similarity matrix computation for compression clustering."""
        # Initialize collection
        embedding_service.initialize_collection()

        # Sample comments with known similarities
        meeting_id = str(uuid4())
        comments = [
            "We should implement rate limiting to protect the API",
            "Rate limiting is essential for API security",
            "The database needs better indexing",
            "Add indexes to speed up database queries",
            "Consider using GraphQL instead of REST",
        ]

        comment_ids = list(range(100, 100 + len(comments)))
        agent_names = ["agent"] * len(comments)

        # Store embeddings
        embedding_service.store_comment_embeddings(
            comment_ids=comment_ids,
            texts=comments,
            meeting_id=meeting_id,
            round_num=1,
            agent_names=agent_names,
        )

        # Compute similarity matrix
        similarity_matrix = embedding_service.compute_similarity_matrix(
            comment_ids=comment_ids,
            threshold=0.7,  # Lower threshold for integration test
        )

        # Verify matrix structure
        assert len(similarity_matrix) == len(comments)

        # Comments 100 and 101 should be similar (rate limiting)
        assert (
            101 in similarity_matrix[100] or 100 in similarity_matrix[101]
        ), "Rate limiting comments should be similar"

        # Comments 102 and 103 should be similar (database indexing)
        assert (
            103 in similarity_matrix[102] or 102 in similarity_matrix[103]
        ), "Database indexing comments should be similar"

        # Comment 104 (GraphQL) should not be highly similar to rate limiting
        assert 100 not in similarity_matrix[104], "GraphQL should not match rate limiting"

    def test_batch_processing_performance(self, embedding_service):
        """Test batch processing with larger number of comments."""
        # Initialize collection
        embedding_service.initialize_collection()

        # Generate 100 comments (simulating a large brainstorming session)
        meeting_id = str(uuid4())
        num_comments = 100
        comments = [f"Technical comment number {i} about architecture" for i in range(num_comments)]
        comment_ids = list(range(1000, 1000 + num_comments))
        agent_names = [f"agent_{i % 5}" for i in range(num_comments)]

        # Store all embeddings in batch
        embedding_service.store_comment_embeddings(
            comment_ids=comment_ids,
            texts=comments,
            meeting_id=meeting_id,
            round_num=1,
            agent_names=agent_names,
        )

        # Verify all embeddings stored
        query_embedding = embedding_service.embed_texts(["architecture comment"])[0]
        similar = embedding_service.find_similar_comments(
            query_embedding=query_embedding,
            limit=num_comments,
            score_threshold=0.5,
        )

        # Should find most comments since they're all about architecture
        assert len(similar) >= 50, f"Should find many similar comments, found {len(similar)}"

    def test_delete_meeting_embeddings(self, embedding_service):
        """Test cleanup of meeting embeddings."""
        # Initialize collection
        embedding_service.initialize_collection()

        # Create two meetings
        meeting_id_1 = str(uuid4())
        meeting_id_2 = str(uuid4())

        # Store embeddings for meeting 1
        embedding_service.store_comment_embeddings(
            comment_ids=[1, 2],
            texts=["Meeting 1 comment A", "Meeting 1 comment B"],
            meeting_id=meeting_id_1,
            round_num=1,
            agent_names=["alice", "bob"],
        )

        # Store embeddings for meeting 2
        embedding_service.store_comment_embeddings(
            comment_ids=[3, 4],
            texts=["Meeting 2 comment A", "Meeting 2 comment B"],
            meeting_id=meeting_id_2,
            round_num=1,
            agent_names=["charlie", "david"],
        )

        # Delete meeting 1 embeddings
        embedding_service.delete_meeting_embeddings(meeting_id_1)

        # Verify meeting 1 embeddings are gone
        query_embedding = embedding_service.embed_texts(["Meeting 1 comment"])[0]
        similar = embedding_service.find_similar_comments(
            query_embedding=query_embedding,
            limit=10,
            score_threshold=0.7,
        )

        # Should not find meeting 1 comments (IDs 1, 2)
        similar_ids = [hit[0] for hit in similar]
        assert 1 not in similar_ids, "Meeting 1 comment should be deleted"
        assert 2 not in similar_ids, "Meeting 1 comment should be deleted"

        # Should still find meeting 2 comments (IDs 3, 4) - optional check
        # depending on similarity threshold


class TestEmbeddingQuality:
    """Tests for embedding model quality and thresholds."""

    def test_semantic_similarity_threshold(self, embedding_service):
        """Test that configured threshold correctly identifies similar comments."""
        # Similar sentences (should be above 0.75)
        similar_pairs = [
            ("Use microservices for scalability", "Microservices help with scaling"),
            ("Redis caching improves performance", "Cache data with Redis for speed"),
            ("Deploy with Kubernetes", "Use Kubernetes for deployment"),
        ]

        for s1, s2 in similar_pairs:
            embeddings = embedding_service.embed_texts([s1, s2])
            similarity = np.dot(embeddings[0], embeddings[1])
            assert similarity > 0.75, f"Similar sentences should score > 0.75: {similarity}"

    def test_dissimilar_comments_threshold(self, embedding_service):
        """Test that dissimilar comments score below threshold."""
        # Dissimilar sentence pairs
        dissimilar_pairs = [
            ("Use microservices for scalability", "The weather is nice today"),
            ("Redis caching improves performance", "I like pizza for dinner"),
            ("Deploy with Kubernetes", "The stock market is volatile"),
        ]

        for s1, s2 in dissimilar_pairs:
            embeddings = embedding_service.embed_texts([s1, s2])
            similarity = np.dot(embeddings[0], embeddings[1])
            assert similarity < 0.5, f"Dissimilar sentences should score < 0.5: {similarity}"
