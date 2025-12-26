"""Embedding service for semantic similarity using Qdrant and sentence-transformers.

This service provides:
- Comment embedding generation using sentence-transformers
- Vector storage in Qdrant
- Cosine similarity search for semantic clustering
- Batch processing for efficient embedding generation

Design Philosophy:
- Lazy initialization (embeddings generated on demand)
- Batch processing (reduce model overhead)
- Caching (avoid re-embedding duplicate comments)
- Environment-driven configuration (model selection, batch size)
"""

import logging
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from theboard.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol for embedding model interface."""

    def encode(
        self, sentences: list[str], batch_size: int = 32, show_progress_bar: bool = False
    ) -> np.ndarray:
        """Encode sentences into embeddings.

        Args:
            sentences: List of text strings to embed
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress

        Returns:
            Numpy array of embeddings (shape: [n_sentences, embedding_dim])
        """
        ...


class EmbeddingService:
    """Service for generating and storing comment embeddings.

    Features:
    - Sentence-transformers for semantic embeddings
    - Qdrant for vector storage and similarity search
    - Batch processing for efficiency
    - Cosine similarity for clustering
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize embedding service.

        Args:
            qdrant_client: Qdrant client (defaults to config-driven initialization)
            embedding_model: Embedding model (defaults to sentence-transformers)
        """
        config = get_settings()

        # Initialize Qdrant client
        self.qdrant_client = qdrant_client or QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key if config.qdrant_api_key else None,
        )

        # Initialize embedding model (sentence-transformers)
        # Default: all-MiniLM-L6-v2 (384-dim, fast, good quality)
        self.embedding_model = embedding_model or SentenceTransformer(config.embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()  # type: ignore

        # Configuration
        self.batch_size = config.embedding_batch_size
        self.collection_name = "comments"

        logger.info(
            "EmbeddingService initialized: model=%s, dim=%d, batch_size=%d",
            config.embedding_model,
            self.embedding_dim,
            self.batch_size,
        )

    def initialize_collection(self) -> None:
        """Initialize Qdrant collection for comments.

        Creates collection with cosine distance metric if it doesn't exist.

        Collection schema:
        - vectors: embedding vectors (384-dim by default)
        - payload: {comment_id, text, meeting_id, round, agent_name}
        """
        # Check if collection exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name in collection_names:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        # Create collection with cosine distance metric
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "Created Qdrant collection '%s' with %d-dim vectors (cosine distance)",
            self.collection_name,
            self.embedding_dim,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if not texts:
            return np.array([])

        # Batch encoding with sentence-transformers
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,  # Disable for production
            convert_to_numpy=True,
        )

        logger.debug("Generated %d embeddings (batch_size=%d)", len(texts), self.batch_size)

        return embeddings  # type: ignore

    def store_comment_embeddings(
        self,
        comment_ids: list[int],
        texts: list[str],
        meeting_id: str,
        round_num: int,
        agent_names: list[str],
    ) -> None:
        """Store comment embeddings in Qdrant.

        Args:
            comment_ids: Database IDs of comments
            texts: Comment text strings
            meeting_id: Meeting UUID
            round_num: Round number
            agent_names: Agent names for each comment

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not (len(comment_ids) == len(texts) == len(agent_names)):
            raise ValueError("Mismatched input lengths")

        if not texts:
            logger.warning("No comments to embed")
            return

        # Generate embeddings
        embeddings = self.embed_texts(texts)

        # Prepare points for Qdrant
        points = [
            PointStruct(
                id=comment_id,
                vector=embedding.tolist(),
                payload={
                    "comment_id": comment_id,
                    "text": text,
                    "meeting_id": meeting_id,
                    "round": round_num,
                    "agent_name": agent_name,
                },
            )
            for comment_id, embedding, text, agent_name in zip(
                comment_ids, embeddings, texts, agent_names
            )
        ]

        # Upsert points to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(
            "Stored %d comment embeddings (meeting=%s, round=%d)",
            len(points),
            meeting_id,
            round_num,
        )

    def find_similar_comments(
        self,
        query_embedding: np.ndarray | list[float],
        limit: int = 10,
        score_threshold: float = 0.85,
    ) -> list[tuple[int, float]]:
        """Find similar comments using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (cosine similarity)

        Returns:
            List of (comment_id, similarity_score) tuples, sorted by score (highest first)
        """
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_vector = query_embedding.tolist()
        else:
            query_vector = query_embedding

        # Search Qdrant
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Extract (comment_id, score) tuples
        similar_comments = [(hit.id, hit.score) for hit in results]

        logger.debug(
            "Found %d similar comments (threshold=%.2f, limit=%d)",
            len(similar_comments),
            score_threshold,
            limit,
        )

        return similar_comments  # type: ignore

    def compute_similarity_matrix(
        self, comment_ids: list[int], threshold: float = 0.85
    ) -> dict[int, list[int]]:
        """Compute pairwise similarity matrix for comments.

        For each comment, find all similar comments above threshold.
        Used for clustering in compression agent.

        Args:
            comment_ids: List of comment IDs to compare
            threshold: Similarity threshold (0.85 = 85% similar)

        Returns:
            Dict mapping comment_id → [similar_comment_ids]
        """
        similarity_matrix: dict[int, list[int]] = {cid: [] for cid in comment_ids}

        # For each comment, search for similar comments
        for comment_id in comment_ids:
            # Get embedding for this comment from Qdrant
            points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[comment_id],
                with_vectors=True,
            )

            if not points:
                continue

            # Get the embedding vector
            embedding = points[0].vector

            # Find similar comments
            similar = self.find_similar_comments(
                query_embedding=embedding,  # type: ignore
                limit=len(comment_ids),  # Search all comments
                score_threshold=threshold,
            )

            # Filter out self-matches and store similar comment IDs
            similarity_matrix[comment_id] = [
                similar_id for similar_id, _ in similar if similar_id != comment_id
            ]

        logger.info(
            "Computed similarity matrix for %d comments (threshold=%.2f)",
            len(comment_ids),
            threshold,
        )

        return similarity_matrix

    def delete_meeting_embeddings(self, meeting_id: str) -> None:
        """Delete all embeddings for a meeting.

        Useful for cleanup or re-embedding.

        Args:
            meeting_id: Meeting UUID
        """
        # Delete by filter (meeting_id matches)
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "meeting_id", "match": {"value": meeting_id}},
                    ]
                }
            },
        )

        logger.info("Deleted embeddings for meeting %s", meeting_id)


# Global embedding service instance (lazy initialization)
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service.

    Returns:
        EmbeddingService instance (singleton per process)
    """
    global _embedding_service

    if _embedding_service is not None:
        return _embedding_service

    _embedding_service = EmbeddingService()
    _embedding_service.initialize_collection()

    return _embedding_service


def reset_embedding_service() -> None:
    """Reset global embedding service (for testing isolation)."""
    global _embedding_service
    _embedding_service = None
