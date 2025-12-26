"""Compressor Agent for intelligent comment compression (Sprint 3 Story 9).

This agent implements a three-tier compression strategy:
- Tier 1: Similarity-based clustering using embeddings (Qdrant)
- Tier 2: LLM semantic merge (Claude Sonnet)
- Tier 3: Outlier removal (support count threshold)

Design Philosophy:
- Preserve information quality while reducing context size
- Track compression metrics for observability
- Maintain audit trail (original comments retained)
- Non-destructive operations (mark as merged, don't delete)
"""

import logging
from collections import defaultdict
from typing import Protocol
from uuid import UUID

import networkx as nx
from agno.agent import Agent as AgnoAgent

from theboard.agents.base import create_agno_agent, extract_agno_metrics
from theboard.config import get_settings
from theboard.database import get_sync_db
from theboard.models.meeting import Comment
from theboard.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class CompressionMetrics:
    """Metrics for compression operation."""

    def __init__(
        self,
        original_count: int,
        compressed_count: int,
        clusters_formed: int,
        outliers_removed: int,
    ) -> None:
        """Initialize compression metrics.

        Args:
            original_count: Number of comments before compression
            compressed_count: Number of comments after compression
            clusters_formed: Number of comment clusters identified
            outliers_removed: Number of outlier comments removed
        """
        self.original_count = original_count
        self.compressed_count = compressed_count
        self.clusters_formed = clusters_formed
        self.outliers_removed = outliers_removed

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (0-1, where 1 = no compression)."""
        if self.original_count == 0:
            return 1.0
        return self.compressed_count / self.original_count

    @property
    def reduction_percentage(self) -> float:
        """Calculate reduction percentage (0-100)."""
        return (1 - self.compression_ratio) * 100

    def __repr__(self) -> str:
        """String representation of metrics."""
        return (
            f"CompressionMetrics(original={self.original_count}, "
            f"compressed={self.compressed_count}, "
            f"ratio={self.compression_ratio:.2f}, "
            f"reduction={self.reduction_percentage:.1f}%)"
        )


class CompressorAgent:
    """Compressor agent for multi-tier comment compression.

    Three-Tier Compression Strategy:
    1. Similarity Clustering: Group similar comments (cosine ≥ 0.85)
    2. LLM Semantic Merge: Merge clustered comments into concise summaries
    3. Outlier Removal: Remove low-support comments (support < threshold)

    Compression Goals:
    - Reduce comment count by 40-60%
    - Preserve information quality and key insights
    - Maintain traceability (original comments retained)
    """

    def __init__(
        self,
        model: str | None = None,
        similarity_threshold: float = 0.85,
        outlier_threshold: int = 2,
    ) -> None:
        """Initialize compressor agent.

        Args:
            model: LLM model for semantic merging (default: from config)
            similarity_threshold: Cosine similarity threshold for clustering (default: 0.85)
            outlier_threshold: Minimum support count to keep comment (default: 2)
        """
        config = get_settings()
        self.model = model or config.default_model
        self.similarity_threshold = similarity_threshold
        self.outlier_threshold = outlier_threshold

        # Initialize embedding service for clustering
        self.embedding_service = get_embedding_service()

        # Agno agent for semantic merging
        self.agno_agent: AgnoAgent | None = None
        self._last_metadata: dict | None = None

        logger.info(
            "CompressorAgent initialized: model=%s, similarity=%.2f, outlier_threshold=%d",
            self.model,
            self.similarity_threshold,
            self.outlier_threshold,
        )

    def compress_comments(
        self,
        meeting_id: UUID,
        round_num: int,
    ) -> CompressionMetrics:
        """Compress comments for a specific meeting round using three-tier strategy.

        Compression Flow:
        1. Load comments from database (round_num)
        2. Tier 1: Cluster similar comments using embeddings
        3. Tier 2: Merge clusters using LLM
        4. Tier 3: Remove outliers (low support)
        5. Update database (mark merged, update support counts)
        6. Return compression metrics

        Args:
            meeting_id: Meeting UUID
            round_num: Round number to compress

        Returns:
            CompressionMetrics with compression statistics

        Raises:
            ValueError: If no comments found for the round
            RuntimeError: If compression fails
        """
        logger.info(
            "Starting compression for meeting %s, round %d",
            meeting_id,
            round_num,
        )

        with get_sync_db() as db:
            # Load comments for this round
            comments = list(
                db.query(Comment)
                .filter(
                    Comment.meeting_id == meeting_id,
                    Comment.round == round_num,
                    Comment.is_merged == False,  # noqa: E712
                )
                .all()
            )

            if not comments:
                logger.warning(
                    "No comments to compress for meeting %s, round %d",
                    meeting_id,
                    round_num,
                )
                return CompressionMetrics(0, 0, 0, 0)

            original_count = len(comments)
            logger.info("Loaded %d comments for compression", original_count)

            # Tier 1: Similarity-based clustering
            clusters = self._tier1_cluster_comments(comments)
            logger.info("Tier 1: Formed %d clusters", len(clusters))

            # Tier 2: LLM semantic merge
            merged_comments = self._tier2_semantic_merge(clusters, meeting_id, round_num, db)
            logger.info("Tier 2: Created %d merged comments", len(merged_comments))

            # Tier 3: Outlier removal
            final_comments = self._tier3_remove_outliers(merged_comments)
            outliers_removed = len(merged_comments) - len(final_comments)
            logger.info("Tier 3: Removed %d outliers", outliers_removed)

            # Commit changes
            db.commit()

            metrics = CompressionMetrics(
                original_count=original_count,
                compressed_count=len(final_comments),
                clusters_formed=len(clusters),
                outliers_removed=outliers_removed,
            )

            logger.info("Compression complete: %s", metrics)
            return metrics

    def _tier1_cluster_comments(
        self,
        comments: list[Comment],
    ) -> list[list[Comment]]:
        """Tier 1: Cluster similar comments using cosine similarity.

        Uses embedding service to compute similarity matrix, then creates
        clusters of comments with similarity >= threshold.

        Graph-Based Clustering Algorithm:
        1. Compute pairwise similarity matrix
        2. Model as graph: nodes = comment IDs, edges = similarity >= threshold
        3. Find connected components (clusters) using networkx
        4. More efficient O(N+E) vs O(N^2) and handles transitive overlaps correctly

        Args:
            comments: List of Comment objects to cluster

        Returns:
            List of comment clusters (each cluster is a list of Comment objects)
        """
        # Extract comment IDs for similarity matrix
        comment_ids = [c.id for c in comments]

        # Compute similarity matrix using embedding service
        similarity_matrix = self.embedding_service.compute_similarity_matrix(
            comment_ids=comment_ids,
            threshold=self.similarity_threshold,
        )

        # Build graph: nodes = comment IDs, edges = similarity >= threshold
        graph = nx.Graph()
        graph.add_nodes_from(comment_ids)

        # Add edges for similar comments
        for comment_id, similar_ids in similarity_matrix.items():
            for similar_id in similar_ids:
                graph.add_edge(comment_id, similar_id)

        # Find connected components (clusters)
        clusters = list(nx.connected_components(graph))

        # Convert cluster IDs to Comment objects
        comment_map = {c.id: c for c in comments}
        comment_clusters = [
            [comment_map[cid] for cid in cluster_ids if cid in comment_map]
            for cluster_ids in clusters
        ]

        # Separate multi-comment clusters from singletons
        multi_clusters = [c for c in comment_clusters if len(c) > 1]
        singletons = [c for c in comment_clusters if len(c) == 1]

        # Return multi-clusters first, then singletons
        return multi_clusters + singletons

    def _tier2_semantic_merge(
        self,
        clusters: list[list[Comment]],
        meeting_id: UUID,
        round_num: int,
        db,
    ) -> list[Comment]:
        """Tier 2: Merge comment clusters using LLM semantic understanding.

        For each cluster:
        - If cluster size = 1: Keep comment as-is
        - If cluster size > 1: Use LLM to merge into concise summary

        LLM Merge Prompt:
        - Extract common themes and key insights
        - Preserve unique perspectives
        - Generate concise merged comment
        - Maintain original comment categories

        Args:
            clusters: List of comment clusters
            meeting_id: Meeting UUID
            round_num: Round number
            db: Database session

        Returns:
            List of merged Comment objects (includes singletons and merged clusters)
        """
        merged_comments = []

        for cluster in clusters:
            if len(cluster) == 1:
                # Singleton: keep as-is with support_count = 1
                comment = cluster[0]
                comment.support_count = 1
                merged_comments.append(comment)
                continue

            # Multi-comment cluster: merge using LLM
            merged_comment = self._merge_cluster_with_llm(
                cluster, meeting_id, round_num, db
            )
            merged_comments.append(merged_comment)

            # Mark original comments as merged
            for original in cluster:
                original.is_merged = True

        return merged_comments

    def _merge_cluster_with_llm(
        self,
        cluster: list[Comment],
        meeting_id: UUID,
        round_num: int,
        db,
    ) -> Comment:
        """Merge a cluster of comments using LLM semantic understanding.

        Creates Agno agent with merge instructions and generates
        a concise summary preserving key insights.

        Args:
            cluster: List of Comment objects to merge
            meeting_id: Meeting UUID
            round_num: Round number
            db: Database session

        Returns:
            New Comment object with merged content
        """
        # Lazy initialize Agno agent
        if self.agno_agent is None:
            self.agno_agent = create_agno_agent(
                name="CompressorAgent",
                instructions=(
                    "You are a compression agent that merges similar comments into concise summaries. "
                    "Your goal is to preserve key insights while reducing redundancy. "
                    "Extract common themes, preserve unique perspectives, and generate a single merged comment. "
                    "Keep the merged comment concise (1-3 sentences) while retaining information value."
                ),
                model=self.model,
                session_id=f"compressor-{meeting_id}",
            )

        # Build merge prompt
        comment_texts = [f"- {c.text} ({c.agent_name})" for c in cluster]
        merge_prompt = (
            f"Merge these {len(cluster)} similar comments into a single concise summary:\n\n"
            + "\n".join(comment_texts)
            + "\n\nProvide ONLY the merged comment text, nothing else."
        )

        # Execute merge
        merged_text = self.agno_agent.run(merge_prompt)
        self._last_metadata = extract_agno_metrics(self.agno_agent)

        # Determine merged category (most common category in cluster)
        category_counts = defaultdict(int)
        for comment in cluster:
            category_counts[comment.category] += 1
        merged_category = max(category_counts.items(), key=lambda x: x[1])[0]

        # Calculate average novelty score
        avg_novelty = sum(c.novelty_score for c in cluster) / len(cluster)

        # Create merged comment
        merged_comment = Comment(
            meeting_id=meeting_id,
            response_id=cluster[0].response_id,  # Use first comment's response
            round=round_num,
            agent_name="CompressorAgent",
            text=merged_text,
            category=merged_category,
            novelty_score=avg_novelty,
            support_count=len(cluster),  # Support = number of merged comments
            is_merged=False,  # This is the result of merging, not merged itself
        )

        db.add(merged_comment)

        logger.debug(
            "Merged %d comments into: %s (support=%d)",
            len(cluster),
            merged_text[:50],
            len(cluster),
        )

        return merged_comment

    def _tier3_remove_outliers(
        self,
        comments: list[Comment],
    ) -> list[Comment]:
        """Tier 3: Remove outlier comments with low support count.

        Filters out comments with support_count < outlier_threshold.
        This removes rare, unsupported ideas while preserving consensus.

        Args:
            comments: List of Comment objects

        Returns:
            Filtered list of Comment objects (outliers removed)
        """
        filtered_comments = [
            c for c in comments if c.support_count >= self.outlier_threshold
        ]

        removed_count = len(comments) - len(filtered_comments)
        if removed_count > 0:
            logger.debug(
                "Removed %d outliers (support < %d)",
                removed_count,
                self.outlier_threshold,
            )

        return filtered_comments

    def get_last_metadata(self) -> dict:
        """Get metadata from last LLM merge operation.

        Returns:
            Dict with tokens_used, cost, model keys
        """
        return self._last_metadata or {"tokens_used": 0, "cost": 0.0, "model": self.model}
