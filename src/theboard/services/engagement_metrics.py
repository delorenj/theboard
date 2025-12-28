"""Engagement metrics calculation for hybrid model strategy.

Sprint 4 Story 13: Hybrid Model Strategy

Calculates agent engagement scores based on:
- Peer references: How often other agents reference this agent's ideas
- Novelty: Average novelty score of agent's comments
- Comment count: Number of comments generated

Engagement formula:
    engagement = (peer_references * 0.5) + (novelty * 0.3) + (comment_count * 0.2)

Higher engagement → promote to premium models (cost optimization)
"""

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select

from theboard.database import get_sync_db
from theboard.models.meeting import Comment, Response

logger = logging.getLogger(__name__)


@dataclass
class AgentEngagement:
    """Agent engagement metrics for a meeting round.

    Attributes:
        agent_name: Agent name
        peer_references: Count of references from other agents
        avg_novelty: Average novelty score of agent's comments
        comment_count: Total comments generated
        engagement_score: Weighted engagement score (0-1 scale)
    """

    agent_name: str
    peer_references: int
    avg_novelty: float
    comment_count: int
    engagement_score: float

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AgentEngagement(agent={self.agent_name}, "
            f"engagement={self.engagement_score:.3f}, "
            f"refs={self.peer_references}, "
            f"novelty={self.avg_novelty:.2f}, "
            f"comments={self.comment_count})"
        )


class EngagementMetricsCalculator:
    """Calculator for agent engagement metrics.

    Used for hybrid model strategy to identify high-performing agents
    for model promotion (budget → premium).
    """

    # Weighting factors for engagement formula
    PEER_REFERENCE_WEIGHT = 0.5
    NOVELTY_WEIGHT = 0.3
    COMMENT_COUNT_WEIGHT = 0.2

    def __init__(self, meeting_id: UUID) -> None:
        """Initialize engagement calculator.

        Args:
            meeting_id: Meeting UUID
        """
        self.meeting_id = meeting_id

    def calculate_round_engagement(self, round_num: int) -> list[AgentEngagement]:
        """Calculate engagement metrics for all agents in a specific round.

        Args:
            round_num: Round number to analyze

        Returns:
            List of AgentEngagement sorted by engagement_score (highest first)
        """
        with get_sync_db() as db:
            # Get all responses for this round
            response_stmt = (
                select(Response)
                .where(Response.meeting_id == self.meeting_id)
                .where(Response.round == round_num)
            )
            responses = db.scalars(response_stmt).all()

            if not responses:
                logger.warning(
                    "No responses found for meeting %s round %d",
                    self.meeting_id,
                    round_num,
                )
                return []

            # Calculate metrics for each agent
            agent_metrics: list[AgentEngagement] = []

            for response in responses:
                agent_name = response.agent_name

                # 1. Calculate peer references
                # Count how many times other agents mention this agent in their comments
                peer_refs = self._count_peer_references(agent_name, round_num, db)

                # 2. Calculate average novelty
                comment_stmt = (
                    select(Comment)
                    .where(Comment.meeting_id == self.meeting_id)
                    .where(Comment.response_id == response.id)
                    .where(Comment.is_merged == False)  # noqa: E712
                )
                comments = db.scalars(comment_stmt).all()

                if comments:
                    avg_novelty = sum(c.novelty_score for c in comments) / len(comments)
                    comment_count = len(comments)
                else:
                    avg_novelty = 0.5  # Neutral novelty
                    comment_count = 0

                # 3. Calculate weighted engagement score
                # Normalize comment_count to 0-1 scale (assume max 10 comments per response)
                normalized_comments = min(comment_count / 10.0, 1.0)

                # Normalize peer_references to 0-1 scale (assume max 5 references)
                normalized_refs = min(peer_refs / 5.0, 1.0)

                engagement_score = (
                    normalized_refs * self.PEER_REFERENCE_WEIGHT
                    + avg_novelty * self.NOVELTY_WEIGHT
                    + normalized_comments * self.COMMENT_COUNT_WEIGHT
                )

                agent_metrics.append(
                    AgentEngagement(
                        agent_name=agent_name,
                        peer_references=peer_refs,
                        avg_novelty=avg_novelty,
                        comment_count=comment_count,
                        engagement_score=engagement_score,
                    )
                )

            # Sort by engagement score (highest first)
            agent_metrics.sort(key=lambda x: x.engagement_score, reverse=True)

            logger.info(
                "Engagement metrics calculated for %d agents (round %d)",
                len(agent_metrics),
                round_num,
            )

            return agent_metrics

    def _count_peer_references(self, agent_name: str, round_num: int, db) -> int:
        """Count how many times other agents reference this agent's ideas.

        Simple heuristic: count occurrences of agent name in other agents' comments.

        Args:
            agent_name: Agent to count references to
            round_num: Round number
            db: Database session

        Returns:
            Count of peer references
        """
        # Get all comments from OTHER agents in this round
        comment_stmt = (
            select(Comment)
            .where(Comment.meeting_id == self.meeting_id)
            .where(Comment.round == round_num)
            .where(Comment.agent_name != agent_name)
            .where(Comment.is_merged == False)  # noqa: E712
        )
        other_comments = db.scalars(comment_stmt).all()

        # Count occurrences of agent name in comment text (case-insensitive)
        ref_count = 0
        for comment in other_comments:
            if agent_name.lower() in comment.text.lower():
                ref_count += 1

        return ref_count

    def get_top_performers(
        self, round_num: int, top_percent: float = 0.2
    ) -> list[AgentEngagement]:
        """Get top-performing agents for model promotion.

        Args:
            round_num: Round number to analyze
            top_percent: Top percentage to select (default 0.2 = top 20%)

        Returns:
            List of top-performing AgentEngagement objects
        """
        all_metrics = self.calculate_round_engagement(round_num)

        if not all_metrics:
            return []

        # Calculate top N agents
        top_n = max(1, int(len(all_metrics) * top_percent))
        top_performers = all_metrics[:top_n]

        logger.info(
            "Top %d performers (%.0f%%) for round %d: %s",
            top_n,
            top_percent * 100,
            round_num,
            [a.agent_name for a in top_performers],
        )

        return top_performers
