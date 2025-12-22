"""Service layer for meeting management."""

import logging
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.orm import joinedload

from theboard.database import get_sync_db
from theboard.models.meeting import Comment, ConvergenceMetric, Meeting, Response
from theboard.schemas import (
    CommentResponse,
    ConvergenceMetricResponse,
    MeetingResponse,
    MeetingStatus,
    MeetingStatusResponse,
    ResponseSummary,
    StrategyType,
)
from theboard.utils.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)


def create_meeting(
    topic: str,
    strategy: StrategyType,
    max_rounds: int,
    agent_count: int,
    auto_select: bool,
) -> MeetingResponse:
    """Create a new brainstorming meeting.

    Args:
        topic: The brainstorming topic
        strategy: Execution strategy (sequential or greedy)
        max_rounds: Maximum number of rounds
        agent_count: Number of agents to select (if auto_select)
        auto_select: Whether to auto-select agents based on topic

    Returns:
        MeetingResponse with created meeting details

    Raises:
        ValueError: If validation fails
    """
    # Validate inputs
    if not (10 <= len(topic) <= 500):
        raise ValueError("Topic must be between 10 and 500 characters")

    if not (1 <= max_rounds <= 10):
        raise ValueError("Max rounds must be between 1 and 10")

    with get_sync_db() as db:
        try:
            # Create meeting
            meeting = Meeting(
                topic=topic,
                strategy=strategy.value,
                max_rounds=max_rounds,
                current_round=0,
                status=MeetingStatus.CREATED.value,
                convergence_detected=False,
            )

            db.add(meeting)
            db.commit()
            db.refresh(meeting)

            logger.info("Created meeting %s: %s", meeting.id, topic)

            # TODO: Sprint 2 - Auto-select agents if requested
            if auto_select:
                logger.info(
                    "Auto-selection of %d agents requested (deferred to Sprint 2)",
                    agent_count,
                )

            return MeetingResponse.model_validate(meeting)

        except Exception as e:
            db.rollback()
            logger.exception("Failed to create meeting")
            raise ValueError(f"Failed to create meeting: {e!s}") from e


def run_meeting(meeting_id: UUID, interactive: bool) -> MeetingResponse:
    """Run a brainstorming meeting.

    Args:
        meeting_id: Meeting UUID
        interactive: Enable human-in-the-loop prompts

    Returns:
        MeetingResponse with final meeting state

    Raises:
        ValueError: If meeting not found or validation fails
    """
    import asyncio

    from theboard.workflows.simple_meeting import SimpleMeetingWorkflow

    with get_sync_db() as db:
        try:
            # Get meeting
            stmt = select(Meeting).where(Meeting.id == meeting_id)
            meeting = db.scalars(stmt).first()

            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            if meeting.status not in [MeetingStatus.CREATED.value, MeetingStatus.PAUSED.value]:
                raise ValueError(f"Meeting cannot be run in status: {meeting.status}")

            # Update status
            meeting.status = MeetingStatus.RUNNING.value
            db.commit()

            logger.info("Running meeting %s", meeting_id)

            # Execute workflow
            workflow = SimpleMeetingWorkflow(meeting_id)
            asyncio.run(workflow.execute())

            # Get updated meeting
            db.refresh(meeting)

            return MeetingResponse.model_validate(meeting)

        except Exception as e:
            if meeting:
                meeting.status = MeetingStatus.FAILED.value
                db.commit()
            logger.exception("Failed to run meeting")
            raise ValueError(f"Failed to run meeting: {e!s}") from e


def get_meeting_status(meeting_id: UUID) -> MeetingStatusResponse:
    """Get detailed meeting status.

    Args:
        meeting_id: Meeting UUID

    Returns:
        MeetingStatusResponse with full meeting details

    Raises:
        ValueError: If meeting not found
    """
    with get_sync_db() as db:
        try:
            # Get meeting with relationships
            stmt = (
                select(Meeting)
                .options(
                    joinedload(Meeting.responses),
                    joinedload(Meeting.comments),
                    joinedload(Meeting.convergence_metrics),
                )
                .where(Meeting.id == meeting_id)
            )
            meeting = db.scalars(stmt).first()

            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            # Get responses
            response_stmt = (
                select(Response)
                .where(Response.meeting_id == meeting_id)
                .order_by(Response.round, Response.created_at)
            )
            responses = db.scalars(response_stmt).all()

            # Get recent comments (last 10)
            comment_stmt = (
                select(Comment)
                .where(Comment.meeting_id == meeting_id)
                .order_by(desc(Comment.created_at))
                .limit(10)
            )
            comments = db.scalars(comment_stmt).all()

            # Get convergence metrics
            metric_stmt = (
                select(ConvergenceMetric)
                .where(ConvergenceMetric.meeting_id == meeting_id)
                .order_by(ConvergenceMetric.round)
            )
            metrics = db.scalars(metric_stmt).all()

            # Build response
            return MeetingStatusResponse(
                meeting=MeetingResponse.model_validate(meeting),
                responses=[
                    ResponseSummary(
                        id=r.id,
                        meeting_id=r.meeting_id,
                        agent_id=r.agent_id,
                        round=r.round,
                        agent_name=r.agent_name,
                        response_text=r.response_text,
                        model_used=r.model_used,
                        tokens_used=r.tokens_used,
                        cost=r.cost,
                        comment_count=len([c for c in comments if c.response_id == r.id]),
                        created_at=r.created_at,
                    )
                    for r in responses
                ],
                recent_comments=[CommentResponse.model_validate(c) for c in comments],
                convergence_metrics=[ConvergenceMetricResponse.model_validate(m) for m in metrics],
            )

        except Exception as e:
            logger.exception("Failed to get meeting status")
            raise ValueError(f"Failed to get meeting status: {e!s}") from e


def delete_meeting(meeting_id: UUID) -> bool:
    """Delete a meeting and all related data.

    Args:
        meeting_id: Meeting UUID

    Returns:
        True if deleted successfully

    Raises:
        ValueError: If meeting not found
    """
    redis = get_redis_manager()

    with get_sync_db() as db:
        try:
            # Get meeting
            stmt = select(Meeting).where(Meeting.id == meeting_id)
            meeting = db.scalars(stmt).first()

            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            # Delete from Redis
            redis.delete_meeting_data(str(meeting_id))

            # Delete from database (cascade will handle related records)
            db.delete(meeting)
            db.commit()

            logger.info("Deleted meeting %s", meeting_id)
            return True

        except Exception as e:
            db.rollback()
            logger.exception("Failed to delete meeting")
            raise ValueError(f"Failed to delete meeting: {e!s}") from e
