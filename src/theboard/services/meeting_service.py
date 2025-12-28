"""Service layer for meeting management."""

import logging
from uuid import UUID

from sqlalchemy import delete, desc, select
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
    model_override: str | None = None,
    hybrid_models: bool = False,
) -> MeetingResponse:
    """Create a new brainstorming meeting.

    Args:
        topic: The brainstorming topic
        strategy: Execution strategy (sequential or greedy)
        max_rounds: Maximum number of rounds
        agent_count: Number of agents to select (if auto_select)
        auto_select: Whether to auto-select agents based on topic
        model_override: CLI model override (--model flag)
        hybrid_models: Enable hybrid model strategy (Story 13)

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
            # Create meeting with model override and hybrid models flag
            meeting = Meeting(
                topic=topic,
                strategy=strategy.value,
                max_rounds=max_rounds,
                current_round=0,
                status=MeetingStatus.CREATED.value,
                convergence_detected=False,
                model_override=model_override,  # Store CLI override for workflow
                hybrid_models=hybrid_models,  # Sprint 4 Story 13
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


def run_meeting(meeting_id: UUID, interactive: bool, rerun: bool = False) -> MeetingResponse:
    """Run a brainstorming meeting.

    Args:
        meeting_id: Meeting UUID
        interactive: Enable human-in-the-loop prompts
        rerun: Reset completed/failed meetings to rerun them

    Returns:
        MeetingResponse with final meeting state

    Raises:
        ValueError: If meeting not found or validation fails
    """
    import asyncio

    from theboard.workflows.multi_agent_meeting import MultiAgentMeetingWorkflow
    from theboard.workflows.simple_meeting import SimpleMeetingWorkflow

    # SESSION LEAK FIX: Validate and update meeting status, then close session
    # before running workflow (workflow opens its own sessions as needed)
    meeting = None
    with get_sync_db() as db:
        try:
            # Get meeting
            stmt = select(Meeting).where(Meeting.id == meeting_id)
            meeting = db.scalars(stmt).first()

            if not meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            # Handle rerun: reset completed/failed meetings
            if rerun and meeting.status in [MeetingStatus.COMPLETED.value, MeetingStatus.FAILED.value]:
                logger.info("Resetting meeting %s for rerun", meeting_id)
                meeting.status = MeetingStatus.CREATED.value
                meeting.current_round = 0
                meeting.stopping_reason = None
                meeting.convergence_detected = False
                meeting.context_size = 0
                meeting.total_comments = 0
                meeting.total_cost = 0.0

                # Delete previous responses, comments, and metrics
                db.execute(delete(Response).where(Response.meeting_id == meeting_id))
                db.execute(delete(Comment).where(Comment.meeting_id == meeting_id))
                db.execute(delete(ConvergenceMetric).where(ConvergenceMetric.meeting_id == meeting_id))

                db.commit()
                db.refresh(meeting)

            if meeting.status not in [MeetingStatus.CREATED.value, MeetingStatus.PAUSED.value]:
                raise ValueError(f"Meeting cannot be run in status: {meeting.status}")

            # Update status and extract model override before closing session
            meeting.status = MeetingStatus.RUNNING.value
            model_override = meeting.model_override
            db.commit()

            logger.info("Running meeting %s", meeting_id)

        except Exception as e:
            # Handle validation errors before workflow execution
            if meeting:
                meeting.status = MeetingStatus.FAILED.value
                db.commit()
            logger.exception("Failed to validate meeting for run")
            raise ValueError(f"Failed to run meeting: {e!s}") from e

    # SESSION LEAK FIX: Execute workflow WITHOUT holding database session
    # Workflow will open its own sessions as needed for each operation
    try:
        # Sprint 2: Use multi-agent workflow for all meetings
        # SimpleMeetingWorkflow kept available for single-agent testing if needed
        workflow = MultiAgentMeetingWorkflow(meeting_id, model_override=model_override)
        asyncio.run(workflow.execute())

    except Exception as e:
        # Handle workflow execution errors
        with get_sync_db() as error_db:
            stmt = select(Meeting).where(Meeting.id == meeting_id)
            failed_meeting = error_db.scalars(stmt).first()
            if failed_meeting:
                failed_meeting.status = MeetingStatus.FAILED.value
                error_db.commit()
        logger.exception("Workflow execution failed")
        raise ValueError(f"Failed to run meeting: {e!s}") from e

    # SESSION LEAK FIX: Reopen session ONLY to get final meeting state
    with get_sync_db() as result_db:
        stmt = select(Meeting).where(Meeting.id == meeting_id)
        final_meeting = result_db.scalars(stmt).first()

        if not final_meeting:
            raise ValueError(f"Meeting not found after execution: {meeting_id}")

        return MeetingResponse.model_validate(final_meeting)


def fork_meeting(meeting_id: UUID) -> MeetingResponse:
    """Fork a meeting, creating a new meeting with the same parameters.

    Args:
        meeting_id: Meeting UUID to fork from

    Returns:
        MeetingResponse with newly created meeting

    Raises:
        ValueError: If source meeting not found
    """
    with get_sync_db() as db:
        try:
            # Get source meeting
            stmt = select(Meeting).where(Meeting.id == meeting_id)
            source_meeting = db.scalars(stmt).first()

            if not source_meeting:
                raise ValueError(f"Meeting not found: {meeting_id}")

            # Create forked meeting with same parameters
            forked_meeting = Meeting(
                topic=source_meeting.topic,
                strategy=source_meeting.strategy,
                max_rounds=source_meeting.max_rounds,
                current_round=0,
                status=MeetingStatus.CREATED.value,
                convergence_detected=False,
                model_override=source_meeting.model_override,
                context_size=0,
                total_comments=0,
                total_cost=0.0,
            )

            db.add(forked_meeting)
            db.commit()
            db.refresh(forked_meeting)

            logger.info(
                "Forked meeting %s -> %s (topic: %s)",
                meeting_id,
                forked_meeting.id,
                source_meeting.topic,
            )

            return MeetingResponse.model_validate(forked_meeting)

        except Exception as e:
            db.rollback()
            logger.exception("Failed to fork meeting")
            raise ValueError(f"Failed to fork meeting: {e!s}") from e


def list_recent_meetings(limit: int = 20) -> list[MeetingResponse]:
    """Get recent meetings for selection.

    Args:
        limit: Maximum number of meetings to return (default 20)

    Returns:
        List of recent meetings, sorted by creation date descending

    Raises:
        ValueError: If query fails
    """
    with get_sync_db() as db:
        try:
            stmt = (
                select(Meeting)
                .order_by(desc(Meeting.created_at))
                .limit(limit)
            )
            meetings = db.scalars(stmt).all()

            return [MeetingResponse.model_validate(m) for m in meetings]

        except Exception as e:
            logger.exception("Failed to list recent meetings")
            raise ValueError(f"Failed to list recent meetings: {e!s}") from e


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
