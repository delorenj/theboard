"""Event schemas for TheBoard event-driven architecture.

All events are immutable pydantic models with strict typing and validation.
Events follow a consistent structure:
- event_type: Discriminator for event routing
- timestamp: ISO 8601 UTC timestamp
- meeting_id: UUID of affected meeting
- payload: Event-specific data

Design Philosophy:
- Immutable events (frozen=True)
- Explicit typing (no implicit conversions)
- Self-contained (all context in payload)
- Traceable (meeting_id + timestamp)
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Base event schema for all TheBoard events.

    Provides common fields for event routing, tracing, and correlation.
    """

    model_config = {"frozen": True}

    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    meeting_id: UUID


class MeetingCreatedEvent(BaseEvent):
    """Emitted when a new meeting is created.

    Payload contains initial meeting configuration.
    """

    event_type: Literal["meeting.created"] = "meeting.created"
    topic: str
    strategy: str
    max_rounds: int
    agent_count: int | None = None


class MeetingStartedEvent(BaseEvent):
    """Emitted when a meeting transitions to RUNNING status.

    Payload contains selected agents and execution configuration.
    """

    event_type: Literal["meeting.started"] = "meeting.started"
    selected_agents: list[str]  # Agent names
    agent_count: int


class RoundCompletedEvent(BaseEvent):
    """Emitted when a meeting round completes.

    Payload contains round metrics and convergence indicators.
    """

    event_type: Literal["meeting.round_completed"] = "meeting.round_completed"
    round_num: int
    agent_name: str
    response_length: int
    comment_count: int
    avg_novelty: float
    tokens_used: int
    cost: float


class CommentExtractedEvent(BaseEvent):
    """Emitted when comments are extracted from agent response.

    Payload contains comment metadata for analytics.
    """

    event_type: Literal["meeting.comment_extracted"] = "meeting.comment_extracted"
    round_num: int
    agent_name: str
    comment_text: str
    category: str
    novelty_score: float


class MeetingConvergedEvent(BaseEvent):
    """Emitted when meeting reaches convergence.

    Payload contains convergence metrics and stopping criteria.
    """

    event_type: Literal["meeting.converged"] = "meeting.converged"
    round_num: int
    avg_novelty: float
    novelty_threshold: float
    total_comments: int


class MeetingCompletedEvent(BaseEvent):
    """Emitted when meeting completes successfully.

    Payload contains final meeting state and metrics.
    """

    event_type: Literal["meeting.completed"] = "meeting.completed"
    total_rounds: int
    total_comments: int
    total_cost: float
    convergence_detected: bool
    stopping_reason: str


class MeetingFailedEvent(BaseEvent):
    """Emitted when meeting execution fails.

    Payload contains error context for debugging.
    """

    event_type: Literal["meeting.failed"] = "meeting.failed"
    error_type: str
    error_message: str
    round_num: int | None = None
    agent_name: str | None = None


# Sprint 4 Story 12: Additional events for human-in-loop


class AgentResponseReadyEvent(BaseEvent):
    """Emitted when agent completes a response and is ready for comments.

    Payload contains response metadata for tracking and human intervention.
    """

    event_type: Literal["agent.response.ready"] = "agent.response.ready"
    round_num: int
    agent_name: str
    response_length: int
    tokens_used: int
    cost: float


class ContextCompressionTriggeredEvent(BaseEvent):
    """Emitted when context compression is triggered.

    Payload contains compression trigger context and metrics.
    """

    event_type: Literal["context.compression.triggered"] = "context.compression.triggered"
    round_num: int
    context_size_before: int
    compression_threshold: int
    trigger_reason: str


class MeetingHumanInputNeededEvent(BaseEvent):
    """Emitted when meeting requires human intervention.

    Payload contains context for decision and available options.
    Used for human-in-loop interactive prompts.
    """

    event_type: Literal["meeting.human.input.needed"] = "meeting.human.input.needed"
    round_num: int
    reason: str  # Why human input is needed
    prompt_text: str  # Question to display to user
    options: list[str]  # Available choices: ["continue", "pause", "modify_context", "stop"]
    timeout_seconds: int = 300  # Default: 5 minutes
