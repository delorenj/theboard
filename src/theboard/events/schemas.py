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


class TopComment(BaseModel):
    """Top comment extracted from meeting for insights."""

    text: str
    category: str
    novelty_score: float
    agent_name: str
    round_num: int


class MeetingCompletedEvent(BaseEvent):
    """Emitted when meeting completes successfully.

    Payload contains final meeting state, metrics, and extracted insights.
    Phase 3A: Enhanced with insights for downstream consumption.
    """

    event_type: Literal["meeting.completed"] = "meeting.completed"

    # Meeting metrics
    total_rounds: int
    total_comments: int
    total_cost: float
    convergence_detected: bool
    stopping_reason: str

    # Insights (Phase 3A)
    top_comments: list[TopComment] = Field(
        default_factory=list,
        description="Top 5 comments ranked by novelty score"
    )
    category_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of comments by category (question, concern, idea, etc.)"
    )
    agent_participation: dict[str, int] = Field(
        default_factory=dict,
        description="Number of responses per agent"
    )


class MeetingFailedEvent(BaseEvent):
    """Emitted when meeting execution fails.

    Payload contains error context for debugging.
    """

    event_type: Literal["meeting.failed"] = "meeting.failed"
    error_type: str
    error_message: str
    round_num: int | None = None
    agent_name: str | None = None


# ============================================================================
# Service Lifecycle Events (Phase 1 - 33GOD Integration)
# ============================================================================


class ServiceRegisteredPayload(BaseModel):
    """Payload for theboard.service.registered event.

    Emitted when TheBoard service starts up and registers with Bloodbank.
    """

    service_id: str = Field(description="Service identifier")
    service_name: str = Field(description="Human-readable service name")
    version: str = Field(description="Service version")
    capabilities: list[str] = Field(
        description="List of capabilities this service provides"
    )
    endpoints: dict[str, str] = Field(
        description="Exposed endpoints (e.g., {'health': 'http://host:port/health'})"
    )


class ServiceHealthPayload(BaseModel):
    """Payload for theboard.service.health event.

    Emitted periodically to report service health status.
    """

    service_id: str = Field(description="Service identifier")
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall health status"
    )
    database: Literal["connected", "disconnected", "error"]
    redis: Literal["connected", "disconnected", "error"]
    bloodbank: Literal["connected", "disconnected", "disabled"]
    uptime_seconds: int = Field(description="Service uptime in seconds")
    details: dict[str, str] | None = None
