"""Pydantic schemas for data validation and API responses."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StrategyType(str, Enum):
    """Meeting execution strategy."""

    SEQUENTIAL = "sequential"
    GREEDY = "greedy"


class MeetingStatus(str, Enum):
    """Meeting status."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class CommentCategory(str, Enum):
    """Comment category types."""

    TECHNICAL_DECISION = "technical_decision"
    RISK = "risk"
    IMPLEMENTATION_DETAIL = "implementation_detail"
    QUESTION = "question"
    CONCERN = "concern"
    SUGGESTION = "suggestion"
    OTHER = "other"


class AgentType(str, Enum):
    """Agent type."""

    PLAINTEXT = "plaintext"
    LETTA = "letta"


# Request schemas


class MeetingCreate(BaseModel):
    """Schema for creating a new meeting."""

    topic: str = Field(..., min_length=10, max_length=500)
    strategy: StrategyType = StrategyType.SEQUENTIAL
    max_rounds: int = Field(default=5, ge=1, le=10)
    agent_names: list[str] = Field(default_factory=list)
    auto_select_agents: bool = False
    agent_count: int = Field(default=5, ge=1, le=10)


class MeetingRun(BaseModel):
    """Schema for running a meeting."""

    meeting_id: UUID
    strategy: StrategyType | None = None
    max_rounds: int | None = None


# Response schemas


class CommentResponse(BaseModel):
    """Schema for comment response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    meeting_id: UUID
    response_id: UUID
    round: int
    agent_name: str
    text: str
    category: CommentCategory
    novelty_score: float
    support_count: int
    is_merged: bool
    created_at: datetime


class ResponseSummary(BaseModel):
    """Schema for response summary."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    meeting_id: UUID
    agent_id: UUID
    round: int
    agent_name: str
    response_text: str
    model_used: str
    tokens_used: int
    cost: float
    comment_count: int = 0
    created_at: datetime


class ConvergenceMetricResponse(BaseModel):
    """Schema for convergence metric response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    meeting_id: UUID
    round: int
    novelty_score: float
    comment_count: int
    unique_comment_count: int
    compression_ratio: float | None
    context_size: int
    created_at: datetime


class MeetingResponse(BaseModel):
    """Schema for meeting response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    topic: str
    strategy: StrategyType
    max_rounds: int
    current_round: int
    status: MeetingStatus
    convergence_detected: bool
    stopping_reason: str | None
    context_size: int
    total_comments: int
    total_cost: float
    created_at: datetime
    updated_at: datetime


class MeetingStatusResponse(BaseModel):
    """Schema for detailed meeting status response."""

    meeting: MeetingResponse
    responses: list[ResponseSummary] = Field(default_factory=list)
    recent_comments: list[CommentResponse] = Field(default_factory=list)
    convergence_metrics: list[ConvergenceMetricResponse] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Schema for agent response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    expertise: str
    persona: str | None
    agent_type: AgentType
    is_active: bool
    default_model: str
    created_at: datetime


# Internal schemas


class Comment(BaseModel):
    """Schema for extracted comment (used by NotetakerAgent)."""

    text: str = Field(..., min_length=10, max_length=1000)
    category: CommentCategory
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CommentList(BaseModel):
    """Schema for list of extracted comments."""

    comments: list[Comment]


class MeetingState(BaseModel):
    """Schema for meeting state stored in Redis."""

    meeting_id: UUID
    current_round: int
    current_agent: str | None
    status: MeetingStatus
    active_context: str
    context_size: int
    convergence_count: int = 0  # Consecutive rounds below threshold


class AgentConfig(BaseModel):
    """Schema for agent configuration."""

    name: str
    expertise: str
    persona: str | None = None
    background: str | None = None
    agent_type: AgentType = AgentType.PLAINTEXT
    default_model: str = "deepseek"
