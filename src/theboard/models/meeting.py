"""Database models for meetings, agents, and responses."""

from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from theboard.models.base import Base


class Meeting(Base):
    """Represents a brainstorming meeting session."""

    __tablename__ = "meetings"

    # Primary key
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Meeting details
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    strategy: Mapped[str] = mapped_column(
        String(20), nullable=False, default="sequential", server_default="sequential"
    )
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    current_round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Meeting state
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="created", server_default="created"
    )
    convergence_detected: Mapped[bool] = mapped_column(default=False, server_default="false")
    stopping_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_override: Mapped[str | None] = mapped_column(String(100), nullable=True)  # CLI --model flag

    # Metadata
    context_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_comments: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost: Mapped[float] = mapped_column(default=0.0, server_default="0.0")

    # Relationships
    responses: Mapped[list["Response"]] = relationship(
        "Response", back_populates="meeting", cascade="all, delete-orphan"
    )
    comments: Mapped[list["Comment"]] = relationship(
        "Comment", back_populates="meeting", cascade="all, delete-orphan"
    )
    convergence_metrics: Mapped[list["ConvergenceMetric"]] = relationship(
        "ConvergenceMetric", back_populates="meeting", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        CheckConstraint(
            "status IN ('created', 'running', 'paused', 'completed', 'failed')",
            name="ck_meeting_status",
        ),
        CheckConstraint("strategy IN ('sequential', 'greedy')", name="ck_meeting_strategy"),
        CheckConstraint("current_round >= 0", name="ck_meeting_current_round"),
        CheckConstraint("max_rounds > 0", name="ck_meeting_max_rounds"),
        Index("ix_meetings_status", "status"),
        Index("ix_meetings_created_at", "created_at"),
    )


class Agent(Base):
    """Represents an AI agent in the agent pool."""

    __tablename__ = "agents"

    # Primary key
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Agent identity
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    expertise: Mapped[str] = mapped_column(Text, nullable=False)
    persona: Mapped[str | None] = mapped_column(Text, nullable=True)
    background: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Agent configuration
    agent_type: Mapped[str] = mapped_column(
        String(20), nullable=False, default="plaintext", server_default="plaintext"
    )
    letta_definition: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")

    # Model assignment
    default_model: Mapped[str] = mapped_column(
        String(50), nullable=False, default="deepseek", server_default="deepseek"
    )

    # Relationships
    responses: Mapped[list["Response"]] = relationship("Response", back_populates="agent")
    memories: Mapped[list["AgentMemory"]] = relationship(
        "AgentMemory", back_populates="agent", cascade="all, delete-orphan"
    )
    performance_metrics: Mapped[list["AgentPerformance"]] = relationship(
        "AgentPerformance", back_populates="agent", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        CheckConstraint("agent_type IN ('plaintext', 'letta')", name="ck_agent_type"),
        Index("ix_agents_name", "name"),
        Index("ix_agents_is_active", "is_active"),
    )


class Response(Base):
    """Represents an agent's response in a meeting round."""

    __tablename__ = "responses"

    # Primary key
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign keys
    meeting_id: Mapped[UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False
    )
    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )

    # Response details
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Metadata
    model_used: Mapped[str] = mapped_column(String(50), nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost: Mapped[float] = mapped_column(default=0.0, server_default="0.0")
    context_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relationships
    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="responses")
    agent: Mapped["Agent"] = relationship("Agent", back_populates="responses")
    comments: Mapped[list["Comment"]] = relationship(
        "Comment", back_populates="response", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_responses_meeting_id", "meeting_id"),
        Index("ix_responses_agent_id", "agent_id"),
        Index("ix_responses_round", "round"),
        Index("ix_responses_meeting_round", "meeting_id", "round"),
    )


class Comment(Base):
    """Represents an extracted comment/idea from a response."""

    __tablename__ = "comments"

    # Primary key
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign keys
    meeting_id: Mapped[UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False
    )
    response_id: Mapped[UUID] = mapped_column(
        ForeignKey("responses.id", ondelete="CASCADE"), nullable=False
    )

    # Comment details
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)

    # Metrics
    novelty_score: Mapped[float] = mapped_column(default=0.0, server_default="0.0")
    support_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_merged: Mapped[bool] = mapped_column(default=False, server_default="false")
    merged_from_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Embedding reference
    embedding_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="comments")
    response: Mapped["Response"] = relationship("Response", back_populates="comments")

    # Indexes
    __table_args__ = (
        CheckConstraint(
            "category IN ('technical_decision', 'risk', 'implementation_detail', "
            "'question', 'concern', 'suggestion', 'other')",
            name="ck_comment_category",
        ),
        Index("ix_comments_meeting_id", "meeting_id"),
        Index("ix_comments_response_id", "response_id"),
        Index("ix_comments_round", "round"),
        Index("ix_comments_category", "category"),
        Index("ix_comments_is_merged", "is_merged"),
    )


class ConvergenceMetric(Base):
    """Tracks convergence metrics per meeting round."""

    __tablename__ = "convergence_metrics"

    # Primary key
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign keys
    meeting_id: Mapped[UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False
    )

    # Metrics
    round: Mapped[int] = mapped_column(Integer, nullable=False)
    novelty_score: Mapped[float] = mapped_column(nullable=False)
    comment_count: Mapped[int] = mapped_column(Integer, nullable=False)
    unique_comment_count: Mapped[int] = mapped_column(Integer, nullable=False)
    compression_ratio: Mapped[float | None] = mapped_column(nullable=True)
    context_size: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    meeting: Mapped["Meeting"] = relationship("Meeting", back_populates="convergence_metrics")

    # Indexes
    __table_args__ = (
        Index("ix_convergence_metrics_meeting_id", "meeting_id"),
        Index("ix_convergence_metrics_round", "round"),
        Index("ix_convergence_metrics_meeting_round", "meeting_id", "round", unique=True),
    )


class AgentMemory(Base):
    """Stores agent memory for cross-meeting recall (Letta integration)."""

    __tablename__ = "agent_memory"

    # Primary key
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign keys
    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )

    # Memory details
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)
    memory_key: Mapped[str] = mapped_column(String(200), nullable=False)
    memory_value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Context
    related_meeting_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True
    )
    related_topic: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="memories")

    # Indexes
    __table_args__ = (
        CheckConstraint(
            "memory_type IN ('previous_meeting', 'learned_pattern', 'decision')",
            name="ck_agent_memory_type",
        ),
        Index("ix_agent_memory_agent_id", "agent_id"),
        Index("ix_agent_memory_type", "memory_type"),
        Index("ix_agent_memory_key", "memory_key"),
    )


class AgentPerformance(Base):
    """Tracks agent performance metrics per meeting."""

    __tablename__ = "agent_performance"

    # Primary key
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Foreign keys
    agent_id: Mapped[UUID] = mapped_column(
        ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )
    meeting_id: Mapped[UUID] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False
    )

    # Performance metrics
    engagement_score: Mapped[float] = mapped_column(default=0.0, server_default="0.0")
    peer_references: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_novelty: Mapped[float] = mapped_column(default=0.0, server_default="0.0")

    # Resource usage
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost: Mapped[float] = mapped_column(default=0.0, server_default="0.0")
    model_promoted: Mapped[bool] = mapped_column(default=False, server_default="false")
    promoted_at_round: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    agent: Mapped["Agent"] = relationship("Agent", back_populates="performance_metrics")

    # Indexes
    __table_args__ = (
        Index("ix_agent_performance_agent_id", "agent_id"),
        Index("ix_agent_performance_meeting_id", "meeting_id"),
        Index("ix_agent_performance_engagement", "engagement_score"),
        Index("ix_agent_performance_agent_meeting", "agent_id", "meeting_id", unique=True),
    )
