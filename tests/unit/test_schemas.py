"""Tests for Pydantic schemas."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from theboard.schemas import (
    Comment,
    CommentCategory,
    CommentList,
    MeetingCreate,
    StrategyType,
)


def test_meeting_create_valid() -> None:
    """Test valid meeting creation schema."""
    meeting = MeetingCreate(
        topic="Test topic for brainstorming",
        strategy=StrategyType.SEQUENTIAL,
        max_rounds=3,
        agent_names=["agent1", "agent2"],
        auto_select_agents=False,
    )
    assert meeting.topic == "Test topic for brainstorming"
    assert meeting.strategy == StrategyType.SEQUENTIAL
    assert meeting.max_rounds == 3


def test_meeting_create_topic_too_short() -> None:
    """Test that topic must be at least 10 characters."""
    with pytest.raises(ValidationError):
        MeetingCreate(topic="short")


def test_meeting_create_defaults() -> None:
    """Test default values for meeting creation."""
    meeting = MeetingCreate(topic="Test topic for meeting")
    assert meeting.strategy == StrategyType.SEQUENTIAL
    assert meeting.max_rounds == 5
    assert meeting.agent_names == []
    assert meeting.auto_select_agents is False
    assert meeting.agent_count == 5


def test_comment_valid() -> None:
    """Test valid comment schema."""
    comment = Comment(
        text="This is a valid comment with enough text",
        category=CommentCategory.TECHNICAL_DECISION,
        novelty_score=0.8,
    )
    assert comment.text == "This is a valid comment with enough text"
    assert comment.category == CommentCategory.TECHNICAL_DECISION
    assert comment.novelty_score == 0.8


def test_comment_text_too_short() -> None:
    """Test that comment text must be at least 10 characters."""
    with pytest.raises(ValidationError):
        Comment(text="short", category=CommentCategory.OTHER)


def test_comment_list_valid() -> None:
    """Test comment list schema."""
    comment_list = CommentList(
        comments=[
            Comment(
                text="First comment with enough characters",
                category=CommentCategory.RISK,
            ),
            Comment(
                text="Second comment with enough characters",
                category=CommentCategory.SUGGESTION,
            ),
        ]
    )
    assert len(comment_list.comments) == 2
    assert comment_list.comments[0].category == CommentCategory.RISK
