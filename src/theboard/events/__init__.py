"""Event foundation for TheBoard event-driven architecture."""

from theboard.events.emitter import EventEmitter, get_event_emitter
from theboard.events.schemas import (
    CommentExtractedEvent,
    ContextModifiedEvent,
    HumanInputNeededEvent,
    MeetingCompletedEvent,
    MeetingConvergedEvent,
    MeetingCreatedEvent,
    MeetingFailedEvent,
    MeetingPausedEvent,
    MeetingResumedEvent,
    MeetingStartedEvent,
    RoundCompletedEvent,
    TopComment,
)

__all__ = [
    "EventEmitter",
    "get_event_emitter",
    "MeetingCreatedEvent",
    "MeetingStartedEvent",
    "RoundCompletedEvent",
    "CommentExtractedEvent",
    "MeetingConvergedEvent",
    "MeetingCompletedEvent",
    "MeetingFailedEvent",
    "TopComment",
    # Sprint 4 Story 12: Human-in-the-loop events
    "HumanInputNeededEvent",
    "MeetingPausedEvent",
    "MeetingResumedEvent",
    "ContextModifiedEvent",
]
