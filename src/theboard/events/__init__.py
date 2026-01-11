"""Event foundation for TheBoard event-driven architecture."""

from theboard.events.emitter import EventEmitter, get_event_emitter
from theboard.events.schemas import (
    CommentExtractedEvent,
    MeetingCompletedEvent,
    MeetingConvergedEvent,
    MeetingCreatedEvent,
    MeetingFailedEvent,
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
]
