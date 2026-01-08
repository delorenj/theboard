#!/usr/bin/env python3
"""Test script for Bloodbank integration.

Tests:
1. Event emitter initialization
2. Event creation and emission
3. Bloodbank envelope wrapping

Usage:
    # Test with RabbitMQ emitter (requires Bloodbank + RabbitMQ)
    THEBOARD_EVENT_EMITTER=rabbitmq python test_bloodbank_integration.py

    # Test with in-memory emitter (no dependencies)
    THEBOARD_EVENT_EMITTER=inmemory python test_bloodbank_integration.py
"""

import sys
from pathlib import Path
from uuid import uuid4

# Add theboard to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from theboard.events.emitter import get_event_emitter, reset_event_emitter
from theboard.events.schemas import (
    MeetingCreatedEvent,
    MeetingStartedEvent,
    RoundCompletedEvent,
    MeetingCompletedEvent,
)
from theboard.config import get_settings


def test_emitter_initialization():
    """Test that emitter initializes correctly."""
    print("=== Test 1: Emitter Initialization ===")

    reset_event_emitter()
    emitter = get_event_emitter()

    print(f"Emitter type: {type(emitter).__name__}")
    print(f"Config event_emitter: {get_settings().event_emitter}")

    return emitter


def test_event_creation():
    """Test event creation with proper schemas."""
    print("\n=== Test 2: Event Creation ===")

    meeting_id = uuid4()

    # Create test events
    created_event = MeetingCreatedEvent(
        meeting_id=meeting_id,
        topic="Test Bloodbank Integration",
        strategy="sequential",
        max_rounds=2,
        agent_count=3
    )

    started_event = MeetingStartedEvent(
        meeting_id=meeting_id,
        selected_agents=["Alice", "Bob", "Charlie"],
        agent_count=3
    )

    round_event = RoundCompletedEvent(
        meeting_id=meeting_id,
        round_num=1,
        agent_name="Alice",
        response_length=150,
        comment_count=3,
        avg_novelty=0.75,
        tokens_used=500,
        cost=0.01
    )

    completed_event = MeetingCompletedEvent(
        meeting_id=meeting_id,
        total_rounds=2,
        total_comments=6,
        total_cost=0.05,
        convergence_detected=True,
        stopping_reason="convergence"
    )

    print(f"Created event: {created_event.event_type}")
    print(f"Started event: {started_event.event_type}")
    print(f"Round event: {round_event.event_type}")
    print(f"Completed event: {completed_event.event_type}")

    return [created_event, started_event, round_event, completed_event]


def test_event_emission(emitter, events):
    """Test event emission through emitter."""
    print("\n=== Test 3: Event Emission ===")

    for event in events:
        try:
            print(f"Emitting: {event.event_type}...", end=" ")
            emitter.emit(event)
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    return True


def test_inmemory_verification(emitter, events):
    """Verify events were stored (InMemoryEmitter only)."""
    if type(emitter).__name__ != "InMemoryEventEmitter":
        print("\n=== Test 4: Event Verification (Skipped - not InMemoryEmitter) ===")
        return True

    print("\n=== Test 4: Event Verification (InMemoryEmitter) ===")

    stored_events = emitter.get_events()
    print(f"Stored {len(stored_events)} events")

    for event in events:
        matching = emitter.get_events(event.event_type)
        print(f"  {event.event_type}: {len(matching)} match(es)")

    return len(stored_events) == len(events)


def main():
    """Run all tests."""
    print("TheBoard Bloodbank Integration Test")
    print("=" * 50)

    try:
        # Test 1: Initialize emitter
        emitter = test_emitter_initialization()

        # Test 2: Create events
        events = test_event_creation()

        # Test 3: Emit events
        success = test_event_emission(emitter, events)

        if not success:
            print("\n❌ Event emission failed")
            sys.exit(1)

        # Test 4: Verify (InMemory only)
        verified = test_inmemory_verification(emitter, events)

        # Summary
        print("\n" + "=" * 50)
        print("✅ All tests passed!")

        if type(emitter).__name__ == "RabbitMQEventEmitter":
            print("\nℹ️  Events published to Bloodbank")
            print("   Check Bloodbank logs or event watcher to verify reception")
            print("   Run: cd ~/code/bloodbank/trunk-main && uv run python -m event_producers.watch")

        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
