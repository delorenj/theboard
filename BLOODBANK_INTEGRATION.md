# Bloodbank Integration

theboard is now integrated with the 33GOD Bloodbank event bus for real-time event streaming.

## Architecture

```
theboard CLI
  ├─> Postgres (primary persistence)
  ├─> Redis (cache)
  └─> Bloodbank (event stream)
       ↓
    RabbitMQ Event Bus
       ↓
    Consumers (theboardroom, analytics, etc.)
```

## Event Flow

When theboard executes meetings, it emits events to Bloodbank:

1. **meeting.created** - Meeting initialized
2. **meeting.started** - Execution began
3. **meeting.round_completed** - Each round finishes
4. **meeting.comment_extracted** - Ideas extracted
5. **meeting.converged** - Convergence detected
6. **meeting.completed** - Meeting finished
7. **meeting.failed** - Execution error

All events are prefixed with `theboard.` namespace (e.g., `theboard.meeting.created`).

## Configuration

### Prerequisites

1. **Bloodbank Repository**: Clone and accessible at `~/code/bloodbank/trunk-main`
2. **RabbitMQ**: Running instance (Docker or native)
3. **Redis**: Required for Bloodbank correlation tracking (optional for theboard)

### Enable Event Emission

**Option 1: Environment Variable**

```bash
export THEBOARD_EVENT_EMITTER=rabbitmq
export THEBOARD_RABBITMQ_URL="amqp://theboard:theboard_rabbit_pass@localhost:5672/"
```

**Option 2: Config File** (`~/.config/theboard/config.yml`)

```yaml
event_emitter: rabbitmq
rabbitmq_url: "amqp://theboard:theboard_rabbit_pass@localhost:5672/"
rabbitmq_host: localhost
rabbitmq_port: 5672
rabbitmq_user: theboard
rabbitmq_password: theboard_rabbit_pass
```

**Option 3: .env File** (project root or CWD)

```env
THEBOARD_EVENT_EMITTER=rabbitmq
THEBOARD_RABBITMQ_URL=amqp://theboard:theboard_rabbit_pass@localhost:5672/
```

### Verify Integration

Run a test meeting and check Bloodbank receives events:

```bash
# Terminal 1: Start Bloodbank event watcher
cd ~/code/bloodbank/trunk-main
uv run python -m event_producers.watch

# Terminal 2: Run theboard meeting
cd ~/code/theboard
theboard meeting run "test integration with bloodbank" --max-rounds 2

# You should see events flowing in Terminal 1:
# theboard.meeting.created
# theboard.meeting.started
# theboard.meeting.round_completed
# theboard.meeting.completed
```

### Docker Compose Integration

Add to `~/docker/trunk-main/stacks/ai/docker-compose.yml`:

```yaml
services:
  theboard-api:
    build: /home/delorenj/code/theboard/trunk-main
    container_name: theboard-api
    environment:
      - THEBOARD_EVENT_EMITTER=rabbitmq
      - THEBOARD_RABBITMQ_URL=amqp://bloodbank:bloodbank_pass@rabbitmq:5672/
      - DATABASE_URL=postgresql+psycopg://theboard:pass@postgres:5432/theboard
      - REDIS_URL=redis://:pass@redis:6379/0
    networks:
      - ai-network
      - bloodbank-network
    depends_on:
      - rabbitmq
      - postgres
      - redis

networks:
  bloodbank-network:
    external: true
```

## Event Schemas

All theboard events include:

```json
{
  "event_id": "uuid",
  "event_type": "theboard.meeting.created",
  "timestamp": "2026-01-05T12:00:00Z",
  "version": "1.0.0",
  "source": {
    "host": "hostname",
    "type": "SYSTEM",
    "app": "theboard",
    "meta": {"version": "1.0.0"}
  },
  "correlation_ids": [],
  "payload": {
    "meeting_id": "uuid",
    "topic": "...",
    ...
  }
}
```

### Payload Schemas

See `/home/delorenj/code/bloodbank/trunk-main/event_producers/events/domains/theboard.py` for complete payload definitions.

**Example: MeetingCreatedPayload**

```python
{
  "meeting_id": UUID,
  "topic": str,
  "strategy": str,  # 'sequential' or 'greedy'
  "max_rounds": int,
  "agent_count": int | None
}
```

## Consuming Events

### Subscribe to All TheBoard Events

```python
from event_producers.events.core.consumer import EventConsumer

consumer = EventConsumer(
    queue_name="my-service",
    binding_keys=["theboard.#"]  # All theboard events
)

async def handle_event(envelope):
    if envelope.event_type == "theboard.meeting.completed":
        # Process completed meeting
        meeting_id = envelope.payload["meeting_id"]
        total_cost = envelope.payload["total_cost"]
        print(f"Meeting {meeting_id} completed, cost: ${total_cost}")

await consumer.start(handle_event)
```

### Subscribe to Specific Events

```python
# Only meeting completion events
consumer = EventConsumer(
    queue_name="analytics",
    binding_keys=["theboard.meeting.completed"]
)

# Only round-level events
consumer = EventConsumer(
    queue_name="round-tracker",
    binding_keys=["theboard.meeting.round_completed"]
)
```

## Troubleshooting

### Events Not Appearing

1. **Check Event Emitter Setting**:
   ```bash
   python -c "from theboard.config import settings; print(settings.event_emitter)"
   # Should print: rabbitmq
   ```

2. **Check RabbitMQ Connection**:
   ```bash
   python -c "from theboard.events.emitter import get_event_emitter; e = get_event_emitter(); print(type(e))"
   # Should print: <class 'theboard.events.bloodbank_emitter.RabbitMQEventEmitter'>
   ```

3. **Check Bloodbank Path**:
   ```bash
   ls ~/code/bloodbank/trunk-main/rabbit.py
   # Should exist
   ```

4. **Enable Debug Logging**:
   ```yaml
   # config.yml
   debug: true
   log_level: DEBUG
   ```

### Fallback Behavior

If RabbitMQ connection fails, theboard automatically falls back to `NullEventEmitter` (no events emitted) and logs a warning. Meetings continue to work normally, storing data in Postgres/Redis.

## Future Extensions

Planned events for theboardroom visualization:

- `theboard.meeting.participant.added`
- `theboard.meeting.participant.turn.completed`
- `theboard.meeting.artifact.created` (via 33god artifact command)

See `/home/delorenj/code/bloodbank/trunk-main/event_producers/events/domains/theboard.py` for placeholder schemas.

## Related Documentation

- Bloodbank README: `/home/delorenj/code/bloodbank/trunk-main/README.md`
- Event Registry: `/home/delorenj/code/bloodbank/trunk-main/event_producers/events/registry.py`
- TheBoard Event Schemas: `/home/delorenj/code/theboard/src/theboard/events/schemas.py`
