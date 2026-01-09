# 33GOD Service Integration Retrospective
**Building the theboard-meeting-trigger Service**

A developer's story of integrating TheBoard with the 33GOD Bloodbank event bus and creating our first event-driven consumer service.

---

## TL;DR - Quick Wins for Future Service Devs

**Copy-paste these patterns:**

1. **Bloodbank expects `RABBIT_URL` env var** - not `RABBITMQ_URL`, not `BLOODBANK_RABBITMQ_URL`. Just `RABBIT_URL`.
2. **Import paths matter** - Use `from event_producers.rabbit import Publisher`, not `from rabbit import Publisher`
3. **TriggerType enum** - Valid values: `MANUAL`, `AGENT`, `SCHEDULED`, `FILE_WATCH`, `HOOK`. No `SYSTEM` value exists.
4. **EventConsumer API** - Use `.run()` method, not `.start()`. No lifecycle hooks like `on_start()` exist.
5. **Context field types** - When using MeetingTriggerPayload, `context: dict[str, str]` means all values must be strings, not booleans.
6. **Clear Python cache** - After fixing Bloodbank code, run `find bloodbank -name __pycache__ -type d -exec rm -rf {} +` and reinstall with `--force-reinstall`.

---

## What We Built

**Goal**: Enable ecosystem-wide meeting creation by consuming Bloodbank events.

**Result**: A working consumer service that listens for `theboard.meeting.trigger` events and auto-creates meetings in TheBoard's database.

**Stack**:
- Bloodbank EventConsumer framework (RabbitMQ + aio-pika)
- Direct SQLAlchemy database access to TheBoard's PostgreSQL
- Event routing via topic exchange (`bloodbank.events.v1`)
- Service running at `localhost:5673` (RabbitMQ) and `localhost:5433` (PostgreSQL)

**Files Created**:
```
/home/delorenj/code/33GOD/services/theboard-meeting-trigger/
├── src/theboard_meeting_trigger/
│   ├── consumer_simple.py          # EventConsumer implementation
│   ├── meeting_creator.py          # Direct database meeting creation
│   ├── models.py                   # Pydantic event payload schemas
│   └── config.py                   # Service configuration
├── pyproject.toml                  # Dependencies (bloodbank, sqlalchemy, psycopg)
└── .env                            # RABBIT_URL + database connection
```

---

## 🎉 What Went Well

### 1. Convention-Over-Configuration EventConsumer Pattern

Bloodbank's EventConsumer is beautifully simple once you understand it:

```python
class TheboardMeetingTriggerConsumer(EventConsumer):
    queue_name = "theboard_meeting_trigger_queue"
    routing_keys = [
        "theboard.meeting.trigger",
        "feature.brainstorm.requested",
    ]

    @EventConsumer.event_handler("theboard.meeting.trigger")
    async def handle_meeting_trigger(self, envelope: EventEnvelope):
        payload = MeetingTriggerPayload(**envelope.payload)
        meeting_id = await self.create_meeting(payload)
        logger.info(f"✓ Created meeting {meeting_id}")
```

**Why this rocks**:
- No manual queue binding code
- No manual message acknowledgment
- No boilerplate RabbitMQ connection management
- Just declare your routing keys and add handler methods

**Tip**: Name your handler methods descriptively. The decorator does all the routing.

### 2. Direct Database Access Pattern

We bypassed TheBoard's CLI and service layer entirely, writing directly to the database:

```python
class MeetingCreator:
    def __init__(self):
        self.engine = create_engine(str(settings.theboard_database_url))
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_meeting_from_trigger(self, payload):
        with self.SessionLocal() as db:
            meeting = Meeting(
                id=uuid4(),
                topic=payload.topic,
                strategy=StrategyType(payload.strategy),
                max_rounds=payload.max_rounds,
            )
            db.add(meeting)
            db.commit()
            return meeting.id
```

**Why this worked**:
- No HTTP overhead
- No API authentication
- Direct access to internal data models
- Fast and simple

**Trade-off**: Tight coupling to TheBoard's database schema. If the `Meeting` model changes, this service breaks. Document this dependency clearly.

### 3. Topic Exchange Routing Key Design

Bloodbank uses a single topic exchange (`bloodbank.events.v1`) with hierarchical routing keys:

```
theboard.meeting.trigger
theboard.meeting.created
feature.brainstorm.requested
architecture.review.needed
```

**Pattern**: `<domain>.<entity>.<action>`

**Why this works**:
- Easy wildcard subscriptions: `theboard.#` subscribes to all TheBoard events
- Logical grouping by domain
- Supports future expansion without breaking consumers

**Tip**: When adding new events, follow this pattern religiously. Future you (and other devs) will thank you.

### 4. Pydantic Payload Validation

Every event payload is a strongly-typed Pydantic model:

```python
class MeetingTriggerPayload(BaseModel):
    topic: str = Field(min_length=10, max_length=500)
    strategy: Literal["sequential", "greedy"] = "sequential"
    max_rounds: int = Field(default=5, ge=1, le=10)
    context: dict[str, str] | None = None
```

**Benefits**:
- Validation happens automatically
- Clear contract for event producers
- Self-documenting schemas
- Type hints for IDE autocomplete

**Gotcha**: Field types are strict. `dict[str, str]` means **all values must be strings**, not `dict[str, Any]`. See "What Didn't Go Well #4" below.

---

## 😬 What Didn't Go Well (And How to Fix It)

### 1. Bloodbank Settings Attribute Name Mismatch

**Problem**: EventConsumer tried to access `settings.rabbitmq_url` but Bloodbank's config defines `settings.rabbit_url`.

**Error**:
```
AttributeError: 'Settings' object has no attribute 'rabbitmq_url'.
Did you mean: 'rabbit_url'?
```

**Root Cause**: Inconsistency in Bloodbank's own codebase. `config.py` defines `rabbit_url` but `consumer.py` tried to access `rabbitmq_url`.

**Fix**:
```python
# In bloodbank/event_producers/consumer.py line 95
connection = await aio_pika.connect_robust(settings.rabbit_url)  # Not rabbitmq_url
```

**Lesson**: When integrating with a shared library, grep for the actual attribute names. Don't trust variable naming conventions to be consistent.

**Prevention**: Add this to your service README:
```bash
# Bloodbank expects RABBIT_URL (not RABBITMQ_URL)
RABBIT_URL=amqp://user:pass@localhost:5673/
```

### 2. Import Path Hell

**Problem**: Bloodbank's internal imports use `from event_producers.rabbit import Publisher` but examples showed `from rabbit import Publisher`.

**Error**:
```
ImportError: cannot import name 'Publisher' from 'rabbit'
```

**Root Cause**: Bloodbank package is installed as `bloodbank` but internal modules use `event_producers` namespace. You can't import `from rabbit` directly.

**Fix**:
```python
# ✗ Wrong
from rabbit import Publisher

# ✓ Correct
from event_producers.rabbit import Publisher
from event_producers.events.base import EventEnvelope, Source, TriggerType
```

**Lesson**: Check the actual `__init__.py` files to see what's exported. Don't rely on top-level imports existing.

**Prevention**: Add this pattern to your imports:
```python
# Standard Bloodbank imports for consumers
from event_producers import EventConsumer
from event_producers.rabbit import Publisher
from event_producers.events.base import EventEnvelope, Source, TriggerType
```

### 3. TriggerType.SYSTEM Doesn't Exist

**Problem**: Initially used `TriggerType.SYSTEM` for service lifecycle events, assuming system-triggered events would have their own enum value.

**Error**:
```
AttributeError: type object 'TriggerType' has no attribute 'SYSTEM'
```

**Actual TriggerType Enum**:
```python
class TriggerType(str, Enum):
    MANUAL = "manual"        # Human-initiated
    AGENT = "agent"          # AI agent triggered
    SCHEDULED = "scheduled"  # Cron/timer
    FILE_WATCH = "file_watch" # File system event
    HOOK = "hook"            # External webhook
```

**Fix**: Use `TriggerType.MANUAL` for service lifecycle events (startup, health checks).

**Lesson**: When working with enums from a library, print them or read the source. Don't assume values exist.

**Prevention**:
```python
# Add this to your docs
# Valid TriggerType values:
# - MANUAL: Human/service-initiated
# - AGENT: AI agent actions
# - SCHEDULED: Cron jobs
# - FILE_WATCH: File system triggers
# - HOOK: External webhooks
```

### 4. Pydantic Type Strictness - dict[str, str] vs dict[str, Any]

**Problem**: Passed `{"test": True, "source": "integration_test"}` to a field typed as `dict[str, str]`.

**Error**:
```
ValidationError: context.test
  Input should be a valid string [type=string_type, input_value=True, input_type=bool]
```

**Root Cause**: Pydantic enforces `dict[str, str]` literally. All values must be strings, not booleans or integers.

**Fix**:
```python
# ✗ Wrong
context={"test": True, "source": "integration_test"}

# ✓ Correct
context={"test": "true", "source": "integration_test"}
```

**Lesson**: Pydantic is strict. If the type says `str`, you can't pass `bool`. Convert everything to strings when dealing with `dict[str, str]` fields.

**Prevention**: Either:
1. Use `dict[str, Any]` if you need mixed types
2. Document that all context values must be strings
3. Add a validator to auto-convert types

### 5. Python Bytecode Cache Issues

**Problem**: Fixed Bloodbank code but service kept failing with the old error.

**Root Cause**: Python caches bytecode in `__pycache__` directories. Even after editing source, old `.pyc` files can be loaded.

**Fix**:
```bash
# Clear all Python cache in Bloodbank
find /path/to/bloodbank -name __pycache__ -type d -exec rm -rf {} +

# Force reinstall the package
uv pip uninstall bloodbank
uv pip install --force-reinstall -e /path/to/bloodbank
```

**Lesson**: After modifying library code, always clear cache and reinstall. Otherwise you'll debug phantom bugs that don't exist in the source anymore.

**Prevention**: Add this to your debugging checklist:
```bash
# After editing shared libraries
1. Clear __pycache__
2. Reinstall with --force-reinstall
3. Restart service
```

### 6. Meeting Model Doesn't Have an `agents` Relationship

**Problem**: Tried to assign agents at meeting creation time with `Meeting(agents=selected_agents)`.

**Error**:
```
TypeError: 'agents' is an invalid keyword argument for Meeting
```

**Root Cause**: TheBoard's architecture assigns agents during execution, not at creation. Agents participate through `Response` records, not a direct relationship on `Meeting`.

**Fix**:
```python
# ✗ Wrong
meeting = Meeting(
    topic=payload.topic,
    strategy=StrategyType(payload.strategy),
    agents=selected_agents,  # Invalid!
)

# ✓ Correct
meeting = Meeting(
    topic=payload.topic,
    strategy=StrategyType(payload.strategy),
    # Agents assigned during execution via Response records
)
```

**Lesson**: Don't assume database model relationships. Read the actual SQLAlchemy model definition to understand what fields exist.

**Prevention**: Before writing to a database model:
1. Read the model file (`models/meeting.py`)
2. Check what fields are defined
3. Look for relationships and understand their cascade behavior

---

## 🧠 Lessons Learned

### 1. Start with Environment Variables, Not Defaults

**Bad Pattern**:
```python
# Service config with defaults
rabbit_url: str = "amqp://guest:guest@rabbitmq:5672/"
```

**Good Pattern**:
```python
# Force explicit configuration
rabbit_url: str = Field(..., description="RabbitMQ URL - set RABBIT_URL env var")
```

**Why**: Services fail loudly if misconfigured instead of silently connecting to the wrong host. Docker hostname `rabbitmq:5672` won't work from localhost.

**Application**: We spent time debugging connection failures because the service tried to connect to a Docker-internal hostname. Explicit env vars force users to provide correct values.

### 2. EventConsumer Uses `.run()`, Not `.start()`

**Bad Assumption**:
```python
consumer = TheboardMeetingTriggerConsumer()
await consumer.start()  # AttributeError: no attribute 'start'
```

**Correct Usage**:
```python
consumer = TheboardMeetingTriggerConsumer()
await consumer.run()  # Blocks and consumes messages
```

**Why**: The EventConsumer base class implements `.run()` as the main loop. No separate `.start()` method exists.

**Application**: Read the base class to understand the API. Don't assume conventional method names exist.

### 3. No Lifecycle Hooks (on_start, on_stop) in EventConsumer

**Bad Assumption**:
```python
class MyConsumer(EventConsumer):
    async def on_start(self):
        # Initialize resources
        pass
```

**Reality**: No `on_start()` or `on_stop()` hooks exist. Do initialization in `__init__()`.

**Correct Pattern**:
```python
class MyConsumer(EventConsumer):
    def __init__(self):
        super().__init__()
        self.meeting_creator = MeetingCreator()  # Initialize here
```

**Why**: EventConsumer is minimal by design. It doesn't provide framework-style lifecycle hooks.

**Application**: We tried to add lifecycle hooks and got no errors, but they never executed. Always verify base class interface.

### 4. Service-to-Service Database Access is Simple (But Risky)

**Pattern We Used**:
```python
# Consumer service directly writes to TheBoard database
engine = create_engine("postgresql://theboard:pass@localhost:5433/theboard")
```

**Benefits**:
- No API layer needed
- Fast writes
- Simple implementation

**Risks**:
- Schema changes in TheBoard break this service silently
- No validation layer
- No audit trail of who created meetings
- Tight coupling between services

**Mitigation**:
1. Document the dependency explicitly
2. Add database migration coordination between services
3. Consider exposing an internal API if schema volatility is high

**Trade-off**: We accepted tight coupling for Phase 2 speed. Future phases may refactor to API-based communication.

### 5. Bloodbank EventEnvelope Requires `source` Field

**Error We Hit**:
```python
envelope = EventEnvelope(
    event_type="theboard.meeting.trigger",
    payload={...},
    # Missing source field!
)
# ValidationError: source - Field required
```

**Correct Pattern**:
```python
from event_producers.events.base import Source, TriggerType

envelope = EventEnvelope(
    event_type="theboard.meeting.trigger",
    payload={...},
    source=Source(
        host=socket.gethostname(),
        type=TriggerType.MANUAL,
        app="integration-test",
    ),
)
```

**Lesson**: EventEnvelope has required fields beyond just `event_type` and `payload`. Always check the schema definition.

### 6. Test with Real Events, Not Mocks

**What We Did**: Published actual events to RabbitMQ and verified end-to-end flow.

**Result**: Discovered:
- Connection string issues (localhost vs Docker hostname)
- Payload validation errors (boolean vs string)
- Cache invalidation bugs

**Alternative (Mock-Based Testing)**: Would have missed all environment-specific issues.

**Recommendation**: Integration tests with real infrastructure beat unit tests for event-driven services. Spin up RabbitMQ and PostgreSQL in Docker, publish real events, verify database state.

---

## 🛠️ Patterns to Steal

### 1. EventConsumer Boilerplate

```python
"""Consumer service template for 33GOD Bloodbank integration."""

import asyncio
import logging
from event_producers import EventConsumer
from event_producers.events.base import EventEnvelope
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    rabbit_url: str  # Bloodbank expects RABBIT_URL env var
    queue_name: str = "my_service_queue"

class MyConsumer(EventConsumer):
    queue_name = Settings().queue_name
    routing_keys = [
        "my.event.type",
    ]

    def __init__(self):
        super().__init__()
        # Initialize resources here (no on_start hook)

    @EventConsumer.event_handler("my.event.type")
    async def handle_my_event(self, envelope: EventEnvelope):
        logger.info(f"📥 Received event: {envelope.event_id}")
        # Process event

if __name__ == "__main__":
    consumer = MyConsumer()
    asyncio.run(consumer.run())  # Not .start()
```

### 2. Direct Database Access Pattern

```python
"""Direct database access for cross-service writes."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str

class DatabaseWriter:
    def __init__(self):
        self.engine = create_engine(
            str(Settings().database_url),
            pool_pre_ping=True,  # Test connections before use
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def write_record(self, data):
        with self.SessionLocal() as db:
            try:
                record = MyModel(**data)
                db.add(record)
                db.commit()
                db.refresh(record)
                return record.id
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to write: {e}")
                raise
```

### 3. Service Configuration Template

```bash
# .env file for 33GOD services

# Bloodbank RabbitMQ (exact var name matters!)
RABBIT_URL=amqp://user:pass@localhost:5673/

# Service database (if direct access)
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5433/dbname

# Service config
SERVICE_ID=my-service
QUEUE_NAME=my_service_queue
LOG_LEVEL=INFO
```

### 4. Event Publishing Test Script

```python
"""Test script to publish Bloodbank events."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4
from event_producers.rabbit import Publisher
from event_producers.events.base import EventEnvelope, Source, TriggerType

async def publish_test_event():
    publisher = Publisher()
    await publisher.start()

    envelope = EventEnvelope(
        event_id=uuid4(),
        event_type="my.test.event",
        timestamp=datetime.now(timezone.utc),
        source=Source(
            host="localhost",
            type=TriggerType.MANUAL,
            app="test-script",
        ),
        payload={
            "test": "true",  # All strings!
            "message": "hello world",
        },
    )

    await publisher.publish(
        routing_key="my.test.event",
        body=envelope.model_dump(mode="json"),
    )

    print(f"✓ Published event: {envelope.event_id}")

if __name__ == "__main__":
    asyncio.run(publish_test_event())
```

---

## 📋 Pre-Flight Checklist for New 33GOD Services

Before you start building:

- [ ] **Read the service registry** (`/home/delorenj/code/33GOD/services/registry.yaml`) to understand existing events
- [ ] **Clone Bloodbank repository** to `~/code/33GOD/bloodbank/trunk-main` (consumers need it installed)
- [ ] **Check RabbitMQ port** - Default is 5673 for local, not 5672
- [ ] **Use correct env var names** - `RABBIT_URL` not `RABBITMQ_URL`
- [ ] **Install Bloodbank as editable** - `uv pip install -e /path/to/bloodbank` for development
- [ ] **Define Pydantic payload models** before writing handlers
- [ ] **Use `dict[str, str]` carefully** - All values must be strings
- [ ] **Test with real RabbitMQ** - Don't rely on mocks for integration testing

During development:

- [ ] **Clear Python cache** after editing Bloodbank code
- [ ] **Force reinstall** with `--force-reinstall` when debugging library issues
- [ ] **Check EventConsumer base class** for actual method signatures
- [ ] **Use `.run()` not `.start()`** for EventConsumer
- [ ] **Import from `event_producers.*`** not top-level `rabbit` or `consumer`
- [ ] **Validate TriggerType values** - No SYSTEM, use MANUAL for services
- [ ] **Test payload serialization** - Ensure all fields are JSON-serializable
- [ ] **Verify database model fields** before writing to them

Before deployment:

- [ ] **Document database dependencies** if using direct access
- [ ] **Add service to registry.yaml** with correct event routing keys
- [ ] **Test end-to-end** with real events and verify database state
- [ ] **Add healthcheck** if running as long-lived service
- [ ] **Configure logging** with structured output for observability

---

## 🎯 What's Next - Phase 3 Ideas

Now that we have event-driven meeting creation working, potential next steps:

1. **Meeting Result Events** - Publish `theboard.meeting.completed` with extracted insights for downstream consumers
2. **RAG Integration** - Index meeting comments in Qdrant and expose semantic search API
3. **Candybar Visualization** - Display TheBoard node in service graph with live meeting status
4. **Agent Selection Service** - Intelligent agent selection based on topic embeddings instead of random selection
5. **Workflow Orchestration** - Chain meetings together: requirements gathering → brainstorming → technical design
6. **Multi-Tenant Support** - Namespace meetings by project/team and filter events accordingly

---

## 🙏 Acknowledgments

**What Worked**:
- Bloodbank's convention-over-configuration EventConsumer pattern
- Direct database access for simple cross-service writes
- Topic exchange routing keys for flexible event subscriptions
- Pydantic validation catching type errors early

**What Needed Fixing**:
- Bloodbank internal inconsistencies (rabbitmq_url vs rabbit_url)
- Missing lifecycle hooks documentation
- TriggerType enum not matching expectations
- Python bytecode cache invalidation

**Key Insight**: Event-driven architecture is simple once you understand the framework's conventions. Most debugging was learning Bloodbank's API surface, not fixing architectural issues.

---

## 📚 Reference Links

**Bloodbank**:
- Repository: `/home/delorenj/code/33GOD/bloodbank/trunk-main`
- EventConsumer: `event_producers/consumer.py`
- Publisher: `event_producers/rabbit.py`
- Event Schemas: `event_producers/events/base.py`

**TheBoard**:
- API Layer: `src/theboard/api.py`
- Event Emitter: `src/theboard/events/bloodbank_emitter.py`
- Meeting Model: `src/theboard/models/meeting.py`

**theboard-meeting-trigger**:
- Service Root: `/home/delorenj/code/33GOD/services/theboard-meeting-trigger`
- Consumer: `src/theboard_meeting_trigger/consumer_simple.py`
- Meeting Creator: `src/theboard_meeting_trigger/meeting_creator.py`

**33GOD Ecosystem**:
- Service Registry: `/home/delorenj/code/33GOD/services/registry.yaml`
- Integration Spec: `docs/33GOD_INTEGRATION_PHASE1.md`

---

**Written by**: Claude Sonnet 4.5 + Jarad DeLorenzo
**Date**: 2026-01-09
**Phase**: Phase 2 Complete (tag: `phase-2-complete`)
**Status**: Production-ready event consumer deployed and tested
