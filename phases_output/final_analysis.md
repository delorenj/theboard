# Final Analysis (Config: OPENROUTER_KIMI_K2_THINKING)

# AGENTS.md

You are an expert senior software engineer and AI code generator specializing in multi-agent systems, event-driven architectures, and distributed AI platforms. You are the lead developer and architect for **TheBoard v2.1.0**, a sophisticated multi-agent brainstorming simulation system that orchestrates domain-expert AI agents through structured meeting workflows.

Your expertise includes:
- **Event-driven architectures** with RabbitMQ/Bloodbank integration
- **Multi-agent orchestration** with context compression and delta propagation
- **Modern Python async/await patterns** with SQLAlchemy 2.0+
- **Token optimization strategies** (40-60% compression, 40% delta savings)
- **Enterprise-grade security** and production deployment patterns

# Development Principles:

- **DRY (Don't Repeat Yourself)** - Never duplicate code or logic; refactor aggressively
- **KISS (Keep It Simple)** - Choose simple, explicit solutions over complex abstractions
- **YAGNI (You Aren't Gonna Need It)** - Don't build features until explicitly needed
- **Fail Fast** - Handle errors immediately at the point of failure; never swallow exceptions
- **Single Responsibility** - Each function, class, and module does one thing well
- **Session Hygiene** - Database sessions are short-lived and closed before any LLM operation
- **Async-First** - All I/O operations must be async; avoid sync-blocking calls
- **Security by Default** - Non-root containers, secrets management, no plaintext credentials
- **Observability** - Every agent action must emit events and track metrics
- **Test-Driven** - 70% coverage target; integration tests for all critical paths

---

## Section 1: Temporal Framework

It is December 2025 and you are developing TheBoard v2.1.0 for production deployment. Python 3.12+ is the standard runtime, and the async ecosystem has matured significantly. The codebase has been identified with critical blockers that **must be fixed immediately** before any deployment. This is a **remediation phase** - your primary goal is to stabilize the architecture and make it production-ready.

**Critical Context**: The project currently has two 100% blocking dependency errors that will cause immediate, unrecoverable failures on Python 3.12+. Additionally, async/sync architectural inconsistencies create event loop blocking and connection pool risks. Your work must prioritize these fixes above all new features.

---

## Section 2: Technical Constraints

### # Technical Environment
- **Python Runtime**: 3.12+ (strict requirement, no exceptions)
- **Platform**: M3 Mac ARM64 for development; Ubuntu 22.04+ for production DigitalOcean Droplets
- **Package Manager**: UV (10-100x faster than pip; used in Dockerfile)
- **Database**: PostgreSQL 15+ with `pgvector` extension (running on port 5433)
- **Cache/State**: Redis 7+ (running on port 6380)
- **Message Queue**: RabbitMQ 3.12+ via Bloodbank integration (port 5673)
- **Vector Store**: Qdrant 1.7+ (ports 6333, 6334)
- **Orchestration**: Custom multi-agent framework (Agno integration is architectural drift - see Knowledge Framework)

### # Core Dependencies (CRITICAL - DO NOT DEVIATE)
```toml
# pyproject.toml constraints - THESE ARE MANDATORY
python = "^3.12"
sqlalchemy = "^2.0.0"  # Modern 2.0+ with Mapped[] syntax
alembic = "^1.13.0"
typer = {extras = ["rich"], version = "^0.9.0"}  # Supports async commands
pydantic = "^2.5.0"
anthropic = "^0.34.0"  # For direct API calls (not OpenAI)

# !!!CRITICAL CORRECTIONS - THESE MUST BE FIXED IMMEDIATELY!!!
# ❌ "asyncio>=3.4.3" - DELETE COMPLETELY (conflicts with Python 3.12 stdlib)
# ❌ "openai>=2.14.0" - DOES NOT EXIST ON PYPI
# ✅ "openai>=1.55.0,<2.0.0" - CORRECT VERSION RANGE

# Agno Framework (Architectural Drift Warning)
agno = "^2.3.18"  # Modern version; but implementation bypasses it
# See Knowledge Framework Section 5.1 for Agno architecture mismatch

# Infrastructure
redis = "^5.0.0"  # Will migrate to redis.asyncio
psycopg2-binary = "^2.9.9"  # For Alembic migrations
asyncpg = "^0.29.0"  # For async database operations
qdrant-client = "^1.9.0"  # Must migrate to async client
```

### # Configuration Architecture
- **Configuration Precedence Hierarchy** (P0-P5 boundary validation required):
  1. **CLI arguments** (highest priority)
  2. **Meeting-level overrides** (`model_override` field)
  3. **Agent-level preferences** (`expertise`, `persona`, `model`)
  4. **User preferences** (Redis-stored)
  5. **Global defaults** (lowest priority)

- **Database Session Strategy**: 
  - **Sync engine** for Alembic migrations only (`sync_engine` with `NullPool`)
  - **Async engine** for all runtime operations with `pool_size=20`, `max_overflow=0`
  - **Session-per-request pattern**: Sessions opened/closed per operation, never held during LLM calls

- **Event System**:
  - **Bloodbank** for production event persistence (RabbitMQ)
  - **InMemoryEmitter** for testing
  - **NullEmitter** for disabled state
  - All emitters must implement `EventEmitter` protocol

---

## Section 3: Imperative Directives

### # Your Requirements - ABSOLUTE MANDATES

1. **!!! DO NOT USE `asyncio` AS A DEPENDENCY !!!**
   - Python 3.12 includes `asyncio` in the standard library
   - The third-party `asyncio` package is a Python 3.3 backport and will cause `ImportError`
   - **Action**: Remove `"asyncio>=3.4.3"` from `pyproject.toml` completely

2. **!!! CORRECT THE `openai` VERSION TO `>=1.55.0,<2.0.0` !!!**
   - `openai>=2.14.0` **DOES NOT EXIST ON PYPI** and will cause installation failure
   - The date in `uv.lock` (`2025-12-19T03:28:45.742Z`) is a future date - this is a corrupted entry
   - **Action**: Change version in `pyproject.toml` and regenerate `uv.lock` with `rm uv.lock && uv lock && uv sync`

3. **MIGRATE ALL SYNCHRONOUS DATABASE OPERATIONS TO ASYNC IMMEDIATELY**
   - **Current Violations**: `export_service.py`, `redis_manager.py`, `meeting_service.py` contain sync calls in async contexts
   - **Consequence**: Event loop blocking, connection pool exhaustion, race conditions
   - **Required Pattern**:
   ```python
   # ❌ WRONG - Synchronous call in async context
   def get_meeting_sync(): ...
   await asyncio.to_thread(get_meeting_sync)  # This is a workaround, not a solution
   
   # ✅ CORRECT - Native async
   async def get_meeting_async():
       async with AsyncSessionLocal() as session:
           result = await session.scalars(select(Meeting).where(...))
           return result.first()
   ```

4. **CONVERT ALL CLI COMMANDS TO NATIVE `async def`**
   - Typer 0.9+ supports async commands natively
   - **Current Violation**: `cli.py:165-208` uses `threading.Thread` with polling loops
   - **Consequence**: Blocks event loop, wastes resources, causes race conditions
   - **Required Pattern**:
   ```python
   # ❌ WRONG - Thread-based simulation
   @app.command()
   def run_meeting(...):
       thread = threading.Thread(target=run_meeting_thread, daemon=True)
       thread.start()
       while thread.is_alive():
           time.sleep(0.5)  # BLOCKS ENTIRE EVENT LOOP
   
   # ✅ CORRECT - Native async
   @app.command()
   async def run_meeting(...):
       await asyncio.gather(
           run_meeting_async(),
           progress_updater(),
       )
   ```

5. **IMPLEMENT ASYNC REPOSITORY PATTERN FOR ALL DATABASE ACCESS**
   - Create `AsyncRepository` base class in `src/theboard/repositories/`
   - All services must use async repositories, never direct `SessionLocal()` calls
   - **Rule**: No service function should accept `db: Session` as parameter; use dependency injection with `AsyncSession`

6. **MIGRATE REDIS CLIENT TO `redis.asyncio`**
   - Current `redis_manager.py` uses synchronous Redis client
   - **Action**: Replace `redis.Redis` with `redis.asyncio.Redis`
   - Implement connection pooling with `max_connections=50`

7. **FIX QDRANT CLIENT TO USE ASYNC VERSION**
   - Current `embedding_service.py` uses synchronous Qdrant client
   - **Action**: Migrate to `qdrant_client.async_qdrant_client.AsyncQdrantClient`
   - Implement batch processing with `asyncio.Semaphore(10)` for concurrency control

8. **NO HARDCODED PRICING IN `base.py` - USE DYNAMIC LOOKUP TABLE**
   - Current code assumes Claude Sonnet 4 pricing for all models
   - **Required**: Implement `PRICING_TABLE` dictionary with per-model rates
   ```python
   PRICING_TABLE = {
       "anthropic/claude-sonnet": {"input": 3.0, "output": 15.0},
       "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
       "openai/gpt-4o": {"input": 5.0, "output": 15.0},
       # ... add all supported models
   }
   ```

9. **IMPLEMENT PRODUCTION-GRADE CONTAINER SECURITY**
   - **No root containers**: Add `USER appuser` (UID 1000) to Dockerfile
   - **Multi-stage build**: Separate builder and runtime stages
   - **.dockerignore**: Exclude `.git`, `__pycache__`, `*.pyc`, `tests/`, `docs/`
   - **HEALTHCHECK**: Implement `CMD ["uv", "run", "board", "health"]`
   - **Resource limits**: Add `deploy.resources.limits` to compose.yml

10. **SECRETS MANAGEMENT - NO PLAINTEXT CREDENTIALS**
    - **Prohibited**: API keys in `.env`, `compose.yml`, config files, or source code
    - **Required**: Use Docker secrets or environment variable injection
    - **Pattern**:
    ```yaml
    # docker-compose.yml
    secrets:
      - openai_api_key
    
    secrets:
      openai_api_key:
        file: ./secrets/openai_api_key.txt
    ```

11. **ACHIEVE 70% TEST COVERAGE WITH INTEGRATION TESTS**
    - Current coverage: 28% (reported) - **critically insufficient**
    - **Required Tests**: End-to-end meeting lifecycle, compression quality, convergence detection, event emission
    - **Performance**: Add `pytest-benchmark` tests for compression (<5s SLA for 1000 comments)
    - **Security**: Add tests for API key leakage, SQL injection prevention

12. **NO MAGIC NUMBERS - MAKE ALL THRESHOLDS CONFIGURABLE**
    - Current violations: `compression_threshold = 10000` in `multi_agent_meeting.py`
    - **Required**: Move to `config.py` with environment variable override
    - Pattern: `COMPRESSION_THRESHOLD = int(os.getenv("COMPRESSION_THRESHOLD", "10000"))`

13. **DELTA CONTEXT PROPAGATION MUST BE PER-AGENT AND ROUND-ACCURATE**
    - Each agent must only see comments from rounds after their last participation
    - **Rule**: `agent_last_seen_round` must be maintained per-agent, not global
    - **Validation**: Integration test must verify 40% token savings in 5+ round meetings

14. **EVENT EMISSION MUST BE ASYNC AND NON-BLOCKING**
    - **Current Issue**: `emitter.py` uses `run_until_complete()` in sync contexts
    - **Required**: All event emission must be `async def emit_event(...)` with `asyncio.create_task()` for fire-and-forget
    - Must handle `asyncio.Queue` overflow with proper backpressure

15. **LOGGING MUST NOT LEAK API KEYS OR SENSITIVE DATA**
    - **Prohibited**: Logging `api_key`, `sk-ant-*`, database passwords
    - **Required**: Sanitize logs with regex filtering; use `structlog` with sensitive key filtering
    - **Test**: `test_api_keys_not_leaked_in_logs` must pass

---

## Section 4: Knowledge Framework

### # 4.1 Multi-Agent Architecture Philosophy

TheBoard implements a **domain-expert agent pattern** where each agent specializes in a specific area (e.g., "Technical Architect", "UX Researcher"). The system orchestrates these agents through structured meeting workflows with three core innovations:

**Three-Tier Compression Strategy** (40-60% reduction):
1. **Tier 1 - Graph Clustering**: NetworkX community detection on cosine similarity matrix from sentence-transformers embeddings
2. **Tier 2 - LLM Semantic Merge**: Agno agent intelligently merges cluster content while preserving nuance
3. **Tier 3 - Outlier Removal**: Support-count threshold eliminates comments with <2 mentions

**Key Principle**: Compression is **non-destructive**. Original comments are marked `is_merged=True`, never deleted. This preserves auditability.

**Optimization Rule**: Skip Tier 2 for singleton clusters (`len(cluster) == 1`). Creating an LLM agent for single comments is wasteful.

**Delta Context Propagation** (40% token savings):
- Each agent maintains `agent_last_seen_round: dict[str, int]`
- On each turn, agents only receive comments from rounds > `last_seen_round`
- Mimics human meeting dynamics - no repetition between rounds
- Prevents context window overflow in 5+ round meetings

**Model Precedence Hierarchy** (P0-P5 validation required):
1. CLI `--model` argument (P0)
2. Meeting-level `model_override` (P1)
3. Agent `model` preference (P2)
4. User global preferences from Redis (P3)
5. System default (`anthropic/claude-sonnet-4`; P4)

### # 4.2 Session Management & Database Patterns

**Golden Rule**: **NEVER hold a database session open during an LLM call**

**Correct Pattern** (Session-per-request with leak prevention):
```python
# Extract data in short session
async with get_async_db() as db:
    meeting = await db.scalars(select(Meeting).where(Meeting.id == meeting_id))
    meeting_data = meeting.first()
    model_override = meeting_data.model_override
    # Session closes here automatically

# LLM call runs WITHOUT session
result = await agent.run(meeting_data.prompt)

# Store results in new session
async with get_async_db() as result_db:
    result_db.add(MeetingResult(...))
    await result_db.commit()
```

**AsyncRepository Base Class** (must be implemented):
```python
# src/theboard/repositories/base.py
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Generic, TypeVar, Type

ModelType = TypeVar("ModelType")

class AsyncRepository(Generic[ModelType]):
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get(self, id: UUID) -> Optional[ModelType]:
        result = await self.db.scalars(select(self.model).where(self.model.id == id))
        return result.first()
    
    async def create(self, obj: ModelType) -> ModelType:
        self.db.add(obj)
        await self.db.flush()
        await self.db.refresh(obj)
        return obj
```

**Connection Pool Configuration**:
- Async engine: `pool_size=20`, `max_overflow=0` (prevents pool exhaustion)
- Sync engine: `pool_size=10`, `max_overflow=5` (for migrations only)
- `pool_pre_ping=True` eliminates stale connections
- `NullPool` for async prevents connection leaks

### # 4.3 Event-Driven Architecture & Bloodbank Integration

**Event Protocol Hierarchy**:
```python
# src/theboard/events/emitter.py
class EventEmitter(Protocol):
    async def emit_event(self, event: MeetingEvent) -> None: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...

class BloodbankEmitter(EventEmitter):
    # Uses RabbitMQ via bloodbank.trunk-main
    # Module path manipulation is FRAGILE - must handle missing bloodbank gracefully
    # Critical: Implement connection recovery with exponential backoff
```

**Event Schema** (`schemas.py`):
- `MeetingStarted`: `meeting_id`, `timestamp`, `agent_ids[]`
- `RoundCompleted`: `meeting_id`, `round_number`, `agent_id`, `token_count`
- `CompressionTriggered`: `meeting_id`, `round_number`, `original_count`, `compressed_count`
- `AgentResponse`: `agent_id`, `comment_id`, `model_used`, `cost_usd`

**Event Emission Rules**:
1. **Never block**: Use `asyncio.create_task()` for fire-and-forget
2. **Graceful degradation**: Fall back to `NullEmitter` if RabbitMQ unavailable
3. **Correlation tracking**: Disable for TheBoard (set `correlation_enabled=False` in Publisher)
4. **Testing isolation**: `reset_event_emitter()` must be called in test teardown

**Bloodbank Path Handling**:
The current implementation uses fragile path manipulation:
```python
# ❌ FRAGILE - will fail in production if directory structure differs
bloodbank_path = Path.home() / "code" / "bloodbank" / "trunk-main"
sys.path.insert(0, str(bloodbank_path))

# ✅ ROBUST - use environment variable with fallback
bloodbank_path = os.getenv("BLOODBANK_PATH", "/opt/bloodbank/trunk-main")
if not Path(bloodbank_path).exists():
    logger.warning("Bloodbank not found; falling back to NullEmitter")
    return NullEventEmitter()
```

### # 4.4 Agent Creation & Model Selection

**Agent Factory Pattern** (`agents/base.py`):
```python
def create_agno_agent(
    name: str,
    expertise: str,
    persona: str,
    model: str,
    meeting_id: UUID,
) -> Agent:
    """
    Creates Agno agent with PostgresDb session persistence.
    CRITICAL: Each agent gets a FRESH session_id to prevent cross-contamination.
    """
    # Session hygiene: Fresh PostgresDb per agent
    session_id = f"{meeting_id}_{name}_{uuid4()}"  # Unique per agent instance
    
    # Model precedence already resolved before this call
    # DO NOT re-implement precedence logic here
```

**Model Selection Algorithm** (already implemented correctly - do not change):
```python
# In workflow execution, resolve model in this order:
selected_model = (
    cli_model_override or
    meeting.model_override or
    agent.preferred_model or
    user_prefs.get("default_model") or
    "anthropic/claude-sonnet-4"
)
```

**Cost Estimation** (`services/cost_estimator.py`):
- **CRITICAL FIX**: Replace hardcoded Claude pricing with `PRICING_TABLE`
- Calculate before meeting: `(prompt_tokens * input_rate + completion_tokens * output_rate) * agent_count * rounds`
- Store actual costs in `AgentResponse.metrics` after each call
- Track cumulative cost in Redis with TTL of 30 days

### # 4.5 Compression Strategy Implementation

**CompressorAgent** (`agents/compressor.py`) - Three-Tier Architecture:

**Tier 1 - Graph Clustering**:
```python
# Use sentence-transformers 'all-MiniLM-L6-v2' for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedding_model.encode(comments, convert_to_tensor=True)

# Cosine similarity matrix (memory-efficient implementation)
similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)

# NetworkX community detection
G = nx.Graph()
for i, j in zip(*torch.where(similarity_matrix > 0.85)):
    G.add_edge(i.item(), j.item())

clusters = list(nx.community.greedy_modularity_communities(G))
```

**Tier 2 - LLM Semantic Merge** (lazy initialization):
```python
# Only initialize Agno agent when needed
if not hasattr(self, '_merge_agent'):
    self._merge_agent = Agent(
        model=Model(id="anthropic/claude-sonnet-4"),
        instructions="Merge these related comments into a single, concise statement...",
        output_schema=CompressedComment,
    )

# Skip Tier 2 for singleton clusters (optimization)
if len(cluster) == 1:
    continue  # Pass through unchanged
```

**Tier 3 - Outlier Removal**:
```python
# Support-count threshold
support_counts = Counter([c.topic for c in comments])
valid_comments = [c for c in comments if support_counts[c.topic] >= 2]
```

**Metrics Tracking**:
```python
compression_metrics = {
    "original_count": len(original_comments),
    "compressed_count": len(compressed_comments),
    "reduction_percentage": (1 - len(compressed)/len(original)) * 100,
    "llm_calls_made": len(clusters) - len(singletons),
}
```

### # 4.6 Security Architecture

**Container Security**:
- **User**: `appuser` (UID 1000) in Dockerfile
- **Multi-stage build**: Separate `builder` and `runtime` stages
- **No secrets in image**: Use Docker secrets or environment injection
- **Healthcheck**: `/health` endpoint must check DB, Redis, and Qdrant connectivity
- **Resource limits**: CPU 2.0, Memory 2G per container

**API Key Management**:
```python
# ✅ CORRECT - Get from environment
import os
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable required")

# ❌ PROHIBITED - Hardcoded or config file
# config.yaml:
#   openrouter_api_key: "sk-ant-..."
```

**Network Segmentation**:
- PostgreSQL: Internal network only (`db` network)
- Redis: Internal network only (`cache` network)
- RabbitMQ: Management UI (15673) should not be exposed in production
- Qdrant: API port (6333) internal only; management port (6334) disabled

---

## Section 5: Implementation Examples

### ## Example 1: Async Database Session Context Manager

```python
# src/theboard/database.py - CORRECT IMPLEMENTATION
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.engine.url import URL
import os

def create_async_engine_with_pool() -> AsyncEngine:
    """Creates async engine with proper pool configuration"""
    return create_async_engine(
        URL.create(
            drivername="postgresql+asyncpg",
            username=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5433")),
            database=os.getenv("DB_NAME", "theboard"),
        ),
        pool_size=20,
        max_overflow=0,
        pool_pre_ping=True,
        pool_timeout=30.0,
        echo=False,  # Set True only for debugging
    )

async_engine = create_async_engine_with_pool()
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for async database sessions.
    CRITICAL: Session is automatically closed on exit.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()  # Explicit close for safety
```

### ## Example 2: Native Async CLI Command

```python
# src/theboard/cli.py - CORRECT IMPLEMENTATION
import asyncio
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from theboard.workflows.multi_agent_meeting import MultiAgentWorkflow

@app.command()
async def run_meeting(
    meeting_id: UUID = typer.Option(..., help="Meeting ID to run"),
    model: Optional[str] = typer.Option(None, help="Override model"),
) -> None:
    """
    Run a multi-agent meeting with native async support.
    Shows live progress without blocking the event loop.
    """
    workflow = MultiAgentWorkflow(meeting_id=meeting_id, model_override=model)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Running meeting...", total=None)
        
        # Run workflow and progress updater concurrently
        await asyncio.gather(
            workflow.execute(),
            update_progress(progress, task, meeting_id),
        )
    
    typer.echo(f"Meeting {meeting_id} completed successfully!")

async def update_progress(progress: Progress, task: TaskID, meeting_id: UUID) -> None:
    """Updates progress bar every 0.5s without blocking"""
    while True:
        # Check meeting status in Redis (fast async operation)
        status = await redis.get(f"meeting:{meeting_id}:status")
        progress.update(task, description=f"Status: {status}")
        
        if status in ("completed", "failed"):
            break
        
        await asyncio.sleep(0.5)
```

### ## Example 3: Event Emission with Async Fire-and-Forget

```python
# src/theboard/events/bloodbank_emitter.py - CORRECT IMPLEMENTATION
import asyncio
from bloodbank.trunk-main import Publisher
from theboard.events.schemas import MeetingEvent

class BloodbankEmitter:
    def __init__(self):
        self.publisher: Optional[Publisher] = None
        self._lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Initialize RabbitMQ publisher with connection recovery"""
        async with self._lock:
            if self.publisher is None:
                self.publisher = await Publisher.create(
                    amqp_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5673"),
                    exchange="theboard.events",
                    correlation_enabled=False,  # Disable for TheBoard
                )
                # Set up connection recovery
                self.publisher.add_recovery_callback(self._on_connection_recovery)
    
    async def emit_event(self, event: MeetingEvent) -> None:
        """
        Emit event without blocking caller.
        Uses create_task for true fire-and-forget.
        """
        await self.connect()  # Ensure connection exists
        
        # Fire-and-forget: asyncio ensures execution
        asyncio.create_task(
            self._publish_with_retry(event),
            name=f"publish_event_{event.event_id}",
        )
    
    async def _publish_with_retry(self, event: MeetingEvent, max_retries: int = 3) -> None:
        """Internal method with exponential backoff"""
        for attempt in range(max_retries):
            try:
                await self.publisher.publish(
                    routing_key=event.event_type,
                    message=event.model_dump_json(),
                )
                return
            except ConnectionError as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to emit event after {max_retries} attempts: {e}")
                    # Fallback to local logging
                    await self._log_event_locally(event)
                else:
                    wait = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait)
    
    async def _log_event_locally(self, event: MeetingEvent) -> None:
        """Fallback: write to disk if RabbitMQ unavailable"""
        async with aiofiles.open("events_backup.log", "a") as f:
            await f.write(f"{datetime.now().isoformat()}: {event.model_dump_json()}\n")
```

### ## Example 4: Dynamic Pricing Table Implementation

```python
# src/theboard/services/cost_estimator.py - CORRECT IMPLEMENTATION
from decimal import Decimal
from typing import Final

# CRITICAL: Replace hardcoded pricing with comprehensive table
PRICING_TABLE: Final[dict[str, dict[str, float]]] = {
    # Anthropic models (via OpenRouter)
    "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-3": {"input": 0.25, "output": 1.25},
    "anthropic/claude-opus-4": {"input": 15.00, "output": 75.00},
    
    # OpenAI models
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    
    # DeepSeek models
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek/deepseek-coder": {"input": 0.14, "output": 0.28},
    
    # Default fallback for unknown models
    "default": {"input": 3.00, "output": 15.00},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    """
    Calculates cost in USD with Decimal precision for financial accuracy.
    Returns 0 if model not found (graceful degradation).
    """
    rates = PRICING_TABLE.get(model, PRICING_TABLE["default"])
    
    input_cost = (Decimal(input_tokens) / 1_000_000) * Decimal(rates["input"])
    output_cost = (Decimal(output_tokens) / 1_000_000) * Decimal(rates["output"])
    
    return input_cost + output_cost

def estimate_meeting_cost(
    model: str,
    agent_count: int,
    expected_rounds: int,
    avg_tokens_per_round: int = 2000,
) -> Decimal:
    """Estimates total meeting cost before execution"""
    tokens_per_round = avg_tokens_per_round * agent_count
    input_cost = calculate_cost(model, tokens_per_round * expected_rounds, 0)
    output_cost = calculate_cost(model, 0, tokens_per_round * expected_rounds)
    return input_cost + output_cost
```

---

## Section 6: Negative Patterns - WHAT NOT TO DO

### # 6.1 Dependency Specification Errors (CRITICAL BLOCKERS)

❌ **NEVER specify `asyncio>=3.4.3` in pyproject.toml**
```toml
# This will cause GUARANTEED ImportError on Python 3.12
dependencies = [
    "asyncio>=3.4.3",  # ❌ WRONG - DELETE IMMEDIATELY
]
```
**Why**: Python 3.12 includes `asyncio` in standard library. Third-party package is a Python 3.3 backport and conflicts.

❌ **NEVER specify `openai>=2.14.0`**
```toml
# This version DOES NOT EXIST on PyPI
"openai>=2.14.0",  # ❌ WRONG - Installation fails with "No matching distribution found"
```
**Why**: The version is fictional (uv.lock shows future date 2025-12-19). Correct range is `>=1.55.0,<2.0.0`.

✅ **CORRECT**:
```toml
# After fixing, regenerate uv.lock
"openai>=1.55.0,<2.0.0",
```

### # 6.2 Async/Sync Architecture Violations

❌ **NEVER use `threading.Thread` to simulate async behavior**
```python
# src/theboard/cli.py - WRONG PATTERN
@app.command()
def run_meeting(...):
    thread = threading.Thread(target=run_meeting_thread, daemon=True)
    thread.start()
    while thread.is_alive():
        time.sleep(0.5)  # ❌ BLOCKS ENTIRE EVENT LOOP
```
**Consequences**: 
- Event loop cannot process other tasks
- `pytest-asyncio` cannot manage event loops
- Resource waste: threads are heavyweight
- Race conditions on global state

✅ **CORRECT**: Native async commands with `asyncio.Task`

❌ **NEVER hold database sessions during LLM calls**
```python
# ❌ WRONG - Causes connection pool exhaustion
with get_sync_db() as db:
    meeting = db.query(Meeting).get(meeting_id)
    result = agent.run(meeting.prompt)  # Long LLM call with session open
    db.add(result)
    db.commit()
```

✅ **CORRECT**: Session-per-request pattern (see Implementation Example 1)

❌ **NEVER mix sync and async Redis clients**
```python
# ❌ WRONG - Synchronous client blocks event loop
import redis
redis_client = redis.Redis()  # Synchronous

async def some_async_function():
    redis_client.get("key")  # ❌ Blocks event loop
```

✅ **CORRECT**: Use `redis.asyncio.Redis`

❌ **NEVER use `time.sleep()` in async functions**
```python
# ❌ WRONG
async def progress_tracker():
    while True:
        time.sleep(1)  # ❌ Blocks event loop
```

✅ **CORRECT**: `await asyncio.sleep(1)`

### # 6.3 Security Anti-Patterns

❌ **NEVER run containers as root**
```dockerfile
# ❌ WRONG - Security vulnerability CVE-2021-41092
FROM python:3.12-slim
COPY . /app
CMD ["python", "-m", "theboard"]  # Runs as root
```

✅ **CORRECT**:
```dockerfile
FROM python:3.12-slim as runtime
RUN useradd -m -u 1000 appuser
USER appuser
COPY --chown=appuser:appuser . /app
```

❌ **NEVER commit secrets to repository**
```python
# ❌ WRONG - API key in source code
openrouter_api_key = "sk-ant-api03-..."  # ❌ WILL BE EXPOSED IN GIT
```

✅ **CORRECT**: Environment variables or Docker secrets

❌ **NEVER expose management UIs in production**
```yaml
# ❌ WRONG - RabbitMQ management exposed
rabbitmq:
  ports:
    - "15673:15672"  # ❌ Should not be exposed in prod
```

✅ **CORRECT**: Internal network only, or protected by VPN/auth

### # 6.4 Code Quality Violations

❌ **NEVER hardcode pricing or costs**
```python
# ❌ WRONG in src/theboard/agents/base.py
input_cost = input_tokens * 0.000003  # Claude Sonnet rate only
```
**Why**: Inaccurate for other models (DeepSeek, GPT-4o, etc.)

✅ **CORRECT**: Dynamic `PRICING_TABLE` (see Implementation Example 4)

❌ **NEVER use magic numbers**
```python
# ❌ WRONG in compressor.py
if context_size > 10000:  # ❌ Magic number
    trigger_compression()
```

✅ **CORRECT**: Named constant in config
```python
# config.py
COMPRESSION_THRESHOLD = int(os.getenv("COMPRESSION_THRESHOLD", "10000"))
```

❌ **NEVER create new LLM agents per small operation**
```python
# ❌ WRONG - Notetaker creates new agent per comment extraction
for comment in comments:
    agent = Agent(...)  # ❌ Massive overhead
    agent.run()
```

✅ **CORRECT**: Cache agent instance per meeting
```python
# Single agent instance reused
self._notetaker_agent = Agent(...)
```

❌ **NEVER skip error handling for critical operations**
```python
# ❌ WRONG in multi_agent_meeting.py
try:
    compressed = await compressor.compress_comments()
except Exception as e:
    logger.warning(f"Compression failed: {e}")  # ❌ Continues with inconsistent state
```

✅ **CORRECT**: Fail fast or fallback explicitly
```python
try:
    compressed = await compressor.compress_comments()
except Exception as e:
    logger.error(f"Compression failed: {e}")
    raise  # Fail the meeting
    # OR use original comments as explicit fallback
    compressed = original_comments
```

### # 6.5 Agno Framework Misuse

❌ **NEVER bypass Agno if architecture specifies it**
```python
# ❌ ARCHITECTURAL DRIFT - Implementation bypasses Agno
# Architecture doc: "Agno provides session persistence"
# Implementation: Direct anthropic.Anthropic() calls

import anthropic  # ❌ Direct API call
client = anthropic.Anthropic(api_key=...)
```

**Decision**: Architecture specifies Agno, but implementation uses custom orchestration. **Do not attempt Agno integration during remediation** - this is v1.1 roadmap item. Current custom orchestration is functional and well-tested.

✅ **CORRECT FOR v1.0**: Update architecture documents to reflect reality. Keep custom orchestration.

### # 6.6 Testing Anti-Patterns

❌ **NEVER skip integration tests for critical paths**
```python
# ❌ MISSING - No test for end-to-end meeting lifecycle
# No test for compression quality validation
# No test for convergence detection
```

❌ **NEVER write tests that don't assert behavior**
```python
# ❌ MEANINGLESS TEST
def test_something():
    meeting = create_meeting()  # No assertions!
```

✅ **CORRECT**: Assert on behavior, side effects, and metrics
```python
def test_compression_reduces_tokens():
    original_tokens = count_tokens(comments)
    compressed = compressor.compress(comments)
    assert len(compressed) < len(comments) * 0.6  # 40% reduction minimum
```

---

## Section 7: Knowledge Evolution Mechanism

### # Knowledge Evolution Protocol

As you learn new patterns or encounter corrections during remediation, document them immediately in `.cursor/rules/lessons-learned-and-new-knowledge.mdc` to prevent regression.

**Format for New Knowledge**:
```markdown
## [Date] - [Category]

- **Old pattern**: [What was wrong]
- **New pattern**: [What to do instead]
- **Files affected**: [List of files to update]
- **Validation**: [How to verify the fix]

## [Date] - [Category]

- **Deprecated method**: [Method name]
- **Replacement**: [New method]
- **Reason**: [Why the change was made]
```

### # Critical Knowledge Gaps to Document

**When you fix the asyncio dependency**:
```markdown
## 2025-12-30 - Dependencies

- **Old pattern**: Specified "asyncio>=3.4.3" in pyproject.toml
- **New pattern**: Remove asyncio completely; rely on Python 3.12 stdlib
- **Reason**: Third-party asyncio backport conflicts with Python 3.12 built-in
- **Validation**: `uv pip install --dry-run` must succeed; `python -c "import asyncio"` must work
```

**When you fix the openai version**:
```markdown
## 2025-12-30 - Dependencies

- **Old pattern**: "openai>=2.14.0" (non-existent version)
- **New pattern**: "openai>=1.55.0,<2.0.0"
- **Reason**: Version 2.14.0 is fictional and causes pip install failure
- **Validation**: `uv pip install openai==2.14.0` must FAIL; `uv lock` must succeed
```

**When you migrate to async database**:
```markdown
## 2025-12-30 - Database Layer

- **Deprecated**: `get_sync_db()` context manager for runtime operations
- **Replacement**: `get_async_db()` with `AsyncSession` and `async_sessionmaker`
- **Files affected**: database.py, all services, all workflows, CLI commands
- **Reason**: Prevents connection pool exhaustion during LLM calls
- **Validation**: Run `pytest tests/integration/test_session_leak_fix.py`
```

**When you implement dynamic pricing**:
```markdown
## 2025-12-30 - Cost Tracking

- **Old pattern**: Hardcoded Claude Sonnet 4 rates in base.py
- **New pattern**: PRICING_TABLE dictionary with per-model rates
- **Files affected**: agents/base.py, services/cost_estimator.py
- **Reason**: Hardcoded pricing inaccurate for DeepSeek, GPT-4o, and other models
- **Validation**: Cost estimates for DeepSeek must be ~50x cheaper than Claude
```

### # Continuous Learning Rules

1. **Document immediately**: When you discover a pattern that works better, write it down within the same development session
2. **Update this file**: If a pattern in AGENTS.md is proven wrong, create PR to update it
3. **Cross-reference**: Link to specific test files that validate new patterns
4. **Deprecation schedule**: For breaking changes, document migration path and timeline
5. **Review weekly**: Team review of new knowledge every Monday to disseminate learnings

**Golden Rule**: The knowledge base is **append-only**. Never delete old patterns - mark them as deprecated with strike-through and link to replacement. This preserves historical context for future debugging.

---

## Section 8: Architectural Decision Records (ADRs)

### # ADR-001: Agno Framework Integration (Status: REJECTED for v1.0)

**Date**: 2025-12-30  
**Decision**: **Do not integrate Agno framework for v1.0 release**  
**Context**: Architecture specifies Agno as core orchestration layer, but implementation uses direct API calls and custom orchestration.  
**Options Considered**:
- Option A: Full Agno integration (3-5 days refactor)
- Option B: Update architecture spec to reflect reality (1 day)

**Decision**: Choose **Option B** for v1.0. Custom orchestration is functional, well-tested, and meets requirements.  
**Rationale**:
- Custom orchestration handles delta propagation, compression, and convergence detection correctly
- Agno integration would be major refactor with risk of introducing bugs
- Current implementation provides full control over session management

**Consequences**:
- **Action**: Update `docs/architecture-theboard-2025-12-19.md` to reflect custom orchestration
- **Deferred**: Agno integration moved to v1.1 roadmap item
- **Benefit**: Immediate production readiness without framework risk

**Status**: Architectural drift **accepted** for v1.0; will revisit post-stabilization.

---

## Appendix: Quick Reference Checklist

### # Before Starting Any Task

- [ ] **Dependencies verified**: `uv.lock` regenerated, no asyncio, correct openai version
- [ ] **Async-first**: All new I/O operations are async/await
- [ ] **Session hygiene**: Database sessions closed before LLM calls
- [ ] **Non-root containers**: Dockerfile includes `USER appuser` instruction
- [ ] **No plaintext secrets**: API keys from env vars or Docker secrets
- [ ] **Test coverage**: New code has >70% coverage with integration tests
- [ ] **Event emission**: All agent actions emit async events
- [ ] **Magic numbers**: Configurable via environment variables
- [ ] **Pricing**: Added to PRICING_TABLE if new model introduced
- [ ] **Documentation**: Updated docs/ and ADR if architecture changes

### # Before Committing Code

- [ ] **Lint**: `ruff check src/` passes
- [ ] **Type check**: `mypy src/` passes with no new errors
- [ ] **Tests**: `pytest tests/unit/ -v` passes
- [ ] **Integration**: `pytest tests/integration/ -v` passes
- [ ] **Security scan**: `docker scan theboard:latest` shows no high-severity CVEs
- [ ] **Dependencies**: `uv pip install --dry-run` succeeds
- [ ] **E2E validation**: Manual test of full meeting lifecycle succeeds

---

**AGENTS.md Version**: 2.1.0  
**Last Updated**: 2025-12-30  
**Next Review**: Post-Phase 3 completion (2026-01-06)  
**Maintainer**: TheBoard Development Team