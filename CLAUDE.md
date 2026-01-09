# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TheBoard is a multi-agent brainstorming simulation system that orchestrates specialized AI agents through structured meeting workflows. Version 2.1.0 is production-ready with event-driven architecture, multi-agent orchestration, context compression, and convergence detection.

## Development Commands

### Package Management & Environment
```bash
# Install/sync dependencies (uses uv for speed)
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run any command through uv without activating
uv run <command>
```

### Database Management
```bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Rollback last migration
alembic downgrade -1
```

### Testing
```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with verbose output and coverage report
pytest -v --cov=theboard --cov-report=html

# Run single test file
pytest tests/unit/test_agents.py

# Run specific test
pytest tests/unit/test_agents.py::test_domain_expert_agent
```

### Code Quality
```bash
# Lint check (ruff is configured in pyproject.toml)
ruff check src/

# Auto-fix linting issues
ruff check --fix src/

# Type checking
mypy src/
```

### CLI Usage
```bash
# Create meeting (wizard mode for guided setup)
uv run board wizard create

# Create meeting (direct command)
uv run board create --topic "Your topic" --max-rounds 5

# Run meeting
uv run board run <meeting-id>

# Run with model override
uv run board run <meeting-id> --model anthropic/claude-sonnet-4

# Check meeting status
uv run board status <meeting-id>

# Export meeting results
uv run board export <meeting-id> --format markdown
```

### Docker Development
```bash
# Start infrastructure only (for local development)
docker compose up -d postgres redis rabbitmq qdrant

# Start all services including app
docker compose up -d

# View logs
docker compose logs -f theboard

# Execute command in container
docker compose exec theboard uv run board create --topic "test"

# Rebuild after code changes
docker compose build theboard
```

## Architecture & Key Patterns

### Multi-Layer Architecture
TheBoard follows a layered service architecture:
- **CLI Layer** (`cli.py`, `cli_commands/`): Typer-based commands with Rich formatting
- **Service Layer** (`services/`): Business logic (meeting, agent, embedding, export, cost estimation)
- **Workflow Layer** (`workflows/`): Multi-agent orchestration and execution strategies
- **Agent Layer** (`agents/`): Domain experts, notetaker, compressor agents
- **Data Layer** (`models/`): SQLAlchemy 2.0+ models with async support
- **Infrastructure** (`events/`, `utils/`): Event emission, Redis cache, database connections

### Session Management Rules
**CRITICAL**: Never hold database sessions open during LLM calls to prevent connection pool exhaustion.

**Pattern**:
```python
# ✓ CORRECT: Short-lived sessions
from theboard.database import get_sync_db

with get_sync_db() as db:
    meeting = db.query(Meeting).filter_by(id=meeting_id).first()
    topic = meeting.topic
    # Session closes here

# LLM call happens WITHOUT database session
result = agent.run(topic)

# New session for storing results
with get_sync_db() as db:
    db.add(Response(...))
    db.commit()
```

**Anti-pattern**:
```python
# ✗ WRONG: Session held during LLM call
with get_sync_db() as db:
    meeting = db.query(Meeting).first()
    result = agent.run(meeting.topic)  # BLOCKS connection pool
    db.add(result)
    db.commit()
```

### Event-Driven Architecture
TheBoard integrates with the 33GOD Bloodbank event bus via RabbitMQ. All critical workflow events are emitted for external consumers.

**Event Types**:
- `theboard.meeting.created` - New meeting initialized
- `theboard.meeting.started` - Execution began
- `theboard.meeting.round_completed` - Round finished
- `theboard.meeting.comment_extracted` - Comments extracted by notetaker
- `theboard.meeting.converged` - Convergence detected
- `theboard.meeting.completed` - Meeting finished successfully
- `theboard.meeting.failed` - Execution failed

**Configuration**: Set `event_emitter: rabbitmq` in `~/.config/theboard/config.yml` or via `THEBOARD_EVENT_EMITTER` env var. Falls back to `null` emitter if RabbitMQ unavailable.

### Model Selection Hierarchy
Models are resolved in this precedence order:
1. CLI `--model` argument (highest priority)
2. Meeting-level `model_override` field
3. Agent-level `preferred_model`
4. User preferences from Redis (`~/.config/theboard/config.yml`)
5. System default (`anthropic/claude-sonnet-4`)

This hierarchy is implemented in `preferences.py:PreferencesManager.get_model_for_agent()`.

### Three-Tier Compression Strategy
When context exceeds threshold (~10K chars), compression is triggered:

**Tier 1 - Graph Clustering**:
- Embeddings via sentence-transformers `all-MiniLM-L6-v2`
- Cosine similarity matrix with NetworkX community detection
- Groups semantically related comments into clusters

**Tier 2 - LLM Semantic Merge**:
- CompressorAgent merges clusters using LLM
- Skips singleton clusters (optimization)
- Preserves meaning while reducing token count

**Tier 3 - Outlier Removal**:
- Support-count threshold filters low-signal comments
- Removes comments mentioned <2 times

**Result**: 40-60% token reduction with quality preservation. Original comments marked `is_merged=True` but never deleted (audit trail).

### Delta Context Propagation
Each agent tracks `agent_last_seen_round` to avoid repetition. On subsequent turns, agents only receive comments from rounds they haven't seen, saving 40% tokens in multi-round meetings.

## Configuration Files

### Environment Configuration
Settings are loaded in this priority order:
1. Environment variables
2. `~/.config/theboard/config.yml` (user config, auto-generated)
3. `.env` file in project root or CWD
4. `.env.example` (fallback defaults)

**Key Settings**:
- `event_emitter`: `null`, `rabbitmq`, or `inmemory`
- `database_url`: PostgreSQL connection string (port 5433 for local)
- `redis_url`: Redis connection string (port 6380 for local)
- `rabbitmq_url`: RabbitMQ AMQP URL (port 5673 for local)
- `openrouter_api_key`: API key for LLM access

### Database Ports
Local development uses custom ports to avoid conflicts:
- **PostgreSQL**: 5433 (not 5432)
- **Redis**: 6380 (not 6379)
- **RabbitMQ**: 5673/15673 (not 5672/15672)
- **Qdrant**: 6333/6334 (standard ports)

### Agent Pool Configuration
Agent definitions are stored in `data/agents/initial_pool.yaml` and seeded into PostgreSQL via `scripts/seed_agents.py`. Each agent has:
- `name`: Display name
- `expertise`: Domain area
- `persona`: Behavioral characteristics
- `model`: Preferred LLM model (optional)

## Important Implementation Details

### Async/Sync Boundary
The codebase is in transition from sync to async. Current state:
- **Sync**: Database operations (`get_sync_db()`), CLI commands (mostly sync with threading)
- **Async**: Event emission (fire-and-forget), some service methods
- **Target**: Full async conversion in v2.2+ (see AGENTS.md for migration plan)

### Cost Tracking
Cost estimation uses hardcoded Claude Sonnet 4 pricing in `agents/base.py`. For accurate multi-model tracking, implement dynamic `PRICING_TABLE` with per-model rates (documented in AGENTS.md Section 4.4).

### Bloodbank Integration
Event emission to Bloodbank requires the `bloodbank` repository at `~/code/bloodbank/trunk-main`. The emitter uses fragile path manipulation (`sys.path.insert`) that should be replaced with environment variable configuration.

**Workaround**: Set `THEBOARD_EVENT_EMITTER=null` to disable if Bloodbank unavailable.

### Testing Strategy
- **Unit tests** (`tests/unit/`): Mock all external dependencies (DB, Redis, LLM APIs)
- **Integration tests** (`tests/integration/`): Real database + mocked LLM calls
- **E2E validation** (`tests/e2e_validation.py`): Full workflow with real infrastructure
- **Coverage target**: 70% (currently ~28%, needs improvement)

### BMAD Workflow Integration
The project uses BMAD (Business, Management, Architecture, Development) methodology. Workflow status tracked in `docs/bmm-workflow-status.yaml`. Agent rules defined in `AGENTS.md` (comprehensive 1000+ line guide for AI developers).

## Common Workflows

### Adding a New Agent Type
1. Define agent class in `src/theboard/agents/` inheriting from `BaseAgent`
2. Add agent to pool in `data/agents/initial_pool.yaml`
3. Run `uv run python scripts/seed_agents.py` to populate database
4. Update `NotetakerAgent` prompts if new comment categories needed
5. Add tests in `tests/unit/test_agents.py`

### Adding a New CLI Command
1. Create command function in `src/theboard/cli.py` or new module in `cli_commands/`
2. Use Typer decorators: `@app.command()` for main commands, `@subapp.command()` for grouped commands
3. Import Rich for formatting: `Console`, `Table`, `Panel`, `Progress`
4. Follow existing pattern: validate inputs, call service layer, emit events, display results
5. Add integration test

### Debugging LLM Calls
Enable debug logging and check `debug.log`:
```python
# config.yml or env var
log_level: DEBUG
debug: true
```

Log file captures:
- Request/response payloads
- Token counts and costs
- Embedding generation
- Compression metrics

### Creating Database Migrations
```bash
# After modifying models in src/theboard/models/
alembic revision --autogenerate -m "Add new_field to Meeting"

# Review generated migration in alembic/versions/
# Edit if needed (especially for complex schema changes)

# Apply migration
alembic upgrade head
```

## Security & Production Notes

### Secrets Management
- **Never commit** `.env` files or API keys to repository
- Use `.env.example` as template with placeholder values
- In production, inject secrets via Docker secrets or environment variables
- API keys should be accessed only via `os.getenv()` or `settings` object

### Container Security
Dockerfile uses multi-stage build and runs as non-root user (`appuser`). Resource limits defined in `compose.yml` prevent resource exhaustion.

### Network Segmentation
All infrastructure services (Postgres, Redis, RabbitMQ, Qdrant) run on internal Docker network. Only application container exposes functionality via CLI.

## Troubleshooting

### Common Issues

**"No module named 'theboard'"**: Ensure virtual environment is activated or use `uv run`

**"Connection to database failed"**: Check PostgreSQL is running on port 5433 (not 5432)

**"Redis connection refused"**: Check Redis is running on port 6380 (not 6379)

**"Event emission failed"**: Bloodbank not found, set `THEBOARD_EVENT_EMITTER=null` to disable

**"Test suite hangs"**: Likely async/sync event loop conflict, run with `pytest -v` for details

**Migration conflicts**: Multiple heads in Alembic, run `alembic heads` and merge if needed

### Debug Checklist
1. Check `debug.log` for detailed error traces
2. Verify all services running: `docker compose ps`
3. Test database connection: `uv run python -c "from theboard.database import get_sync_db; print('OK')"`
4. Test Redis connection: `redis-cli -p 6380 ping`
5. Check environment config: `uv run python -c "from theboard.config import settings; print(settings.model_dump())"`

## Related Documentation

- **README.md**: Installation, quick start, project status
- **AGENTS.md**: Comprehensive AI developer guide (1000+ lines, architectural patterns)
- **BLOODBANK_INTEGRATION.md**: Event bus integration details
- **docs/USER_GUIDE.md**: CLI usage examples and best practices
- **docs/DEVELOPER.md**: Architecture deep-dive and API reference
- **docs/TROUBLESHOOTING.md**: Extended troubleshooting guide
- **docs/WIZARD_GUIDE.md**: Interactive setup wizard documentation
