# TheBoard Developer Documentation

**Version:** 1.0
**Last Updated:** 2025-12-30

Technical documentation for developers working on TheBoard.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [API Reference](#api-reference)
4. [Database Schema](#database-schema)
5. [Adding New Features](#adding-new-features)
6. [Testing](#testing)
7. [Deployment](#deployment)

---

## Architecture Overview

TheBoard follows a layered architecture pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                              │
│                     (cli.py, commands/)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                     Service Layer                             │
│   (services/meeting_service.py, export_service.py, etc.)      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Workflow Layer                              │
│     (workflows/multi_agent_meeting.py, simple_meeting.py)     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                      Data Layer                               │
│           (models/, database.py, redis_manager.py)            │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- Python 3.11+
- SQLAlchemy 2.0 (ORM)
- PostgreSQL 16 (primary database)
- Redis 7 (caching, session management)
- RabbitMQ 3.12 (event bus for human-in-loop)

**LLM Integration:**
- LiteLLM (unified API for multiple providers)
- OpenAI, Anthropic, or compatible models
- Hybrid model strategy (cost optimization)

**Vector Database:**
- Qdrant (embeddings for novelty detection)

**CLI:**
- Typer (command-line framework)
- Rich (terminal UI, progress, tables)

---

## Core Components

### 1. Meeting Workflow (`workflows/multi_agent_meeting.py`)

**Purpose**: Orchestrates multi-round agent brainstorming sessions.

**Key Features**:
- Sequential and greedy execution strategies
- Convergence detection
- Context compression (lazy, graph-based)
- Delta propagation (Sprint 5 optimization)
- Human-in-the-loop support (via RabbitMQ events)

**Main Methods**:
```python
class MultiAgentMeetingWorkflow:
    async def execute(self) -> MeetingResponse:
        """Execute the full meeting workflow."""

    async def _execute_round(self, round_num: int, agents: list[Agent]) -> float:
        """Execute a single round with given agents."""

    async def _execute_agent_turn(self, agent: Agent, context: str, round_num: int) -> float:
        """Execute a single agent's turn in a round."""

    async def _build_context(self, current_round: int, agent_id: str | None = None) -> str:
        """Build context for an agent (with optional delta propagation)."""
```

**Delta Propagation** (Sprint 5):
- Tracks `agent_last_seen_round: dict[str, int]`
- Only sends comments since agent's last participation
- Reduces token usage by ~40-60%

**Lazy Compression** (Sprint 5):
- Only compresses when `context_size > compression_threshold` (default: 10K chars)
- Tracks `compression_trigger_count`
- Logs compression decisions for observability

### 2. Comment Extractor (`utils/comment_extractor.py`)

**Purpose**: Parses LLM responses into structured comments.

**Schema**:
```python
class Comment:
    category: CommentCategory  # question, idea, concern, observation, recommendation
    text: str
    novelty_score: float  # 0.0-1.0
```

**LLM Prompt Format**:
```
Extract structured comments from the response in this JSON format:
[
  {"category": "idea", "text": "..."},
  {"category": "question", "text": "..."}
]

Categories: question, idea, concern, observation, recommendation
```

### 3. Context Compressor (`intelligence/context_compressor.py`)

**Purpose**: Reduces context size while preserving semantic meaning.

**Algorithms**:
- **Graph-based clustering** (Sprint 4): Groups similar comments using TF-IDF + cosine similarity
- **Semantic summarization**: Uses LLM to summarize clusters
- **Batch operations** (Sprint 4): Processes comments in batches for efficiency

**Key Methods**:
```python
class ContextCompressor:
    def compress_comments(self, meeting_id: UUID, round_num: int) -> CompressionMetrics:
        """Compress comments for a meeting round."""

    def _cluster_comments(self, comments: list[Comment]) -> list[list[Comment]]:
        """Cluster similar comments using graph-based approach."""

    def _summarize_cluster(self, cluster: list[Comment]) -> Comment:
        """Summarize a cluster into a single representative comment."""
```

**Metrics**:
```python
class CompressionMetrics:
    original_count: int
    compressed_count: int
    reduction_percentage: float
    compression_ratio: float  # compressed / original
```

### 4. Convergence Detector (`intelligence/convergence_detector.py`)

**Purpose**: Determines when meeting should stop (ideas exhausted).

**Detection Criteria**:
- **Novelty decline**: Average novelty score < threshold (default: 0.3) for 2+ rounds
- **Low new ideas**: <20% new comments compared to previous round
- **High compression**: Compression ratio <0.5 (context compresses well = repetitive)

**Key Methods**:
```python
class ConvergenceDetector:
    def check_convergence(self, meeting_id: UUID) -> ConvergenceCheckResult:
        """Check if meeting has converged."""

    def _calculate_novelty_trend(self, metrics: list[ConvergenceMetric]) -> float:
        """Calculate novelty score trend (negative = declining)."""
```

### 5. Export Service (`services/export_service.py`)

**Purpose**: Generate meeting artifacts in multiple formats.

**Supported Formats**:
- **Markdown**: Formatted text with headers, tables, code blocks
- **JSON**: Structured data with metadata wrapper
- **HTML**: Self-contained styled page with embedded CSS
- **Template**: Custom Jinja2 templates

**Key Methods**:
```python
class ExportService:
    def export_markdown(self, meeting_id: UUID, output_path: Path | None) -> str:
        """Export as markdown."""

    def export_json(self, meeting_id: UUID, output_path: Path | None, pretty: bool = True) -> str:
        """Export as JSON."""

    def export_html(self, meeting_id: UUID, output_path: Path | None) -> str:
        """Export as HTML with embedded CSS."""

    def export_with_template(self, meeting_id: UUID, template_name: str, output_path: Path | None) -> str:
        """Export using custom Jinja2 template."""
```

---

## API Reference

### Pydantic Models (`schemas.py`)

All API boundaries use Pydantic models for validation.

**MeetingCreate**:
```python
class MeetingCreate(BaseModel):
    topic: str  # 10-500 chars
    strategy: StrategyType  # sequential | greedy
    max_rounds: int  # 1-10
    agent_ids: list[UUID] | None  # Optional manual selection
```

**MeetingResponse**:
```python
class MeetingResponse(BaseModel):
    id: UUID
    topic: str
    strategy: StrategyType
    status: MeetingStatus  # created | running | paused | completed | failed
    current_round: int
    max_rounds: int
    total_comments: int
    total_cost: float
    context_size: int
    convergence_detected: bool
    stopping_reason: str | None
```

**CommentResponse**:
```python
class CommentResponse(BaseModel):
    id: UUID
    meeting_id: UUID
    agent_name: str
    round: int
    category: CommentCategory
    text: str
    novelty_score: float
    created_at: datetime
```

### Database Models (`models/`)

**Meeting** (`models/meeting.py`):
```python
class Meeting(Base):
    __tablename__ = "meetings"

    id: UUID = Column(UUID(as_uuid=True), primary_key=True)
    topic: str
    strategy: StrategyType
    status: MeetingStatus
    current_round: int
    max_rounds: int
    total_comments: int
    context_size: int
    total_cost: Decimal
    convergence_detected: bool
    stopping_reason: str | None

    # Relationships
    responses: list[Response] = relationship("Response", back_populates="meeting")
    comments: list[Comment] = relationship("Comment", back_populates="meeting")
    convergence_metrics: list[ConvergenceMetric] = relationship("ConvergenceMetric", back_populates="meeting")
```

**Agent** (`models/agent.py`):
```python
class Agent(Base):
    __tablename__ = "agents"

    id: UUID
    name: str
    role: str
    expertise: str
    system_prompt: str
    is_active: bool
    letta_definition: dict | None  # JSONB (Sprint 5: Letta integration)
```

**Comment** (`models/meeting.py`):
```python
class Comment(Base):
    __tablename__ = "comments"

    id: UUID
    meeting_id: UUID
    agent_id: UUID
    round: int
    category: CommentCategory
    text: str
    novelty_score: float
    parent_comment_id: UUID | None  # For compression clusters
    is_compressed: bool
```

---

## Database Schema

### Entity Relationship Diagram

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Meeting    │1      n │   Response   │n      1 │    Agent     │
│──────────────│◄────────┤──────────────├────────►│──────────────│
│ id (PK)      │         │ id (PK)      │         │ id (PK)      │
│ topic        │         │ meeting_id   │         │ name         │
│ strategy     │         │ agent_id     │         │ role         │
│ status       │         │ round        │         │ expertise    │
│ current_round│         │ response_text│         │ system_prompt│
│ max_rounds   │         │ cost         │         │ is_active    │
│ ...          │         └──────────────┘         │ letta_def    │
└──────┬───────┘                                  └──────────────┘
       │
       │1
       │
       │n
┌──────▼───────┐
│   Comment    │
│──────────────│
│ id (PK)      │
│ meeting_id   │
│ agent_id     │
│ round        │
│ category     │
│ text         │
│ novelty_score│
│ parent_id    │  ◄────┐ (self-referential for compression)
│ is_compressed│       │
└──────────────┘       │
       │               │
       └───────────────┘
```

### Key Tables

**meetings**: Core meeting records
**agents**: AI agent definitions
**responses**: Raw LLM responses per agent per round
**comments**: Extracted structured comments
**convergence_metrics**: Per-round metrics for convergence detection

### Indexes

```sql
CREATE INDEX idx_comments_meeting_round ON comments(meeting_id, round);
CREATE INDEX idx_comments_novelty ON comments(novelty_score DESC);
CREATE INDEX idx_responses_meeting ON responses(meeting_id);
CREATE INDEX idx_convergence_meeting ON convergence_metrics(meeting_id);
```

---

## Adding New Features

### Adding a New Agent

1. **Create agent definition:**
   ```python
   # In scripts/seed_agents.py or via SQL
   agent = Agent(
       id=uuid4(),
       name="Security Engineer",
       role="security",
       expertise="Application security, threat modeling, secure coding",
       system_prompt="You are a security engineer with expertise in...",
       is_active=True,
   )
   ```

2. **Seed to database:**
   ```bash
   python scripts/seed_agents.py
   ```

3. **Verify:**
   ```bash
   docker-compose exec postgres psql -U theboard -d theboard -c "SELECT name, role FROM agents WHERE name='Security Engineer';"
   ```

### Adding a New Export Format

1. **Extend `ExportService`:**
   ```python
   # In services/export_service.py
   def export_pdf(self, meeting_id: UUID, output_path: Path | None) -> bytes:
       """Export meeting as PDF."""
       data = self._get_meeting_data(meeting_id)

       # Generate PDF using reportlab or weasyprint
       pdf_content = self._generate_pdf(data)

       if output_path:
           output_path.write_bytes(pdf_content)

       return pdf_content
   ```

2. **Update CLI command:**
   ```python
   # In cli.py export command
   valid_formats = ["markdown", "json", "html", "template", "pdf"]

   # Add elif branch
   elif format == "pdf":
       result = export_service.export_pdf(uuid_id, output)
   ```

3. **Add tests:**
   ```python
   def test_export_pdf():
       service = ExportService()
       pdf = service.export_pdf(meeting_id)
       assert pdf.startswith(b'%PDF-1.')
   ```

### Adding a New Convergence Criterion

1. **Extend `ConvergenceDetector`:**
   ```python
   # In intelligence/convergence_detector.py
   def _check_semantic_similarity(self, meeting_id: UUID) -> bool:
       """Check if recent comments are semantically similar."""
       # Use embeddings to detect semantic convergence
       ...
       return similarity_score > 0.85
   ```

2. **Update `check_convergence`:**
   ```python
   def check_convergence(self, meeting_id: UUID) -> ConvergenceCheckResult:
       novelty_converged = self._check_novelty_decline(meeting_id)
       semantic_converged = self._check_semantic_similarity(meeting_id)

       if novelty_converged or semantic_converged:
           return ConvergenceCheckResult(converged=True, reason="...")
   ```

---

## Testing

### Running Tests

**Full test suite:**
```bash
pytest
```

**Specific test file:**
```bash
pytest tests/unit/test_comment_extractor.py
```

**With coverage:**
```bash
pytest --cov=src/theboard --cov-report=html
```

### Test Structure

```
tests/
├── unit/               # Isolated unit tests (no DB)
│   ├── test_comment_extractor.py
│   ├── test_convergence_detector.py
│   └── test_schemas.py
├── integration/        # Tests with real DB/Redis
│   ├── test_meeting_workflow.py
│   ├── test_export_service.py
│   └── test_redis_manager.py
└── fixtures/           # Shared test data
    └── sample_meetings.py
```

### Example Test

```python
import pytest
from uuid import uuid4
from theboard.services.meeting_service import create_meeting
from theboard.schemas import StrategyType

def test_create_meeting():
    """Test meeting creation with auto-agent selection."""
    meeting = create_meeting(
        topic="Test brainstorming topic",
        strategy=StrategyType.SEQUENTIAL,
        max_rounds=3,
        agent_count=5,
        auto_select=True,
    )

    assert meeting.id is not None
    assert meeting.topic == "Test brainstorming topic"
    assert meeting.strategy == StrategyType.SEQUENTIAL
    assert meeting.status == MeetingStatus.CREATED
```

---

## Deployment

### Docker Deployment

**1. Build image:**
```bash
docker build -t theboard:latest .
```

**2. Run with docker-compose:**
```bash
docker-compose up -d
```

**3. Verify:**
```bash
docker-compose ps
docker-compose exec theboard board --help
```

### Environment Variables

**Required:**
```bash
# LLM Configuration
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4-turbo-preview

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=theboard
POSTGRES_USER=theboard
POSTGRES_PASSWORD=theboard

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# RabbitMQ
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
```

**Optional:**
```bash
# Logging
LOG_LEVEL=INFO

# Convergence
CONVERGENCE_NOVELTY_THRESHOLD=0.3

# Compression
COMPRESSION_THRESHOLD=10000
```

### Production Checklist

- [ ] Set strong database passwords
- [ ] Use managed services (RDS, ElastiCache) instead of Docker containers
- [ ] Enable SSL/TLS for all connections
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure log aggregation (ELK, Loki)
- [ ] Set up backups (automated PostgreSQL dumps)
- [ ] Implement API rate limiting
- [ ] Use secrets management (AWS Secrets Manager, Vault)
- [ ] Set resource limits in docker-compose (memory, CPU)
- [ ] Enable health checks for all services

---

## Contributing

### Code Style

**Follow PEP 8:**
```bash
# Format with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/

# Type check with mypy
mypy src/
```

### Commit Messages

```
feat: Add PDF export functionality
fix: Resolve convergence detection bug
docs: Update API reference
test: Add unit tests for export service
refactor: Simplify context compression logic
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

---

## Architecture Decisions

### Why SQLAlchemy 2.0?

- Modern async support
- Improved type hints
- Better ORM performance
- Strong migration tooling (Alembic)

### Why Redis for Caching?

- Fast in-memory storage
- Built-in TTL support
- Pub/sub for events
- Widely deployed and stable

### Why RabbitMQ for Events?

- Reliable message delivery
- Support for complex routing
- Dead-letter queues for failed events
- Industry standard for event-driven systems

### Why Qdrant for Embeddings?

- Purpose-built vector database
- Fast similarity search
- Easy Docker deployment
- Excellent Python client

---

**Questions?** Check the [User Guide](./USER_GUIDE.md) or [Troubleshooting](./TROUBLESHOOTING.md).
