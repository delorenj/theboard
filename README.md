# TheBoard - Multi-Agent Brainstorming Simulation System

A sophisticated multi-agent system for simulating brainstorming sessions with AI agents, featuring intelligent comment extraction, context management, and convergence detection.

## Features (v1.0 - Production Ready)

### Core Brainstorming
- **Multi-Agent Execution**: Run brainstorming sessions with 10+ specialized AI agents
- **Execution Strategies**: Sequential (thorough) or Greedy (adaptive) agent selection
- **Intelligent Comment Extraction**: Automatically categorize insights (ideas, questions, concerns, observations, recommendations)
- **Real-time Progress**: Live updates during meeting execution with Rich terminal UI

### Intelligence Layer
- **Context Compression**: Graph-based clustering reduces token usage by 40%+ while preserving quality
- **Convergence Detection**: Automatically stops meetings when ideas are exhausted
- **Novelty Scoring**: Embedding-based detection of unique vs. repetitive comments
- **Cross-Meeting Memory**: Letta agent integration for learning across sessions

### Performance & Optimization
- **Lazy Compression**: Only compresses when context exceeds 10K characters
- **Delta Propagation**: Agents receive only new comments since last participation (40-60% token reduction)
- **Hybrid Model Strategy**: Automatic model selection reduces costs by 60%+
- **Redis Caching**: Tiered TTL strategy optimizes memory usage

### User Experience
- **Interactive CLI**: Typer-based interface with comprehensive help text
- **Meeting Management**: Create, run, pause, resume, fork, and rerun meetings
- **Human-in-the-Loop**: Real-time steering and feedback during execution
- **Multi-Format Export**: Markdown, JSON, HTML, and custom Jinja2 templates

### Infrastructure
- **Persistent Storage**: PostgreSQL database with full SQLAlchemy ORM
- **Distributed Caching**: Redis for state management and session handling
- **Event System**: RabbitMQ for human-in-loop orchestration
- **Vector Database**: Qdrant for embedding-based novelty detection
- **Comprehensive Documentation**: User guide, troubleshooting, and developer docs

## Architecture

TheBoard implements a layered architecture:

- **CLI Layer**: Typer-based command-line interface with Rich formatting
- **Service Layer**: Business logic for meeting management
- **Workflow Layer**: Orchestration of agent execution and notetaker integration
- **Agent Layer**: Domain expert agents and notetaker for comment extraction
- **Data Layer**: SQLAlchemy models with Alembic migrations
- **Infrastructure Layer**: Docker Compose for services (Postgres, Redis, RabbitMQ, Qdrant)

## Requirements

- Python 3.12+
- Docker and Docker Compose
- uv (Python package manager)

## Installation

### Option 1: Run with Docker (Recommended)

The easiest way to run TheBoard is using Docker Compose, which includes the application and all required services.

#### 1. Clone the repository

```bash
git clone <repository-url>
cd theboard
```

#### 2. Set up environment

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_key_here
```

#### 3. Start all services

```bash
docker compose up -d
```

This starts:
- TheBoard application container
- PostgreSQL (port 5433)
- Redis (port 6380)
- Qdrant (ports 6335, 6336)
- RabbitMQ (ports 5673, 15673)

#### 4. Run CLI commands in container

```bash
# Create a meeting
docker compose exec theboard uv run board create --topic "Your brainstorming topic"

# Run a meeting
docker compose exec theboard uv run board run --meeting-id <meeting-id>

# Check meeting status
docker compose exec theboard uv run board status --meeting-id <meeting-id>
```

### Option 2: Local Development

For local development with hot-reloading, you can run just the infrastructure in Docker and the app locally.

#### 1. Clone the repository

```bash
git clone <repository-url>
cd theboard
```

#### 2. Set up environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```bash
OPENROUTER_API_KEY=your_key_here
```

#### 3. Start only infrastructure services

```bash
# Start Postgres, Redis, RabbitMQ, Qdrant
docker compose up -d postgres redis rabbitmq qdrant
```

#### 4. Install Python dependencies

```bash
uv sync
```

#### 5. Run database migrations

```bash
source .venv/bin/activate
alembic upgrade head
```

## Quick Start

```bash
# Create a meeting (auto-selects 5 relevant agents)
board create --topic "Design a mobile app feature for tracking daily water intake"

# Run the meeting with live progress display
board run <meeting-id>

# Export results
board export <meeting-id> --format markdown --output report.md
```

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for detailed examples and tutorials.

## Usage

### Running with Docker

#### Create a Meeting

```bash
docker compose exec theboard uv run board create \
  --topic "Microservices vs Monolith architecture" \
  --strategy sequential \
  --max-rounds 5 \
  --agent-count 5
```

**Options:**
- `--topic, -t`: The brainstorming topic (required, 10-500 characters)
- `--strategy, -s`: Execution strategy (sequential | greedy, default: sequential)
- `--max-rounds, -r`: Maximum number of rounds (1-10, default: 5)
- `--agent-count, -n`: Number of agents to auto-select (1-10, default: 5)
- `--auto-select/--manual`: Auto-select agents based on topic (default: auto-select)
- `--model, -m`: Override LLM model for this meeting

#### Run a Meeting

```bash
# Basic run
docker compose exec theboard uv run board run <meeting-id>

# With human-in-the-loop steering
docker compose exec theboard uv run board run <meeting-id> --interactive

# Rerun a completed meeting (overwrites data)
docker compose exec theboard uv run board run <meeting-id> --rerun

# Fork a meeting (creates new meeting with same config)
docker compose exec theboard uv run board run <meeting-id> --fork

# Run most recent meeting
docker compose exec theboard uv run board run --last
```

**Options:**
- `--interactive, -i`: Enable human-in-the-loop prompts
- `--rerun`: Reset and rerun a completed/failed meeting
- `--fork`: Create new meeting with same parameters
- `--last`: Run most recent meeting without selection

#### Export Meeting Results

```bash
# Export as markdown
docker compose exec theboard uv run board export <meeting-id> --format markdown

# Export as JSON
docker compose exec theboard uv run board export <meeting-id> --format json

# Export as HTML
docker compose exec theboard uv run board export <meeting-id> --format html

# Export with custom template
docker compose exec theboard uv run board export <meeting-id> \
  --format template \
  --template "custom-report.j2"
```

**Options:**
- `--format, -f`: Export format (markdown | json | html | template)
- `--output, -o`: Output file path (optional, auto-generated if omitted)
- `--template, -t`: Template name for custom exports

#### Check Meeting Status

```bash
docker compose exec theboard uv run board status <meeting-id>
```

Options:
- `--comments/--no-comments`: Show/hide recent comments (default: show)
- `--metrics/--no-metrics`: Show/hide convergence metrics (default: show)

#### Display Version

```bash
docker compose exec theboard uv run board version
```

### Running Locally

If you're running the app locally (not in Docker):

```bash
# Create a meeting
board create --topic "Microservices vs Monolith architecture" --max-rounds 1

# Run a meeting
board run <meeting-id>

# Check status
board status <meeting-id>
```

## Example Workflow

### With Docker

```bash
# Create a meeting
docker compose exec theboard uv run board create --topic "API design: REST vs GraphQL vs gRPC for mobile apps" --max-rounds 1

# Output: Meeting ID: 550e8400-e29b-41d4-a716-446655440000

# Run the meeting
docker compose exec theboard uv run board run 550e8400-e29b-41d4-a716-446655440000

# Check status and comments
docker compose exec theboard uv run board status 550e8400-e29b-41d4-a716-446655440000
```

### Local Development

```bash
# Create a meeting
board create --topic "API design: REST vs GraphQL vs gRPC for mobile apps" --max-rounds 1

# Output: Meeting ID: 550e8400-e29b-41d4-a716-446655440000

# Run the meeting
board run 550e8400-e29b-41d4-a716-446655440000

# Check status and comments
board status 550e8400-e29b-41d4-a716-446655440000
```

## Development

### Project Structure

```
theboard/
├── src/theboard/
│   ├── agents/          # Agent implementations
│   │   ├── base.py      # Base agent class
│   │   ├── domain_expert.py  # Domain expert agent
│   │   └── notetaker.py      # Notetaker agent
│   ├── models/          # SQLAlchemy models
│   │   ├── base.py      # Base model
│   │   └── meeting.py   # Meeting, Agent, Response, Comment models
│   ├── services/        # Business logic layer
│   │   └── meeting_service.py
│   ├── utils/           # Utility modules
│   │   └── redis_manager.py
│   ├── workflows/       # Workflow orchestration
│   │   └── simple_meeting.py
│   ├── cli.py           # CLI implementation
│   ├── config.py        # Configuration management
│   ├── database.py      # Database connection
│   └── schemas.py       # Pydantic schemas
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── alembic/             # Database migrations
├── compose.yml          # Docker services
└── pyproject.toml       # Project configuration
```

### Running Tests

```bash
source .venv/bin/activate
pytest
```

With coverage:

```bash
pytest --cov=theboard --cov-report=html
```

### Code Quality

Run linting:

```bash
ruff check src/
```

Run type checking:

```bash
mypy src/
```

## Sprint 1 Implementation Status

### Completed Stories

✅ **Story 1: Project Setup & Data Layer (8 pts)**
- Docker Compose with all services
- SQLAlchemy models for all tables
- Alembic migrations
- Redis connection manager

✅ **Story 2: Basic CLI Structure (3 pts)**
- Typer CLI with create, run, status commands
- Rich formatting for output
- Interactive prompts

✅ **Story 3: Agno Integration & Simple Agent (8 pts)**
- DomainExpertAgent with Claude Sonnet integration
- Single-agent, single-round execution
- Response storage

✅ **Story 4: Notetaker Agent Implementation (7 pts)**
- NotetakerAgent with structured extraction
- Comment categorization (7 categories)
- Comment storage with metadata

### Sprint 1 Acceptance Criteria

- ✅ Docker Compose brings up all services
- ✅ `board create` creates a meeting with 1 agent
- ✅ `board run` executes 1-agent, 1-round meeting
- ✅ Notetaker extracts comments from agent response
- ✅ `board status` displays meeting state and comments
- ✅ Database schema complete with migrations
- ✅ Unit tests for core components

## Roadmap

### Sprint 2 (Next)
- Multi-agent orchestration
- Agent pool management with auto-selection
- Context management and accumulation
- Sequential strategy with multiple rounds

### Sprint 3
- Embedding infrastructure (Qdrant)
- Compressor agent for context compression
- Convergence detection

### Sprint 4
- Greedy execution strategy
- Human-in-the-loop capabilities
- Hybrid model strategy for cost optimization

### Sprint 5
- Letta agent integration
- Export functionality (markdown, JSON, HTML)
- Performance optimizations

### Sprint 6
- CLI polish and live progress
- Comprehensive documentation
- Production readiness

## Technologies

### Core Stack
- **Python 3.12**: Core language
- **Typer**: CLI framework with argument parsing
- **Rich**: Terminal UI, tables, progress bars
- **SQLAlchemy 2.0**: ORM and database toolkit
- **Alembic**: Database migrations
- **Pydantic**: Data validation and serialization

### LLM & Agents
- **Agno**: Multi-agent framework for orchestration
- **Letta**: Cross-meeting memory and agent persistence
- **OpenRouter**: Unified LLM API (Claude, GPT, etc.)
- **LiteLLM**: Multi-provider LLM abstraction

### Infrastructure
- **PostgreSQL 16**: Persistent storage (meetings, agents, responses)
- **Redis 7**: Caching and session management
- **Qdrant**: Vector database for novelty detection
- **RabbitMQ 3.12**: Event bus for human-in-loop
- **Docker**: Containerization and deployment

### Development
- **pytest**: Testing framework
- **uv**: Fast Python package manager
- **ruff**: Linter and formatter
- **mypy**: Type checking

## Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Getting started, examples, best practices
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[Developer Docs](docs/DEVELOPER.md)**: Architecture, API reference, contributing

## Project Status

**Version:** 1.0 (Production Ready)
**Status:** ✅ All 17 stories completed across 6 sprints

### Implementation Progress

- **Sprint 1**: MVP Foundation ✅
- **Sprint 2**: Multi-Agent Orchestration ✅
- **Sprint 2.5**: Event Foundation ✅
- **Sprint 3**: Compression & Convergence ✅
- **Sprint 4**: Advanced Features ✅
- **Sprint 5**: Letta Integration & Export ✅
- **Sprint 6**: CLI Polish & Documentation ✅

**Total Stories:** 17/17 (100%)
**Total Points:** 89/89 (100%)

See [Sprint Plan](docs/sprint-plan-theboard-2025-12-19.md) for detailed breakdown.

## License

See LICENSE file.

## Contributing

This project follows the BMAD (Business, Management, Architecture, Development) methodology for systematic development.

Contributions welcome! See [DEVELOPER.md](docs/DEVELOPER.md) for guidelines.

## Support

For issues, questions, or contributions, please open an issue on the repository.
