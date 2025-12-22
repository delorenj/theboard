# TheBoard - Multi-Agent Brainstorming Simulation System

A sophisticated multi-agent system for simulating brainstorming sessions with AI agents, featuring intelligent comment extraction, context management, and convergence detection.

## Features (Sprint 1 MVP)

- **Single-Agent Execution**: Run brainstorming sessions with domain expert agents
- **Intelligent Comment Extraction**: Automatically extract and categorize key insights
- **Persistent Storage**: PostgreSQL database for meetings, agents, responses, and comments
- **State Management**: Redis caching for meeting state and context
- **CLI Interface**: Rich terminal interface for creating and managing meetings
- **Structured Data**: Pydantic models for type-safe data validation

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

Edit `.env` and add your Anthropic API key:

```bash
ANTHROPIC_API_KEY=your_key_here
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

Edit `.env` and add your Anthropic API key:

```bash
ANTHROPIC_API_KEY=your_key_here
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

## Usage

### Running with Docker

#### Create a Meeting

```bash
docker compose exec theboard uv run board create --topic "Microservices vs Monolith architecture" --max-rounds 1
```

Options:
- `--topic, -t`: The brainstorming topic (required, 10-500 characters)
- `--strategy, -s`: Execution strategy (sequential or greedy, default: sequential)
- `--max-rounds, -r`: Maximum number of rounds (default: 5)
- `--agent-count, -n`: Number of agents to auto-select (default: 5)
- `--auto-select/--manual`: Auto-select agents based on topic (default: auto-select)

#### Run a Meeting

```bash
docker compose exec theboard uv run board run <meeting-id>
```

Options:
- `--interactive, -i`: Enable human-in-the-loop prompts (Sprint 4 feature)

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
├── docker-compose.yml   # Docker services
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

- **Python 3.12**: Core language
- **Typer**: CLI framework
- **Rich**: Terminal formatting
- **SQLAlchemy**: ORM and database toolkit
- **Alembic**: Database migrations
- **Pydantic**: Data validation
- **Anthropic Claude**: LLM provider
- **PostgreSQL**: Persistent storage
- **Redis**: Caching and state management
- **Qdrant**: Vector database for embeddings
- **RabbitMQ**: Message broker for events
- **Docker**: Containerization
- **pytest**: Testing framework

## License

See LICENSE file.

## Contributing

This project follows the BMAD (Business, Management, Architecture, Development) methodology for systematic development.

## Support

For issues, questions, or contributions, please open an issue on the repository
