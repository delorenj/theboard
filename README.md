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
- PostgreSQL (host port 5433 → container 5432)
- Redis (host port 6380 → container 6379)
- Qdrant (host ports 6333, 6334)
- RabbitMQ (host port 5673 → container 5672, management 15673 → 15672)

**Note on RabbitMQ:** The containerized RabbitMQ is mapped to port **5673** to avoid conflicts with native RabbitMQ instances. If you already run RabbitMQ natively on port 5672 (as part of Bloodbank or other pipelines), you can skip starting the RabbitMQ container and the app will connect to your native instance via the `RABBITMQ_URL` in `.env`

#### 4. Run CLI commands in container

```bash
# Create a meeting
docker compose exec theboard uv run board create --topic "Your brainstorming topic"

# Run a meeting
docker compose exec theboard uv run board run --meeting-id <meeting-id>

# Check meeting status
docker compose exec theboard uv run board status --meeting-id <meeting-id>
```

### Option 2: Local Development (Using Native Services)

For local development with hot-reloading, you can run just the infrastructure in Docker and the app locally. If you already have some services running natively (like RabbitMQ), you can skip those containers.

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

**Important:** If you're using native services (e.g., native RabbitMQ on port 5672), the `.env` file is already configured correctly. Just ensure:
- `RABBITMQ_URL=amqp://theboard:theboard_rabbit_pass@localhost:5672/` points to your native RabbitMQ
- Your native RabbitMQ has the credentials configured (user: `theboard`, password: `theboard_rabbit_pass`)

#### 3. Start only required infrastructure services

```bash
# Option A: Start only core services (if you have native RabbitMQ)
docker compose up -d postgres redis qdrant

# Option B: Start all infrastructure services (if you don't have native RabbitMQ)
docker compose up -d postgres redis rabbitmq qdrant
```

**Note:** The containerized RabbitMQ runs on port **5673** (not 5672) to avoid conflicts with native instances.

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

## Infrastructure Configuration

### RabbitMQ Setup

TheBoard uses RabbitMQ for event-driven communication (Sprint 4 Story 12). You have two options:

**Option 1: Use Native RabbitMQ (Recommended if you already have it)**

If you run RabbitMQ natively (e.g., as part of Bloodbank or another pipeline), TheBoard will connect to your existing instance:

1. Ensure your native RabbitMQ is running on port **5672** (default)
2. Create credentials for TheBoard:
   ```bash
   rabbitmqctl add_user theboard theboard_rabbit_pass
   rabbitmqctl set_permissions -p / theboard ".*" ".*" ".*"
   ```
3. Your `.env` file is already configured to use `localhost:5672`
4. Start Docker services **without** RabbitMQ: `docker compose up -d postgres redis qdrant`

**Option 2: Use Containerized RabbitMQ**

If you don't have native RabbitMQ, use the containerized version:

1. The containerized RabbitMQ runs on port **5673** (mapped from container's 5672) to avoid conflicts
2. Start all services: `docker compose up -d`
3. Update `.env` to use the containerized instance:
   ```bash
   RABBITMQ_URL=amqp://theboard:theboard_rabbit_pass@localhost:5673/
   ```

**Important:** The containerized RabbitMQ is on port **5673**, not 5672, specifically to avoid conflicts with native instances.

### Port Reference

| Service | Container Port | Host Port (Docker) | Native Port |
|---------|---------------|-------------------|-------------|
| PostgreSQL | 5432 | 5433 | 5432 |
| Redis | 6379 | 6380 | 6379 |
| RabbitMQ | 5672 | 5673 | 5672 |
| RabbitMQ Mgmt | 15672 | 15673 | 15672 |
| Qdrant | 6333 | 6333 | 6333 |
| Qdrant gRPC | 6334 | 6334 | 6334 |

## Technologies

- **Python 3.12**: Core language
- **Typer**: CLI framework
- **Rich**: Terminal formatting
- **SQLAlchemy**: ORM and database toolkit
- **Alembic**: Database migrations
- **Pydantic**: Data validation
- **Agno**: Multi-agent framework for LLM orchestration
- **OpenRouter**: LLM provider (unified access to Claude, GPT, etc.)
- **PostgreSQL**: Persistent storage
- **Redis**: Caching and state management
- **Qdrant**: Vector database for embeddings
- **RabbitMQ**: Message broker for events (port 5672 native, 5673 containerized)
- **Docker**: Containerization
- **pytest**: Testing framework

## License

See LICENSE file.

## Contributing

This project follows the BMAD (Business, Management, Architecture, Development) methodology for systematic development.

## Support

For issues, questions, or contributions, please open an issue on the repository
