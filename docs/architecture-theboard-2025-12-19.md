# System Architecture: TheBoard

**Project:** TheBoard - Multi-Agent Brainstorming Simulation System
**Date:** 2025-12-19
**Version:** 1.0
**Status:** Planning
**Author:** System Architect (BMAD Method)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architectural Drivers](#architectural-drivers)
3. [High-Level Architecture](#high-level-architecture)
4. [Technology Stack](#technology-stack)
5. [System Components](#system-components)
6. [Data Architecture](#data-architecture)
7. [API Design](#api-design)
8. [NFR Coverage](#nfr-coverage)
9. [Security Architecture](#security-architecture)
10. [Scalability & Performance](#scalability--performance)
11. [Reliability & Availability](#reliability--availability)
12. [Development & Deployment](#development--deployment)
13. [Traceability & Trade-offs](#traceability--trade-offs)

---

## 1. Executive Summary

TheBoard implements an event-driven, layered architecture designed for multi-agent AI orchestration with emphasis on token cost optimization, context management, and observability. The system coordinates domain-expert AI agents through structured brainstorming rounds, applying mathematical convergence detection and three-tier compression to produce refined artifacts.

**Architectural Pattern:** Event-Driven Layered Architecture with Modular Orchestration

**Key Characteristics:**
- **Separation of Concerns:** CLI → Orchestration → Agents → Data
- **Event-Driven Coordination:** RabbitMQ for async human-in-loop and progress notifications
- **Stateful Orchestration:** Agno workflows for round management with Redis state caching
- **Cost-Optimized LLM Usage:** Hybrid model strategy (cheap workers → expensive leaders)
- **Mathematical Convergence:** Embedding-based novelty detection for automatic stopping

**Target Scale:**
- 10 concurrent agents per meeting
- 100+ meetings in agent long-term memory
- <5 minute total meeting execution (5 rounds, 5 agents)
- <$2 cost per meeting with hybrid strategy

---

## 2. Architectural Drivers

Architectural drivers are NFRs that heavily influence design decisions.

### 2.1 Critical Drivers

#### AD-001: Performance - Meeting Execution Latency
**Requirement:** Sequential strategy <30s per round for 5 agents, total meeting <5 minutes

**Architecture Impact:**
- Async/await throughout (Python asyncio)
- Redis caching for hot data (meeting state, active context)
- Postgres connection pooling
- Lazy compression (only when context exceeds threshold)
- Delta propagation (agents receive only new comments)

**Validation:** Load testing with 5-agent, 5-round meetings under 5 minutes

---

#### AD-002: Token Cost Optimization
**Requirement:** Hybrid model strategy must reduce costs by >60% compared to all-Opus baseline

**Architecture Impact:**
- Dynamic model promotion based on engagement metrics
- Selective agent activation (only engaged agents in round 2+)
- Three-tier compression to minimize context size
- Lazy compression triggers
- Per-agent, per-round token tracking

**Validation:** Cost tracking dashboard, A/B comparison with all-Opus baseline

---

#### AD-003: Context Management
**Requirement:** Context size <15K tokens for 5-round, 5-agent meeting

**Architecture Impact:**
- Three-tier compression (embedding clustering → LLM merge → outlier removal)
- Delta propagation to reduce redundant context transmission
- Context size monitoring with alerts
- Aggressive similarity thresholds (0.85 for clustering)

**Validation:** Context size tracking per round, alert at 15K tokens

---

#### AD-004: Scalability - Agent Memory
**Requirement:** Handle 100+ past meetings in agent memory without degradation

**Architecture Impact:**
- Letta framework for persistent agent memory
- agent_memory table in Postgres with JSONB for flexible schema
- Vector similarity search for memory recall (Qdrant)
- Memory retrieval only when relevant (topic similarity > 0.7)

**Validation:** Test with 100+ meeting history, measure memory retrieval latency <1s

---

#### AD-005: Reliability - State Persistence
**Requirement:** Meeting state persisted to survive crashes, meeting resume capability

**Architecture Impact:**
- Redis for active state (current_round, current_agent, turn_queue)
- Postgres for full audit trail (all responses pre-compression)
- Idempotent event handlers (RabbitMQ)
- Transaction boundaries for database writes

**Validation:** Crash recovery testing, resume from mid-meeting

---

#### AD-006: Observability - Structured Logging
**Requirement:** Structured logging with correlation IDs, metrics for token usage/costs/latency

**Architecture Impact:**
- JSON logging format for all key events
- Correlation IDs per meeting execution
- Token/cost tracking per agent, per round, per meeting
- Latency metrics (round duration, compression time, convergence)

**Validation:** Log aggregation, metrics dashboard, trace correlation

---

### 2.2 Secondary Drivers

#### AD-007: Compression Quality
**Requirement:** Compress comments by 40-60% while preserving information

**Architecture Impact:**
- Embedding-based clustering using sentence-transformers + Qdrant
- LLM semantic merge using Claude Sonnet
- Full audit trail for quality validation
- Compression quality metrics

---

#### AD-008: Convergence Detection
**Requirement:** Automatically stop when novelty < 0.2 for 2 consecutive rounds

**Architecture Impact:**
- Novelty calculation using embedding similarity
- Per-round convergence metrics in database
- Configurable thresholds

---

## 3. High-Level Architecture

### 3.1 Architectural Pattern

**Event-Driven Layered Architecture with Modular Orchestration**

This pattern combines:
- **Layered Architecture:** Clear vertical separation (CLI → Orchestration → Agents → Data)
- **Event-Driven Architecture:** Async communication via RabbitMQ for human-in-loop
- **Modular Orchestration:** Agno workflows as pluggable orchestration logic

**Rationale:**
- **Layered:** Enforces separation of concerns, testability, maintainability
- **Event-Driven:** Decouples execution from human availability, enables async workflows
- **Modular:** Agno provides state management and async coordination without custom plumbing

**Trade-offs:**
- ✓ Gain: Clear boundaries, easy testing, async human-in-loop, observable state
- ✗ Lose: Slight latency overhead from event bus, Agno framework learning curve

---

### 3.2 Component Overview

```
┌──────────────────────────────────────────────────────┐
│              CLI Layer (Typer + Rich)                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │   create   │  │    run     │  │   status   │    │
│  │  command   │  │  command   │  │  command   │    │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘    │
│         │ Invoke        │ Invoke        │ Query     │
└─────────┼───────────────┼───────────────┼───────────┘
          ↓               ↓               ↓
┌─────────────────────────────────────────────────────┐
│        Orchestration Layer (Agno Workflows)         │
│  ┌─────────────────┐  ┌──────────────┐            │
│  │ Meeting         │  │ Agent        │            │
│  │ Coordinator     │──│ Registry     │            │
│  │ (Workflow)      │  │              │            │
│  └────┬────────────┘  └──────────────┘            │
│       │ Manages                                     │
│  ┌────▼────────────┐  ┌──────────────┐            │
│  │ Context         │  │ Artifact     │            │
│  │ Manager         │  │ Manager      │            │
│  └─────────────────┘  └──────────────┘            │
└──────────┬──────────────────────────────┬───────────┘
           │ Coordinates                  │ Emits Events
           ↓                              ↓
┌──────────────────────────┐    ┌────────────────────┐
│   Agent Layer (Letta)    │    │  Event Bus         │
│  ┌────────────────────┐  │    │  (RabbitMQ)        │
│  │ Participant Agents │  │    │  ┌──────────────┐  │
│  │ (Domain Experts)   │  │    │  │ meeting.*    │  │
│  └────────────────────┘  │    │  │ events       │  │
│  ┌────────────────────┐  │    │  └──────────────┘  │
│  │ Notetaker Agent    │  │    │         │          │
│  │ (Extraction)       │  │    │         │ Consumed │
│  └────────────────────┘  │    │         ↓          │
│  ┌────────────────────┐  │    │  ┌──────────────┐  │
│  │ Compressor Agent   │  │    │  │ CLI Event    │  │
│  │ (Compression)      │  │    │  │ Consumer     │  │
│  └────────────────────┘  │    │  └──────────────┘  │
└──────────┬───────────────┘    └────────────────────┘
           │ LLM calls + memory access
           ↓
┌──────────────────────────────────────────────────────┐
│                    Data Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Postgres   │  │    Redis    │  │   Qdrant    │ │
│  │ (Persistent │  │  (Session   │  │  (Vector    │ │
│  │  Storage)   │  │   State)    │  │ Embeddings) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│  - meetings       - meeting:*:state - comment       │
│  - agents         - meeting:*:round:* embeddings    │
│  - responses      - agent:*:performance             │
│  - comments       - compression:*                   │
│  - agent_memory   (TTL: 7 days)                     │
└──────────────────────────────────────────────────────┘
```

---

### 3.3 Data Flow

#### Meeting Creation Flow
```
User
  → CLI: board create <topic>
    → Interactive wizard (agent selection)
      → Meeting Coordinator: initialize_meeting()
        → Postgres: INSERT into meetings, meeting_agents
        → Redis: SET meeting:{id}:state (initial state)
        → RabbitMQ: PUBLISH meeting.created event
          → Return: meeting_id to CLI
```

#### Round Execution Flow (Sequential Strategy)
```
Meeting Coordinator: sequential_round(context, round_num)
  → For each agent in turn_queue:
    ┌─ Agent: full_response(context)
    │   → LLM API: Claude/DeepSeek (async call)
    │     → Return: response_text
    │       → Postgres: INSERT into responses
    │         → Notetaker: extract_comments(response_text)
    │           → LLM API: Claude Sonnet (structured extraction)
    │             → Postgres: INSERT into comments
    │               → Compressor: compress(comments) [if context > threshold]
    │                 ├─ Qdrant: embed_texts(comments)
    │                 ├─ Qdrant: similarity_search() [clustering]
    │                 └─ LLM API: Claude Sonnet (semantic merge)
    │                   → Return: compressed_comments
    │                     → Context Manager: append_to_context(context, compressed)
    │                       → Redis: UPDATE meeting:{id}:state (active_context)
    └─ Next agent in sequence

  → Convergence Check: check_convergence(current, previous)
    ├─ Qdrant: embed_texts(current_comments, prev_comments)
    ├─ Calculate: novelty score
    └─ Postgres: INSERT into convergence_metrics
      → If converged: STOP, emit meeting.convergence.detected
      → Else: Next round

  → RabbitMQ: PUBLISH meeting.round.completed event
```

#### Artifact Export Flow
```
User
  → CLI: board export <meeting-id> --format markdown
    → Artifact Manager: generate_artifact(meeting_id, format)
      → Postgres: SELECT final_artifact FROM meetings
      → Context Manager: retrieve_context(meeting_id)
        → Redis: GET meeting:{id}:state (active_context)
          → Format: apply_template(context, format)
            → Return: artifact_text to CLI
```

---

## 4. Technology Stack

### 4.1 Core Technologies

#### Python 3.12
**Purpose:** Primary programming language

**Rationale:**
- Native async/await support (critical for parallel agent execution)
- Rich ecosystem for AI/ML (sentence-transformers, pydantic)
- Your preferred language per associative cloud
- Type hints + mypy for strict typing

**Trade-offs:**
- ✓ Gain: Rapid development, extensive libraries, strong typing
- ✗ Lose: Slightly slower than compiled languages (mitigated by async)

---

#### Agno (Agent Orchestration Framework)
**Purpose:** Workflow orchestration for meeting coordination

**Rationale:**
- Native state management (eliminates custom state machine)
- Async task composition (sequential/parallel execution)
- Tool/skill abstraction (agents as composable skills)
- Built for agentic workflows

**Trade-offs:**
- ✓ Gain: State management, async coordination, observability hooks
- ✗ Lose: Early-stage framework risk, learning curve

**Mitigation:** Prototype MVP early, abstract orchestration layer, fallback to direct async/await

---

#### Letta (Agent Framework)
**Purpose:** Agent definitions with persistent memory

**Rationale:**
- Native memory persistence (previous_meetings, learned_patterns)
- Tool integration for agent capabilities
- Supports cross-meeting learning
- Memory retrieval via semantic search

**Trade-offs:**
- ✓ Gain: Long-term memory, structured agent definitions, tool integration
- ✗ Lose: Integration complexity, migration effort from plaintext

**Mitigation:** Start with plaintext, migrate incrementally, support both formats

---

#### Typer + Rich (CLI)
**Purpose:** Command-line interface with rich formatting

**Rationale:**
- Typer: Type-safe CLI argument parsing
- Rich: Tables, progress bars, live updates
- Aligns with your preference for observability
- Interactive prompts for human-in-loop

**Trade-offs:**
- ✓ Gain: Type safety, excellent UX, live progress display
- ✗ Lose: None significant

---

### 4.2 Data Stores

#### PostgreSQL 15
**Purpose:** Primary persistent storage

**Rationale:**
- Relational model fits structured data (meetings, agents, responses)
- JSONB for flexible schemas (agent_memory, letta_definition)
- Foreign key constraints enforce data integrity
- Transaction support for atomic writes
- Your preferred SQL database per associative cloud

**Trade-offs:**
- ✓ Gain: ACID guarantees, relational integrity, JSONB flexibility
- ✗ Lose: Vertical scaling limits (mitigated by read replicas)

**Schema Highlights:**
- `meetings`: Meeting definitions and final artifacts
- `responses`: Full audit trail (all agent responses pre-compression)
- `comments`: Extracted comments with novelty scores
- `agent_memory`: Letta memory persistence (JSONB)
- `convergence_metrics`: Per-round convergence tracking

---

#### Redis 7
**Purpose:** Session state cache and hot data

**Rationale:**
- Sub-millisecond latency for active meeting state
- TTL support (auto-expire completed meetings after 7 days)
- In-memory performance for real-time operations
- Your preferred cache per associative cloud

**Trade-offs:**
- ✓ Gain: Extreme speed, TTL automation, simple key-value model
- ✗ Lose: Volatile (mitigated by Postgres persistence)

**Key Patterns:**
- `meeting:{id}:state`: Active meeting state (current_round, current_agent, context)
- `meeting:{id}:round:{r}:comments`: Round comments cache (TTL: 24h)
- `agent:{name}:performance`: Per-agent engagement metrics (TTL: 7 days)

---

#### Qdrant (Vector Database)
**Purpose:** Comment embeddings for compression and convergence

**Rationale:**
- Optimized for vector similarity search
- Cosine similarity for embedding clustering
- Batch processing for embedding operations
- Your preference for Qdrant per associative cloud

**Trade-offs:**
- ✓ Gain: Fast similarity search, batch embeddings, scalable
- ✗ Lose: Additional infrastructure component

**Usage:**
- Comment embedding storage
- Clustering for compression (similarity > 0.85)
- Novelty calculation (overlap detection via embeddings)

---

#### RabbitMQ 3.12
**Purpose:** Event bus for async communication

**Rationale:**
- Event-driven architecture for human-in-loop
- Decouples execution from CLI consumer
- Reliable message delivery (acknowledgments)
- Your preference for RabbitMQ per associative cloud

**Trade-offs:**
- ✓ Gain: Async workflows, decoupling, human-in-loop flexibility
- ✗ Lose: Additional complexity, message broker management

**Exchange/Routing:**
- Exchange: `theboard.events`
- Routing keys: `meeting.{event_type}`
- Events: `agent.response.ready`, `meeting.round.completed`, `meeting.convergence.detected`, etc.

---

### 4.3 LLM Providers

#### Anthropic Claude (Opus 4.5, Sonnet 4.5)
**Purpose:** High-quality LLM for leader agents, notetaker, compressor

**Rationale:**
- Opus 4.5: Best-in-class reasoning for promoted leader agents
- Sonnet 4.5: Fast, cost-effective for notetaker/compressor
- Structured output support (Pydantic models)

**Cost Model:**
- Opus: ~$0.015 per 1K input tokens
- Sonnet: ~$0.003 per 1K input tokens

---

#### DeepSeek R3
**Purpose:** Cheap worker agents for hybrid model strategy

**Rationale:**
- ~$0.001 per 1K input tokens (15x cheaper than Opus)
- Adequate quality for initial rounds
- Cost optimization: start cheap, promote top 20% to Opus

**Cost Model:**
- DeepSeek: ~$0.001 per 1K input tokens

**Hybrid Strategy:**
- Round 1: All agents use DeepSeek
- Round 2+: Top 20% (by engagement score) promoted to Opus
- Target: >60% cost reduction

---

### 4.4 Supporting Libraries

#### Pydantic
**Purpose:** Data validation and strict typing

**Rationale:**
- Enforces strict schemas for Comment, Meeting, Agent models
- Validation errors caught early
- Your preference for strict typing per associative cloud

---

#### SQLAlchemy
**Purpose:** ORM for Postgres

**Rationale:**
- Type-safe database operations
- Transaction management
- Migration support via Alembic

---

#### sentence-transformers
**Purpose:** Text embeddings for similarity

**Rationale:**
- Fast local embeddings (no API calls)
- Cosine similarity for clustering
- Batch processing support

---

#### aiormq
**Purpose:** Async RabbitMQ client

**Rationale:**
- Native async/await support
- Integrates with Python asyncio

---

### 4.5 Development Tools

- **uv:** Python package manager (faster than pip, your preference)
- **ruff:** Linting and formatting (replaces black + isort)
- **mypy:** Type checking with --strict mode
- **pytest:** Unit and integration testing
- **Docker Compose:** Local dev environment

---

## 5. System Components

### 5.1 CLI Layer

#### Component: Typer CLI Application
**Purpose:** Command-line interface for user interaction

**Responsibilities:**
- Parse commands: `create`, `run`, `status`, `export`
- Interactive wizards (agent selection, configuration)
- Live progress display using Rich
- Event consumption for human-in-loop

**Interfaces:**
- Input: Command-line arguments, interactive prompts
- Output: Rich-formatted tables, progress bars, status messages
- Dependencies: Orchestration Layer (Meeting Coordinator, Agent Registry)

**FRs Addressed:** F6 (CLI Interface)

**Implementation Notes:**
- Single Typer app instance at /home/delorenj/code/theboard/src/theboard/cli.py
- Command handlers delegate to orchestration layer
- Rich Live() for streaming progress
- Event consumer runs in background thread

---

### 5.2 Orchestration Layer

#### Component: Meeting Coordinator (Agno Workflow)
**Purpose:** Orchestrate multi-round agent discussions with convergence detection

**Responsibilities:**
- Initialize meeting state (Postgres + Redis)
- Execute sequential or greedy strategies
- Manage turn ordering and round progression
- Detect convergence via novelty calculation
- Emit events for human-in-loop decision points
- Generate final artifacts

**Interfaces:**
- Input: `run()` invoked by CLI with meeting_id
- Output: Final artifact (string), convergence metrics
- Dependencies: Agent Registry, Context Manager, Notetaker, Compressor, Participant Agents

**FRs Addressed:** F2 (Agent Orchestration), F5 (Convergence Detection)

**Implementation Notes:**
- Agno Workflow class: `TheboardMeeting`
- State management via Agno context + Redis caching
- Async execution with asyncio.gather for parallel agent calls (greedy)
- Convergence check after each round
- Event emission via RabbitMQ producer

**Key Methods:**
```python
class TheboardMeeting(Workflow):
    async def run() -> str
    async def sequential_round(context: str, round_num: int) -> list[Comment]
    async def greedy_round(context: str, round_num: int) -> list[Comment]
    async def check_convergence(current: list[Comment], all_comments: list[Comment]) -> bool
    async def generate_artifact(context: str, comments: list[Comment]) -> str
```

---

#### Component: Agent Registry
**Purpose:** Manage agent pool, capability indexing, auto-selection

**Responsibilities:**
- Load agents from plaintext or Letta format
- Index agent expertise for semantic search
- Auto-select teams based on topic embedding similarity
- Track agent performance across meetings
- Manage agent lifecycle (load, cache, update)

**Interfaces:**
- Input: `load_pool(pool_dir)`, `auto_select_team(topic, artifact_type)`
- Output: List of Agent objects
- Dependencies: Qdrant (for expertise embeddings), Postgres (agent storage)

**FRs Addressed:** F7 (Agent Pool Management)

**Implementation Notes:**
- Singleton pattern (single registry instance per process)
- Agent pool loaded from /home/delorenj/code/DeLoDocs/AI/Agents/Generic
- Expertise embeddings stored in Qdrant
- Auto-selection: cosine similarity between topic embedding and agent expertise
- Performance tracking: agent_performance table in Postgres

**Key Methods:**
```python
class AgentRegistry:
    async def load_pool(pool_dir: Path) -> list[Agent]
    async def auto_select_team(topic: str, artifact_type: str, count: int = 5) -> list[Agent]
    async def get_agent_by_name(name: str) -> Agent
    async def track_performance(meeting_id: str, agent_name: str, metrics: dict)
    async def get_performance(meeting_id: str, agent_name: str) -> dict
```

---

#### Component: Context Manager
**Purpose:** Manage cumulative context evolution and compression orchestration

**Responsibilities:**
- Build cumulative context (Xr = Xr-1 ∪ σ(comments))
- Cache active context in Redis
- Archive context history to Postgres
- Delta propagation (return only new comments since agent's last turn)
- Context size tracking with alerts
- Trigger compression when context exceeds threshold

**Interfaces:**
- Input: `append_to_context(context, comments)`, `get_delta(agent_name, since_round)`
- Output: Updated context (string), delta comments (list)
- Dependencies: Redis (caching), Postgres (archival), Compressor (compression)

**FRs Addressed:** F4 (Context Management)

**Implementation Notes:**
- Context stored as plain text string (cumulative)
- Redis key: `meeting:{id}:state` → `active_context` field
- Delta tracking: per-agent `last_seen_round` in state
- Lazy compression: only when len(context) > 10K chars
- Alert at 15K tokens (estimate: 4 chars per token)

**Key Methods:**
```python
class ContextManager:
    async def append_to_context(context: str, comments: list[Comment]) -> str
    async def get_delta(agent_name: str, since_round: int) -> list[Comment]
    def get_context_size() -> int
    async def cache_context(meeting_id: str, context: str)
    async def retrieve_context(meeting_id: str) -> str
```

---

#### Component: Artifact Manager
**Purpose:** Generate final artifacts in multiple formats

**Responsibilities:**
- Parse cumulative context into structured artifact
- Apply format templates (markdown, JSON, HTML)
- Validate artifact quality
- Store final artifact to Postgres

**Interfaces:**
- Input: `generate_artifact(context, comments, format)`
- Output: Artifact text (string)
- Dependencies: Context Manager (final context), Postgres (storage)

**FRs Addressed:** F6.4 (Export command)

**Implementation Notes:**
- Template-based generation (Jinja2 templates for markdown/HTML)
- JSON: structured representation of comments + metadata
- Markdown: human-readable with sections
- HTML: styled with CSS

**Key Methods:**
```python
class ArtifactManager:
    async def generate_artifact(context: str, comments: list[Comment], format: str) -> str
    async def apply_template(context: str, template_name: str) -> str
    async def validate_artifact(artifact: str) -> bool
```

---

### 5.3 Agent Layer

#### Component: Participant Agents (Domain Experts)
**Purpose:** Provide domain-specific responses and critiques

**Responsibilities:**
- Generate full responses to context (initial round)
- Generate comment-responses to peer comments (greedy strategy)
- Recall past meeting outcomes via Letta memory
- Track collaboration history

**Interfaces:**
- Input: `full_response(context)`, `comment_response(peer_comments)`
- Output: Response text (string)
- Dependencies: LLM API (Claude/DeepSeek), Letta (memory), Postgres (agent_memory)

**FRs Addressed:** F2.1 (Agent Orchestration), F7 (Agent Pool)

**Implementation Notes:**
- Agno Skill wrapper around Letta agents
- Memory retrieval: similarity search on past meetings (threshold: 0.7)
- Dynamic model switching: start DeepSeek, promote to Opus based on engagement

**Key Methods:**
```python
class DomainExpertAgent(Agent):
    async def full_response(context: str) -> str
    async def comment_response(peer_comments: list[Comment]) -> str
    async def recall_similar_meeting(topic: str) -> Optional[dict]
```

---

#### Component: Notetaker Agent
**Purpose:** Extract structured comments from agent responses

**Responsibilities:**
- Parse response text into atomic comments
- Categorize comments (technical_decision, risk, implementation_detail, etc.)
- Calculate novelty score per comment
- Store extracted comments to Postgres

**Interfaces:**
- Input: `extract_comments(response_text, agent_name)`
- Output: List of Comment objects (Pydantic)
- Dependencies: LLM API (Claude Sonnet for structured extraction), Postgres (storage)

**FRs Addressed:** F3.1, F3.2 (Comment Extraction)

**Implementation Notes:**
- Uses Claude Sonnet with structured output (Pydantic model)
- Extraction prompt: "Extract key ideas, technical decisions, risks, and implementation details"
- Novelty score: embedding similarity with existing comments (1 - max_similarity)

**Key Methods:**
```python
class NotetakerAgent(Agent):
    async def extract_comments(response: str, agent_name: str) -> list[Comment]
    async def categorize_comment(text: str) -> str
    async def calculate_novelty(comment: str, existing: list[Comment]) -> float
```

---

#### Component: Compressor Agent
**Purpose:** Apply three-tier compression to reduce context size

**Responsibilities:**
- Tier 1: Embedding-based clustering (cosine similarity > 0.85)
- Tier 2: LLM semantic merge of clusters
- Tier 3: Outlier removal (support < 2 agents)
- Track compression metrics (original count, final count, ratio)

**Interfaces:**
- Input: `compress(comments)`
- Output: Compressed comments (list)
- Dependencies: Qdrant (embeddings), LLM API (Claude Sonnet for merge), Postgres (metrics)

**FRs Addressed:** F3.3, F3.4 (Compression)

**Implementation Notes:**
- Batch embedding processing (all comments at once)
- Clustering: greedy algorithm with used-set tracking
- Merge prompt: "Combine these similar ideas into one coherent comment"
- Support counting: how many agents mentioned this idea

**Key Methods:**
```python
class CompressorAgent(Agent):
    async def compress(comments: list[Comment]) -> list[Comment]
    async def cluster_similar_comments(comments: list[Comment]) -> list[list[Comment]]
    async def merge_cluster(cluster: list[Comment]) -> Comment
    def has_support(comment: Comment, all_comments: list[Comment], min_support: int = 2) -> bool
```

---

### 5.4 Data Layer

#### Component: PostgreSQL Database
**Purpose:** Persistent storage for all meeting data

**Responsibilities:**
- Store meeting definitions, agent pool, responses, comments
- Enforce referential integrity (foreign keys)
- Provide transaction support (ACID)
- Archive full context history
- Store agent memory (JSONB)

**Interfaces:**
- Input: SQLAlchemy ORM operations (INSERT, SELECT, UPDATE)
- Output: Query results
- Dependencies: None (base layer)

**FRs Addressed:** All storage requirements

**Schema Highlights:**
- 9 tables (meetings, agents, meeting_agents, responses, comments, convergence_metrics, agent_memory, agent_performance, migrations)
- Indexes: meeting_id + round, agent_id, compressed flag
- JSONB for flexible schemas (agent_memory, letta_definition)

---

#### Component: Redis Cache
**Purpose:** High-speed session state storage

**Responsibilities:**
- Cache active meeting state (current_round, active_context)
- Cache round comments (TTL: 24 hours)
- Cache agent performance metrics (TTL: 7 days)
- Auto-expire completed meetings (TTL: 7 days)

**Interfaces:**
- Input: SET/GET/DEL operations
- Output: Cached data (JSON strings)
- Dependencies: None (base layer)

**FRs Addressed:** F4.2 (Active context caching)

**Key Patterns:**
- `meeting:{id}:state`: Meeting state dict
- `meeting:{id}:round:{r}:comments`: Comment list
- `agent:{name}:performance`: Metrics dict

---

#### Component: Qdrant Vector DB
**Purpose:** Store and search comment embeddings

**Responsibilities:**
- Store comment embeddings (768-dimensional vectors)
- Cosine similarity search for clustering
- Batch embedding operations

**Interfaces:**
- Input: Embed texts, similarity search
- Output: Embeddings, similar vectors
- Dependencies: None (base layer)

**FRs Addressed:** F3.3 (Compression clustering)

**Collection:**
- `comments`: Collection with 768-dim vectors (sentence-transformers)
- Metadata: comment_id, meeting_id, round, agent

---

#### Component: RabbitMQ Event Bus
**Purpose:** Async event communication

**Responsibilities:**
- Publish events (meeting.*, agent.*)
- Route events to consumers
- Ensure reliable delivery (ack/nack)

**Interfaces:**
- Input: PUBLISH messages to exchange
- Output: Delivered messages to consumers
- Dependencies: None (base layer)

**FRs Addressed:** F8 (Event-Driven Human-in-Loop)

**Exchange/Queue:**
- Exchange: `theboard.events` (topic exchange)
- Queue: `theboard.cli.events` (binds to meeting.*)
- Routing: `meeting.{event_type}`

---

## 6. Data Architecture

### 6.1 Data Model

**Entity Relationship Diagram (Text):**

```
meetings (1) ──< meeting_agents >── (N) agents
   │
   ├──< (1:N) responses
   │      │
   │      └──< (1:N) comments
   │
   ├──< (1:N) convergence_metrics
   │
   └──< (N:1) [final_artifact stored in meetings table]

agents (1) ──< (1:N) agent_memory
   │
   └──< (1:N) agent_performance
```

**Core Entities:**

1. **Meeting:** Brainstorming session definition
   - Attributes: id (UUID), topic, artifact_type, strategy, max_rounds, convergence_threshold, status
   - Relationships: Has many agents (via meeting_agents), has many responses, has many convergence_metrics

2. **Agent:** Domain expert from pool
   - Attributes: id (UUID), name, expertise (array), persona, model, letta_definition (JSONB)
   - Relationships: Belongs to many meetings (via meeting_agents), has many responses, has agent_memory

3. **Response:** Agent's full output (pre-extraction)
   - Attributes: id (UUID), meeting_id, round, agent_id, response_type, response_text, token_count
   - Relationships: Belongs to meeting, belongs to agent, has many comments

4. **Comment:** Extracted atomic idea
   - Attributes: id (UUID), response_id, meeting_id, round, agent_id, text, category, novelty_score, compressed, merged_into
   - Relationships: Belongs to response, belongs to meeting, belongs to agent, may reference another comment (merged_into)

5. **ConvergenceMetric:** Per-round convergence tracking
   - Attributes: meeting_id, round (composite PK), novelty_score, comment_count, compression_ratio
   - Relationships: Belongs to meeting

6. **AgentMemory:** Letta memory persistence
   - Attributes: agent_id, memory_type, memory_key (composite PK), memory_value (JSONB)
   - Relationships: Belongs to agent

7. **AgentPerformance:** Per-meeting agent metrics
   - Attributes: meeting_id, agent_id (composite PK), comments_generated, peer_references, novelty_avg, engagement_score, total_tokens, cost
   - Relationships: Belongs to meeting, belongs to agent

---

### 6.2 Database Design

**Normalization Level:** 3NF (Third Normal Form)

**Rationale:**
- Eliminates redundancy (responses stored once, referenced by comments)
- Maintains referential integrity (foreign keys)
- Flexible via JSONB where needed (agent_memory, letta_definition)

**Indexing Strategy:**

1. **Primary Keys:** All tables use UUID primary keys (except composite PKs)
2. **Foreign Key Indexes:** Automatic indexes on all FK columns
3. **Composite Indexes:**
   - `idx_meeting_round` on `responses(meeting_id, round)` - Fast round queries
   - `idx_meeting_round_compressed` on `comments(meeting_id, round, compressed)` - Compression filtering
4. **GIN Indexes:**
   - `agent_memory.memory_value` (JSONB) - Fast JSON queries

**Partitioning:**
- None required for v1 (scale: 100s of meetings)
- Future: Partition responses/comments by meeting_id if >10K meetings

---

### 6.3 Data Flow

**Write Path:**

1. **Meeting Creation:**
   - Postgres: INSERT into meetings (id, topic, strategy, ...)
   - Postgres: INSERT into meeting_agents (meeting_id, agent_id, ...)
   - Redis: SET meeting:{id}:state (initial state)

2. **Agent Response:**
   - Postgres: INSERT into responses (meeting_id, round, agent_id, response_text, ...)
   - Notetaker: extract_comments(response)
   - Postgres: INSERT into comments (response_id, meeting_id, agent_id, text, ...)

3. **Compression:**
   - Qdrant: embed_texts(comments) → embeddings
   - Qdrant: similarity_search() → clusters
   - Compressor: merge_cluster() → merged_comment
   - Postgres: UPDATE comments SET compressed=true, merged_into={id}
   - Redis: SET meeting:{id}:compression:{round} (metrics)

4. **Context Update:**
   - Redis: UPDATE meeting:{id}:state (active_context += compressed_comments)

5. **Convergence Metric:**
   - Postgres: INSERT into convergence_metrics (meeting_id, round, novelty_score, ...)

**Read Path:**

1. **Meeting State Retrieval:**
   - Redis: GET meeting:{id}:state
   - If miss: Postgres: SELECT FROM meetings WHERE id={id}

2. **Context Retrieval:**
   - Redis: GET meeting:{id}:state → active_context
   - If miss: Postgres: SELECT final_artifact FROM meetings WHERE id={id}

3. **Agent Performance:**
   - Redis: GET agent:{name}:performance
   - If miss: Postgres: SELECT FROM agent_performance WHERE meeting_id={id} AND agent_id={id}

**Caching Strategy:**

- **Hot Data (Redis):** Active meeting state, current round comments, agent performance
- **Cold Data (Postgres):** Historical responses, archived meetings, agent memory
- **Write-Through:** Updates go to both Redis and Postgres (Redis for speed, Postgres for persistence)
- **TTL:** Redis keys expire after 7 days (completed meetings), 24 hours (round comments)

---

## 7. API Design

### 7.1 Internal Python APIs

TheBoard does not expose REST endpoints (CLI-only application). All APIs are internal Python module interfaces.

**API Philosophy:**
- Type-safe via Pydantic models
- Async/await throughout
- Clear separation: CLI → Orchestration → Agents → Data
- Dependency injection for testability

---

### 7.2 Core API Contracts

#### Meeting Coordinator API
```python
from agno import Workflow
from typing import Literal

class TheboardMeeting(Workflow):
    """Agno Workflow for meeting orchestration"""

    topic: str
    agents: list[DomainExpertAgent]
    strategy: Literal["sequential", "greedy"]
    max_rounds: int = 5
    convergence_threshold: float = 0.2

    async def run() -> str:
        """
        Execute full meeting lifecycle.

        Returns:
            Final artifact as string

        Raises:
            MeetingExecutionError: If LLM API fails after retries
            ConvergenceError: If convergence never reached and max_rounds hit
        """

    async def sequential_round(context: str, round_num: int) -> list[Comment]:
        """
        Execute sequential strategy round.

        Args:
            context: Cumulative context from previous rounds
            round_num: Current round number (1-indexed)

        Returns:
            List of comments extracted this round
        """

    async def greedy_round(context: str, round_num: int) -> list[Comment]:
        """
        Execute greedy strategy round (parallel + comment-responses).

        Returns:
            List of all comments (full responses + comment-responses)
        """

    async def check_convergence(
        current: list[Comment],
        all_comments: list[Comment]
    ) -> bool:
        """
        Calculate novelty score and check convergence.

        Returns:
            True if converged (novelty < threshold for 2 consecutive rounds)
        """
```

---

#### Agent Registry API
```python
from pathlib import Path

class AgentRegistry:
    """Singleton registry for agent pool management"""

    async def load_pool(pool_dir: Path) -> list[Agent]:
        """
        Load agents from plaintext or Letta format.

        Args:
            pool_dir: Directory containing agent files

        Returns:
            List of loaded agents

        Side Effects:
            - Inserts agents into Postgres (if not exists)
            - Indexes expertise embeddings in Qdrant
        """

    async def auto_select_team(
        topic: str,
        artifact_type: str,
        count: int = 5
    ) -> list[Agent]:
        """
        Auto-select agents based on topic similarity.

        Args:
            topic: Meeting topic
            artifact_type: Desired artifact type
            count: Number of agents to select

        Returns:
            List of selected agents (ordered by relevance)

        Algorithm:
            1. Embed topic
            2. Similarity search in Qdrant (expertise embeddings)
            3. Filter by artifact_type requirements
            4. Ensure diversity (different domains)
            5. Return top N
        """

    async def track_performance(
        meeting_id: str,
        agent_name: str,
        metrics: dict
    ):
        """
        Track agent performance metrics.

        Args:
            meeting_id: Meeting UUID
            agent_name: Agent name
            metrics: Dict with keys: comments_generated, peer_references,
                     novelty_avg, total_tokens, cost

        Side Effects:
            - Updates agent_performance table in Postgres
            - Caches to Redis (agent:{name}:performance)
        """
```

---

#### Context Manager API
```python
class ContextManager:
    """Manage cumulative context evolution"""

    async def append_to_context(
        context: str,
        comments: list[Comment]
    ) -> str:
        """
        Append compressed comments to cumulative context.

        Args:
            context: Existing context (Xr-1)
            comments: New comments to append (σ(C(Tir)))

        Returns:
            Updated context (Xr)

        Side Effects:
            - Updates Redis (meeting:{id}:state)
            - Archives Xr-1 to Postgres if round complete
        """

    async def get_delta(
        agent_name: str,
        since_round: int
    ) -> list[Comment]:
        """
        Get comments added since agent's last turn (delta propagation).

        Args:
            agent_name: Agent requesting delta
            since_round: Last round agent participated

        Returns:
            List of new comments (rounds > since_round)
        """

    def get_context_size() -> int:
        """
        Estimate token count (chars / 4).

        Returns:
            Estimated token count

        Raises:
            ContextSizeExceededError: If context > 20K tokens
        """
```

---

#### Compressor Agent API
```python
class CompressorAgent(Agent):
    """Three-tier compression agent"""

    similarity_threshold: float = 0.85

    async def compress(comments: list[Comment]) -> list[Comment]:
        """
        Apply three-tier compression.

        Algorithm:
            Tier 1: Embedding clustering (similarity > 0.85)
            Tier 2: LLM semantic merge
            Tier 3: Outlier removal (support < 2)

        Returns:
            Compressed comments (40-60% of original count)

        Side Effects:
            - Updates comments table (compressed=true, merged_into={id})
            - Stores compression metrics to Redis
        """

    async def cluster_similar_comments(
        comments: list[Comment]
    ) -> list[list[Comment]]:
        """
        Cluster comments using Qdrant embeddings.

        Returns:
            List of clusters (each cluster is a list of similar comments)
        """

    async def merge_cluster(cluster: list[Comment]) -> Comment:
        """
        Merge cluster into single comment using LLM.

        Args:
            cluster: List of similar comments

        Returns:
            Merged comment (combines ideas from cluster)
        """
```

---

### 7.3 Data Models (Pydantic)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

class Comment(BaseModel):
    """Extracted atomic idea from agent response"""
    id: UUID
    response_id: UUID
    meeting_id: UUID
    round: int
    agent_name: str
    text: str
    category: str  # technical_decision, risk, implementation_detail, etc.
    novelty_score: float = Field(ge=0.0, le=1.0)
    compressed: bool = False
    merged_into: UUID | None = None
    created_at: datetime

class Meeting(BaseModel):
    """Meeting definition"""
    id: UUID
    topic: str
    artifact_type: str
    strategy: Literal["sequential", "greedy"]
    max_rounds: int = Field(ge=1, le=20)
    convergence_threshold: float = Field(ge=0.0, le=1.0)
    status: Literal["active", "paused", "completed"]
    created_at: datetime
    completed_at: datetime | None = None
    final_artifact: str | None = None

class Agent(BaseModel):
    """Domain expert agent"""
    id: UUID
    name: str
    expertise: list[str]
    persona: str
    model: str = "deepseek-r3"
    letta_definition: dict | None = None
```

---

## 8. NFR Coverage

### 8.1 Performance (NFR 7.1)

#### NFR-PERF-001: Meeting Execution Latency
**Requirement:** Sequential strategy <30s per round for 5 agents, total meeting <5 minutes

**Architecture Solution:**
- Async/await throughout (Python asyncio)
- Parallel LLM calls for independent operations (greedy strategy)
- Redis caching for meeting state (<100ms retrieval)
- Postgres connection pooling (max 10 connections)
- Lazy compression (only when context > 10K chars)

**Implementation Notes:**
- Use asyncio.gather() for parallel agent responses in greedy strategy
- Connection pool configured in SQLAlchemy: `pool_size=10, max_overflow=20`
- Redis pipelining for batch operations

**Validation:**
- Load testing: 5-agent, 5-round meeting under 5 minutes
- Per-round latency tracking in convergence_metrics table

---

#### NFR-PERF-002: Compression Latency
**Requirement:** <5s for 50 comments, <10s for 100 comments

**Architecture Solution:**
- Batch embedding processing (all comments at once)
- Qdrant optimized for vector similarity search
- LLM semantic merge uses fast Sonnet model
- Parallel cluster merging (asyncio.gather)

**Implementation Notes:**
- Batch size: 100 comments per embedding call
- Qdrant query: top_k=50 for clustering
- Merge prompt optimized for speed (concise output)

**Validation:**
- Benchmark compression on 50, 100, 200 comment datasets
- Track compression_time metric per round

---

#### NFR-PERF-003: Convergence Calculation
**Requirement:** <2s per round for novelty score computation

**Architecture Solution:**
- Pre-computed embeddings (stored in Qdrant during comment extraction)
- Cosine similarity batch computation
- In-memory overlap counting

**Implementation Notes:**
- Embeddings computed once during extraction, reused for convergence
- Similarity threshold: 0.85 (pre-configured, no tuning per round)

**Validation:**
- Benchmark novelty calculation on 50, 100 comment datasets

---

#### NFR-PERF-004: CLI Responsiveness
**Requirement:** Status checks <500ms, live progress updates <200ms per update

**Architecture Solution:**
- Redis for sub-millisecond meeting state retrieval
- Rich Live() for non-blocking progress updates
- Background event consumer (no blocking on CLI thread)

**Implementation Notes:**
- Status command: single Redis GET (meeting:{id}:state)
- Live updates: async event consumption with buffering

**Validation:**
- Measure CLI response time with profiler
- Status checks: p95 <500ms

---

#### NFR-PERF-005: Database Performance
**Requirement:** Meeting state retrieval <100ms, comment insertion <50ms per batch

**Architecture Solution:**
- Redis primary cache (sub-millisecond)
- Postgres indexes on hot columns (meeting_id, round, agent_id)
- Batch inserts for comments (10-50 per batch)

**Implementation Notes:**
- Batch insert: `INSERT INTO comments VALUES (...), (...), (...)`
- Connection pooling prevents connection overhead

**Validation:**
- Measure query latency with pg_stat_statements
- Batch insert: p95 <50ms for 50 comments

---

#### NFR-PERF-006: Scalability
**Requirement:** Support 10 concurrent agents, 100+ past meetings in memory, <20K tokens context

**Architecture Solution:**
- Async execution handles 10 concurrent LLM calls without blocking
- Qdrant similarity search for memory recall (fast even at 100+ meetings)
- Three-tier compression keeps context <15K tokens

**Implementation Notes:**
- Memory recall: similarity threshold 0.7 (only retrieve relevant meetings)
- Context alert at 15K tokens, hard limit at 20K tokens (truncate with warning)

**Validation:**
- Test with 10 agents in greedy strategy (parallel)
- Test memory recall with 100+ meeting history

---

### 8.2 Security (NFR 7.2)

#### NFR-SEC-001: Credential Management
**Requirement:** API keys via environment variables, no hardcoded secrets

**Architecture Solution:**
- Environment variables for all secrets (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, DB_PASSWORD, etc.)
- .env file for local development (not committed to git)
- Docker secrets for production deployment

**Implementation Notes:**
- Load env vars using python-dotenv
- .gitignore includes .env
- CI/CD pipeline validates no hardcoded secrets (ruff rule: no string literals matching API key patterns)

**Validation:**
- Pre-commit hook scans for hardcoded secrets
- CI fails on secret detection

---

#### NFR-SEC-002: Network Security
**Requirement:** TLS 1.2+ for LLM API calls, RabbitMQ auth, Redis AUTH, Postgres SSL

**Architecture Solution:**
- All LLM API clients enforce TLS (httpx with verify=True)
- RabbitMQ: username/password auth (from env vars)
- Redis: AUTH password (from env vars)
- Postgres: SSL mode=require (production)

**Implementation Notes:**
- LLM clients: anthropic SDK, DeepSeek SDK (both enforce TLS)
- RabbitMQ: aiormq with credentials
- Redis: redis-py with password
- Postgres: SQLAlchemy with sslmode=require

**Validation:**
- Verify TLS with network inspection (Wireshark)
- Test auth failures with wrong credentials

---

#### NFR-SEC-003: Data Privacy
**Requirement:** No PII in agent responses, single-user system, audit trail

**Architecture Solution:**
- User responsible for input sanitization (documented in user guide)
- Single-user system (no multi-tenancy in v1)
- Full audit trail: all responses stored pre-compression

**Implementation Notes:**
- Warning in CLI: "Ensure input does not contain PII"
- Audit trail: responses table never deleted

**Validation:**
- Document PII handling in user guide
- Test audit trail completeness

---

#### NFR-SEC-004: Input Validation
**Requirement:** All user inputs validated via Pydantic, SQL injection prevention, CLI injection prevention

**Architecture Solution:**
- Pydantic models for all CLI inputs (topic, agent_name, meeting_id)
- SQLAlchemy parameterized queries (no string concatenation)
- Typer argument parsing (no shell execution of user input)

**Implementation Notes:**
- Pydantic Field() validators for string length, format
- SQLAlchemy: `session.execute(text("SELECT ... WHERE id = :id"), {"id": meeting_id})`
- No subprocess calls with user input

**Validation:**
- Test SQL injection attempts (should be blocked)
- Test CLI injection attempts (should be blocked)

---

### 8.3 Reliability (NFR 7.3)

#### NFR-REL-001: Fault Tolerance
**Requirement:** State persisted to survive crashes, graceful degradation on LLM API failure

**Architecture Solution:**
- Redis + Postgres dual persistence (Redis for speed, Postgres for durability)
- Idempotent event handlers (RabbitMQ ack only after successful processing)
- LLM API retry with exponential backoff (max 3 retries)

**Implementation Notes:**
- State write: both Redis SET and Postgres UPDATE (within transaction)
- Event handler: process → ack (not ack → process)
- Retry: tenacity library with exponential backoff (1s, 2s, 4s)

**Validation:**
- Kill process mid-meeting, verify resume from Redis state
- Simulate LLM API failures, verify retries

---

#### NFR-REL-002: Data Integrity
**Requirement:** Transaction boundaries, atomic updates, foreign key constraints

**Architecture Solution:**
- Postgres transactions for multi-row writes
- Redis atomic operations (SET, GET, DEL)
- Foreign key constraints enforced in schema

**Implementation Notes:**
- SQLAlchemy: `with session.begin(): ...` for transactions
- Redis: MULTI/EXEC for atomic multi-key operations
- Foreign keys: ON DELETE CASCADE for meeting data cleanup

**Validation:**
- Test partial failure rollback
- Test foreign key constraint violations

---

#### NFR-REL-003: Recovery
**Requirement:** Meeting resume after interruption, rollback on failure, event replay

**Architecture Solution:**
- Meeting state in Redis includes current_round, current_agent (resume point)
- Database transactions for automatic rollback
- RabbitMQ: unacknowledged messages redelivered (event replay)

**Implementation Notes:**
- Resume: load state from Redis, continue from current_round + 1
- Rollback: SQLAlchemy transaction.rollback() on exception
- Event replay: RabbitMQ consumer crashes → messages redelivered

**Validation:**
- Pause and resume meeting mid-round
- Crash during database write, verify rollback

---

#### NFR-REL-004: Availability
**Requirement:** 99% uptime for local development, graceful shutdown, health checks

**Architecture Solution:**
- Docker Compose healthchecks for all services
- Graceful shutdown: SIGTERM handler saves state before exit
- Service dependencies: depends_on with condition: service_healthy

**Implementation Notes:**
- Healthchecks: Postgres (pg_isready), Redis (redis-cli ping), RabbitMQ (rabbitmqctl status)
- SIGTERM handler: save active context to Postgres, close connections
- depends_on: ensures services start in order

**Validation:**
- Monitor uptime over 7 days
- Test graceful shutdown (SIGTERM)

---

### 8.4 Observability (NFR 7.4)

#### NFR-OBS-001: Structured Logging
**Requirement:** JSON logging, correlation IDs, DEBUG/INFO/ERROR levels, log rotation

**Architecture Solution:**
- Python logging with structlog for JSON formatting
- Correlation IDs: meeting_id propagated through all log entries
- Log rotation: 100MB per file, 10 files retained

**Implementation Notes:**
- Structured logging: `logger.info("round_completed", meeting_id=..., round=..., comments=...)`
- Correlation ID middleware: adds meeting_id to all log entries within meeting execution
- Log rotation: logging.handlers.RotatingFileHandler

**Validation:**
- Verify JSON format with log parser
- Trace single meeting via correlation ID

---

#### NFR-OBS-002: Metrics
**Requirement:** Track token usage, costs, latency, convergence rounds

**Architecture Solution:**
- Per-agent, per-round token tracking (stored in agent_performance table)
- Cost calculation: tokens × model_price
- Latency metrics: round_duration, compression_time, convergence_calc_time
- Convergence rounds: count of rounds until convergence (in convergence_metrics)

**Implementation Notes:**
- Token tracking: LLM API response includes token_count
- Cost: stored as DECIMAL(10, 4) in Postgres
- Latency: recorded with time.perf_counter()

**Validation:**
- Cost dashboard: sum costs per meeting, per agent
- Latency histogram: p50, p95, p99

---

#### NFR-OBS-003: Monitoring
**Requirement:** CLI progress display, event stream for external monitoring, service health endpoints

**Architecture Solution:**
- CLI: Rich Live() for live progress (round, agent, comment count, novelty)
- Event stream: RabbitMQ events consumable by external tools (DataDog, Prometheus)
- Health endpoints: Docker Compose healthchecks

**Implementation Notes:**
- CLI progress: subscribe to meeting.round.completed events
- External monitoring: consume RabbitMQ events, parse JSON payloads
- Health checks: /healthz endpoints (future REST API)

**Validation:**
- Verify CLI updates in real-time
- Consume events with external tool (e.g., curl RabbitMQ API)

---

#### NFR-OBS-004: Debugging
**Requirement:** Full audit trail, debug mode with verbose logging, meeting replay

**Architecture Solution:**
- Audit trail: responses table stores all agent responses pre-compression
- Debug mode: LOG_LEVEL=DEBUG environment variable
- Replay: re-execute meeting from stored state (round-by-round)

**Implementation Notes:**
- Audit trail: never delete responses (even after compression)
- Debug mode: structlog level filter
- Replay: load state from round N, execute subsequent rounds

**Validation:**
- Query audit trail for full meeting history
- Enable debug mode, verify verbose logs
- Replay meeting from mid-point

---

### 8.5 Maintainability (NFR 7.5)

#### NFR-MAINT-001: Code Quality
**Requirement:** Type hints enforced via mypy --strict, Pydantic models, docstrings (Google style)

**Architecture Solution:**
- mypy --strict in CI pipeline (fails on type errors)
- Pydantic models for all data structures (Comment, Meeting, Agent)
- Docstrings for all public APIs (Google style)

**Implementation Notes:**
- mypy config: `strict = true, warn_return_any = true`
- Pydantic: strict=True mode
- Docstring format: Google (Args, Returns, Raises)

**Validation:**
- CI: mypy --strict passes
- Test: Pydantic validation errors caught

---

#### NFR-MAINT-002: Code Style
**Requirement:** ruff for formatting, max line length 100, conventional commits

**Architecture Solution:**
- ruff for linting + formatting (replaces black, isort, flake8)
- Line length: 100 characters
- Commit messages: conventional commits (feat:, fix:, docs:, etc.)

**Implementation Notes:**
- ruff config: `line-length = 100, select = ["E", "F", "I"]`
- Pre-commit hook: ruff format, ruff check
- Commit template: .gitmessage with conventional format

**Validation:**
- CI: ruff check passes
- Pre-commit: ruff format auto-fixes

---

#### NFR-MAINT-003: Dependency Management
**Requirement:** uv for package management, pinned versions in production

**Architecture Solution:**
- uv for fast dependency resolution (replaces pip)
- pyproject.toml with pinned versions (==1.2.3)
- Automated security updates (dependabot)

**Implementation Notes:**
- uv.lock file: committed to git
- pyproject.toml: dependencies with exact versions in production
- dependabot config: check weekly

**Validation:**
- uv lock generates reproducible environment
- Security updates applied within 1 week

---

## 9. Security Architecture

### 9.1 Authentication & Authorization

**Current State (v1):**
- Single-user CLI application (no authentication required)
- No authorization model (user has full access)

**Future State (v2+):**
- Multi-user: OAuth 2.0 via Anthropic Workspaces
- RBAC: meeting_owner, meeting_viewer, agent_admin roles

---

### 9.2 Data Encryption

**At Rest:**
- Postgres: pgcrypto for sensitive columns (future)
- Redis: Not encrypted (ephemeral data with TTL)
- Agent memory: JSONB not encrypted (contains domain knowledge, not PII)

**In Transit:**
- LLM APIs: TLS 1.2+ (enforced by SDK)
- RabbitMQ: TLS optional (not required for local dev)
- Postgres: SSL mode=require (production)
- Redis: No TLS (local network only)

---

### 9.3 Security Best Practices

**Input Validation:**
- Pydantic models validate all CLI inputs (type, length, format)
- SQLAlchemy parameterized queries prevent SQL injection
- No subprocess calls with user input (prevent CLI injection)

**Output Sanitization:**
- LLM responses not sanitized (trusted domain: agent output)
- User input (topic) included in prompts: no sanitization needed (LLM handles safely)

**Rate Limiting:**
- LLM API: handled by provider (10 RPM free tier, 100 RPM paid)
- No rate limiting on CLI commands (single-user)

**Security Headers:**
- Not applicable (CLI application, no HTTP)

---

## 10. Scalability & Performance

### 10.1 Scaling Strategy

**Vertical Scaling:**
- Increase RAM for Redis (1GB → 4GB for larger contexts)
- Increase Postgres storage (10GB → 100GB for more meetings)
- Increase CPU cores for Qdrant (embedding search)

**Horizontal Scaling (Future):**
- Multiple Meeting Coordinator instances (partition by meeting_id)
- Qdrant cluster (sharded by meeting_id)
- Postgres read replicas (for historical data queries)

**Auto-Scaling:**
- Not applicable for v1 (single-user CLI)
- Future: containerized deployment with auto-scaling based on active meetings

---

### 10.2 Performance Optimization

**Query Optimization:**
- Indexes on hot columns (meeting_id, round, agent_id)
- Batch inserts for comments (10-50 per batch)
- Connection pooling (Postgres, Redis)

**N+1 Query Prevention:**
- SQLAlchemy joinedload for meeting_agents eager loading
- Batch queries for agent performance (single SELECT with IN clause)

**Lazy Loading:**
- Agent memory: loaded on-demand (only when relevant)
- Context history: retrieved from Postgres only on export

**Compression:**
- Three-tier compression reduces context size by 40-60%
- Lazy compression: only when context > 10K chars

---

### 10.3 Caching Strategy

**Cache Hierarchy:**

1. **L1 (Application Memory):**
   - Agent Registry: in-memory cache of loaded agents
   - Embeddings: cached during round execution

2. **L2 (Redis):**
   - Meeting state: `meeting:{id}:state` (TTL: 7 days)
   - Round comments: `meeting:{id}:round:{r}:comments` (TTL: 24h)
   - Agent performance: `agent:{name}:performance` (TTL: 7 days)

3. **L3 (Postgres):**
   - Full audit trail (no TTL)
   - Agent memory (persistent)

**Cache Invalidation:**
- **Write-Through:** Updates go to Redis + Postgres simultaneously
- **TTL-Based:** Redis keys expire automatically
- **Manual Invalidation:** On meeting completion, update final_artifact in Postgres, keep Redis cache

**Cache Hit Ratio Target:** >90% for meeting state retrieval (Redis)

---

### 10.4 Load Balancing

**Not Applicable for v1:**
- Single CLI process (no load balancer needed)

**Future (Multi-User):**
- Load balancer distributes requests across Meeting Coordinator instances
- Sticky sessions by meeting_id (same meeting always routes to same instance)
- Algorithm: Least connections (balance by active meetings)

---

## 11. Reliability & Availability

### 11.1 High Availability

**Current State (v1):**
- Single-node deployment (local dev)
- No HA required (single-user CLI)

**Future State (Production):**
- Multi-AZ deployment for Postgres, Redis, RabbitMQ
- Redundant Meeting Coordinator instances
- Health checks + automatic failover

---

### 11.2 Disaster Recovery

**RPO (Recovery Point Objective):** 5 minutes
- Redis snapshots every 5 minutes
- Postgres continuous archiving (WAL)

**RTO (Recovery Time Objective):** 15 minutes
- Restore Redis from snapshot (<5 min)
- Restore Postgres from backup (<10 min)

**Backup Frequency:**
- Postgres: Daily full backup, continuous WAL archiving
- Redis: Snapshot every 5 minutes (RDB)
- Agent pool: Committed to git (no backup needed)

**Restore Procedures:**
1. Restore Postgres from latest backup
2. Restore Redis from latest snapshot
3. Restart services
4. Verify health checks

---

### 11.3 Monitoring & Alerting

**Metrics to Track:**

1. **System Metrics:**
   - CPU usage (Postgres, Redis, Qdrant)
   - Memory usage (Redis cache size, Postgres shared_buffers)
   - Disk usage (Postgres, Qdrant)

2. **Application Metrics:**
   - Meeting execution time (p50, p95, p99)
   - Token usage per meeting (sum, avg)
   - Cost per meeting (sum, avg)
   - Convergence rounds (histogram)
   - Compression ratio (avg, min, max)

3. **Error Metrics:**
   - LLM API failures (count, rate)
   - Database errors (count, rate)
   - Event delivery failures (count, rate)

**Alerting Thresholds:**

| Metric | Warning | Critical |
|--------|---------|----------|
| Meeting execution time | >7 min | >10 min |
| LLM API failure rate | >5% | >10% |
| Database error rate | >1% | >5% |
| Redis memory usage | >80% | >95% |
| Postgres disk usage | >80% | >95% |

**Alerting Channels:**
- CLI: Warnings displayed inline
- Logs: ERROR level logged
- Future: Email, Slack notifications

---

## 12. Development & Deployment

### 12.1 Code Organization

**Project Structure:**
```
theboard/
├── src/theboard/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI commands
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── meeting_coordinator.py  # Agno Workflow
│   │   ├── agent_registry.py
│   │   ├── context_manager.py
│   │   └── artifact_manager.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── domain_expert.py   # Participant agents
│   │   ├── notetaker.py       # Notetaker agent
│   │   └── compressor.py      # Compressor agent
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py          # SQLAlchemy models
│   │   ├── redis_client.py
│   │   ├── qdrant_client.py
│   │   └── rabbitmq_client.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic models
│   │   ├── config.py          # Settings (env vars)
│   │   └── logging.py         # Structured logging setup
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py      # Embedding utilities
│       └── cost_calculator.py
├── tests/
│   ├── unit/
│   │   ├── test_compressor.py
│   │   ├── test_convergence.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_meeting_e2e.py
│   │   └── ...
│   └── fixtures/
│       ├── sample_agents.py
│       └── ...
├── docs/
│   ├── architecture-theboard-2025-12-19.md
│   ├── tech-spec-theboard-2025-12-19.md
│   └── PRD.md
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
└── README.md
```

**Module Boundaries:**
- `cli.py`: Entry point, delegates to orchestration
- `orchestration/`: Meeting logic, no LLM calls
- `agents/`: LLM interactions, no database access
- `data/`: Database/cache interactions, no business logic
- `core/`: Shared utilities (schemas, config, logging)

**Naming Conventions:**
- Classes: PascalCase (MeetingCoordinator)
- Functions: snake_case (extract_comments)
- Constants: UPPER_SNAKE_CASE (MAX_ROUNDS)
- Files: snake_case (meeting_coordinator.py)

---

### 12.2 Testing Strategy

**Unit Testing:**
- Coverage target: >70%
- Frameworks: pytest, pytest-asyncio
- Mocking: pytest-mock for LLM API calls, database operations
- Focus: Core logic (compression, convergence, context management)

**Integration Testing:**
- Scenarios: End-to-end meeting execution (3+ scenarios)
- Frameworks: pytest with Docker Compose fixtures
- Focus: CLI → Orchestration → Agents → Data flow

**E2E Testing:**
- Scenarios: Full meeting with real LLM API calls (1 scenario per sprint)
- Focus: Artifact quality, cost validation

**Performance Testing:**
- Tools: pytest-benchmark, locust (future)
- Scenarios: 5-agent, 5-round meeting under 5 minutes
- Load: 10 concurrent agents in greedy strategy

**Test Data:**
- Sample agents: fixtures/sample_agents.py (3 plaintext agents)
- Sample meetings: fixtures/sample_meetings.py (topic, expected comments)

---

### 12.3 CI/CD Pipeline

**Pipeline Stages:**

1. **Lint & Format:**
   - ruff check (linting)
   - ruff format --check (formatting)
   - mypy --strict (type checking)

2. **Test:**
   - pytest tests/unit (unit tests)
   - pytest tests/integration (integration tests with Docker Compose)

3. **Build:**
   - docker build -t theboard:latest .

4. **Deploy (Future):**
   - docker push theboard:latest
   - Deploy to staging → manual approval → production

**Automated Testing Gates:**
- Linting must pass (ruff)
- Type checking must pass (mypy --strict)
- Unit tests must pass (>70% coverage)
- Integration tests must pass

**Deployment Strategy:**
- **Local Dev:** docker-compose up (Postgres, Redis, RabbitMQ, Qdrant)
- **Staging (Future):** Docker Swarm or Kubernetes
- **Production (Future):** Kubernetes with auto-scaling

---

### 12.4 Environments

**Development:**
- Docker Compose for services (Postgres, Redis, RabbitMQ, Qdrant)
- .env file for secrets (not committed)
- LOG_LEVEL=DEBUG

**Staging (Future):**
- Mirrored production infrastructure
- Separate databases (no shared data with prod)
- LOG_LEVEL=INFO

**Production (Future):**
- Multi-AZ deployment
- Managed services (RDS, ElastiCache, Amazon MQ)
- LOG_LEVEL=INFO

**Environment Parity:**
- Same Docker images across all environments
- Different configs via environment variables
- Staging mirrors production (same resource limits)

---

## 13. Traceability & Trade-offs

### 13.1 FR Traceability

| FR ID | FR Name | Components | Implementation Notes |
|-------|---------|------------|----------------------|
| F1 | Meeting Configuration | CLI, Meeting Coordinator, Postgres, Redis | `board create` command, interactive wizard, state initialization |
| F2 | Agent Orchestration | Meeting Coordinator, Participant Agents | Sequential/greedy strategies, turn ordering, dynamic model switching |
| F3 | Comment Extraction & Compression | Notetaker, Compressor, Qdrant | Three-tier compression, embedding clustering, LLM merge |
| F4 | Context Management | Context Manager, Redis, Postgres | Cumulative context (Xr evolution), delta propagation, caching |
| F5 | Convergence Detection | Meeting Coordinator, Qdrant | Novelty calculation using embeddings, automatic stopping |
| F6 | CLI Interface | Typer CLI, Rich | Commands: create, run, status, export; live progress display |
| F7 | Agent Pool Management | Agent Registry, Qdrant, Postgres | Load plaintext/Letta agents, auto-selection via topic similarity |
| F8 | Event-Driven Human-in-Loop | RabbitMQ, CLI Event Consumer | Events emitted at decision points, async human steering |

---

### 13.2 NFR Traceability

| NFR ID | NFR Name | Architecture Solution | Validation Method |
|--------|----------|------------------------|-------------------|
| NFR-PERF-001 | Meeting execution latency <5 min | Async/await, Redis caching, lazy compression | Load testing 5-agent meetings |
| NFR-PERF-002 | Compression latency <5s for 50 comments | Batch embeddings, parallel merging | Benchmark compression |
| NFR-PERF-003 | Convergence calc <2s | Pre-computed embeddings, in-memory overlap | Benchmark novelty calculation |
| NFR-PERF-004 | CLI responsiveness <500ms | Redis for state retrieval, async updates | Profile CLI commands |
| NFR-PERF-005 | DB query <200ms | Indexes, connection pooling, batch inserts | pg_stat_statements |
| NFR-PERF-006 | Scalability (10 agents, 100+ meetings) | Async execution, vector search, compression | Test with 10 agents, 100+ history |
| NFR-SEC-001 | Credential management | Environment variables, Docker secrets | Pre-commit secret scan |
| NFR-SEC-002 | Network security | TLS 1.2+, RabbitMQ auth, Redis AUTH | Network inspection |
| NFR-SEC-003 | Data privacy | User input sanitization, audit trail | Document in user guide |
| NFR-SEC-004 | Input validation | Pydantic, SQLAlchemy parameterized queries | Test injection attempts |
| NFR-REL-001 | Fault tolerance | Redis + Postgres persistence, retry logic | Crash recovery testing |
| NFR-REL-002 | Data integrity | Transactions, foreign keys | Test rollback scenarios |
| NFR-REL-003 | Recovery | Resume from Redis state, event replay | Pause/resume testing |
| NFR-REL-004 | Availability (99% uptime) | Health checks, graceful shutdown | Monitor uptime 7 days |
| NFR-OBS-001 | Structured logging | JSON logging, correlation IDs | Trace via correlation ID |
| NFR-OBS-002 | Metrics tracking | Token/cost/latency tracking | Metrics dashboard |
| NFR-OBS-003 | Monitoring | CLI progress, RabbitMQ events, health checks | External event consumption |
| NFR-OBS-004 | Debugging | Audit trail, debug mode, replay | Query audit trail, replay meeting |
| NFR-MAINT-001 | Code quality | mypy --strict, Pydantic, docstrings | CI type checking |
| NFR-MAINT-002 | Code style | ruff, line length 100, conventional commits | CI linting |
| NFR-MAINT-003 | Dependency management | uv, pinned versions, dependabot | Reproducible builds |

---

### 13.3 Major Trade-offs

#### Trade-off 1: Agno Framework vs. Custom Orchestration
**Decision:** Use Agno for meeting orchestration

**Trade-off:**
- ✓ Gain: State management, async coordination, observability hooks, reduced boilerplate
- ✗ Lose: Early-stage framework risk, learning curve, potential limitations

**Rationale:** State management is complex (current_round, current_agent, turn_queue, context). Agno provides this out-of-the-box with async task composition. Benefits outweigh risks given early prototyping to validate fit.

**Mitigation:** Abstract orchestration layer (interface), fallback plan to direct async/await if Agno proves inadequate.

---

#### Trade-off 2: Hybrid Model Strategy vs. All-Opus
**Decision:** Hybrid (start cheap, promote top 20% to Opus)

**Trade-off:**
- ✓ Gain: 60-80% cost reduction, maintains quality via selective promotion
- ✗ Lose: Complexity (engagement tracking, model switching), potential quality loss if promotion logic flawed

**Rationale:** Token costs are a primary driver. All-Opus is $15 per 5-round meeting, hybrid is $3-5. Quality maintained by promoting high-engagement agents.

**Mitigation:** A/B testing to validate quality, configurable promotion threshold, fallback to all-Opus for critical meetings.

---

#### Trade-off 3: Three-Tier Compression vs. Simple LLM Summarization
**Decision:** Three-tier compression (embedding clustering → LLM merge → outlier removal)

**Trade-off:**
- ✓ Gain: 40-60% compression ratio, preserves quality via semantic merge
- ✗ Lose: Complexity (Qdrant embeddings, clustering logic), latency (5s for 50 comments)

**Rationale:** Simple LLM summarization loses information (no semantic clustering). Three-tier approach combines speed (embeddings) with quality (LLM merge).

**Mitigation:** Full audit trail enables quality validation, configurable compression thresholds, lazy compression (only when needed).

---

#### Trade-off 4: Redis Caching vs. Postgres-Only
**Decision:** Redis for active state, Postgres for persistence

**Trade-off:**
- ✓ Gain: Sub-millisecond retrieval for hot data, reduces Postgres load
- ✗ Lose: Dual writes (complexity), Redis volatility (mitigated by Postgres backup)

**Rationale:** Meeting state accessed frequently (every round, every status check). Redis provides <100ms retrieval vs. Postgres 200ms+.

**Mitigation:** Write-through caching (both Redis + Postgres), TTL for auto-expiration, Redis backup snapshots.

---

#### Trade-off 5: Event-Driven Human-in-Loop vs. Synchronous CLI Prompts
**Decision:** Event-driven via RabbitMQ

**Trade-off:**
- ✓ Gain: Async workflows (human can respond later), decouples execution from CLI
- ✗ Lose: Complexity (message broker), latency overhead (50-100ms per event)

**Rationale:** Synchronous prompts block meeting execution. Event-driven allows meeting to pause, human to review, then resume. Critical for 5-15 minute meetings.

**Mitigation:** Timeout defaults (auto-continue after 5min), local RabbitMQ (low latency), simple event schema.

---

#### Trade-off 6: Letta vs. Plaintext Agents
**Decision:** Support both (start plaintext, migrate to Letta)

**Trade-off:**
- ✓ Gain: Incremental migration path, backward compatibility, long-term memory via Letta
- ✗ Lose: Dual code paths (complexity), migration effort

**Rationale:** Plaintext agents already exist (/home/delorenj/code/DeLoDocs/AI/Agents/Generic). Letta provides memory persistence (previous_meetings). Hybrid support minimizes risk.

**Mitigation:** Migration script (plaintext → Letta), comprehensive testing, phase out plaintext in v2.

---

#### Trade-off 7: Sequential vs. Greedy Strategy
**Decision:** Support both, default to sequential

**Trade-off:**
- **Sequential:**
  - ✓ Gain: Simpler logic, lower token cost (n responses), cumulative context evolution
  - ✗ Lose: Slower (sequential execution), less parallelism
- **Greedy:**
  - ✓ Gain: Faster (parallel execution), more peer interaction (n² responses)
  - ✗ Lose: Higher token cost, complex orchestration

**Rationale:** Sequential for most use cases (simplicity, cost). Greedy for critical meetings requiring exhaustive critique.

**Mitigation:** User chooses strategy via `--strategy` flag, document trade-offs in user guide.

---

## 14. Future Enhancements

**Out of Scope for v1 (Deferred to Future Versions):**

1. **Vision/Multimodal Support:** Vision agents review screenshots, diagrams, wireframes
2. **Web UI:** Real-time collaborative editing, visualization of agent discussions
3. **Advanced Agent Features:** Cross-meeting meta-learning, agent specialization via fine-tuning
4. **Integration:** Git repo integration (commit artifacts), project management tools (Jira, Linear)
5. **Enterprise Features:** Multi-user concurrent access, RBAC, team workspaces, SSO

---

## 15. Appendix

### 15.1 Glossary

- **Agent:** AI entity with domain expertise, powered by LLM
- **Meeting:** Brainstorming session with multiple rounds
- **Round:** One iteration where agents respond and context accumulates
- **Turn:** Single agent's response within a round
- **Comment:** Extracted atomic idea from agent response
- **Context (Xr):** Cumulative content at round r
- **Compression:** Reducing comment count while preserving meaning
- **Convergence:** State where novelty score falls below threshold
- **Novelty:** Measure of new ideas introduced in a round (1 - overlap ratio)
- **Sequential Strategy:** Agents respond one at a time, building on cumulative context
- **Greedy Strategy:** All agents respond in parallel, then comment on each other (n² responses)
- **Notetaker:** Agent that extracts structured comments from responses
- **Compressor:** Agent that reduces comment count via three-tier compression

---

### 15.2 References

- **Tech Spec:** /home/delorenj/code/theboard/docs/tech-spec-theboard-2025-12-19.md
- **Brainstorming Session:** /home/delorenj/code/theboard/brainstorming-theboard-architecture-2025-12-19.md
- **PRD:** /home/delorenj/code/theboard/PRD.md
- **Agent Pool:** /home/delorenj/code/DeLoDocs/AI/Agents/Generic

---

### 15.3 Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-19 | System Architect (BMAD) | Initial architecture document |

---

*Generated by BMAD Method v6 - System Architect*
*Architecture for Level 2+ Project (17 stories, 5 phases)*
*All 8 FRs addressed, all 21 NFRs addressed*
*Total Pages: ~50 (comprehensive architecture)*
