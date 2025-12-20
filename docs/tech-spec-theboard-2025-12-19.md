# Technical Specification: TheBoard

**Project:** TheBoard - Multi-Agent Brainstorming Simulation System
**Date:** 2025-12-19
**Version:** 1.0
**Status:** Planning
**Author:** Product Manager (BMAD Method)

---

## 1. Problem Statement

LLM-generated artifacts (implementation plans, diffs, requirements docs, roadmaps) lack the critical refinement that comes from diverse domain expertise and structured debate. Teams need a systematic way to simulate multi-perspective brainstorming retreats that produce meticulously refined outputs examined through multiple domains, personalities, and disciplines.

Without this capability:
- Artifacts reflect single-perspective thinking
- Domain-specific risks and opportunities are missed
- Quality improvements require manual coordination of multiple experts
- No systematic process for iterative refinement through debate

---

## 2. Proposed Solution

TheBoard is a multi-agent brainstorming simulation system that orchestrates domain-expert AI agents through structured rounds of discussion, critique, and refinement. It mimics corporate retreat dynamics where agents take turns sharing ideas, building on peer comments, and converging toward high-quality artifacts.

**Core Capabilities:**
- Define meetings with topics and desired output artifacts
- Auto-select or manually compose agent teams from an expert pool
- Execute multi-round discussions using sequential or greedy strategies
- Extract and compress comments to manage context size
- Detect convergence automatically using novelty metrics
- Export refined artifacts in multiple formats

**Value Proposition:**
- Transform rough ideas into meticulously examined artifacts
- Leverage diverse domain expertise systematically
- Reduce human coordination overhead
- Produce audit trails of the refinement process

---

## 3. Requirements

### 3.1 Functional Requirements

#### F1: Meeting Configuration
- **F1.1:** User can create a meeting by specifying:
  - Topic/objective
  - Desired output artifact type (refined-doc, roadmap, recommendations, etc.)
  - Execution strategy (sequential or greedy)
  - Maximum rounds (default: 5)
  - Convergence threshold (default: 0.2)
- **F1.2:** User can select agents via:
  - Auto-selection based on topic analysis
  - Manual selection from agent pool
  - Interactive team builder wizard
- **F1.3:** Meeting configuration persisted to database
- **F1.4:** Meeting assigned unique ID for tracking

#### F2: Agent Orchestration
- **F2.1:** System supports two execution strategies:
  - **Sequential:** Agents respond one at a time, building on cumulative context
  - **Greedy:** All agents respond in parallel, then comment on each other (n² responses)
- **F2.2:** Turn ordering managed by Meeting Coordinator
- **F2.3:** Agent state tracked per round (current agent, completed turns)
- **F2.4:** Support for dynamic model switching:
  - Start all agents on cheap model (DeepSeek R3)
  - Promote top 20% to expensive model (Claude Opus) based on engagement metrics
- **F2.5:** Agent performance metrics tracked:
  - Comments generated
  - Peer references received
  - Novelty score average
  - Token usage and cost

#### F3: Comment Extraction & Compression
- **F3.1:** Notetaker agent extracts structured comments from responses
- **F3.2:** Comment schema includes:
  - Agent name
  - Text content
  - Category (technical_decision, implementation_detail, risk, etc.)
  - Timestamp
  - Novelty score
- **F3.3:** Three-tier compression applied to comment sets:
  1. **Embedding-based clustering:** Group similar comments using cosine similarity (threshold: 0.85)
  2. **LLM semantic merge:** Combine clusters into single coherent comments
  3. **Outlier removal:** Drop comments with low support (<2 agent mentions)
- **F3.4:** Compression metrics tracked:
  - Original count
  - Clustered count
  - Merged count
  - Final count
  - Compression ratio
- **F3.5:** Full audit trail of pre-compression responses maintained in database

#### F4: Context Management
- **F4.1:** Build cumulative context (Xr) across rounds:
  - X₀ = initial artifact
  - Xr = Xr-1 ∪ σ(⋃ᵢ C(Tir))
- **F4.2:** Active context cached in Redis for fast access
- **F4.3:** Context size tracking with alerts if exceeds threshold
- **F4.4:** Delta propagation: agents receive only new comments since last turn
- **F4.5:** Full context history archived in Postgres

#### F5: Convergence Detection
- **F5.1:** Calculate novelty score after each round:
  ```
  novelty(Rr) = 1 - |C(Rr) ∩ C(Rr-1)| / |C(Rr)|
  Where intersection uses embedding similarity
  ```
- **F5.2:** Stop meeting when:
  - novelty(Rr) < threshold for k consecutive rounds (default: 2)
  - OR max_rounds reached
  - OR human override
- **F5.3:** Convergence metrics persisted per round
- **F5.4:** User notified of convergence reason

#### F6: CLI Interface
- **F6.1:** `board create <topic>` command:
  - Interactive configuration wizard
  - Agent selection (auto or manual)
  - Display selected team with expertise
  - Output: meeting ID
- **F6.2:** `board run <meeting-id>` command:
  - Execute meeting with configured strategy
  - Optional `--watch` flag for live progress streaming
  - Optional `--resume` flag to continue paused meeting
  - Display round-by-round progress
- **F6.3:** `board status <meeting-id>` command:
  - Show current meeting state
  - Display round, agent, comment count, novelty score
  - Optional `--verbose` for detailed history
- **F6.4:** `board export <meeting-id>` command:
  - Support formats: markdown, JSON, HTML
  - Optional `--output` flag for file path
  - Default: print to stdout
- **F6.5:** Rich formatting for console output:
  - Tables for agent teams and statistics
  - Progress bars for live execution
  - Color coding for status

#### F7: Agent Pool Management
- **F7.1:** Load agents from file-based pool at /home/delorenj/code/DeLoDocs/AI/Agents/Generic
- **F7.2:** Parse plaintext agent descriptions:
  - Name
  - Expertise areas
  - Persona
  - Background
- **F7.3:** Index agent capabilities for auto-selection
- **F7.4:** Auto-select agents based on:
  - Topic embedding similarity with agent expertise
  - Artifact type requirements
  - Team diversity (different domains)
- **F7.5:** Support migration from plaintext to Letta format
- **F7.6:** Track agent performance across meetings

#### F8: Event-Driven Human-in-Loop
- **F8.1:** Emit RabbitMQ events at key decision points:
  - `agent.response.ready`
  - `context.compression.triggered`
  - `meeting.round.completed`
  - `meeting.convergence.detected`
  - `meeting.human.input.needed`
- **F8.2:** CLI consumer listens for events
- **F8.3:** User can pause meeting and provide steering input
- **F8.4:** Meeting state persisted for resume capability
- **F8.5:** Timeout defaults enable autonomous operation (5min → auto-continue)

### 3.2 Out of Scope (v1)

The following are explicitly deferred to future versions:

- Vision/multimodal agent support (images, diagrams, screenshots)
- Real-time collaborative editing UI (web interface)
- Integration with external git repos for artifact commit
- Advanced analytics dashboard with visualizations
- Multi-user concurrent meeting access
- Agent training/fine-tuning capabilities
- Meeting templates marketplace
- Integration with project management tools (Jira, Linear)
- Automated testing of generated artifacts
- Cross-meeting trend analysis

---

## 4. Technical Approach

### 4.1 Technology Stack

- **Language/Framework:** Python 3.12 + Agno (agent orchestration) + Typer (CLI)
- **Agent Framework:** Letta (agent definitions with persistent memory)
- **Database:** PostgreSQL 15 (persistent storage), Redis 7 (session state), Qdrant (vector embeddings)
- **Message Broker:** RabbitMQ 3.12 (event-driven workflows)
- **LLM Providers:**
  - Anthropic Claude (Opus 4.5 for leaders, Sonnet 4.5 for notetaker/compressor)
  - DeepSeek R3 (cheap worker agents)
- **Key Libraries:**
  - Pydantic (strict typing and data validation)
  - SQLAlchemy (ORM)
  - Rich (CLI formatting)
  - sentence-transformers (embeddings)
  - aiormq (RabbitMQ async client)
- **Containerization:** Docker + Docker Compose
- **Development Tools:** uv (Python package manager), ruff (linting/formatting), mypy (type checking)

### 4.2 Architecture Overview

TheBoard follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│      CLI Layer (Typer + Rich)           │
│  - Commands: create, run, status, export│
│  - Interactive prompts                   │
│  - Live progress display                 │
└──────────────┬──────────────────────────┘
               │ Command invocation
┌──────────────▼──────────────────────────┐
│  Orchestration Layer (Agno Workflows)    │
│  - Meeting Coordinator                   │
│  - Agent Registry                        │
│  - Context Manager                       │
│  - Artifact Manager                      │
└──────────────┬──────────────────────────┘
               │ Agent coordination
┌──────────────▼──────────────────────────┐
│   Agent Layer (Letta agents)             │
│  - Participant Agents (domain experts)   │
│  - Notetaker Agent (comment extraction)  │
│  - Compressor Agent (compression)        │
└──────────────┬──────────────────────────┘
               │ LLM calls + memory
┌──────────────▼──────────────────────────┐
│         Data Layer                       │
│  - Postgres: persistent storage          │
│  - Redis: session state                  │
│  - Qdrant: vector embeddings             │
│  - RabbitMQ: event bus                   │
└──────────────────────────────────────────┘
```

**Key Components:**

1. **Meeting Coordinator (Agno Workflow):**
   - Orchestrates round progression
   - Manages turn ordering
   - Executes sequential or greedy strategies
   - Detects convergence
   - Emits events for human-in-loop

2. **Agent Registry:**
   - Indexes agent capabilities
   - Tracks availability and performance
   - Auto-selects teams based on topic
   - Manages agent lifecycle

3. **Context Manager:**
   - Builds cumulative context (Xr)
   - Orchestrates compression
   - Handles delta propagation
   - Caches active context in Redis
   - Archives full history to Postgres

4. **Notetaker Agent:**
   - Extracts structured comments from responses
   - Categorizes comments
   - Calculates novelty scores
   - Uses Claude Sonnet for extraction

5. **Compressor Agent:**
   - Applies three-tier compression
   - Clusters using Qdrant embeddings
   - Merges semantically using Claude Sonnet
   - Removes outliers based on support

6. **Participant Agents:**
   - Domain experts from pool or custom
   - Letta-based with persistent memory
   - Recall past meetings
   - Track collaboration history

7. **Event Bus:**
   - RabbitMQ for async communication
   - Human-in-loop decision points
   - Progress notifications
   - State change events

### 4.3 Data Flow

**Meeting Execution Flow:**

1. **Initialization:**
   ```
   User → CLI create → Meeting definition → Postgres
                    → Agent selection → Meeting state → Redis
                    → Event: meeting.created → RabbitMQ
   ```

2. **Per Round:**
   ```
   Meeting Coordinator → Agents (parallel/sequential)
                      ↓
   Agent Responses → Notetaker → Comments → Postgres
                      ↓
   Comments → Compressor → Qdrant clustering
                        → LLM merge
                        → Compressed comments → Redis
                      ↓
   Context Manager → Append to Xr → Redis
                  → Archive Xr-1 → Postgres
                      ↓
   Convergence Check → Novelty calculation
                    → If converged: Stop
                    → Else: Next round
                      ↓
   Event: round.completed → RabbitMQ → CLI display
   ```

3. **Completion:**
   ```
   Final context → Artifact Manager → Generate artifact
                                   → Export (markdown/JSON/HTML)
                                   → Postgres
                                   ↓
   Event: meeting.completed → RabbitMQ → CLI notification
   ```

### 4.4 Data Model

**PostgreSQL Schema:**

```sql
-- Meeting definitions
CREATE TABLE meetings (
    id UUID PRIMARY KEY,
    topic TEXT NOT NULL,
    artifact_type VARCHAR(50),
    strategy VARCHAR(20) CHECK (strategy IN ('sequential', 'greedy')),
    max_rounds INT DEFAULT 5,
    convergence_threshold FLOAT DEFAULT 0.2,
    status VARCHAR(20) CHECK (status IN ('active', 'paused', 'completed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    final_artifact TEXT
);

-- Agent pool
CREATE TABLE agents (
    id UUID PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    expertise TEXT[] NOT NULL,
    persona TEXT,
    background TEXT,
    model VARCHAR(50) DEFAULT 'deepseek-r3',
    letta_definition JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Meeting participants (many-to-many)
CREATE TABLE meeting_agents (
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (meeting_id, agent_id)
);

-- Historical responses (full audit trail)
CREATE TABLE responses (
    id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    round INT NOT NULL,
    agent_id UUID REFERENCES agents(id),
    response_type VARCHAR(20) CHECK (response_type IN ('full', 'comment')),
    response_text TEXT NOT NULL,
    token_count INT,
    model_used VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_meeting_round (meeting_id, round)
);

-- Extracted comments (structured)
CREATE TABLE comments (
    id UUID PRIMARY KEY,
    response_id UUID REFERENCES responses(id) ON DELETE CASCADE,
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    round INT NOT NULL,
    agent_id UUID REFERENCES agents(id),
    text TEXT NOT NULL,
    category VARCHAR(50),
    novelty_score FLOAT,
    compressed BOOLEAN DEFAULT FALSE,
    merged_into UUID REFERENCES comments(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_meeting_round_compressed (meeting_id, round, compressed)
);

-- Convergence tracking
CREATE TABLE convergence_metrics (
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    round INT NOT NULL,
    novelty_score FLOAT NOT NULL,
    comment_count INT NOT NULL,
    compression_ratio FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (meeting_id, round)
);

-- Agent memory (for Letta integration)
CREATE TABLE agent_memory (
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    memory_type VARCHAR(50) NOT NULL,
    memory_key TEXT NOT NULL,
    memory_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, memory_type, memory_key)
);

-- Agent performance metrics
CREATE TABLE agent_performance (
    meeting_id UUID REFERENCES meetings(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id),
    comments_generated INT DEFAULT 0,
    peer_references INT DEFAULT 0,
    novelty_avg FLOAT,
    engagement_score FLOAT,
    total_tokens INT DEFAULT 0,
    cost DECIMAL(10, 4) DEFAULT 0,
    PRIMARY KEY (meeting_id, agent_id)
);
```

**Redis Schema:**

```python
# Active meeting state
meeting:{id}:state = {
    "id": str,
    "current_round": int,
    "current_agent": str,
    "active_context": str,  # Cumulative Xr
    "turn_queue": list[str],
    "status": str,
    "strategy": str,
    "created_at": str,
    "updated_at": str
}
# TTL: 7 days for completed meetings

# Round comments cache
meeting:{id}:round:{r}:comments = list[Comment]
# TTL: 24 hours

# Compression metadata per round
meeting:{id}:compression:{round} = {
    "round": int,
    "original_count": int,
    "clustered_count": int,
    "merged_count": int,
    "final_count": int,
    "compression_ratio": float,
    "timestamp": str
}
# TTL: 7 days

# Agent performance cache
meeting:{id}:agent:{name}:performance = {
    "comments_generated": int,
    "peer_references": int,
    "novelty_avg": float,
    "engagement_score": float,
    "total_tokens": int,
    "cost": float
}
# TTL: 7 days
```

**RabbitMQ Event Schema:**

```python
# Exchange: theboard.events
# Routing keys: meeting.{event_type}

# Event: agent.response.ready
{
    "event": "agent.response.ready",
    "meeting_id": str,
    "round": int,
    "agent": str,
    "response_type": "full" | "comment",
    "response_length": int,
    "timestamp": str  # ISO 8601
}

# Event: context.compression.triggered
{
    "event": "context.compression.triggered",
    "meeting_id": str,
    "round": int,
    "comment_count_before": int,
    "comment_count_after": int,
    "compression_ratio": float,
    "timestamp": str
}

# Event: meeting.round.completed
{
    "event": "meeting.round.completed",
    "meeting_id": str,
    "round": int,
    "comments_generated": int,
    "novelty_score": float,
    "context_size": int,
    "timestamp": str
}

# Event: meeting.convergence.detected
{
    "event": "meeting.convergence.detected",
    "meeting_id": str,
    "round": int,
    "novelty_score": float,
    "threshold": float,
    "stopping_early": bool,
    "timestamp": str
}

# Event: meeting.human.input.needed
{
    "event": "meeting.human.input.needed",
    "meeting_id": str,
    "round": int,
    "reason": str,
    "options": list[str],
    "timeout_seconds": int,
    "timestamp": str
}
```

### 4.5 Core Algorithms

**Algorithm 1: Sequential Strategy**

```python
def sequential_round(context: str, round_num: int) -> list[Comment]:
    """Each agent responds to cumulative context in sequence"""
    comments = []

    for agent in agents:
        # Agent full response
        response = await agent.full_response(context)

        # Extract comments
        agent_comments = await notetaker.extract_comments(response, agent.name)
        comments.extend(agent_comments)

        # Compress incrementally
        compressed = await compressor.compress(agent_comments)

        # Update context for next agent
        context = append_to_context(context, compressed)

    return comments
```

**Algorithm 2: Greedy Strategy**

```python
def greedy_round(context: str, round_num: int) -> list[Comment]:
    """All agents respond in parallel, then comment on each other"""
    comments = []

    # Phase 1: Parallel full responses
    responses = await asyncio.gather(*[
        agent.full_response(context) for agent in agents
    ])

    # Extract comments from all full responses
    for agent, response in zip(agents, responses):
        agent_comments = await notetaker.extract_comments(response, agent.name)
        comments.extend(agent_comments)

    # Phase 2: Comment-responses (each agent responds to others)
    for i, agent in enumerate(agents):
        other_comments = [c for j, c in enumerate(comments) if j != i]
        comment_response = await agent.comment_response(other_comments)

        response_comments = await notetaker.extract_comments(
            comment_response,
            agent.name
        )
        comments.extend(response_comments)

    return comments
```

**Algorithm 3: Three-Tier Compression**

```python
def compress(comments: list[Comment]) -> list[Comment]:
    """
    σ(C) = merge_similar ∘ summarize_verbose ∘ drop_outliers
    """
    # Tier 1: Embedding-based clustering
    clusters = cluster_similar_comments(comments, threshold=0.85)

    # Tier 2: LLM semantic merge
    merged = []
    for cluster in clusters:
        if len(cluster) > 1:
            merged_comment = await merge_cluster(cluster)
            merged.append(merged_comment)
        else:
            merged.append(cluster[0])

    # Tier 3: Drop outliers (support < 2)
    supported = [c for c in merged if has_support(c, comments, min_support=2)]

    return supported

def cluster_similar_comments(comments: list[Comment], threshold: float):
    """Use Qdrant cosine similarity to cluster"""
    embeddings = embed_texts([c.text for c in comments])
    clusters = []
    used = set()

    for i, emb_i in enumerate(embeddings):
        if i in used:
            continue
        cluster = [comments[i]]
        used.add(i)

        for j, emb_j in enumerate(embeddings[i+1:], i+1):
            if j in used:
                continue
            if cosine_similarity(emb_i, emb_j) > threshold:
                cluster.append(comments[j])
                used.add(j)

        clusters.append(cluster)

    return clusters
```

**Algorithm 4: Convergence Detection**

```python
def check_convergence(
    current_comments: list[Comment],
    previous_comments: list[Comment],
    threshold: float = 0.2
) -> bool:
    """
    novelty(Rr) = 1 - |C(Rr) ∩ C(Rr-1)| / |C(Rr)|
    """
    if not previous_comments:
        return False

    # Embed all comments
    curr_embs = embed_texts([c.text for c in current_comments])
    prev_embs = embed_texts([c.text for c in previous_comments])

    # Count overlaps (similarity > 0.85)
    overlap_count = 0
    for curr_emb in curr_embs:
        for prev_emb in prev_embs:
            if cosine_similarity(curr_emb, prev_emb) > 0.85:
                overlap_count += 1
                break

    # Calculate novelty
    novelty = 1 - (overlap_count / len(current_comments))

    return novelty < threshold
```

### 4.6 API Design

**Internal Python APIs (not REST endpoints):**

```python
# Meeting Coordinator (Agno Workflow)
class TheboardMeeting(Workflow):
    topic: str
    agents: list[DomainExpertAgent]
    strategy: Literal["sequential", "greedy"]
    max_rounds: int = 5
    convergence_threshold: float = 0.2

    async def run() -> str
    async def sequential_round(context: str, round_num: int) -> list[Comment]
    async def greedy_round(context: str, round_num: int) -> list[Comment]
    async def check_convergence(current: list[Comment], all_comments: list[Comment]) -> bool
    async def generate_artifact(context: str, comments: list[Comment]) -> str

# Context Manager
class ContextManager:
    async def append_to_context(context: str, comments: list[Comment]) -> str
    async def get_delta(agent_name: str, since_round: int) -> list[Comment]
    def get_context_size() -> int
    async def cache_context(meeting_id: str, context: str)
    async def retrieve_context(meeting_id: str) -> str

# Compressor Agent
class CompressorAgent(Agent):
    similarity_threshold: float = 0.85

    async def compress(comments: list[Comment]) -> list[Comment]
    async def cluster_similar_comments(comments: list[Comment]) -> list[list[Comment]]
    async def merge_cluster(cluster: list[Comment]) -> Comment
    def has_support(comment: Comment, all_comments: list[Comment], min_support: int = 2) -> bool

# Agent Registry
class AgentRegistry:
    async def load_pool(pool_dir: Path) -> list[Agent]
    async def auto_select_team(topic: str, artifact_type: str, count: int = 5) -> list[Agent]
    async def get_agent_by_name(name: str) -> Agent
    async def track_performance(meeting_id: str, agent_name: str, metrics: dict)
    async def get_performance(meeting_id: str, agent_name: str) -> dict

# Notetaker Agent
class NotetakerAgent(Agent):
    async def extract_comments(response: str, agent_name: str) -> list[Comment]
    async def categorize_comment(text: str) -> str
    async def calculate_novelty(comment: str, existing: list[Comment]) -> float
```

---

## 5. Implementation Plan

### Phase 1: Foundation (MVP Core)

**Story 1: Project Setup & Data Layer**
- Initialize Python project with uv
- Set up Docker Compose with Postgres, Redis, RabbitMQ, Qdrant
- Define SQLAlchemy models for meetings, agents, responses, comments
- Create Alembic migrations
- Implement Redis connection manager
- Test database connectivity and basic CRUD

**Story 2: Basic CLI Structure**
- Install and configure Typer
- Implement CLI app skeleton with commands: create, run, status, export
- Add Rich formatting for console output
- Implement basic `board create` with minimal inputs
- Implement basic `board status` for meeting display
- Test CLI user experience

**Story 3: Agno Integration & Simple Agent**
- Install and configure Agno framework
- Create DomainExpertAgent as Agno skill
- Implement single-agent single-round execution
- Integrate Claude Sonnet for LLM calls
- Test Agno workflow execution
- Validate agent response generation

**Story 4: Notetaker Agent Implementation**
- Create NotetakerAgent with structured extraction
- Define Comment pydantic model with validation
- Implement comment extraction using Claude Sonnet
- Test extraction accuracy on sample responses
- Store extracted comments in Postgres
- Display comments via CLI

### Phase 2: Multi-Agent Orchestration

**Story 5: Meeting Coordinator Workflow**
- Implement TheboardMeeting as Agno workflow
- Add round management logic (loop, counter, state tracking)
- Implement sequential strategy execution
- Track meeting state in Redis (current_round, current_agent, turn_queue)
- Emit basic events to RabbitMQ
- Test multi-round execution with 2 agents

**Story 6: Agent Pool Management**
- Create agent pool loader for plaintext descriptions
- Parse agent files: name, expertise, persona, background
- Implement AgentRegistry with in-memory index
- Add auto-select team logic using topic keywords
- Support manual agent selection via CLI interactive prompt
- Test team composition for various topics

**Story 7: Context Management**
- Implement ContextManager for cumulative context building
- Add context persistence to Redis with TTL
- Implement context size tracking and warnings
- Test multi-round context accumulation
- Validate context coherence across rounds
- Archive context history to Postgres

### Phase 3: Compression & Convergence

**Story 8: Embedding Infrastructure**
- Set up Qdrant vector database in Docker Compose
- Implement comment embedding pipeline using sentence-transformers
- Add cosine similarity computation
- Test embedding quality and similarity thresholds
- Optimize embedding batch processing
- Add embedding caching

**Story 9: Compressor Agent**
- Implement CompressorAgent with three-tier strategy
- Add similarity-based clustering using Qdrant
- Implement LLM semantic merge using Claude Sonnet
- Add outlier removal logic based on support counts
- Track compression metrics (ratio, counts)
- Test compression quality vs. information retention

**Story 10: Convergence Detection**
- Implement novelty score calculation using embeddings
- Add convergence threshold checking (default: 0.2)
- Test stopping criteria with various scenarios
- Persist convergence metrics per round to database
- Emit convergence events to RabbitMQ
- Display convergence reason to user

### Phase 4: Advanced Features

**Story 11: Greedy Execution Strategy**
- Implement parallel agent response collection using asyncio.gather
- Add comment-response phase (each agent responds to others)
- Optimize for token efficiency (track n² cost)
- Compare performance: greedy vs. sequential
- Test convergence behavior with greedy strategy
- Document trade-offs in user guide

**Story 12: Event-Driven Human-in-Loop**
- Set up RabbitMQ event publishing for 5 key event types
- Implement CLI event consumer with async listener
- Add interactive prompts for human steering
- Implement meeting pause/resume capability
- Add timeout defaults (5min auto-continue)
- Test async human intervention scenarios

**Story 13: Hybrid Model Strategy**
- Implement engagement metric calculation (weighted: peer_references, novelty, comment_count)
- Add dynamic model promotion logic (top 20% → Opus after round 1)
- Track per-agent costs and token usage
- Test cost savings vs. quality (A/B comparison)
- Document optimal promotion thresholds
- Add budget limits per meeting

### Phase 5: Letta Integration & Polish

**Story 14: Letta Agent Migration**
- Implement plaintext → Letta migration script with regex parsing
- Add Letta memory persistence to agent_memory table
- Implement cross-meeting memory recall (previous_meetings, learned_patterns)
- Test memory-enhanced responses
- Migrate sample agents from pool
- Document migration process

**Story 15: Export & Artifact Generation**
- Implement markdown export with formatting
- Add JSON export with structured schema
- Add HTML export with CSS styling
- Test artifact quality and readability
- Support custom export templates
- Add artifact validation

**Story 16: Performance Optimization**
- Implement lazy compression (only when context > 10K chars)
- Add delta propagation (agents receive only new comments)
- Optimize Redis caching strategy
- Add selective agent activation (only engaged agents in round 2+)
- Benchmark latency and token usage
- Document optimization settings

**Story 17: CLI Polish & Documentation**
- Enhance `board export` command with format options
- Implement live progress streaming with Rich.Live
- Add comprehensive help text for all commands
- Write user guide with examples
- Create troubleshooting guide
- Add developer documentation

---

## 6. Acceptance Criteria

The project is considered complete when all of the following criteria are met:

### Core Functionality
- [ ] User can create a meeting via `board create` with topic and agent selection (auto or manual)
- [ ] Sequential strategy executes multiple rounds with context accumulation (tested with 5 agents, 5 rounds)
- [ ] Greedy strategy executes with parallel responses and comment-response phase
- [ ] Notetaker successfully extracts structured comments from agent responses (>90% extraction rate)
- [ ] Compressor reduces comment count by 40-60% while preserving quality (validated via spot checks)
- [ ] Convergence detection stops meeting when novelty < 0.2 for 2 consecutive rounds
- [ ] Context size stays under 15K tokens for 5-round, 5-agent meeting
- [ ] Agent pool auto-selection chooses relevant agents based on topic (>80% relevance)

### CLI Experience
- [ ] `board status` displays current round, agent, comment count, novelty score with Rich formatting
- [ ] `board run --watch` provides live progress streaming with round-by-round updates
- [ ] `board export` generates readable markdown, JSON, and HTML artifacts
- [ ] Interactive team selection wizard allows manual agent composition
- [ ] All commands respond within 500ms for status checks

### Persistence & State
- [ ] All responses and comments persisted to Postgres for audit trail
- [ ] Meeting state cached in Redis for fast retrieval (<100ms)
- [ ] Convergence metrics tracked per round in database
- [ ] Meeting can be paused and resumed from same state (tested)

### Advanced Features
- [ ] Human-in-loop can pause meeting via event consumer and resume
- [ ] Hybrid model strategy reduces cost by >60% compared to all-Opus baseline (measured)
- [ ] Letta agents recall past meeting outcomes in new meetings (tested)
- [ ] Delta propagation reduces token usage by >30% (measured)

### Quality & Testing
- [ ] Unit test coverage >70% for core logic (orchestration, compression, convergence)
- [ ] Integration tests validate end-to-end meeting execution (at least 3 scenarios)
- [ ] Type checking passes with mypy --strict
- [ ] Linting passes with ruff
- [ ] Docker Compose brings up full stack (Postgres, Redis, Qdrant, RabbitMQ) in <60s

### Documentation
- [ ] User guide with 3+ example scenarios
- [ ] API documentation for core classes
- [ ] Troubleshooting guide for common issues
- [ ] Architecture diagram (Mermaid or similar)

---

## 7. Non-Functional Requirements

### 7.1 Performance

- **Meeting Execution Latency:**
  - Sequential strategy: <30s per round for 5 agents
  - Greedy strategy: <45s per round for 5 agents
  - Total meeting: <5 minutes for 5 rounds, 5 agents (sequential)

- **Compression Latency:**
  - <5s for compressing 50 comments
  - <10s for compressing 100 comments

- **Convergence Calculation:**
  - <2s per round for novelty score computation
  - Embedding similarity: <1s for 100 comment comparisons

- **CLI Responsiveness:**
  - Status checks: <500ms
  - Live progress updates: <200ms per update
  - Command initialization: <1s

- **Database Performance:**
  - Meeting state retrieval: <100ms (Redis)
  - Comment insertion: <50ms per batch (Postgres)
  - Query response times: <200ms for typical queries

- **Scalability:**
  - Support up to 10 concurrent agents per meeting
  - Handle 100+ past meetings in agent memory without degradation
  - Context management: up to 20K tokens without truncation

### 7.2 Security

- **Credential Management:**
  - API keys stored via environment variables (.env file, not committed)
  - Database credentials managed via Docker secrets in production
  - No hardcoded secrets in codebase

- **Network Security:**
  - LLM API calls use TLS 1.2+
  - RabbitMQ authentication required (username/password)
  - Redis AUTH enabled in production
  - Postgres SSL mode: require (production)

- **Data Privacy:**
  - No PII stored in agent responses (user responsible for input sanitization)
  - Meeting data isolated per user (single-user system for v1)
  - Audit trail maintained for all operations

- **Input Validation:**
  - All user inputs validated via Pydantic models
  - SQL injection prevention via SQLAlchemy parameterized queries
  - Command injection prevention in CLI arguments

### 7.3 Reliability

- **Fault Tolerance:**
  - Meeting state persisted to survive crashes (Redis + Postgres)
  - Idempotent event handlers (RabbitMQ consumer)
  - Graceful degradation if LLM API fails (retry with exponential backoff, max 3 retries)

- **Data Integrity:**
  - Transaction boundaries for database writes
  - Atomic updates to meeting state in Redis
  - Foreign key constraints enforced in Postgres

- **Recovery:**
  - Meeting resume capability after interruption
  - Rollback on partial failures (database transactions)
  - Event replay capability for RabbitMQ

- **Availability:**
  - Target: 99% uptime for local development
  - Graceful shutdown on SIGTERM
  - Health checks for all services (Docker Compose healthcheck)

### 7.4 Observability

- **Logging:**
  - Structured logging (JSON format) for all key events
  - Log levels: DEBUG (dev), INFO (prod), ERROR (always)
  - Log rotation configured (max 100MB per file)
  - Correlation IDs for meeting execution traces

- **Metrics:**
  - Token usage per agent, per round, per meeting
  - Costs tracked ($ per meeting, per agent)
  - Latency metrics (round duration, compression time, convergence calc)
  - Convergence rounds histogram

- **Monitoring:**
  - CLI progress display with Rich (live updates)
  - Event stream for external monitoring (RabbitMQ)
  - Health endpoints for services (Postgres, Redis, Qdrant, RabbitMQ)

- **Debugging:**
  - Full audit trail in database (all responses pre-compression)
  - Debug mode: verbose logging with stack traces
  - Replay capability for meeting execution

### 7.5 Maintainability

- **Code Quality:**
  - Type hints throughout (enforced via mypy --strict)
  - Pydantic models for all data structures
  - Clear separation: CLI → Orchestration → Agents → Data
  - Docstrings for all public APIs (Google style)

- **Code Style:**
  - Formatted via ruff (PEP 8 compliant)
  - Import sorting via ruff
  - Max line length: 100 characters
  - Consistent naming conventions

- **Version Control:**
  - Git commit messages follow conventional commits
  - Semantic versioning for releases
  - Changelog maintained (CHANGELOG.md)

- **Dependency Management:**
  - All dependencies via uv (pyproject.toml)
  - Pinned versions in production
  - Security updates applied regularly

### 7.6 Usability

- **CLI UX:**
  - Rich formatting: tables, progress bars, color coding
  - Clear error messages with suggestions
  - Interactive prompts with defaults
  - Help text for all commands

- **Documentation:**
  - User guide with examples
  - Troubleshooting guide
  - API reference
  - Architecture diagram

- **Defaults:**
  - Sensible defaults for all parameters
  - Override via CLI flags or config file
  - Config file: ~/.theboard/config.yaml (future)

---

## 8. Dependencies, Risks, and Timeline

### 8.1 Dependencies

**External Services:**
- **Anthropic Claude API**
  - Models: Opus 4.5, Sonnet 4.5
  - Rate limits: 10 RPM (free tier), 100 RPM (paid tier)
  - Pricing: ~$0.015 per 1K input tokens (Opus)
  - Status: Production-ready, high availability

- **DeepSeek API**
  - Model: R3
  - Pricing: ~$0.001 per 1K input tokens
  - Status: Production-ready

**Infrastructure:**
- **Docker + Docker Compose**
  - Version: 24.0+
  - Required for: Local development, containerized deployment

- **PostgreSQL 15**
  - Docker image: postgres:15-alpine
  - Storage: 10GB initial, 100GB max

- **Redis 7**
  - Docker image: redis:7-alpine
  - Storage: 1GB RAM

- **Qdrant**
  - Docker image: qdrant/qdrant:latest
  - Storage: 5GB initial

- **RabbitMQ 3.12**
  - Docker image: rabbitmq:3.12-management
  - Storage: 1GB

**Python Ecosystem:**
- **Agno Framework**
  - Version: Latest (0.x)
  - Purpose: Agent orchestration
  - Risk: Early-stage framework (mitigation: fallback to direct async/await)

- **Letta Framework**
  - Version: Latest
  - Purpose: Agent definitions with memory
  - Risk: Integration complexity (mitigation: start with plaintext, migrate incrementally)

- **Core Libraries:**
  - Typer: CLI framework
  - Rich: CLI formatting
  - Pydantic: Data validation
  - SQLAlchemy: ORM
  - sentence-transformers: Embeddings

**Agent Pool:**
- **Initial Pool Location:** /home/delorenj/code/DeLoDocs/AI/Agents/Generic
- **Format:** Plaintext descriptions
- **Migration Path:** Plaintext → Letta definitions

### 8.2 Risks & Mitigation

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|-----------|---------|---------------------|
| **Token costs exceed budget** | Medium | High | 1. Implement hybrid model strategy (cheap workers, expensive leaders)<br>2. Lazy compression (only when context > threshold)<br>3. Selective agent activation (only engaged agents in round 2+)<br>4. Set per-meeting budget limits with alerts<br>5. Monitor costs real-time via dashboard |
| **Convergence never reached** | Medium | Medium | 1. Set max_rounds hard limit (default: 5)<br>2. Human-in-loop can force stop<br>3. Monitor novelty trends per round<br>4. Adjust convergence threshold dynamically<br>5. Document when manual intervention needed |
| **Compression loses critical information** | Medium | High | 1. Maintain full audit trail pre-compression<br>2. Add compression quality metrics (spot checks)<br>3. Human review option for critical meetings<br>4. Tunable compression aggressiveness<br>5. Warning if compression ratio > 0.7 |
| **Agno framework limitations** | Low | High | 1. Prototype MVP early to validate Agno suitability<br>2. Fallback plan: Direct async/await orchestration<br>3. Abstract orchestration layer (loose coupling)<br>4. Monitor Agno community and updates |
| **Letta integration complexity** | Medium | Medium | 1. Start with plaintext agents (simpler)<br>2. Migrate incrementally (hybrid support)<br>3. Support both formats long-term<br>4. Comprehensive migration testing<br>5. Document migration process |
| **LLM API rate limits** | Medium | Medium | 1. Implement exponential backoff (max 3 retries)<br>2. Queue management for parallel requests<br>3. Option to use local models (Ollama) as fallback<br>4. Monitor rate limit headers<br>5. Consider paid tier for production |
| **Context explosion despite compression** | Medium | High | 1. Aggressive compression thresholds (θ_merge = 0.85)<br>2. Delta propagation (agents get only new comments)<br>3. Truncation as last resort with user warning<br>4. Context size alerts at 15K tokens<br>5. Test with large meetings (10 agents, 10 rounds) |
| **Agent disagreement deadlock** | Low | Medium | 1. Timeout → human escalation<br>2. Consider mediator agent role (future)<br>3. Document as known limitation for v1<br>4. Voting mechanism (future)<br>5. Allow manual override |
| **Docker Compose complexity for users** | Low | Medium | 1. Provide one-command startup script<br>2. Comprehensive setup documentation<br>3. Pre-built Docker images (future)<br>4. Health checks for all services<br>5. Troubleshooting guide |
| **Qdrant embedding performance** | Low | Low | 1. Batch embedding processing<br>2. Embedding caching in Redis<br>3. Monitor query latency<br>4. Consider Qdrant cloud for scale (future)<br>5. Optimize embedding model choice |

### 8.3 Timeline & Milestones

**Effort Estimates (Not Time-Based):**

- **Phase 1 (Foundation):** M effort
  - 4 stories, foundational infrastructure

- **Phase 2 (Multi-Agent):** M effort
  - 3 stories, core orchestration logic

- **Phase 3 (Intelligence):** M-L effort
  - 3 stories, compression and convergence (more complex)

- **Phase 4 (Advanced):** M effort
  - 3 stories, optimization features

- **Phase 5 (Polish):** M-L effort
  - 4 stories, Letta integration and finalization

**Total Effort:** L-XL effort for full v1 implementation

**Key Milestones:**

1. **Milestone 1: MVP Foundation Complete**
   - Single-agent execution working end-to-end
   - Basic CLI functional (create, run, status)
   - Data layer operational (Postgres, Redis)
   - **Success Criteria:** Can run 1-agent, 3-round meeting and export artifact
   - **Completion:** After Phase 1 (Stories 1-4)

2. **Milestone 2: Multi-Agent Orchestration Working**
   - Sequential strategy executing 5-agent meetings
   - Context accumulation validated across rounds
   - Agent pool management functional (auto-select + manual)
   - **Success Criteria:** Can run 5-agent, 5-round meeting with coherent context
   - **Completion:** After Phase 2 (Stories 5-7)

3. **Milestone 3: Intelligence Layer Complete**
   - Compression reducing tokens by 40%+ with quality preservation
   - Convergence detection stopping at appropriate rounds
   - Embedding infrastructure operational (Qdrant)
   - **Success Criteria:** Meeting stops automatically when converged, context manageable
   - **Completion:** After Phase 3 (Stories 8-10)

4. **Milestone 4: Advanced Features Working**
   - Greedy strategy functional
   - Human-in-loop pause/resume working
   - Hybrid model strategy reducing costs by >60%
   - **Success Criteria:** User can steer meetings, costs optimized
   - **Completion:** After Phase 4 (Stories 11-13)

5. **Milestone 5: Production-Ready v1**
   - All 17 stories implemented and tested
   - Acceptance criteria met (see Section 6)
   - Documentation complete (user guide, API docs, troubleshooting)
   - Docker deployment validated
   - **Success Criteria:** System ready for real-world usage
   - **Completion:** After Phase 5 (Stories 14-17)

**Critical Path:**
```
Foundation → Multi-Agent → Compression → Convergence → Export
```

These are sequential dependencies. Advanced features (greedy, human-in-loop, Letta) can be parallelized.

**Dependencies on External Blockers:**
- **None:** All dependencies are established OSS projects or stable APIs
- **Agno/Letta:** Early-stage frameworks, but fallback plans exist

---

## 9. Success Metrics

Post-launch, success will be measured by:

**Quality Metrics:**
- Artifact quality improvement (subjective, via human eval)
- Comment compression ratio: 40-60% (measured)
- Convergence rate: >80% of meetings converge before max_rounds (measured)
- Context coherence: >90% of users report coherent artifacts (survey)

**Performance Metrics:**
- Meeting execution time: <5 minutes for typical meeting (measured)
- Token efficiency: <100K tokens for 5-agent, 5-round meeting (measured)
- Cost per meeting: <$2 with hybrid strategy (measured)

**Usability Metrics:**
- CLI responsiveness: <500ms for status checks (measured)
- User satisfaction: >80% would use again (survey)
- Setup time: <10 minutes from clone to first meeting (measured)

**Adoption Metrics (future):**
- Active users
- Meetings per week
- Agent pool growth

---

## 10. Future Enhancements (Out of Scope for v1)

**Vision/Multimodal Support:**
- Vision agents review screenshots, diagrams
- Architecture agents review system diagrams
- UI/UX agents review wireframes

**Web UI:**
- Real-time collaborative editing
- Visualization of agent discussions
- Meeting dashboard with analytics

**Advanced Agent Features:**
- Cross-meeting meta-learning
- Agent specialization via fine-tuning
- Dynamic capability discovery

**Integration:**
- Git repo integration (commit artifacts)
- Project management tools (Jira, Linear)
- Slack/Discord notifications

**Enterprise Features:**
- Multi-user concurrent access
- Role-based access control
- Team workspaces
- SSO authentication

---

## 11. Appendix

### A. Glossary

- **Agent:** AI entity with domain expertise, powered by LLM
- **Meeting:** Brainstorming session with multiple rounds
- **Round:** One iteration where agents respond and context accumulates
- **Turn:** Single agent's response within a round
- **Comment:** Extracted atomic idea from agent response
- **Context (Xr):** Cumulative content at round r
- **Compression:** Reducing comment count while preserving meaning
- **Convergence:** State where novelty score falls below threshold
- **Novelty:** Measure of new ideas introduced in a round
- **Sequential Strategy:** Agents respond one at a time
- **Greedy Strategy:** All agents respond in parallel, then comment
- **Notetaker:** Agent that extracts comments from responses
- **Compressor:** Agent that reduces comment count

### B. References

- PRD: /home/delorenj/code/theboard/PRD.md
- Brainstorming Session: /home/delorenj/code/theboard/brainstorming-theboard-architecture-2025-12-19.md
- Agent Pool: /home/delorenj/code/DeLoDocs/AI/Agents/Generic

### C. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-19 | Product Manager (BMAD) | Initial technical specification |

---

*Generated by BMAD Method v6 - Product Manager*
*Tech Spec for Level 2+ Project*
*Estimated Total Effort: L-XL*
