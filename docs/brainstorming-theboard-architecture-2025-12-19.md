---
modified: 2025-12-20T00:10:16-05:00
---
# Brainstorming Session: TheBoard Architecture & Formalization

**Date:** 2025-12-19
**Objective:** Expand PRD components with formalized algorithms, Agno/Typer/Letta integration patterns, and mathematical notation
**Context:** Multi-agent brainstorming simulation system requiring production-ready architecture

---

## Techniques Used

1. **SCAMPER** - Generate algorithm variations and optimization strategies
2. **Mind Mapping** - Map component hierarchy and data flow
3. **Starbursting** - Surface critical technical questions across domains

---

## Ideas Generated

### 1. Mathematical Formalization

**Core Algorithm Notation:**
```
Let:
  P = {a₁, a₂, ..., aₙ} be the set of n participating agents
  X₀ = initial context (input artifact)
  Rᵣ = round r (r ∈ ℕ)
  Tᵢᵣ = agent aᵢ's turn in round r
  C(Tᵢᵣ) = set of comments extracted from turn Tᵢᵣ
  Xᵣ = cumulative context after round r
  σ(C) = compression function applied to comment set C

Full Response Turn:
  Tᵢᵣ: aᵢ(Xᵣ₋₁) → Response_text
  C(Tᵢᵣ) = Notetaker(Response_text)

Comment-Response Turn:
  Tᵢⱼᵣ: aᵢ responds to aⱼ's comments from same round
  C(Tᵢⱼᵣ) = Notetaker(aᵢ(C(Tⱼᵣ)))

Context Evolution:
  X₀ = initial_artifact
  Xᵣ = Xᵣ₋₁ ∪ σ(⋃ᵢ₌₁ⁿ C(Tᵢᵣ))

Greedy Strategy (Pⁿ responses per round):
  Round_r = [T₁ᵣ, T₁₂ᵣ, ..., T₁ₙᵣ, T₂ᵣ, T₂₁ᵣ, ..., Tₙᵣ]
  Total_responses_per_round = n + n(n-1) = n²

Sequential Strategy:
  Round_r = [T₁ᵣ, T₂ᵣ, ..., Tₙᵣ]
  Xᵣ = Xᵣ₋₁ ∪ σ(C(T₁ᵣ)) ∪ σ(C(T₂ᵣ)) ∪ ... ∪ σ(C(Tₙᵣ))

Compression Function:
  σ(C) : {c₁, c₂, ..., cₘ} → {c'₁, c'₂, ..., c'ₖ} where k ≤ m

  σ = merge_similar ∘ summarize_verbose ∘ drop_outliers

  merge_similar: cosine_similarity(emb(cᵢ), emb(cⱼ)) > θ_merge → merge
  summarize_verbose: len(cᵢ) > θ_len → LLM_summarize(cᵢ)
  drop_outliers: support(cᵢ) < θ_support → remove

Convergence Detection:
  novelty(Rᵣ) = 1 - |C(Rᵣ) ∩ C(Rᵣ₋₁)| / |C(Rᵣ)|

  Where intersection uses embedding similarity:
    cᵢ ∈ C(Rᵣ) ∩ C(Rᵣ₋₁) iff ∃cⱼ ∈ C(Rᵣ₋₁): sim(cᵢ, cⱼ) > θ_sim

  Stop when: novelty(Rᵣ) < θ_novelty for k consecutive rounds

Cost Model:
  Cost per round (greedy):
    Cᵣ = n × cost_full + n(n-1) × cost_comment

    Where:
      cost_full = avg_tokens_full × model_price
      cost_comment = avg_tokens_comment × model_price

  Hybrid model optimization:
    Leaders ⊆ P use Claude Opus (expensive, high quality)
    Workers ⊆ P use DeepSeek (cheap, adequate)

    Cᵣ = |Leaders| × opus_cost + |Workers| × deepseek_cost

  Example savings:
    Static (all Opus): 10 agents × 5 rounds × $0.15/1K × 2K = $15
    Hybrid (2 Opus, 8 DeepSeek): $3.20 (78% reduction)
```

---

### 2. Component Architecture (Mind Map)

```
TheBoard System
│
├── CLI Layer (Typer)
│   ├── Commands
│   │   ├── board create <topic> <output-artifact>
│   │   ├── board select-team [--auto|--custom]
│   │   ├── board run <meeting-id> [--rounds N] [--watch]
│   │   ├── board status <meeting-id>
│   │   └── board export <meeting-id> <format>
│   │
│   └── Interactive Features
│       ├── Meeting configuration wizard
│       ├── Agent selection interface
│       ├── Live progress display (Rich.Live)
│       └── Human-in-loop decision prompts
│
├── Orchestration Layer (Agno)
│   ├── Meeting Coordinator (Agno Workflow)
│   │   ├── Session initialization
│   │   ├── Round management
│   │   ├── Turn ordering strategy
│   │   └── Convergence detection
│   │
│   ├── Agent Registry
│   │   ├── Capability indexing
│   │   ├── Availability tracking
│   │   └── Performance metrics
│   │
│   ├── Context Manager
│   │   ├── Cumulative context builder (Xᵣ evolution)
│   │   ├── Compression orchestrator
│   │   └── Delta propagation
│   │
│   └── Artifact Manager
│       ├── Input artifact parser
│       ├── Comment aggregator
│       └── Output artifact generator
│
├── Agent Layer (Letta)
│   ├── Participant Agents (Agno Skills)
│   │   ├── Domain specialists (from pool)
│   │   ├── Custom agents (ad-hoc)
│   │   └── Agent state persistence
│   │
│   ├── Notetaker Agent
│   │   ├── Comment extraction (C(Tᵢᵣ))
│   │   ├── Fact identification
│   │   └── Idea categorization
│   │
│   └── Compressor Agent
│       ├── Similarity detection (embedding-based)
│       ├── Merge logic (LLM semantic combination)
│       └── Summarization
│
├── Data Layer
│   ├── Meeting State (Redis)
│   │   ├── Current round/agent tracking
│   │   ├── Active context (Xᵣ)
│   │   ├── Turn queue
│   │   └── Convergence metrics
│   │
│   ├── Persistent Storage (Postgres)
│   │   ├── Meeting definitions
│   │   ├── Agent pool
│   │   ├── Historical responses (all rounds)
│   │   └── Final artifacts
│   │
│   ├── Vector Store (Qdrant)
│   │   ├── Comment embeddings
│   │   ├── Similarity search for compression
│   │   └── Meeting topic clustering
│   │
│   └── Event Stream (RabbitMQ) see [bloodbank-quickref](bloodbank-quickref.md)
│       ├── Agent response events
│       ├── Compression triggers
│       ├── Human intervention requests
│       └── Round completion notifications
│
└── Integration Layer
    ├── Agent Pool Loader
    │   ├── File-based (plaintext descriptions)
    │   ├── Letta-native definitions
    │   └── Migration tooling
    │
    ├── LLM Backends
    │   ├── Claude Opus 4.5 (leader agents)
    │   ├── DeepSeek R3 (worker agents)
    │   ├── Fallback routing
    │   └── Dynamic model promotion
    │
    └── Export Formats
        ├── Markdown report
        ├── Git diff
        ├── Structured JSON
        └── HTML dashboard
```

---

### 3. Agno Integration Patterns

**Agent as Agno Skill:**
```python
from agno import Agent, Tool, task
from pydantic import BaseModel

class Comment(BaseModel):
    agent: str
    text: str
    timestamp: str
    category: str
    novelty_score: float

class DomainExpertAgent(Agent):
    """Base agent for TheBoard participants"""

    name: str
    expertise: list[str]
    model: str = "claude-opus-4.5"

    @task
    async def full_response(self, context: str) -> str:
        """Generate comprehensive response to context"""
        return await self.run(
            f"Review this {context} from your expertise in {self.expertise}. "
            f"Provide detailed analysis, identify risks, and suggest improvements."
        )

    @task
    async def comment_response(self, peer_comments: list[Comment]) -> str:
        """React to peer agent's comments"""
        comments_text = "\n".join([f"- {c.agent}: {c.text}" for c in peer_comments])
        return await self.run(
            f"Respond to these ideas from your {self.expertise} perspective:\n{comments_text}"
        )

class NotetakerAgent(Agent):
    """Extracts structured comments from responses"""

    model: str = "claude-sonnet-4.5"

    @task
    async def extract_comments(self, response: str, agent_name: str) -> list[Comment]:
        """Parse response into atomic comments"""
        # Use structured extraction with pydantic
        result = await self.run(
            f"Extract key points from this response as structured comments:\n{response}",
            response_model=list[Comment]
        )
        return result

class CompressorAgent(Agent):
    """Manages context compression"""

    model: str = "claude-sonnet-4.5"
    similarity_threshold: float = 0.85

    @task
    async def compress(self, comments: list[Comment]) -> list[Comment]:
        """Apply three-tier compression strategy"""
        # 1. Embedding-based clustering
        clusters = await self.cluster_similar_comments(comments)

        # 2. LLM semantic merge
        merged = []
        for cluster in clusters:
            if len(cluster) > 1:
                merged_comment = await self.merge_cluster(cluster)
                merged.append(merged_comment)
            else:
                merged.append(cluster[0])

        # 3. Drop outliers (low support)
        supported = [c for c in merged if self.has_support(c, comments)]

        return supported

    async def cluster_similar_comments(self, comments: list[Comment]) -> list[list[Comment]]:
        """Cluster comments using embedding similarity"""
        # Qdrant similarity search implementation
        pass

    async def merge_cluster(self, cluster: list[Comment]) -> Comment:
        """Merge similar comments using LLM"""
        texts = [c.text for c in cluster]
        merged_text = await self.run(
            f"Combine these similar ideas into one coherent comment:\n" +
            "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        )
        return Comment(
            agent="compressor",
            text=merged_text,
            timestamp=cluster[0].timestamp,
            category=cluster[0].category,
            novelty_score=max(c.novelty_score for c in cluster)
        )
```

**Meeting as Agno Workflow:**
```python
from agno import Workflow, task
from typing import Literal

class TheboardMeeting(Workflow):
    """Orchestrates multi-round agent collaboration"""

    topic: str
    agents: list[DomainExpertAgent]
    notetaker: NotetakerAgent
    compressor: CompressorAgent
    strategy: Literal["greedy", "sequential"] = "sequential"
    max_rounds: int = 5
    convergence_threshold: float = 0.2

    @task
    async def run(self) -> str:
        """Execute multi-round brainstorming meeting"""
        context = await self.load_initial_context()
        all_comments = []

        for round_num in range(1, self.max_rounds + 1):
            self.log(f"Starting Round {round_num}")

            # Execute round based on strategy
            if self.strategy == "greedy":
                round_comments = await self.greedy_round(context, round_num)
            else:
                round_comments = await self.sequential_round(context, round_num)

            # Compress comments
            compressed = await self.compressor.compress(round_comments)
            all_comments.extend(compressed)

            # Update context
            context = await self.append_to_context(context, compressed)

            # Check convergence
            if round_num > 1 and await self.check_convergence(compressed, all_comments):
                self.log(f"Convergence detected at round {round_num}")
                break

            # Emit event for human-in-loop
            await self.emit_event("round.completed", {
                "round": round_num,
                "comments": len(compressed),
                "context_size": len(context)
            })

        # Generate final artifact
        artifact = await self.generate_artifact(context, all_comments)
        return artifact

    @task
    async def sequential_round(self, context: str, round_num: int) -> list[Comment]:
        """Execute sequential round: each agent responds to cumulative context"""
        comments = []

        for agent in self.agents:
            # Agent full response
            response = await agent.full_response(context)

            # Extract comments
            agent_comments = await self.notetaker.extract_comments(response, agent.name)
            comments.extend(agent_comments)

            # Update context incrementally for next agent
            context = await self.append_to_context(context, agent_comments)

        return comments

    @task
    async def greedy_round(self, context: str, round_num: int) -> list[Comment]:
        """Execute greedy round: all agents respond, then all comment on each other"""
        comments = []

        # Phase 1: All agents do full response in parallel
        full_responses = await asyncio.gather(*[
            agent.full_response(context) for agent in self.agents
        ])

        # Extract comments from full responses
        for agent, response in zip(self.agents, full_responses):
            agent_comments = await self.notetaker.extract_comments(response, agent.name)
            comments.extend(agent_comments)

        # Phase 2: Each agent comments on all other agents' comments
        for i, agent in enumerate(self.agents):
            other_comments = [c for j, c in enumerate(comments) if j != i]
            comment_response = await agent.comment_response(other_comments)

            response_comments = await self.notetaker.extract_comments(
                comment_response,
                agent.name
            )
            comments.extend(response_comments)

        return comments

    @task
    async def check_convergence(
        self,
        current_comments: list[Comment],
        all_comments: list[Comment]
    ) -> bool:
        """Detect convergence using novelty metric"""
        # Get previous round comments
        prev_comments = all_comments[:-len(current_comments)]

        # Calculate novelty score
        novelty = await self.calculate_novelty(current_comments, prev_comments)

        return novelty < self.convergence_threshold

    async def calculate_novelty(
        self,
        current: list[Comment],
        previous: list[Comment]
    ) -> float:
        """
        novelty(Rᵣ) = 1 - |C(Rᵣ) ∩ C(Rᵣ₋₁)| / |C(Rᵣ)|
        Where intersection uses embedding similarity
        """
        if not previous:
            return 1.0

        # Compute embeddings
        current_embs = await self.get_embeddings([c.text for c in current])
        prev_embs = await self.get_embeddings([c.text for c in previous])

        # Count overlapping comments (similarity > threshold)
        overlap_count = 0
        for curr_emb in current_embs:
            for prev_emb in prev_embs:
                if cosine_similarity(curr_emb, prev_emb) > 0.85:
                    overlap_count += 1
                    break

        novelty = 1 - (overlap_count / len(current))
        return novelty
```

---

### 4. Typer CLI Architecture

**Command Structure:**
```python
import typer
from rich import print
from rich.table import Table
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

app = typer.Typer(
    name="board",
    help="TheBoard: Multi-agent brainstorming simulation"
)

@app.command()
def create(
    topic: str = typer.Argument(..., help="Meeting topic or objective"),
    artifact_type: str = typer.Option(
        "refined-doc",
        "--artifact", "-a",
        help="Output artifact type: refined-doc, roadmap, recommendations"
    ),
    rounds: int = typer.Option(5, "--rounds", "-r", help="Maximum rounds"),
    strategy: str = typer.Option(
        "sequential",
        "--strategy", "-s",
        help="Execution strategy: sequential or greedy"
    ),
):
    """Create and configure a new brainstorming meeting"""

    # Initialize meeting
    meeting_id = initialize_meeting(topic, artifact_type, rounds, strategy)
    print(f"[green]✓ Meeting {meeting_id} created[/green]")
    print(f"  Topic: {topic}")
    print(f"  Artifact: {artifact_type}")
    print(f"  Strategy: {strategy}")
    print()

    # Interactive team selection
    if typer.confirm("Auto-select team based on topic?", default=True):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Analyzing topic and selecting agents...", total=None)
            agents = auto_select_team(topic, artifact_type)
            progress.update(task, completed=True)
    else:
        agents = interactive_team_builder()

    assign_agents(meeting_id, agents)

    # Display team
    table = Table(title="Selected Team")
    table.add_column("Agent", style="cyan")
    table.add_column("Expertise", style="magenta")
    table.add_column("Model", style="green")

    for agent in agents:
        table.add_row(agent.name, ", ".join(agent.expertise), agent.model)

    print(table)
    print()
    print(f"[blue]Run meeting with: board run {meeting_id}[/blue]")

@app.command()
def run(
    meeting_id: str = typer.Argument(..., help="Meeting ID to execute"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Stream live progress"),
    resume: bool = typer.Option(False, "--resume", help="Resume paused meeting"),
):
    """Execute a brainstorming meeting"""

    meeting = load_meeting(meeting_id)

    if not watch:
        # Simple execution with final result
        print(f"[yellow]Starting meeting {meeting_id}...[/yellow]")
        result = meeting.run()
        print()
        print("[green]✓ Meeting completed[/green]")
        print(result.summary)
    else:
        # Live progress display
        with Live(auto_refresh=True) as live:
            for update in meeting.run_async():
                live.update(render_progress(update))

        print()
        print("[green]✓ Meeting completed[/green]")

@app.command()
def status(
    meeting_id: str = typer.Argument(..., help="Meeting ID to check"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status"),
):
    """Check meeting progress and statistics"""

    meeting = load_meeting(meeting_id)

    # Summary table
    summary = Table(title=f"Meeting {meeting_id} Status")
    summary.add_column("Attribute", style="cyan")
    summary.add_column("Value", style="yellow")

    summary.add_row("Topic", meeting.topic)
    summary.add_row("Status", meeting.status)
    summary.add_row("Current Round", str(meeting.current_round))
    summary.add_row("Total Agents", str(len(meeting.agents)))
    summary.add_row("Comments Generated", str(meeting.total_comments))
    summary.add_row("Context Size", f"{len(meeting.context):,} chars")

    print(summary)

    if verbose:
        # Detailed round-by-round breakdown
        print()
        rounds_table = Table(title="Round History")
        rounds_table.add_column("Round")
        rounds_table.add_column("Agent")
        rounds_table.add_column("Comments")
        rounds_table.add_column("Novelty")

        for round_data in meeting.history:
            rounds_table.add_row(
                str(round_data.round),
                round_data.agent,
                str(round_data.comment_count),
                f"{round_data.novelty:.2f}"
            )

        print(rounds_table)

@app.command()
def export(
    meeting_id: str = typer.Argument(..., help="Meeting ID to export"),
    format: str = typer.Option("markdown", "--format", "-f", help="Export format"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export meeting results to various formats"""

    meeting = load_meeting(meeting_id)

    if format == "markdown":
        content = meeting.export_markdown()
    elif format == "json":
        content = meeting.export_json()
    elif format == "html":
        content = meeting.export_html()
    else:
        print(f"[red]Unknown format: {format}[/red]")
        raise typer.Exit(1)

    if output:
        Path(output).write_text(content)
        print(f"[green]✓ Exported to {output}[/green]")
    else:
        print(content)

def interactive_team_builder() -> list[Agent]:
    """Interactive agent selection wizard"""

    # Load agent pool
    pool = load_agent_pool()

    print("[bold]Agent Pool:[/bold]")
    for i, agent in enumerate(pool, 1):
        print(f"  {i}. {agent.name} - {', '.join(agent.expertise)}")

    print()
    selected_indices = typer.prompt(
        "Select agents (comma-separated numbers)",
        type=str
    )

    indices = [int(x.strip()) - 1 for x in selected_indices.split(",")]
    selected = [pool[i] for i in indices]

    return selected

def render_progress(update: dict) -> Table:
    """Render live progress update as Rich table"""

    table = Table(title="Meeting Progress")
    table.add_column("Round", style="cyan")
    table.add_column("Agent", style="magenta")
    table.add_column("Action", style="yellow")
    table.add_column("Comments", style="green")

    table.add_row(
        str(update["round"]),
        update["agent"],
        update["action"],
        str(update["comment_count"])
    )

    return table
```

---

### 5. Letta Agent Definitions

**Agent Pool Migration:**
```python
from letta import Agent, Memory
from pathlib import Path
import re

def migrate_plaintext_to_letta(agent_file: Path) -> Agent:
    """Convert plaintext agent description to Letta definition"""

    description = agent_file.read_text()

    # Parse structured fields from plaintext
    name = extract_field(description, r"Name:\s*(.+)")
    expertise = extract_list(description, r"Expertise:\s*(.+)")
    persona = extract_field(description, r"Persona:\s*(.+)", multiline=True)
    background = extract_field(description, r"Background:\s*(.+)", multiline=True)

    # Create Letta agent with memory
    agent = Agent(
        name=name,
        persona=persona,
        human=f"A participant in TheBoard meetings. Expertise: {', '.join(expertise)}",
        memory=Memory(
            human=f"Domain expert in: {', '.join(expertise)}",
            persona=persona,
        ),
    )

    # Add TheBoard-specific context to memory
    agent.memory.update({
        "role": "meeting_participant",
        "expertise_areas": expertise,
        "background": background,
        "previous_meetings": [],
        "learned_patterns": {},
        "collaboration_history": {},
    })

    return agent

def extract_field(text: str, pattern: str, multiline: bool = False) -> str:
    """Extract field from plaintext using regex"""
    flags = re.MULTILINE | re.DOTALL if multiline else 0
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else ""

def extract_list(text: str, pattern: str) -> list[str]:
    """Extract comma-separated list from plaintext"""
    match = re.search(pattern, text)
    if not match:
        return []
    return [x.strip() for x in match.group(1).split(",")]

# Bulk migration
def migrate_agent_pool(pool_dir: Path, output_dir: Path):
    """Migrate entire agent pool from plaintext to Letta"""

    pool_files = pool_dir.glob("*.txt")

    for agent_file in pool_files:
        print(f"Migrating {agent_file.name}...")

        agent = migrate_plaintext_to_letta(agent_file)

        # Save Letta agent definition
        output_file = output_dir / f"{agent.name}.json"
        agent.save(output_file)

        print(f"  ✓ Saved to {output_file}")
```

**Persistent Agent Memory:**
```python
class TheboardAgentMemory:
    """Extended memory for agents participating in multiple meetings"""

    previous_meetings: list[dict]
    learned_patterns: dict[str, str]
    collaboration_history: dict[str, int]
    topic_expertise_growth: dict[str, float]

    def recall_similar_meeting(self, current_topic: str) -> Optional[dict]:
        """Retrieve relevant past meeting context using vector similarity"""

        if not self.previous_meetings:
            return None

        # Embed current topic
        current_emb = embed_text(current_topic)

        # Find most similar past meeting
        best_match = None
        best_similarity = 0.0

        for meeting in self.previous_meetings:
            past_emb = embed_text(meeting["topic"])
            similarity = cosine_similarity(current_emb, past_emb)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = meeting

        return best_match if best_similarity > 0.7 else None

    def update_after_meeting(
        self,
        meeting_id: str,
        topic: str,
        agents_interacted: list[str],
        outcome: str
    ):
        """Update agent memory after meeting completion"""

        # Add to meeting history
        self.previous_meetings.append({
            "id": meeting_id,
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
        })

        # Update collaboration history
        for other_agent in agents_interacted:
            self.collaboration_history[other_agent] = \
                self.collaboration_history.get(other_agent, 0) + 1

        # Extract learned patterns (would use LLM in production)
        # Example: "API redesign meetings prioritize backward compatibility"
        topic_category = categorize_topic(topic)
        self.learned_patterns[topic_category] = outcome

        # Update expertise growth
        self.topic_expertise_growth[topic_category] = \
            self.topic_expertise_growth.get(topic_category, 0.0) + 0.1

# Integration with Letta Agent
class TheboardLettaAgent(Agent):
    """Letta agent with TheBoard-specific memory"""

    theboard_memory: TheboardAgentMemory

    async def full_response_with_memory(self, context: str, topic: str) -> str:
        """Generate response using long-term memory"""

        # Recall similar past meetings
        similar_meeting = self.theboard_memory.recall_similar_meeting(topic)

        # Build enhanced prompt
        memory_context = ""
        if similar_meeting:
            memory_context = f"\n\nPast experience: In a similar meeting about '{similar_meeting['topic']}', the outcome was: {similar_meeting['outcome']}"

        # Generate response with memory context
        response = await self.run(
            f"{context}{memory_context}\n\nProvide your analysis from your expertise."
        )

        return response
```

---

### 6. Data Flow & State Management

**Redis State Schema:**
```python
# Meeting state key pattern: meeting:{id}:state
{
    "id": "mtg_abc123",
    "topic": "API redesign for backward compatibility",
    "artifact_type": "refined-spec",
    "current_round": 2,
    "current_agent": "architect_1",
    "active_context": "...",  # Current Xᵣ (compressed)
    "turn_queue": ["dev_1", "pm_1", "qa_1"],
    "status": "active" | "paused" | "completed",
    "strategy": "sequential",
    "convergence_scores": [0.8, 0.5],  # Novelty per round
    "created_at": "2025-12-19T10:00:00Z",
    "updated_at": "2025-12-19T10:15:00Z"
}

# Comment cache: meeting:{id}:round:{r}:comments
[
    {
        "agent": "architect_1",
        "text": "Versioning strategy should use semantic versioning",
        "timestamp": "2025-12-19T10:05:00Z",
        "category": "technical_decision",
        "novelty_score": 0.9
    },
    {
        "agent": "dev_1",
        "text": "Agree with semver. Suggest v2 namespace for breaking changes.",
        "timestamp": "2025-12-19T10:06:00Z",
        "category": "implementation_detail",
        "novelty_score": 0.6
    }
]

# Compression metadata: meeting:{id}:compression:{round}
{
    "round": 2,
    "original_count": 47,
    "clustered_count": 18,
    "merged_count": 12,
    "summarized_count": 8,
    "dropped_count": 3,
    "final_count": 24,
    "compression_ratio": 0.51,
    "timestamp": "2025-12-19T10:07:00Z"
}

# Agent performance: meeting:{id}:agent:{name}:performance
{
    "agent": "architect_1",
    "full_responses": 2,
    "comment_responses": 5,
    "comments_generated": 12,
    "peer_references": 8,  # How many times other agents referenced this agent
    "novelty_avg": 0.72,
    "engagement_score": 0.85,  # Weighted metric
    "model_used": "claude-opus-4.5",
    "total_tokens": 8450,
    "cost": 1.27
}
```

**Postgres Schema:**
```sql
-- Meeting definitions
CREATE TABLE meetings (
    id UUID PRIMARY KEY,
    topic TEXT NOT NULL,
    artifact_type VARCHAR(50),
    strategy VARCHAR(20),
    max_rounds INT,
    status VARCHAR(20),
    created_at TIMESTAMP,
    completed_at TIMESTAMP,
    final_artifact TEXT
);

-- Agent pool
CREATE TABLE agents (
    id UUID PRIMARY KEY,
    name VARCHAR(100) UNIQUE,
    expertise TEXT[],
    persona TEXT,
    background TEXT,
    model VARCHAR(50),
    letta_definition JSONB,
    created_at TIMESTAMP
);

-- Meeting participants
CREATE TABLE meeting_agents (
    meeting_id UUID REFERENCES meetings(id),
    agent_id UUID REFERENCES agents(id),
    joined_at TIMESTAMP,
    PRIMARY KEY (meeting_id, agent_id)
);

-- Historical responses (full audit trail)
CREATE TABLE responses (
    id UUID PRIMARY KEY,
    meeting_id UUID REFERENCES meetings(id),
    round INT,
    agent_id UUID REFERENCES agents(id),
    response_type VARCHAR(20),  -- 'full' or 'comment'
    response_text TEXT,
    token_count INT,
    created_at TIMESTAMP
);

-- Extracted comments
CREATE TABLE comments (
    id UUID PRIMARY KEY,
    response_id UUID REFERENCES responses(id),
    meeting_id UUID REFERENCES meetings(id),
    round INT,
    agent_id UUID REFERENCES agents(id),
    text TEXT,
    category VARCHAR(50),
    novelty_score FLOAT,
    compressed BOOLEAN DEFAULT FALSE,
    merged_into UUID REFERENCES comments(id),
    created_at TIMESTAMP
);

-- Convergence tracking
CREATE TABLE convergence_metrics (
    meeting_id UUID REFERENCES meetings(id),
    round INT,
    novelty_score FLOAT,
    comment_count INT,
    compression_ratio FLOAT,
    timestamp TIMESTAMP,
    PRIMARY KEY (meeting_id, round)
);

-- Agent memory (for Letta integration)
CREATE TABLE agent_memory (
    agent_id UUID REFERENCES agents(id),
    memory_type VARCHAR(50),  -- 'meeting', 'pattern', 'collaboration'
    memory_key TEXT,
    memory_value JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (agent_id, memory_type, memory_key)
);
```

**RabbitMQ Event Schema:**
Let’s use Bloodbank events/commands [Bloodbank CLI PRD](Bloodbank%20CLI%20PRD.md)
```python
# Exchange: theboard.events
# Routing keys: meeting.{event_type}

# Event: agent.response.ready
{
    "event": "agent.response.ready",
    "meeting_id": "mtg_abc123",
    "round": 2,
    "agent": "architect_1",
    "response_type": "full",
    "response_length": 1247,
    "timestamp": "2025-12-19T10:05:00Z"
}

# Event: context.compression.triggered
{
    "event": "context.compression.triggered",
    "meeting_id": "mtg_abc123",
    "round": 2,
    "comment_count_before": 47,
    "comment_count_after": 24,
    "compression_ratio": 0.51,
    "timestamp": "2025-12-19T10:07:00Z"
}

# Event: meeting.round.completed
{
    "event": "meeting.round.completed",
    "meeting_id": "mtg_abc123",
    "round": 2,
    "comments_generated": 24,
    "novelty_score": 0.5,
    "context_size": 12450,
    "timestamp": "2025-12-19T10:08:00Z"
}

# Event: meeting.convergence.detected
{
    "event": "meeting.convergence.detected",
    "meeting_id": "mtg_abc123",
    "round": 4,
    "novelty_score": 0.15,
    "threshold": 0.2,
    "stopping_early": True,
    "timestamp": "2025-12-19T10:20:00Z"
}

# Event: meeting.human.input.needed
{
    "event": "meeting.human.input.needed",
    "meeting_id": "mtg_abc123",
    "round": 2,
    "reason": "low_novelty_but_no_consensus",
    "options": ["continue", "stop", "add_agents"],
    "timeout_seconds": 300,
    "timestamp": "2025-12-19T10:08:00Z"
}
```

---

### 7. Optimization Strategies

**1. Early Stopping via Novelty Detection**
- Track comment similarity across rounds using embeddings
- Stop when novelty < θ_novelty for k consecutive rounds
- **Savings:** ~40% token cost on converged topics
- **Implementation:** After each round, compute novelty(Rᵣ) and check stopping condition

**2. Selective Agent Activation**
- Not all agents needed in all rounds
- Round 1: Full team participation
- Round 2+: Only agents whose comments got peer responses
- **Reduces:** Pn² to dynamic subset (typically 30-50% reduction)
- **Implementation:** Track peer_reference_count per agent, threshold = 2

**3. Lazy Compression**
- Don't compress until context size > threshold (e.g., 10K chars)
- Batch compress every N rounds instead of per-turn
- **Trade-off:** Higher peak memory vs. fewer LLM calls
- **Savings:** ~50% compression LLM calls
- **Implementation:** Check context_size after round, trigger compression if > threshold

**4. Hybrid Execution Strategy**
- Parallel for independent full responses
- Sequential for building on peers
- **Auto-detect:** Parse comment references to determine dependencies
- **Savings:** ~30% latency reduction on independent rounds

**5. Dynamic Model Promotion**
- Start all agents on cheap models (DeepSeek R3)
- Promote top 20% to expensive models (Claude Opus) based on engagement
- **Engagement metric:** weighted_sum(peer_references, novelty_score, comment_count)
- **Savings:** 60-80% cost reduction with minimal quality loss
- **Implementation:** After round 1, compute engagement scores, promote top agents

**6. Context Delta Propagation**
- Don't propagate full context Xᵣ to every agent
- Only send delta: new comments since agent's last turn
- **Trade-off:** Context coherence vs. token efficiency
- **Savings:** ~40% input token reduction
- **Implementation:** Track last_seen_round per agent, only send comments from rounds > last_seen

**7. Preemptive Agent Specialization**
- Use topic classification to pre-filter agent pool
- Example: "API redesign" → architects, backend devs, API specialists
- **Reduces:** Team size from 10+ to 5-7 targeted experts
- **Savings:** ~30% total tokens
- **Implementation:** Embed topic + agent expertise, cosine similarity ranking

---

### 8. Extension Points

**Multi-Artifact Meetings:**
```python
class MultiArtifactMeeting(TheboardMeeting):
    """Handle multiple input artifacts simultaneously"""

    artifacts: list[tuple[str, str]]  # [(type, content), ...]

    async def run(self):
        # Each artifact gets separate comment stream
        artifact_comments = {}

        for artifact_type, artifact_content in self.artifacts:
            comments = await self.process_artifact(artifact_type, artifact_content)
            artifact_comments[artifact_type] = comments

        # Final merge: Cross-artifact insights
        cross_insights = await self.extract_cross_artifact_insights(artifact_comments)

        return self.generate_unified_artifact(artifact_comments, cross_insights)
```

**Vision Agent Integration:**
```python
class VisionAgent(DomainExpertAgent):
    """Agent capable of analyzing images (screenshots, diagrams)"""

    model: str = "claude-opus-4.5"  # Supports vision

    @task
    async def analyze_image(self, image_path: str, context: str) -> str:
        """Analyze screenshot or diagram"""
        image_data = load_image(image_path)

        return await self.run(
            f"Analyze this image in context of: {context}",
            images=[image_data]
        )

# Usage in meeting
class UIReviewMeeting(TheboardMeeting):
    """Meeting type for UI/UX review with screenshots"""

    screenshots: list[str]

    async def run(self):
        # UI/UX agents review screenshots
        for screenshot in self.screenshots:
            visual_comments = await self.vision_agents_review(screenshot)
            # Integrate with text-based discussion
```

**Persistent Agent Learning:**
```python
class LearningAgent(TheboardLettaAgent):
    """Agent with cross-meeting learning"""

    async def full_response_with_learning(self, context: str, topic: str) -> str:
        # Recall similar past meetings
        past_meeting = self.theboard_memory.recall_similar_meeting(topic)

        # Recall learned patterns
        topic_category = categorize_topic(topic)
        learned_pattern = self.theboard_memory.learned_patterns.get(topic_category)

        # Build enhanced prompt
        learning_context = ""
        if past_meeting:
            learning_context += f"\n\nPast experience: {past_meeting['outcome']}"
        if learned_pattern:
            learning_context += f"\n\nLearned pattern: {learned_pattern}"

        response = await self.run(f"{context}{learning_context}")
        return response
```

**Meeting Templates:**
```python
MEETING_TEMPLATES = {
    "product_roadmap": {
        "agents": ["pm", "architect", "designer", "market_research"],
        "artifact_type": "roadmap",
        "max_rounds": 3,
        "strategy": "sequential"
    },
    "bug_triage": {
        "agents": ["qa", "dev", "support", "devops"],
        "artifact_type": "triage-report",
        "max_rounds": 2,
        "strategy": "greedy"
    },
    "api_redesign": {
        "agents": ["backend_architect", "api_specialist", "dev", "tech_writer"],
        "artifact_type": "api-spec",
        "max_rounds": 4,
        "strategy": "sequential"
    },
    "ui_review": {
        "agents": ["ux_designer", "frontend_dev", "accessibility_expert", "product_designer"],
        "artifact_type": "design-recommendations",
        "max_rounds": 3,
        "strategy": "sequential",
        "vision_enabled": True
    }
}

def create_from_template(template_name: str, topic: str) -> TheboardMeeting:
    """Instantiate meeting from template"""
    template = MEETING_TEMPLATES[template_name]

    agents = [load_agent(name) for name in template["agents"]]

    return TheboardMeeting(
        topic=topic,
        agents=agents,
        artifact_type=template["artifact_type"],
        max_rounds=template["max_rounds"],
        strategy=template["strategy"]
    )
```

---

## Critical Technical Questions (Starbursting)

### WHO
- ✓ **Convergence decision:** Meeting Coordinator detects via novelty metric, human override available
- ✓ **Agent performance metrics:** Agent Registry tracks engagement, Compressor tracks novelty
- ✓ **Compression trigger:** Context Manager based on size threshold + round completion
- **LLM cost management:** Need budget tracking service with per-meeting limits
- **Artifact quality validation:** Need QA agent or human review step before export

### WHAT
- ✓ **Comment vs. response:** Comment = extracted atomic idea (via Notetaker), Response = full LLM output
- ✓ **Compression threshold:** θ_merge = 0.85 (cosine similarity), θ_len = 500 chars, θ_support = 2 agents
- ✓ **Context data structure:** String accumulation with Redis caching, Postgres archival
- ✓ **Novelty metric:** 1 - (overlap ratio using embedding similarity)
- **Consensus failure:** Need tiebreaker mechanism or human escalation

### WHERE
- ✓ **Agent state:** Letta native memory + Redis for session state
- ✓ **Response archive:** Postgres with JSONB for full audit trail
- ✓ **Human-in-loop:** CLI prompts via RabbitMQ event consumer
- ✓ **Agent pool:** File-based plaintext + Letta JSON (migration path defined)
- **Orchestration boundary:** Agno Workflow owns round logic, Agents own domain logic

### WHEN
- ✓ **Execution switch:** Sequential by default, greedy opt-in via CLI flag
- **Meeting archival:** After 30 days inactive or explicit user archive command
- **Redis eviction:** TTL = 7 days for completed meetings
- **Model promotion:** After round 1 completion based on engagement scores
- **Re-compression:** Lazy strategy - only when context_size > 10K chars

### WHY
- ✓ **Separate Notetaker/Compressor:** Specialization - extraction vs. optimization, different prompts
- ✓ **RabbitMQ events:** Decouple execution from human-in-loop, enable async workflows
- ✓ **Persist all responses:** Audit trail, debugging, future analysis/learning
- **Text-only agents:** Start simple, vision extension defined as future enhancement
- **Corporate retreat model:** Proven pattern for structured collaboration, maps to agent orchestration

### HOW
- ✓ **Context explosion prevention:** Lazy compression + selective agent activation + delta propagation
- ✓ **Agent diversity:** Topic-based filtering + capability indexing in Agent Registry
- **Deadlock handling:** Timeout → human escalation OR introduce mediator agent
- **Plaintext → Letta migration:** Batch migration script with regex parsing (defined above)
- **Artifact validation:** QA agent reviews against requirements OR structured output validation

---

## Key Insights

### 1. Layered Compression Strategy
**Impact:** High | **Effort:** M-L

Three-tier compression (embedding clustering → LLM merge → outlier removal) addresses core scalability bottleneck. Enables 10+ agent teams with measurable quality metrics.

### 2. Agno Workflow as Meeting Orchestrator
**Impact:** High | **Effort:** M

Agno provides native async execution, state management, and observability. Avoids building custom orchestration.

### 3. Typer + Rich for Production CLI UX
**Impact:** Medium | **Effort:** M

Live progress streaming and interactive prompts critical for 5-15 minute meetings with human-in-loop workflows.

### 4. Formalized Convergence Detection
**Impact:** High | **Effort:** L

Novelty metric (1 - overlap ratio) provides objective stopping criteria. Prevents token waste on redundant rounds.

### 5. Letta for Agent Memory Persistence
**Impact:** Medium-High | **Effort:** L

Long-term memory enables meta-learning: agents recall past outcomes, collaboration patterns, learned domain knowledge.

### 6. Hybrid Model Strategy with Dynamic Promotion
**Impact:** Medium | **Effort:** M

Start cheap (DeepSeek), promote top 20% to expensive (Opus) based on engagement. 60-80% cost reduction.

### 7. Event-Driven Architecture for Human-in-Loop
**Impact:** Medium | **Effort:** M

RabbitMQ events at decision points enable async human review. Decouples execution from human availability.

---

## Statistics

- **Total ideas generated:** 75+
- **Architectural domains:** 8 (Math, Agno, Typer, Letta, Data, Optimization, Extensions, Questions)
- **Key insights:** 7
- **Techniques applied:** 3 (SCAMPER, Mind Mapping, Starbursting)
- **Code examples:** 12 (Python/SQL/CLI)

---

## Recommended Next Steps

1. **Formalize Mathematical Notation**
   - Document all formulas in LaTeX for academic rigor
   - Create reference implementation in Python
   - Build unit tests for compression/convergence algorithms

2. **Prototype Core Agno Workflow**
   - Implement single-round meeting with 2 agents
   - Validate Agno state management and async execution
   - Measure token costs and latency

3. **Design Redis + Postgres Schema**
   - Implement state management layer
   - Build migration scripts
   - Test context persistence and retrieval

4. **Build Typer CLI MVP**
   - Implement `create`, `run`, `status` commands
   - Add Rich progress display
   - Test interactive team selection

5. **Migrate Agent Pool to Letta**
   - Run migration script on sample agents
   - Validate memory persistence
   - Test cross-meeting recall

6. **Implement Compression Pipeline**
   - Integrate Qdrant for embeddings
   - Build clustering + merge logic
   - Benchmark compression ratio vs. quality

7. **Create Architecture Diagram**
   - Mermaid diagram of full system
   - Component interaction flows
   - Data flow visualization

8. **Write Technical Specification**
   - Use BMAD /tech-spec workflow
   - Detail each component's API
   - Define integration contracts

---

*Generated by BMAD Method Creative Intelligence*
*Session duration: Comprehensive brainstorming*
*Date: 2025-12-19*
