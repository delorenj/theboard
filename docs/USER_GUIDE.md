# TheBoard User Guide

**Version:** 1.0
**Last Updated:** 2025-12-30

Welcome to TheBoard! This guide will help you get started with multi-agent brainstorming simulations.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Example Scenarios](#example-scenarios)
4. [Advanced Features](#advanced-features)
5. [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database (via Docker)
- Redis cache (via Docker)
- RabbitMQ message broker (via Docker)
- LLM API key (OpenAI, Anthropic, or compatible)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd theboard
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Install TheBoard:**
   ```bash
   pip install -e .
   ```

5. **Verify installation:**
   ```bash
   board --help
   ```

---

## Core Concepts

### Meetings

A **meeting** is a multi-round brainstorming session where AI agents discuss a topic. Each meeting has:

- **Topic**: The brainstorming subject (10-500 characters)
- **Strategy**: Execution approach (sequential or greedy)
- **Max Rounds**: Number of discussion rounds (1-10)
- **Agents**: Participants with diverse expertise

### Agents

**Agents** are AI personas with specialized knowledge and perspectives. TheBoard includes:

- Software Engineer
- UX Designer
- Product Manager
- Data Scientist
- Marketing Strategist
- Business Analyst
- Security Expert
- DevOps Engineer
- Mobile Developer
- AI/ML Engineer

Agents can be auto-selected based on topic relevance or manually chosen.

### Strategies

**Sequential Strategy**: Agents respond in order, one at a time. Predictable and thorough.

**Greedy Strategy**: Agents selected dynamically based on contribution potential. Adaptive and efficient.

### Comments

**Comments** are structured insights extracted from agent responses:

- **Category**: Question, idea, concern, observation, or recommendation
- **Novelty Score**: Measure of uniqueness (0.0-1.0)
- **Text**: The actual comment content

### Convergence

**Convergence** detection stops meetings automatically when:

- Novelty scores drop below threshold (ideas becoming repetitive)
- Comment diversity decreases
- Compression ratio increases (context efficiently summarizable)

---

## Agent Pool Management

TheBoard includes a powerful agent pool management system that allows you to create, manage, and organize AI agents with diverse expertise.

### Seeding Initial Agent Pool

To get started with a pre-configured pool of 10 diverse agents, run:

```bash
python scripts/seed_agents.py
```

This creates agents including: backend-architect, frontend-specialist, devops-engineer, security-analyst, data-architect, ml-engineer, product-manager, ux-designer, qa-engineer, and tech-lead.

### Listing Agents

View all agents in the pool:

```bash
board agents list
```

View only active agents:

```bash
board agents list --active-only
```

### Creating Individual Agents

Create a custom agent:

```bash
board agents create \
  --name "cloud-architect" \
  --expertise "Expert in AWS, Azure, GCP cloud architecture and multi-cloud strategies" \
  --persona "Cost-conscious architect who balances scalability with budget constraints" \
  --background "10+ years managing cloud infrastructure for Fortune 500 companies" \
  --type plaintext \
  --model deepseek
```

### Viewing Agent Details

Get detailed information about a specific agent:

```bash
board agents show backend-architect
```

Or use the agent's UUID:

```bash
board agents show a0742ce4-4078-4a0e-b4d8-b457a89cdf79
```

### Importing Agents from File

Create multiple agents at once from a YAML or JSON file:

```bash
board agents import data/agents/initial_pool.yaml
```

**Example YAML format:**

```yaml
- name: devops-specialist
  expertise: |
    Expert in CI/CD, container orchestration, and infrastructure automation.
    Strong knowledge of Docker, Kubernetes, and cloud platforms.
  persona: |
    Automation-first mindset focused on reliability and reproducibility.
  background: |
    Built deployment pipelines for high-traffic applications.
  agent_type: plaintext
  default_model: deepseek
```

### Updating Agents

Modify an existing agent:

```bash
board agents update backend-architect \
  --expertise "Updated expertise description" \
  --model gpt-4-turbo
```

### Deactivating and Activating Agents

Temporarily deactivate an agent (soft delete):

```bash
board agents deactivate backend-architect
```

Reactivate a deactivated agent:

```bash
board agents activate backend-architect
```

### Deleting Agents

Permanently delete an agent (use with caution):

```bash
board agents delete backend-architect --force
```

Without `--force`, agents are only deactivated:

```bash
board agents delete backend-architect
```

### Agent Selection

When creating a meeting with `--auto-select`, agents are automatically selected based on keyword matching between the meeting topic and agent expertise. The selection algorithm:

1. Extracts keywords from the topic
2. Matches keywords against agent expertise, persona, and background
3. Ranks agents by relevance score
4. Selects top-matching agents

**Example:**

```bash
board create --topic "Design a scalable microservices backend with database sharding and caching"
```

This will automatically select agents like backend-architect, data-architect, and devops-engineer based on keywords: "microservices", "backend", "database", "caching".

---

## Example Scenarios

### Scenario 1: Simple Sequential Meeting

**Goal**: Brainstorm a mobile app feature with 5 agents using sequential strategy.

**Step 1: Create the meeting**
```bash
board create \
  --topic "Design a feature for tracking daily water intake in a health app" \
  --strategy sequential \
  --max-rounds 5 \
  --agent-count 5 \
  --auto-select
```

**Expected output:**
```
✓ Meeting Created!

Meeting ID: 12345678-1234-1234-1234-123456789012
Topic: Design a feature for tracking daily water intake in a health app
Strategy: sequential
Max Rounds: 5

Run the meeting with: board run 12345678-1234-1234-1234-123456789012
```

**Step 2: Run the meeting**
```bash
board run 12345678-1234-1234-1234-123456789012
```

**Expected output:**
```
Meeting Progress (live updates):
┌───────┬──────────┬──────────┬─────────────┬──────────────┐
│ Round │ Status   │ Comments │ Avg Novelty │ Context Size │
├───────┼──────────┼──────────┼─────────────┼──────────────┤
│ 3/5   │ running  │ 14       │ 0.65        │ 8,234        │
└───────┴──────────┴──────────┴─────────────┴──────────────┘

✓ Meeting Completed!

Total Rounds: 4
Total Comments: 23
Total Cost: $0.0485
Status: completed
Stopping Reason: Convergence detected

View details with: board status 12345678-1234-1234-1234-123456789012
```

**Step 3: View detailed status**
```bash
board status 12345678-1234-1234-1234-123456789012
```

**Step 4: Export results**
```bash
board export 12345678-1234-1234-1234-123456789012 \
  --format markdown \
  --output water-intake-brainstorm.md
```

**Result**: A comprehensive markdown report with all agent responses, categorized comments, and convergence metrics.

---

### Scenario 2: Greedy Strategy with Human Steering

**Goal**: Use greedy strategy with human-in-the-loop to steer discussion.

**Step 1: Create meeting with greedy strategy**
```bash
board create \
  --topic "Brainstorm security vulnerabilities in a payment processing system" \
  --strategy greedy \
  --max-rounds 8 \
  --agent-count 6
```

**Step 2: Run interactively**
```bash
board run <meeting-id> --interactive
```

**During execution**, you'll be prompted:

```
Round 2 complete. What would you like to do?

Options:
1. Continue to next round
2. Pause and review
3. Steer discussion (provide guidance)
4. Stop meeting

Choice: 3

Enter steering message:
> Focus on SQL injection and XSS vulnerabilities specifically.

✓ Steering applied. Agents will consider this in round 3.
```

**Step 3: Export results as HTML**
```bash
board export <meeting-id> --format html --output security-analysis.html
```

**Result**: Styled HTML report with color-coded sections, perfect for sharing with stakeholders.

---

### Scenario 3: Rerunning and Forking Meetings

**Goal**: Iterate on previous meeting results.

**Step 1: List recent meetings**
```bash
board run
# Shows interactive selector
```

**Output:**
```
Recent Meetings:

1. Design a feature for tracking daily water intake... (completed) - $0.05 - 4 rounds
2. Brainstorm security vulnerabilities in payment... (completed) - $0.12 - 6 rounds
3. Create a gamification system for fitness app (created) - Ready to run

Select meeting: 2
```

**Step 2: Fork the security meeting**
```bash
board run --fork <security-meeting-id>
```

**Expected behavior:**
- Creates a new meeting with same topic and settings
- Preserves original meeting history
- New meeting starts fresh with round 1

**Step 3: Or rerun to replace**
```bash
board run --rerun <security-meeting-id>
```

**Expected behavior:**
- Resets meeting to round 1
- Clears all previous responses and comments
- Overwrites original meeting data (use with caution!)

**Step 4: Use --last for quick access**
```bash
board run --last
# Runs most recent meeting automatically
```

---

## Advanced Features

### Custom Agent Selection

Instead of auto-select, manually choose agents:

```bash
board create \
  --topic "Your topic here" \
  --auto-select=false
# Then interactively select agents from available pool
```

### Model Override

Use a different LLM model for specific meetings:

```bash
board create \
  --topic "Your topic here" \
  --model "gpt-4-turbo"
```

### Template-Based Export

Use custom Jinja2 templates for export:

```bash
board export <meeting-id> \
  --format template \
  --template "executive-summary.j2" \
  --output summary.md
```

**Template location**: `src/theboard/templates/exports/`

---

## Best Practices

### 1. Topic Formulation

**Good topics:**
- ✅ "Design a feature for real-time collaboration in a document editor"
- ✅ "Identify potential bottlenecks in our microservices architecture"
- ✅ "Brainstorm ways to improve user onboarding completion rate"

**Poor topics:**
- ❌ "Help" (too vague)
- ❌ "Fix bug" (too narrow)
- ❌ "Design, develop, test, and deploy a complete enterprise system" (too broad)

### 2. Agent Count

- **3-5 agents**: Quick explorations (5-15 minutes)
- **5-7 agents**: Balanced discussions (15-30 minutes)
- **8-10 agents**: Deep dives (30-60 minutes)

More agents = more diverse perspectives but higher cost and longer runtime.

### 3. Round Planning

- **3 rounds**: Quick gut-check
- **5 rounds**: Standard exploration
- **8-10 rounds**: Deep analysis

Convergence usually happens around round 4-6. Meetings stop early if detected.

### 4. Strategy Selection

**Use Sequential when:**
- You want predictable execution
- Topic benefits from structured input
- Cost is not a primary concern

**Use Greedy when:**
- You want adaptive agent selection
- Topic is exploratory
- You want to minimize cost (unused agents don't run)

### 5. Cost Management

- **Hybrid model strategy** (default): Uses cheaper models for routine tasks, premium for complex reasoning. Reduces costs by 60%+.
- **Lazy compression**: Only compresses when context exceeds 10K characters.
- **Delta propagation**: Agents receive only new comments since their last turn.

**Estimated costs** (with default settings):
- 5 agents, 5 rounds, sequential: $0.05-0.15
- 8 agents, 8 rounds, greedy: $0.10-0.25

### 6. Export Formats

- **Markdown**: Best for documentation, version control
- **JSON**: Best for programmatic processing, integrations
- **HTML**: Best for presentations, sharing with non-technical stakeholders
- **Template**: Best for custom report formats

---

## Next Steps

- **Troubleshooting**: See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **Developer Docs**: See [DEVELOPER.md](./DEVELOPER.md)
- **API Reference**: Check code documentation in `src/theboard/`

---

**Questions or issues?** Open an issue on GitHub or contact support.
