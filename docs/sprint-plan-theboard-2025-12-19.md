# Sprint Plan: TheBoard

**Project:** TheBoard - Multi-Agent Brainstorming Simulation System
**Date:** 2025-12-19
**Version:** 1.0
**Status:** Planning
**Author:** Scrum Master (BMAD Method)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Sprint Overview](#sprint-overview)
3. [Story Point Estimates](#story-point-estimates)
4. [Sprint Iterations](#sprint-iterations)
5. [Dependency Graph](#dependency-graph)
6. [Risk Analysis](#risk-analysis)
7. [Velocity Planning](#velocity-planning)
8. [Sprint Ceremonies](#sprint-ceremonies)

---

## 1. Executive Summary

This sprint plan organizes the 17 implementation stories from the tech spec into 6 actionable sprint iterations. The plan accounts for:
- Story dependencies (critical path: Foundation → Multi-Agent → Compression → Convergence)
- Complexity estimates (story points based on size, risk, unknowns)
- Team velocity assumptions (targeting 13-21 points per 2-week sprint)
- Milestone-driven planning (each sprint delivers a testable milestone)

**Total Effort:** 89 story points across 6 sprints (12-15 weeks at recommended velocity)

**Key Constraints:**
- Sequential dependencies require Foundation → Intelligence Layer completion before advanced features
- Agno/Letta integration carries technical risk (early prototyping critical)
- Token cost optimization is cross-cutting (impacts multiple sprints)

---

## 2. Sprint Overview

### Sprint Structure
- **Duration:** 2 weeks per sprint (10 working days)
- **Velocity Target:** 13-21 story points per sprint
- **Team Size:** 1-2 developers (solo or pair)
- **Sprint Ceremonies:**
  - Sprint Planning: Day 1 (2-3 hours)
  - Daily Standup: Daily (15 minutes)
  - Sprint Review: Last day (1 hour)
  - Sprint Retro: Last day (30 minutes)

### Milestone Mapping
- **Sprint 1:** Milestone 1 (MVP Foundation Complete)
- **Sprint 2:** Milestone 2 (Multi-Agent Orchestration Working)
- **Sprint 3:** Milestone 3 (Intelligence Layer Complete)
- **Sprint 4:** Milestone 4 (Advanced Features Working)
- **Sprint 5:** Milestone 5 Partial (Letta Integration)
- **Sprint 6:** Milestone 5 Complete (Production-Ready v1)

---

## 3. Story Point Estimates

**Estimation Scale:** Fibonacci (1, 2, 3, 5, 8, 13, 21)

**Estimation Factors:**
1. **Size:** Lines of code, number of modules
2. **Complexity:** Algorithmic difficulty, integration points
3. **Risk:** Unknowns, dependencies on external frameworks
4. **Testing:** Test coverage requirements, edge cases

### Story Point Breakdown

| Story ID | Story Name | Points | Complexity | Risk | Rationale |
|----------|-----------|--------|-----------|------|-----------|
| **Phase 1: Foundation** | | **26 pts** | | | |
| 1 | Project Setup & Data Layer | 8 | Medium | Low | Docker setup, schema design, migrations |
| 2 | Basic CLI Structure | 3 | Low | Low | Typer skeleton, minimal commands |
| 3 | Agno Integration & Simple Agent | 8 | High | High | **Risk:** Early-stage framework, learning curve |
| 4 | Notetaker Agent Implementation | 7 | Medium | Medium | LLM extraction, structured output validation |
| **Phase 2: Multi-Agent Orchestration** | | **20 pts** | | | |
| 5 | Meeting Coordinator Workflow | 8 | High | Medium | Agno workflow, round management, state tracking |
| 6 | Agent Pool Management | 5 | Medium | Low | File parsing, auto-selection logic |
| 7 | Context Management | 7 | Medium | Medium | Redis caching, delta propagation, size tracking |
| **Phase 3: Compression & Convergence** | | **18 pts** | | | |
| 8 | Embedding Infrastructure | 5 | Medium | Low | Qdrant setup, embedding pipeline |
| 9 | Compressor Agent | 8 | High | Medium | Three-tier compression, clustering, LLM merge |
| 10 | Convergence Detection | 5 | Medium | Low | Novelty calculation, threshold checking |
| **Phase 4: Advanced Features** | | **13 pts** | | | |
| 11 | Greedy Execution Strategy | 5 | Medium | Low | Parallel execution with asyncio.gather |
| 12 | Event-Driven Human-in-Loop | 5 | Medium | Medium | RabbitMQ events, CLI consumer, pause/resume |
| 13 | Hybrid Model Strategy | 3 | Low | Low | Engagement scoring, model promotion logic |
| **Phase 5: Letta Integration & Polish** | | **12 pts** | | | |
| 14 | Letta Agent Migration | 5 | Medium | High | **Risk:** Integration complexity, migration script |
| 15 | Export & Artifact Generation | 3 | Low | Low | Template-based generation (markdown, JSON, HTML) |
| 16 | Performance Optimization | 2 | Low | Low | Lazy compression, selective activation |
| 17 | CLI Polish & Documentation | 2 | Low | Low | Help text, user guide, troubleshooting |

**Total:** 89 story points

---

## 4. Sprint Iterations

### Sprint 1: Foundation & MVP Core (26 points)
**Goal:** Establish foundational infrastructure and validate single-agent execution end-to-end

**Duration:** 2 weeks (10 days)

**Stories:**
1. **Story 1: Project Setup & Data Layer** (8 pts)
   - Initialize Python project with uv
   - Docker Compose: Postgres, Redis, RabbitMQ, Qdrant
   - SQLAlchemy models: meetings, agents, responses, comments
   - Alembic migrations
   - Redis connection manager
   - Test database connectivity

2. **Story 2: Basic CLI Structure** (3 pts)
   - Typer app skeleton
   - Commands: create, run, status, export (stubs)
   - Rich formatting setup
   - Basic `board create` with minimal inputs
   - Basic `board status` display

3. **Story 3: Agno Integration & Simple Agent** (8 pts)
   - Install and configure Agno framework
   - DomainExpertAgent as Agno skill
   - Single-agent, single-round execution
   - Claude Sonnet integration for LLM calls
   - Test Agno workflow execution

4. **Story 4: Notetaker Agent Implementation** (7 pts)
   - NotetakerAgent with structured extraction
   - Comment Pydantic model
   - Extract comments using Claude Sonnet
   - Store comments in Postgres
   - Display comments via CLI

**Acceptance Criteria:**
- [ ] Docker Compose brings up all services (Postgres, Redis, RabbitMQ, Qdrant)
- [ ] `board create` creates a meeting with 1 agent, stores to database
- [ ] `board run` executes 1-agent, 1-round meeting
- [ ] Notetaker extracts comments from agent response (>90% extraction rate)
- [ ] `board status` displays meeting state (round, agent, comments)
- [ ] All unit tests pass (>70% coverage for core logic)

**Sprint Deliverable:** Working MVP with single-agent execution and comment extraction

**Real-Life User Story:**
> Sarah, a tech lead, needs quick feedback on a microservices architecture proposal. She runs:
> ```bash
> board create --topic "Microservices vs Monolith for our e-commerce platform"
> board run --agents "backend-architect" --rounds 1
> board status
> ```
>
> The backend architect agent provides a structured response analyzing the trade-offs. The notetaker extracts 8 key comments including technical decisions (microservices complexity), risks (operational overhead), and implementation details (service boundaries). Sarah can see these comments in the CLI output and they're stored in the database for future reference.
>
> **What Sarah Can Do Now:**
> - Create a meeting with a topic
> - Execute a single-agent, single-round brainstorming session
> - View extracted comments categorized by type (technical_decision, risk, implementation_detail)
> - Access stored meeting data from Postgres

**Risks:**
- Agno learning curve may slow Story 3
- LLM API rate limits during testing

**Mitigation:**
- Allocate extra time for Agno prototyping (2-3 days)
- Use mocked LLM responses for unit tests

---

### Sprint 2: Multi-Agent Orchestration (20 points)
**Goal:** Enable multi-agent, multi-round meetings with context accumulation

**Duration:** 2 weeks (10 days)

**Stories:**
5. **Story 5: Meeting Coordinator Workflow** (8 pts)
   - TheboardMeeting as Agno workflow
   - Round management (loop, counter, state tracking)
   - Sequential strategy execution
   - Meeting state in Redis
   - RabbitMQ event emission (basic events)
   - Test multi-round execution with 2 agents

6. **Story 6: Agent Pool Management** (5 pts)
   - Agent pool loader for plaintext descriptions
   - Parse agent files (name, expertise, persona)
   - AgentRegistry with in-memory index
   - Auto-select team using topic keywords
   - Manual agent selection via CLI interactive prompt

7. **Story 7: Context Management** (7 pts)
   - ContextManager for cumulative context building
   - Context persistence to Redis (TTL)
   - Context size tracking and warnings
   - Test multi-round context accumulation
   - Archive context history to Postgres

**Acceptance Criteria:**
- [ ] `board create` supports auto-select or manual agent team composition
- [ ] `board run` executes 5-agent, 5-round meeting (sequential strategy)
- [ ] Context accumulates across rounds (Xr = Xr-1 + new comments)
- [ ] Meeting state persisted to Redis (current_round, current_agent)
- [ ] Agent pool auto-selects relevant agents based on topic (>80% relevance)
- [ ] Integration tests validate multi-round flow

**Sprint Deliverable:** Multi-agent orchestration with context accumulation

**Real-Life User Story:**
> Marcus is planning a new payment processing system. He needs diverse perspectives from security, backend, and frontend experts. He runs:
> ```bash
> board create --topic "Payment processing system with PCI compliance"
> board run --auto-select --agents 5 --rounds 3 --strategy sequential
> board status --show-context
> ```
>
> The system auto-selects a team: security-engineer, backend-architect, payment-specialist, compliance-officer, and frontend-developer. Over 3 rounds, each agent builds on previous comments:
> - **Round 1:** Initial ideas (security suggests tokenization, backend proposes API design)
> - **Round 2:** Agents respond to each other (compliance validates tokenization approach, frontend asks about error handling)
> - **Round 3:** Refined consensus emerges (team aligns on vault-based tokenization with dual-write pattern)
>
> Marcus sees the context grow from 2K tokens (Round 1) to 8K tokens (Round 3), with cumulative context preserving all prior discussion. The agent pool automatically selected the most relevant experts based on the payment/compliance keywords in the topic.
>
> **What Marcus Can Do Now:**
> - Auto-select relevant agents based on meeting topic
> - Run multi-agent brainstorming with sequential turn-taking
> - Execute multiple rounds with context accumulation
> - Monitor context size and round progression via Redis state
> - See how agents build on each other's ideas across rounds

**Risks:**
- Context explosion without compression
- Turn ordering complexity in greedy strategy (deferred to Sprint 4)

**Mitigation:**
- Implement basic context size alerts (threshold: 15K tokens)
- Focus on sequential strategy first (greedy deferred)

---

### Sprint 3: Compression & Convergence (18 points)
**Goal:** Implement intelligent compression and automatic convergence detection

**Duration:** 2 weeks (10 days)

**Stories:**
8. **Story 8: Embedding Infrastructure** (5 pts)
   - Qdrant vector database setup in Docker Compose
   - Comment embedding pipeline (sentence-transformers)
   - Cosine similarity computation
   - Test embedding quality (threshold tuning)
   - Batch embedding processing

9. **Story 9: Compressor Agent** (8 pts)
   - CompressorAgent with three-tier strategy
   - Tier 1: Similarity-based clustering (Qdrant)
   - Tier 2: LLM semantic merge (Claude Sonnet)
   - Tier 3: Outlier removal (support < 2)
   - Track compression metrics (ratio, counts)
   - Test compression quality vs. information retention

10. **Story 10: Convergence Detection** (5 pts)
    - Novelty score calculation using embeddings
    - Convergence threshold checking (default: 0.2)
    - Test stopping criteria
    - Persist convergence metrics per round (Postgres)
    - Emit convergence events (RabbitMQ)
    - Display convergence reason to user

**Acceptance Criteria:**
- [ ] Compressor reduces comment count by 40-60% (measured)
- [ ] Compression preserves information quality (spot checks)
- [ ] Convergence detection stops meeting when novelty < 0.2 for 2 consecutive rounds
- [ ] Context size stays under 15K tokens for 5-round, 5-agent meeting
- [ ] Convergence metrics tracked per round in database
- [ ] Full audit trail maintained (all responses pre-compression stored)

**Sprint Deliverable:** Intelligent compression and automatic convergence

**Real-Life User Story:**
> Lisa is brainstorming a complex data migration strategy with 7 domain experts. Without compression, the discussion would quickly exceed token limits. She runs:
> ```bash
> board create --topic "PostgreSQL to Cassandra migration for 100TB dataset"
> board run --auto-select --agents 7 --max-rounds 5
> board status --show-compression
> ```
>
> **What Happens:**
> - **Round 1:** 7 agents generate 42 comments (6 each). Context: 12K tokens.
> - **Compression triggered:** Compressor clusters similar ideas (e.g., "partition by date" mentioned by 4 agents → merged). Reduces to 18 unique comments. Context: 5K tokens (58% reduction).
> - **Round 2:** Agents respond with 35 new comments. Before compression: 17K tokens. After compression: 8K tokens.
> - **Round 3:** Only 12 new ideas emerge. Novelty score: 0.18 (most ideas are refinements).
> - **Round 4:** Novelty drops to 0.12 (very few new ideas). System detects convergence after 2 consecutive rounds below 0.2 threshold.
> - **Meeting auto-stops** with message: "Convergence detected after Round 4 (novelty: 0.12, threshold: 0.20)"
>
> Lisa reviews the compressed output: 45 unique ideas instead of 100+ raw comments, with full audit trail preserved in database. The compression identified that "incremental migration with dual-write" was the emerging consensus (mentioned by 5/7 agents).
>
> **What Lisa Can Do Now:**
> - Run large brainstorming sessions without hitting token limits
> - Automatically compress similar ideas while preserving unique insights
> - Detect when discussion has converged (no new ideas emerging)
> - Have meetings stop automatically when productive discussion ends
> - Review compression metrics (ratio, cluster counts) per round
> - Access full audit trail (all pre-compression responses in database)

**Risks:**
- Compression may lose critical information
- Convergence may never be reached (infinite loops)

**Mitigation:**
- Maintain full audit trail for quality validation
- Set max_rounds hard limit (default: 5)
- Human-in-loop override (deferred to Sprint 4)

---

### Sprint 4: Advanced Features (13 points)
**Goal:** Implement greedy strategy, human-in-loop, and hybrid model optimization

**Duration:** 2 weeks (10 days)

**Stories:**
11. **Story 11: Greedy Execution Strategy** (5 pts)
    - Parallel agent responses using asyncio.gather
    - Comment-response phase (each agent responds to others)
    - Token efficiency tracking (n² cost)
    - Compare performance: greedy vs. sequential
    - Test convergence behavior with greedy

12. **Story 12: Event-Driven Human-in-Loop** (5 pts)
    - RabbitMQ event publishing (5 event types)
    - CLI event consumer with async listener
    - Interactive prompts for human steering
    - Meeting pause/resume capability
    - Timeout defaults (5min auto-continue)

13. **Story 13: Hybrid Model Strategy** (3 pts)
    - Engagement metric calculation (weighted: peer_references, novelty, comment_count)
    - Dynamic model promotion logic (top 20% → Opus after round 1)
    - Track per-agent costs and token usage
    - Test cost savings vs. quality (A/B comparison)

**Acceptance Criteria:**
- [ ] Greedy strategy executes with parallel responses and comment-response phase
- [ ] Human-in-loop can pause meeting, provide input, and resume
- [ ] Hybrid model strategy reduces cost by >60% compared to all-Opus baseline
- [ ] Quality maintained with hybrid strategy (validated via A/B testing)
- [ ] Event consumer handles timeout defaults (auto-continue after 5min)
- [ ] Performance benchmarks: greedy vs. sequential

**Sprint Deliverable:** Advanced orchestration features and cost optimization

**Real-Life User Story:**
> Ahmed is exploring a critical API design decision with a tight budget. He needs fast iteration and wants to guide the discussion if needed. He runs:
> ```bash
> board create --topic "REST vs GraphQL vs gRPC for mobile API"
> board run --auto-select --agents 6 --strategy greedy --hybrid-models
> ```
>
> **Greedy Strategy in Action:**
> - **Round 1:** All 6 agents respond in parallel (execution time: 8s vs 45s sequential)
> - **Comment-Response Phase:** Each agent sees others' ideas and responds (36 total responses in this phase)
> - Agents debate: mobile-dev prefers GraphQL for flexibility, backend-arch counters with REST simplicity
>
> **Hybrid Model Optimization:**
> - Round 1: All agents use DeepSeek (cost: $0.12)
> - After Round 1: System calculates engagement scores
> - Top 2 agents (mobile-dev, api-specialist) promoted to Opus 4.5 based on peer references
> - Remaining 4 agents continue with DeepSeek
> - **Total cost: $0.89** vs all-Opus baseline: $2.34 (62% savings)
>
> **Human-in-the-Loop:**
> - After Round 2, Ahmed receives prompt: "Meeting paused. Options: continue, modify context, stop"
> - He adds steering: "Focus on offline-first mobile requirements"
> - Agents incorporate this constraint in Round 3
> - Discussion shifts toward GraphQL with Apollo Client cache
>
> Ahmed reviews the final recommendation with confidence: the hybrid strategy maintained quality (top performers got premium models) while staying under budget. The greedy strategy surfaced debates faster by letting agents respond to each other immediately.
>
> **What Ahmed Can Do Now:**
> - Run parallel brainstorming (greedy strategy) for faster iteration
> - Optimize costs with hybrid models (DeepSeek → Opus promotion)
> - Pause meetings and inject human guidance when needed
> - Compare performance: greedy vs sequential execution time
> - Track per-agent costs and engagement metrics
> - Achieve 60%+ cost reduction while maintaining quality

**Risks:**
- Greedy strategy may have unexpected token cost explosion
- Hybrid model quality degradation

**Mitigation:**
- Budget limits per meeting (configurable)
- A/B testing to validate hybrid quality
- Configurable promotion thresholds

---

### Sprint 5: Letta Integration & Export (10 points)
**Goal:** Migrate to Letta agents with persistent memory and implement artifact export

**Duration:** 2 weeks (10 days)

**Stories:**
14. **Story 14: Letta Agent Migration** (5 pts)
    - Plaintext → Letta migration script
    - Letta memory persistence to agent_memory table
    - Cross-meeting memory recall (previous_meetings, learned_patterns)
    - Test memory-enhanced responses
    - Migrate sample agents from pool

15. **Story 15: Export & Artifact Generation** (3 pts)
    - Markdown export with formatting
    - JSON export with structured schema
    - HTML export with CSS styling
    - Test artifact quality and readability
    - Support custom export templates

16. **Story 16: Performance Optimization** (2 pts)
    - Lazy compression (only when context > 10K chars)
    - Delta propagation (agents receive only new comments)
    - Optimize Redis caching strategy
    - Selective agent activation (only engaged agents in round 2+)

**Acceptance Criteria:**
- [ ] Letta agents recall past meeting outcomes in new meetings
- [ ] Memory retrieval works with 100+ meeting history (latency <1s)
- [ ] `board export` generates readable markdown, JSON, and HTML artifacts
- [ ] Lazy compression triggered only when context exceeds threshold
- [ ] Delta propagation reduces token usage by >30%
- [ ] Optimization settings documented

**Sprint Deliverable:** Letta integration with memory and optimized export

**Real-Life User Story:**
> Priya is running a series of architecture discussions over several weeks. She needs agents to remember past decisions and build institutional knowledge. She also wants shareable artifacts for her team. She runs:
> ```bash
> # Week 1: Initial discussion
> board create --topic "Microservices deployment strategy"
> board run --auto-select --letta-agents
> board export --format markdown --output deployment-strategy-v1.md
>
> # Week 2: Follow-up with memory
> board create --topic "Microservices observability and monitoring"
> board run --auto-select --letta-agents
> ```
>
> **Letta Memory in Action:**
> - **Week 1 Meeting:** sre-specialist suggests Kubernetes with Istio service mesh
> - Memory stored: `deployment_pattern: "k8s + istio"` in agent memory
> - **Week 2 Meeting:** Same SRE agent recalls: *"Based on our previous discussion about Kubernetes deployment, I recommend Prometheus + Grafana for observability, which integrates well with Istio's telemetry."*
> - Cross-meeting recall improves coherence: agents don't contradict past decisions
>
> **Performance Optimizations:**
> - Lazy compression: Only triggered when context exceeds 10K tokens (saved 2 compression cycles)
> - Delta propagation: Agents receive only new comments since their last turn (30% token reduction)
> - Selective activation: Only agents with engagement >0.3 participate in later rounds (filtered out 2 low-engagement agents)
>
> **Export Artifacts:**
> Priya exports the final results in multiple formats:
> - **Markdown:** Formatted with headings, bullet points, and decision summary - perfect for Confluence
> - **JSON:** Structured data with all comments, rounds, and metrics - feeds into internal docs generator
> - **HTML:** Styled report with CSS - shareable via email to stakeholders
>
> **What Priya Can Do Now:**
> - Run multi-week discussion series with agent memory continuity
> - Agents recall past meeting outcomes and build on them
> - Export meeting results in markdown, JSON, or HTML formats
> - Optimize performance with lazy compression and delta propagation
> - Reduce token costs by selectively activating engaged agents
> - Query agent memory for past decisions and patterns
> - Share formatted artifacts with team members

**Risks:**
- Letta integration complexity higher than expected
- Memory retrieval performance degradation at scale

**Mitigation:**
- Hybrid support: both plaintext and Letta agents
- Vector similarity search for memory recall (Qdrant)
- Performance testing with 100+ meetings

---

### Sprint 6: CLI Polish & Production Readiness (2 points)
**Goal:** Finalize CLI experience, complete documentation, and achieve production-ready state

**Duration:** 1 week (5 days)

**Stories:**
17. **Story 17: CLI Polish & Documentation** (2 pts)
    - Enhance `board export` with format options
    - Implement live progress streaming (Rich.Live)
    - Comprehensive help text for all commands
    - User guide with 3+ example scenarios
    - Troubleshooting guide for common issues
    - Developer documentation (API reference)

**Additional Tasks (Not Stories):**
- Final integration testing (end-to-end scenarios)
- Performance benchmarking (validate NFR compliance)
- Security audit (credential management, input validation)
- Deployment documentation (Docker Compose setup)
- CI/CD pipeline finalization

**Acceptance Criteria:**
- [ ] All 17 stories implemented and tested
- [ ] Unit test coverage >70% for core logic
- [ ] Integration tests validate 3+ end-to-end scenarios
- [ ] Type checking passes (mypy --strict)
- [ ] Linting passes (ruff)
- [ ] Docker Compose brings up full stack in <60s
- [ ] User guide with examples published
- [ ] Troubleshooting guide for common issues
- [ ] API documentation complete
- [ ] All NFRs validated (performance, security, reliability, observability)

**Sprint Deliverable:** Production-ready v1 with complete documentation

**Real-Life User Story:**
> The entire engineering team at TechCorp is now using TheBoard for architecture decisions. Jordan, a new developer, joins and needs to get productive quickly. They run:
> ```bash
> # First time setup
> board --help
> # Shows comprehensive help with all commands and examples
>
> # Follow getting started guide
> board create --topic "Should we adopt TypeScript for our React components?"
> board run --interactive
> ```
>
> **Production-Ready Experience:**
> - **Live Progress Streaming:** Jordan sees real-time updates during execution:
>   ```
>   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
>   ┃ Round 1/5 │ Agent: frontend-architect │ ┃
>   ┃ Comments: 7 │ Novelty: 1.00 │ Tokens: 2.1K ┃
>   ┃ Status: Extracting comments... ⣾          ┃
>   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
>   ```
>
> - **Comprehensive Help:** Every command has detailed help text:
>   ```bash
>   board create --help
>   # Shows: topic, agent selection, strategy options, examples
>   board export --help
>   # Shows: format options (markdown/json/html), output paths, templates
>   ```
>
> - **Example Scenarios in Docs:**
>   - Simple meeting (5 agents, sequential)
>   - Greedy strategy for fast iteration
>   - Human-in-loop steering for critical decisions
>
> **Troubleshooting Support:**
> Jordan encounters an error: "LLM API rate limit exceeded"
> - Checks troubleshooting guide in docs/troubleshooting.md
> - Finds solution: Configure `--rate-limit-delay 2000` or use `--mock-mode` for testing
> - Issue resolved in 2 minutes
>
> **Developer Experience:**
> - Docker Compose brings up full stack in 45 seconds
> - All NFRs validated: meetings execute in <5 minutes, token costs optimized, context <15K
> - API documentation shows Pydantic models, core services, integration patterns
> - Type checking (mypy --strict) and linting (ruff) pass
> - 72% unit test coverage across core logic
>
> **What Jordan (and the Team) Can Do Now:**
> - Onboard new users quickly with comprehensive docs and examples
> - Monitor meeting progress in real-time with Rich live updates
> - Troubleshoot common issues using the troubleshooting guide
> - Deploy to production with Docker Compose in <1 minute
> - Extend the system using developer API documentation
> - Run with confidence: all 17 stories implemented and tested
> - Export and share results in team-preferred formats
> - Trust system reliability: NFRs validated, monitoring in place

**Risks:**
- Scope creep (additional features requested)
- Documentation takes longer than estimated

**Mitigation:**
- Strict scope adherence (defer new features to v2)
- Allocate buffer time for documentation (3 days)

---

## 5. Dependency Graph

### Critical Path
```
Story 1 (Project Setup)
  ↓
Story 2 (CLI Structure)
  ↓
Story 3 (Agno Integration) ← CRITICAL: Validates framework choice
  ↓
Story 4 (Notetaker)
  ↓
Story 5 (Meeting Coordinator)
  ↓
Story 6 (Agent Pool) + Story 7 (Context Manager) ← Can parallelize
  ↓
Story 8 (Embedding Infrastructure)
  ↓
Story 9 (Compressor) ← CRITICAL: Core intelligence
  ↓
Story 10 (Convergence)
  ↓
Story 11, 12, 13 (Advanced Features) ← Can parallelize
  ↓
Story 14 (Letta Migration)
  ↓
Story 15, 16 (Export & Optimization) ← Can parallelize
  ↓
Story 17 (Polish & Docs)
```

### Parallelization Opportunities

**Sprint 2:**
- Story 6 (Agent Pool) and Story 7 (Context Manager) can be developed in parallel
- Both depend on Story 5 (Meeting Coordinator)

**Sprint 4:**
- Story 11 (Greedy), Story 12 (Human-in-Loop), Story 13 (Hybrid Models) can be developed in parallel
- All depend on Sprint 3 completion

**Sprint 5:**
- Story 15 (Export) and Story 16 (Optimization) can be developed in parallel
- Both depend on Story 14 (Letta Migration)

### Dependency Constraints

**Hard Dependencies (Sequential):**
1. Story 1 must complete before Story 2 (database required for CLI)
2. Story 3 must complete before Story 5 (Agno required for workflows)
3. Story 8 must complete before Story 9 (embeddings required for compression)
4. Story 9 must complete before Story 10 (compression required for convergence)

**Soft Dependencies (Recommended Sequential):**
1. Story 4 before Story 5 (notetaker used in orchestration, but can mock)
2. Story 7 before Story 9 (context manager used in compression, but can mock)

---

## 6. Risk Analysis

### High-Risk Stories

#### Story 3: Agno Integration & Simple Agent (8 pts, High Risk)
**Risk:** Early-stage framework, learning curve, potential limitations

**Impact:** CRITICAL - Agno is core to orchestration layer

**Probability:** Medium (40%)

**Mitigation:**
- Allocate extra time for prototyping (2-3 days)
- Fallback plan: direct async/await orchestration
- Abstract orchestration layer (loose coupling)
- Early spike to validate Agno fit

**Contingency:**
- If Agno proves inadequate by Sprint 1 end, pivot to custom orchestration
- Budget 1 additional sprint for custom implementation

---

#### Story 9: Compressor Agent (8 pts, Medium Risk)
**Risk:** Compression may lose critical information, algorithm complexity

**Impact:** HIGH - Core intelligence feature, affects context management

**Probability:** Medium (30%)

**Mitigation:**
- Full audit trail for quality validation
- Compression quality metrics (spot checks)
- Tunable compression thresholds
- Human review option for critical meetings

**Contingency:**
- If compression quality insufficient, reduce compression ratio (accept higher token cost)
- Add manual compression override

---

#### Story 14: Letta Agent Migration (5 pts, High Risk)
**Risk:** Integration complexity, migration effort underestimated

**Impact:** MEDIUM - Letta provides memory, but plaintext fallback exists

**Probability:** Medium (30%)

**Mitigation:**
- Hybrid support (both plaintext and Letta)
- Incremental migration (not all-at-once)
- Comprehensive migration testing
- Migration script for automation

**Contingency:**
- If Letta integration too complex, defer to v2
- Use plaintext agents for v1, add memory in v2

---

### Medium-Risk Stories

#### Story 5: Meeting Coordinator Workflow (8 pts, Medium Risk)
**Risk:** State management complexity, round orchestration edge cases

**Mitigation:**
- Leverage Agno state management (reduces custom code)
- Comprehensive integration tests
- State machine visualization

---

#### Story 10: Convergence Detection (5 pts, Low-Medium Risk)
**Risk:** Convergence may never be reached (infinite loops)

**Mitigation:**
- Set max_rounds hard limit (default: 5)
- Human-in-loop override
- Convergence monitoring and alerts

---

### Risk Burn-Down Plan

**Sprint 1:**
- Validate Agno framework choice (Story 3 spike)
- Decision point: Continue with Agno or pivot to custom (end of Sprint 1)

**Sprint 3:**
- Validate compression quality (Story 9 testing)
- Compression ratio vs. information retention trade-off analysis

**Sprint 5:**
- Validate Letta integration (Story 14 spike)
- Decision point: Full Letta or hybrid plaintext support

---

## 7. Velocity Planning

### Velocity Assumptions

**Team Composition:**
- 1-2 developers (solo or pair)
- Experienced Python developer with LLM/AI background
- Familiarity with async/await, Docker, Postgres

**Velocity Targets:**
- **Conservative:** 13 points per 2-week sprint
- **Moderate:** 16 points per 2-week sprint
- **Optimistic:** 21 points per 2-week sprint

**Recommended:** 16 points per sprint (moderate velocity)

### Sprint Capacity Planning

| Sprint | Points | Velocity | Duration | Risk Buffer |
|--------|--------|----------|----------|-------------|
| Sprint 1 | 26 | 16 | 2 weeks | **Overflow:** Story 3 may carry over |
| Sprint 2 | 20 | 16 | 2 weeks | On track |
| Sprint 3 | 18 | 16 | 2 weeks | On track |
| Sprint 4 | 13 | 16 | 2 weeks | **Underutilized:** Can pull from Sprint 5 |
| Sprint 5 | 10 | 16 | 2 weeks | **Underutilized:** Can pull Story 17 |
| Sprint 6 | 2 | 16 | 1 week | **Light sprint:** Focus on polish |

### Recommended Adjustments

**Option 1: Extend Sprint 1 (Conservative)**
- Sprint 1: 3 weeks (26 pts at 13 pts/sprint velocity)
- Total timeline: 13 weeks

**Option 2: Rebalance Sprints (Recommended)**
- Sprint 1: 2 weeks (16 pts - defer Story 4 to Sprint 2)
- Sprint 2: 2 weeks (24 pts - add Story 4)
- Sprint 3: 2 weeks (18 pts)
- Sprint 4: 2 weeks (13 pts)
- Sprint 5: 2 weeks (15 pts - pull Story 17)
- Sprint 6: 1 week (3 pts - final polish)
- Total timeline: 11 weeks

**Option 3: Aggressive (Optimistic Velocity)**
- Sprint 1: 2 weeks (26 pts at 21 pts/sprint)
- Sprint 2: 2 weeks (20 pts)
- Sprint 3: 2 weeks (18 pts)
- Sprint 4: 2 weeks (13 pts)
- Sprint 5: 1 week (12 pts)
- Total timeline: 9 weeks

**Chosen Approach:** Option 2 (Rebalanced, 11 weeks)

---

## 8. Sprint Ceremonies

### Sprint Planning (Day 1, 2-3 hours)

**Agenda:**
1. Review sprint goal and milestone
2. Review backlog stories (priority order)
3. Story point estimation (if not pre-estimated)
4. Capacity planning (team availability)
5. Task breakdown (stories → tasks)
6. Commitment (which stories to pull into sprint)

**Outputs:**
- Sprint backlog (committed stories)
- Task breakdown (sub-tasks per story)
- Sprint goal statement

**Example Sprint Goal (Sprint 1):**
> Establish foundational infrastructure with Docker, Postgres, Redis, and validate single-agent execution end-to-end using Agno framework.

---

### Daily Standup (Daily, 15 minutes)

**Format:**
1. What did you complete yesterday?
2. What are you working on today?
3. Any blockers or impediments?

**Notes:**
- For solo developer: use as daily reflection
- Update todo list (TodoWrite tool)
- Track story progress (in_progress, completed)

---

### Sprint Review (Last Day, 1 hour)

**Agenda:**
1. Demo sprint deliverables
2. Review acceptance criteria (met/unmet)
3. Validate milestone achievement
4. Discuss incomplete stories (carry over?)
5. Stakeholder feedback (if applicable)

**Outputs:**
- Completed stories (marked as done)
- Incomplete stories (carry over to next sprint)
- Demo artifacts (CLI output, database state)

**Example Demo (Sprint 1):**
- Show `board create` creating a meeting
- Show `board run` executing 1-agent, 1-round meeting
- Show `board status` displaying meeting state
- Show database tables with meeting data

---

### Sprint Retrospective (Last Day, 30 minutes)

**Agenda:**
1. What went well?
2. What could be improved?
3. Action items for next sprint

**Outputs:**
- Retrospective notes
- Action items (process improvements)

**Example Action Items:**
- Improve test coverage (target: >70%)
- Add more logging for debugging
- Refactor context management (reduce complexity)

---

## 9. Story Breakdown (Detailed)

### Sprint 1 Stories (Detailed)

#### Story 1: Project Setup & Data Layer (8 points)

**Tasks:**
1. Initialize Python project with uv
   - `uv init theboard`
   - Configure pyproject.toml (dependencies, dev dependencies)
   - Set up .gitignore

2. Docker Compose setup
   - Create docker-compose.yml
   - Services: Postgres 15, Redis 7, RabbitMQ 3.12, Qdrant
   - Health checks for all services
   - Volume mounts for persistence

3. SQLAlchemy models
   - Define models: Meeting, Agent, Response, Comment, ConvergenceMetric, AgentMemory, AgentPerformance
   - Relationships (FKs): meeting_agents (many-to-many), responses → comments (one-to-many)
   - JSONB fields: agent_memory.memory_value, agents.letta_definition

4. Alembic migrations
   - Initialize Alembic: `alembic init alembic`
   - Create initial migration: `alembic revision --autogenerate -m "initial schema"`
   - Apply migration: `alembic upgrade head`

5. Redis connection manager
   - Redis client setup (redis-py)
   - Connection pooling
   - Test connectivity

6. Test database connectivity
   - Unit tests: CREATE, READ, UPDATE, DELETE operations
   - Integration test: Docker Compose up → database accessible

**Acceptance Criteria:**
- [ ] `docker-compose up` brings up all services (Postgres, Redis, RabbitMQ, Qdrant)
- [ ] Health checks pass for all services
- [ ] SQLAlchemy models defined with correct relationships
- [ ] Alembic migrations applied successfully
- [ ] Redis connection manager functional
- [ ] Unit tests pass (database CRUD)

**Definition of Done:**
- Code committed to git
- Tests passing (unit + integration)
- Documentation updated (README with setup instructions)

---

#### Story 2: Basic CLI Structure (3 points)

**Tasks:**
1. Install Typer and Rich
   - Add to pyproject.toml dependencies

2. Create CLI app skeleton
   - `src/theboard/cli.py` with Typer app
   - Commands: create, run, status, export (stubs)

3. Rich formatting setup
   - Tables for agent teams
   - Progress bars for live execution (placeholder)

4. Implement `board create` (minimal)
   - Interactive prompt for topic
   - Store meeting to database (minimal fields)
   - Output meeting ID

5. Implement `board status` (basic)
   - Query meeting by ID from database
   - Display: topic, status, current_round

6. Test CLI user experience
   - Manual testing: run commands, verify output
   - Unit tests: CLI argument parsing

**Acceptance Criteria:**
- [ ] `board create` prompts for topic, creates meeting, outputs meeting ID
- [ ] `board status <meeting-id>` displays meeting state
- [ ] Help text available for all commands (`board --help`)
- [ ] Rich formatting used for output (tables)

**Definition of Done:**
- Code committed to git
- Tests passing (CLI argument parsing)
- README updated with CLI usage examples

---

#### Story 3: Agno Integration & Simple Agent (8 points)

**Tasks:**
1. Install Agno framework
   - Add to pyproject.toml dependencies
   - Review Agno documentation

2. Create DomainExpertAgent as Agno skill
   - Define agent skill with LLM call
   - Integrate Claude Sonnet for response generation
   - Test LLM API call (mocked for unit tests)

3. Implement single-agent, single-round execution
   - Create simple Agno workflow
   - Execute workflow with 1 agent, 1 round
   - Store response to database

4. Test Agno workflow execution
   - Unit tests: workflow state management
   - Integration test: end-to-end single-agent execution

**Acceptance Criteria:**
- [ ] Agno framework installed and configured
- [ ] DomainExpertAgent responds to context input (LLM call)
- [ ] Single-agent, single-round execution works
- [ ] Response stored to database (responses table)
- [ ] Agno workflow state managed correctly

**Definition of Done:**
- Code committed to git
- Tests passing (unit + integration)
- Agno feasibility validated (decision point for Sprint 2)

---

#### Story 4: Notetaker Agent Implementation (7 points)

**Tasks:**
1. Define Comment Pydantic model
   - Fields: id, response_id, meeting_id, round, agent_name, text, category, novelty_score
   - Validation: text length, category enum

2. Create NotetakerAgent
   - LLM prompt: "Extract key ideas, technical decisions, risks"
   - Use Claude Sonnet with structured output (Pydantic model)

3. Implement comment extraction
   - Parse response text → list of Comment objects
   - Categorize comments (technical_decision, risk, implementation_detail)
   - Calculate novelty score (placeholder for now, full implementation in Sprint 3)

4. Store extracted comments to Postgres
   - INSERT into comments table
   - Batch insert for performance

5. Display comments via CLI
   - Enhance `board status` to show comments
   - Rich table formatting

6. Test extraction accuracy
   - Sample responses → expected comments
   - Validate >90% extraction rate

**Acceptance Criteria:**
- [ ] NotetakerAgent extracts comments from agent response
- [ ] Comments stored to database (comments table)
- [ ] `board status` displays comments
- [ ] Extraction accuracy >90% (validated with sample data)

**Definition of Done:**
- Code committed to git
- Tests passing (extraction accuracy validation)
- README updated with comment extraction details

---

### Sprint 2 Stories (Detailed)

#### Story 5: Meeting Coordinator Workflow (8 points)

**Tasks:**
1. Implement TheboardMeeting as Agno workflow
   - Define workflow class with state: topic, agents, strategy, max_rounds
   - Implement `run()` method

2. Round management logic
   - Loop: for round in range(1, max_rounds + 1)
   - Counter: current_round incremented per round
   - State tracking: current_agent, turn_queue

3. Sequential strategy execution
   - For each agent in turn_queue:
     - Agent.full_response(context)
     - Notetaker.extract_comments(response)
     - Context += compressed_comments

4. Track meeting state in Redis
   - SET meeting:{id}:state (current_round, current_agent, active_context)
   - TTL: 7 days

5. Emit basic RabbitMQ events
   - meeting.created
   - meeting.round.completed
   - agent.response.ready

6. Test multi-round execution with 2 agents
   - Integration test: 2-agent, 3-round meeting
   - Validate context accumulation

**Acceptance Criteria:**
- [ ] TheboardMeeting workflow executes sequential strategy
- [ ] Round management (loop, counter, state) works correctly
- [ ] Meeting state persisted to Redis
- [ ] RabbitMQ events emitted (meeting.created, round.completed)
- [ ] Multi-round execution validated (2 agents, 3 rounds)

**Definition of Done:**
- Code committed to git
- Tests passing (integration test with 2 agents)
- RabbitMQ events logged

---

#### Story 6: Agent Pool Management (5 points)

**Tasks:**
1. Agent pool loader for plaintext descriptions
   - Parse files in /home/delorenj/code/DeLoDocs/AI/Agents/Generic
   - Extract: name, expertise, persona, background

2. Parse agent files (regex or structured parsing)
   - Name: first line
   - Expertise: lines starting with "Expertise:"
   - Persona: lines starting with "Persona:"

3. AgentRegistry with in-memory index
   - Singleton pattern
   - Index: {agent_name: Agent}
   - Load agents on startup

4. Auto-select team using topic keywords
   - Embed topic using sentence-transformers
   - Similarity search in Qdrant (expertise embeddings)
   - Return top N agents (default: 5)

5. Manual agent selection via CLI interactive prompt
   - List all agents
   - User selects by name or number
   - Validate selection (agents exist)

6. Test team composition
   - Sample topics → expected agents
   - Validate >80% relevance

**Acceptance Criteria:**
- [ ] Agent pool loaded from plaintext files
- [ ] AgentRegistry indexes agents by name and expertise
- [ ] Auto-select team returns relevant agents (>80% relevance)
- [ ] Manual selection via CLI works
- [ ] Agent pool persisted to database (agents table)

**Definition of Done:**
- Code committed to git
- Tests passing (auto-selection validation)
- Agent pool documented (README)

---

#### Story 7: Context Management (7 points)

**Tasks:**
1. Implement ContextManager
   - Build cumulative context (Xr = Xr-1 + comments)
   - Append comments to context string

2. Context persistence to Redis
   - SET meeting:{id}:state (active_context field)
   - TTL: 7 days

3. Context size tracking
   - Calculate token count (chars / 4 estimate)
   - Warning if context > 15K tokens
   - Alert if context > 20K tokens (hard limit)

4. Test multi-round context accumulation
   - Integration test: 3-round meeting
   - Validate context growth (Xr > Xr-1)

5. Archive context history to Postgres
   - INSERT into responses table (full context per round)
   - Enable replay from database

**Acceptance Criteria:**
- [ ] Context accumulates across rounds (Xr = Xr-1 + new comments)
- [ ] Context persisted to Redis
- [ ] Context size tracking with alerts (15K, 20K tokens)
- [ ] Context history archived to Postgres (full audit trail)
- [ ] Multi-round accumulation validated

**Definition of Done:**
- Code committed to git
- Tests passing (context accumulation)
- Context size alerts functional

---

### Sprint 3 Stories (Detailed)

#### Story 8: Embedding Infrastructure (5 points)

**Tasks:**
1. Qdrant setup in Docker Compose
   - Add qdrant/qdrant:latest service
   - Volume mount for persistence
   - Health check

2. Comment embedding pipeline
   - Use sentence-transformers (model: all-MiniLM-L6-v2)
   - Embed comments → 768-dimensional vectors
   - Store embeddings in Qdrant (collection: comments)

3. Cosine similarity computation
   - Qdrant search API (similarity threshold: 0.85)
   - Batch processing for embeddings

4. Test embedding quality
   - Similar comments should have similarity > 0.85
   - Different comments should have similarity < 0.5

5. Optimize batch embedding processing
   - Batch size: 100 comments
   - Parallel processing if needed

**Acceptance Criteria:**
- [ ] Qdrant running in Docker Compose
- [ ] Comments embedded with sentence-transformers
- [ ] Embeddings stored in Qdrant (collection: comments)
- [ ] Cosine similarity computation works
- [ ] Embedding quality validated (similarity thresholds)

**Definition of Done:**
- Code committed to git
- Tests passing (embedding quality validation)
- Qdrant collection created

---

#### Story 9: Compressor Agent (8 points)

**Tasks:**
1. Implement CompressorAgent
   - Three-tier compression strategy

2. Tier 1: Similarity-based clustering
   - Qdrant cosine similarity search (threshold: 0.85)
   - Greedy clustering algorithm (used-set tracking)

3. Tier 2: LLM semantic merge
   - Prompt: "Combine these similar ideas into one coherent comment"
   - Use Claude Sonnet
   - Merge clusters into single comments

4. Tier 3: Outlier removal
   - Count support: how many agents mentioned this idea
   - Drop comments with support < 2

5. Track compression metrics
   - Original count, clustered count, merged count, final count
   - Compression ratio: (original - final) / original
   - Store metrics to Redis (meeting:{id}:compression:{round})

6. Test compression quality vs. information retention
   - Sample comment sets → expected compression ratio (40-60%)
   - Spot checks: merged comments preserve original meaning

**Acceptance Criteria:**
- [ ] Compressor reduces comment count by 40-60%
- [ ] Compression preserves information (spot checks)
- [ ] Compression metrics tracked (ratio, counts)
- [ ] Full audit trail maintained (pre-compression responses stored)
- [ ] Compression quality validated

**Definition of Done:**
- Code committed to git
- Tests passing (compression ratio validation)
- Compression metrics logged

---

#### Story 10: Convergence Detection (5 points)

**Tasks:**
1. Implement novelty score calculation
   - Embed current comments (Rr)
   - Embed previous comments (Rr-1)
   - Count overlaps (similarity > 0.85)
   - novelty = 1 - (overlap_count / len(current_comments))

2. Convergence threshold checking
   - Default threshold: 0.2
   - Check if novelty < threshold for k consecutive rounds (default: k=2)

3. Test stopping criteria
   - Scenarios: converged, not converged, max_rounds reached
   - Validate stopping reason

4. Persist convergence metrics per round
   - INSERT into convergence_metrics (meeting_id, round, novelty_score, comment_count)

5. Emit convergence events
   - meeting.convergence.detected (RabbitMQ)

6. Display convergence reason to user
   - CLI output: "Meeting converged after round X (novelty: 0.15)"

**Acceptance Criteria:**
- [ ] Novelty score calculated using embeddings
- [ ] Convergence detection stops meeting when novelty < 0.2 for 2 consecutive rounds
- [ ] Convergence metrics persisted per round
- [ ] Convergence event emitted (RabbitMQ)
- [ ] User notified of convergence reason

**Definition of Done:**
- Code committed to git
- Tests passing (convergence scenarios)
- Convergence metrics logged

---

### Sprint 4 Stories (Detailed)

#### Story 11: Greedy Execution Strategy (5 points)

**Tasks:**
1. Implement parallel agent responses
   - Use asyncio.gather() to call all agents in parallel
   - Collect responses

2. Comment-response phase
   - Each agent responds to other agents' comments
   - N² responses total (N agents)

3. Token efficiency tracking
   - Count tokens per agent, per round
   - Compare greedy vs. sequential (cost analysis)

4. Performance comparison
   - Benchmark: greedy vs. sequential execution time
   - Latency per round

5. Test convergence behavior with greedy
   - Validate convergence detection still works

**Acceptance Criteria:**
- [ ] Greedy strategy executes with parallel responses
- [ ] Comment-response phase implemented (N² responses)
- [ ] Token efficiency tracked (greedy vs. sequential)
- [ ] Performance benchmarks completed
- [ ] Convergence detection works with greedy strategy

**Definition of Done:**
- Code committed to git
- Tests passing (greedy execution)
- Performance benchmarks documented

---

#### Story 12: Event-Driven Human-in-Loop (5 points)

**Tasks:**
1. RabbitMQ event publishing
   - 5 event types: agent.response.ready, context.compression.triggered, meeting.round.completed, meeting.convergence.detected, meeting.human.input.needed
   - Exchange: theboard.events
   - Routing keys: meeting.{event_type}

2. CLI event consumer
   - Async listener for RabbitMQ events
   - Subscribe to meeting.* events

3. Interactive prompts for human steering
   - On meeting.human.input.needed: prompt user for input
   - Options: continue, pause, modify context, stop

4. Meeting pause/resume capability
   - Save state to Redis (status: paused)
   - Resume from Redis state

5. Timeout defaults
   - Auto-continue after 5 minutes if no human input

**Acceptance Criteria:**
- [ ] RabbitMQ events published at key decision points
- [ ] CLI consumer listens for events
- [ ] Human-in-loop prompts functional
- [ ] Meeting pause/resume works
- [ ] Timeout defaults tested (auto-continue)

**Definition of Done:**
- Code committed to git
- Tests passing (pause/resume)
- Event schema documented

---

#### Story 13: Hybrid Model Strategy (3 points)

**Tasks:**
1. Engagement metric calculation
   - Weighted formula: engagement = (peer_references * 0.5) + (novelty * 0.3) + (comment_count * 0.2)

2. Dynamic model promotion logic
   - Round 1: All agents use DeepSeek
   - After round 1: Calculate engagement per agent
   - Promote top 20% to Opus (based on engagement score)

3. Track per-agent costs and token usage
   - Store to agent_performance table
   - Cost calculation: tokens * model_price

4. Test cost savings vs. quality
   - A/B comparison: hybrid vs. all-Opus
   - Validate >60% cost reduction
   - Quality maintained (spot checks)

**Acceptance Criteria:**
- [ ] Engagement metric calculated per agent
- [ ] Model promotion logic works (top 20% → Opus)
- [ ] Cost savings >60% compared to all-Opus
- [ ] Quality maintained (validated via A/B testing)
- [ ] Per-agent costs tracked

**Definition of Done:**
- Code committed to git
- Tests passing (cost calculation)
- A/B testing results documented

---

### Sprint 5 Stories (Detailed)

#### Story 14: Letta Agent Migration (5 points)

**Tasks:**
1. Plaintext → Letta migration script
   - Parse plaintext agent files
   - Convert to Letta format (JSON or API call)

2. Letta memory persistence
   - Store to agent_memory table (JSONB)
   - Memory types: previous_meetings, learned_patterns

3. Cross-meeting memory recall
   - Query agent_memory by agent_id
   - Similarity search on topic (Qdrant)
   - Retrieve relevant past meetings (threshold: 0.7)

4. Test memory-enhanced responses
   - Agent recalls past meeting outcomes
   - Validate memory improves response quality

5. Migrate sample agents
   - Migrate 3-5 agents from plaintext to Letta
   - Test migration script

**Acceptance Criteria:**
- [ ] Migration script converts plaintext → Letta
- [ ] Letta memory persisted to agent_memory table
- [ ] Cross-meeting memory recall works (similarity search)
- [ ] Memory-enhanced responses validated
- [ ] Sample agents migrated

**Definition of Done:**
- Code committed to git
- Tests passing (memory recall)
- Migration script documented

---

#### Story 15: Export & Artifact Generation (3 points)

**Tasks:**
1. Markdown export
   - Template: heading, sections, bullet points
   - Generate from final context

2. JSON export
   - Structured schema: {meeting_id, topic, comments, rounds, ...}

3. HTML export
   - CSS styling for readability
   - Generate from final context

4. Test artifact quality
   - Readability validation (manual spot checks)

5. Support custom export templates
   - Jinja2 templates (future enhancement)

**Acceptance Criteria:**
- [ ] `board export` generates markdown, JSON, HTML
- [ ] Artifacts readable and well-formatted
- [ ] Custom templates supported (Jinja2)

**Definition of Done:**
- Code committed to git
- Tests passing (export formats)
- Sample artifacts generated

---

#### Story 16: Performance Optimization (2 points)

**Tasks:**
1. Lazy compression
   - Only compress when context > 10K chars
   - Track compression trigger count

2. Delta propagation
   - Agents receive only new comments since last turn
   - Track per-agent last_seen_round

3. Optimize Redis caching strategy
   - Review TTLs, eviction policies

4. Selective agent activation
   - Only engaged agents participate in round 2+ (based on engagement score)

**Acceptance Criteria:**
- [ ] Lazy compression triggered only when needed
- [ ] Delta propagation reduces token usage by >30%
- [ ] Redis caching optimized
- [ ] Selective activation tested

**Definition of Done:**
- Code committed to git
- Tests passing (delta propagation)
- Optimization settings documented

---

### Sprint 6 Stories (Detailed)

#### Story 17: CLI Polish & Documentation (2 points)

**Tasks:**
1. Enhance `board export` with format options
   - `--format markdown|json|html`
   - `--output <file_path>`

2. Implement live progress streaming
   - Rich.Live() for real-time updates
   - Display: round, agent, comment count, novelty score

3. Comprehensive help text
   - `board --help`
   - `board create --help`, etc.

4. User guide with 3+ example scenarios
   - Scenario 1: Simple meeting (5 agents, sequential)
   - Scenario 2: Greedy strategy
   - Scenario 3: Human-in-loop steering

5. Troubleshooting guide
   - Common issues: Docker not running, LLM API rate limits, etc.

6. Developer documentation
   - API reference (Pydantic models, core APIs)
   - Architecture diagram (Mermaid or similar)

**Acceptance Criteria:**
- [ ] `board export` enhanced with format options
- [ ] Live progress streaming functional
- [ ] Help text comprehensive
- [ ] User guide with 3+ examples published
- [ ] Troubleshooting guide available
- [ ] Developer documentation complete

**Definition of Done:**
- Code committed to git
- Documentation published (README, docs/)
- All acceptance criteria met

---

## 10. Validation Checklist

### Milestone 1: MVP Foundation Complete
- [ ] Single-agent execution works end-to-end
- [ ] Basic CLI functional (create, run, status)
- [ ] Data layer operational (Postgres, Redis)
- [ ] Can run 1-agent, 3-round meeting and export artifact
- [ ] Database schema validated (all tables created)

### Milestone 2: Multi-Agent Orchestration Working
- [ ] Sequential strategy executing 5-agent meetings
- [ ] Context accumulation validated across rounds
- [ ] Agent pool management functional (auto-select + manual)
- [ ] Can run 5-agent, 5-round meeting with coherent context
- [ ] Meeting state persisted to Redis and Postgres

### Milestone 3: Intelligence Layer Complete
- [ ] Compression reducing tokens by 40%+ with quality preservation
- [ ] Convergence detection stopping at appropriate rounds
- [ ] Embedding infrastructure operational (Qdrant)
- [ ] Meeting stops automatically when converged
- [ ] Context manageable (<15K tokens for 5-round, 5-agent meeting)

### Milestone 4: Advanced Features Working
- [ ] Greedy strategy functional
- [ ] Human-in-loop pause/resume working
- [ ] Hybrid model strategy reducing costs by >60%
- [ ] User can steer meetings
- [ ] Costs optimized and tracked

### Milestone 5: Production-Ready v1
- [ ] All 17 stories implemented and tested
- [ ] Acceptance criteria met (see Section 6 of tech spec)
- [ ] Documentation complete (user guide, API docs, troubleshooting)
- [ ] Docker deployment validated
- [ ] System ready for real-world usage

---

## 11. Appendix

### A. Fibonacci Estimation Guide

**Story Point Scale:**
- **1 point:** Trivial task (1-2 hours)
  - Example: Add help text to CLI command

- **2 points:** Small task (2-4 hours)
  - Example: Implement simple utility function

- **3 points:** Medium task (4-8 hours, ~1 day)
  - Example: Basic CLI command with database interaction

- **5 points:** Large task (8-16 hours, 1-2 days)
  - Example: Agent pool loader with auto-selection

- **8 points:** Very large task (16-24 hours, 2-3 days)
  - Example: Agno integration, Compressor agent

- **13 points:** Epic-sized task (3-5 days)
  - Example: Complete orchestration layer refactor

- **21 points:** Multi-sprint epic (5+ days)
  - Example: Full system redesign (not recommended, break down further)

---

### B. Definition of Done (DoD)

**Code:**
- [ ] Code committed to git (feature branch)
- [ ] Code reviewed (self-review or peer review)
- [ ] No linting errors (ruff check passes)
- [ ] Type checking passes (mypy --strict)

**Testing:**
- [ ] Unit tests written (>70% coverage for new code)
- [ ] Integration tests passing (if applicable)
- [ ] Manual testing completed (CLI user experience)

**Documentation:**
- [ ] Code comments added (where necessary)
- [ ] Docstrings updated (Google style)
- [ ] README updated (if user-facing change)
- [ ] API documentation updated (if API change)

**Deployment:**
- [ ] Merged to main branch
- [ ] CI/CD pipeline passes
- [ ] Docker Compose tested (if infrastructure change)

---

### C. Glossary

- **Story:** Unit of work (feature, task, bug fix)
- **Sprint:** Time-boxed iteration (2 weeks)
- **Story Point:** Relative size estimate (Fibonacci scale)
- **Velocity:** Story points completed per sprint
- **Milestone:** Major achievement (end of phase)
- **Backlog:** Prioritized list of stories
- **Acceptance Criteria:** Conditions for story completion
- **Definition of Done:** Checklist for story quality
- **Critical Path:** Sequential dependencies that determine minimum timeline

---

### D. References

- **Tech Spec:** /home/delorenj/code/theboard/docs/tech-spec-theboard-2025-12-19.md
- **Architecture:** /home/delorenj/code/theboard/docs/architecture-theboard-2025-12-19.md
- **PRD:** /home/delorenj/code/theboard/PRD.md

---

### E. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-19 | Scrum Master (BMAD) | Initial sprint plan |

---

*Generated by BMAD Method v6 - Scrum Master*
*Sprint Plan for 17 stories across 6 sprints (11 weeks at 16 pts/sprint velocity)*
*Total Effort: 89 story points*
