# Sprint 5 Plan: TheBoard - Letta Integration & Export

**Date:** 2025-12-30
**Scrum Master:** delorenj (Steve)
**Project:** TheBoard
**Project Level:** 3 (Complex integration, 12-40 stories)
**Sprint Duration:** 2 weeks (2025-12-30 to 2026-01-10)
**Total Stories:** 3
**Total Points:** 10
**Team Capacity:** 13 points
**Utilization:** 77%

---

## Executive Summary

Sprint 5 focuses on production readiness enhancements: implementing Letta agent memory for cross-meeting institutional knowledge, export functionality for shareable artifacts, and performance optimizations for scalability. This sprint transitions TheBoard from a working prototype to a production-grade brainstorming platform suitable for multi-week discussion series.

**Key Deliverables:**
- Letta agent migration with persistent memory across meetings
- Multi-format export (Markdown, JSON, HTML) for stakeholder sharing
- Performance optimizations (lazy compression, delta propagation, selective activation)

**Strategic Value:**
- Enables multi-week brainstorming campaigns with memory continuity
- Provides professional artifacts for Confluence, email, and documentation systems
- Reduces token costs by 30%+ through intelligent optimization

**Risk Profile:** Medium-High
- Story 14 (Letta integration) carries technical risk requiring early prototyping
- Holiday season (late Dec/early Jan) may impact availability
- 23% capacity buffer allocated for risk mitigation

---

## Sprint Goal

**"Enable agents to build institutional knowledge across meetings and deliver shareable artifacts with optimized resource usage"**

**Success Criteria:**
- [ ] Agents recall past meeting outcomes in new discussions (Story 14)
- [ ] `board export` generates readable MD/JSON/HTML artifacts (Story 15)
- [ ] Token usage reduced by ≥30% via delta propagation (Story 16)
- [ ] All acceptance criteria met for Stories 14, 15, 16
- [ ] No critical bugs in Letta integration
- [ ] Demo: 2-meeting series with memory recall and multi-format export

---

## Story Inventory

### STORY-14: Letta Agent Migration
**Priority:** Must Have
**Estimate:** 5 points (1-2 days)
**Risk:** High
**Dependencies:** Qdrant collection, `agent_memory` table migration, Letta SDK

**User Story:**
As a meeting facilitator
I want agents to remember past discussions across meetings
So that brainstorming sessions build institutional knowledge instead of starting from scratch each time

**Acceptance Criteria:**
- [ ] Migration script converts plaintext agents to Letta agents
  - Preserves agent role, expertise, and description
  - Maps to new `agent_memory` table schema
  - Handles edge cases (missing fields, duplicate agents)
- [ ] Letta memory persists to `agent_memory` table with columns:
  - `agent_id`, `meeting_id`, `memory_type` (decision/pattern/context)
  - `content` (JSON), `relevance_score`, `created_at`
- [ ] Cross-meeting memory recall implemented:
  - Query Qdrant for similar past discussions (vector similarity search)
  - Return top 5 relevant memories per agent
  - Memory retrieval latency <1s with 100+ meeting history
- [ ] Memory-enhanced responses validated:
  - Agents reference past decisions in responses (A/B testing)
  - No contradictions with previous meeting outcomes
  - Memory improves coherence score by ≥20%
- [ ] Sample agents migrated from pool:
  - At least 7 agents from existing pool converted
  - Test with both Letta and plaintext agents (hybrid support)

**Technical Implementation:**
- Use Letta 0.4.x API for memory persistence
- Qdrant vector search for memory retrieval (collection: `agent_memories`)
- Fallback to plaintext agents if Letta initialization fails
- Memory schema: `{"meeting_id": UUID, "decision": str, "confidence": float, "timestamp": ISO8601}`
- Migration path: `src/theboard/migrations/migrate_to_letta.py`

**Testing Strategy:**
- Unit tests: memory persistence, retrieval, migration script
- Integration tests: cross-meeting recall with 10+ meetings
- Performance tests: retrieval latency with 100+ meetings
- A/B comparison: Letta vs. plaintext coherence scores

**Risks:**
- **High:** Letta integration complexity higher than expected
- **Medium:** Memory retrieval performance degradation at scale
- **Medium:** Letta API changes (0.4.x → 0.5.x breaking changes)

**Mitigation:**
- Prototype early in sprint (Days 1-2)
- Performance testing with 100+ meetings
- Hybrid support: both plaintext and Letta agents
- Vector similarity search for fast memory recall
- Graceful degradation if Letta unavailable

---

### STORY-15: Export & Artifact Generation
**Priority:** Should Have
**Estimate:** 3 points (4-8 hours)
**Risk:** Low
**Dependencies:** None (works with existing data model)

**User Story:**
As a meeting organizer
I want to export meeting results in multiple formats
So that I can share brainstorming outcomes with stakeholders via Confluence, email, or internal docs

**Acceptance Criteria:**
- [ ] Markdown export implemented:
  - Formatted with H1/H2/H3 headings per round
  - Bullet points for comments
  - Decision summary section at end
  - Syntax: `board export <meeting_id> --format markdown --output <path>`
- [ ] JSON export implemented:
  - Structured schema with rounds, agents, comments, metrics
  - Include convergence metrics, compression ratios
  - Support filtering: `--include-metadata`, `--comments-only`
  - Valid JSON schema (pass `jq` validation)
- [ ] HTML export implemented:
  - CSS styling for readability (light/dark theme toggle)
  - Responsive design (mobile-friendly)
  - Embedded charts for metrics (convergence, engagement)
  - Shareable via email or web hosting
- [ ] Export quality validated:
  - Manual review of 3+ exported artifacts
  - Readability score >8/10 (Flesch-Kincaid)
  - No data loss (comment count matches)
- [ ] Custom export templates supported:
  - Jinja2 templates in `templates/export/`
  - Template variables: `{{meeting_topic}}`, `{{rounds}}`, `{{comments}}`
  - User can specify: `--template custom-report.md.j2`

**Technical Implementation:**
- Use Jinja2 for templating
- Markdown: Python `markdown` library for rendering
- JSON: Pydantic schemas for serialization (`ExportSchema`)
- HTML: Bootstrap 5 CSS framework
- Export directory: `exports/{meeting_id}/`
- CLI command: `board export` with Rich progress indicator

**Testing Strategy:**
- Unit tests: each export format (MD/JSON/HTML)
- Integration tests: end-to-end export with real meeting data
- Manual QA: readability, formatting, responsiveness
- Edge cases: empty meetings, large meetings (1000+ comments)

**Risks:**
- **Low:** Template customization complexity
- **Low:** Large exports (>1MB) slow to generate

**Mitigation:**
- Provide 3 default templates (simple, detailed, presentation)
- Async export for large meetings (background task)
- Progress indicator with Rich.Live
- Size limits: warn at 10MB, fail at 50MB

---

### STORY-16: Performance Optimization
**Priority:** Should Have
**Estimate:** 2 points (2-4 hours)
**Risk:** Low
**Dependencies:** Engagement metrics (Story 13, Sprint 4 - already implemented)

**User Story:**
As a system operator
I want TheBoard to optimize resource usage automatically
So that meetings run efficiently without manual configuration tuning

**Acceptance Criteria:**
- [ ] Lazy compression implemented:
  - Compression triggered only when `len(context) > 10,000` chars
  - Skip compression for rounds 1-2 (low context)
  - Log: "Compression skipped (context: 4,523 chars < 10K threshold)"
  - Validate 30% reduction in compression cycles
- [ ] Delta propagation implemented:
  - Agents receive only new comments since last turn (not full history)
  - Delta calculation: `comments_since(agent_id, last_round)`
  - Token usage reduced by ≥30% (measured via OpenRouter API)
  - Test with 5-round, 7-agent meeting
- [ ] Redis caching optimized:
  - Cache context per agent (key: `meeting:{id}:agent:{id}:context`)
  - TTL: 24 hours (auto-expire old meetings)
  - Cache hit rate >80% for active meetings
  - Eviction policy: LRU (least recently used)
- [ ] Selective agent activation implemented:
  - Calculate engagement score: `peer_references + novelty + comment_count`
  - Filter out agents with engagement <0.3 after round 2
  - Log: "Agent {name} filtered (engagement: 0.18 < 0.3 threshold)"
  - Validate 20% reduction in API calls

**Technical Implementation:**
- Context size tracking: `len(json.dumps(context))`
- Delta calculation: SQLAlchemy query with `created_at > agent_last_turn`
- Redis caching: `RedisManager.set_with_ttl(key, value, ttl=86400)`
- Engagement threshold: configurable via `config.py` (default: 0.3)
- Configuration: `src/theboard/config.py` optimization settings

**Testing Strategy:**
- Unit tests: delta calculation, engagement filtering
- Integration tests: full meeting with optimizations enabled
- Performance benchmarks: before/after token usage, compression cycles
- A/B testing: optimized vs. baseline coherence

**Risks:**
- **Low:** Lazy compression delays may cause OOM on very large meetings
- **Low:** Delta propagation breaks coherence (agents miss context)

**Mitigation:**
- Hard limit: compress at 50K chars regardless of threshold
- Configurable delta window: `--delta-window-rounds 2` (default: 1)
- A/B testing to validate coherence maintained
- Monitor: memory usage, API token consumption

---

## Sprint Allocation & Sequencing

### Week 1 (Days 1-5): Letta Foundation
**Focus:** STORY-14 (Letta Agent Migration)

**Day 1-2: Prototype & Infrastructure**
- Set up Letta SDK (install, configure)
- Create `agent_memory` table migration (Alembic)
- Prototype memory persistence and retrieval
- Validate Qdrant integration for vector search
- **Checkpoint:** Memory write/read working with 1 agent

**Day 3-4: Migration & Testing**
- Implement migration script (plaintext → Letta)
- Migrate 7 sample agents from pool
- Test cross-meeting memory recall
- Performance testing: 100+ meeting history
- **Checkpoint:** Migration complete, memory recall <1s

**Day 5: Validation & Refinement**
- A/B testing: Letta vs. plaintext coherence
- Edge case handling (missing fields, duplicates)
- Documentation: migration guide, Letta setup
- **Deliverable:** Letta agents with cross-meeting memory

---

### Week 2 (Days 6-10): Export & Optimization
**Focus:** STORY-15 (Export) + STORY-16 (Performance) - Parallel tracks

**Days 6-7: Export Implementation (STORY-15)**
- Implement markdown export (Jinja2 templates)
- Implement JSON export (Pydantic schemas)
- Implement HTML export (Bootstrap CSS)
- Test export quality (readability, formatting)
- **Deliverable:** `board export` with MD/JSON/HTML formats

**Days 8-9: Performance Optimization (STORY-16)**
- Implement lazy compression (10K threshold)
- Implement delta propagation (comment deltas)
- Optimize Redis caching (TTL, LRU)
- Implement selective agent activation (engagement filter)
- **Deliverable:** 30% token reduction via optimizations

**Day 10: Integration & Sprint Review Prep**
- Integration testing: all 3 stories together
- Performance benchmarks: token usage, compression cycles
- Prepare demo: 2-meeting series with memory and export
- Documentation updates
- **Deliverable:** Sprint 5 complete, ready for review

---

## Epic Traceability

Sprint 5 stories map to the following epics:

| Epic | Epic Name | Stories | Total Points | Sprint |
|------|-----------|---------|--------------|--------|
| Agent Infrastructure | Persistent Memory & Intelligence | STORY-14 | 5 points | Sprint 5 |
| Reporting & Shareability | Export & Artifacts | STORY-15 | 3 points | Sprint 5 |
| Performance & Scalability | Resource Optimization | STORY-16 | 2 points | Sprint 5 |

**Coverage:**
- All Sprint 5 functional requirements covered
- No orphaned requirements
- Clear epic-to-story mapping

---

## Functional Requirements Coverage

Sprint 5 implements the following requirements from the PRD (if exists) or high-level features:

| FR ID | FR Name | Story | Sprint | Status |
|-------|---------|-------|--------|--------|
| FR-14 | Agent memory across meetings | STORY-14 | 5 | Not Started |
| FR-15 | Export artifacts (MD/JSON/HTML) | STORY-15 | 5 | Not Started |
| FR-16 | Performance optimization | STORY-16 | 5 | Not Started |

**Note:** Sprint 1-4 FRs already completed (26 + 20 + 18 + 13 = 77 points delivered)

---

## Risks & Mitigation

### High Risks

**Risk 1: Letta Integration Complexity**
- **Impact:** Story 14 may take longer than estimated (5 points → 8 points)
- **Probability:** Medium (40%)
- **Mitigation:**
  - Prototype early (Days 1-2) to validate approach
  - Hybrid support: fallback to plaintext agents
  - Timebox to 5 days; descope memory features if needed
  - Consult Letta docs and examples proactively

### Medium Risks

**Risk 2: Holiday Availability**
- **Impact:** Reduced productivity during late Dec/early Jan
- **Probability:** Medium (30%)
- **Mitigation:**
  - 23% capacity buffer (3 points)
  - Stories 15 & 16 are independent (can slip to Sprint 6 if needed)
  - Story 14 prioritized (must complete)

**Risk 3: Memory Retrieval Performance**
- **Impact:** Latency >1s with 100+ meetings, poor UX
- **Probability:** Low (20%)
- **Mitigation:**
  - Vector similarity search (Qdrant optimized)
  - Performance testing with realistic data
  - Caching layer for frequent queries
  - Pagination: return top 5 memories only

### Low Risks

**Risk 4: Export Template Complexity**
- **Impact:** Custom templates may be difficult to create
- **Probability:** Low (10%)
- **Mitigation:**
  - Provide 3 well-documented default templates
  - Template variables documented in user guide
  - Example custom templates in `templates/export/examples/`

---

## Dependencies

### Internal Dependencies
- **Story 14 → Stories 15/16:** None (all independent)
- **Story 15 ← Story 16:** None (parallel execution)
- **Sprint 4 complete:** ✓ (all 13 stories delivered)

### External Dependencies
- **Letta SDK:** 0.4.x available on PyPI (✓ verified)
- **Qdrant:** Already deployed and configured (Sprint 3)
- **PostgreSQL:** `agent_memory` table migration required (Alembic)
- **Redis:** Already deployed and configured (Sprint 1)

### Infrastructure
- **Native Services:** Postgres (192.168.1.12:5432), Redis (localhost:6379), RabbitMQ (localhost:5672), Qdrant (proxy network)
- **API Keys:** OpenRouter API key configured (✓)
- **Disk Space:** Export directory may grow (monitor `exports/` size)

---

## Definition of Done

For a story to be considered complete:
- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage for new code)
- [ ] Integration tests passing (end-to-end scenarios)
- [ ] Code reviewed (self-review for solo developer, checklist-based)
- [ ] Documentation updated (user guide, API reference, migration guide)
- [ ] Acceptance criteria validated (manual testing or automated)
- [ ] Performance benchmarks run (token usage, latency, memory)
- [ ] No critical bugs (P0/P1 issues resolved)
- [ ] Feature branch merged to `main` via PR

**Sprint 5 Specific:**
- [ ] Letta agents demonstrate memory recall in 2-meeting series
- [ ] Export generates readable artifacts (MD/JSON/HTML)
- [ ] Performance optimizations reduce token usage by ≥30%
- [ ] Demo scenario ready for sprint review

---

## Demo Scenario (Sprint Review)

**Scenario:** Multi-week architecture discussion series with memory and export

**Setup:**
```bash
# Week 1: Initial discussion about deployment strategy
board create --topic "Microservices deployment strategy for mobile backend"
board run --auto-select --letta-agents --agents 7 --strategy sequential
board export <meeting_id> --format markdown --output deployment-strategy-v1.md
```

**Expected Behavior (Week 1):**
- 7 Letta agents participate (migrated from plaintext pool)
- SRE specialist suggests "Kubernetes with Istio service mesh"
- Memory persists: `deployment_pattern: "k8s + istio"` stored in agent memory
- Markdown export generates formatted report with decision summary

**Setup (Week 2):**
```bash
# Week 2: Follow-up about observability
board create --topic "Observability and monitoring for microservices"
board run --auto-select --letta-agents --agents 7 --strategy sequential
board export <meeting_id> --format html --output observability-report.html
```

**Expected Behavior (Week 2):**
- SRE specialist recalls Week 1 decision: *"Based on our previous discussion about Kubernetes with Istio, I recommend Prometheus + Grafana for observability, which integrates well with Istio's telemetry."*
- Cross-meeting memory improves coherence: no contradictions
- HTML export generates styled report with charts (convergence, engagement)

**Performance Metrics:**
- Token usage reduced by ≥30% (delta propagation)
- Compression cycles reduced by 30% (lazy compression)
- Memory retrieval latency <1s (vector search)
- Export generation <5s (markdown/JSON/HTML)

**Demo Flow:**
1. Show Week 1 meeting creation and export (markdown)
2. Show Week 2 meeting with memory recall (SRE specialist quote)
3. Show HTML export with styling and charts
4. Show performance metrics dashboard (token savings, compression)
5. Q&A: memory accuracy, export customization, optimization impact

---

## Team Velocity & Capacity

**Historical Velocity:**
- Sprint 1: 26 points / 2 weeks = 13 pts/week
- Sprint 2: 20 points / 2 weeks = 10 pts/week
- Sprint 3: 18 points / 2 weeks = 9 pts/week
- Sprint 4: 13 points / 2 weeks = 6.5 pts/week

**Trend:** Declining velocity (-23% per sprint avg) due to increasing complexity

**Sprint 5 Capacity:**
- **Team size:** 1 developer
- **Sprint length:** 2 weeks (10 workdays)
- **Productive hours:** 6 hours/day (senior developer)
- **Total hours:** 60 hours
- **Velocity:** 6.5 pts/week × 2 weeks = 13 points capacity
- **Committed:** 10 points
- **Buffer:** 3 points (23%)

**Utilization:** 77% (healthy, accounts for high-risk Story 14 and holiday season)

---

## Retrospective Planning

**What to Review:**
- Letta integration complexity: Was 5-point estimate accurate?
- Holiday impact: Did late Dec/early Jan affect productivity?
- Velocity trend: Declining from 13 → 6.5 pts/week - why?
- Parallel execution: Did Stories 15/16 benefit from independence?
- Buffer usage: Was 23% buffer sufficient?

**Metrics to Track:**
- Story completion rate (target: 100%)
- Velocity: actual vs. predicted (6.5 pts/week)
- Defect rate: bugs found post-sprint
- Code coverage: maintain ≥80%
- Performance benchmarks: token reduction, latency

---

## Next Steps

**Immediate (Sprint 5):**
1. **Day 1:** Begin STORY-14 (Letta prototype)
2. **Day 5:** Complete STORY-14 (Letta migration)
3. **Days 6-10:** Complete STORY-15 (Export) + STORY-16 (Performance)
4. **Day 10:** Sprint review demo

**After Sprint 5 (Sprint 6):**
- **Sprint 6 Goal:** CLI polish, comprehensive documentation, production readiness
- **Story 17:** CLI Polish & Documentation (2 points)
  - Enhance `board export` with format options
  - Implement live progress streaming (Rich.Live)
  - Comprehensive help text for all commands
  - User guide with 3+ example scenarios
  - Troubleshooting guide, developer documentation

**Long-term Roadmap:**
- Production deployment (Docker Compose, CI/CD)
- Performance benchmarking (NFR validation)
- Security audit (credential management, input validation)
- User feedback collection (alpha testing)

**Command to Start:**
```bash
# Begin Sprint 5 implementation
board status  # Check current state
# Create Story 14 feature branch
git checkout -b feature/story-14-letta-migration
# Start development...
```

---

**This plan was created using BMAD Method v6 - Phase 4 (Sprint Planning)**
**Generated:** 2025-12-30
**Next Review:** 2026-01-10 (Sprint 5 Review)
