# Brainstorming Session: Meeting Configuration & Wizard Design

**Date:** 2026-01-02
**Objective:** Ideate on optimal meeting configuration options and design wizard-based creation flow with JSON template support

**Context:**
- Current state: CLI with 6 parameters (topic, strategy, max_rounds, agent_count, auto_select, model)
- Workflow system: MultiAgentMeetingWorkflow with configurable convergence
- Architecture: Event-driven, modular agents, Redis/Postgres persistence
- Constraints: Backward compatibility, JSON serializable, schema validation

---

## Techniques Used

1. **SCAMPER** - Creative variations on existing config options to discover new parameters
2. **Starbursting** - Question-based exploration to surface implicit requirements
3. **Six Thinking Hats** - Multi-perspective analysis balancing user needs and technical constraints

---

## Ideas Generated

### Category 1: Core Meeting Parameters (5 options)
- `topic` - Meeting subject (10-500 chars, required)
- `meeting_type` - brainstorm, risk_assessment, technical_review, debate, research
- `meeting_brief` - Structured object: topic + background + constraints + desired_output
- `priority_level` - urgent/high/medium/low (influences model selection)
- `meeting_length_strategy` - quick/standard/thorough/unlimited (combines rounds + convergence)

### Category 2: Agent Selection & Configuration (6 options)
- `agent_selection_mode` - auto/manual/hybrid/preset/exclude_list
- `agent_pool_preset` - small(3)/medium(5)/large(8)/custom
- `persona_config` - Array of {agent_id, role, model_tier, creativity_level}
- `exclude_perspectives` - Negative selection (perspectives to avoid)
- `difficulty_level` - novice_agents/balanced/expert_agents
- `creativity_spectrum` - 0-1 float (conservative to radical thinking)

### Category 3: Execution Strategy (5 options)
- `strategy` - sequential/greedy/parallel/hybrid
- `max_rounds` - 1-10 (default 5)
- `min_rounds` - 1-5 (default 2)
- `round_configuration` - sync_mode, duration_limit, intermission_enabled
- `adaptive_meeting` - Evolve based on early rounds

### Category 4: Convergence & Quality Control (6 options)
- `convergence_profile` - strict/balanced/exploratory/disabled
- `novelty_threshold` - 0-1 float (default 0.3)
- `early_exit_conditions` - time_limit, cost_limit, comment_count, quality_target
- `min_quality_threshold` - Run until quality target hit
- `compression_enabled` - Default true
- `compression_threshold` - Chars before compression (default 10000)

### Category 5: Model & Cost Management (5 options)
- `model_override` - Apply single model to all agents
- `model_strategy` - uniform/tiered/adaptive/cost_optimized
- `model_budget` - Max spend per meeting
- `cost_optimizer_enabled` - Suggest cheaper alternatives
- `dry_run_mode` - Preview costs/agents without execution

### Category 6: Output & Artifacts (5 options)
- `artifact_bundle` - Array: markdown, json, metrics, transcript, summary
- `synthesis_style` - bullet_points/narrative/structured_doc/executive_summary
- `recording_mode` - full_transcript/summary_only/none
- `output_format` - standard/compact/detailed
- `auto_export_enabled` - Generate artifacts on completion

### Category 7: Specialized Modes (4 options)
- `research_mode` - Agents cite sources, fact-check claims
- `debate_mode` - Argue opposing viewpoints
- `role_play_mode` - Simulate stakeholder perspectives
- `expert_review_mode` - Specialized critique

### Category 8: Wizard & Template Support (4 options)
- `template_name` - Load named template
- `template_path` - Load from file
- `save_as_template` - Save current config as template
- `smart_defaults_enabled` - Infer settings from topic
- `wizard_mode` - full/quick/advanced/none

---

## Key Insights

### Insight 1: Progressive Disclosure via Three-Tier Config System
**Description:** Organize 48 options into Essential (3-5 params), Common (10-12 params), Advanced (rest) tiers to prevent analysis paralysis

**Source:** SCAMPER (Eliminate), Six Hats (Black: config explosion risk)

**Impact:** High | **Effort:** Medium

**Why it matters:** New users see 3-5 essential options via wizard, power users access full config via JSON templates. Prevents overwhelming beginners while preserving advanced control.

**Recommended structure:**
- **Essential Tier (Wizard):** topic, meeting_type, meeting_length_strategy
- **Common Tier (CLI flags):** agent_pool_preset, convergence_profile, model_strategy, output_format
- **Advanced Tier (Templates):** All 40+ remaining options

**Implementation:**
```python
# Essential (required, wizard-friendly)
class EssentialConfig(BaseModel):
    topic: str = Field(..., min_length=10, max_length=500)
    meeting_type: MeetingType = Field(default=MeetingType.BRAINSTORM)
    meeting_length_strategy: LengthStrategy = Field(default=LengthStrategy.STANDARD)

# Common (optional, exposed as CLI flags)
class CommonConfig(BaseModel):
    agent_pool_preset: AgentPoolSize = Field(default=AgentPoolSize.MEDIUM)
    convergence_profile: ConvergenceProfile = Field(default=ConvergenceProfile.BALANCED)
    model_strategy: ModelStrategy = Field(default=ModelStrategy.TIERED)

# Advanced (templates only)
class AdvancedConfig(BaseModel):
    persona_config: list[PersonaConfig] | None = None
    round_configuration: RoundConfig | None = None
    early_exit_conditions: ExitConditions | None = None
    # ... 35+ more options
```

---

### Insight 2: Template System as First-Class Feature
**Description:** JSON templates not just for automation, but primary mechanism for config sharing, discovery, and education

**Source:** Starbursting (Where/When/How), Six Hats (Yellow: team alignment)

**Impact:** High | **Effort:** High

**Why it matters:** Templates solve multiple problems simultaneously:
- **Repeatability** - Same config for recurring meeting types
- **Sharing** - Team standards enforced through templates
- **Documentation** - Templates show best practices by example
- **Onboarding** - New users learn config options through curated examples

**Template Features:**
1. **Schema Validation** - Strict JSON schema with helpful error messages
2. **Template Inheritance** - Base templates + scenario-specific overrides
3. **Discovery** - `board templates list` with descriptions
4. **Quick Application** - `board create --template risk_assessment`
5. **Template Export** - Save successful meeting configs as templates

**Default Templates to Ship:**
```json
// quick_brainstorm.json - Fast ideation (2 rounds, 3 agents, low cost)
{
  "meeting_type": "brainstorm",
  "meeting_length_strategy": "quick",
  "agent_pool_preset": "small",
  "convergence_profile": "balanced",
  "max_rounds": 2,
  "model_strategy": "cost_optimized"
}

// thorough_analysis.json - Deep exploration (5 rounds, 8 agents, quality focus)
{
  "meeting_type": "technical_review",
  "meeting_length_strategy": "thorough",
  "agent_pool_preset": "large",
  "convergence_profile": "exploratory",
  "max_rounds": 5,
  "model_strategy": "uniform"
}

// risk_assessment.json - Cautious evaluation (debate mode, expert agents)
{
  "meeting_type": "risk_assessment",
  "debate_mode": true,
  "difficulty_level": "expert_agents",
  "convergence_profile": "strict",
  "synthesis_style": "structured_doc"
}
```

**Template Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "TheBoard Meeting Template",
  "type": "object",
  "required": ["topic"],
  "properties": {
    "template_metadata": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "author": {"type": "string"},
        "version": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}}
      }
    },
    "topic": {"type": "string", "minLength": 10, "maxLength": 500},
    "meeting_type": {"enum": ["brainstorm", "risk_assessment", "technical_review", "debate", "research"]},
    "meeting_length_strategy": {"enum": ["quick", "standard", "thorough", "unlimited"]},
    ...
  }
}
```

---

### Insight 3: Wizard as Separate Command, Not Flag
**Description:** Implement `board wizard` as dedicated interactive flow instead of `board create --wizard` flag

**Source:** Six Hats (Blue: process design), Starbursting (How: integration)

**Impact:** Medium | **Effort:** Low

**Why it matters:**
- **Clear separation** - Power users never accidentally trigger wizard
- **Teaching tool** - Wizard educates about config options through interaction
- **Template generation** - Wizard outputs JSON template at end, bridging guided and automated workflows
- **Discoverability** - `board wizard` is more memorable than flag combination

**Wizard Flow:**
```bash
$ board wizard

┌─────────────────────────────────────────┐
│ TheBoard Meeting Creation Wizard       │
└─────────────────────────────────────────┘

Step 1/5: Meeting Topic
> What are we brainstorming?
  How do I optimize database query performance for user analytics?

Step 2/5: Meeting Type
  1. Brainstorm (general ideation)
  2. Risk Assessment (identify risks)
  3. Technical Review (evaluate approach)
  4. Debate (opposing viewpoints)
  5. Research (fact-based analysis)
> Select: 3

Step 3/5: Meeting Length
  1. Quick (2 rounds, ~5 min, $0.20-$0.40)
  2. Standard (3-4 rounds, ~10 min, $0.40-$0.80) [Recommended]
  3. Thorough (5+ rounds, ~15 min, $0.80-$1.50)
> Select: 2

Step 4/5: Agent Team Size
  1. Small (3 agents - focused)
  2. Medium (5 agents - balanced) [Recommended]
  3. Large (8 agents - comprehensive)
> Select: 2

Step 5/5: Review Configuration

┌─────────────────────────────────────────┐
│ Meeting Preview                         │
├─────────────────────────────────────────┤
│ Topic: How do I optimize database...    │
│ Type: Technical Review                  │
│ Length: Standard (3-4 rounds)           │
│ Agents: 5 (auto-selected)               │
│                                         │
│ Estimated Cost: $0.45 - $0.75          │
│ Estimated Time: 8-12 minutes           │
└─────────────────────────────────────────┘

[C]reate and run  [S]ave as template  [E]dit  [c]ancel
> c

✓ Creating meeting...
✓ Auto-selecting 5 agents based on topic...
✓ Meeting created: a3f2e1d4-...

[Optional] Save this configuration as a template?
  Template name (or press Enter to skip): db-optimization

✓ Template saved: ~/.theboard/templates/db-optimization.json

Run: board run --last
View: board status --last
Reuse: board create --template db-optimization
```

**Implementation:**
- Separate command: `src/theboard/cli_commands/wizard.py`
- Uses Rich prompts for interactive flow
- Generates `MeetingConfig` object
- Optionally saves to `~/.theboard/templates/`
- Seamlessly hands off to existing `create_meeting()` service

---

### Insight 4: Smart Defaults via Topic Analysis
**Description:** Parse topic text to infer optimal config (agent count, meeting type, convergence profile) automatically

**Source:** SCAMPER (Reverse: smart defaults), Six Hats (Green: NLP config)

**Impact:** High | **Effort:** High

**Why it matters:** Eliminates 80% of config decisions for users. Topic analysis provides intelligent defaults that users can override, reducing cognitive load while maintaining control.

**Examples:**
```
Topic: "Assess security risks in authentication redesign"
→ meeting_type: risk_assessment
→ persona_config: security expert, backend architect, compliance officer
→ convergence_profile: strict
→ synthesis_style: structured_doc

Topic: "Brainstorm creative marketing campaigns for Q2 launch"
→ meeting_type: brainstorm
→ creativity_spectrum: 0.8 (high creativity)
→ convergence_profile: exploratory
→ agent_pool_preset: large

Topic: "Should we migrate from MySQL to PostgreSQL?"
→ meeting_type: debate
→ debate_mode: true
→ agent_pool_preset: small (focused debate)
→ convergence_profile: balanced
```

**Implementation Strategy:**
1. **Keyword Extraction** - Extract domain keywords (security, marketing, migration)
2. **Intent Classification** - Question vs imperative → meeting_type mapping
3. **Complexity Estimation** - Token count, technical density → agent_pool_preset
4. **Domain Matching** - Keywords → relevant agent personas

**Algorithm:**
```python
def analyze_topic(topic: str) -> SmartDefaults:
    # Extract keywords
    keywords = extract_keywords(topic)

    # Classify intent
    if is_question(topic):
        if "risk" in keywords or "concern" in keywords:
            meeting_type = MeetingType.RISK_ASSESSMENT
        elif "should" in topic or "vs" in topic:
            meeting_type = MeetingType.DEBATE
        else:
            meeting_type = MeetingType.RESEARCH
    else:
        meeting_type = MeetingType.BRAINSTORM

    # Estimate complexity
    complexity = estimate_complexity(topic, keywords)
    agent_pool = {
        "low": AgentPoolSize.SMALL,
        "medium": AgentPoolSize.MEDIUM,
        "high": AgentPoolSize.LARGE
    }[complexity]

    # Match agent personas
    relevant_agents = match_personas_to_keywords(keywords)

    return SmartDefaults(
        meeting_type=meeting_type,
        agent_pool_preset=agent_pool,
        persona_config=relevant_agents,
        convergence_profile=infer_convergence(meeting_type)
    )
```

**User Experience:**
```bash
$ board create -t "Assess security risks in authentication redesign"

Smart defaults detected:
  Meeting Type: Risk Assessment (from keywords: assess, risks, security)
  Agent Pool: Medium (5 agents including security experts)
  Convergence: Strict (thorough risk coverage)

[A]ccept defaults  [C]ustomize  [v]iew details
> a

✓ Meeting created with smart defaults
```

---

### Insight 5: Cost Preview as Required Step
**Description:** Before executing, show estimated cost/time based on config + topic complexity, with interactive adjustment

**Source:** Six Hats (Red: anxiety about costs), Starbursting (When: see estimates)

**Impact:** Medium | **Effort:** Medium

**Why it matters:**
- **Prevents surprises** - Users know cost before committing
- **Builds trust** - Transparency about resource consumption
- **Enables optimization** - Users can adjust config to meet budget
- **Educates** - Shows cost drivers (agent count, rounds, model tiers)

**Calculation Logic:**
```python
def estimate_meeting_cost(config: MeetingConfig) -> CostEstimate:
    # Base token estimates per agent per round
    tokens_per_response = estimate_tokens(config.topic)

    # Agent count from preset or manual
    agent_count = resolve_agent_count(config.agent_pool_preset)

    # Expected rounds (max_rounds weighted by convergence profile)
    convergence_factor = {
        ConvergenceProfile.STRICT: 0.6,      # Likely exits early
        ConvergenceProfile.BALANCED: 0.8,    # Usually hits 80% of max
        ConvergenceProfile.EXPLORATORY: 1.0  # Often hits max_rounds
    }
    expected_rounds = config.max_rounds * convergence_factor[config.convergence_profile]

    # Model pricing
    model_cost = get_model_cost(config.model_strategy)

    # Calculate
    total_tokens = tokens_per_response * agent_count * expected_rounds
    estimated_cost = (total_tokens / 1_000_000) * model_cost

    # Duration estimate (2-4 min per round)
    estimated_duration = expected_rounds * 3  # minutes

    return CostEstimate(
        min_cost=estimated_cost * 0.7,  # Best case (early convergence)
        max_cost=estimated_cost * 1.5,  # Worst case (max rounds + retries)
        expected_cost=estimated_cost,
        min_duration=expected_rounds * 2,
        max_duration=config.max_rounds * 4,
        breakdown={
            "agents": agent_count,
            "expected_rounds": expected_rounds,
            "tokens_per_round": tokens_per_response * agent_count,
            "model": config.model_strategy,
        }
    )
```

**Display:**
```
┌─────────────────────────────────────────┐
│ Cost & Time Estimate                    │
├─────────────────────────────────────────┤
│ Estimated Cost: $0.45 - $0.85          │
│ Expected: $0.62                         │
│                                         │
│ Estimated Time: 8-15 minutes           │
│ Expected: 11 minutes                    │
│                                         │
│ Estimated Comments: 35-50              │
├─────────────────────────────────────────┤
│ Breakdown                               │
│ • 5 agents × ~3.2 rounds               │
│ • ~2000 tokens/response                │
│ • Model: claude-sonnet-3.5             │
│   ($3.00 per 1M tokens)                │
│ • Convergence likely after round 3     │
└─────────────────────────────────────────┘

Budget: $0.62 of $5.00 available (12%)

[C]ontinue  [A]djust config  [c]ancel
```

**Adjustment Options:**
```
Cost too high? Try these optimizations:
  1. Reduce agent count (5 → 3 agents): ~$0.40
  2. Switch to cost-optimized models: ~$0.25
  3. Lower max rounds (5 → 3 rounds): ~$0.38
  4. Combine all: ~$0.18

[S]elect optimization  [k]eep current config
```

---

### Insight 6: Validation as Guidance, Not Gatekeeping
**Description:** Use warnings for non-optimal configs, errors only for impossible combinations

**Source:** Six Hats (Black: validation complexity), Starbursting (What: validation rules)

**Impact:** Medium | **Effort:** Low

**Why it matters:** Users experiment and learn through iteration. Blocking valid but suboptimal configs frustrates exploration. Warnings educate without preventing execution.

**Validation Categories:**

**1. Errors (Blocks Execution):**
```python
# Impossible configurations
- model_budget < minimum_viable_cost
- max_rounds < min_rounds
- agent_count = 0 with auto_select = False
- conflicting modes (debate_mode + research_mode)
- invalid template schema

# Examples:
❌ Error: Budget $0.10 insufficient for min_rounds=2
   Minimum cost for this config: $0.25
   Increase budget or reduce min_rounds

❌ Error: Cannot enable both debate_mode and research_mode
   Choose one specialized mode or use default
```

**2. Warnings (Allows Execution):**
```python
# Suboptimal but valid
- high agent_count + low max_rounds (may not converge)
- low novelty_threshold + exploratory convergence (redundant)
- expensive model + cost_optimized strategy (contradictory)
- large agent pool + quick meeting length (mismatch)

# Examples:
⚠️  Warning: High agent count (8) with low max_rounds (2)
   Large teams typically need 4-5 rounds to converge
   Recommendation: Increase max_rounds to 4

⚠️  Warning: Cost-optimized strategy with premium model override
   Override will prevent cost optimization
   Remove model override to enable cost optimization
```

**3. Tips (Suggestions):**
```python
# Helpful but non-critical
- topic suggests debate_mode
- similar templates available
- past meetings with similar config
- alternative model strategies

# Examples:
💡 Tip: Topic contains "vs" - consider debate_mode
   This enables agents to argue opposing viewpoints

💡 Tip: Template "security-audit" matches this topic
   Load with: --template security-audit
```

**Implementation:**
```python
class ConfigValidator:
    def validate(self, config: MeetingConfig) -> ValidationResult:
        errors = []
        warnings = []
        tips = []

        # Error: Budget validation
        min_cost = estimate_minimum_cost(config)
        if config.model_budget and config.model_budget < min_cost:
            errors.append(
                f"Budget ${config.model_budget:.2f} insufficient for min_rounds={config.min_rounds}. "
                f"Minimum cost: ${min_cost:.2f}"
            )

        # Warning: Agent/round mismatch
        if config.agent_pool_preset == AgentPoolSize.LARGE and config.max_rounds < 4:
            warnings.append(
                f"High agent count ({get_agent_count(config.agent_pool_preset)}) with low max_rounds ({config.max_rounds}). "
                f"Recommendation: Increase max_rounds to 4"
            )

        # Tip: Mode suggestion
        if "vs" in config.topic.lower() and not config.debate_mode:
            tips.append(
                "Topic contains 'vs' - consider debate_mode for opposing viewpoints"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            tips=tips
        )
```

**Display:**
```bash
$ board create --config meeting.json

Validating configuration...

⚠️  2 warnings found:
  • High agent count (8) with low max_rounds (2)
    Recommendation: Increase max_rounds to 4

  • Cost-optimized strategy with premium model override
    Remove model override to enable cost optimization

💡 1 tip:
  • Template "database-optimization" matches this topic
    Load with: --template database-optimization

[C]ontinue anyway  [F]ix warnings  [c]ancel
```

---

### Insight 7: Template Marketplace Pattern (Future)
**Description:** Community-contributed templates with ratings, usage stats, version compatibility (v2 feature)

**Source:** Six Hats (Green: innovation), Starbursting (Who: team templates)

**Impact:** Low (future feature) | **Effort:** High

**Why it matters:** Captures collective intelligence about effective meeting configurations. Teams share successful patterns. Users learn through curated examples. Not critical for v1, but designing schema now enables future extension.

**Template Metadata (v2 Schema):**
```json
{
  "template_metadata": {
    "name": "security-audit-template",
    "version": "1.2.0",
    "author": "security-team@company.com",
    "description": "Comprehensive security risk assessment with compliance focus",
    "tags": ["security", "compliance", "risk-assessment", "audit"],
    "created_at": "2026-01-15T10:30:00Z",
    "updated_at": "2026-02-01T14:20:00Z",
    "compatibility": {
      "min_theboard_version": "0.2.0",
      "required_features": ["debate_mode", "expert_review_mode"]
    },
    "usage_stats": {
      "adoption_count": 47,
      "success_rate": 0.89,
      "avg_cost": 0.72,
      "avg_duration_minutes": 12
    },
    "rating": {
      "average": 4.6,
      "count": 23
    }
  },
  "config": {
    // Actual meeting configuration
  }
}
```

**Marketplace Features (Future):**
- **Template Discovery:** `board templates browse --tags security,compliance`
- **Template Rating:** `board templates rate security-audit 5 "Very thorough coverage"`
- **Template Sharing:** `board templates publish my-template --public`
- **Template Forking:** `board templates fork security-audit --name custom-security`
- **Usage Analytics:** Track which templates drive successful meetings
- **Version Control:** Template updates with backward compatibility

**Design Now, Implement Later:**
- Schema supports metadata fields (ignored if missing)
- Template storage directory structure allows remote templates
- Validation gracefully handles unknown fields
- CLI commands stubbed with "Coming soon" messages

---

## Statistics

- **Total Ideas:** 48 configuration options
- **Categories:** 8
- **Key Insights:** 7
- **Techniques Applied:** 3 (SCAMPER, Starbursting, Six Thinking Hats)

---

## Recommended Next Steps

### Immediate (Sprint Implementation)

1. **Define JSON Schema**
   - Create `src/theboard/schemas/meeting_config.py`
   - Implement three-tier validation (Essential, Common, Advanced)
   - Add schema validation with error messages

2. **Implement Template System**
   - Template storage: `~/.theboard/templates/`
   - Default templates: quick_brainstorm, thorough_analysis, risk_assessment
   - CLI commands: `board templates list`, `board create --template <name>`

3. **Build Wizard Command**
   - New command: `board wizard`
   - Rich interactive prompts for essential tier
   - Cost preview before execution
   - Template export at completion

4. **Add Cost Estimation**
   - `estimate_meeting_cost()` function
   - Display before execution
   - Interactive adjustment options

5. **Enhance Validation**
   - Error/Warning/Tip categorization
   - Config conflict detection
   - Helpful suggestions

### Future Enhancements (v2)

6. **Smart Defaults via NLP**
   - Topic analysis for meeting_type inference
   - Complexity estimation for agent_pool_preset
   - Keyword matching for persona_config

7. **Template Marketplace**
   - Template metadata schema
   - Rating and adoption tracking
   - Community sharing platform
   - Version compatibility checking

### Documentation

8. **Create Documentation**
   - Template format guide
   - Config option reference (all 48 params)
   - Wizard walkthrough with screenshots
   - Best practices for each meeting_type

---

*Generated by BMAD Method v6 - Creative Intelligence*
*Session duration: 45 minutes*
*Project: TheBoard - Multi-Agent Brainstorming System*
