# Meeting Creation Wizard Guide

## Overview

The wizard provides a guided, interactive experience for creating meetings. Perfect for:
- **New users** - No need to memorize CLI flags
- **Quick setup** - 4 simple questions, cost preview, done
- **Learning tool** - Understand config options through interaction
- **Template generation** - Export wizard configs for reuse

---

## Quick Start

```bash
board wizard create
```

That's it! The wizard handles everything.

---

## Wizard Flow

### Step 1: Meeting Topic
```
What are we brainstorming? (10-500 characters)

> How do I optimize database query performance for user analytics?
```

**Validation:**
- Minimum 10 characters
- Maximum 500 characters
- Topic influences cost estimates and agent selection

---

### Step 2: Meeting Type
```
Select the meeting style:

  1. General ideation and creative thinking
  2. Identify and evaluate risks
  3. Evaluate technical approach or design
  4. Explore opposing viewpoints
  5. Fact-based analysis and investigation

> Select [1/2/3/4/5]: 3
```

**Options:**
1. **Brainstorm** - Open-ended ideation
2. **Risk Assessment** - Identify potential problems
3. **Technical Review** - Evaluate architecture/approach
4. **Debate** - Argue pros/cons
5. **Research** - Fact-based investigation

**Note:** Meeting type is stored for future features (specialized agent selection, output formatting)

---

### Step 3: Meeting Length
```
How thorough should the discussion be?

  1. Quick (2 rounds, ~5 min, $0.10-$0.30)
  2. Standard (4 rounds, ~12 min, $0.30-$0.70) [Recommended]
  3. Thorough (5 rounds, ~17 min, $0.70-$1.50)

> Select [1/2/3] (2): 2
```

**Length Strategies:**

| Strategy | Rounds | Duration | Est. Cost | Use Case |
|----------|--------|----------|-----------|----------|
| Quick | 2 | ~5 min | $0.10-$0.30 | Fast ideation, time pressure |
| Standard | 4 | ~12 min | $0.30-$0.70 | Balanced coverage (default) |
| Thorough | 5 | ~17 min | $0.70-$1.50 | Deep analysis, complex topics |

**Convergence behavior:**
- Quick: Exits early if novelty drops (90% chance before max rounds)
- Standard: Moderate convergence (75% chance by round 3)
- Thorough: Runs full duration (100% likely to hit max rounds)

---

### Step 4: Agent Team Size
```
How many agents should participate?

  1. Small (3 agents - focused discussion)
  2. Medium (5 agents - balanced perspectives) [Recommended]
  3. Large (8 agents - comprehensive coverage)

> Select [1/2/3] (2): 2
```

**Agent Pool Sizes:**

| Size | Agents | Best For |
|------|--------|----------|
| Small | 3 | Focused debates, quick decisions |
| Medium | 5 | Balanced perspectives (default) |
| Large | 8 | Comprehensive analysis, diverse viewpoints |

**Agent Selection:**
- Agents auto-selected based on topic keywords
- Example: Topic with "security" → includes SecurityExpert agent
- See auto-selected agents in cost preview

---

## Cost Preview

Before creating the meeting, wizard displays comprehensive cost/time estimate:

```
┌─────────────────────────────────────────┐
│ Cost & Time Estimate                    │
├─────────────────────────────────────────┤
│ Estimated Cost: $0.25 - $0.65          │
│ Expected: $0.42                         │
│                                         │
│ Estimated Time: 8-15 minutes           │
│ Expected: 12 minutes                    │
│                                         │
│ Expected Comments: 45                   │
│                                         │
│ Breakdown:                              │
│ • 5 agents × ~3.2 rounds               │
│ • ~2,000 tokens/round                  │
│ • Model: deepseek ($0.14 per 1M tokens)│
│ • Convergence likelihood: 75%          │
└─────────────────────────────────────────┘

Create and run this meeting? [Y/n]:
```

**Cost Calculation:**
1. **Base tokens/response** - Estimated from topic complexity (1500-2200 tokens)
2. **Notetaker overhead** - Additional 500 tokens per response
3. **Expected rounds** - Factored by convergence likelihood
4. **Model pricing** - Currently uses DeepSeek ($0.14/1M tokens)

**Ranges:**
- **Min cost** - Early convergence (70% of expected rounds)
- **Max cost** - No convergence + retries (120% of max rounds)
- **Expected** - Realistic estimate (convergence factor applied)

---

## Template Export

After creating a meeting, wizard offers to save config as template:

```
Save this configuration as a template for future use? [y/N]: y

Template name (e.g., 'quick-brainstorm', 'security-review'): db-optimization

✓ Template saved: /home/user/.theboard/templates/db-optimization.json

Reuse with: board create --template db-optimization
```

**Template Structure:**
```json
{
  "name": "db-optimization",
  "description": "Technical Review meeting template",
  "config": {
    "meeting_type": "technical_review",
    "length_strategy": "standard",
    "agent_pool_size": "medium",
    "topic": "<TOPIC_PLACEHOLDER>"
  }
}
```

**Using Templates:**
```bash
# List available templates
ls ~/.theboard/templates/

# Use template (topic required)
board create --template db-optimization -t "Optimize query performance"
```

---

## Examples

### Example 1: Quick Brainstorm
```
Topic: Ideas for improving user onboarding flow
Type: Brainstorm
Length: Quick
Agents: Small

Result:
- 2 rounds, 3 agents
- ~4 minutes
- $0.18
- 12-18 comments
```

### Example 2: Security Review
```
Topic: Assess authentication redesign for security vulnerabilities
Type: Risk Assessment
Length: Thorough
Agents: Large

Result:
- 5 rounds, 8 agents
- ~18 minutes
- $1.35
- 80-120 comments
```

### Example 3: Technical Decision
```
Topic: Should we migrate from MySQL to PostgreSQL?
Type: Debate
Length: Standard
Agents: Medium

Result:
- 4 rounds, 5 agents
- ~11 minutes
- $0.52
- 40-50 comments
```

---

## Tips

### Choosing Length Strategy

**Use Quick when:**
- Time constrained (standups, quick checks)
- Simple, focused topic
- Just need 2-3 perspectives

**Use Standard when:**
- Most scenarios (default for good reason)
- Balanced coverage needed
- Moderate complexity

**Use Thorough when:**
- Complex, high-stakes topics
- Need comprehensive analysis
- Cost is not a concern

### Choosing Agent Pool Size

**Use Small when:**
- Focused debate (2-3 viewpoints enough)
- Quick decision needed
- Topic is narrow in scope

**Use Medium when:**
- Most scenarios (balanced default)
- Need diverse perspectives
- Moderate topic complexity

**Use Large when:**
- Complex, multi-faceted topics
- Want comprehensive coverage
- Multiple domains involved (security + backend + frontend)

### Cost Optimization

To reduce costs:
1. Choose **Quick** length strategy
2. Select **Small** agent pool
3. Keep topic concise (complexity drives token count)

To maximize quality:
1. Choose **Thorough** length strategy
2. Select **Large** agent pool
3. Accept higher cost for comprehensive coverage

---

## Comparison: Wizard vs Direct CLI

### Wizard Approach
```bash
board wizard create
# Interactive prompts
# Cost preview
# Template export option
```

**Pros:**
- Guided experience
- No flag memorization
- Cost visibility
- Learning tool

**Cons:**
- Slower (4 prompts)
- Not scriptable

### Direct CLI Approach
```bash
board create \
  -t "Optimize database performance" \
  --strategy sequential \
  --max-rounds 4 \
  --agent-count 5 \
  --auto-select \
  --model deepseek
```

**Pros:**
- Fast (one command)
- Scriptable
- Full control

**Cons:**
- Must remember flags
- No cost preview
- Easy to misconfigure

### Recommendation

- **New users:** Start with wizard, export templates
- **Power users:** Use templates or direct CLI
- **Teams:** Share wizard-generated templates

---

## Troubleshooting

### "Topic too short"
```
Topic too short (minimum 10 characters)
```

**Fix:** Provide more detailed topic (at least 10 characters)

### "Template name already exists"
```
Error: Template 'quick-brainstorm' already exists
```

**Fix:** Choose different name or delete existing template:
```bash
rm ~/.theboard/templates/quick-brainstorm.json
```

### "Meeting creation failed"
```
Error: Failed to create meeting: Database connection refused
```

**Fix:** Ensure database is running:
```bash
# Check services
docker ps

# Start services if needed
docker-compose up -d
```

---

## Advanced Features (Future)

### Smart Defaults (v2)
Wizard will analyze topic text to suggest:
- Meeting type (detect "risk", "vs", "how to")
- Agent pool size (estimate complexity)
- Relevant agents (keyword matching)

### Template Marketplace (v2)
- Community-contributed templates
- Rating and adoption tracking
- Version compatibility
- Usage analytics

### Natural Language Config (v3)
```
"I want a quick brainstorm with 3 agents focusing on technical risks"
→ Automatically configures meeting
```

---

## Architecture Notes

### Components

**Schemas** (`/home/delorenj/code/theboard/src/theboard/schemas.py`)
- `MeetingType` - Enum for meeting types
- `LengthStrategy` - Enum for length presets
- `AgentPoolSize` - Enum for agent pool sizes
- `WizardConfig` - Config model with `.to_meeting_create_params()`
- `CostEstimate` - Cost/time estimation result

**Cost Estimator** (`/home/delorenj/code/theboard/src/theboard/services/cost_estimator.py`)
- `estimate_meeting_cost(config)` - Returns CostEstimate
- Factors: topic complexity, rounds, agents, model pricing, convergence
- Ranges: min (early exit), expected (realistic), max (no convergence)

**Wizard Command** (`/home/delorenj/code/theboard/src/theboard/cli_commands/wizard.py`)
- Interactive Rich prompts
- 4-step flow (topic, type, length, agents)
- Cost preview before creation
- Template export after creation
- Seamless handoff to `create_meeting()` service

### Integration Points

**Backward Compatible:**
- Existing `board create` unchanged
- Wizard uses same `create_meeting()` service
- Templates map to existing parameters

**Template Storage:**
- Location: `~/.theboard/templates/*.json`
- Format: JSON with name, description, config
- Future: Database storage for marketplace

**Cost Model:**
- Currently: Static DeepSeek pricing ($0.14/1M)
- Future: Dynamic pricing from preferences
- Future: Multi-model cost comparison

---

## See Also

- [Meeting CLI Improvements](./MEETING_CLI_IMPROVEMENTS.md) - TUI enhancements, cost preview
- [Brainstorming Session](./brainstorming-meeting-config-wizard.md) - Full ideation document
- [Configuration Reference](./CONFIG.md) - All 48 config options (when implemented)
