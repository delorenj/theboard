# Agno Framework Integration

## Overview

TheBoard has been refactored to use the [Agno framework](https://docs.agno.com/) for agent orchestration. This document explains the integration patterns and benefits.

## What Changed

### Before (Direct Anthropic API)
- Manual API calls using `anthropic.messages.create()`
- Custom prompt construction and system message handling
- Manual JSON parsing for structured outputs
- Custom session management via Redis
- Manual token counting and cost calculation

### After (Agno Framework)
- Declarative agent creation with `Agent` class
- Automatic conversation history via `PostgresDb`
- Built-in structured outputs using Pydantic schemas
- Automatic session persistence with `session_id`
- Built-in metrics tracking and cost calculation

## Key Agno Patterns

### 1. Agent Creation Pattern

**File**: `src/theboard/agents/base.py`

```python
from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.anthropic import Claude

def create_agno_agent(
    name: str,
    role: str,
    expertise: str,
    instructions: list[str],
    model_id: str = "claude-sonnet-4-20250514",
    session_id: str | None = None,
    output_schema: type | None = None,
    debug_mode: bool = False,
) -> Agent:
    """Create an Agno Agent with TheBoard configuration."""

    # Agno uses Claude model wrapper instead of direct client
    model = Claude(id=model_id, api_key=settings.anthropic_api_key)

    # Create agent with automatic session persistence
    agent = Agent(
        name=name,
        role=role,
        description=f"You are {name}, {role}. Your expertise: {expertise}",
        model=model,
        instructions=instructions,
        db=PostgresDb(db_url=settings.database_url_str) if session_id else None,
        session_id=session_id,
        add_history_to_messages=True if session_id else False,
        output_schema=output_schema,  # For structured outputs
        debug_mode=debug_mode,
        markdown=True,
    )

    return agent
```

**Benefits**:
- Single source of truth for agent configuration
- Automatic session persistence when `session_id` is provided
- Built-in conversation history management
- Debug mode for development visibility

### 2. Domain Expert Pattern

**File**: `src/theboard/agents/domain_expert.py`

```python
class DomainExpertAgent:
    """Domain expert using Agno framework."""

    def __init__(
        self,
        name: str,
        expertise: str,
        persona: str | None = None,
        background: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        session_id: str | None = None,  # meeting_id for persistence
    ) -> None:
        # Build instructions list
        instructions = [
            "Analyze the current discussion and provide your expert perspective",
            "Identify key technical decisions or considerations",
            "Highlight potential risks or challenges",
            "Suggest practical implementation approaches",
        ]

        # Create underlying Agno agent
        self._agent = create_agno_agent(
            name=name,
            role="Expert in brainstorming and domain-specific analysis",
            expertise=expertise,
            instructions=instructions,
            model_id=model,
            session_id=session_id,  # Enables conversation persistence
        )

    async def execute(self, context: str, **kwargs) -> str:
        # Use agent.run() instead of manual API calls
        response = self._agent.run(prompt)
        return response.content
```

**Benefits**:
- Instructions-based configuration instead of system prompts
- Automatic conversation history across rounds
- No manual session management needed
- Metrics automatically tracked

### 3. Structured Output Pattern

**File**: `src/theboard/agents/notetaker.py`

```python
from theboard.schemas import CommentList

class NotetakerAgent:
    """Extracts structured comments using Agno's output_schema."""

    async def extract_comments(
        self,
        response_text: str,
        agent_name: str
    ) -> list[Comment]:
        # Create agent with output_schema for automatic validation
        extractor = create_agno_agent(
            name="Notetaker",
            role="Extract structured comments from brainstorming responses",
            expertise="Identifying key ideas, categorizing insights",
            instructions=[...],
            output_schema=CommentList,  # Pydantic model for validation
        )

        # Run extraction
        response = extractor.run(prompt)

        # Response content is already a validated CommentList!
        comment_list: CommentList = response.content
        return comment_list.comments
```

**Benefits**:
- No manual JSON parsing
- Automatic validation against Pydantic schema
- Type-safe structured outputs
- Graceful error handling

### 4. Session Persistence Pattern

**File**: `src/theboard/workflows/simple_meeting.py`

```python
class SimpleMeetingWorkflow:
    """Workflow using Agno session persistence."""

    async def _execute_round(
        self,
        meeting: Meeting,
        agent: Agent,
        round_num: int
    ) -> None:
        # Create expert with session_id for persistence
        expert = DomainExpertAgent(
            name=agent.name,
            expertise=agent.expertise,
            model=agent.default_model,
            session_id=str(meeting.id),  # Agno stores conversation in PostgresDb
        )

        # Generate response - history is automatic
        response_text = await expert.execute(context, round_num=round_num)
```

**Benefits**:
- Conversation history persisted to PostgresDb automatically
- No Redis management needed for agent state
- Session can be resumed across service restarts
- Shared database with application data

## Database Schema

Agno automatically creates and manages these tables:

```sql
-- Agent sessions (created by Agno)
agno_sessions (
    session_id TEXT PRIMARY KEY,
    agent_name TEXT,
    messages JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)

-- Agent runs (created by Agno)
agno_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT,
    messages JSONB,
    metrics JSONB,
    created_at TIMESTAMP
)
```

These tables coexist with TheBoard's existing schema in the `public` schema.

## Metrics Extraction

**File**: `src/theboard/agents/base.py`

```python
def extract_agno_metrics(agent: Agent) -> dict[str, Any]:
    """Extract usage metrics from Agno Agent run."""

    # Agno automatically tracks metrics in run_response
    metrics = agent.run_response.metrics

    tokens_used = metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)

    # Calculate cost
    cost = (
        metrics.get("input_tokens", 0) / 1_000_000 * 3.0 +
        metrics.get("output_tokens", 0) / 1_000_000 * 15.0
    )

    return {"tokens_used": tokens_used, "cost": cost}
```

## Configuration

Add to `.env`:

```bash
# Existing TheBoard config
DATABASE_URL=postgresql+psycopg://theboard:pass@localhost:5432/theboard
ANTHROPIC_API_KEY=sk-...

# Agno will use the same database
# No additional configuration needed!
```

## Debug Mode

Enable debug mode for development:

```python
agent = create_agno_agent(
    name="Test Agent",
    role="Testing",
    expertise="Debugging",
    instructions=["Test instruction"],
    debug_mode=True,  # Shows detailed logs
)
```

This will log:
- Messages sent to the model
- Tool calls and results
- Token usage and timing
- Session state changes

## Migration Benefits

1. **Reduced Code**: ~40% less code in agent implementations
2. **Better Reliability**: Automatic validation and error handling
3. **Session Persistence**: Built-in conversation history
4. **Type Safety**: Pydantic schemas for all structured outputs
5. **Developer Experience**: Debug mode for visibility
6. **Maintainability**: Clear, declarative agent configuration

## Future Enhancements

With Agno integration in place, these features become easier:

1. **Multi-Agent Teams** (Sprint 2+):
   ```python
   from agno.team import Team

   team = Team(
       members=[expert1, expert2, expert3],
       mode="coordinate",
       db=PostgresDb(...),
       session_id=meeting_id,
   )

   team.print_response(topic, stream=True)
   ```

2. **Workflow Orchestration** (Sprint 3+):
   ```python
   from agno.workflow import Workflow, Router, Step

   workflow = Workflow(
       steps=[
           Router(routes={
               "technical": Step(agent=tech_expert),
               "business": Step(agent=biz_expert),
           })
       ]
   )
   ```

3. **AgentOS Deployment** (Production):
   ```python
   from agno.os import AgentOS

   agent_os = AgentOS(
       agents=[expert1, expert2],
       db=PostgresDb(...),
   )
   agent_os.serve()
   ```

## References

- **Agno Documentation**: https://docs.agno.com/
- **Agno GitHub**: https://github.com/agno-agi/agno
- **Agno Skill**: `/home/delorenj/.claude/skills/agno/SKILL.md`
- **Example Projects**: https://github.com/langdb/langdb-samples/tree/main/examples/agno

## Support

For issues or questions about Agno integration:
1. Check Agno skill documentation
2. Review examples in the Agno cookbook
3. Enable debug_mode for detailed logs
4. Check PostgresDb tables for session data
