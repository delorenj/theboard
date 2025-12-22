"""Base agent interface using Agno framework.

This module provides the base agent functionality using Agno's Agent class
instead of direct Anthropic API calls. Agno handles session management,
memory, and structured outputs automatically.
"""

import logging
from typing import Any

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.models.openrouter import OpenRouter

from theboard.config import settings
from theboard.preferences import get_preferences_manager

logger = logging.getLogger(__name__)


def get_agno_db() -> PostgresDb:
    """Get Agno PostgresDb instance for session persistence.

    Agno Pattern: Use PostgresDb for automatic session storage and retrieval.
    Sessions are stored in the database and linked to meeting_id via session_id.

    Returns:
        PostgresDb instance configured with TheBoard connection
    """
    return PostgresDb(
        db_url=settings.database_url_str,
        db_schema="public",  # Use default schema where meetings table exists
    )


def create_agno_agent(
    name: str,
    role: str,
    expertise: str,
    instructions: list[str],
    model_id: str | None = None,
    agent_type: str = "worker",
    session_id: str | None = None,
    output_schema: type | None = None,
    debug_mode: bool = False,
    model_override: str | None = None,
) -> Agent:
    """Create an Agno Agent with TheBoard configuration.

    Agno Pattern: Use Agent class for all LLM interactions. This provides:
    - Automatic session persistence via PostgresDb
    - Built-in conversation history management
    - Structured outputs via Pydantic schemas
    - Debug mode for development visibility
    - Cost and token tracking

    Args:
        name: Agent name
        role: Agent role description
        expertise: Domain expertise description
        instructions: List of specific instructions for the agent
        model_id: OpenRouter model ID (optional, uses preferences if None)
        agent_type: Agent type for preference lookup (worker, leader, notetaker, compressor)
        session_id: Optional session ID for persistence (typically meeting_id)
        output_schema: Optional Pydantic model for structured output
        debug_mode: Enable debug logging for Agno
        model_override: CLI flag override (highest precedence)

    Returns:
        Configured Agno Agent instance
    """
    # Model selection with precedence logic
    if model_id is None:
        manager = get_preferences_manager()
        model_id = manager.get_model_for_agent(
            agent_name=name,
            agent_type=agent_type,
            cli_override=model_override,
        )
    # Agno Pattern: OpenRouter model for unified LLM access
    model = OpenRouter(
        id=model_id,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    # Agno Pattern: Build complete description combining role and expertise
    description = f"You are {name}, {role}. Your expertise: {expertise}"

    # Agno Pattern: Create agent with all configuration
    # Session persistence is automatic when db and session_id are provided
    agent = Agent(
        name=name,
        role=role,
        description=description,
        model=model,
        instructions=instructions,
        db=get_agno_db() if session_id else None,
        session_id=session_id,
        num_history_messages=10 if session_id else 0,  # Include history when session exists
        output_schema=output_schema,
        debug_mode=debug_mode or settings.debug,
        markdown=True,  # Enable markdown formatting in responses
    )

    logger.info(
        "Created Agno agent: %s (model=%s, session=%s, structured_output=%s)",
        name,
        model_id,
        session_id or "none",
        output_schema.__name__ if output_schema else "none",
    )

    return agent


def extract_agno_metrics(agent: Agent) -> dict[str, Any]:
    """Extract usage metrics from Agno Agent run.

    Agno Pattern: Agents track metrics automatically during execution.
    Access via agent.run_response after calling agent.run() or agent.print_response().

    Args:
        agent: Agno Agent that has executed a run

    Returns:
        Dictionary with tokens_used and estimated cost
    """
    if not hasattr(agent, "run_response") or not agent.run_response:
        logger.warning("No run_response available for agent %s", agent.name)
        return {"tokens_used": 0, "cost": 0.0}

    # Agno stores metrics in run_response.metrics
    metrics = getattr(agent.run_response, "metrics", {})

    # Extract token usage
    tokens_used = metrics.get("input_tokens", 0) + metrics.get("output_tokens", 0)

    # Calculate cost (Claude Sonnet 4 rates)
    input_cost_per_mtok = 3.0  # $3 per million tokens
    output_cost_per_mtok = 15.0  # $15 per million tokens
    cost = (
        metrics.get("input_tokens", 0) / 1_000_000 * input_cost_per_mtok
        + metrics.get("output_tokens", 0) / 1_000_000 * output_cost_per_mtok
    )

    return {
        "tokens_used": tokens_used,
        "cost": cost,
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
    }
