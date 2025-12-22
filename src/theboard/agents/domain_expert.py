"""Domain Expert Agent implementation using Agno framework.

This module implements domain expert agents using Agno's Agent class.
Instead of manual prompt construction and API calls, Agno handles:
- Agent role and expertise configuration
- Automatic session persistence
- Conversation history management
- Token and cost tracking
"""

import logging
from typing import Any

from agno.agent import Agent

from theboard.agents.base import create_agno_agent, extract_agno_metrics

logger = logging.getLogger(__name__)


class DomainExpertAgent:
    """Domain expert agent using Agno framework for brainstorming sessions.

    Agno Pattern: This class is a wrapper around Agno Agent that provides
    TheBoard-specific configuration and execution logic. The actual Agent
    instance is created with the proper instructions and session management.
    """

    def __init__(
        self,
        name: str,
        expertise: str,
        persona: str | None = None,
        background: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        session_id: str | None = None,
    ) -> None:
        """Initialize Domain Expert Agent.

        Args:
            name: Agent name
            expertise: Agent expertise description
            persona: Optional persona description
            background: Optional background information
            model: LLM model to use
            session_id: Optional session ID for conversation persistence (meeting_id)
        """
        self.name = name
        self.expertise = expertise
        self.persona = persona
        self.background = background
        self.model = model
        self.session_id = session_id

        # Agno Pattern: Build instructions list for the agent
        # Instructions guide the agent's behavior without system prompts
        instructions = [
            "Analyze the current discussion and provide your expert perspective",
            "Identify key technical decisions or considerations",
            "Highlight potential risks or challenges",
            "Suggest practical implementation approaches",
            "Be concise but thorough - focus on actionable insights",
            "Build on previous ideas when they exist",
        ]

        # Add persona and background to instructions if provided
        if self.persona:
            instructions.append(f"Adopt this persona: {self.persona}")

        if self.background:
            instructions.append(f"Draw on your background: {self.background}")

        # Agno Pattern: Create the underlying Agno Agent
        # This handles all LLM interaction, session persistence, and metrics
        self._agent: Agent = create_agno_agent(
            name=name,
            role="Expert in brainstorming and domain-specific analysis",
            expertise=expertise,
            instructions=instructions,
            model_id=model,
            session_id=session_id,
            debug_mode=False,  # Set True for development debugging
        )

        logger.info(
            "Created DomainExpertAgent: %s (expertise=%s, session=%s)",
            name,
            expertise,
            session_id or "none",
        )

    async def execute(self, context: str, **kwargs: Any) -> str:
        """Generate expert response based on context using Agno Agent.

        Agno Pattern: Use agent.run() instead of manual API calls.
        The agent automatically manages conversation history when session_id is set.

        Args:
            context: Current discussion context
            **kwargs: Additional arguments (round_num, etc.)

        Returns:
            Agent's response as string

        Raises:
            RuntimeError: If agent execution fails
        """
        round_num = kwargs.get("round_num", 1)
        logger.info("Agent %s executing for round %d", self.name, round_num)

        # Build context-aware prompt based on round
        if round_num == 1:
            prompt = f"""This is the first round of brainstorming on the following topic:

{context}

Please provide your initial thoughts and expert analysis."""
        else:
            # Agno Pattern: When session_id is set, conversation history is automatic
            # No need to manually include previous context - Agno handles it
            prompt = f"""Based on the discussion so far:

{context}

Please build on these ideas with your expert perspective. Consider what has been said and add your unique insights."""

        try:
            # Agno Pattern: Use agent.run() for execution
            # This returns a RunResponse with content and metrics
            response = self._agent.run(prompt)

            # Extract response text from Agno RunResponse
            response_text = response.content

            # Agno Pattern: Extract metrics automatically tracked by Agno
            metrics = extract_agno_metrics(self._agent)

            logger.info(
                "Agent %s completed round %d: %d tokens, $%.4f",
                self.name,
                round_num,
                metrics["tokens_used"],
                metrics["cost"],
            )

            # Store metadata for retrieval
            self._last_metrics = metrics

            return response_text

        except Exception as e:
            logger.exception("Agent %s failed to execute", self.name)
            raise RuntimeError(f"Agent execution failed: {e!s}") from e

    def get_last_metadata(self) -> dict[str, Any]:
        """Get metadata from last execution.

        Returns:
            Dictionary with tokens_used, cost, and model info
        """
        metrics = getattr(self, "_last_metrics", {})
        return {
            "tokens_used": metrics.get("tokens_used", 0),
            "cost": metrics.get("cost", 0.0),
            "model": self.model,
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
        }
