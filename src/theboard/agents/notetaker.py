"""Notetaker Agent for extracting structured comments using Agno.

This module implements the notetaker agent using Agno's structured output feature.
Instead of manual JSON parsing and validation, Agno automatically:
- Ensures the LLM returns valid structured data
- Validates against Pydantic schemas
- Handles parsing errors gracefully
"""

import logging
from typing import Any

from agno.agent import Agent

from theboard.agents.base import create_agno_agent, extract_agno_metrics
from theboard.preferences import get_preferences_manager
from theboard.schemas import Comment, CommentCategory, CommentList

logger = logging.getLogger(__name__)


class NotetakerAgent:
    """Agent that extracts structured comments using Agno's output_schema.

    Agno Pattern: Use output_schema parameter to get structured outputs directly.
    This eliminates manual JSON parsing and validation - Agno handles it automatically.
    """

    def __init__(self, model: str | None = None) -> None:
        """Initialize Notetaker Agent.

        Args:
            model: LLM model to use for extraction (defaults to preferences)
        """
        self.name = "Notetaker"

        # Use preferences if model not provided
        if model is None:
            prefs = get_preferences_manager()
            model = prefs.get_model_for_agent(
                agent_name="notetaker",
                agent_type="notetaker",
            )

        self.model = model

        # Agno Pattern: Define instructions for structured extraction
        instructions = [
            "You are a meticulous note-taker analyzing brainstorming discussions",
            "Extract key ideas, decisions, and insights from agent responses",
            "For each distinct idea, create a comment with text, category, and novelty_score",
            "Categories: technical_decision, risk, implementation_detail, question, concern, suggestion, other",
            "Extract 3-10 comments per response",
            "Focus on substantive ideas, not pleasantries",
            "Be precise and concise in your extractions",
        ]

        # Agno Pattern: Create agent WITHOUT output_schema initially
        # We'll create a new agent with output_schema for each extraction
        # This allows flexibility in the response format
        self._base_agent_config = {
            "name": self.name,
            "role": "Extract structured comments from brainstorming responses",
            "expertise": "Identifying key ideas, categorizing insights, and extracting actionable points",
            "instructions": instructions,
            "model_id": model,
        }

        logger.info("Created NotetakerAgent (model=%s)", model)

    async def extract_comments(
        self, response_text: str, agent_name: str
    ) -> list[Comment]:
        """Extract structured comments from agent response.

        Agno Pattern: Create agent with output_schema=CommentList for automatic
        structured output. Agno ensures the response matches the Pydantic schema.

        Args:
            response_text: The agent's response text
            agent_name: Name of the agent who provided the response

        Returns:
            List of extracted Comment objects

        Raises:
            RuntimeError: If extraction fails
        """
        logger.info("Extracting comments from %s response", agent_name)

        try:
            # Agno Pattern: Create agent with CommentList output schema
            # This tells Agno to return structured data matching CommentList model
            extractor = create_agno_agent(
                **self._base_agent_config,
                output_schema=CommentList,  # Agno automatically validates against this
                debug_mode=False,
            )

            # Build extraction prompt
            prompt = f"""Extract structured comments from this response by {agent_name}:

{response_text}

Identify 3-10 distinct ideas, decisions, or insights. For each one:
- Write clear, concise text (10-1000 characters)
- Choose the most appropriate category:
  - technical_decision: Architecture or design decisions
  - risk: Potential risks, challenges, or concerns
  - implementation_detail: Specific implementation suggestions
  - question: Questions that need answers
  - concern: Issues to address
  - suggestion: General recommendations
  - other: Important points that don't fit above
- Set novelty_score to 0.0 (will be calculated later)

Return the comments in the structured format."""

            # Agno Pattern: Use agent.run() with output_schema
            # Returns RunResponse where content is already a CommentList instance
            response = extractor.run(prompt)

            # Agno Pattern: response.content is automatically a CommentList Pydantic model
            # No manual JSON parsing needed!
            comment_list: CommentList = response.content
            comments = comment_list.comments

            # Extract metrics
            metrics = extract_agno_metrics(extractor)

            logger.info(
                "Extracted %d comments from %s: %d tokens, $%.4f",
                len(comments),
                agent_name,
                metrics["tokens_used"],
                metrics["cost"],
            )

            # Store metadata
            self._last_metrics = metrics

            return comments

        except Exception as e:
            logger.exception("Comment extraction failed for %s", agent_name)
            # Fallback: create a single comment from the response
            # This handles cases where structured extraction fails
            return [
                Comment(
                    text=response_text[:500] if len(response_text) > 500 else response_text,
                    category=CommentCategory.OTHER,
                    novelty_score=0.0,
                )
            ]

    def get_last_metadata(self) -> dict[str, Any]:
        """Get metadata from last extraction.

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
