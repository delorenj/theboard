"""Letta agent integration for cross-meeting memory (Story 14).

This module provides memory persistence and recall functionality for Letta agents,
enabling cross-meeting institutional knowledge.

Architecture:
- Letta SDK handles agent state and memory management
- PostgreSQL stores memory records in agent_memory table
- Qdrant provides vector similarity search for memory retrieval
- Hybrid approach: Support both Letta and plaintext agents

Memory Types:
- decision: Key decisions made in meetings (e.g., "Use Kubernetes for deployment")
- pattern: Recurring patterns observed (e.g., "Team prefers gradual migration")
- context: Domain context learned (e.g., "Project uses microservices architecture")
- learning: Agent-specific learnings (e.g., "Mobile-first design is priority")
"""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from theboard.models.meeting import Agent, AgentMemory, Meeting


class LettaMemoryManager:
    """Manages agent memory persistence and retrieval for Letta integration."""

    def __init__(self, db_session: AsyncSession):
        """Initialize memory manager with database session.

        Args:
            db_session: SQLAlchemy async session for database operations
        """
        self.db = db_session

    async def store_memory(
        self,
        agent_id: UUID,
        meeting_id: UUID,
        memory_type: str,
        content: dict[str, Any],
        relevance_score: float | None = None,
        embedding: list[float] | None = None,
    ) -> AgentMemory:
        """Store a memory for an agent from a meeting.

        Args:
            agent_id: UUID of the agent
            meeting_id: UUID of the meeting where memory was formed
            memory_type: Type of memory (decision, pattern, context, learning)
            content: Memory content as JSON (flexible schema per type)
            relevance_score: Optional relevance score (0.0-1.0)
            embedding: Optional vector embedding for similarity search

        Returns:
            AgentMemory: Created memory record

        Example content schemas:
            decision: {"decision": "Use k8s", "rationale": "...", "confidence": 0.9}
            pattern: {"pattern": "Gradual migration", "occurrences": 3, "context": "..."}
            context: {"topic": "Microservices", "facts": [...], "source_meeting": "..."}
            learning: {"insight": "Mobile-first", "evidence": [...], "agent_specific": True}
        """
        memory = AgentMemory(
            agent_id=agent_id,
            meeting_id=meeting_id,
            memory_type=memory_type,
            content=content,
            relevance_score=relevance_score,
            embedding=embedding,
        )

        self.db.add(memory)
        await self.db.commit()
        await self.db.refresh(memory)

        return memory

    async def recall_memories(
        self,
        agent_id: UUID,
        memory_type: str | None = None,
        limit: int = 5,
        min_relevance: float = 0.0,
    ) -> list[AgentMemory]:
        """Recall agent memories, optionally filtered by type.

        Args:
            agent_id: UUID of the agent
            memory_type: Optional filter by memory type
            limit: Maximum number of memories to return (default: 5)
            min_relevance: Minimum relevance score filter (default: 0.0)

        Returns:
            List of AgentMemory records, ordered by created_at DESC

        Note:
            For vector similarity search, use recall_similar_memories() instead.
            This method returns most recent memories without similarity ranking.
        """
        query = select(AgentMemory).where(AgentMemory.agent_id == agent_id)

        if memory_type:
            query = query.where(AgentMemory.memory_type == memory_type)

        if min_relevance > 0.0:
            query = query.where(AgentMemory.relevance_score >= min_relevance)

        query = query.order_by(AgentMemory.created_at.desc()).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def recall_similar_memories(
        self,
        agent_id: UUID,
        query_embedding: list[float],
        limit: int = 5,
        memory_type: str | None = None,
    ) -> list[AgentMemory]:
        """Recall agent memories using vector similarity search.

        Args:
            agent_id: UUID of the agent
            query_embedding: Vector embedding of the query (current meeting topic/context)
            limit: Maximum number of memories to return (default: 5)
            memory_type: Optional filter by memory type

        Returns:
            List of AgentMemory records, ordered by similarity score DESC

        Note:
            This uses PostgreSQL's vector similarity functions.
            For production, integrate with Qdrant for scalable similarity search.
            Story 14 acceptance criteria: <1s latency with 100+ meetings.
        """
        # TODO: Integrate with Qdrant for production vector search
        # For now, use simple PostgreSQL array operations
        # In production, query Qdrant collection 'agent_memories' with:
        # - filter: agent_id, memory_type
        # - query_vector: query_embedding
        # - limit: limit
        # - score_threshold: 0.7 (configurable)

        # Placeholder implementation - will be replaced with Qdrant in next iteration
        query = select(AgentMemory).where(
            AgentMemory.agent_id == agent_id, AgentMemory.embedding.isnot(None)
        )

        if memory_type:
            query = query.where(AgentMemory.memory_type == memory_type)

        query = query.order_by(AgentMemory.created_at.desc()).limit(limit)

        result = await self.db.execute(query)
        memories = list(result.scalars().all())

        # TODO: Calculate cosine similarity and re-rank
        # For now, return by recency
        return memories

    async def get_agent_context(
        self, agent_id: UUID, meeting_topic: str, limit: int = 5
    ) -> str:
        """Build context string for agent from past memories.

        Args:
            agent_id: UUID of the agent
            meeting_topic: Current meeting topic for similarity matching
            limit: Maximum memories to include in context

        Returns:
            Formatted context string for agent prompt

        Example output:
            "Based on your past experience:
            - Decision (Meeting abc123): Use Kubernetes for deployment (confidence: 0.9)
            - Pattern (3 occurrences): Team prefers gradual migration strategies
            - Context: This project uses microservices architecture with event-driven design"
        """
        # Recall recent memories (by recency for now, similarity later)
        memories = await self.recall_memories(agent_id, limit=limit)

        if not memories:
            return ""

        context_lines = ["Based on your past experience:"]

        for memory in memories:
            content = memory.content
            memory_type = memory.memory_type

            if memory_type == "decision":
                decision = content.get("decision", "Unknown decision")
                confidence = content.get("confidence", "unknown")
                context_lines.append(
                    f"- Decision: {decision} (confidence: {confidence})"
                )

            elif memory_type == "pattern":
                pattern = content.get("pattern", "Unknown pattern")
                occurrences = content.get("occurrences", "multiple")
                context_lines.append(
                    f"- Pattern ({occurrences} occurrences): {pattern}"
                )

            elif memory_type == "context":
                topic = content.get("topic", "Unknown topic")
                facts = content.get("facts", [])
                context_lines.append(f"- Context ({topic}): {', '.join(facts[:3])}")

            elif memory_type == "learning":
                insight = content.get("insight", "Unknown insight")
                context_lines.append(f"- Learning: {insight}")

        return "\n".join(context_lines)

    async def extract_memories_from_response(
        self, agent_id: UUID, meeting_id: UUID, response_text: str
    ) -> list[AgentMemory]:
        """Extract and store memories from agent response.

        Args:
            agent_id: UUID of the agent
            meeting_id: UUID of the meeting
            response_text: Agent's response text to analyze

        Returns:
            List of created AgentMemory records

        Note:
            This is a placeholder for LLM-based memory extraction.
            In production, use an LLM to identify key decisions, patterns, etc.
            from the agent's response and store as structured memories.
        """
        # TODO: Implement LLM-based memory extraction
        # For now, return empty list (manual memory storage via store_memory)
        return []


class LettaAgentAdapter:
    """Adapter for Letta agents to work with TheBoard's agent interface.

    This class bridges Letta's agent model with TheBoard's existing plaintext
    agent system, enabling hybrid support (both Letta and plaintext agents).
    """

    def __init__(self, agent: Agent, memory_manager: LettaMemoryManager):
        """Initialize Letta agent adapter.

        Args:
            agent: TheBoard Agent model (with agent_type='letta')
            memory_manager: LettaMemoryManager for memory operations
        """
        self.agent = agent
        self.memory_manager = memory_manager
        self._letta_definition = agent.letta_definition or {}

    async def get_enhanced_prompt(
        self, base_prompt: str, meeting_topic: str
    ) -> str:
        """Enhance agent prompt with memory context.

        Args:
            base_prompt: Original agent prompt (expertise, persona, background)
            meeting_topic: Current meeting topic

        Returns:
            Enhanced prompt with memory context injected

        Example:
            Original: "You are a mobile development expert..."
            Enhanced: "You are a mobile development expert...
                      Based on your past experience:
                      - Decision: Mobile-first design (confidence: 0.9)
                      - Pattern (3 occurrences): Team prefers React Native
                      ..."
        """
        # Get memory context from past meetings
        memory_context = await self.memory_manager.get_agent_context(
            self.agent.id, meeting_topic, limit=5
        )

        if memory_context:
            # Inject memory context after base prompt
            enhanced = f"{base_prompt}\n\n{memory_context}"
        else:
            enhanced = base_prompt

        return enhanced

    def is_letta_agent(self) -> bool:
        """Check if agent is Letta-enabled.

        Returns:
            True if agent_type is 'letta', False otherwise
        """
        return self.agent.agent_type == "letta"
