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
import os
from datetime import datetime
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from theboard.models.meeting import Agent, AgentMemory, Meeting


class LettaMemoryManager:
    """Manages agent memory persistence and retrieval for Letta integration."""

    COLLECTION_NAME = "agent_memories"
    EMBEDDING_DIM = 1536  # OpenAI text-embedding-ada-002 dimension

    def __init__(
        self,
        db_session: AsyncSession,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ):
        """Initialize memory manager with database session and Qdrant client.

        Args:
            db_session: SQLAlchemy async session for database operations
            qdrant_url: Qdrant server URL (default: from QDRANT_URL env or localhost)
            qdrant_api_key: Qdrant API key (default: from QDRANT_API_KEY env)
        """
        self.db = db_session
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.qdrant_client = AsyncQdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

    async def ensure_collection_exists(self) -> None:
        """Ensure Qdrant collection exists with proper schema.

        Creates collection if it doesn't exist with:
        - Vector size: 1536 (OpenAI ada-002)
        - Distance metric: Cosine similarity
        - Payload schema: agent_id, memory_type, created_at
        """
        collections = await self.qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.COLLECTION_NAME not in collection_names:
            await self.qdrant_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )

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

        # Store embedding in Qdrant if provided
        if embedding:
            await self.ensure_collection_exists()
            await self.qdrant_client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=str(memory.id),  # Use memory UUID as Qdrant point ID
                        vector=embedding,
                        payload={
                            "agent_id": str(agent_id),
                            "meeting_id": str(meeting_id),
                            "memory_type": memory_type,
                            "created_at": memory.created_at.isoformat(),
                        },
                    )
                ],
            )

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
        score_threshold: float = 0.7,
    ) -> list[AgentMemory]:
        """Recall agent memories using vector similarity search via Qdrant.

        Args:
            agent_id: UUID of the agent
            query_embedding: Vector embedding of the query (current meeting topic/context)
            limit: Maximum number of memories to return (default: 5)
            memory_type: Optional filter by memory type
            score_threshold: Minimum similarity score (0.0-1.0, default: 0.7)

        Returns:
            List of AgentMemory records, ordered by similarity score DESC

        Note:
            Uses Qdrant for scalable vector similarity search.
            Story 14 acceptance criteria: <1s latency with 100+ meetings.
        """
        await self.ensure_collection_exists()

        # Build Qdrant filter conditions
        filter_conditions = [
            FieldCondition(
                key="agent_id",
                match=MatchValue(value=str(agent_id)),
            )
        ]

        if memory_type:
            filter_conditions.append(
                FieldCondition(
                    key="memory_type",
                    match=MatchValue(value=memory_type),
                )
            )

        # Query Qdrant for similar memories
        search_result = await self.qdrant_client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=Filter(must=filter_conditions) if filter_conditions else None,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Extract memory IDs from Qdrant results
        memory_ids = [UUID(point.id) for point in search_result]

        if not memory_ids:
            return []

        # Fetch full AgentMemory records from PostgreSQL
        query = select(AgentMemory).where(AgentMemory.id.in_(memory_ids))
        result = await self.db.execute(query)
        memories = list(result.scalars().all())

        # Re-order memories by Qdrant similarity score
        memory_dict = {mem.id: mem for mem in memories}
        sorted_memories = [
            memory_dict[UUID(point.id)]
            for point in search_result
            if UUID(point.id) in memory_dict
        ]

        return sorted_memories

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

    async def delete_memory(self, memory_id: UUID) -> None:
        """Delete a memory from both PostgreSQL and Qdrant.

        Args:
            memory_id: UUID of the memory to delete
        """
        # Delete from Qdrant
        try:
            await self.qdrant_client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=[str(memory_id)],
            )
        except Exception:
            # Log error but continue - memory might not have embedding
            pass

        # Delete from PostgreSQL
        query = select(AgentMemory).where(AgentMemory.id == memory_id)
        result = await self.db.execute(query)
        memory = result.scalar_one_or_none()

        if memory:
            await self.db.delete(memory)
            await self.db.commit()

    async def close(self) -> None:
        """Close Qdrant client connection."""
        await self.qdrant_client.close()


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
