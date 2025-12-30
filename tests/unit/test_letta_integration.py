"""Unit tests for Letta integration (Story 14)."""

import pytest
from uuid import uuid4

from theboard.agents.letta_integration import LettaMemoryManager, LettaAgentAdapter
from theboard.models.meeting import Agent, AgentMemory


@pytest.mark.asyncio
async def test_store_memory(db_session, sample_agent, sample_meeting):
    """Test storing agent memory."""
    memory_manager = LettaMemoryManager(db_session)

    # Store a decision memory
    memory = await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={
            "decision": "Use Kubernetes for deployment",
            "rationale": "Scalability and orchestration needs",
            "confidence": 0.9,
        },
        relevance_score=0.95,
    )

    assert memory.id is not None
    assert memory.agent_id == sample_agent.id
    assert memory.meeting_id == sample_meeting.id
    assert memory.memory_type == "decision"
    assert memory.content["decision"] == "Use Kubernetes for deployment"
    assert memory.relevance_score == 0.95


@pytest.mark.asyncio
async def test_recall_memories(db_session, sample_agent, sample_meeting):
    """Test recalling agent memories."""
    memory_manager = LettaMemoryManager(db_session)

    # Store multiple memories
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={"decision": "Decision 1"},
    )
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="pattern",
        content={"pattern": "Pattern 1", "occurrences": 3},
    )
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={"decision": "Decision 2"},
    )

    # Recall all memories
    memories = await memory_manager.recall_memories(sample_agent.id, limit=10)
    assert len(memories) == 3

    # Recall only decisions
    decisions = await memory_manager.recall_memories(
        sample_agent.id, memory_type="decision"
    )
    assert len(decisions) == 2
    assert all(m.memory_type == "decision" for m in decisions)

    # Recall with limit
    limited = await memory_manager.recall_memories(sample_agent.id, limit=2)
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_get_agent_context(db_session, sample_agent, sample_meeting):
    """Test building agent context from memories."""
    memory_manager = LettaMemoryManager(db_session)

    # Store memories of different types
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={
            "decision": "Use microservices architecture",
            "confidence": 0.9,
        },
    )
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="pattern",
        content={
            "pattern": "Team prefers gradual migration",
            "occurrences": 3,
        },
    )

    # Build context
    context = await memory_manager.get_agent_context(
        sample_agent.id, meeting_topic="System architecture"
    )

    assert "Based on your past experience:" in context
    assert "Use microservices architecture" in context
    assert "confidence: 0.9" in context
    assert "Team prefers gradual migration" in context
    assert "3 occurrences" in context


@pytest.mark.asyncio
async def test_letta_agent_adapter(db_session, sample_meeting):
    """Test LettaAgentAdapter for hybrid agent support."""
    # Create a Letta agent
    letta_agent = Agent(
        name="letta-sre-specialist",
        expertise="SRE and infrastructure",
        persona="Pragmatic SRE with cloud experience",
        agent_type="letta",
        letta_definition={"model": "gpt-4", "temperature": 0.7},
    )
    db_session.add(letta_agent)
    await db_session.commit()
    await db_session.refresh(letta_agent)

    memory_manager = LettaMemoryManager(db_session)
    adapter = LettaAgentAdapter(letta_agent, memory_manager)

    # Verify Letta agent detection
    assert adapter.is_letta_agent()

    # Store a memory
    await memory_manager.store_memory(
        agent_id=letta_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={"decision": "Use Kubernetes", "confidence": 0.9},
    )

    # Enhance prompt with memory
    base_prompt = "You are an SRE specialist focusing on cloud infrastructure."
    enhanced = await adapter.get_enhanced_prompt(
        base_prompt, meeting_topic="Deployment strategy"
    )

    assert base_prompt in enhanced
    assert "Based on your past experience:" in enhanced
    assert "Use Kubernetes" in enhanced


@pytest.mark.asyncio
async def test_plaintext_agent_compatibility(db_session):
    """Test that plaintext agents still work (backward compatibility)."""
    # Create a plaintext agent (existing agent type)
    plaintext_agent = Agent(
        name="plaintext-backend-arch",
        expertise="Backend architecture",
        persona="Experienced backend architect",
        agent_type="plaintext",  # Not 'letta'
    )
    db_session.add(plaintext_agent)
    await db_session.commit()
    await db_session.refresh(plaintext_agent)

    memory_manager = LettaMemoryManager(db_session)
    adapter = LettaAgentAdapter(plaintext_agent, memory_manager)

    # Verify NOT detected as Letta agent
    assert not adapter.is_letta_agent()

    # Prompt enhancement should still work (no memories yet)
    base_prompt = "You are a backend architect."
    enhanced = await adapter.get_enhanced_prompt(
        base_prompt, meeting_topic="API design"
    )

    # Should return base prompt unchanged (no memories)
    assert enhanced == base_prompt


@pytest.mark.asyncio
async def test_memory_relevance_filtering(db_session, sample_agent, sample_meeting):
    """Test filtering memories by relevance score."""
    memory_manager = LettaMemoryManager(db_session)

    # Store memories with different relevance scores
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={"decision": "High relevance"},
        relevance_score=0.9,
    )
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={"decision": "Medium relevance"},
        relevance_score=0.6,
    )
    await memory_manager.store_memory(
        agent_id=sample_agent.id,
        meeting_id=sample_meeting.id,
        memory_type="decision",
        content={"decision": "Low relevance"},
        relevance_score=0.3,
    )

    # Recall only high-relevance memories
    high_relevance = await memory_manager.recall_memories(
        sample_agent.id, min_relevance=0.8
    )
    assert len(high_relevance) == 1
    assert high_relevance[0].content["decision"] == "High relevance"

    # Recall medium+ relevance
    medium_plus = await memory_manager.recall_memories(
        sample_agent.id, min_relevance=0.5
    )
    assert len(medium_plus) == 2
