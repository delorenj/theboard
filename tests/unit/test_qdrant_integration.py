"""Unit tests for Qdrant vector search integration (Story 14)."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from theboard.agents.letta_integration import LettaMemoryManager
from theboard.models.meeting import Agent, AgentMemory


@pytest.mark.asyncio
async def test_qdrant_collection_creation(db_session):
    """Test Qdrant collection is created on first use."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection doesn't exist
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)
        await memory_manager.ensure_collection_exists()

        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "agent_memories"
        assert call_args.kwargs["vectors_config"].size == 1536
        assert call_args.kwargs["vectors_config"].distance.value == "Cosine"


@pytest.mark.asyncio
async def test_store_memory_with_qdrant_embedding(db_session, sample_agent, sample_meeting):
    """Test storing memory with embedding in Qdrant."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection exists
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="agent_memories")]
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)

        # Create test embedding (1536 dimensions)
        test_embedding = [0.1] * 1536

        memory = await memory_manager.store_memory(
            agent_id=sample_agent.id,
            meeting_id=sample_meeting.id,
            memory_type="decision",
            content={"decision": "Use Kubernetes", "confidence": 0.9},
            relevance_score=0.95,
            embedding=test_embedding,
        )

        # Verify memory stored in PostgreSQL
        assert memory.id is not None
        assert memory.embedding == test_embedding

        # Verify Qdrant upsert was called
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == "agent_memories"

        points = call_args.kwargs["points"]
        assert len(points) == 1
        assert points[0].id == str(memory.id)
        assert points[0].vector == test_embedding
        assert points[0].payload["agent_id"] == str(sample_agent.id)
        assert points[0].payload["memory_type"] == "decision"


@pytest.mark.asyncio
async def test_recall_similar_memories_with_qdrant(db_session, sample_agent, sample_meeting):
    """Test recalling memories using Qdrant vector search."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection exists
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="agent_memories")]
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)

        # Store test memory
        test_embedding = [0.1] * 1536
        memory = await memory_manager.store_memory(
            agent_id=sample_agent.id,
            meeting_id=sample_meeting.id,
            memory_type="decision",
            content={"decision": "Use Kubernetes"},
            embedding=test_embedding,
        )

        # Mock Qdrant search result
        mock_search_result = [
            MagicMock(id=str(memory.id), score=0.95),
        ]
        mock_client.search.return_value = mock_search_result

        # Query similar memories
        query_embedding = [0.11] * 1536  # Slightly different embedding
        results = await memory_manager.recall_similar_memories(
            agent_id=sample_agent.id,
            query_embedding=query_embedding,
            limit=5,
            score_threshold=0.7,
        )

        # Verify Qdrant search was called with correct parameters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs["collection_name"] == "agent_memories"
        assert call_args.kwargs["query_vector"] == query_embedding
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["score_threshold"] == 0.7

        # Verify filter for agent_id
        filter_obj = call_args.kwargs["query_filter"]
        assert filter_obj is not None
        assert len(filter_obj.must) == 1
        assert filter_obj.must[0].key == "agent_id"
        assert filter_obj.must[0].match.value == str(sample_agent.id)

        # Verify results
        assert len(results) == 1
        assert results[0].id == memory.id
        assert results[0].content["decision"] == "Use Kubernetes"


@pytest.mark.asyncio
async def test_recall_similar_with_memory_type_filter(db_session, sample_agent, sample_meeting):
    """Test vector search with memory_type filter."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection exists
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="agent_memories")]
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)

        # Store test memories of different types
        decision_memory = await memory_manager.store_memory(
            agent_id=sample_agent.id,
            meeting_id=sample_meeting.id,
            memory_type="decision",
            content={"decision": "Use Kubernetes"},
            embedding=[0.1] * 1536,
        )

        pattern_memory = await memory_manager.store_memory(
            agent_id=sample_agent.id,
            meeting_id=sample_meeting.id,
            memory_type="pattern",
            content={"pattern": "Team prefers gradual migration"},
            embedding=[0.2] * 1536,
        )

        # Mock Qdrant search result (only decision)
        mock_search_result = [
            MagicMock(id=str(decision_memory.id), score=0.95),
        ]
        mock_client.search.return_value = mock_search_result

        # Query with memory_type filter
        query_embedding = [0.11] * 1536
        results = await memory_manager.recall_similar_memories(
            agent_id=sample_agent.id,
            query_embedding=query_embedding,
            memory_type="decision",
            limit=5,
        )

        # Verify filter includes both agent_id and memory_type
        call_args = mock_client.search.call_args
        filter_obj = call_args.kwargs["query_filter"]
        assert len(filter_obj.must) == 2

        filter_keys = [f.key for f in filter_obj.must]
        assert "agent_id" in filter_keys
        assert "memory_type" in filter_keys

        # Verify only decision memory returned
        assert len(results) == 1
        assert results[0].memory_type == "decision"


@pytest.mark.asyncio
async def test_delete_memory_removes_from_qdrant(db_session, sample_agent, sample_meeting):
    """Test deleting memory removes it from both PostgreSQL and Qdrant."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection exists
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="agent_memories")]
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)

        # Store memory
        memory = await memory_manager.store_memory(
            agent_id=sample_agent.id,
            meeting_id=sample_meeting.id,
            memory_type="decision",
            content={"decision": "Use Kubernetes"},
            embedding=[0.1] * 1536,
        )

        memory_id = memory.id

        # Delete memory
        await memory_manager.delete_memory(memory_id)

        # Verify Qdrant delete was called
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "agent_memories"
        assert str(memory_id) in call_args.kwargs["points_selector"]


@pytest.mark.asyncio
async def test_qdrant_client_close(db_session):
    """Test closing Qdrant client connection."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        memory_manager = LettaMemoryManager(db_session)
        await memory_manager.close()

        # Verify client close was called
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_no_qdrant_upsert_without_embedding(db_session, sample_agent, sample_meeting):
    """Test that memories without embeddings don't trigger Qdrant upsert."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        memory_manager = LettaMemoryManager(db_session)

        # Store memory without embedding
        memory = await memory_manager.store_memory(
            agent_id=sample_agent.id,
            meeting_id=sample_meeting.id,
            memory_type="decision",
            content={"decision": "Use Kubernetes"},
            # No embedding provided
        )

        # Verify memory stored in PostgreSQL
        assert memory.id is not None
        assert memory.embedding is None

        # Verify Qdrant upsert was NOT called
        mock_client.upsert.assert_not_called()
