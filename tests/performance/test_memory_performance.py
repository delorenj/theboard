"""Performance tests for Letta memory retrieval (Story 14).

Validates <1s latency requirement with 100+ meetings.

Run with:
    pytest tests/performance/test_memory_performance.py -v --no-cov -s
"""

import asyncio
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from theboard.agents.letta_integration import LettaMemoryManager
from theboard.models.meeting import Agent, AgentMemory, Meeting


@pytest.mark.asyncio
async def test_memory_retrieval_latency_100_meetings(db_session, sample_agent):
    """Test memory retrieval <1s with 100+ meetings.

    Story 14 acceptance criteria: <1s latency with 100+ meetings.
    """
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection exists
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="agent_memories")]
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)

        # Create 100+ meetings with memories
        num_meetings = 120
        num_memories_per_meeting = 5
        total_memories = num_meetings * num_memories_per_meeting

        print(f"\n\nCreating {total_memories} memories across {num_meetings} meetings...")

        meetings = []
        memories = []

        # Batch create meetings
        for i in range(num_meetings):
            meeting = Meeting(
                topic=f"Meeting {i}: Architecture discussion",
                strategy="sequential",
                max_rounds=5,
                status="completed",
            )
            db_session.add(meeting)
            meetings.append(meeting)

        await db_session.commit()

        # Batch create memories
        for meeting_idx, meeting in enumerate(meetings):
            for mem_idx in range(num_memories_per_meeting):
                # Create varied memory types
                memory_types = ["decision", "pattern", "context", "learning"]
                memory_type = memory_types[mem_idx % len(memory_types)]

                embedding = [0.1 + (meeting_idx * 0.001) + (mem_idx * 0.0001)] * 1536

                memory = AgentMemory(
                    agent_id=sample_agent.id,
                    meeting_id=meeting.id,
                    memory_type=memory_type,
                    content={
                        "text": f"Memory {meeting_idx}-{mem_idx}",
                        "meeting_num": meeting_idx,
                        "memory_num": mem_idx,
                    },
                    relevance_score=0.5 + (mem_idx * 0.1),
                    embedding=embedding,
                )
                db_session.add(memory)
                memories.append(memory)

        await db_session.commit()

        print(f"✓ Created {total_memories} memories")

        # Mock Qdrant search to return realistic results (top 10 matches)
        def mock_search_side_effect(*args, **kwargs):
            """Simulate Qdrant search with realistic latency."""
            # Simulate Qdrant search latency (50-200ms typical)
            time.sleep(0.05)  # 50ms

            # Return top 10 most recent memories
            limit = kwargs.get("limit", 5)
            recent_memories = sorted(
                memories, key=lambda m: m.created_at, reverse=True
            )[:limit]

            return [
                MagicMock(id=str(mem.id), score=0.9 - (i * 0.05))
                for i, mem in enumerate(recent_memories)
            ]

        mock_client.search.side_effect = mock_search_side_effect

        # Test 1: Recall by recency (no vector search)
        print(f"\nTest 1: Recall by recency (no Qdrant)")
        start_time = time.time()

        recent_memories = await memory_manager.recall_memories(
            agent_id=sample_agent.id,
            limit=10,
        )

        recency_latency = time.time() - start_time

        print(f"  Latency: {recency_latency:.3f}s")
        print(f"  Memories returned: {len(recent_memories)}")
        assert len(recent_memories) == 10
        assert recency_latency < 1.0, f"Recency query too slow: {recency_latency:.3f}s"

        # Test 2: Recall by similarity (Qdrant vector search)
        print(f"\nTest 2: Recall by similarity (Qdrant)")
        query_embedding = [0.15] * 1536

        start_time = time.time()

        similar_memories = await memory_manager.recall_similar_memories(
            agent_id=sample_agent.id,
            query_embedding=query_embedding,
            limit=10,
            score_threshold=0.7,
        )

        similarity_latency = time.time() - start_time

        print(f"  Latency: {similarity_latency:.3f}s")
        print(f"  Memories returned: {len(similar_memories)}")
        assert len(similar_memories) == 10
        assert similarity_latency < 1.0, f"Similarity query too slow: {similarity_latency:.3f}s"

        # Test 3: Build agent context (combines recall + formatting)
        print(f"\nTest 3: Build agent context")
        start_time = time.time()

        context = await memory_manager.get_agent_context(
            agent_id=sample_agent.id,
            meeting_topic="Architecture discussion",
            limit=10,
        )

        context_latency = time.time() - start_time

        print(f"  Latency: {context_latency:.3f}s")
        print(f"  Context length: {len(context)} chars")
        assert len(context) > 0
        assert context_latency < 1.0, f"Context building too slow: {context_latency:.3f}s"

        # Test 4: Filtered recall with memory type
        print(f"\nTest 4: Filtered recall (memory_type filter)")
        start_time = time.time()

        decision_memories = await memory_manager.recall_memories(
            agent_id=sample_agent.id,
            memory_type="decision",
            limit=10,
        )

        filtered_latency = time.time() - start_time

        print(f"  Latency: {filtered_latency:.3f}s")
        print(f"  Memories returned: {len(decision_memories)}")
        assert all(m.memory_type == "decision" for m in decision_memories)
        assert filtered_latency < 1.0, f"Filtered query too slow: {filtered_latency:.3f}s"

        # Summary
        print(f"\n" + "="*60)
        print(f"PERFORMANCE SUMMARY ({num_meetings} meetings, {total_memories} memories)")
        print(f"="*60)
        print(f"Recency query:      {recency_latency:.3f}s  {'✓ PASS' if recency_latency < 1.0 else '✗ FAIL'}")
        print(f"Similarity query:   {similarity_latency:.3f}s  {'✓ PASS' if similarity_latency < 1.0 else '✗ FAIL'}")
        print(f"Context building:   {context_latency:.3f}s  {'✓ PASS' if context_latency < 1.0 else '✗ FAIL'}")
        print(f"Filtered query:     {filtered_latency:.3f}s  {'✓ PASS' if filtered_latency < 1.0 else '✗ FAIL'}")
        print(f"="*60)
        print(f"Story 14 Target: <1s latency with 100+ meetings ✓ ACHIEVED")
        print(f"="*60)


@pytest.mark.asyncio
async def test_bulk_memory_storage_performance(db_session, sample_agent, sample_meeting):
    """Test bulk memory storage performance."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection exists
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="agent_memories")]
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)

        # Test storing 100 memories
        num_memories = 100
        print(f"\n\nStoring {num_memories} memories...")

        start_time = time.time()

        for i in range(num_memories):
            embedding = [0.1 + (i * 0.001)] * 1536

            await memory_manager.store_memory(
                agent_id=sample_agent.id,
                meeting_id=sample_meeting.id,
                memory_type="decision",
                content={"decision": f"Decision {i}"},
                embedding=embedding,
            )

        storage_latency = time.time() - start_time

        print(f"  Total latency: {storage_latency:.3f}s")
        print(f"  Per-memory latency: {storage_latency/num_memories:.4f}s")
        print(f"  Throughput: {num_memories/storage_latency:.1f} memories/sec")

        # Should be able to store at least 10 memories/sec
        assert (num_memories / storage_latency) > 10, f"Storage too slow: {num_memories/storage_latency:.1f} mem/sec"


@pytest.mark.asyncio
async def test_concurrent_memory_queries(db_session, sample_agent):
    """Test concurrent memory queries performance."""
    with patch("theboard.agents.letta_integration.AsyncQdrantClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock collection exists
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="agent_memories")]
        mock_client.get_collections.return_value = mock_collections

        memory_manager = LettaMemoryManager(db_session)

        # Create test memories
        for i in range(50):
            embedding = [0.1 + (i * 0.001)] * 1536
            meeting = Meeting(
                topic=f"Meeting {i}",
                strategy="sequential",
                max_rounds=5,
            )
            db_session.add(meeting)
            await db_session.commit()

            await memory_manager.store_memory(
                agent_id=sample_agent.id,
                meeting_id=meeting.id,
                memory_type="decision",
                content={"decision": f"Decision {i}"},
                embedding=embedding,
            )

        # Test concurrent queries
        num_concurrent = 10
        print(f"\n\nRunning {num_concurrent} concurrent queries...")

        async def run_query():
            return await memory_manager.recall_memories(
                agent_id=sample_agent.id,
                limit=5,
            )

        start_time = time.time()

        # Run concurrent queries
        results = await asyncio.gather(*[run_query() for _ in range(num_concurrent)])

        concurrent_latency = time.time() - start_time

        print(f"  Total latency: {concurrent_latency:.3f}s")
        print(f"  Per-query latency: {concurrent_latency/num_concurrent:.4f}s")
        print(f"  Throughput: {num_concurrent/concurrent_latency:.1f} queries/sec")

        # All queries should succeed
        assert len(results) == num_concurrent
        assert all(len(r) == 5 for r in results)

        # Concurrent execution should be faster than sequential
        # (At least 2x faster due to async)
        expected_sequential = 0.05 * num_concurrent  # Assume 50ms per query
        assert concurrent_latency < expected_sequential, \
            f"Concurrent queries not faster than sequential: {concurrent_latency:.3f}s vs {expected_sequential:.3f}s"
