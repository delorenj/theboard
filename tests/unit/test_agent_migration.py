"""Unit tests for agent migration service (Story 14)."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from theboard.agents.agent_migration import AgentMigrationService
from theboard.models.meeting import Agent


@pytest.mark.asyncio
async def test_migrate_plaintext_agent_to_letta(db_session):
    """Test migrating a plaintext agent to Letta type."""
    # Create plaintext agent
    agent = Agent(
        name="test-backend-arch",
        expertise="Backend architecture and microservices",
        persona="Experienced architect with cloud focus",
        background="15 years in distributed systems",
        agent_type="plaintext",
        default_model="deepseek",
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    # Migrate to Letta
    service = AgentMigrationService(db_session)
    updated = await service.migrate_agent_to_letta(agent, model="gpt-4", temperature=0.8)

    # Verify migration
    assert updated.agent_type == "letta"
    assert updated.default_model == "gpt-4"
    assert updated.letta_definition is not None
    assert updated.letta_definition["model"] == "gpt-4"
    assert updated.letta_definition["temperature"] == 0.8
    assert updated.letta_definition["memory_enabled"] is True
    assert updated.letta_definition["cross_meeting_recall"] is True

    # Verify original attributes preserved in letta_definition
    assert updated.letta_definition["original_expertise"] == agent.expertise
    assert updated.letta_definition["original_persona"] == agent.persona
    assert updated.letta_definition["original_background"] == agent.background

    # Verify original attributes still on agent
    assert updated.expertise == "Backend architecture and microservices"
    assert updated.persona == "Experienced architect with cloud focus"
    assert updated.background == "15 years in distributed systems"


@pytest.mark.asyncio
async def test_migrate_already_letta_agent_fails(db_session):
    """Test that migrating an already-Letta agent raises error."""
    # Create Letta agent
    agent = Agent(
        name="test-letta-agent",
        expertise="SRE",
        agent_type="letta",
        letta_definition={"model": "gpt-4", "temperature": 0.7},
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    # Try to migrate - should fail
    service = AgentMigrationService(db_session)
    with pytest.raises(ValueError, match="already a Letta agent"):
        await service.migrate_agent_to_letta(agent)


@pytest.mark.asyncio
async def test_revert_letta_agent_to_plaintext(db_session):
    """Test reverting a Letta agent back to plaintext."""
    # Create Letta agent with preserved attributes
    agent = Agent(
        name="test-letta-agent",
        expertise="Mobile development",
        persona="React Native expert",
        background="5 years in mobile apps",
        agent_type="letta",
        letta_definition={
            "model": "gpt-4",
            "temperature": 0.7,
            "original_expertise": "Mobile development",
            "original_persona": "React Native expert",
            "original_background": "5 years in mobile apps",
        },
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    # Revert to plaintext
    service = AgentMigrationService(db_session)
    updated = await service.revert_agent_to_plaintext(agent)

    # Verify reversion
    assert updated.agent_type == "plaintext"
    assert updated.letta_definition is None

    # Verify original attributes restored
    assert updated.expertise == "Mobile development"
    assert updated.persona == "React Native expert"
    assert updated.background == "5 years in mobile apps"


@pytest.mark.asyncio
async def test_revert_plaintext_agent_fails(db_session):
    """Test that reverting a plaintext agent raises error."""
    # Create plaintext agent
    agent = Agent(
        name="test-plaintext-agent",
        expertise="Backend",
        agent_type="plaintext",
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    # Try to revert - should fail
    service = AgentMigrationService(db_session)
    with pytest.raises(ValueError, match="not a Letta agent"):
        await service.revert_agent_to_plaintext(agent)


@pytest.mark.asyncio
async def test_migrate_all_agents(db_session):
    """Test migrating all plaintext agents at once."""
    # Create multiple plaintext agents
    agents = [
        Agent(name="backend-arch", expertise="Backend", agent_type="plaintext"),
        Agent(name="frontend-dev", expertise="Frontend", agent_type="plaintext"),
        Agent(name="sre-specialist", expertise="SRE", agent_type="plaintext"),
    ]
    for agent in agents:
        db_session.add(agent)
    await db_session.commit()

    # Migrate all
    service = AgentMigrationService(db_session)
    result = await service.migrate_all_agents(model="deepseek", temperature=0.7)

    # Verify all migrated
    assert len(result["migrated"]) == 3
    assert "backend-arch" in result["migrated"]
    assert "frontend-dev" in result["migrated"]
    assert "sre-specialist" in result["migrated"]
    assert len(result["skipped"]) == 0

    # Verify agents are now Letta type
    agents_by_type = await service.list_agents_by_type()
    assert len(agents_by_type["letta"]) == 3
    assert len(agents_by_type["plaintext"]) == 0


@pytest.mark.asyncio
async def test_migrate_all_with_exclusions(db_session):
    """Test migrating all agents except excluded ones."""
    # Create multiple plaintext agents
    agents = [
        Agent(name="backend-arch", expertise="Backend", agent_type="plaintext"),
        Agent(name="frontend-dev", expertise="Frontend", agent_type="plaintext"),
        Agent(name="sre-specialist", expertise="SRE", agent_type="plaintext"),
    ]
    for agent in agents:
        db_session.add(agent)
    await db_session.commit()

    # Migrate all except "frontend-dev"
    service = AgentMigrationService(db_session)
    result = await service.migrate_all_agents(
        model="deepseek",
        temperature=0.7,
        exclude_names=["frontend-dev"],
    )

    # Verify migration results
    assert len(result["migrated"]) == 2
    assert "backend-arch" in result["migrated"]
    assert "sre-specialist" in result["migrated"]
    assert len(result["skipped"]) == 1
    assert "frontend-dev" in result["skipped"]

    # Verify types
    agents_by_type = await service.list_agents_by_type()
    assert len(agents_by_type["letta"]) == 2
    assert len(agents_by_type["plaintext"]) == 1
    assert "frontend-dev" in agents_by_type["plaintext"]


@pytest.mark.asyncio
async def test_dry_run_migration(db_session):
    """Test dry-run migration doesn't commit changes."""
    # Create plaintext agent
    agent = Agent(
        name="test-agent",
        expertise="Testing",
        agent_type="plaintext",
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    original_id = agent.id

    # Dry-run migration
    service = AgentMigrationService(db_session)
    await service.migrate_agent_to_letta(agent, dry_run=True)

    # Verify changes not committed
    await db_session.rollback()  # Rollback session to see DB state
    reloaded = await service.get_agent_by_name("test-agent")

    assert reloaded.agent_type == "plaintext"  # Should still be plaintext
    assert reloaded.letta_definition is None


@pytest.mark.asyncio
async def test_get_agent_by_name(db_session):
    """Test getting agent by name."""
    # Create agent
    agent = Agent(name="test-agent", expertise="Testing", agent_type="plaintext")
    db_session.add(agent)
    await db_session.commit()

    # Get by name
    service = AgentMigrationService(db_session)
    found = await service.get_agent_by_name("test-agent")

    assert found is not None
    assert found.name == "test-agent"

    # Try non-existent agent
    not_found = await service.get_agent_by_name("nonexistent")
    assert not_found is None


@pytest.mark.asyncio
async def test_list_agents_by_type(db_session):
    """Test listing agents grouped by type."""
    # Create mixed agent types
    plaintext_agents = [
        Agent(name="plain-1", expertise="Test", agent_type="plaintext"),
        Agent(name="plain-2", expertise="Test", agent_type="plaintext"),
    ]
    letta_agents = [
        Agent(
            name="letta-1",
            expertise="Test",
            agent_type="letta",
            letta_definition={"model": "gpt-4"},
        ),
        Agent(
            name="letta-2",
            expertise="Test",
            agent_type="letta",
            letta_definition={"model": "gpt-4"},
        ),
        Agent(
            name="letta-3",
            expertise="Test",
            agent_type="letta",
            letta_definition={"model": "gpt-4"},
        ),
    ]

    for agent in plaintext_agents + letta_agents:
        db_session.add(agent)
    await db_session.commit()

    # List by type
    service = AgentMigrationService(db_session)
    agents_by_type = await service.list_agents_by_type()

    assert len(agents_by_type["plaintext"]) == 2
    assert "plain-1" in agents_by_type["plaintext"]
    assert "plain-2" in agents_by_type["plaintext"]

    assert len(agents_by_type["letta"]) == 3
    assert "letta-1" in agents_by_type["letta"]
    assert "letta-2" in agents_by_type["letta"]
    assert "letta-3" in agents_by_type["letta"]


@pytest.mark.asyncio
async def test_migration_preserves_all_attributes(db_session):
    """Test that migration preserves all agent attributes."""
    # Create agent with all attributes
    agent = Agent(
        name="complete-agent",
        expertise="Full stack development",
        persona="Pragmatic engineer with DevOps mindset",
        background="10 years in startups and enterprise",
        agent_type="plaintext",
        default_model="deepseek",
        is_active=True,
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)

    # Migrate
    service = AgentMigrationService(db_session)
    updated = await service.migrate_agent_to_letta(agent)

    # Verify all attributes preserved
    assert updated.name == "complete-agent"
    assert updated.expertise == "Full stack development"
    assert updated.persona == "Pragmatic engineer with DevOps mindset"
    assert updated.background == "10 years in startups and enterprise"
    assert updated.is_active is True
    assert updated.agent_type == "letta"
    assert updated.default_model == "deepseek"  # Updated in migration
