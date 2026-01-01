"""Service layer for agent pool management."""

import logging
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from theboard.database import get_sync_db
from theboard.models.meeting import Agent
from theboard.schemas import AgentConfig, AgentResponse, AgentType

logger = logging.getLogger(__name__)


def create_agent(
    name: str,
    expertise: str,
    persona: str | None = None,
    background: str | None = None,
    agent_type: AgentType = AgentType.PLAINTEXT,
    default_model: str = "deepseek",
    is_active: bool = True,
) -> AgentResponse:
    """Create a new agent in the pool.

    Args:
        name: Unique agent name (used for identification)
        expertise: Agent's area of expertise (required)
        persona: Optional persona description
        background: Optional background context
        agent_type: Type of agent (plaintext or letta)
        default_model: Default model to use for this agent
        is_active: Whether agent is active in the pool

    Returns:
        AgentResponse with created agent details

    Raises:
        ValueError: If validation fails or agent name already exists
    """
    # Validate inputs
    if not (3 <= len(name) <= 100):
        raise ValueError("Agent name must be between 3 and 100 characters")

    if not (10 <= len(expertise) <= 5000):
        raise ValueError("Expertise must be between 10 and 5000 characters")

    with get_sync_db() as db:
        try:
            # Check if agent name already exists
            stmt = select(Agent).where(Agent.name == name)
            existing = db.scalars(stmt).first()
            if existing:
                raise ValueError(f"Agent with name '{name}' already exists")

            # Create agent
            agent = Agent(
                name=name,
                expertise=expertise,
                persona=persona,
                background=background,
                agent_type=agent_type.value,
                default_model=default_model,
                is_active=is_active,
            )

            db.add(agent)
            db.commit()
            db.refresh(agent)

            logger.info("Created agent %s: %s", agent.id, agent.name)

            return AgentResponse.model_validate(agent)

        except IntegrityError as e:
            db.rollback()
            logger.exception("Database integrity error creating agent")
            raise ValueError(f"Agent creation failed: {e!s}") from e
        except Exception as e:
            db.rollback()
            logger.exception("Failed to create agent")
            raise ValueError(f"Failed to create agent: {e!s}") from e


def list_agents(
    active_only: bool = False, limit: int = 100, offset: int = 0
) -> list[AgentResponse]:
    """List agents in the pool.

    Args:
        active_only: If True, only return active agents
        limit: Maximum number of agents to return
        offset: Number of agents to skip (pagination)

    Returns:
        List of AgentResponse objects

    Raises:
        ValueError: If query fails
    """
    with get_sync_db() as db:
        try:
            stmt = select(Agent).order_by(Agent.name)

            if active_only:
                stmt = stmt.where(Agent.is_active == True)  # noqa: E712

            stmt = stmt.limit(limit).offset(offset)

            agents = db.scalars(stmt).all()

            return [AgentResponse.model_validate(a) for a in agents]

        except Exception as e:
            logger.exception("Failed to list agents")
            raise ValueError(f"Failed to list agents: {e!s}") from e


def get_agent(agent_id: UUID) -> AgentResponse:
    """Get a specific agent by ID.

    Args:
        agent_id: Agent UUID

    Returns:
        AgentResponse with agent details

    Raises:
        ValueError: If agent not found
    """
    with get_sync_db() as db:
        try:
            stmt = select(Agent).where(Agent.id == agent_id)
            agent = db.scalars(stmt).first()

            if not agent:
                raise ValueError(f"Agent not found: {agent_id}")

            return AgentResponse.model_validate(agent)

        except Exception as e:
            logger.exception("Failed to get agent")
            raise ValueError(f"Failed to get agent: {e!s}") from e


def get_agent_by_name(name: str) -> AgentResponse:
    """Get a specific agent by name.

    Args:
        name: Agent name

    Returns:
        AgentResponse with agent details

    Raises:
        ValueError: If agent not found
    """
    with get_sync_db() as db:
        try:
            stmt = select(Agent).where(Agent.name == name)
            agent = db.scalars(stmt).first()

            if not agent:
                raise ValueError(f"Agent not found: {name}")

            return AgentResponse.model_validate(agent)

        except Exception as e:
            logger.exception("Failed to get agent by name")
            raise ValueError(f"Failed to get agent: {e!s}") from e


def update_agent(
    agent_id: UUID,
    expertise: str | None = None,
    persona: str | None = None,
    background: str | None = None,
    default_model: str | None = None,
    is_active: bool | None = None,
) -> AgentResponse:
    """Update an existing agent.

    Args:
        agent_id: Agent UUID
        expertise: Updated expertise (optional)
        persona: Updated persona (optional)
        background: Updated background (optional)
        default_model: Updated default model (optional)
        is_active: Updated active status (optional)

    Returns:
        AgentResponse with updated agent details

    Raises:
        ValueError: If agent not found or validation fails
    """
    with get_sync_db() as db:
        try:
            stmt = select(Agent).where(Agent.id == agent_id)
            agent = db.scalars(stmt).first()

            if not agent:
                raise ValueError(f"Agent not found: {agent_id}")

            # Update fields if provided
            if expertise is not None:
                if not (10 <= len(expertise) <= 5000):
                    raise ValueError("Expertise must be between 10 and 5000 characters")
                agent.expertise = expertise

            if persona is not None:
                agent.persona = persona

            if background is not None:
                agent.background = background

            if default_model is not None:
                agent.default_model = default_model

            if is_active is not None:
                agent.is_active = is_active

            db.commit()
            db.refresh(agent)

            logger.info("Updated agent %s", agent_id)

            return AgentResponse.model_validate(agent)

        except Exception as e:
            db.rollback()
            logger.exception("Failed to update agent")
            raise ValueError(f"Failed to update agent: {e!s}") from e


def deactivate_agent(agent_id: UUID) -> AgentResponse:
    """Deactivate an agent (soft delete).

    Args:
        agent_id: Agent UUID

    Returns:
        AgentResponse with deactivated agent details

    Raises:
        ValueError: If agent not found
    """
    return update_agent(agent_id, is_active=False)


def activate_agent(agent_id: UUID) -> AgentResponse:
    """Activate a previously deactivated agent.

    Args:
        agent_id: Agent UUID

    Returns:
        AgentResponse with activated agent details

    Raises:
        ValueError: If agent not found
    """
    return update_agent(agent_id, is_active=True)


def delete_agent(agent_id: UUID, force: bool = False) -> bool:
    """Delete an agent from the pool.

    Args:
        agent_id: Agent UUID
        force: If True, permanently delete. If False, only deactivate.

    Returns:
        True if deleted successfully

    Raises:
        ValueError: If agent not found or has associated data
    """
    with get_sync_db() as db:
        try:
            stmt = select(Agent).where(Agent.id == agent_id)
            agent = db.scalars(stmt).first()

            if not agent:
                raise ValueError(f"Agent not found: {agent_id}")

            if not force:
                # Soft delete: just deactivate
                agent.is_active = False
                db.commit()
                logger.info("Deactivated agent %s", agent_id)
            else:
                # Hard delete: permanently remove
                # Note: This will cascade delete responses and performance metrics
                db.delete(agent)
                db.commit()
                logger.info("Permanently deleted agent %s", agent_id)

            return True

        except Exception as e:
            db.rollback()
            logger.exception("Failed to delete agent")
            raise ValueError(f"Failed to delete agent: {e!s}") from e


def bulk_create_agents(agents: list[AgentConfig]) -> list[AgentResponse]:
    """Create multiple agents in bulk.

    Args:
        agents: List of AgentConfig objects

    Returns:
        List of created AgentResponse objects

    Raises:
        ValueError: If any agent creation fails
    """
    created_agents = []

    for agent_config in agents:
        try:
            agent_response = create_agent(
                name=agent_config.name,
                expertise=agent_config.expertise,
                persona=agent_config.persona,
                background=agent_config.background,
                agent_type=agent_config.agent_type,
                default_model=agent_config.default_model,
            )
            created_agents.append(agent_response)
        except ValueError as e:
            logger.warning("Skipping agent %s: %s", agent_config.name, e)
            continue

    logger.info("Bulk created %d/%d agents", len(created_agents), len(agents))

    return created_agents
