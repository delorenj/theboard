"""Agent migration utilities for Story 14: Letta Integration.

This module provides tools to migrate plaintext agents to Letta agents,
preserving their expertise, persona, and background while enabling
cross-meeting memory capabilities.

Usage:
    python -m theboard.agents.agent_migration migrate-all
    python -m theboard.agents.agent_migration migrate-agent <agent_name>
    python -m theboard.agents.agent_migration revert-agent <agent_name>
"""

import asyncio
import argparse
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from theboard.config import settings
from theboard.models.meeting import Agent


class AgentMigrationService:
    """Service for migrating agents between plaintext and Letta types."""

    def __init__(self, db_session: AsyncSession):
        """Initialize migration service.

        Args:
            db_session: SQLAlchemy async session for database operations
        """
        self.db = db_session

    async def migrate_agent_to_letta(
        self,
        agent: Agent,
        model: str = "deepseek",
        temperature: float = 0.7,
        dry_run: bool = False,
    ) -> Agent:
        """Migrate a plaintext agent to Letta agent type.

        Args:
            agent: Agent to migrate (must be agent_type='plaintext')
            model: Model to use for Letta agent (default: deepseek)
            temperature: Temperature for Letta agent (default: 0.7)
            dry_run: If True, don't commit changes (default: False)

        Returns:
            Updated Agent record

        Raises:
            ValueError: If agent is already a Letta agent
        """
        if agent.agent_type == "letta":
            raise ValueError(
                f"Agent '{agent.name}' is already a Letta agent. "
                "No migration needed."
            )

        # Preserve original attributes
        original_expertise = agent.expertise
        original_persona = agent.persona
        original_background = agent.background

        # Create Letta definition
        letta_definition = {
            "model": model,
            "temperature": temperature,
            "memory_enabled": True,
            "cross_meeting_recall": True,
            # Preserve original attributes in definition for reference
            "original_expertise": original_expertise,
            "original_persona": original_persona,
            "original_background": original_background,
        }

        # Update agent
        agent.agent_type = "letta"
        agent.letta_definition = letta_definition
        agent.default_model = model

        if not dry_run:
            await self.db.commit()
            await self.db.refresh(agent)

        return agent

    async def revert_agent_to_plaintext(
        self,
        agent: Agent,
        dry_run: bool = False,
    ) -> Agent:
        """Revert a Letta agent back to plaintext type.

        Args:
            agent: Agent to revert (must be agent_type='letta')
            dry_run: If True, don't commit changes (default: False)

        Returns:
            Updated Agent record

        Raises:
            ValueError: If agent is not a Letta agent
        """
        if agent.agent_type != "letta":
            raise ValueError(
                f"Agent '{agent.name}' is not a Letta agent. "
                "Cannot revert to plaintext."
            )

        # Restore original attributes from letta_definition if available
        if agent.letta_definition:
            original_expertise = agent.letta_definition.get(
                "original_expertise", agent.expertise
            )
            original_persona = agent.letta_definition.get(
                "original_persona", agent.persona
            )
            original_background = agent.letta_definition.get(
                "original_background", agent.background
            )

            agent.expertise = original_expertise
            agent.persona = original_persona
            agent.background = original_background

        # Revert to plaintext
        agent.agent_type = "plaintext"
        agent.letta_definition = None
        # Keep default_model as-is (might have been changed)

        if not dry_run:
            await self.db.commit()
            await self.db.refresh(agent)

        return agent

    async def migrate_all_agents(
        self,
        model: str = "deepseek",
        temperature: float = 0.7,
        dry_run: bool = False,
        exclude_names: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Migrate all plaintext agents to Letta.

        Args:
            model: Model to use for Letta agents (default: deepseek)
            temperature: Temperature for Letta agents (default: 0.7)
            dry_run: If True, don't commit changes (default: False)
            exclude_names: List of agent names to skip (default: None)

        Returns:
            Dict with 'migrated' and 'skipped' lists of agent names
        """
        exclude_names = exclude_names or []

        # Get all plaintext agents
        query = select(Agent).where(Agent.agent_type == "plaintext")
        result = await self.db.execute(query)
        plaintext_agents = list(result.scalars().all())

        migrated = []
        skipped = []

        for agent in plaintext_agents:
            if agent.name in exclude_names:
                skipped.append(agent.name)
                continue

            await self.migrate_agent_to_letta(
                agent, model=model, temperature=temperature, dry_run=dry_run
            )
            migrated.append(agent.name)

        return {
            "migrated": migrated,
            "skipped": skipped,
        }

    async def get_agent_by_name(self, name: str) -> Agent | None:
        """Get agent by name.

        Args:
            name: Agent name

        Returns:
            Agent if found, None otherwise
        """
        query = select(Agent).where(Agent.name == name)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def list_agents_by_type(self) -> dict[str, list[str]]:
        """List all agents grouped by type.

        Returns:
            Dict with 'plaintext' and 'letta' lists of agent names
        """
        query = select(Agent).order_by(Agent.name)
        result = await self.db.execute(query)
        agents = list(result.scalars().all())

        plaintext_agents = [a.name for a in agents if a.agent_type == "plaintext"]
        letta_agents = [a.name for a in agents if a.agent_type == "letta"]

        return {
            "plaintext": plaintext_agents,
            "letta": letta_agents,
        }


async def main():
    """CLI entry point for agent migration."""
    parser = argparse.ArgumentParser(
        description="Migrate agents between plaintext and Letta types"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # migrate-all command
    migrate_all_parser = subparsers.add_parser(
        "migrate-all", help="Migrate all plaintext agents to Letta"
    )
    migrate_all_parser.add_argument(
        "--model", default="deepseek", help="Model to use (default: deepseek)"
    )
    migrate_all_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature (default: 0.7)"
    )
    migrate_all_parser.add_argument(
        "--dry-run", action="store_true", help="Don't commit changes"
    )
    migrate_all_parser.add_argument(
        "--exclude", nargs="+", help="Agent names to exclude"
    )

    # migrate-agent command
    migrate_agent_parser = subparsers.add_parser(
        "migrate-agent", help="Migrate a specific agent to Letta"
    )
    migrate_agent_parser.add_argument("name", help="Agent name")
    migrate_agent_parser.add_argument(
        "--model", default="deepseek", help="Model to use (default: deepseek)"
    )
    migrate_agent_parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature (default: 0.7)"
    )
    migrate_agent_parser.add_argument(
        "--dry-run", action="store_true", help="Don't commit changes"
    )

    # revert-agent command
    revert_agent_parser = subparsers.add_parser(
        "revert-agent", help="Revert a Letta agent back to plaintext"
    )
    revert_agent_parser.add_argument("name", help="Agent name")
    revert_agent_parser.add_argument(
        "--dry-run", action="store_true", help="Don't commit changes"
    )

    # list-agents command
    list_parser = subparsers.add_parser(
        "list-agents", help="List all agents by type"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create database session
    engine = create_async_engine(settings.database_url)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        service = AgentMigrationService(session)

        if args.command == "list-agents":
            agents = await service.list_agents_by_type()
            print("\nPlaintext Agents:")
            for name in agents["plaintext"]:
                print(f"  - {name}")
            print(f"\nTotal: {len(agents['plaintext'])}")

            print("\nLetta Agents:")
            for name in agents["letta"]:
                print(f"  - {name}")
            print(f"\nTotal: {len(agents['letta'])}")

        elif args.command == "migrate-all":
            print(f"\nMigrating all plaintext agents to Letta...")
            if args.dry_run:
                print("(DRY RUN - no changes will be committed)")

            result = await service.migrate_all_agents(
                model=args.model,
                temperature=args.temperature,
                dry_run=args.dry_run,
                exclude_names=args.exclude or [],
            )

            print(f"\nMigrated ({len(result['migrated'])}):")
            for name in result["migrated"]:
                print(f"  ✓ {name}")

            if result["skipped"]:
                print(f"\nSkipped ({len(result['skipped'])}):")
                for name in result["skipped"]:
                    print(f"  - {name}")

        elif args.command == "migrate-agent":
            agent = await service.get_agent_by_name(args.name)
            if not agent:
                print(f"Error: Agent '{args.name}' not found")
                return

            if args.dry_run:
                print("(DRY RUN - no changes will be committed)")

            try:
                updated = await service.migrate_agent_to_letta(
                    agent,
                    model=args.model,
                    temperature=args.temperature,
                    dry_run=args.dry_run,
                )
                print(f"\n✓ Migrated '{args.name}' to Letta agent")
                print(f"  Model: {args.model}")
                print(f"  Temperature: {args.temperature}")
            except ValueError as e:
                print(f"Error: {e}")

        elif args.command == "revert-agent":
            agent = await service.get_agent_by_name(args.name)
            if not agent:
                print(f"Error: Agent '{args.name}' not found")
                return

            if args.dry_run:
                print("(DRY RUN - no changes will be committed)")

            try:
                updated = await service.revert_agent_to_plaintext(
                    agent, dry_run=args.dry_run
                )
                print(f"\n✓ Reverted '{args.name}' to plaintext agent")
            except ValueError as e:
                print(f"Error: {e}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
