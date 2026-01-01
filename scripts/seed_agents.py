#!/usr/bin/env python3
"""Seed the database with initial agent pool.

This script loads agent definitions from data/agents/initial_pool.yaml
and creates them in the database.

Usage:
    python scripts/seed_agents.py [--clear]

Options:
    --clear     Delete all existing agents before seeding (WARNING: destructive!)
"""

import argparse
import logging
from pathlib import Path

import yaml

from theboard.schemas import AgentConfig
from theboard.services.agent_service import bulk_create_agents, list_agents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clear_agents() -> None:
    """Delete all existing agents (WARNING: destructive!)."""
    from theboard.database import get_sync_db
    from theboard.models.meeting import Agent
    from sqlalchemy import delete

    with get_sync_db() as db:
        result = db.execute(delete(Agent))
        db.commit()
        logger.warning("Deleted %d existing agents", result.rowcount)


def seed_agents(file_path: Path, clear_first: bool = False) -> None:
    """Seed agents from YAML file.

    Args:
        file_path: Path to YAML file with agent definitions
        clear_first: If True, delete all existing agents first
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Agent definitions file not found: {file_path}")

    # Clear existing agents if requested
    if clear_first:
        logger.warning("Clearing all existing agents...")
        clear_agents()

    # Load agent definitions
    logger.info("Loading agent definitions from %s", file_path)
    with open(file_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("YAML file must contain a list of agent configurations")

    # Parse agent configs
    agent_configs = [AgentConfig.model_validate(item) for item in data]
    logger.info("Found %d agent definition(s)", len(agent_configs))

    # Create agents
    logger.info("Creating agents...")
    created_agents = bulk_create_agents(agent_configs)

    logger.info(
        "Successfully created %d/%d agent(s)",
        len(created_agents),
        len(agent_configs),
    )

    if len(created_agents) < len(agent_configs):
        logger.warning(
            "%d agent(s) were skipped (likely duplicates)",
            len(agent_configs) - len(created_agents),
        )

    # Display summary
    all_agents = list_agents(active_only=False)
    logger.info("Total agents in database: %d", len(all_agents))
    logger.info("Active agents: %d", sum(1 for a in all_agents if a.is_active))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed the database with initial agent pool"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing agents before seeding (WARNING: destructive!)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=Path("data/agents/initial_pool.yaml"),
        help="Path to agent definitions YAML file",
    )

    args = parser.parse_args()

    try:
        seed_agents(args.file, clear_first=args.clear)
        logger.info("✓ Agent seeding completed successfully")
    except Exception as e:
        logger.exception("Failed to seed agents")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
