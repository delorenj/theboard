"""CLI commands for agent pool management."""

import logging
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from theboard.schemas import AgentType

# Initialize console
console = Console()
logger = logging.getLogger(__name__)

# Create agent subcommand app
agents_app = typer.Typer(
    name="agents",
    help="Manage the agent pool",
    add_completion=False,
)


@agents_app.command(name="create")
def create_agent(
    name: str = typer.Option(..., "--name", "-n", help="Unique agent name"),
    expertise: str = typer.Option(
        ..., "--expertise", "-e", help="Agent's area of expertise"
    ),
    persona: str | None = typer.Option(
        None, "--persona", "-p", help="Optional persona description"
    ),
    background: str | None = typer.Option(
        None, "--background", "-b", help="Optional background context"
    ),
    agent_type: AgentType = typer.Option(
        AgentType.PLAINTEXT,
        "--type",
        "-t",
        help="Agent type (plaintext or letta)",
    ),
    model: str = typer.Option(
        "deepseek", "--model", "-m", help="Default model for this agent"
    ),
    inactive: bool = typer.Option(
        False, "--inactive", help="Create agent as inactive"
    ),
) -> None:
    """Create a new agent in the pool.

    Creates an agent with specified expertise and optional persona/background.
    Agents can be of type 'plaintext' (simple text-based) or 'letta' (Letta-integrated).
    """
    try:
        from theboard.services.agent_service import create_agent

        with console.status("[bold green]Creating agent...", spinner="dots"):
            agent = create_agent(
                name=name,
                expertise=expertise,
                persona=persona,
                background=background,
                agent_type=agent_type,
                default_model=model,
                is_active=not inactive,
            )

        console.print(
            Panel.fit(
                f"[green]✓[/green] Agent created successfully!\n\n"
                f"[bold]ID:[/bold] {agent.id}\n"
                f"[bold]Name:[/bold] {agent.name}\n"
                f"[bold]Type:[/bold] {agent.agent_type.value}\n"
                f"[bold]Model:[/bold] {agent.default_model}\n"
                f"[bold]Status:[/bold] {'Active' if agent.is_active else 'Inactive'}",
                title="[bold green]Agent Created[/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        logger.exception("Failed to create agent")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@agents_app.command(name="list")
def list_agents(
    active_only: bool = typer.Option(
        False, "--active-only", "-a", help="Show only active agents"
    ),
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum agents to display"),
) -> None:
    """List all agents in the pool.

    Displays a table with agent details including name, expertise, type, and status.
    """
    try:
        from theboard.services.agent_service import list_agents

        agents = list_agents(active_only=active_only, limit=limit)

        if not agents:
            console.print(
                "[yellow]No agents found. Create one with:[/yellow] [bold]board agents create[/bold]"
            )
            return

        # Build table
        table = Table(title=f"Agent Pool ({len(agents)} agents)")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Expertise", style="white")
        table.add_column("Type", style="blue")
        table.add_column("Model", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("ID", style="dim")

        for agent in agents:
            # Truncate expertise for display
            expertise_preview = (
                agent.expertise[:50] + "..."
                if len(agent.expertise) > 50
                else agent.expertise
            )

            status = "[green]Active[/green]" if agent.is_active else "[dim]Inactive[/dim]"

            table.add_row(
                agent.name,
                expertise_preview,
                agent.agent_type.value,
                agent.default_model,
                status,
                str(agent.id)[:8],
            )

        console.print(table)

        console.print(
            f"\n[dim]View agent details:[/dim] [bold]board agents show <name>[/bold]"
        )

    except Exception as e:
        logger.exception("Failed to list agents")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@agents_app.command(name="show")
def show_agent(
    identifier: str = typer.Argument(..., help="Agent name or ID"),
) -> None:
    """Show detailed information about a specific agent.

    Args:
        identifier: Agent name or UUID
    """
    try:
        from theboard.services.agent_service import get_agent, get_agent_by_name

        # Try to parse as UUID first, otherwise treat as name
        try:
            agent_id = UUID(identifier)
            agent = get_agent(agent_id)
        except ValueError:
            agent = get_agent_by_name(identifier)

        # Build detailed panel
        details = f"""[bold]ID:[/bold] {agent.id}
[bold]Name:[/bold] {agent.name}
[bold]Type:[/bold] {agent.agent_type.value}
[bold]Model:[/bold] {agent.default_model}
[bold]Status:[/bold] {'[green]Active[/green]' if agent.is_active else '[dim]Inactive[/dim]'}
[bold]Created:[/bold] {agent.created_at.strftime('%Y-%m-%d %H:%M:%S')}

[bold]Expertise:[/bold]
{agent.expertise}"""

        if agent.persona:
            details += f"\n\n[bold]Persona:[/bold]\n{agent.persona}"

        console.print(
            Panel.fit(
                details,
                title=f"[bold cyan]{agent.name}[/bold cyan]",
                border_style="cyan",
            )
        )

    except Exception as e:
        logger.exception("Failed to show agent")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@agents_app.command(name="update")
def update_agent(
    identifier: str = typer.Argument(..., help="Agent name or ID"),
    expertise: str | None = typer.Option(None, "--expertise", "-e", help="Updated expertise"),
    persona: str | None = typer.Option(None, "--persona", "-p", help="Updated persona"),
    background: str | None = typer.Option(None, "--background", "-b", help="Updated background"),
    model: str | None = typer.Option(None, "--model", "-m", help="Updated default model"),
) -> None:
    """Update an existing agent's details.

    At least one field must be provided to update.
    """
    try:
        from theboard.services.agent_service import get_agent, get_agent_by_name, update_agent

        # Validate at least one field provided
        if not any([expertise, persona, background, model]):
            console.print("[red]Error: At least one field must be provided to update[/red]")
            raise typer.Exit(1)

        # Try to parse as UUID first, otherwise treat as name
        try:
            agent_id = UUID(identifier)
            get_agent(agent_id)
        except ValueError:
            agent = get_agent_by_name(identifier)
            agent_id = agent.id

        with console.status("[bold green]Updating agent...", spinner="dots"):
            updated_agent = update_agent(
                agent_id=agent_id,
                expertise=expertise,
                persona=persona,
                background=background,
                default_model=model,
            )

        console.print(
            Panel.fit(
                f"[green]✓[/green] Agent updated successfully!\n\n"
                f"[bold]Name:[/bold] {updated_agent.name}",
                title="[bold green]Agent Updated[/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        logger.exception("Failed to update agent")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@agents_app.command(name="deactivate")
def deactivate_agent(
    identifier: str = typer.Argument(..., help="Agent name or ID"),
) -> None:
    """Deactivate an agent (soft delete).

    Deactivated agents will not be selected for meetings but remain in the database.
    """
    try:
        from theboard.services.agent_service import deactivate_agent, get_agent, get_agent_by_name

        # Try to parse as UUID first, otherwise treat as name
        try:
            agent_id = UUID(identifier)
            get_agent(agent_id)
        except ValueError:
            agent = get_agent_by_name(identifier)
            agent_id = agent.id

        with console.status("[bold yellow]Deactivating agent...", spinner="dots"):
            deactivate_agent(agent_id)

        console.print(
            f"[green]✓[/green] Agent deactivated successfully. "
            f"Reactivate with: [bold]board agents activate {identifier}[/bold]"
        )

    except Exception as e:
        logger.exception("Failed to deactivate agent")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@agents_app.command(name="activate")
def activate_agent(
    identifier: str = typer.Argument(..., help="Agent name or ID"),
) -> None:
    """Activate a previously deactivated agent.

    Reactivates an agent so it can be selected for meetings again.
    """
    try:
        from theboard.services.agent_service import activate_agent, get_agent, get_agent_by_name

        # Try to parse as UUID first, otherwise treat as name
        try:
            agent_id = UUID(identifier)
            get_agent(agent_id)
        except ValueError:
            agent = get_agent_by_name(identifier)
            agent_id = agent.id

        with console.status("[bold green]Activating agent...", spinner="dots"):
            activate_agent(agent_id)

        console.print("[green]✓[/green] Agent activated successfully.")

    except Exception as e:
        logger.exception("Failed to activate agent")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@agents_app.command(name="delete")
def delete_agent(
    identifier: str = typer.Argument(..., help="Agent name or ID"),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Permanently delete (default: deactivate only)",
    ),
) -> None:
    """Delete an agent from the pool.

    By default, this deactivates the agent (soft delete).
    Use --force to permanently delete the agent and all associated data.
    """
    try:
        from theboard.services.agent_service import delete_agent, get_agent, get_agent_by_name

        # Try to parse as UUID first, otherwise treat as name
        try:
            agent_id = UUID(identifier)
            agent = get_agent(agent_id)
        except ValueError:
            agent = get_agent_by_name(identifier)
            agent_id = agent.id

        # Confirm if force delete
        if force:
            confirm = typer.confirm(
                f"Are you sure you want to PERMANENTLY delete agent '{agent.name}'? "
                f"This will also delete all associated responses and performance data."
            )
            if not confirm:
                console.print("[yellow]Deletion cancelled[/yellow]")
                return

        with console.status("[bold red]Deleting agent...", spinner="dots"):
            delete_agent(agent_id, force=force)

        if force:
            console.print(f"[green]✓[/green] Agent '{agent.name}' permanently deleted.")
        else:
            console.print(
                f"[green]✓[/green] Agent '{agent.name}' deactivated. "
                f"Reactivate with: [bold]board agents activate {identifier}[/bold]"
            )

    except Exception as e:
        logger.exception("Failed to delete agent")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@agents_app.command(name="import")
def import_agents(
    file_path: Path = typer.Argument(..., help="Path to YAML or JSON file with agent definitions"),
) -> None:
    """Import agents from a YAML or JSON file.

    File should contain a list of agent configurations with fields:
    - name (required)
    - expertise (required)
    - persona (optional)
    - background (optional)
    - agent_type (optional, default: plaintext)
    - default_model (optional, default: deepseek)
    """
    try:
        from theboard.services.agent_service import bulk_create_agents
        from theboard.schemas import AgentConfig
        import yaml
        import json

        if not file_path.exists():
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)

        # Load file based on extension
        with open(file_path) as f:
            if file_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                data = json.load(f)
            else:
                console.print(
                    f"[red]Error: Unsupported file format: {file_path.suffix}[/red]"
                )
                raise typer.Exit(1)

        # Validate data structure
        if not isinstance(data, list):
            console.print("[red]Error: File must contain a list of agent configurations[/red]")
            raise typer.Exit(1)

        # Parse agent configs
        agent_configs = [AgentConfig.model_validate(item) for item in data]

        console.print(f"[blue]Found {len(agent_configs)} agent(s) to import...[/blue]")

        with console.status("[bold green]Importing agents...", spinner="dots"):
            created_agents = bulk_create_agents(agent_configs)

        console.print(
            f"\n[green]✓[/green] Successfully imported {len(created_agents)}/{len(agent_configs)} agent(s)"
        )

        if len(created_agents) < len(agent_configs):
            console.print(
                f"[yellow]⚠[/yellow] {len(agent_configs) - len(created_agents)} agent(s) skipped (see logs)"
            )

    except Exception as e:
        logger.exception("Failed to import agents")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e
