"""Configuration management CLI commands."""

import asyncio
import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from theboard.preferences import get_preferences_manager
from theboard.services.openrouter_service import OpenRouterService

logger = logging.getLogger(__name__)
console = Console()

# Config command group
config_app = typer.Typer(
    name="config",
    help="Manage TheBoard configuration and preferences",
)


@config_app.command("models")
def config_models(
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Force refresh model cache")
    ] = False,
) -> None:
    """Interactive TUI for selecting OpenRouter models.

    Displays available models grouped by cost tier with pricing information.
    Allows selection of global default and per-agent-type models.
    """
    try:
        # Fetch models
        with console.status("[bold green]Fetching OpenRouter models...", spinner="dots"):
            service = OpenRouterService()
            models = asyncio.run(service.fetch_models(force_refresh=refresh))

        if not models:
            console.print("[red]No models found. Check your OpenRouter API key.[/red]")
            raise typer.Exit(1)

        # Group by cost tier
        tiers = service.group_by_cost_tier(models)

        # Display models by tier
        console.print("\n[bold]Available OpenRouter Models[/bold]\n")

        for tier_name in ["budget", "standard", "premium"]:
            tier_models = tiers[tier_name]
            if not tier_models:
                continue

            table = Table(
                title=f"{tier_name.capitalize()} Tier",
                show_header=True,
                border_style=(
                    "cyan"
                    if tier_name == "budget"
                    else "yellow" if tier_name == "standard" else "red"
                ),
            )
            table.add_column("ID", style="bold")
            table.add_column("Name", style="cyan")
            table.add_column("Input ($/MTok)", justify="right")
            table.add_column("Output ($/MTok)", justify="right")
            table.add_column("Context", justify="right")

            for model in tier_models:
                table.add_row(
                    model.id,
                    model.name,
                    f"${model.input_cost_per_mtok:.2f}",
                    f"${model.output_cost_per_mtok:.2f}",
                    f"{model.context_length:,}",
                )

            console.print(table)
            console.print()

        # Interactive selection
        console.print("[bold]Select models for configuration:[/bold]\n")

        manager = get_preferences_manager()
        current_prefs = manager.load()

        # Show current configuration
        current_table = Table(title="Current Configuration", show_header=True)
        current_table.add_column("Setting", style="bold cyan")
        current_table.add_column("Model ID", style="yellow")

        current_table.add_row("Global Default", current_prefs.models.default)
        current_table.add_row("Worker Agents", current_prefs.models.by_agent_type.worker)
        current_table.add_row("Leader Agents", current_prefs.models.by_agent_type.leader)
        current_table.add_row("Notetaker", current_prefs.models.by_agent_type.notetaker)
        current_table.add_row("Compressor", current_prefs.models.by_agent_type.compressor)

        console.print(current_table)
        console.print()

        # Prompt for changes
        if not typer.confirm("Update model configuration?", default=False):
            console.print("[yellow]No changes made.[/yellow]")
            return

        # Get new default model
        new_default = typer.prompt(
            "Global default model ID",
            default=current_prefs.models.default,
            show_default=True,
        )
        manager.set_default_model(new_default)

        # Get agent type models
        for agent_type in ["worker", "leader", "notetaker", "compressor"]:
            current_value = getattr(current_prefs.models.by_agent_type, agent_type)
            new_value = typer.prompt(
                f"{agent_type.capitalize()} model ID",
                default=current_value,
                show_default=True,
            )
            if new_value != current_value:
                manager.set_agent_type_model(agent_type, new_value)

        console.print(
            Panel.fit(
                "[green]✓[/green] Configuration updated successfully!\n\n"
                f"Preferences saved to:\n[cyan]{manager.config_path}[/cyan]",
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        logger.exception("Failed to configure models")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@config_app.command("show")
def config_show() -> None:
    """Display current configuration."""
    try:
        manager = get_preferences_manager()
        preferences = manager.load()

        # Configuration table
        table = Table(title="TheBoard Configuration", show_header=True)
        table.add_column("Setting", style="bold cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Config File", str(manager.config_path))
        table.add_row("", "")
        table.add_row("[bold]Models", "")
        table.add_row("  Global Default", preferences.models.default)
        table.add_row("  Worker Agents", preferences.models.by_agent_type.worker)
        table.add_row("  Leader Agents", preferences.models.by_agent_type.leader)
        table.add_row("  Notetaker", preferences.models.by_agent_type.notetaker)
        table.add_row("  Compressor", preferences.models.by_agent_type.compressor)

        if preferences.models.overrides:
            table.add_row("", "")
            table.add_row("[bold]Per-Agent Overrides", "")
            for agent_name, model_id in preferences.models.overrides.items():
                table.add_row(f"  {agent_name}", model_id)

        console.print(table)

    except Exception as e:
        logger.exception("Failed to show configuration")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@config_app.command("init")
def config_init(
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite existing config")
    ] = False,
) -> None:
    """Initialize preferences file with defaults."""
    try:
        manager = get_preferences_manager()

        if manager.config_path.exists() and not force:
            console.print(
                f"[yellow]Configuration file already exists:[/yellow]\n"
                f"[cyan]{manager.config_path}[/cyan]\n\n"
                f"Use [bold]--force[/bold] to overwrite."
            )
            raise typer.Exit(1)

        # Create default preferences
        from theboard.preferences import Preferences

        preferences = Preferences()
        manager.save(preferences)

        console.print(
            Panel.fit(
                f"[green]✓[/green] Configuration file created!\n\n"
                f"Location: [cyan]{manager.config_path}[/cyan]\n\n"
                f"Edit this file to customize model preferences.\n"
                f"Use [bold]board config models[/bold] for interactive selection.",
                title="[bold green]Initialized[/bold green]",
                border_style="green",
            )
        )

    except Exception as e:
        logger.exception("Failed to initialize configuration")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e
