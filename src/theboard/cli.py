"""Command-line interface for TheBoard."""

import logging
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from theboard.cli_commands.config import config_app
from theboard.config import settings
from theboard.schemas import MeetingStatus, StrategyType

# Initialize Typer app
app = typer.Typer(
    name="board",
    help="TheBoard - Multi-Agent Brainstorming Simulation System",
    add_completion=False,
)

# Register config command group
app.add_typer(config_app, name="config")

# Initialize Rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)
logger = logging.getLogger(__name__)


@app.command()
def create(
    topic: str = typer.Option(
        ..., "--topic", "-t", help="The brainstorming topic (10-500 characters)"
    ),
    strategy: StrategyType = typer.Option(
        StrategyType.SEQUENTIAL,
        "--strategy",
        "-s",
        help="Execution strategy: sequential or greedy",
    ),
    max_rounds: int = typer.Option(
        5, "--max-rounds", "-r", help="Maximum number of rounds (1-10)"
    ),
    agent_count: int = typer.Option(
        5, "--agent-count", "-n", help="Number of agents to auto-select (1-10)"
    ),
    auto_select: bool = typer.Option(
        True, "--auto-select/--manual", help="Auto-select agents based on topic"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Override default model for this meeting"
    ),
    hybrid_models: bool = typer.Option(
        False,
        "--hybrid-models",
        help="Enable hybrid model strategy (budget→premium for top performers)",
    ),
) -> None:
    """Create a new brainstorming meeting.

    Creates a meeting with the specified topic and configuration.
    Agents can be auto-selected based on topic relevance or manually chosen.

    Sprint 4 Story 13: Use --hybrid-models to enable cost optimization via dynamic model promotion.
    """
    try:
        # Import here to avoid circular dependencies
        from theboard.services.meeting_service import create_meeting

        # Validate inputs
        if not (10 <= len(topic) <= 500):
            console.print(
                "[red]Error: Topic must be between 10 and 500 characters[/red]"
            )
            raise typer.Exit(1)

        if not (1 <= max_rounds <= 10):
            console.print("[red]Error: Max rounds must be between 1 and 10[/red]")
            raise typer.Exit(1)

        if not (1 <= agent_count <= 10):
            console.print("[red]Error: Agent count must be between 1 and 10[/red]")
            raise typer.Exit(1)

        # Create meeting
        with console.status("[bold green]Creating meeting...", spinner="dots"):
            meeting = create_meeting(
                topic=topic,
                strategy=strategy,
                max_rounds=max_rounds,
                agent_count=agent_count if auto_select else 0,
                auto_select=auto_select,
                model_override=model,
                hybrid_models=hybrid_models,
            )

        # Display success
        console.print(
            Panel.fit(
                f"[green]✓[/green] Meeting created successfully!\n\n"
                f"[bold]Meeting ID:[/bold] {meeting.id}\n"
                f"[bold]Topic:[/bold] {meeting.topic}\n"
                f"[bold]Strategy:[/bold] {meeting.strategy.value}\n"
                f"[bold]Max Rounds:[/bold] {meeting.max_rounds}",
                title="[bold green]Meeting Created[/bold green]",
                border_style="green",
            )
        )

        console.print(
            f"\n[dim]Run the meeting with:[/dim] [bold]board run {meeting.id}[/bold]"
        )

    except Exception as e:
        logger.exception("Failed to create meeting")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@app.command()
def run(
    meeting_id: str | None = typer.Argument(None, help="Meeting ID to run (optional - shows selector if omitted)"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enable human-in-the-loop prompts"
    ),
    rerun: bool = typer.Option(
        False, "--rerun", help="Reset and rerun a completed/failed meeting"
    ),
    fork: bool = typer.Option(
        False, "--fork", help="Fork meeting (create new meeting with same parameters)"
    ),
    last: bool = typer.Option(
        False, "--last", help="Run most recent meeting"
    ),
) -> None:
    """Run a brainstorming meeting.

    Executes the meeting with the configured strategy and agents.
    Displays real-time progress and results.

    If no meeting ID is provided, shows an interactive selector.
    Use --last to run the most recent meeting without selection.
    Use --rerun to reset and rerun a completed/failed meeting (overwrites data).
    Use --fork to create a new meeting with the same parameters (preserves history).
    """
    try:
        # Import here to avoid circular dependencies
        from rich.prompt import Prompt
        from theboard.services.meeting_service import fork_meeting, list_recent_meetings, run_meeting

        # Mutual exclusivity check
        if rerun and fork:
            console.print(
                "[red]Error: Cannot use --rerun and --fork together. Choose one.[/red]"
            )
            raise typer.Exit(1)

        # Determine meeting ID
        uuid_id: UUID

        if meeting_id is None:
            # No meeting ID provided - show selector or use --last
            if last:
                # Get most recent meeting
                meetings = list_recent_meetings(limit=1)
                if not meetings:
                    console.print("[red]Error: No meetings found. Create one with 'board create'[/red]")
                    raise typer.Exit(1)
                uuid_id = meetings[0].id
                console.print(f"[dim]Using most recent meeting:[/dim] {meetings[0].topic[:60]}")
            else:
                # Show interactive selector
                meetings = list_recent_meetings(limit=20)
                if not meetings:
                    console.print("[red]Error: No meetings found. Create one with 'board create'[/red]")
                    raise typer.Exit(1)

                # Display meetings
                console.print("\n[bold]Recent Meetings:[/bold]\n")
                for i, meeting in enumerate(meetings, 1):
                    status_color = {
                        "created": "cyan",
                        "running": "yellow",
                        "paused": "blue",
                        "completed": "green",
                        "failed": "red",
                    }.get(meeting.status.value, "white")

                    # Format display based on status
                    if meeting.status.value == "completed":
                        details = f"${meeting.total_cost:.2f} - {meeting.current_round} rounds"
                    elif meeting.status.value == "failed":
                        details = f"${meeting.total_cost:.2f} - {meeting.current_round} rounds"
                    else:
                        details = "Ready to run"

                    console.print(
                        f"  {i}. {meeting.topic[:60]} "
                        f"[{status_color}]({meeting.status.value})[/{status_color}] "
                        f"[dim]- {details}[/dim]"
                    )

                # Get user choice
                choice = Prompt.ask(
                    "\n[bold]Select meeting[/bold]",
                    choices=[str(i) for i in range(1, len(meetings) + 1)],
                    default="1"
                )
                selected = meetings[int(choice) - 1]
                uuid_id = selected.id
                console.print(f"[dim]Selected:[/dim] {selected.topic[:60]}\n")
        else:
            # Meeting ID provided - parse it
            try:
                uuid_id = UUID(meeting_id)
            except ValueError as e:
                console.print(f"[red]Error: Invalid meeting ID format: {meeting_id}[/red]")
                raise typer.Exit(1) from e

        # Handle fork: create new meeting with same parameters
        if fork:
            console.print(f"\n[bold]Forking meeting {uuid_id}...[/bold]\n")
            with console.status("[bold cyan]Creating fork...", spinner="dots"):
                forked = fork_meeting(meeting_id=uuid_id)

            console.print(
                Panel.fit(
                    f"[cyan]✓[/cyan] Meeting forked!\n\n"
                    f"[bold]Original ID:[/bold] {uuid_id}\n"
                    f"[bold]Forked ID:[/bold] {forked.id}\n"
                    f"[bold]Topic:[/bold] {forked.topic}",
                    title="[bold cyan]Meeting Forked[/bold cyan]",
                    border_style="cyan",
                )
            )

            # Use the forked meeting ID for running
            uuid_id = forked.id
            console.print(f"\n[bold]Running forked meeting {uuid_id}...[/bold]\n")

        else:
            # Regular run or rerun
            action = "Rerunning" if rerun else "Starting"
            console.print(f"\n[bold]{action} meeting {uuid_id}...[/bold]\n")

        # Run meeting (with rerun flag if specified)
        with console.status("[bold green]Running meeting...", spinner="dots"):
            result = run_meeting(meeting_id=uuid_id, interactive=interactive, rerun=rerun)

        # Display completion
        console.print(
            Panel.fit(
                f"[green]✓[/green] Meeting completed!\n\n"
                f"[bold]Total Rounds:[/bold] {result.current_round}\n"
                f"[bold]Total Comments:[/bold] {result.total_comments}\n"
                f"[bold]Total Cost:[/bold] ${result.total_cost:.4f}\n"
                f"[bold]Status:[/bold] {result.status.value}\n"
                + (
                    f"[bold]Stopping Reason:[/bold] {result.stopping_reason}"
                    if result.stopping_reason
                    else ""
                ),
                title="[bold green]Meeting Completed[/bold green]",
                border_style="green",
            )
        )

        console.print(
            f"\n[dim]View details with:[/dim] [bold]board status {uuid_id}[/bold]"
        )

    except Exception as e:
        logger.exception("Failed to run meeting")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@app.command()
def status(
    meeting_id: str = typer.Argument(..., help="Meeting ID to check"),
    show_comments: bool = typer.Option(
        True, "--comments/--no-comments", help="Show recent comments"
    ),
    show_metrics: bool = typer.Option(
        True, "--metrics/--no-metrics", help="Show convergence metrics"
    ),
) -> None:
    """Display meeting status and details.

    Shows meeting state, responses, comments, and convergence metrics.
    """
    try:
        # Import here to avoid circular dependencies
        from theboard.services.meeting_service import get_meeting_status

        # Parse meeting ID
        try:
            uuid_id = UUID(meeting_id)
        except ValueError as e:
            console.print(f"[red]Error: Invalid meeting ID format: {meeting_id}[/red]")
            raise typer.Exit(1) from e

        # Get meeting status
        status_data = get_meeting_status(meeting_id=uuid_id)

        # Display meeting info
        meeting_table = Table(title="Meeting Information", show_header=False)
        meeting_table.add_column("Field", style="bold cyan")
        meeting_table.add_column("Value")

        meeting_table.add_row("ID", str(status_data.meeting.id))
        meeting_table.add_row("Topic", status_data.meeting.topic)
        meeting_table.add_row("Strategy", status_data.meeting.strategy.value)
        meeting_table.add_row("Status", status_data.meeting.status.value)
        meeting_table.add_row(
            "Current Round", f"{status_data.meeting.current_round}/{status_data.meeting.max_rounds}"
        )
        meeting_table.add_row("Total Comments", str(status_data.meeting.total_comments))
        meeting_table.add_row("Context Size", f"{status_data.meeting.context_size:,} chars")
        meeting_table.add_row("Total Cost", f"${status_data.meeting.total_cost:.4f}")
        if status_data.meeting.convergence_detected:
            meeting_table.add_row(
                "Convergence",
                "[green]✓ Detected[/green]"
                + (
                    f" ({status_data.meeting.stopping_reason})"
                    if status_data.meeting.stopping_reason
                    else ""
                ),
            )

        console.print(meeting_table)

        # Display comments
        if show_comments and status_data.recent_comments:
            console.print("\n")
            comments_table = Table(title="Recent Comments (Last 10)", show_lines=True)
            comments_table.add_column("Round", style="bold")
            comments_table.add_column("Agent", style="cyan")
            comments_table.add_column("Category", style="yellow")
            comments_table.add_column("Comment", style="white", no_wrap=False)
            comments_table.add_column("Novelty", style="green")

            for comment in status_data.recent_comments[:10]:
                comments_table.add_row(
                    str(comment.round),
                    comment.agent_name,
                    comment.category.value,
                    comment.text[:100] + "..." if len(comment.text) > 100 else comment.text,
                    f"{comment.novelty_score:.2f}",
                )

            console.print(comments_table)

        # Display metrics
        if show_metrics and status_data.convergence_metrics:
            console.print("\n")
            metrics_table = Table(title="Convergence Metrics", show_header=True)
            metrics_table.add_column("Round", style="bold")
            metrics_table.add_column("Comments", justify="right")
            metrics_table.add_column("Unique", justify="right")
            metrics_table.add_column("Novelty", justify="right", style="green")
            metrics_table.add_column("Compression", justify="right", style="yellow")
            metrics_table.add_column("Context", justify="right")

            for metric in status_data.convergence_metrics:
                metrics_table.add_row(
                    str(metric.round),
                    str(metric.comment_count),
                    str(metric.unique_comment_count),
                    f"{metric.novelty_score:.2f}",
                    (
                        f"{metric.compression_ratio * 100:.1f}%"
                        if metric.compression_ratio
                        else "N/A"
                    ),
                    f"{metric.context_size:,}",
                )

            console.print(metrics_table)

    except Exception as e:
        logger.exception("Failed to get meeting status")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@app.command()
def export(
    meeting_id: str = typer.Argument(..., help="Meeting ID to export"),
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Export format: markdown, json, or html"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path (optional)"
    ),
) -> None:
    """Export meeting results to file.

    Generates formatted artifacts from meeting data.
    Supports markdown, JSON, and HTML formats.
    """
    console.print("[yellow]Export functionality not yet implemented (Sprint 5)[/yellow]")
    raise typer.Exit(1)


@app.command()
def listen(
    timeout: int = typer.Option(
        300, "--timeout", "-t", help="Timeout for human input prompts (seconds)"
    ),
) -> None:
    """Listen for meeting events and handle human-in-loop prompts.

    Subscribe to meeting.* events from RabbitMQ and display them in real-time.
    Prompts for human input when meeting.human.input.needed events are received.

    Sprint 4 Story 12: Event-driven human-in-loop
    """
    import asyncio
    import signal
    from datetime import datetime

    from rich.panel import Panel
    from rich.prompt import Prompt

    from theboard.events.consumer import get_event_consumer

    console.print("[bold cyan]TheBoard Event Listener[/bold cyan]")
    console.print("Connecting to RabbitMQ and subscribing to meeting.* events...")
    console.print("[dim]Press Ctrl+C to stop listening[/dim]\n")

    consumer = get_event_consumer()
    shutdown_event = asyncio.Event()

    # Event handlers
    def handle_round_completed(event: dict) -> None:
        """Display round completion events."""
        timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        console.print(
            Panel(
                f"[green]Round {event['round_num']} completed[/green]\n"
                f"Agent: {event['agent_name']}\n"
                f"Comments: {event['comment_count']}, "
                f"Novelty: {event['avg_novelty']:.2f}, "
                f"Cost: ${event['cost']:.4f}",
                title=f"[bold]{timestamp.strftime('%H:%M:%S')}[/bold]",
                border_style="green",
            )
        )

    def handle_human_input_needed(event: dict) -> None:
        """Handle human-in-loop prompts with timeout."""
        timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))

        console.print(
            Panel(
                f"[yellow bold]{event['reason']}[/yellow bold]\n\n"
                f"{event['prompt_text']}\n\n"
                f"Options: {', '.join(event['options'])}\n"
                f"[dim]Timeout: {timeout}s (auto-continue if no response)[/dim]",
                title=f"[bold yellow]{timestamp.strftime('%H:%M:%S')} - Human Input Needed[/bold yellow]",
                border_style="yellow",
            )
        )

        # Prompt with timeout handling
        try:
            # Note: typer.prompt doesn't support timeout, so we use rich.prompt
            choice = Prompt.ask(
                "[bold]Your choice[/bold]",
                choices=event["options"],
                default="continue",
            )
            console.print(f"[green]✓ Selected: {choice}[/green]\n")

            # TODO: Send choice back to meeting workflow via RabbitMQ
            # (requires response queue or Redis state management)

        except KeyboardInterrupt:
            console.print("\n[yellow]Input cancelled - defaulting to 'continue'[/yellow]\n")

    def handle_convergence_detected(event: dict) -> None:
        """Display convergence detection events."""
        timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        console.print(
            Panel(
                f"[cyan]Meeting converged at round {event['round_num']}[/cyan]\n"
                f"Average novelty: {event['avg_novelty']:.2f} "
                f"(threshold: {event['novelty_threshold']:.2f})\n"
                f"Total comments: {event['total_comments']}",
                title=f"[bold]{timestamp.strftime('%H:%M:%S')}[/bold]",
                border_style="cyan",
            )
        )

    def handle_meeting_completed(event: dict) -> None:
        """Display meeting completion events."""
        timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        console.print(
            Panel(
                f"[green bold]Meeting completed successfully[/green bold]\n\n"
                f"Total rounds: {event['total_rounds']}\n"
                f"Total comments: {event['total_comments']}\n"
                f"Total cost: ${event['total_cost']:.4f}\n"
                f"Convergence: {'Yes' if event['convergence_detected'] else 'No'}\n"
                f"Stopping reason: {event['stopping_reason']}",
                title=f"[bold]{timestamp.strftime('%H:%M:%S')}[/bold]",
                border_style="green",
            )
        )

    def handle_generic_event(event: dict) -> None:
        """Display all other events in compact format."""
        timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        console.print(
            f"[dim]{timestamp.strftime('%H:%M:%S')}[/dim] "
            f"[blue]{event['event_type']}[/blue] "
            f"[dim]meeting={str(event['meeting_id'])[:8]}...[/dim]"
        )

    # Register handlers
    consumer.register_handler("meeting.round_completed", handle_round_completed)
    consumer.register_handler("meeting.human.input.needed", handle_human_input_needed)
    consumer.register_handler("meeting.converged", handle_convergence_detected)
    consumer.register_handler("meeting.completed", handle_meeting_completed)

    # Generic handler for other events
    consumer.register_handler("meeting.created", handle_generic_event)
    consumer.register_handler("meeting.started", handle_generic_event)
    consumer.register_handler("agent.response.ready", handle_generic_event)
    consumer.register_handler("context.compression.triggered", handle_generic_event)

    # Signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        console.print("\n[yellow]Shutting down event listener...[/yellow]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start consumer
    async def run_consumer():
        try:
            await consumer.start()
        except Exception as e:
            console.print(f"[red]Consumer error: {e}[/red]")
            raise

    try:
        asyncio.run(run_consumer())
    except KeyboardInterrupt:
        console.print("\n[yellow]Event listener stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Display version information."""
    from theboard import __version__

    console.print(f"[bold]TheBoard[/bold] version [cyan]{__version__}[/cyan]")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
