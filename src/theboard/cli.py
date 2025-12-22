"""Command-line interface for TheBoard."""

import logging
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from theboard.config import settings
from theboard.schemas import MeetingStatus, StrategyType

# Initialize Typer app
app = typer.Typer(
    name="board",
    help="TheBoard - Multi-Agent Brainstorming Simulation System",
    add_completion=False,
)

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
) -> None:
    """Create a new brainstorming meeting.

    Creates a meeting with the specified topic and configuration.
    Agents can be auto-selected based on topic relevance or manually chosen.
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
    meeting_id: str = typer.Argument(..., help="Meeting ID to run"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Enable human-in-the-loop prompts"
    ),
) -> None:
    """Run a brainstorming meeting.

    Executes the meeting with the configured strategy and agents.
    Displays real-time progress and results.
    """
    try:
        # Import here to avoid circular dependencies
        from theboard.services.meeting_service import run_meeting

        # Parse meeting ID
        try:
            uuid_id = UUID(meeting_id)
        except ValueError as e:
            console.print(f"[red]Error: Invalid meeting ID format: {meeting_id}[/red]")
            raise typer.Exit(1) from e

        # Run meeting
        console.print(f"\n[bold]Starting meeting {uuid_id}...[/bold]\n")

        with console.status("[bold green]Running meeting...", spinner="dots"):
            result = run_meeting(meeting_id=uuid_id, interactive=interactive)

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
            f"\n[dim]View details with:[/dim] [bold]board status {meeting_id}[/bold]"
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
def version() -> None:
    """Display version information."""
    from theboard import __version__

    console.print(f"[bold]TheBoard[/bold] version [cyan]{__version__}[/cyan]")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
