"""Command-line interface for TheBoard."""

import logging
from pathlib import Path
from uuid import UUID

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from theboard.cli_commands.agents import agents_app
from theboard.cli_commands.config import config_app
from theboard.config import settings
from theboard.schemas import MeetingResponse, MeetingStatus, StrategyType

# Initialize Typer app
app = typer.Typer(
    name="board",
    help="TheBoard - Multi-Agent Brainstorming Simulation System",
    add_completion=False,
)

# Register command groups
app.add_typer(agents_app, name="agents")
app.add_typer(config_app, name="config")

# Initialize Rich console
console = Console()

# Configure logging
log_file = Path("debug.log")

file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, console=console),
        file_handler,
    ],
    force=True,
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
                model_override=model,
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
            f"\n[dim]Run the meeting with:[/dim] [bold]board run --last[/bold] [dim](or select from [bold]board run[/bold])[/dim]"
        )

    except Exception as e:
        logger.exception("Failed to create meeting")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


def _run_with_live_progress(meeting_id: UUID, interactive: bool, rerun: bool) -> MeetingResponse:
    """Run meeting with live progress display (Sprint 6 Story 17).

    Displays real-time updates: round, agent, comment count, novelty score.
    Uses Rich.Live() for smooth updates.
    """
    import threading
    import time
    from rich.live import Live
    from rich.table import Table
    from theboard.services.meeting_service import run_meeting
    from theboard.database import get_sync_db
    from theboard.models.meeting import Meeting as DBMeeting
    from sqlalchemy import select

    # Shared state
    result = None
    error = None
    running = threading.Event()
    running.set()

    def run_meeting_thread():
        """Execute meeting in background thread."""
        nonlocal result, error
        try:
            result = run_meeting(meeting_id=meeting_id, interactive=interactive, rerun=rerun)
        except Exception as e:
            error = e
        finally:
            running.clear()

    def generate_progress_table() -> Table:
        """Generate progress table from current database state."""
        table = Table(title="Meeting Progress", show_header=True, header_style="bold magenta")
        table.add_column("Round", style="cyan", justify="center", width=8)
        table.add_column("Status", style="yellow", justify="center", width=12)
        table.add_column("Comments", style="green", justify="right", width=10)
        table.add_column("Context Size", style="white", justify="right", width=14)

        # Query current meeting state
        with get_sync_db() as db:
            stmt = select(DBMeeting).where(DBMeeting.id == meeting_id)
            meeting = db.scalars(stmt).first()

            if meeting:
                table.add_row(
                    f"{meeting.current_round}/{meeting.max_rounds}",
                    meeting.status,
                    str(meeting.total_comments),
                    f"{meeting.context_size:,}",
                )

        return table

    # Start meeting execution in background thread
    thread = threading.Thread(target=run_meeting_thread, daemon=True)
    thread.start()

    # Display live progress updates
    with Live(generate_progress_table(), refresh_per_second=2, console=console) as live:
        while running.is_set():
            time.sleep(0.5)  # Poll every 500ms
            live.update(generate_progress_table())

    # Check for errors
    if error:
        raise error

    return result


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
                console.print(f"[dim]Using most recent meeting:[/dim] {meetings[0].topic}")
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
                        f"  {i}. {meeting.topic} "
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
                console.print(f"[dim]Selected:[/dim] {selected.topic}\n")
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

        # Run meeting with live progress updates (Sprint 6 Story 17)
        result = _run_with_live_progress(uuid_id, interactive, rerun)

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
            f"\n[dim]View details with:[/dim] [bold]board status[/bold] [dim](or [bold]board status --last[/bold])[/dim]"
        )

    except Exception as e:
        logger.exception("Failed to run meeting")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


@app.command()
def status(
    meeting_id: str | None = typer.Argument(None, help="Meeting ID to check (optional - shows selector if omitted)"),
    show_comments: bool = typer.Option(
        True, "--comments/--no-comments", help="Show recent comments"
    ),
    show_metrics: bool = typer.Option(
        True, "--metrics/--no-metrics", help="Show convergence metrics"
    ),
    last: bool = typer.Option(
        False, "--last", help="Show status of most recent meeting"
    ),
) -> None:
    """Display meeting status and details.

    Shows meeting state, responses, comments, and convergence metrics.
    If no meeting ID is provided, shows an interactive selector.
    Use --last to show the most recent meeting without selection.
    """
    try:
        # Import here to avoid circular dependencies
        from rich.prompt import Prompt
        from theboard.services.meeting_service import get_meeting_status, list_recent_meetings

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
                console.print(f"[dim]Showing most recent meeting:[/dim] {meetings[0].topic}\n")
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
                        f"  {i}. {meeting.topic} "
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
                console.print(f"[dim]Selected:[/dim] {selected.topic}\n")
        else:
            # Meeting ID provided - parse it
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

        # Generate and display meeting artifact (replaces truncated table)
        if show_comments:
            try:
                from theboard.services.export_service import ExportService
                import re

                export_service = ExportService()

                # Create artifacts directory
                artifacts_dir = Path("artifacts/meetings")
                artifacts_dir.mkdir(parents=True, exist_ok=True)

                # Generate safe filename
                meeting_topic = status_data.meeting.topic
                safe_topic = re.sub(r"[^a-zA-Z0-9]", "_", meeting_topic)[:50]
                timestamp = status_data.meeting.created_at.strftime("%Y%m%d_%H%M%S") if status_data.meeting.created_at else "unknown"
                filename = f"meeting_{timestamp}_{safe_topic}_{uuid_id}.md"
                artifact_path = artifacts_dir / filename

                # Export to markdown
                export_service.export_markdown(uuid_id, artifact_path)

                console.print("\n")
                console.print(
                    Panel.fit(
                        f"[green]✓[/green] Meeting artifact generated\n\n"
                        f"[bold]Path:[/bold] {artifact_path}\n"
                        f"[bold]Comments:[/bold] {status_data.meeting.total_comments}\n"
                        f"[bold]Rounds:[/bold] {status_data.meeting.current_round}",
                        title="[bold cyan]Meeting Document[/bold cyan]",
                        border_style="cyan",
                    )
                )

                console.print(f"\n[dim]View full details:[/dim] [bold]cat {artifact_path}[/bold]")
                console.print(f"[dim]Or open in editor:[/dim] [bold]$EDITOR {artifact_path}[/bold]\n")

            except Exception as e:
                logger.warning("Failed to generate meeting artifact: %s", e)
                # Fallback to truncated table if artifact generation fails
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
        "markdown", "--format", "-f", help="Export format: markdown, json, html, or template"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path (optional)"
    ),
    template: str | None = typer.Option(
        None, "--template", "-t", help="Template name for template format"
    ),
) -> None:
    """Export meeting results to file.

    Generates formatted artifacts from meeting data.
    Supports markdown, JSON, HTML, and custom template formats.
    """
    try:
        # Import here to avoid circular dependencies
        from theboard.services.export_service import ExportService

        # Parse meeting ID
        try:
            uuid_id = UUID(meeting_id)
        except ValueError as e:
            console.print(f"[red]Error: Invalid meeting ID format: {meeting_id}[/red]")
            raise typer.Exit(1) from e

        # Validate format
        valid_formats = ["markdown", "json", "html", "template"]
        if format not in valid_formats:
            console.print(
                f"[red]Error: Invalid format '{format}'. Must be one of: {', '.join(valid_formats)}[/red]"
            )
            raise typer.Exit(1)

        # Validate template requirement
        if format == "template" and not template:
            console.print(
                "[red]Error: --template is required when using template format[/red]"
            )
            raise typer.Exit(1)

        # Auto-generate output path if not provided
        if output is None:
            extensions = {
                "markdown": "md",
                "json": "json",
                "html": "html",
                "template": "txt",
            }
            ext = extensions.get(format, "txt")
            output = Path(f"meeting_{meeting_id}_{format}.{ext}")

        # Initialize export service
        export_service = ExportService()

        # Export based on format
        with console.status(f"[bold green]Exporting as {format}...", spinner="dots"):
            if format == "markdown":
                result = export_service.export_markdown(uuid_id, output)
            elif format == "json":
                result = export_service.export_json(uuid_id, output)
            elif format == "html":
                result = export_service.export_html(uuid_id, output)
            elif format == "template":
                result = export_service.export_with_template(uuid_id, template, output)

        # Display success
        console.print(
            Panel.fit(
                f"[green]✓[/green] Export completed successfully!\\n\\n"
                f"[bold]Meeting ID:[/bold] {uuid_id}\\n"
                f"[bold]Format:[/bold] {format}\\n"
                + (f"[bold]Template:[/bold] {template}\\n" if template else "")
                + f"[bold]Output:[/bold] {output}\\n"
                f"[bold]Size:[/bold] {len(result):,} characters",
                title="[bold green]Export Complete[/bold green]",
                border_style="green",
            )
        )

        console.print(f"\\n[dim]View the exported file at:[/dim] [bold]{output}[/bold]")

    except Exception as e:
        logger.exception("Failed to export meeting")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e


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
