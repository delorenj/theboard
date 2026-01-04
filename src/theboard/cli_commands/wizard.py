"""Wizard command for guided meeting creation."""

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from theboard.schemas import (
    AgentPoolSize,
    LengthStrategy,
    MeetingType,
    WizardConfig,
)
from theboard.services.cost_estimator import estimate_meeting_cost
from theboard.services.meeting_service import create_meeting

logger = logging.getLogger(__name__)
console = Console()

# Create wizard subcommand
wizard_app = typer.Typer(
    name="wizard",
    help="Interactive wizard for meeting creation",
    add_completion=False,
)


@wizard_app.command()
def create() -> None:
    """Launch interactive meeting creation wizard.

    Guides users through 4 steps:
    1. Meeting topic
    2. Meeting type
    3. Meeting length
    4. Agent team size

    Shows cost preview before creation and optionally saves as template.
    """
    try:
        # Welcome
        console.print(
            Panel.fit(
                "[bold]TheBoard Meeting Creation Wizard[/bold]\n\n"
                "I'll guide you through creating an effective brainstorming meeting.\n"
                "This wizard focuses on essential settings - advanced options available via templates.",
                title="[bold cyan]Welcome[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print()

        # Step 1: Topic
        console.print("[bold]Step 1/4: Meeting Topic[/bold]")
        console.print("[dim]What are we brainstorming? (10-500 characters)[/dim]\n")

        topic = ""
        while not (10 <= len(topic) <= 500):
            topic = Prompt.ask("> Topic")
            if len(topic) < 10:
                console.print("[red]Topic too short (minimum 10 characters)[/red]")
            elif len(topic) > 500:
                console.print("[red]Topic too long (maximum 500 characters)[/red]")

        console.print()

        # Step 2: Meeting Type
        console.print("[bold]Step 2/4: Meeting Type[/bold]")
        console.print("[dim]Select the meeting style:[/dim]\n")

        meeting_type_choices = {
            "1": (MeetingType.BRAINSTORM, "General ideation and creative thinking"),
            "2": (MeetingType.RISK_ASSESSMENT, "Identify and evaluate risks"),
            "3": (MeetingType.TECHNICAL_REVIEW, "Evaluate technical approach or design"),
            "4": (MeetingType.DEBATE, "Explore opposing viewpoints"),
            "5": (MeetingType.RESEARCH, "Fact-based analysis and investigation"),
        }

        for key, (_, desc) in meeting_type_choices.items():
            console.print(f"  {key}. {desc}")

        meeting_type_choice = Prompt.ask(
            "\n> Select", choices=["1", "2", "3", "4", "5"], default="1"
        )
        meeting_type, _ = meeting_type_choices[meeting_type_choice]

        console.print()

        # Step 3: Length Strategy
        console.print("[bold]Step 3/4: Meeting Length[/bold]")
        console.print("[dim]How thorough should the discussion be?[/dim]\n")

        length_choices = {
            "1": (
                LengthStrategy.QUICK,
                "Quick (2 rounds, ~5 min, $0.10-$0.30)",
            ),
            "2": (
                LengthStrategy.STANDARD,
                "Standard (4 rounds, ~12 min, $0.30-$0.70) [Recommended]",
            ),
            "3": (
                LengthStrategy.THOROUGH,
                "Thorough (5 rounds, ~17 min, $0.70-$1.50)",
            ),
        }

        for key, (_, desc) in length_choices.items():
            console.print(f"  {key}. {desc}")

        length_choice = Prompt.ask(
            "\n> Select", choices=["1", "2", "3"], default="2"
        )
        length_strategy, _ = length_choices[length_choice]

        console.print()

        # Step 4: Agent Pool Size
        console.print("[bold]Step 4/4: Agent Team Size[/bold]")
        console.print("[dim]How many agents should participate?[/dim]\n")

        pool_choices = {
            "1": (AgentPoolSize.SMALL, "Small (3 agents - focused discussion)"),
            "2": (AgentPoolSize.MEDIUM, "Medium (5 agents - balanced perspectives) [Recommended]"),
            "3": (AgentPoolSize.LARGE, "Large (8 agents - comprehensive coverage)"),
        }

        for key, (_, desc) in pool_choices.items():
            console.print(f"  {key}. {desc}")

        pool_choice = Prompt.ask(
            "\n> Select", choices=["1", "2", "3"], default="2"
        )
        agent_pool_size, _ = pool_choices[pool_choice]

        console.print()

        # Build wizard config
        config = WizardConfig(
            topic=topic,
            meeting_type=meeting_type,
            length_strategy=length_strategy,
            agent_pool_size=agent_pool_size,
        )

        # Generate cost estimate
        estimate = estimate_meeting_cost(config)

        # Display preview
        console.print("[bold]Configuration Preview[/bold]\n")

        preview_table = Table(show_header=False, box=None)
        preview_table.add_column("Field", style="bold cyan")
        preview_table.add_column("Value")

        preview_table.add_row("Topic", topic[:60] + "..." if len(topic) > 60 else topic)
        preview_table.add_row("Type", meeting_type.value.replace("_", " ").title())
        preview_table.add_row("Length", length_strategy.value.title())
        preview_table.add_row(
            "Agents",
            f"{estimate.breakdown['agents']} (auto-selected based on topic)",
        )

        console.print(preview_table)
        console.print()

        # Cost estimate panel
        console.print(
            Panel.fit(
                f"[bold]Estimated Cost:[/bold] ${estimate.min_cost:.2f} - ${estimate.max_cost:.2f}\n"
                f"[bold]Expected:[/bold] ${estimate.expected_cost:.2f}\n\n"
                f"[bold]Estimated Time:[/bold] {estimate.min_duration_minutes}-{estimate.max_duration_minutes} minutes\n"
                f"[bold]Expected:[/bold] {estimate.expected_duration_minutes} minutes\n\n"
                f"[bold]Expected Comments:[/bold] {estimate.expected_comment_count}\n\n"
                f"[dim]Breakdown:[/dim]\n"
                f"[dim]• {estimate.breakdown['agents']} agents × ~{estimate.breakdown['expected_rounds']} rounds[/dim]\n"
                f"[dim]• ~{estimate.breakdown['tokens_per_round']:,} tokens/round[/dim]\n"
                f"[dim]• Model: {estimate.breakdown['model']} (${estimate.breakdown['model_rate_per_1m_tokens']:.2f} per 1M tokens)[/dim]\n"
                f"[dim]• Convergence likelihood: {estimate.breakdown['convergence_likelihood']}[/dim]",
                title="[bold green]Cost & Time Estimate[/bold green]",
                border_style="green",
            )
        )
        console.print()

        # Confirm creation
        if not Confirm.ask("[bold]Create and run this meeting?[/bold]", default=True):
            console.print("[yellow]Meeting creation cancelled[/yellow]")
            raise typer.Exit(0)

        # Create meeting
        meeting_params = config.to_meeting_create_params()

        with console.status("[bold green]Creating meeting...", spinner="dots"):
            meeting = create_meeting(**meeting_params)

        # Success
        console.print(
            Panel.fit(
                f"[green]✓[/green] Meeting created successfully!\n\n"
                f"[bold]Meeting ID:[/bold] {meeting.id}\n"
                f"[bold]Topic:[/bold] {meeting.topic}\n"
                f"[bold]Agents:[/bold] {meeting_params['agent_count']} (auto-selected)\n"
                f"[bold]Max Rounds:[/bold] {meeting.max_rounds}",
                title="[bold green]Meeting Created[/bold green]",
                border_style="green",
            )
        )
        console.print()

        # Optional: Save as template
        if Confirm.ask(
            "[dim]Save this configuration as a template for future use?[/dim]",
            default=False,
        ):
            template_name = Prompt.ask(
                "Template name (e.g., 'quick-brainstorm', 'security-review')"
            )

            template_dir = Path.home() / ".theboard" / "templates"
            template_dir.mkdir(parents=True, exist_ok=True)

            template_path = template_dir / f"{template_name}.json"

            template_data = {
                "name": template_name,
                "description": f"{meeting_type.value.replace('_', ' ').title()} meeting template",
                "config": {
                    "meeting_type": meeting_type.value,
                    "length_strategy": length_strategy.value,
                    "agent_pool_size": agent_pool_size.value,
                    "topic": "<TOPIC_PLACEHOLDER>",
                },
            }

            template_path.write_text(json.dumps(template_data, indent=2))
            console.print(
                f"\n[green]✓[/green] Template saved: [bold]{template_path}[/bold]"
            )
            console.print(
                f"[dim]Reuse with:[/dim] [bold]board create --template {template_name}[/bold]"
            )

        # Show next steps
        console.print(
            f"\n[dim]Run the meeting with:[/dim] [bold]board run --last[/bold]"
        )
        console.print(
            f"[dim]View status:[/dim] [bold]board status --last[/bold]"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Wizard cancelled[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        logger.exception("Wizard failed")
        console.print(f"[red]Error: {e!s}[/red]")
        raise typer.Exit(1) from e
