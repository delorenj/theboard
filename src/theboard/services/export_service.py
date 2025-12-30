"""Export service for generating meeting artifacts in various formats.

Supports:
- Markdown: Formatted text with headings, lists, and code blocks
- JSON: Structured data export
- HTML: Styled web page with CSS
- Template-based: Custom Jinja2 templates
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from jinja2 import Environment, FileSystemLoader, Template
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from theboard.database import get_sync_db
from theboard.models.meeting import Comment, ConvergenceMetric, Meeting, Response

logger = logging.getLogger(__name__)


class ExportService:
    """Service for exporting meeting artifacts in multiple formats."""

    DEFAULT_TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "exports"

    def __init__(self, templates_dir: Path | None = None):
        """Initialize export service.

        Args:
            templates_dir: Custom templates directory (default: built-in templates)
        """
        self.templates_dir = templates_dir or self.DEFAULT_TEMPLATES_DIR
        self.jinja_env = None

        # Initialize Jinja2 environment if templates directory exists
        if self.templates_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=True,
            )

    def _get_meeting_data(self, meeting_id: UUID) -> dict[str, Any]:
        """Fetch complete meeting data with all relationships.

        Args:
            meeting_id: Meeting UUID

        Returns:
            Dict with meeting, responses, comments, and metrics

        Raises:
            ValueError: If meeting not found
        """
        with get_sync_db() as db:
            # Fetch meeting with all relationships
            query = (
                select(Meeting)
                .where(Meeting.id == meeting_id)
                .options(
                    joinedload(Meeting.responses).joinedload(Response.comments),
                    joinedload(Meeting.comments),
                    joinedload(Meeting.convergence_metrics),
                )
            )

            result = db.execute(query)
            meeting = result.scalar_one_or_none()

            if not meeting:
                raise ValueError(f"Meeting {meeting_id} not found")

            # Build data structure
            return {
                "meeting": {
                    "id": str(meeting.id),
                    "topic": meeting.topic,
                    "strategy": meeting.strategy,
                    "status": meeting.status,
                    "max_rounds": meeting.max_rounds,
                    "current_round": meeting.current_round,
                    "convergence_detected": meeting.convergence_detected,
                    "stopping_reason": meeting.stopping_reason,
                    "model_override": meeting.model_override,
                    "context_size": meeting.context_size,
                    "total_comments": meeting.total_comments,
                    "total_cost": meeting.total_cost,
                    "created_at": meeting.created_at.isoformat() if meeting.created_at else None,
                    "updated_at": meeting.updated_at.isoformat() if meeting.updated_at else None,
                },
                "responses": [
                    {
                        "id": str(resp.id),
                        "round": resp.round,
                        "agent_name": resp.agent_name,
                        "response_text": resp.response_text,
                        "model_used": resp.model_used,
                        "tokens_used": resp.tokens_used,
                        "cost": resp.cost,
                        "context_size": resp.context_size,
                        "created_at": resp.created_at.isoformat() if resp.created_at else None,
                        "comment_count": len(resp.comments),
                    }
                    for resp in sorted(meeting.responses, key=lambda r: (r.round, r.created_at or datetime.min))
                ],
                "comments": [
                    {
                        "id": str(comment.id),
                        "round": comment.round,
                        "agent_name": comment.agent_name,
                        "text": comment.text,
                        "category": comment.category,
                        "novelty_score": comment.novelty_score,
                        "support_count": comment.support_count,
                        "is_merged": comment.is_merged,
                        "merged_from_ids": comment.merged_from_ids,
                    }
                    for comment in sorted(meeting.comments, key=lambda c: (c.round, c.created_at or datetime.min))
                ],
                "convergence_metrics": [
                    {
                        "round": metric.round,
                        "novelty_score": metric.novelty_score,
                        "comment_count": metric.comment_count,
                        "unique_comment_count": metric.unique_comment_count,
                        "compression_ratio": metric.compression_ratio,
                        "context_size": metric.context_size,
                    }
                    for metric in sorted(meeting.convergence_metrics, key=lambda m: m.round)
                ],
            }

    def export_markdown(self, meeting_id: UUID, output_path: Path | None = None) -> str:
        """Export meeting as formatted Markdown.

        Args:
            meeting_id: Meeting UUID
            output_path: Optional path to save markdown file

        Returns:
            Markdown string

        Raises:
            ValueError: If meeting not found
        """
        data = self._get_meeting_data(meeting_id)
        meeting = data["meeting"]
        responses = data["responses"]
        comments = data["comments"]
        metrics = data["convergence_metrics"]

        # Build markdown
        lines = []

        # Header
        lines.append(f"# Meeting: {meeting['topic']}")
        lines.append("")
        lines.append(f"**Status:** {meeting['status']}")
        lines.append(f"**Strategy:** {meeting['strategy']}")
        lines.append(f"**Rounds:** {meeting['current_round']}/{meeting['max_rounds']}")
        lines.append(f"**Total Comments:** {meeting['total_comments']}")
        lines.append(f"**Total Cost:** ${meeting['total_cost']:.4f}")
        if meeting["convergence_detected"]:
            lines.append(f"**Convergence:** Detected ({meeting['stopping_reason']})")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"This {meeting['strategy']} meeting ran for {meeting['current_round']} rounds, ")
        lines.append(f"generating {meeting['total_comments']} comments across {len(responses)} responses.")
        lines.append(f"Final context size: {meeting['context_size']:,} characters.")
        lines.append("")

        # Comments by Category
        if comments:
            lines.append("## Key Insights")
            lines.append("")

            # Group by category
            by_category: dict[str, list[dict]] = {}
            for comment in comments:
                category = comment["category"]
                by_category.setdefault(category, []).append(comment)

            for category, cat_comments in sorted(by_category.items()):
                lines.append(f"### {category.replace('_', ' ').title()}")
                lines.append("")
                for comment in cat_comments:
                    novelty = f" (novelty: {comment['novelty_score']:.2f})" if comment["novelty_score"] > 0 else ""
                    merged = " [MERGED]" if comment["is_merged"] else ""
                    lines.append(f"- **{comment['agent_name']}** (Round {comment['round']}){novelty}{merged}: {comment['text']}")
                lines.append("")

        # Round-by-Round Detail
        lines.append("## Round-by-Round Responses")
        lines.append("")

        # Group responses by round
        by_round: dict[int, list[dict]] = {}
        for resp in responses:
            by_round.setdefault(resp["round"], []).append(resp)

        for round_num in sorted(by_round.keys()):
            lines.append(f"### Round {round_num}")
            lines.append("")

            # Metrics for this round
            round_metrics = next((m for m in metrics if m["round"] == round_num), None)
            if round_metrics:
                lines.append(f"**Metrics:** {round_metrics['comment_count']} comments, ")
                lines.append(f"{round_metrics['unique_comment_count']} unique, ")
                lines.append(f"novelty: {round_metrics['novelty_score']:.2f}")
                if round_metrics["compression_ratio"]:
                    lines.append(f", compression: {round_metrics['compression_ratio']:.1f}%")
                lines.append("")
                lines.append("")

            # Responses
            for resp in by_round[round_num]:
                lines.append(f"#### {resp['agent_name']} ({resp['model_used']})")
                lines.append("")
                lines.append(f"**Tokens:** {resp['tokens_used']} | **Cost:** ${resp['cost']:.4f} | **Comments Extracted:** {resp['comment_count']}")
                lines.append("")
                lines.append("```")
                lines.append(resp["response_text"])
                lines.append("```")
                lines.append("")

        # Convergence Metrics Table
        if metrics:
            lines.append("## Convergence Metrics")
            lines.append("")
            lines.append("| Round | Comments | Unique | Novelty | Compression | Context Size |")
            lines.append("|-------|----------|--------|---------|-------------|--------------|")
            for metric in metrics:
                compression = f"{metric['compression_ratio']:.1f}%" if metric["compression_ratio"] else "N/A"
                lines.append(
                    f"| {metric['round']} | {metric['comment_count']} | {metric['unique_comment_count']} | "
                    f"{metric['novelty_score']:.2f} | {compression} | {metric['context_size']:,} |"
                )
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by TheBoard Export Service*")

        markdown = "\n".join(lines)

        # Save to file if requested
        if output_path:
            output_path.write_text(markdown, encoding="utf-8")
            logger.info(f"Exported markdown to {output_path}")

        return markdown

    def export_json(self, meeting_id: UUID, output_path: Path | None = None, pretty: bool = True) -> str:
        """Export meeting as structured JSON.

        Args:
            meeting_id: Meeting UUID
            output_path: Optional path to save JSON file
            pretty: Whether to pretty-print JSON (default: True)

        Returns:
            JSON string

        Raises:
            ValueError: If meeting not found
        """
        data = self._get_meeting_data(meeting_id)

        # Add metadata
        export_data = {
            "export_metadata": {
                "exported_at": datetime.now().isoformat(),
                "format_version": "1.0",
                "meeting_id": str(meeting_id),
            },
            **data,
        }

        # Serialize to JSON
        json_str = json.dumps(export_data, indent=2 if pretty else None, ensure_ascii=False)

        # Save to file if requested
        if output_path:
            output_path.write_text(json_str, encoding="utf-8")
            logger.info(f"Exported JSON to {output_path}")

        return json_str

    def export_html(self, meeting_id: UUID, output_path: Path | None = None) -> str:
        """Export meeting as styled HTML page.

        Args:
            meeting_id: Meeting UUID
            output_path: Optional path to save HTML file

        Returns:
            HTML string

        Raises:
            ValueError: If meeting not found
        """
        data = self._get_meeting_data(meeting_id)
        meeting = data["meeting"]
        responses = data["responses"]
        comments = data["comments"]
        metrics = data["convergence_metrics"]

        # Build HTML
        html_parts = []

        # HTML header with CSS
        html_parts.append("<!DOCTYPE html>")
        html_parts.append('<html lang="en">')
        html_parts.append("<head>")
        html_parts.append('    <meta charset="UTF-8">')
        html_parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append(f"    <title>Meeting: {meeting['topic']}</title>")
        html_parts.append("    <style>")
        html_parts.append(self._get_default_css())
        html_parts.append("    </style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append('    <div class="container">')

        # Header
        html_parts.append(f'        <h1>Meeting: {meeting["topic"]}</h1>')
        html_parts.append('        <div class="metadata">')
        html_parts.append(f'            <span class="badge status-{meeting["status"]}">{meeting["status"]}</span>')
        html_parts.append(f'            <span>Strategy: {meeting["strategy"]}</span>')
        html_parts.append(f'            <span>Rounds: {meeting["current_round"]}/{meeting["max_rounds"]}</span>')
        html_parts.append(f'            <span>Comments: {meeting["total_comments"]}</span>')
        html_parts.append(f'            <span>Cost: ${meeting["total_cost"]:.4f}</span>')
        html_parts.append('        </div>')

        # Summary card
        html_parts.append('        <div class="card summary">')
        html_parts.append('            <h2>Summary</h2>')
        html_parts.append(f'            <p>This {meeting["strategy"]} meeting ran for {meeting["current_round"]} rounds, ')
        html_parts.append(f'            generating {meeting["total_comments"]} comments across {len(responses)} responses.')
        html_parts.append(f'            Final context size: {meeting["context_size"]:,} characters.</p>')
        if meeting["convergence_detected"]:
            html_parts.append(f'            <p class="convergence">✓ Convergence detected: {meeting["stopping_reason"]}</p>')
        html_parts.append('        </div>')

        # Comments by category
        if comments:
            html_parts.append('        <div class="card">')
            html_parts.append('            <h2>Key Insights</h2>')

            by_category: dict[str, list[dict]] = {}
            for comment in comments:
                by_category.setdefault(comment["category"], []).append(comment)

            for category, cat_comments in sorted(by_category.items()):
                html_parts.append(f'            <h3>{category.replace("_", " ").title()}</h3>')
                html_parts.append('            <ul class="comments">')
                for comment in cat_comments:
                    badges = []
                    if comment["novelty_score"] > 0:
                        badges.append(f'<span class="badge novelty">novelty: {comment["novelty_score"]:.2f}</span>')
                    if comment["is_merged"]:
                        badges.append('<span class="badge merged">MERGED</span>')
                    badges_html = " ".join(badges)
                    html_parts.append(f'                <li>')
                    html_parts.append(f'                    <strong>{comment["agent_name"]}</strong> (Round {comment["round"]}) {badges_html}')
                    html_parts.append(f'                    <p>{comment["text"]}</p>')
                    html_parts.append('                </li>')
                html_parts.append('            </ul>')

            html_parts.append('        </div>')

        # Round-by-round responses
        html_parts.append('        <div class="card">')
        html_parts.append('            <h2>Round-by-Round Responses</h2>')

        by_round: dict[int, list[dict]] = {}
        for resp in responses:
            by_round.setdefault(resp["round"], []).append(resp)

        for round_num in sorted(by_round.keys()):
            html_parts.append(f'            <div class="round">')
            html_parts.append(f'                <h3>Round {round_num}</h3>')

            # Metrics
            round_metrics = next((m for m in metrics if m["round"] == round_num), None)
            if round_metrics:
                compression = f"{round_metrics['compression_ratio']:.1f}%" if round_metrics["compression_ratio"] else "N/A"
                html_parts.append('                <div class="metrics">')
                html_parts.append(f'                    <span>{round_metrics["comment_count"]} comments</span>')
                html_parts.append(f'                    <span>{round_metrics["unique_comment_count"]} unique</span>')
                html_parts.append(f'                    <span>novelty: {round_metrics["novelty_score"]:.2f}</span>')
                html_parts.append(f'                    <span>compression: {compression}</span>')
                html_parts.append('                </div>')

            # Responses
            for resp in by_round[round_num]:
                html_parts.append('                <div class="response">')
                html_parts.append(f'                    <h4>{resp["agent_name"]} <span class="model">({resp["model_used"]})</span></h4>')
                html_parts.append('                    <div class="response-meta">')
                html_parts.append(f'                        <span>Tokens: {resp["tokens_used"]}</span>')
                html_parts.append(f'                        <span>Cost: ${resp["cost"]:.4f}</span>')
                html_parts.append(f'                        <span>Comments: {resp["comment_count"]}</span>')
                html_parts.append('                    </div>')
                html_parts.append(f'                    <pre class="response-text">{resp["response_text"]}</pre>')
                html_parts.append('                </div>')

            html_parts.append('            </div>')

        html_parts.append('        </div>')

        # Convergence metrics table
        if metrics:
            html_parts.append('        <div class="card">')
            html_parts.append('            <h2>Convergence Metrics</h2>')
            html_parts.append('            <table>')
            html_parts.append('                <thead>')
            html_parts.append('                    <tr>')
            html_parts.append('                        <th>Round</th>')
            html_parts.append('                        <th>Comments</th>')
            html_parts.append('                        <th>Unique</th>')
            html_parts.append('                        <th>Novelty</th>')
            html_parts.append('                        <th>Compression</th>')
            html_parts.append('                        <th>Context Size</th>')
            html_parts.append('                    </tr>')
            html_parts.append('                </thead>')
            html_parts.append('                <tbody>')
            for metric in metrics:
                compression = f"{metric['compression_ratio']:.1f}%" if metric["compression_ratio"] else "N/A"
                html_parts.append('                    <tr>')
                html_parts.append(f'                        <td>{metric["round"]}</td>')
                html_parts.append(f'                        <td>{metric["comment_count"]}</td>')
                html_parts.append(f'                        <td>{metric["unique_comment_count"]}</td>')
                html_parts.append(f'                        <td>{metric["novelty_score"]:.2f}</td>')
                html_parts.append(f'                        <td>{compression}</td>')
                html_parts.append(f'                        <td>{metric["context_size"]:,}</td>')
                html_parts.append('                    </tr>')
            html_parts.append('                </tbody>')
            html_parts.append('            </table>')
            html_parts.append('        </div>')

        # Footer
        html_parts.append('        <footer>')
        html_parts.append(f'            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by TheBoard Export Service')
        html_parts.append('        </footer>')
        html_parts.append('    </div>')
        html_parts.append('</body>')
        html_parts.append('</html>')

        html = "\n".join(html_parts)

        # Save to file if requested
        if output_path:
            output_path.write_text(html, encoding="utf-8")
            logger.info(f"Exported HTML to {output_path}")

        return html

    def _get_default_css(self) -> str:
        """Get default CSS styling for HTML export."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }

        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }

        h2 {
            color: #2c3e50;
            margin-top: 30px;
            margin-bottom: 15px;
        }

        h3 {
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        h4 {
            color: #7f8c8d;
            margin-bottom: 8px;
        }

        .metadata {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 4px;
        }

        .metadata span {
            color: #555;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge.status-completed {
            background: #27ae60;
            color: white;
        }

        .badge.status-running {
            background: #3498db;
            color: white;
        }

        .badge.status-paused {
            background: #f39c12;
            color: white;
        }

        .badge.novelty {
            background: #9b59b6;
            color: white;
        }

        .badge.merged {
            background: #e74c3c;
            color: white;
        }

        .card {
            margin-bottom: 30px;
            padding: 20px;
            background: #fafafa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }

        .card.summary {
            background: #e8f4f8;
        }

        .convergence {
            color: #27ae60;
            font-weight: 600;
            margin-top: 10px;
        }

        .comments {
            list-style: none;
        }

        .comments li {
            margin-bottom: 15px;
            padding: 12px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }

        .comments li strong {
            color: #2c3e50;
        }

        .comments li p {
            margin-top: 5px;
            color: #555;
        }

        .round {
            margin-bottom: 30px;
        }

        .metrics {
            display: flex;
            gap: 20px;
            margin: 10px 0 20px 0;
            padding: 10px;
            background: white;
            border-radius: 4px;
            font-size: 0.9em;
            color: #666;
        }

        .response {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 4px;
            border: 1px solid #ddd;
        }

        .model {
            color: #7f8c8d;
            font-size: 0.9em;
            font-weight: normal;
        }

        .response-meta {
            display: flex;
            gap: 15px;
            margin: 10px 0;
            font-size: 0.9em;
            color: #666;
        }

        .response-text {
            background: #f8f8f8;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.9em;
            line-height: 1.5;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background: #ecf0f1;
            color: #2c3e50;
            font-weight: 600;
        }

        tr:hover {
            background: #f8f9fa;
        }

        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }

            .container {
                box-shadow: none;
                padding: 20px;
            }
        }
        """

    def export_with_template(
        self,
        meeting_id: UUID,
        template_name: str,
        output_path: Path | None = None,
    ) -> str:
        """Export meeting using custom Jinja2 template.

        Args:
            meeting_id: Meeting UUID
            template_name: Template filename (e.g., 'custom.md.j2')
            output_path: Optional path to save output

        Returns:
            Rendered template string

        Raises:
            ValueError: If meeting not found or template not found
        """
        if not self.jinja_env:
            raise ValueError(f"Templates directory not found: {self.templates_dir}")

        # Get meeting data
        data = self._get_meeting_data(meeting_id)

        # Load and render template
        try:
            template = self.jinja_env.get_template(template_name)
        except Exception as e:
            raise ValueError(f"Template '{template_name}' not found: {e}")

        # Add helpers to template context
        template_data = {
            **data,
            "now": datetime.now(),
        }

        rendered = template.render(**template_data)

        # Save to file if requested
        if output_path:
            output_path.write_text(rendered, encoding="utf-8")
            logger.info(f"Exported template to {output_path}")

        return rendered
