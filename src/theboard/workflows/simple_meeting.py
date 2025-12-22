"""Simple meeting workflow for Sprint 1 MVP using Agno framework.

This workflow orchestrates a simple 1-agent, 1-round meeting using Agno-based agents.
Session persistence is handled by Agno's PostgresDb integration, eliminating the need
for manual Redis state management.
"""

import logging
from uuid import UUID

from sqlalchemy import select

from theboard.agents.domain_expert import DomainExpertAgent
from theboard.agents.notetaker import NotetakerAgent
from theboard.database import get_sync_db
from theboard.models.meeting import Agent, Comment as DBComment, Meeting, Response
from theboard.schemas import MeetingStatus

logger = logging.getLogger(__name__)


class SimpleMeetingWorkflow:
    """Simple workflow for single-agent, single-round meetings (Sprint 1 MVP).

    Agno Pattern: This workflow uses Agno-based agents instead of direct API calls.
    Session management is handled by Agno's PostgresDb, so Redis state management
    is no longer needed for agent conversations (only for meeting coordination).
    """

    def __init__(self, meeting_id: UUID) -> None:
        """Initialize workflow.

        Args:
            meeting_id: Meeting UUID
        """
        self.meeting_id = meeting_id
        self.db = get_sync_db()

        # Agno Pattern: Pass meeting_id as session_id for conversation persistence
        # Notetaker doesn't need session persistence as it's stateless extraction
        self.notetaker = NotetakerAgent()

    async def execute(self) -> None:
        """Execute the simple meeting workflow.

        Agno Pattern: Agents automatically persist their conversation state to PostgresDb
        when session_id is provided. No need for manual Redis state management.

        Raises:
            ValueError: If meeting not found or invalid state
            RuntimeError: If workflow execution fails
        """
        try:
            # Get meeting
            stmt = select(Meeting).where(Meeting.id == self.meeting_id)
            meeting = self.db.scalars(stmt).first()

            if not meeting:
                raise ValueError(f"Meeting not found: {self.meeting_id}")

            if meeting.status != MeetingStatus.RUNNING.value:
                raise ValueError(f"Meeting not in RUNNING state: {meeting.status}")

            logger.info("Executing simple workflow for meeting %s", self.meeting_id)

            # For Sprint 1: Create a single test agent
            # In Sprint 2, this will be replaced with agent pool selection
            test_agent = await self._get_or_create_test_agent()

            # Execute single round
            await self._execute_round(meeting, test_agent, round_num=1)

            # Update meeting status
            meeting.status = MeetingStatus.COMPLETED.value
            meeting.current_round = 1
            meeting.stopping_reason = "Single round completed (Sprint 1 MVP)"
            self.db.commit()

            logger.info("Simple workflow completed for meeting %s", self.meeting_id)

        except Exception as e:
            self.db.rollback()
            logger.exception("Simple workflow failed")
            raise RuntimeError(f"Workflow execution failed: {e!s}") from e
        finally:
            self.db.close()

    async def _get_or_create_test_agent(self) -> Agent:
        """Get or create a test agent for Sprint 1.

        Returns:
            Agent instance from database

        Raises:
            RuntimeError: If agent creation fails
        """
        # Check if test agent exists
        stmt = select(Agent).where(Agent.name == "test-architect")
        agent = self.db.scalars(stmt).first()

        if agent:
            return agent

        # Create test agent
        agent = Agent(
            name="test-architect",
            expertise="Software architecture, system design, and technical decision-making",
            persona="A pragmatic architect who balances technical excellence with practical concerns",
            background="15 years of experience designing scalable distributed systems",
            agent_type="plaintext",
            default_model="claude-sonnet-4-20250514",
            is_active=True,
        )

        self.db.add(agent)
        self.db.commit()
        self.db.refresh(agent)

        logger.info("Created test agent: %s", agent.name)
        return agent

    async def _execute_round(
        self, meeting: Meeting, agent: Agent, round_num: int
    ) -> None:
        """Execute a single round with one agent.

        Agno Pattern: Create DomainExpertAgent with session_id=meeting_id for
        automatic conversation persistence. Agno stores the conversation in
        PostgresDb, allowing the agent to maintain context across rounds.

        Args:
            meeting: Meeting instance
            agent: Agent instance
            round_num: Round number

        Raises:
            RuntimeError: If round execution fails
        """
        logger.info(
            "Executing round %d for meeting %s with agent %s",
            round_num,
            meeting.id,
            agent.name,
        )

        # Build context (for round 1, just the topic)
        context = f"Topic: {meeting.topic}"

        # Agno Pattern: Create domain expert with session_id for persistence
        # The agent will automatically save conversation to PostgresDb
        expert = DomainExpertAgent(
            name=agent.name,
            expertise=agent.expertise,
            persona=agent.persona,
            background=agent.background,
            model=agent.default_model,
            session_id=str(meeting.id),  # Agno uses this for conversation persistence
        )

        # Generate response using Agno agent
        response_text = await expert.execute(context, round_num=round_num)
        metadata = expert.get_last_metadata()

        # Store response in database
        response = Response(
            meeting_id=meeting.id,
            agent_id=agent.id,
            round=round_num,
            agent_name=agent.name,
            response_text=response_text,
            model_used=metadata["model"],
            tokens_used=metadata["tokens_used"],
            cost=metadata["cost"],
            context_size=len(context),
        )

        self.db.add(response)
        self.db.commit()
        self.db.refresh(response)

        logger.info(
            "Stored response from %s: %d tokens, $%.4f",
            agent.name,
            metadata["tokens_used"],
            metadata["cost"],
        )

        # Agno Pattern: Extract comments using Agno agent with structured output
        # The notetaker agent uses output_schema for automatic JSON validation
        comments = await self.notetaker.extract_comments(response_text, agent.name)
        notetaker_metadata = self.notetaker.get_last_metadata()

        # Store comments in database
        for comment in comments:
            db_comment = DBComment(
                meeting_id=meeting.id,
                response_id=response.id,
                round=round_num,
                agent_name=agent.name,
                text=comment.text,
                category=comment.category.value,
                novelty_score=comment.novelty_score,
                support_count=1,
                is_merged=False,
            )
            self.db.add(db_comment)

        self.db.commit()

        logger.info(
            "Extracted and stored %d comments from %s (notetaker: %d tokens, $%.4f)",
            len(comments),
            agent.name,
            notetaker_metadata["tokens_used"],
            notetaker_metadata["cost"],
        )

        # Update meeting metrics
        meeting.total_comments += len(comments)
        meeting.total_cost += metadata["cost"] + notetaker_metadata["cost"]
        meeting.context_size = len(context) + len(response_text)

        # Note: Redis state management for meeting coordination could be added here
        # if needed, but agent conversation state is now in Agno's PostgresDb

        self.db.commit()

        logger.info("Round %d completed successfully", round_num)
