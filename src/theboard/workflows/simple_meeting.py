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
from theboard.preferences import get_preferences_manager
from theboard.schemas import MeetingStatus

logger = logging.getLogger(__name__)


class SimpleMeetingWorkflow:
    """Simple workflow for single-agent, single-round meetings (Sprint 1 MVP).

    Agno Pattern: This workflow uses Agno-based agents instead of direct API calls.
    Session management is handled by Agno's PostgresDb, so Redis state management
    is no longer needed for agent conversations (only for meeting coordination).
    """

    def __init__(self, meeting_id: UUID, model_override: str | None = None) -> None:
        """Initialize workflow.

        Args:
            meeting_id: Meeting UUID
            model_override: Optional CLI model override (--model flag)
        """
        self.meeting_id = meeting_id
        self.model_override = model_override

        # Agno Pattern: Pass meeting_id as session_id for conversation persistence
        # Notetaker doesn't need session persistence as it's stateless extraction
        # Use model_override or preferences for notetaker
        prefs = get_preferences_manager()
        notetaker_model = prefs.get_model_for_agent(
            agent_name="notetaker",
            agent_type="notetaker",
            cli_override=model_override,
        )
        self.notetaker = NotetakerAgent(model=notetaker_model)

    async def execute(self) -> None:
        """Execute the simple meeting workflow.

        Agno Pattern: Agents automatically persist their conversation state to PostgresDb
        when session_id is provided. No need for manual Redis state management.

        Raises:
            ValueError: If meeting not found or invalid state
            RuntimeError: If workflow execution fails
        """
        with get_sync_db() as db:
            try:
                # Get meeting
                stmt = select(Meeting).where(Meeting.id == self.meeting_id)
                meeting = db.scalars(stmt).first()

                if not meeting:
                    raise ValueError(f"Meeting not found: {self.meeting_id}")

                if meeting.status != MeetingStatus.RUNNING.value:
                    raise ValueError(f"Meeting not in RUNNING state: {meeting.status}")

                logger.info("Executing simple workflow for meeting %s", self.meeting_id)

                # For Sprint 1: Create a single test agent
                # In Sprint 2, this will be replaced with agent pool selection
                test_agent = await self._get_or_create_test_agent(db)

                # Execute single round
                await self._execute_round(db, meeting, test_agent, round_num=1)

                # Update meeting status
                meeting.status = MeetingStatus.COMPLETED.value
                meeting.current_round = 1
                meeting.stopping_reason = "Single round completed (Sprint 1 MVP)"
                db.commit()

                logger.info("Simple workflow completed for meeting %s", self.meeting_id)

            except Exception as e:
                db.rollback()
                logger.exception("Simple workflow failed")
                raise RuntimeError(f"Workflow execution failed: {e!s}") from e

    async def _get_or_create_test_agent(self, db) -> Agent:
        """Get or create a test agent for Sprint 1.

        Args:
            db: Database session

        Returns:
            Agent instance from database

        Raises:
            RuntimeError: If agent creation fails
        """
        # Check if test agent exists
        stmt = select(Agent).where(Agent.name == "test-architect")
        agent = db.scalars(stmt).first()

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

        db.add(agent)
        db.commit()
        db.refresh(agent)

        logger.info("Created test agent: %s", agent.name)
        return agent

    async def _execute_round(
        self, db, meeting: Meeting, agent: Agent, round_num: int
    ) -> None:
        """Execute a single round with one agent.

        Agno Pattern: Create DomainExpertAgent with session_id=meeting_id for
        automatic conversation persistence. Agno stores the conversation in
        PostgresDb, allowing the agent to maintain context across rounds.

        CRITICAL FIX: Database session is closed BEFORE async LLM calls to prevent
        connection pool exhaustion. Session is reopened to store results.

        Args:
            db: Database session
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

        # Extract data needed for LLM calls (meeting_id, agent details, topic)
        meeting_id = meeting.id
        agent_id = agent.id
        agent_name = agent.name
        topic = meeting.topic

        # Build context (for round 1, just the topic)
        context = f"Topic: {topic}"

        # Agno Pattern: Create domain expert with session_id for persistence
        # Use preferences for model selection with full precedence hierarchy
        prefs = get_preferences_manager()
        model_to_use = prefs.get_model_for_agent(
            agent_name=agent.name,
            agent_type="domain_expert",
            cli_override=self.model_override,
        )

        expert = DomainExpertAgent(
            name=agent.name,
            expertise=agent.expertise,
            persona=agent.persona,
            background=agent.background,
            model=model_to_use,
            session_id=str(meeting_id),  # Agno uses this for conversation persistence
        )

        # SESSION LEAK FIX: Execute LLM calls WITHOUT holding database session
        # This prevents connection pool exhaustion during long-running API calls
        logger.debug("Executing LLM calls (session closed during API operations)")

        response_text = await expert.execute(context, round_num=round_num)
        metadata = expert.get_last_metadata()

        # Extract comments (also async LLM call, no session held)
        comments = await self.notetaker.extract_comments(response_text, agent_name)
        notetaker_metadata = self.notetaker.get_last_metadata()

        # SESSION LEAK FIX: Reopen session ONLY for storing results
        # This ensures minimal connection hold time
        with get_sync_db() as storage_db:
            # Store response in database
            response = Response(
                meeting_id=meeting_id,
                agent_id=agent_id,
                round=round_num,
                agent_name=agent_name,
                response_text=response_text,
                model_used=metadata["model"],
                tokens_used=metadata["tokens_used"],
                cost=metadata["cost"],
                context_size=len(context),
            )

            storage_db.add(response)
            storage_db.commit()
            storage_db.refresh(response)

            logger.info(
                "Stored response from %s: %d tokens, $%.4f",
                agent_name,
                metadata["tokens_used"],
                metadata["cost"],
            )

            # Store comments in database
            for comment in comments:
                db_comment = DBComment(
                    meeting_id=meeting_id,
                    response_id=response.id,
                    round=round_num,
                    agent_name=agent_name,
                    text=comment.text,
                    category=comment.category.value,
                    novelty_score=comment.novelty_score,
                    support_count=1,
                    is_merged=False,
                )
                storage_db.add(db_comment)

            storage_db.commit()

            logger.info(
                "Extracted and stored %d comments from %s (notetaker: %d tokens, $%.4f)",
                len(comments),
                agent_name,
                notetaker_metadata["tokens_used"],
                notetaker_metadata["cost"],
            )

            # Update meeting metrics (need to refresh meeting object in new session)
            stmt = select(Meeting).where(Meeting.id == meeting_id)
            meeting_obj = storage_db.scalars(stmt).first()

            if meeting_obj:
                meeting_obj.total_comments += len(comments)
                meeting_obj.total_cost += metadata["cost"] + notetaker_metadata["cost"]
                meeting_obj.context_size = len(context) + len(response_text)
                storage_db.commit()

        logger.info("Round %d completed successfully", round_num)
