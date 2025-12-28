"""Multi-agent meeting workflow for Sprint 2.

This workflow orchestrates multi-agent, multi-round brainstorming meetings with:
- Multiple agents per round (sequential turn-taking)
- Multi-round execution with context accumulation
- Basic convergence detection via novelty scores
- Session management following Sprint 1.5 patterns
"""

import logging
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from theboard.agents.domain_expert import DomainExpertAgent
from theboard.agents.notetaker import NotetakerAgent
from theboard.database import get_sync_db
from theboard.events import (
    CommentExtractedEvent,
    MeetingCompletedEvent,
    MeetingConvergedEvent,
    MeetingFailedEvent,
    MeetingStartedEvent,
    RoundCompletedEvent,
    get_event_emitter,
)
from theboard.models.meeting import Agent, Comment as DBComment, Meeting, Response
from theboard.preferences import get_preferences_manager
from theboard.schemas import MeetingStatus

# Sprint 3: Import embedding service and compressor agent
from theboard.agents.compressor import CompressorAgent
from theboard.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class MultiAgentMeetingWorkflow:
    """Multi-agent, multi-round meeting workflow (Sprint 2).

    Key Features:
    - Multiple agents per meeting (selected pool)
    - Sequential strategy (agents take turns each round)
    - Context accumulation across rounds
    - Basic convergence detection (novelty threshold)
    - Proper session management (Sprint 1.5 pattern)
    """

    def __init__(
        self,
        meeting_id: UUID,
        model_override: str | None = None,
        novelty_threshold: float = 0.3,
        enable_compression: bool = True,
    ) -> None:
        """Initialize multi-agent workflow.

        Args:
            meeting_id: Meeting UUID
            model_override: Optional CLI model override (--model flag)
            novelty_threshold: Convergence threshold for novelty scores (default 0.3)
            enable_compression: Enable comment compression (default True, Sprint 3)
        """
        self.meeting_id = meeting_id
        self.model_override = model_override
        self.novelty_threshold = novelty_threshold
        self.enable_compression = enable_compression

        # Initialize event emitter (Sprint 2.5)
        self.emitter = get_event_emitter()

        # Hybrid model strategy tracking (Sprint 4 Story 13)
        self.promoted_agents: set[str] = set()  # Agents promoted to premium models

        # Initialize notetaker (stateless, doesn't need session persistence)
        prefs = get_preferences_manager()
        notetaker_model = prefs.get_model_for_agent(
            agent_name="notetaker",
            agent_type="notetaker",
            cli_override=model_override,
        )
        self.notetaker = NotetakerAgent(model=notetaker_model)

        # Initialize compressor agent (Sprint 3 Story 9)
        if self.enable_compression:
            compressor_model = prefs.get_model_for_agent(
                agent_name="compressor",
                agent_type="compressor",
                cli_override=model_override,
            )
            self.compressor = CompressorAgent(model=compressor_model)
        else:
            self.compressor = None

    def pause_meeting(self, reason: str = "User requested pause") -> bool:
        """Pause meeting execution and save state to Redis.

        Sprint 4 Story 12: Meeting pause capability for human-in-loop.

        Args:
            reason: Reason for pausing (for logging/audit)

        Returns:
            True if paused successfully, False if Redis unavailable
        """
        from theboard.utils.redis_manager import RedisManager

        redis = RedisManager()

        with get_sync_db() as db:
            meeting = db.query(Meeting).filter(Meeting.id == self.meeting_id).first()
            if not meeting:
                logger.error("Meeting %s not found for pause", self.meeting_id)
                return False

            # Update meeting status to paused
            meeting.status = "paused"
            db.commit()

            # Save state to Redis
            state = {
                "status": "paused",
                "reason": reason,
                "current_round": meeting.current_round,
                "context_size": meeting.context_size,
                "total_comments": meeting.total_comments,
                "paused_at": datetime.now(UTC).isoformat(),
            }

            if redis.set_meeting_state(str(self.meeting_id), state):
                logger.info(
                    "Meeting %s paused: %s (round %d)",
                    self.meeting_id,
                    reason,
                    meeting.current_round,
                )
                return True

            logger.warning("Failed to save pause state to Redis for meeting %s", self.meeting_id)
            return False

    def resume_meeting(self) -> bool:
        """Resume paused meeting from Redis state.

        Sprint 4 Story 12: Meeting resume capability for human-in-loop.

        Returns:
            True if resumed successfully, False if not paused or Redis unavailable
        """
        from theboard.utils.redis_manager import RedisManager

        redis = RedisManager()

        # Load state from Redis
        state = redis.get_meeting_state(str(self.meeting_id))
        if not state:
            logger.warning("No pause state found in Redis for meeting %s", self.meeting_id)
            return False

        with get_sync_db() as db:
            meeting = db.query(Meeting).filter(Meeting.id == self.meeting_id).first()
            if not meeting:
                logger.error("Meeting %s not found for resume", self.meeting_id)
                return False

            # Update meeting status to running
            meeting.status = "running"
            db.commit()

            logger.info(
                "Meeting %s resumed from round %d (paused: %s)",
                self.meeting_id,
                state.get("current_round", 0),
                state.get("reason", "unknown"),
            )
            return True

    def check_pause_requested(self) -> bool:
        """Check if meeting pause has been requested via Redis.

        Used during execution to check for external pause requests
        (e.g., from CLI human-in-loop prompts).

        Returns:
            True if pause requested, False otherwise
        """
        from theboard.utils.redis_manager import RedisManager

        redis = RedisManager()
        state = redis.get_meeting_state(str(self.meeting_id))

        if state and state.get("status") == "pause_requested":
            logger.info("Pause requested for meeting %s", self.meeting_id)
            return True

        return False

    def _promote_top_performers(self, round_num: int) -> None:
        """Promote top-performing agents to premium models.

        Sprint 4 Story 13: Hybrid Model Strategy

        After round 1, calculate engagement metrics and promote top 20%
        of agents from budget models (DeepSeek) to premium models (Opus).

        Args:
            round_num: Round number just completed
        """
        from theboard.services.engagement_metrics import EngagementMetricsCalculator

        calculator = EngagementMetricsCalculator(self.meeting_id)

        # Get top 20% performers
        top_performers = calculator.get_top_performers(round_num, top_percent=0.2)

        if not top_performers:
            logger.warning("No top performers found for promotion (round %d)", round_num)
            return

        # Track promoted agents
        for perf in top_performers:
            self.promoted_agents.add(perf.agent_name)

        logger.info(
            "Promoted %d agents to premium models: %s",
            len(top_performers),
            [p.agent_name for p in top_performers],
        )

        # Log engagement scores
        for perf in top_performers:
            logger.info(
                "  %s: engagement=%.3f (refs=%d, novelty=%.2f, comments=%d)",
                perf.agent_name,
                perf.engagement_score,
                perf.peer_references,
                perf.avg_novelty,
                perf.comment_count,
            )

    def _get_agent_model(self, agent_name: str, agent_type: str = "domain_expert") -> str:
        """Get model for an agent considering hybrid model strategy.

        Sprint 4 Story 13: Hybrid Model Strategy

        Logic:
        1. If hybrid_models disabled: Use preferences (normal behavior)
        2. If hybrid_models enabled:
           - Round 1: Use budget model (DeepSeek)
           - Round 2+: Use premium model (Opus) for promoted agents, budget for others

        Args:
            agent_name: Agent name
            agent_type: Agent type (default: domain_expert)

        Returns:
            Model identifier
        """
        from theboard.models.pricing import get_promotion_model

        prefs = get_preferences_manager()

        with get_sync_db() as db:
            meeting = db.query(Meeting).filter(Meeting.id == self.meeting_id).first()

            if not meeting or not meeting.hybrid_models:
                # Hybrid models disabled: use normal preferences
                return prefs.get_model_for_agent(
                    agent_name=agent_name,
                    agent_type=agent_type,
                    cli_override=self.model_override,
                )

            # Hybrid models enabled
            current_round = meeting.current_round

            if current_round == 1:
                # Round 1: Everyone uses budget model (DeepSeek)
                return "deepseek/deepseek-chat"

            # Round 2+: Promoted agents use premium, others use budget
            if agent_name in self.promoted_agents:
                # Get budget model first, then promote it
                budget_model = "deepseek/deepseek-chat"
                promoted_model = get_promotion_model(budget_model)
                logger.debug(
                    "Agent %s promoted: %s → %s",
                    agent_name,
                    budget_model,
                    promoted_model,
                )
                return promoted_model
            else:
                # Non-promoted agents stay on budget
                return "deepseek/deepseek-chat"

    async def execute(self) -> None:
        """Execute the multi-agent meeting workflow.

        Workflow:
        1. Load meeting and selected agents
        2. For each round (1 to max_rounds):
           a. Build cumulative context from previous rounds
           b. For each agent sequentially:
              - Execute agent response with context
              - Extract comments from response
           c. Check convergence (average novelty score)
           d. Break if converged
        3. Update meeting status

        Session Management:
        - Follows Sprint 1.5 pattern: extract → execute → store
        - No session held during async LLM calls
        - New session for each storage operation

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

                logger.info(
                    "Executing multi-agent workflow for meeting %s (max_rounds=%d)",
                    self.meeting_id,
                    meeting.max_rounds,
                )

                # Get selected agents for this meeting
                # Sprint 2 Phase 2: Topic-based agent selection
                agents = await self._get_selected_agents(db, meeting.topic)

                if not agents:
                    raise ValueError("No agents available for meeting")

                logger.info("Selected %d agents for meeting", len(agents))

                # Emit meeting started event (Sprint 2.5)
                self.emitter.emit(
                    MeetingStartedEvent(
                        meeting_id=self.meeting_id,
                        selected_agents=[agent.name for agent in agents],
                        agent_count=len(agents),
                    )
                )

            except Exception as e:
                db.rollback()
                logger.exception("Failed to initialize multi-agent workflow")
                raise RuntimeError(f"Workflow initialization failed: {e!s}") from e

        # Execute rounds (session closed during round execution)
        try:
            converged = False
            for round_num in range(1, meeting.max_rounds + 1):
                logger.info("Starting round %d of %d", round_num, meeting.max_rounds)

                # Sprint 4 Story 11: Execute round based on strategy
                if meeting.strategy == "greedy":
                    avg_novelty = await self._execute_round_greedy(agents, round_num)
                else:
                    # Default: sequential strategy
                    avg_novelty = await self._execute_round(agents, round_num)

                # Sprint 3 Story 9: Compress comments after each round
                if self.enable_compression and self.compressor:
                    try:
                        compression_metrics = self.compressor.compress_comments(
                            meeting_id=self.meeting_id,
                            round_num=round_num,
                        )
                        logger.info(
                            "Round %d compression: %d → %d comments (%.1f%% reduction)",
                            round_num,
                            compression_metrics.original_count,
                            compression_metrics.compressed_count,
                            compression_metrics.reduction_percentage,
                        )

                        # Update meeting compression metrics
                        with get_sync_db() as compression_db:
                            compression_stmt = select(Meeting).where(Meeting.id == self.meeting_id)
                            compression_meeting = compression_db.scalars(compression_stmt).first()
                            if compression_meeting:
                                # Track compression cost
                                compressor_metadata = self.compressor.get_last_metadata()
                                compression_meeting.total_cost += compressor_metadata["cost"]
                                compression_db.commit()

                    except Exception as e:
                        # Non-fatal: compression failure doesn't block workflow
                        logger.warning(
                            "Compression failed for round %d: %s",
                            round_num,
                            str(e),
                            exc_info=True,
                        )

                # Sprint 4 Story 13: Promote top performers after round 1
                if round_num == 1 and meeting.hybrid_models:
                    logger.info("Calculating engagement metrics for round 1...")
                    self._promote_top_performers(round_num)

                # Check convergence
                if avg_novelty < self.novelty_threshold:
                    logger.info(
                        "Convergence detected at round %d (novelty=%.3f < threshold=%.3f)",
                        round_num,
                        avg_novelty,
                        self.novelty_threshold,
                    )
                    converged = True

                    # Emit convergence event (Sprint 2.5)
                    # Need to get total comments count for the event
                    with get_sync_db() as convergence_db:
                        convergence_stmt = select(Meeting).where(Meeting.id == self.meeting_id)
                        convergence_meeting = convergence_db.scalars(convergence_stmt).first()
                        if convergence_meeting:
                            self.emitter.emit(
                                MeetingConvergedEvent(
                                    meeting_id=self.meeting_id,
                                    round_num=round_num,
                                    avg_novelty=avg_novelty,
                                    novelty_threshold=self.novelty_threshold,
                                    total_comments=convergence_meeting.total_comments,
                                )
                            )

                    break

            # Update meeting status
            with get_sync_db() as final_db:
                stmt = select(Meeting).where(Meeting.id == self.meeting_id)
                final_meeting = final_db.scalars(stmt).first()

                if final_meeting:
                    final_meeting.status = MeetingStatus.COMPLETED.value
                    final_meeting.current_round = round_num
                    final_meeting.convergence_detected = converged
                    final_meeting.stopping_reason = (
                        f"Converged at round {round_num} (novelty={avg_novelty:.3f})"
                        if converged
                        else f"Max rounds reached ({meeting.max_rounds})"
                    )
                    final_db.commit()

                    # Emit meeting completed event (Sprint 2.5)
                    self.emitter.emit(
                        MeetingCompletedEvent(
                            meeting_id=self.meeting_id,
                            total_rounds=round_num,
                            total_comments=final_meeting.total_comments,
                            total_cost=final_meeting.total_cost,
                            convergence_detected=converged,
                            stopping_reason=final_meeting.stopping_reason,
                        )
                    )

                logger.info(
                    "Multi-agent workflow completed for meeting %s (converged=%s)",
                    self.meeting_id,
                    converged,
                )

        except Exception as e:
            # Update meeting status to failed
            with get_sync_db() as error_db:
                stmt = select(Meeting).where(Meeting.id == self.meeting_id)
                failed_meeting = error_db.scalars(stmt).first()
                if failed_meeting:
                    failed_meeting.status = MeetingStatus.FAILED.value
                    failed_meeting.stopping_reason = f"Workflow error: {e!s}"
                    error_db.commit()

                    # Emit meeting failed event (Sprint 2.5)
                    self.emitter.emit(
                        MeetingFailedEvent(
                            meeting_id=self.meeting_id,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            round_num=failed_meeting.current_round,
                            agent_name=None,  # Workflow-level failure
                        )
                    )

            logger.exception("Multi-agent workflow failed")
            raise RuntimeError(f"Workflow execution failed: {e!s}") from e

    async def _get_selected_agents(self, db, topic: str) -> list[Agent]:
        """Get selected agents for this meeting using topic-based selection.

        Sprint 2 Phase 2: Keyword-based agent selection algorithm.
        Extracts keywords from topic and matches against agent expertise/persona/background.

        Selection Algorithm:
        1. Extract keywords from topic (lowercase, remove stopwords)
        2. For each agent, count keyword matches in expertise/persona/background
        3. Calculate relevance score (matches / total_keywords)
        4. Return agents sorted by relevance score (highest first)
        5. Include all agents with at least one keyword match

        Args:
            db: Database session
            topic: Meeting topic string

        Returns:
            List of Agent instances sorted by relevance score (highest first)

        Raises:
            RuntimeError: If agent query fails
        """
        # Extract keywords from topic
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "how", "what", "when", "where", "who", "why"
        }

        # Tokenize and clean topic
        topic_lower = topic.lower()
        # Remove punctuation and split
        import re
        words = re.findall(r'\b\w+\b', topic_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        if not keywords:
            # If no keywords extracted, return all active agents
            logger.warning("No keywords extracted from topic, returning all active agents")
            stmt = select(Agent).where(Agent.is_active == True).order_by(Agent.name)  # noqa: E712
            agents = db.scalars(stmt).all()
            return list(agents)

        logger.info("Extracted %d keywords from topic: %s", len(keywords), keywords)

        # Get all active agents
        stmt = select(Agent).where(Agent.is_active == True)  # noqa: E712
        all_agents = db.scalars(stmt).all()

        # Score each agent based on keyword matches
        agent_scores = []
        for agent in all_agents:
            # Combine all agent text fields for matching
            agent_text = " ".join([
                agent.expertise.lower(),
                agent.persona.lower() if agent.persona else "",
                agent.background.lower() if agent.background else "",
            ])

            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in agent_text)

            # Calculate relevance score
            relevance_score = matches / len(keywords) if keywords else 0

            # Only include agents with at least one match
            if matches > 0:
                agent_scores.append((agent, relevance_score, matches))
                logger.debug(
                    "Agent %s: %d matches, %.2f relevance",
                    agent.name,
                    matches,
                    relevance_score,
                )

        # Sort by relevance score (highest first), then by number of matches
        agent_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Extract agents from scored tuples
        selected_agents = [agent for agent, score, matches in agent_scores]

        if not selected_agents:
            # Fallback: if no agents match, return all active agents
            logger.warning("No agents matched topic keywords, returning all active agents")
            return list(all_agents)

        logger.info(
            "Selected %d/%d agents based on topic relevance (avg score: %.2f)",
            len(selected_agents),
            len(all_agents),
            sum(score for _, score, _ in agent_scores) / len(agent_scores) if agent_scores else 0,
        )

        return selected_agents

    async def _execute_round(self, agents: list[Agent], round_num: int) -> float:
        """Execute a single round with all agents sequentially.

        Session Management (Sprint 1.5 pattern):
        - Build context (read-only session for previous responses)
        - For each agent:
          - Execute LLM calls WITHOUT holding session
          - Store results in new session
        - Calculate round novelty score

        Args:
            agents: List of Agent instances for this round
            round_num: Current round number

        Returns:
            Average novelty score for this round

        Raises:
            RuntimeError: If round execution fails
        """
        logger.info("Executing round %d with %d agents", round_num, len(agents))

        # Build cumulative context from previous rounds
        context = await self._build_context(round_num)

        # Execute each agent sequentially
        round_novelty_scores = []

        for agent in agents:
            try:
                # Execute agent response and extract comments
                # (NO session held during LLM calls - Sprint 1.5 pattern)
                agent_novelty = await self._execute_agent_turn(
                    agent, context, round_num
                )
                round_novelty_scores.append(agent_novelty)

            except Exception as e:
                logger.error(
                    "Agent %s failed in round %d: %s", agent.name, round_num, e
                )
                # Continue with other agents even if one fails
                continue

        # Calculate average novelty for convergence detection
        avg_novelty = (
            sum(round_novelty_scores) / len(round_novelty_scores)
            if round_novelty_scores
            else 1.0
        )

        logger.info(
            "Round %d completed: %d/%d agents succeeded, avg_novelty=%.3f",
            round_num,
            len(round_novelty_scores),
            len(agents),
            avg_novelty,
        )

        # Emit round completed event (Sprint 2.5)
        # Get round metrics for event payload
        with get_sync_db() as round_db:
            # Get responses for this round to calculate metrics
            response_stmt = (
                select(Response)
                .where(Response.meeting_id == self.meeting_id)
                .where(Response.round == round_num)
            )
            round_responses = round_db.scalars(response_stmt).all()

            # Get comments for this round
            comment_stmt = (
                select(DBComment)
                .where(DBComment.meeting_id == self.meeting_id)
                .where(DBComment.round == round_num)
            )
            round_comments = round_db.scalars(comment_stmt).all()

            # Calculate round metrics
            total_tokens = sum(r.tokens_used for r in round_responses)
            total_cost = sum(r.cost for r in round_responses)
            total_response_length = sum(len(r.response_text) for r in round_responses)
            comment_count = len(round_comments)

            # Emit event for each agent in the round
            for response in round_responses:
                agent_comments = [c for c in round_comments if c.agent_name == response.agent_name]
                agent_avg_novelty = (
                    sum(c.novelty_score for c in agent_comments) / len(agent_comments)
                    if agent_comments
                    else 0.0
                )

                self.emitter.emit(
                    RoundCompletedEvent(
                        meeting_id=self.meeting_id,
                        round_num=round_num,
                        agent_name=response.agent_name,
                        response_length=len(response.response_text),
                        comment_count=len(agent_comments),
                        avg_novelty=agent_avg_novelty,
                        tokens_used=response.tokens_used,
                        cost=response.cost,
                    )
                )

        return avg_novelty

    async def _execute_round_greedy(self, agents: list[Agent], round_num: int) -> float:
        """Execute a single round with greedy strategy (Sprint 4 Story 11).

        Greedy Strategy:
        1. All agents respond in parallel (asyncio.gather)
        2. Comment-response phase: Agents respond to each other's comments
        3. Higher token cost (N² responses) but faster convergence

        Args:
            agents: List of Agent instances for this round
            round_num: Current round number

        Returns:
            Average novelty score for this round

        Raises:
            RuntimeError: If round execution fails
        """
        import asyncio

        logger.info(
            "Executing round %d with greedy strategy (%d agents in parallel)",
            round_num,
            len(agents),
        )

        # Build cumulative context from previous rounds
        context = await self._build_context(round_num)

        # Phase 1: Parallel agent responses
        logger.info("Phase 1: Parallel agent responses")

        # Execute all agents in parallel using asyncio.gather
        tasks = [
            self._execute_agent_turn(agent, context, round_num)
            for agent in agents
        ]

        try:
            # Gather all responses in parallel
            round_novelty_scores = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and log errors
            valid_novelty_scores = []
            for i, result in enumerate(round_novelty_scores):
                if isinstance(result, Exception):
                    logger.error(
                        "Agent %s failed in round %d: %s",
                        agents[i].name,
                        round_num,
                        result,
                    )
                else:
                    valid_novelty_scores.append(result)

            round_novelty_scores = valid_novelty_scores

        except Exception as e:
            logger.error("Parallel execution failed in round %d: %s", round_num, e)
            raise RuntimeError(f"Greedy strategy execution failed: {e!s}") from e

        # Phase 2: Comment-response phase
        # Each agent responds to other agents' comments
        logger.info("Phase 2: Comment-response phase (%d × %d responses)", len(agents), len(agents))

        # Get all comments from this round for comment-response phase
        with get_sync_db() as comment_db:
            comment_stmt = (
                select(DBComment)
                .where(DBComment.meeting_id == self.meeting_id)
                .where(DBComment.round == round_num)
                .where(DBComment.is_merged == False)  # noqa: E712
            )
            round_comments = comment_db.scalars(comment_stmt).all()

        # Build comment context for response phase
        if round_comments:
            comment_context = "\n\n".join([
                f"[{comment.agent_name}] {comment.text}"
                for comment in round_comments
            ])

            # Each agent responds to the collective comments
            # This creates N² responses total
            comment_response_tasks = [
                self._execute_agent_comment_response(
                    agent, comment_context, round_num
                )
                for agent in agents
            ]

            try:
                # Execute comment-response phase in parallel
                comment_novelty_scores = await asyncio.gather(
                    *comment_response_tasks, return_exceptions=True
                )

                # Filter out exceptions
                valid_comment_scores = []
                for i, result in enumerate(comment_novelty_scores):
                    if isinstance(result, Exception):
                        logger.error(
                            "Agent %s comment-response failed: %s",
                            agents[i].name,
                            result,
                        )
                    else:
                        valid_comment_scores.append(result)

                # Include comment-response novelty in average
                round_novelty_scores.extend(valid_comment_scores)

            except Exception as e:
                logger.warning("Comment-response phase failed: %s", e)
                # Non-fatal: continue with initial responses

        # Calculate average novelty for convergence detection
        avg_novelty = (
            sum(round_novelty_scores) / len(round_novelty_scores)
            if round_novelty_scores
            else 1.0
        )

        logger.info(
            "Round %d (greedy) completed: %d total responses, avg_novelty=%.3f",
            round_num,
            len(round_novelty_scores),
            avg_novelty,
        )

        # Emit per-response RoundCompletedEvents (Sprint 2.5)
        # This matches the sequential strategy pattern
        with get_sync_db() as round_db:
            # Get all responses for this round
            response_stmt = (
                select(Response)
                .where(Response.meeting_id == self.meeting_id)
                .where(Response.round == round_num)
            )
            round_responses = round_db.scalars(response_stmt).all()

            # Emit individual events for each response
            for response in round_responses:
                # Get comments for this specific response
                comment_stmt = (
                    select(DBComment)
                    .where(DBComment.meeting_id == self.meeting_id)
                    .where(DBComment.response_id == response.id)
                )
                response_comments = round_db.scalars(comment_stmt).all()

                # Calculate average novelty for this response
                response_avg_novelty = (
                    sum(c.novelty_score for c in response_comments) / len(response_comments)
                    if response_comments
                    else 0.5  # Default novelty
                )

                self.emitter.emit(
                    RoundCompletedEvent(
                        meeting_id=self.meeting_id,
                        round_num=round_num,
                        agent_name=response.agent_name,
                        response_length=len(response.response_text),
                        comment_count=len(response_comments),
                        avg_novelty=response_avg_novelty,
                        tokens_used=response.tokens_used,
                        cost=response.cost,
                    )
                )

        return avg_novelty

    async def _execute_agent_comment_response(
        self, agent: Agent, comment_context: str, round_num: int
    ) -> float:
        """Execute agent response to collective comments (greedy strategy).

        Args:
            agent: Agent instance
            comment_context: Collective comments from all agents
            round_num: Current round number

        Returns:
            Novelty score for this response
        """
        prefs = get_preferences_manager()
        agent_model = prefs.get_model_for_agent(
            agent_name=agent.name,
            agent_type="domain_expert",
            cli_override=self.model_override,
        )

        # Create domain expert for this response
        domain_expert = DomainExpertAgent(
            name=agent.name,
            expertise=agent.expertise or "General expertise",
            model=agent_model,
            session_id=f"meeting-{self.meeting_id}-{agent.name}",
        )

        # Build prompt for comment response
        comment_response_prompt = (
            f"Review the following comments from your fellow agents:\n\n{comment_context}\n\n"
            "Provide your response addressing key points, agreements, or counterpoints. "
            "Be concise and focused on advancing the discussion."
        )

        # Execute agent response
        response_text = domain_expert.respond_to_context(comment_response_prompt)

        # Store response in database
        with get_sync_db() as db:
            # Get agent_id from database
            agent_stmt = select(Agent).where(Agent.name == agent.name)
            db_agent = db.scalars(agent_stmt).first()

            if not db_agent:
                logger.warning("Agent %s not found in database", agent.name)
                return 0.5  # Default novelty

            response = Response(
                meeting_id=self.meeting_id,
                agent_id=db_agent.id,
                round=round_num,
                agent_name=agent.name,
                response_text=response_text,
                tokens_used=domain_expert.get_last_metadata()["tokens_used"],
                cost=domain_expert.get_last_metadata()["cost"],
                model_used=agent_model,
            )
            db.add(response)
            db.flush()  # Get response ID

            # Extract comments from response
            comments = self.notetaker.extract_comments(
                response_text=response_text,
                agent_name=agent.name,
                meeting_id=self.meeting_id,
                round_num=round_num,
                response_id=response.id,
            )

            # Store comments to database
            db_comments = []
            for comment in comments:
                db_comment = DBComment(
                    meeting_id=self.meeting_id,
                    response_id=response.id,
                    round=round_num,
                    agent_name=agent.name,
                    text=comment.text,
                    category=comment.category,
                    novelty_score=comment.novelty_score,
                    support_count=1,
                    is_merged=False,
                )
                db.add(db_comment)
                db_comments.append(db_comment)

            db.commit()

            # Generate and store embeddings for comments (Sprint 3)
            if db_comments:
                try:
                    embedding_service = get_embedding_service()

                    comment_ids = [c.id for c in db_comments]
                    texts = [c.text for c in db_comments]
                    agent_names = [c.agent_name for c in db_comments]

                    embedding_service.store_comment_embeddings(
                        comment_ids=comment_ids,
                        texts=texts,
                        meeting_id=str(self.meeting_id),
                        round_num=round_num,
                        agent_names=agent_names,
                    )

                except Exception as e:
                    logger.warning("Failed to generate embeddings: %s", str(e))

            # Calculate novelty score (average of comment novelty scores)
            novelty_score = (
                sum(c.novelty_score for c in comments) / len(comments)
                if comments
                else 0.5
            )

            return novelty_score

    async def _build_context(self, current_round: int) -> str:
        """Build cumulative context from previous rounds.

        Context Structure:
        - Round 1: Topic only
        - Round 2+: Topic + all comments from previous rounds

        Formula: Context_r = Topic + Σ(Comments from rounds 1 to r-1)

        Args:
            current_round: Current round number

        Returns:
            Formatted context string

        Raises:
            RuntimeError: If context building fails
        """
        with get_sync_db() as db:
            # Get meeting topic
            stmt = select(Meeting).where(Meeting.id == self.meeting_id)
            meeting = db.scalars(stmt).first()

            if not meeting:
                raise RuntimeError(f"Meeting not found: {self.meeting_id}")

            context_parts = [f"Topic: {meeting.topic}\n"]

            # Add comments from previous rounds
            if current_round > 1:
                comment_stmt = (
                    select(DBComment)
                    .where(DBComment.meeting_id == self.meeting_id)
                    .where(DBComment.round < current_round)
                    .order_by(DBComment.round, DBComment.created_at)
                )
                prev_comments = db.scalars(comment_stmt).all()

                if prev_comments:
                    context_parts.append("\nPrevious Discussion:\n")
                    for comment in prev_comments:
                        context_parts.append(
                            f"[Round {comment.round}, {comment.agent_name}] "
                            f"{comment.category.upper()}: {comment.text}\n"
                        )

            context = "".join(context_parts)
            logger.debug("Built context for round %d: %d chars", current_round, len(context))

            return context

    async def _execute_agent_turn(
        self, agent: Agent, context: str, round_num: int
    ) -> float:
        """Execute a single agent's turn (response + comment extraction).

        Session Management (Sprint 1.5 pattern):
        1. Extract agent data (no ongoing session needed)
        2. Execute LLM calls WITHOUT holding session
        3. Store results in new session

        Args:
            agent: Agent instance
            context: Cumulative context for this round
            round_num: Current round number

        Returns:
            Average novelty score for this agent's comments

        Raises:
            RuntimeError: If agent execution fails
        """
        logger.info("Executing agent %s for round %d", agent.name, round_num)

        # Extract agent data (from object, not requiring active session)
        agent_id = agent.id
        agent_name = agent.name

        # Get model for this agent (considers hybrid model strategy - Story 13)
        model_to_use = self._get_agent_model(
            agent_name=agent.name,
            agent_type="domain_expert",
        )

        # Create domain expert (Agno handles its own session persistence)
        expert = DomainExpertAgent(
            name=agent.name,
            expertise=agent.expertise,
            persona=agent.persona,
            background=agent.background,
            model=model_to_use,
            session_id=str(self.meeting_id),  # Agno persistence per meeting
        )

        # SESSION LEAK FIX: Execute LLM calls WITHOUT holding database session
        response_text = await expert.execute(context, round_num=round_num)
        metadata = expert.get_last_metadata()

        # Extract comments (also async LLM call, no session held)
        comments = await self.notetaker.extract_comments(response_text, agent_name)
        notetaker_metadata = self.notetaker.get_last_metadata()

        # SESSION LEAK FIX: Reopen session ONLY for storing results
        with get_sync_db() as storage_db:
            # Store response
            response = Response(
                meeting_id=self.meeting_id,
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

            # Store comments
            novelty_scores = []
            db_comments = []  # Sprint 3: Track for embedding generation

            for comment in comments:
                db_comment = DBComment(
                    meeting_id=self.meeting_id,
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
                db_comments.append(db_comment)  # Sprint 3: Collect for embeddings
                novelty_scores.append(comment.novelty_score)

            storage_db.commit()

            logger.info(
                "Extracted and stored %d comments from %s (notetaker: %d tokens, $%.4f)",
                len(comments),
                agent_name,
                notetaker_metadata["tokens_used"],
                notetaker_metadata["cost"],
            )

            # Sprint 3: Generate and store embeddings for comments
            if db_comments:
                try:
                    embedding_service = get_embedding_service()

                    # Extract data for embedding service
                    comment_ids = [c.id for c in db_comments]
                    texts = [c.text for c in db_comments]
                    agent_names = [c.agent_name for c in db_comments]

                    # Store embeddings in Qdrant
                    embedding_service.store_comment_embeddings(
                        comment_ids=comment_ids,
                        texts=texts,
                        meeting_id=str(self.meeting_id),
                        round_num=round_num,
                        agent_names=agent_names,
                    )

                    logger.info(
                        "Stored embeddings for %d comments from %s in round %d",
                        len(db_comments),
                        agent_name,
                        round_num,
                    )

                except Exception as e:
                    # Non-fatal: embedding generation failure doesn't block workflow
                    logger.warning(
                        "Failed to generate embeddings for comments: %s",
                        str(e),
                        exc_info=True,
                    )

            # Update meeting metrics
            stmt = select(Meeting).where(Meeting.id == self.meeting_id)
            meeting_obj = storage_db.scalars(stmt).first()

            if meeting_obj:
                meeting_obj.total_comments += len(comments)
                meeting_obj.total_cost += metadata["cost"] + notetaker_metadata["cost"]
                meeting_obj.context_size = len(context) + len(response_text)
                meeting_obj.current_round = round_num
                storage_db.commit()

        # Return average novelty for this agent's comments
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        logger.info(
            "Agent %s completed round %d: %d comments, avg_novelty=%.3f",
            agent_name,
            round_num,
            len(comments),
            avg_novelty,
        )

        return avg_novelty
