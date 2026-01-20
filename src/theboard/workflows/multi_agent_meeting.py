"""Multi-agent meeting workflow for Sprint 2 & 4.

This workflow orchestrates multi-agent, multi-round brainstorming meetings with:
- Multiple agents per round (sequential or greedy/parallel turn-taking)
- Multi-round execution with context accumulation
- Basic convergence detection via novelty scores
- Session management following Sprint 1.5 patterns

Sprint 4 Story 11: Greedy Execution Strategy
- Parallel agent responses using asyncio.gather
- Comment-response phase (each agent responds to others)
- Token efficiency tracking (n² cost for greedy)
- Performance benchmarks (greedy vs sequential)
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select

from theboard.agents.domain_expert import DomainExpertAgent
from theboard.agents.notetaker import NotetakerAgent
from theboard.database import get_sync_db
from theboard.events import (
    ContextModifiedEvent,
    HumanInputNeededEvent,
    MeetingCompletedEvent,
    MeetingConvergedEvent,
    MeetingFailedEvent,
    MeetingPausedEvent,
    MeetingResumedEvent,
    MeetingStartedEvent,
    RoundCompletedEvent,
    TopComment,
    get_event_emitter,
)
from theboard.models.meeting import Agent, Comment as DBComment, Meeting, Response
from theboard.preferences import get_preferences_manager
from theboard.schemas import MeetingStatus, StrategyType

# Sprint 3: Import embedding service and compressor agent
from theboard.agents.compressor import CompressorAgent
from theboard.services.embedding_service import get_embedding_service

# Sprint 4 Story 12: Import Redis manager for pause state
from theboard.utils.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics for a single round execution (Sprint 4 Story 11).

    Used for tracking token efficiency and performance benchmarks.
    """

    round_num: int
    strategy: str
    execution_time_seconds: float
    agent_count: int
    total_responses: int
    total_comments: int
    total_tokens: int
    total_cost: float
    avg_novelty: float
    parallel_responses: int = 0  # For greedy: N parallel + N² comment-responses
    comment_response_count: int = 0  # N² responses in comment-response phase


@dataclass
class ExecutionBenchmark:
    """Performance benchmark comparing greedy vs sequential (Sprint 4 Story 11)."""

    strategy: str
    total_rounds: int
    total_execution_time_seconds: float
    total_tokens: int
    total_cost: float
    avg_tokens_per_round: float
    avg_time_per_round_seconds: float
    round_metrics: list[RoundMetrics] = field(default_factory=list)


class MultiAgentMeetingWorkflow:
    """Multi-agent, multi-round meeting workflow (Sprint 2 & 4).

    Key Features:
    - Multiple agents per meeting (selected pool)
    - Sequential strategy (agents take turns each round)
    - Greedy strategy (parallel execution with asyncio.gather) - Sprint 4 Story 11
    - Context accumulation across rounds
    - Basic convergence detection (novelty threshold)
    - Proper session management (Sprint 1.5 pattern)
    - Token efficiency tracking and benchmarks
    """

    def __init__(
        self,
        meeting_id: UUID,
        model_override: str | None = None,
        novelty_threshold: float = 0.3,
        min_rounds: int = 2,
        enable_compression: bool = True,
        compression_threshold: int = 10000,
        strategy: StrategyType = StrategyType.SEQUENTIAL,
        interactive: bool = False,
        human_input_timeout: int = 300,
    ) -> None:
        """Initialize multi-agent workflow.

        Args:
            meeting_id: Meeting UUID
            model_override: Optional CLI model override (--model flag)
            novelty_threshold: Convergence threshold for novelty scores (default 0.3)
            min_rounds: Minimum rounds before convergence can be detected (default 2)
            enable_compression: Enable comment compression (default True, Sprint 3)
            compression_threshold: Context size threshold for lazy compression (default 10000 chars, Sprint 5)
            strategy: Execution strategy - sequential or greedy (default sequential, Sprint 4 Story 11)
            interactive: Enable human-in-the-loop mode (default False, Sprint 4 Story 12)
            human_input_timeout: Seconds to wait for human input before auto-continue (default 300 = 5 min)
        """
        self.meeting_id = meeting_id
        self.model_override = model_override
        self.novelty_threshold = novelty_threshold
        self.min_rounds = min_rounds
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.compression_trigger_count = 0  # Track how many times compression was triggered
        self.strategy = strategy  # Sprint 4 Story 11: Execution strategy

        # Sprint 4 Story 12: Human-in-the-loop settings
        self.interactive = interactive
        self.human_input_timeout = human_input_timeout
        self.steering_context: str | None = None  # Human steering text (accumulated)

        # Sprint 4 Story 11: Token efficiency tracking
        self.round_metrics: list[RoundMetrics] = []
        self.execution_start_time: float | None = None

        # Sprint 5 Story 16: Delta propagation - track what each agent has seen
        self.agent_last_seen_round: dict[str, int] = {}  # agent_id -> last round they saw

        # Initialize event emitter (Sprint 2.5)
        self.emitter = get_event_emitter()

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

        logger.info(
            "MultiAgentMeetingWorkflow initialized: strategy=%s, novelty_threshold=%.2f, min_rounds=%d, interactive=%s",
            self.strategy.value,
            self.novelty_threshold,
            self.min_rounds,
            self.interactive,
        )

    def _extract_insights(self) -> tuple[list[TopComment], dict[str, int], dict[str, int]]:
        """Extract insights from meeting for completed event.

        Phase 3A: Extract top comments, category distribution, and agent participation.

        Returns:
            Tuple of (top_comments, category_distribution, agent_participation)
        """
        with get_sync_db() as db:
            # Query all comments for this meeting
            comments = db.scalars(
                select(DBComment)
                .where(DBComment.meeting_id == self.meeting_id)
                .where(DBComment.is_merged == False)  # Only non-merged comments
                .order_by(DBComment.novelty_score.desc())
            ).all()

            # Extract top 5 comments
            top_comments = [
                TopComment(
                    text=c.text,
                    category=c.category,
                    novelty_score=c.novelty_score,
                    agent_name=c.agent_name,
                    round_num=c.round
                )
                for c in comments[:5]
            ]

            # Calculate category distribution
            category_dist: dict[str, int] = {}
            for comment in comments:
                category_dist[comment.category] = category_dist.get(comment.category, 0) + 1

            # Calculate agent participation from responses
            responses = db.scalars(
                select(Response)
                .where(Response.meeting_id == self.meeting_id)
            ).all()

            agent_participation: dict[str, int] = {}
            for response in responses:
                agent_participation[response.agent_name] = agent_participation.get(response.agent_name, 0) + 1

            return top_comments, category_dist, agent_participation

    # =========================================================================
    # Sprint 4 Story 12: Human-in-the-Loop Methods
    # =========================================================================

    async def _human_input_checkpoint(
        self, round_num: int, avg_novelty: float, topic: str
    ) -> str:
        """Check for human input after round completion.

        Sprint 4 Story 12: Human-in-the-loop checkpoint.

        Emits human.input.needed event and waits for response via Redis.
        Auto-continues after timeout if no human input received.

        Args:
            round_num: Current round number
            avg_novelty: Average novelty score for this round
            topic: Meeting topic

        Returns:
            Action to take: "continue", "stop", "paused", or "modify_context"
        """
        redis = get_redis_manager()

        # Get current meeting state for event payload
        with get_sync_db() as db:
            stmt = select(Meeting).where(Meeting.id == self.meeting_id)
            meeting = db.scalars(stmt).first()
            total_comments = meeting.total_comments if meeting else 0

        # Emit human.input.needed event
        self.emitter.emit(
            HumanInputNeededEvent(
                meeting_id=self.meeting_id,
                round_num=round_num,
                reason="round_complete",
                current_topic=topic,
                total_comments=total_comments,
                avg_novelty=avg_novelty,
                timeout_seconds=self.human_input_timeout,
            )
        )

        logger.info(
            "Round %d complete. Waiting for human input (timeout=%ds)...",
            round_num,
            self.human_input_timeout,
        )

        # Poll for human input or timeout
        # In a real implementation, this would use async event waiting
        # For now, we use a polling approach with Redis state
        import asyncio

        poll_interval = 1.0  # Check every second
        elapsed = 0.0

        while elapsed < self.human_input_timeout:
            # Check if meeting was paused by external command
            pause_state = redis.get_pause_state(str(self.meeting_id))
            if pause_state:
                # Meeting was paused externally
                self._handle_pause(round_num, pause_state.get("steering_context"))
                return "paused"

            # Check for human action stored in Redis
            action_key = f"meeting:{self.meeting_id}:human_action"
            action_data = redis.client.get(action_key)
            if action_data:
                action = action_data.decode("utf-8") if isinstance(action_data, bytes) else action_data
                redis.client.delete(action_key)  # Clear action after reading

                if action == "stop":
                    return "stop"
                elif action == "pause":
                    self._handle_pause(round_num)
                    return "paused"
                elif action.startswith("modify_context:"):
                    # Extract steering text
                    steering = action[len("modify_context:"):]
                    self._apply_steering_context(round_num, steering)
                    return "continue"
                else:
                    return "continue"

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout: auto-continue
        logger.info(
            "Human input timeout (%ds) - auto-continuing meeting",
            self.human_input_timeout,
        )
        return "continue"

    def _handle_pause(self, round_num: int, steering_context: str | None = None) -> None:
        """Handle meeting pause request.

        Sprint 4 Story 12: Pause meeting and update state.

        Args:
            round_num: Round number when paused
            steering_context: Optional steering text from human
        """
        redis = get_redis_manager()

        # Set pause state in Redis
        redis.set_pause_state(
            str(self.meeting_id),
            round_num,
            steering_context,
            self.human_input_timeout,
        )

        # Update meeting status in database
        with get_sync_db() as db:
            stmt = select(Meeting).where(Meeting.id == self.meeting_id)
            meeting = db.scalars(stmt).first()
            if meeting:
                meeting.status = MeetingStatus.PAUSED.value
                meeting.current_round = round_num
                db.commit()

        # Emit paused event
        self.emitter.emit(
            MeetingPausedEvent(
                meeting_id=self.meeting_id,
                round_num=round_num,
                paused_by="user",
                reason=None,
            )
        )

        logger.info("Meeting %s paused at round %d", self.meeting_id, round_num)

    def _apply_steering_context(self, round_num: int, steering_text: str) -> None:
        """Apply human steering context to meeting.

        Sprint 4 Story 12: Add steering context for next rounds.

        Args:
            round_num: Current round number
            steering_text: Human steering text to incorporate
        """
        # Accumulate steering context
        if self.steering_context:
            self.steering_context = f"{self.steering_context}\n\n[Round {round_num} Steering]: {steering_text}"
        else:
            self.steering_context = f"[Round {round_num} Steering]: {steering_text}"

        # Emit context modified event
        self.emitter.emit(
            ContextModifiedEvent(
                meeting_id=self.meeting_id,
                round_num=round_num,
                modification_type="add_constraint",
                steering_text=steering_text,
            )
        )

        logger.info(
            "Applied steering context at round %d: %s",
            round_num,
            steering_text[:100] + "..." if len(steering_text) > 100 else steering_text,
        )

    def get_steering_context(self) -> str | None:
        """Get accumulated steering context for agents.

        Sprint 4 Story 12: Return steering text to include in agent prompts.

        Returns:
            Accumulated steering context or None
        """
        return self.steering_context

    async def resume_from_pause(self) -> None:
        """Resume a paused meeting.

        Sprint 4 Story 12: Resume meeting from paused state.

        Retrieves pause state from Redis and continues execution.
        """
        redis = get_redis_manager()
        pause_state = redis.get_pause_state(str(self.meeting_id))

        if not pause_state:
            raise ValueError(f"Meeting {self.meeting_id} is not paused")

        round_num = pause_state.get("round_num", 0)
        steering = pause_state.get("steering_context")

        # Apply any pending steering context
        if steering:
            self._apply_steering_context(round_num, steering)

        # Clear pause state
        redis.clear_pause_state(str(self.meeting_id))

        # Update meeting status
        with get_sync_db() as db:
            stmt = select(Meeting).where(Meeting.id == self.meeting_id)
            meeting = db.scalars(stmt).first()
            if meeting:
                meeting.status = MeetingStatus.RUNNING.value
                db.commit()

        # Emit resumed event
        self.emitter.emit(
            MeetingResumedEvent(
                meeting_id=self.meeting_id,
                round_num=round_num,
                resumed_by="user",
                context_modified=steering is not None,
                steering_context=steering,
            )
        )

        logger.info(
            "Meeting %s resumed at round %d (steering=%s)",
            self.meeting_id,
            round_num,
            "yes" if steering else "no",
        )

        # Continue execution from where we left off
        await self.execute()

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
            self.execution_start_time = time.time()

            for round_num in range(1, meeting.max_rounds + 1):
                logger.info(
                    "Starting round %d of %d (strategy=%s)",
                    round_num,
                    meeting.max_rounds,
                    self.strategy.value,
                )

                # Sprint 4 Story 11: Execute round based on strategy
                round_start_time = time.time()

                if self.strategy == StrategyType.GREEDY:
                    # Greedy: parallel execution with asyncio.gather
                    avg_novelty, round_metrics = await self._execute_round_greedy(agents, round_num)
                else:
                    # Sequential: agents take turns one by one
                    avg_novelty = await self._execute_round(agents, round_num)
                    # Create metrics for sequential execution
                    round_metrics = RoundMetrics(
                        round_num=round_num,
                        strategy=self.strategy.value,
                        execution_time_seconds=time.time() - round_start_time,
                        agent_count=len(agents),
                        total_responses=len(agents),
                        total_comments=0,  # Will be calculated below
                        total_tokens=0,
                        total_cost=0.0,
                        avg_novelty=avg_novelty,
                    )

                self.round_metrics.append(round_metrics)

                # Sprint 5 Story 16: Lazy compression - only compress when context > threshold
                if self.enable_compression and self.compressor:
                    # Check current context size to decide if compression is needed
                    with get_sync_db() as check_db:
                        check_stmt = select(Meeting).where(Meeting.id == self.meeting_id)
                        check_meeting = check_db.scalars(check_stmt).first()
                        current_context_size = check_meeting.context_size if check_meeting else 0

                    # Only compress if context exceeds threshold (lazy compression)
                    if current_context_size > self.compression_threshold:
                        self.compression_trigger_count += 1
                        try:
                            compression_metrics = self.compressor.compress_comments(
                                meeting_id=self.meeting_id,
                                round_num=round_num,
                            )
                            logger.info(
                                "Round %d lazy compression triggered (context=%d > threshold=%d): %d → %d comments (%.1f%% reduction)",
                                round_num,
                                current_context_size,
                                self.compression_threshold,
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
                    else:
                        logger.debug(
                            "Round %d: skipping compression (context=%d < threshold=%d)",
                            round_num,
                            current_context_size,
                            self.compression_threshold,
                        )

                # Sprint 4 Story 12: Human-in-the-loop checkpoint
                if self.interactive:
                    action = await self._human_input_checkpoint(round_num, avg_novelty, meeting.topic)
                    if action == "stop":
                        logger.info("Meeting stopped by human at round %d", round_num)
                        # Update meeting with user-stopped reason
                        with get_sync_db() as stop_db:
                            stop_stmt = select(Meeting).where(Meeting.id == self.meeting_id)
                            stop_meeting = stop_db.scalars(stop_stmt).first()
                            if stop_meeting:
                                stop_meeting.status = MeetingStatus.COMPLETED.value
                                stop_meeting.current_round = round_num
                                stop_meeting.stopping_reason = "Stopped by user"
                                stop_db.commit()
                        break
                    elif action == "paused":
                        # Meeting is paused, exit loop but don't complete
                        logger.info("Meeting paused at round %d", round_num)
                        return  # Exit execute() - meeting remains in PAUSED state

                # Check convergence (only after minimum rounds)
                if round_num >= self.min_rounds and avg_novelty < self.novelty_threshold:
                    logger.info(
                        "Convergence detected at round %d (novelty=%.3f < threshold=%.3f, min_rounds=%d satisfied)",
                        round_num,
                        avg_novelty,
                        self.novelty_threshold,
                        self.min_rounds,
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
                elif round_num < self.min_rounds:
                    logger.debug(
                        "Round %d: convergence check skipped (min_rounds=%d not reached, current novelty=%.3f)",
                        round_num,
                        self.min_rounds,
                        avg_novelty,
                    )

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

                    # Extract insights for event payload (Phase 3A)
                    top_comments, category_dist, agent_participation = self._extract_insights()

                    # Emit meeting completed event (Sprint 2.5, enhanced in Phase 3A)
                    self.emitter.emit(
                        MeetingCompletedEvent(
                            meeting_id=self.meeting_id,
                            total_rounds=round_num,
                            total_comments=final_meeting.total_comments,
                            total_cost=final_meeting.total_cost,
                            convergence_detected=converged,
                            stopping_reason=final_meeting.stopping_reason,
                            top_comments=top_comments,
                            category_distribution=category_dist,
                            agent_participation=agent_participation,
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

            # Tokenize agent text using same approach as topic tokenization
            # to ensure exact word matches and avoid false positives
            agent_words = set(re.findall(r'\b\w+\b', agent_text))
            
            # Count keyword matches using word boundary matching
            matches = sum(1 for keyword in keywords if keyword in agent_words)

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

        # Execute each agent sequentially
        round_novelty_scores = []

        for agent in agents:
            try:
                # Sprint 5 Story 16: Build delta context for this specific agent
                # This reduces token usage by only sending new comments since their last turn
                agent_context = await self._build_context(round_num, agent_id=str(agent.id))

                # Execute agent response and extract comments
                # (NO session held during LLM calls - Sprint 1.5 pattern)
                agent_novelty = await self._execute_agent_turn(
                    agent, agent_context, round_num
                )
                round_novelty_scores.append(agent_novelty)

                # Update agent's last seen round for delta propagation
                self.agent_last_seen_round[str(agent.id)] = round_num

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

    async def _build_context(self, current_round: int, agent_id: str | None = None) -> str:
        """Build cumulative context from previous rounds with optional delta propagation.

        Context Structure:
        - Round 1: Topic only
        - Round 2+: Topic + comments (all or delta based on agent tracking)

        Sprint 5 Story 16: Delta propagation - if agent_id provided, only include
        comments since agent's last seen round to reduce token usage.

        Formula:
        - Full: Context_r = Topic + Σ(Comments from rounds 1 to r-1)
        - Delta: Context_r = Topic + Σ(Comments from rounds last_seen+1 to r-1)

        Args:
            current_round: Current round number
            agent_id: Optional agent ID for delta propagation (Sprint 5)

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
                # Sprint 5: Determine starting round for delta propagation
                if agent_id and agent_id in self.agent_last_seen_round:
                    start_round = self.agent_last_seen_round[agent_id] + 1
                    is_delta = True
                else:
                    start_round = 1
                    is_delta = False

                comment_stmt = (
                    select(DBComment)
                    .where(DBComment.meeting_id == self.meeting_id)
                    .where(DBComment.round >= start_round)
                    .where(DBComment.round < current_round)
                    .order_by(DBComment.round, DBComment.created_at)
                )
                prev_comments = db.scalars(comment_stmt).all()

                if prev_comments:
                    if is_delta:
                        context_parts.append(f"\nNew Comments (since round {start_round}):\n")
                    else:
                        context_parts.append("\nPrevious Discussion:\n")

                    for comment in prev_comments:
                        context_parts.append(
                            f"[Round {comment.round}, {comment.agent_name}] "
                            f"{comment.category.upper()}: {comment.text}\n"
                        )

            context = "".join(context_parts)
            logger.debug(
                "Built context for round %d%s: %d chars",
                current_round,
                f" (delta for agent {agent_id})" if agent_id and agent_id in self.agent_last_seen_round else "",
                len(context),
            )

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

        # Get model for this agent using preferences
        prefs = get_preferences_manager()
        model_to_use = prefs.get_model_for_agent(
            agent_name=agent.name,
            agent_type="domain_expert",
            cli_override=self.model_override,
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

    async def _execute_round_greedy(
        self, agents: list[Agent], round_num: int
    ) -> tuple[float, RoundMetrics]:
        """Execute a single round with all agents in parallel using greedy strategy.

        Sprint 4 Story 11: Greedy Execution Strategy

        Greedy strategy phases:
        1. Parallel Response Phase: All agents respond simultaneously to context
           - Uses asyncio.gather for parallel LLM calls
           - Significantly faster than sequential (N parallel vs N sequential)
        2. Comment-Response Phase: Each agent responds to other agents' comments
           - Creates N² interactions (each agent responds to each other agent's comments)
           - Higher token cost but richer discussion

        Session Management (Sprint 1.5 pattern):
        - Build context (read-only session for previous responses)
        - Execute all LLM calls WITHOUT holding session (parallel)
        - Store results in new session

        Args:
            agents: List of Agent instances for this round
            round_num: Current round number

        Returns:
            Tuple of (average_novelty_score, round_metrics)

        Raises:
            RuntimeError: If round execution fails
        """
        round_start_time = time.time()
        logger.info(
            "Executing GREEDY round %d with %d agents (parallel)",
            round_num,
            len(agents),
        )

        # Build shared context for all agents (before parallel execution)
        shared_context = await self._build_context(round_num, agent_id=None)

        # Phase 1: Parallel Response Phase
        # All agents respond to the same context simultaneously
        parallel_start_time = time.time()

        async def execute_agent_parallel(agent: Agent) -> tuple[Agent, float, dict]:
            """Execute a single agent's turn and return results."""
            try:
                novelty = await self._execute_agent_turn(agent, shared_context, round_num)
                # Get metrics from database
                with get_sync_db() as db:
                    response_stmt = (
                        select(Response)
                        .where(Response.meeting_id == self.meeting_id)
                        .where(Response.round == round_num)
                        .where(Response.agent_name == agent.name)
                    )
                    response = db.scalars(response_stmt).first()
                    metrics = {
                        "tokens_used": response.tokens_used if response else 0,
                        "cost": response.cost if response else 0.0,
                    }
                return (agent, novelty, metrics)
            except Exception as e:
                logger.error("Agent %s failed in greedy round %d: %s", agent.name, round_num, e)
                return (agent, 0.0, {"tokens_used": 0, "cost": 0.0})

        # Execute all agents in parallel using asyncio.gather
        parallel_results = await asyncio.gather(
            *[execute_agent_parallel(agent) for agent in agents],
            return_exceptions=True,
        )

        parallel_execution_time = time.time() - parallel_start_time

        # Process results from parallel phase
        phase1_novelty_scores = []
        phase1_total_tokens = 0
        phase1_total_cost = 0.0
        successful_agents = []

        for result in parallel_results:
            if isinstance(result, Exception):
                logger.error("Agent execution raised exception: %s", result)
                continue

            agent, novelty, metrics = result
            phase1_novelty_scores.append(novelty)
            phase1_total_tokens += metrics["tokens_used"]
            phase1_total_cost += metrics["cost"]
            successful_agents.append(agent)

            # Update agent's last seen round for delta propagation
            self.agent_last_seen_round[str(agent.id)] = round_num

        logger.info(
            "Greedy round %d parallel phase completed: %d/%d agents, %.1fs, %d tokens, $%.4f",
            round_num,
            len(successful_agents),
            len(agents),
            parallel_execution_time,
            phase1_total_tokens,
            phase1_total_cost,
        )

        # Phase 2: Comment-Response Phase (N² interactions)
        # Each agent responds to other agents' comments from this round
        comment_response_start_time = time.time()
        phase2_novelty_scores = []
        phase2_total_tokens = 0
        phase2_total_cost = 0.0
        comment_response_count = 0

        # Get all comments from this round for the comment-response phase
        with get_sync_db() as db:
            comment_stmt = (
                select(DBComment)
                .where(DBComment.meeting_id == self.meeting_id)
                .where(DBComment.round == round_num)
                .order_by(DBComment.created_at)
            )
            round_comments = db.scalars(comment_stmt).all()

        if round_comments and len(successful_agents) > 1:
            # Build comment-response context for each agent pair
            async def execute_comment_response(
                responding_agent: Agent, target_agent_name: str, target_comments: list[DBComment]
            ) -> tuple[str, float, dict]:
                """Have responding_agent respond to target_agent's comments."""
                # Build context with target agent's comments
                comment_context = f"Topic: {shared_context.split(chr(10))[0]}\n\n"
                comment_context += f"Comments from {target_agent_name} to respond to:\n"
                for comment in target_comments:
                    comment_context += f"- [{comment.category.upper()}] {comment.text}\n"
                comment_context += "\nProvide your thoughts and responses to these comments."

                try:
                    novelty = await self._execute_agent_turn(
                        responding_agent, comment_context, round_num
                    )
                    # Get metrics
                    with get_sync_db() as db:
                        response_stmt = (
                            select(Response)
                            .where(Response.meeting_id == self.meeting_id)
                            .where(Response.round == round_num)
                            .where(Response.agent_name == responding_agent.name)
                            .order_by(Response.created_at.desc())
                        )
                        response = db.scalars(response_stmt).first()
                        metrics = {
                            "tokens_used": response.tokens_used if response else 0,
                            "cost": response.cost if response else 0.0,
                        }
                    return (responding_agent.name, novelty, metrics)
                except Exception as e:
                    logger.error(
                        "Comment-response failed: %s responding to %s: %s",
                        responding_agent.name,
                        target_agent_name,
                        e,
                    )
                    return (responding_agent.name, 0.0, {"tokens_used": 0, "cost": 0.0})

            # Create comment-response tasks (N² - N, excluding self-responses)
            comment_response_tasks = []
            agent_comments_map: dict[str, list[DBComment]] = {}
            for comment in round_comments:
                if comment.agent_name not in agent_comments_map:
                    agent_comments_map[comment.agent_name] = []
                agent_comments_map[comment.agent_name].append(comment)

            for responding_agent in successful_agents:
                for target_agent_name, target_comments in agent_comments_map.items():
                    # Skip self-responses
                    if responding_agent.name != target_agent_name:
                        comment_response_tasks.append(
                            execute_comment_response(
                                responding_agent, target_agent_name, target_comments
                            )
                        )

            if comment_response_tasks:
                # Execute comment-responses in parallel
                comment_results = await asyncio.gather(
                    *comment_response_tasks, return_exceptions=True
                )

                for result in comment_results:
                    if isinstance(result, Exception):
                        logger.error("Comment-response raised exception: %s", result)
                        continue

                    responder_name, novelty, metrics = result
                    phase2_novelty_scores.append(novelty)
                    phase2_total_tokens += metrics["tokens_used"]
                    phase2_total_cost += metrics["cost"]
                    comment_response_count += 1

            comment_response_time = time.time() - comment_response_start_time
            logger.info(
                "Greedy round %d comment-response phase completed: %d interactions, %.1fs, %d tokens, $%.4f",
                round_num,
                comment_response_count,
                comment_response_time,
                phase2_total_tokens,
                phase2_total_cost,
            )

        # Calculate overall metrics
        all_novelty_scores = phase1_novelty_scores + phase2_novelty_scores
        avg_novelty = sum(all_novelty_scores) / len(all_novelty_scores) if all_novelty_scores else 1.0
        total_execution_time = time.time() - round_start_time

        # Get comment count for this round
        with get_sync_db() as db:
            comment_count_stmt = (
                select(DBComment)
                .where(DBComment.meeting_id == self.meeting_id)
                .where(DBComment.round == round_num)
            )
            total_comments = len(db.scalars(comment_count_stmt).all())

        round_metrics = RoundMetrics(
            round_num=round_num,
            strategy="greedy",
            execution_time_seconds=total_execution_time,
            agent_count=len(agents),
            total_responses=len(successful_agents) + comment_response_count,
            total_comments=total_comments,
            total_tokens=phase1_total_tokens + phase2_total_tokens,
            total_cost=phase1_total_cost + phase2_total_cost,
            avg_novelty=avg_novelty,
            parallel_responses=len(successful_agents),
            comment_response_count=comment_response_count,
        )

        logger.info(
            "Greedy round %d completed: %d agents, %d comment-responses, "
            "%.1fs total, %d tokens, $%.4f, avg_novelty=%.3f",
            round_num,
            len(successful_agents),
            comment_response_count,
            total_execution_time,
            round_metrics.total_tokens,
            round_metrics.total_cost,
            avg_novelty,
        )

        # Emit round completed events
        with get_sync_db() as round_db:
            response_stmt = (
                select(Response)
                .where(Response.meeting_id == self.meeting_id)
                .where(Response.round == round_num)
            )
            round_responses = round_db.scalars(response_stmt).all()

            comment_stmt = (
                select(DBComment)
                .where(DBComment.meeting_id == self.meeting_id)
                .where(DBComment.round == round_num)
            )
            round_comments = round_db.scalars(comment_stmt).all()

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

        return avg_novelty, round_metrics

    def get_execution_benchmark(self) -> ExecutionBenchmark:
        """Get execution benchmark data for this workflow run.

        Sprint 4 Story 11: Performance benchmark for greedy vs sequential comparison.

        Returns:
            ExecutionBenchmark with aggregated metrics across all rounds
        """
        if not self.round_metrics:
            return ExecutionBenchmark(
                strategy=self.strategy.value,
                total_rounds=0,
                total_execution_time_seconds=0.0,
                total_tokens=0,
                total_cost=0.0,
                avg_tokens_per_round=0.0,
                avg_time_per_round_seconds=0.0,
            )

        total_tokens = sum(m.total_tokens for m in self.round_metrics)
        total_cost = sum(m.total_cost for m in self.round_metrics)
        total_time = sum(m.execution_time_seconds for m in self.round_metrics)
        num_rounds = len(self.round_metrics)

        return ExecutionBenchmark(
            strategy=self.strategy.value,
            total_rounds=num_rounds,
            total_execution_time_seconds=total_time,
            total_tokens=total_tokens,
            total_cost=total_cost,
            avg_tokens_per_round=total_tokens / num_rounds if num_rounds > 0 else 0.0,
            avg_time_per_round_seconds=total_time / num_rounds if num_rounds > 0 else 0.0,
            round_metrics=self.round_metrics,
        )
