"""Multi-agent meeting workflow for Sprint 2.

This workflow orchestrates multi-agent, multi-round brainstorming meetings with:
- Multiple agents per round (sequential turn-taking)
- Multi-round execution with context accumulation
- Basic convergence detection via novelty scores
- Session management following Sprint 1.5 patterns
"""

import logging
import re
from uuid import UUID

from sqlalchemy import select

from theboard.agents.domain_expert import DomainExpertAgent
from theboard.agents.notetaker import NotetakerAgent
from theboard.database import get_sync_db
from theboard.events import (
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
        compression_threshold: int = 10000,
    ) -> None:
        """Initialize multi-agent workflow.

        Args:
            meeting_id: Meeting UUID
            model_override: Optional CLI model override (--model flag)
            novelty_threshold: Convergence threshold for novelty scores (default 0.3)
            enable_compression: Enable comment compression (default True, Sprint 3)
            compression_threshold: Context size threshold for lazy compression (default 10000 chars, Sprint 5)
        """
        self.meeting_id = meeting_id
        self.model_override = model_override
        self.novelty_threshold = novelty_threshold
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.compression_trigger_count = 0  # Track how many times compression was triggered

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

                # Execute round with all agents sequentially
                avg_novelty = await self._execute_round(agents, round_num)

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
