"""Cost estimation service for meeting configuration."""

import logging

from theboard.schemas import AgentPoolSize, CostEstimate, LengthStrategy, WizardConfig

logger = logging.getLogger(__name__)

# Model pricing (per 1M tokens)
MODEL_PRICING = {
    "claude-sonnet-3.5": 3.00,
    "claude-sonnet-4": 3.00,
    "deepseek": 0.14,  # DeepSeek pricing
    "gpt-4o-mini": 0.15,
}


def estimate_meeting_cost(config: WizardConfig) -> CostEstimate:
    """Estimate cost and duration for a meeting configuration.

    Args:
        config: Wizard configuration with topic, meeting_type, length_strategy, agent_pool_size

    Returns:
        CostEstimate with min/max/expected costs and durations

    Algorithm:
        1. Estimate tokens per response based on topic complexity (~1500-2500 tokens)
        2. Calculate total tokens = tokens_per_response × agent_count × expected_rounds
        3. Apply model pricing (default to deepseek)
        4. Factor in convergence likelihood (quick exits early, thorough runs full)
        5. Add notetaker overhead (~500 tokens per agent response)
    """
    # Length strategy defines rounds and convergence behavior
    length_params = {
        LengthStrategy.QUICK: {
            "max_rounds": 2,
            "convergence_factor": 0.9,  # Likely converges early
            "minutes_per_round": 2.5,
        },
        LengthStrategy.STANDARD: {
            "max_rounds": 4,
            "convergence_factor": 0.75,  # Usually 3 rounds
            "minutes_per_round": 3.0,
        },
        LengthStrategy.THOROUGH: {
            "max_rounds": 5,
            "convergence_factor": 1.0,  # Often hits max
            "minutes_per_round": 3.5,
        },
    }

    # Agent pool size mapping
    pool_sizes = {
        AgentPoolSize.SMALL: 3,
        AgentPoolSize.MEDIUM: 5,
        AgentPoolSize.LARGE: 8,
    }

    params = length_params[config.length_strategy]
    agent_count = pool_sizes[config.agent_pool_size]

    # Estimate tokens per response based on topic length (proxy for complexity)
    topic_length = len(config.topic)
    if topic_length < 50:
        base_tokens_per_response = 1500
    elif topic_length < 100:
        base_tokens_per_response = 1800
    else:
        base_tokens_per_response = 2200

    # Notetaker adds ~500 tokens per agent response
    notetaker_tokens_per_response = 500

    # Calculate expected rounds (factoring in convergence)
    max_rounds = params["max_rounds"]
    expected_rounds = max_rounds * params["convergence_factor"]

    # Total tokens calculation
    # Each agent × rounds × (response tokens + notetaker tokens)
    tokens_per_round = agent_count * (base_tokens_per_response + notetaker_tokens_per_response)
    expected_total_tokens = tokens_per_round * expected_rounds
    min_total_tokens = tokens_per_round * (expected_rounds * 0.7)  # Best case (early convergence)
    max_total_tokens = tokens_per_round * max_rounds * 1.2  # Worst case (no convergence + retries)

    # Model pricing (default to deepseek)
    model_cost_per_million = MODEL_PRICING["deepseek"]

    # Calculate costs
    expected_cost = (expected_total_tokens / 1_000_000) * model_cost_per_million
    min_cost = (min_total_tokens / 1_000_000) * model_cost_per_million
    max_cost = (max_total_tokens / 1_000_000) * model_cost_per_million

    # Duration estimates
    minutes_per_round = params["minutes_per_round"]
    expected_duration = int(expected_rounds * minutes_per_round)
    min_duration = int(expected_rounds * 0.7 * minutes_per_round)
    max_duration = int(max_rounds * minutes_per_round * 1.3)

    # Comment count estimate (2-4 comments per agent per round)
    comments_per_agent_per_round = 3
    expected_comments = int(agent_count * expected_rounds * comments_per_agent_per_round)

    # Build breakdown for display
    breakdown = {
        "agents": agent_count,
        "expected_rounds": round(expected_rounds, 1),
        "max_rounds": max_rounds,
        "tokens_per_round": int(tokens_per_round),
        "total_tokens": int(expected_total_tokens),
        "model": "deepseek",
        "model_rate_per_1m_tokens": model_cost_per_million,
        "convergence_likelihood": f"{params['convergence_factor'] * 100:.0f}%",
    }

    logger.info(
        "Cost estimate for %s strategy with %d agents: $%.2f (expected)",
        config.length_strategy.value,
        agent_count,
        expected_cost,
    )

    return CostEstimate(
        min_cost=round(min_cost, 2),
        max_cost=round(max_cost, 2),
        expected_cost=round(expected_cost, 2),
        min_duration_minutes=min_duration,
        max_duration_minutes=max_duration,
        expected_duration_minutes=expected_duration,
        expected_comment_count=expected_comments,
        breakdown=breakdown,
    )
