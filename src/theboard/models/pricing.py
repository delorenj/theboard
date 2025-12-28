"""Model pricing configuration for cost tracking and hybrid model strategy.

Sprint 4 Story 13: Hybrid Model Strategy

Prices are in USD per million tokens (input + output combined for simplicity).
Based on OpenRouter pricing as of 2025-01.
"""

from typing import Literal

# Model pricing in USD per million tokens
# Simplified: single price for both input and output (typically output is 3x)
MODEL_PRICES = {
    # Budget tier (DeepSeek)
    "deepseek/deepseek-chat": 0.14,  # $0.14/MTok
    "deepseek/deepseek-r1": 0.55,  # $0.55/MTok (reasoning model)
    # Mid tier (Haiku, Sonnet 3.5)
    "anthropic/claude-3-haiku-20240307": 0.80,  # $0.25 input, $1.25 output avg
    "anthropic/claude-3-5-haiku-20241022": 0.80,  # Similar to Haiku
    "anthropic/claude-3-5-sonnet-20241022": 3.00,  # $3 input, $15 output avg
    # Premium tier (Opus, Sonnet 4)
    "anthropic/claude-sonnet-4-20250514": 3.00,  # Sonnet 4
    "anthropic/claude-opus-4-20241113": 15.00,  # $15 input, $75 output avg
}

# Model tier classification for promotion logic
ModelTier = Literal["budget", "mid", "premium"]

MODEL_TIERS: dict[str, ModelTier] = {
    "deepseek/deepseek-chat": "budget",
    "deepseek/deepseek-r1": "budget",
    "anthropic/claude-3-haiku-20240307": "mid",
    "anthropic/claude-3-5-haiku-20241022": "mid",
    "anthropic/claude-3-5-sonnet-20241022": "mid",
    "anthropic/claude-sonnet-4-20250514": "premium",
    "anthropic/claude-opus-4-20241113": "premium",
}


def get_model_price(model: str) -> float:
    """Get price per million tokens for a model.

    Args:
        model: Model identifier (e.g., "deepseek/deepseek-chat")

    Returns:
        Price in USD per million tokens

    Raises:
        ValueError: If model not found in pricing table
    """
    if model not in MODEL_PRICES:
        # Default to mid-tier pricing for unknown models
        return 3.0

    return MODEL_PRICES[model]


def calculate_cost(tokens_used: int, model: str) -> float:
    """Calculate cost in USD for token usage.

    Args:
        tokens_used: Total tokens (input + output)
        model: Model identifier

    Returns:
        Cost in USD (rounded to 4 decimal places)
    """
    price_per_million = get_model_price(model)
    cost = (tokens_used / 1_000_000) * price_per_million
    return round(cost, 4)


def get_model_tier(model: str) -> ModelTier:
    """Get tier classification for a model.

    Args:
        model: Model identifier

    Returns:
        Model tier: "budget", "mid", or "premium"
    """
    return MODEL_TIERS.get(model, "mid")


def get_promotion_model(current_model: str) -> str:
    """Get promoted model for high-performing agents.

    Promotion logic (Story 13):
    - Budget → Premium (DeepSeek → Opus)
    - Mid → Premium (Sonnet 3.5 → Sonnet 4 or Opus)
    - Premium → No promotion (already at top)

    Args:
        current_model: Current model being used

    Returns:
        Promoted model identifier
    """
    tier = get_model_tier(current_model)

    if tier == "budget":
        # Promote to Opus for best quality
        return "anthropic/claude-opus-4-20241113"
    elif tier == "mid":
        # Promote to Sonnet 4 (balance of cost and quality)
        return "anthropic/claude-sonnet-4-20250514"
    else:
        # Already premium, no promotion
        return current_model
