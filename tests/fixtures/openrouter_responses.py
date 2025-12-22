"""OpenRouter API response fixtures for testing."""


def sample_api_response():
    """Sample API response with diverse models across all tiers.

    Returns a realistic OpenRouter API response containing models in:
    - Budget tier (<$1/MTok): DeepSeek Chat
    - Standard tier ($1-10/MTok): Claude Sonnet 4
    - Premium tier (>$10/MTok): Claude Opus 4.5

    Also includes invalid models to test filtering:
    - Missing pricing
    - Short context length
    - No chat modality

    Returns:
        dict: OpenRouter API response structure
    """
    return {
        "data": [
            # Budget tier (<$1/MTok output cost)
            {
                "id": "deepseek/deepseek-chat",
                "name": "DeepSeek Chat",
                "context_length": 32000,
                "pricing": {"prompt": "0.00000014", "completion": "0.00000028"},  # $0.28/MTok
                "architecture": {"modality": "text->text,chat"}
            },
            # Standard tier ($1-10/MTok output cost)
            {
                "id": "anthropic/claude-sonnet-4-20250514",
                "name": "Claude Sonnet 4",
                "context_length": 200000,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},  # $15/MTok (premium actually)
                "architecture": {"modality": "text->text,chat"}
            },
            {
                "id": "anthropic/claude-3-haiku-20240307",
                "name": "Claude 3 Haiku",
                "context_length": 200000,
                "pricing": {"prompt": "0.00000025", "completion": "0.00000125"},  # $1.25/MTok
                "architecture": {"modality": "text->text,chat"}
            },
            # Premium tier (>$10/MTok output cost)
            {
                "id": "anthropic/claude-opus-4-5-20251101",
                "name": "Claude Opus 4.5",
                "context_length": 200000,
                "pricing": {"prompt": "0.000015", "completion": "0.000075"},  # $75/MTok
                "architecture": {"modality": "text->text,chat"}
            },
            # Invalid: Missing pricing (should be filtered)
            {
                "id": "invalid/no-pricing",
                "name": "Invalid Model (No Pricing)",
                "context_length": 10000,
                "architecture": {"modality": "text->text,chat"}
            },
            # Invalid: Short context length (should be filtered)
            {
                "id": "invalid/short-context",
                "name": "Invalid Model (Short Context)",
                "context_length": 4000,  # Below MIN_CONTEXT_LENGTH (8000)
                "pricing": {"prompt": "0.001", "completion": "0.002"},
                "architecture": {"modality": "text->text,chat"}
            },
            # Invalid: No chat modality (should be filtered)
            {
                "id": "invalid/no-chat",
                "name": "Invalid Model (No Chat)",
                "context_length": 10000,
                "pricing": {"prompt": "0.001", "completion": "0.002"},
                "architecture": {"modality": "text->text"}  # Missing 'chat'
            },
            # Invalid: Zero pricing (should be filtered)
            {
                "id": "invalid/zero-pricing",
                "name": "Invalid Model (Zero Pricing)",
                "context_length": 10000,
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"modality": "text->text,chat"}
            },
        ]
    }


def empty_api_response():
    """Empty API response for testing edge cases.

    Returns:
        dict: OpenRouter API response with no models
    """
    return {"data": []}


def malformed_api_response():
    """Malformed API response for error handling tests.

    Returns:
        dict: Invalid response structure
    """
    return {"invalid": "structure"}


def budget_tier_models():
    """Only budget tier models for focused testing.

    Returns:
        dict: API response with only budget models
    """
    return {
        "data": [
            {
                "id": "deepseek/deepseek-chat",
                "name": "DeepSeek Chat",
                "context_length": 32000,
                "pricing": {"prompt": "0.00014", "completion": "0.00028"},
                "architecture": {"modality": "text->text,chat"}
            },
            {
                "id": "meta-llama/llama-3.1-8b-instruct",
                "name": "Meta Llama 3.1 8B",
                "context_length": 128000,
                "pricing": {"prompt": "0.00006", "completion": "0.00006"},
                "architecture": {"modality": "text->text,chat"}
            },
        ]
    }


def standard_tier_models():
    """Only standard tier models for focused testing.

    Returns:
        dict: API response with only standard models
    """
    return {
        "data": [
            {
                "id": "anthropic/claude-sonnet-4-20250514",
                "name": "Claude Sonnet 4",
                "context_length": 200000,
                "pricing": {"prompt": "0.003", "completion": "0.015"},
                "architecture": {"modality": "text->text,chat"}
            },
            {
                "id": "anthropic/claude-3-haiku-20240307",
                "name": "Claude 3 Haiku",
                "context_length": 200000,
                "pricing": {"prompt": "0.00025", "completion": "0.00125"},
                "architecture": {"modality": "text->text,chat"}
            },
        ]
    }


def premium_tier_models():
    """Only premium tier models for focused testing.

    Returns:
        dict: API response with only premium models
    """
    return {
        "data": [
            {
                "id": "anthropic/claude-opus-4-5-20251101",
                "name": "Claude Opus 4.5",
                "context_length": 200000,
                "pricing": {"prompt": "0.015", "completion": "0.075"},
                "architecture": {"modality": "text->text,chat"}
            },
            {
                "id": "openai/o1",
                "name": "OpenAI o1",
                "context_length": 200000,
                "pricing": {"prompt": "0.015", "completion": "0.060"},
                "architecture": {"modality": "text->text,chat"}
            },
        ]
    }
