"""TOML configuration fixtures for preferences testing."""


def minimal_toml():
    """Minimal valid TOML configuration.

    Contains only the required fields with default values.

    Returns:
        str: TOML content
    """
    return """[models]
default = "deepseek/deepseek-chat"

[models.by_agent_type]
worker = "deepseek/deepseek-chat"
leader = "anthropic/claude-sonnet-4-20250514"
notetaker = "anthropic/claude-3-haiku-20240307"
compressor = "anthropic/claude-3-haiku-20240307"
"""


def toml_with_overrides():
    """TOML configuration with per-agent overrides.

    Includes custom model assignments for specific agent instances.

    Returns:
        str: TOML content
    """
    return """[models]
default = "deepseek/deepseek-chat"

[models.by_agent_type]
worker = "deepseek/deepseek-chat"
leader = "anthropic/claude-sonnet-4-20250514"
notetaker = "anthropic/claude-3-haiku-20240307"
compressor = "anthropic/claude-3-haiku-20240307"

[models.overrides]
test_agent = "anthropic/claude-opus-4-5-20251101"
special_worker = "openai/gpt-4"
debug_agent = "anthropic/claude-sonnet-4-20250514"
"""


def corrupted_toml():
    """Invalid TOML syntax for error handling tests.

    Missing closing bracket to trigger parse error.

    Returns:
        str: Invalid TOML content
    """
    return """[models
default = "invalid"
this is not valid TOML
"""


def empty_toml():
    """Empty TOML file for edge case testing.

    Returns:
        str: Empty string
    """
    return ""


def toml_with_custom_default():
    """TOML with non-standard default model.

    Returns:
        str: TOML content
    """
    return """[models]
default = "anthropic/claude-opus-4-5-20251101"

[models.by_agent_type]
worker = "anthropic/claude-sonnet-4-20250514"
leader = "anthropic/claude-opus-4-5-20251101"
notetaker = "anthropic/claude-3-haiku-20240307"
compressor = "anthropic/claude-3-haiku-20240307"
"""


def toml_missing_agent_types():
    """TOML with incomplete agent type definitions.

    Missing some agent type fields to test partial configuration.

    Returns:
        str: TOML content
    """
    return """[models]
default = "deepseek/deepseek-chat"

[models.by_agent_type]
worker = "deepseek/deepseek-chat"
leader = "anthropic/claude-sonnet-4-20250514"
"""
