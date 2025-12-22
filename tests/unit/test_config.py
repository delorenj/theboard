"""Tests for configuration management."""

from theboard.config import get_settings, settings


def test_settings_singleton() -> None:
    """Test that settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_settings_has_required_fields() -> None:
    """Test that settings has all required configuration fields."""
    assert hasattr(settings, "database_url")
    assert hasattr(settings, "redis_url")
    assert hasattr(settings, "anthropic_api_key")
    assert hasattr(settings, "max_rounds")
    assert hasattr(settings, "convergence_threshold")


def test_settings_defaults() -> None:
    """Test that settings has reasonable defaults."""
    assert settings.max_rounds == 5
    assert settings.convergence_threshold == 0.2
    assert settings.default_strategy in ["sequential", "greedy"]
    assert settings.embedding_model == "all-MiniLM-L6-v2"
