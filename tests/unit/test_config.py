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
    assert hasattr(settings, "openrouter_api_key")
    assert hasattr(settings, "max_rounds")
    assert hasattr(settings, "convergence_threshold")


def test_settings_defaults() -> None:
    """Test that settings has reasonable defaults."""
    assert settings.max_rounds == 5
    assert settings.convergence_threshold == 0.2
    assert settings.default_strategy in ["sequential", "greedy"]
    assert settings.embedding_model == "all-MiniLM-L6-v2"


def test_settings_credential_fallbacks(monkeypatch) -> None:
    """Test that credentials fallback to DEFAULT_USERNAME and DEFAULT_PASSWORD."""
    from theboard.config import Settings

    monkeypatch.setenv("DEFAULT_USERNAME", "fallback_user")
    monkeypatch.setenv("DEFAULT_PASSWORD", "fallback_pass")
    monkeypatch.setenv("DEFAULT_DATABASE", "fallback_db")

    # Clear specific env vars if they exist
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("RABBITMQ_USER", raising=False)
    monkeypatch.delenv("RABBITMQ_PASSWORD", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("RABBITMQ_URL", raising=False)

    new_settings = Settings(_env_file=None)

    assert new_settings.postgres_user == "fallback_user"
    assert new_settings.postgres_password == "fallback_pass"
    assert new_settings.postgres_db == "fallback_db"
    assert new_settings.redis_password == "fallback_pass"
    assert new_settings.rabbitmq_user == "fallback_user"
    assert new_settings.rabbitmq_password == "fallback_pass"

    assert "fallback_user:fallback_pass" in str(new_settings.database_url)
    assert "fallback_db" in str(new_settings.database_url)
    assert "fallback_pass" in str(new_settings.redis_url)
    assert "fallback_user:fallback_pass" in str(new_settings.rabbitmq_url)
