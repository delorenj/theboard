"""Shared pytest fixtures for Phase 1 testing."""

import pytest
from datetime import datetime
from pathlib import Path


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary cache directory for OpenRouter service.

    Uses pytest's tmp_path fixture to create an isolated cache directory
    for each test.

    Returns:
        Path: Temporary cache directory
    """
    cache_dir = tmp_path / ".cache" / "theboard"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Temporary config directory for preferences.

    Uses pytest's tmp_path fixture to create an isolated config directory
    for each test.

    Returns:
        Path: Temporary config directory
    """
    config_dir = tmp_path / ".config" / "theboard"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables for predictable testing.

    Removes THEBOARD_DEFAULT_MODEL and OPENROUTER_API_KEY to ensure
    tests start with clean environment state.

    Args:
        monkeypatch: pytest fixture for patching
    """
    monkeypatch.delenv("THEBOARD_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def set_env(monkeypatch):
    """Helper to set environment variables in tests.

    Returns:
        Function to set environment variables
    """
    def _set_env(key: str, value: str):
        monkeypatch.setenv(key, value)
    return _set_env


@pytest.fixture
def frozen_time(mocker):
    """Freeze datetime.now() for cache expiration tests.

    Returns a fixed datetime and patches datetime.now() to return it.
    This ensures deterministic cache expiration testing.

    Args:
        mocker: pytest-mock fixture

    Returns:
        datetime: Frozen datetime instance
    """
    frozen = datetime(2025, 1, 1, 12, 0, 0)

    # Mock datetime in openrouter_service module
    mock_datetime = mocker.patch("theboard.services.openrouter_service.datetime")
    mock_datetime.now.return_value = frozen
    # Keep fromisoformat working for cache deserialization
    mock_datetime.fromisoformat = datetime.fromisoformat
    mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

    return frozen
