"""Unit tests for PreferencesManager.

Priority breakdown:
- P0 (Critical): Precedence hierarchy, TOML persistence
- P1 (High): Validation, setters
- P2 (Medium): Singleton, edge cases
"""

import pytest
from pathlib import Path

from theboard.preferences import (
    PreferencesManager,
    Preferences,
    ModelPreferences,
    ModelsByAgentType,
    get_preferences_manager,
)
from tests.fixtures.toml_configs import (
    minimal_toml,
    toml_with_overrides,
    corrupted_toml,
    empty_toml,
    toml_with_custom_default,
)


class TestPreferencesPersistence:
    """Test TOML file persistence (P0 Critical)."""

    def test_load_creates_default_if_missing(self, tmp_config_dir, clean_env):
        """Test default preferences creation when file doesn't exist.

        Verifies that PreferencesManager automatically creates a default
        preferences file when none exists.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        # Load should create default file
        preferences = manager.load()

        assert config_file.exists()
        assert preferences.models.default == "deepseek/deepseek-chat"
        assert preferences.models.by_agent_type.worker == "deepseek/deepseek-chat"

    def test_load_valid_toml(self, tmp_config_dir, clean_env):
        """Test loading valid TOML file.

        Verifies that PreferencesManager correctly parses a valid TOML file
        and returns the expected Preferences object.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(minimal_toml())

        manager = PreferencesManager(config_path=config_file)
        preferences = manager.load()

        assert preferences.models.default == "deepseek/deepseek-chat"
        assert preferences.models.by_agent_type.worker == "deepseek/deepseek-chat"
        assert preferences.models.by_agent_type.leader == "anthropic/claude-sonnet-4-20250514"

    def test_load_corrupted_toml_falls_back(self, tmp_config_dir, clean_env):
        """Test corrupted TOML falls back to defaults.

        Verifies that PreferencesManager handles corrupted TOML files gracefully
        by falling back to default preferences.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(corrupted_toml())

        manager = PreferencesManager(config_path=config_file)
        preferences = manager.load()

        # Should fall back to defaults
        assert preferences.models.default == "deepseek/deepseek-chat"

    def test_save_toml_round_trip(self, tmp_config_dir, clean_env):
        """Test save + load preserves data (P0 CRITICAL).

        This is the most critical test for TOML persistence. Verifies that:
        1. Saving preferences writes valid TOML
        2. Loading the saved file returns identical data
        3. No data loss or corruption occurs
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        # Create preferences with custom data
        original = Preferences(
            models=ModelPreferences(
                default="anthropic/claude-opus-4-5-20251101",
                by_agent_type=ModelsByAgentType(
                    worker="deepseek/deepseek-chat",
                    leader="anthropic/claude-sonnet-4-20250514",
                    notetaker="anthropic/claude-3-haiku-20240307",
                    compressor="anthropic/claude-3-haiku-20240307",
                ),
                overrides={"test_agent": "openai/gpt-4"},
            )
        )

        # Save and reload
        manager.save(original)
        loaded = manager.load()

        # Verify exact match
        assert loaded.models.default == original.models.default
        assert loaded.models.by_agent_type.worker == original.models.by_agent_type.worker
        assert loaded.models.by_agent_type.leader == original.models.by_agent_type.leader
        assert loaded.models.overrides == original.models.overrides

    def test_save_creates_directory(self, tmp_path, clean_env):
        """Test config directory creation during init.

        Verifies that PreferencesManager.__init__ creates the config directory.
        Note: This creates CONFIG_DIR, not custom config_path directories.
        """
        # Create a custom path but PreferencesManager will create CONFIG_DIR
        config_file = tmp_path / "custom" / "preferences.toml"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        manager = PreferencesManager(config_path=config_file)
        preferences = Preferences()

        manager.save(preferences)

        assert config_file.exists()
        # Also verify CONFIG_DIR was created (even though we're using custom path)
        assert PreferencesManager.CONFIG_DIR.exists()


class TestPrecedenceHierarchy:
    """Test 6-level precedence hierarchy (P0 CRITICAL).

    Precedence order (highest to lowest):
    1. CLI flag override (--model)
    2. Environment variable (THEBOARD_DEFAULT_MODEL)
    3. Per-agent override in preferences.toml
    4. Agent type default in preferences.toml
    5. Global default in preferences.toml
    6. Hardcoded fallback
    """

    def test_precedence_level_1_cli_override(self, tmp_config_dir, clean_env):
        """Test CLI flag has highest precedence (Level 1).

        When CLI override is provided, it should be used regardless
        of any other configuration.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        manager = PreferencesManager(config_path=config_file)
        cli_model = "anthropic/claude-opus-4-5-20251101"

        # CLI override should win
        model = manager.get_model_for_agent(
            agent_name="test_worker",
            agent_type="worker",
            cli_override=cli_model,
        )

        assert model == cli_model

    def test_precedence_level_2_env_var(self, tmp_config_dir, monkeypatch):
        """Test environment variable precedence (Level 2).

        When no CLI override is provided, environment variable should be used.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        env_model = "anthropic/claude-sonnet-4-20250514"
        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", env_model)

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="test_worker",
            agent_type="worker",
            cli_override=None,
        )

        assert model == env_model

    def test_precedence_level_3_per_agent_override(self, tmp_config_dir, clean_env):
        """Test per-agent override in config (Level 3).

        When agent name matches an override, that model should be used.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="test_agent",  # Has override
            agent_type="worker",
            cli_override=None,
        )

        assert model == "anthropic/claude-opus-4-5-20251101"

    def test_precedence_level_4_agent_type_default(self, tmp_config_dir, clean_env):
        """Test agent type default (Level 4).

        When no override exists, agent type default should be used.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(minimal_toml())

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="some_worker",
            agent_type="worker",
            cli_override=None,
        )

        assert model == "deepseek/deepseek-chat"

    def test_precedence_level_5_global_default(self, tmp_config_dir, clean_env):
        """Test global default in config (Level 5).

        When agent type doesn't exist, global default should be used.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(minimal_toml())

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="some_agent",
            agent_type="unknown_type",  # Not in by_agent_type
            cli_override=None,
        )

        assert model == "deepseek/deepseek-chat"

    def test_precedence_level_6_hardcoded_fallback(self, tmp_config_dir, clean_env):
        """Test hardcoded fallback (Level 6).

        When config file is empty/invalid, hardcoded defaults should be used.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(empty_toml())

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="some_agent",
            agent_type="worker",
            cli_override=None,
        )

        # Should use hardcoded default from ModelPreferences
        assert model == "deepseek/deepseek-chat"

    def test_precedence_cli_beats_env(self, tmp_config_dir, monkeypatch):
        """Test CLI override beats environment variable.

        Verifies Level 1 > Level 2 precedence.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(minimal_toml())

        cli_model = "anthropic/claude-opus-4-5-20251101"
        env_model = "anthropic/claude-sonnet-4-20250514"

        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", env_model)

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="test_agent",
            agent_type="worker",
            cli_override=cli_model,
        )

        assert model == cli_model  # CLI wins

    def test_precedence_env_beats_per_agent(self, tmp_config_dir, monkeypatch):
        """Test environment variable beats per-agent override.

        Verifies Level 2 > Level 3 precedence.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        env_model = "anthropic/claude-sonnet-4-20250514"
        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", env_model)

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="test_agent",  # Has override in config
            agent_type="worker",
            cli_override=None,
        )

        assert model == env_model  # Env var wins

    def test_precedence_per_agent_beats_agent_type(self, tmp_config_dir, clean_env):
        """Test per-agent override beats agent type default.

        Verifies Level 3 > Level 4 precedence.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="test_agent",  # Has override
            agent_type="worker",  # Has agent type default
            cli_override=None,
        )

        # Override should win over agent type
        assert model == "anthropic/claude-opus-4-5-20251101"
        assert model != "deepseek/deepseek-chat"  # Not worker default

    def test_precedence_agent_type_beats_global(self, tmp_config_dir, clean_env):
        """Test agent type default beats global default.

        Verifies Level 4 > Level 5 precedence.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_custom_default())

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="some_worker",
            agent_type="worker",  # Has specific default
            cli_override=None,
        )

        # Agent type default should win
        assert model == "anthropic/claude-sonnet-4-20250514"
        assert model != "anthropic/claude-opus-4-5-20251101"  # Not global default

    def test_precedence_all_levels_set(self, tmp_config_dir, monkeypatch):
        """Test with all precedence levels set (CLI should win).

        Edge case: All 6 levels configured, CLI override should take precedence.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        cli_model = "test/cli-model"
        env_model = "test/env-model"

        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", env_model)

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="test_agent",  # Has per-agent override
            agent_type="worker",  # Has agent type default
            cli_override=cli_model,
        )

        assert model == cli_model

    def test_precedence_no_levels_set(self, tmp_config_dir, clean_env):
        """Test with no precedence levels set (hardcoded fallback).

        Edge case: Empty config, no env var, no CLI override.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(empty_toml())

        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="random_agent",
            agent_type="random_type",
            cli_override=None,
        )

        # Should use hardcoded fallback
        assert model == "deepseek/deepseek-chat"


class TestPreferencesSetters:
    """Test preference modification methods (P1 High)."""

    def test_set_default_model(self, tmp_config_dir, clean_env):
        """Test setting global default model.

        Verifies that set_default_model updates the config and persists changes.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        new_model = "anthropic/claude-opus-4-5-20251101"
        manager.set_default_model(new_model)

        # Reload and verify
        preferences = manager.load()
        assert preferences.models.default == new_model

    def test_set_agent_type_model_valid(self, tmp_config_dir, clean_env):
        """Test setting agent type model with valid type.

        Verifies that set_agent_type_model updates the config correctly.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        new_model = "anthropic/claude-sonnet-4-20250514"
        manager.set_agent_type_model("worker", new_model)

        # Reload and verify
        preferences = manager.load()
        assert preferences.models.by_agent_type.worker == new_model

    def test_set_agent_type_model_invalid_type(self, tmp_config_dir, clean_env):
        """Test invalid agent type raises ValueError.

        Verifies that set_agent_type_model validates agent type.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        with pytest.raises(ValueError, match="Invalid agent type"):
            manager.set_agent_type_model("invalid_type", "some/model")

    def test_set_agent_type_all_valid_types(self, tmp_config_dir, clean_env):
        """Test setting all valid agent types.

        Verifies that all expected agent types can be set.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        agent_types = ["worker", "leader", "notetaker", "compressor"]
        test_model = "test/model"

        for agent_type in agent_types:
            manager.set_agent_type_model(agent_type, test_model)

        # Reload and verify
        preferences = manager.load()
        assert preferences.models.by_agent_type.worker == test_model
        assert preferences.models.by_agent_type.leader == test_model
        assert preferences.models.by_agent_type.notetaker == test_model
        assert preferences.models.by_agent_type.compressor == test_model


class TestPreferencesSingleton:
    """Test singleton pattern (P2 Medium)."""

    def test_singleton_returns_same_instance(self):
        """Test get_preferences_manager returns singleton.

        Verifies that multiple calls return the same instance.
        """
        manager1 = get_preferences_manager()
        manager2 = get_preferences_manager()

        assert manager1 is manager2

    def test_singleton_isolation_with_custom_path(self, tmp_config_dir):
        """Test custom path doesn't affect singleton.

        Verifies that creating a manager with custom path doesn't
        interfere with the global singleton.
        """
        config_file = tmp_config_dir / "preferences.toml"
        custom_manager = PreferencesManager(config_path=config_file)
        singleton_manager = get_preferences_manager()

        # Different instances
        assert custom_manager is not singleton_manager

        # Different config paths
        assert custom_manager.config_path != singleton_manager.config_path
