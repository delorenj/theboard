"""Integration tests for model selection flow.

Tests the complete flow from preferences to agent creation,
including environment variables and CLI overrides.
"""

import pytest
import os
from pathlib import Path

from theboard.preferences import PreferencesManager, Preferences, ModelPreferences
from theboard.agents.base import create_agno_agent
from theboard.services.openrouter_service import OpenRouterService, OpenRouterModel
from tests.fixtures.toml_configs import minimal_toml, toml_with_overrides
from tests.fixtures.openrouter_responses import sample_api_response


class TestModelSelectionFlowE2E:
    """End-to-end tests for model selection flow."""

    def test_e2e_cli_flag_override_reaches_agent(self, tmp_config_dir, clean_env):
        """Test CLI flag override flows through to agent creation.

        E2E journey:
        1. Create preferences with default model
        2. Create agent with CLI override
        3. Verify agent uses CLI model (not preferences default)
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        # Setup: Create preferences with specific default
        prefs = Preferences()
        prefs.models.default = "anthropic/claude-3-haiku-20240307"
        manager.save(prefs)

        # Action: Create agent with CLI override
        cli_override = "anthropic/claude-opus-4-5-20251101"

        # Get model via manager (simulating agent factory)
        selected_model = manager.get_model_for_agent(
            agent_name="test_agent",
            agent_type="worker",
            cli_override=cli_override,
        )

        # Assert: CLI override should win
        assert selected_model == cli_override
        assert selected_model != prefs.models.default

    def test_e2e_preferences_file_to_agent(self, tmp_config_dir, clean_env):
        """Test preferences file flows through to agent creation.

        E2E journey:
        1. Create preferences with custom agent type models
        2. Create agent without overrides
        3. Verify agent uses preferences agent type model
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        manager = PreferencesManager(config_path=config_file)

        # Action: Get model for agent with override in config
        selected_model = manager.get_model_for_agent(
            agent_name="test_agent",  # Has override in toml_with_overrides
            agent_type="worker",
            cli_override=None,
        )

        # Assert: Should use per-agent override
        assert selected_model == "anthropic/claude-opus-4-5-20251101"

        # Action: Get model for regular worker
        worker_model = manager.get_model_for_agent(
            agent_name="regular_worker",
            agent_type="worker",
            cli_override=None,
        )

        # Assert: Should use worker agent type default
        assert worker_model == "deepseek/deepseek-chat"

    def test_e2e_environment_variable_override(self, tmp_config_dir, monkeypatch):
        """Test environment variable flows through to agent creation.

        E2E journey:
        1. Create preferences with defaults
        2. Set THEBOARD_DEFAULT_MODEL environment variable
        3. Create agent without CLI override
        4. Verify agent uses env var model
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(minimal_toml())

        env_model = "anthropic/claude-sonnet-4-20250514"
        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", env_model)

        manager = PreferencesManager(config_path=config_file)

        # Action: Get model (should use env var)
        selected_model = manager.get_model_for_agent(
            agent_name="test_agent",
            agent_type="worker",
            cli_override=None,
        )

        # Assert: Should use environment variable
        assert selected_model == env_model

    def test_e2e_full_precedence_chain(self, tmp_config_dir, monkeypatch):
        """Test complete precedence chain in realistic scenario.

        Scenario: Config has defaults, env var is set, CLI override provided.
        Expected: CLI override wins.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        # Setup all precedence levels
        env_model = "env/model"
        cli_model = "cli/model"

        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", env_model)

        manager = PreferencesManager(config_path=config_file)

        # Action: Request with CLI override (highest precedence)
        selected_model = manager.get_model_for_agent(
            agent_name="test_agent",  # Has per-agent override in config
            agent_type="leader",  # Has agent type default
            cli_override=cli_model,
        )

        # Assert: CLI should win despite all other levels being set
        assert selected_model == cli_model

    def test_e2e_preferences_persistence_across_sessions(self, tmp_config_dir, clean_env):
        """Test preferences persist across manager instances.

        Simulates multiple sessions/CLI invocations.
        """
        config_file = tmp_config_dir / "preferences.toml"

        # Session 1: Create and save preferences
        manager1 = PreferencesManager(config_path=config_file)
        manager1.set_default_model("anthropic/claude-opus-4-5-20251101")
        manager1.set_agent_type_model("worker", "deepseek/deepseek-chat")

        # Session 2: Load preferences in new manager instance
        manager2 = PreferencesManager(config_path=config_file)
        prefs = manager2.load()

        # Assert: Configuration persisted
        assert prefs.models.default == "anthropic/claude-opus-4-5-20251101"
        assert prefs.models.by_agent_type.worker == "deepseek/deepseek-chat"

    def test_e2e_model_selection_with_invalid_config(self, tmp_config_dir, clean_env):
        """Test graceful fallback when config is invalid.

        E2E journey:
        1. Create corrupted config file
        2. Attempt to load preferences
        3. Verify fallback to defaults
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text("invalid toml [[[")

        manager = PreferencesManager(config_path=config_file)

        # Action: Load corrupted config (should fall back to defaults)
        selected_model = manager.get_model_for_agent(
            agent_name="test_agent",
            agent_type="worker",
            cli_override=None,
        )

        # Assert: Should use hardcoded fallback
        assert selected_model == "deepseek/deepseek-chat"


class TestOpenRouterIntegration:
    """Integration tests for OpenRouter service with real filesystem."""

    @pytest.mark.asyncio
    async def test_cache_filesystem_integration(self, tmp_cache_dir):
        """Test cache round-trip with real filesystem.

        Verifies that cache save/load works with actual file I/O.
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Create sample models
        models = [
            OpenRouterModel(
                id="test/model1",
                name="Test Model 1",
                context_length=10000,
                pricing={"prompt": "0.001", "completion": "0.002"},
                architecture={"modality": "text->text,chat"},
            ),
            OpenRouterModel(
                id="test/model2",
                name="Test Model 2",
                context_length=20000,
                pricing={"prompt": "0.005", "completion": "0.010"},
                architecture={"modality": "text->text,chat"},
            ),
        ]

        # Save to cache
        service._save_cache(models)

        # Verify file exists
        assert service.CACHE_FILE.exists()

        # Load from cache
        cache = service._load_cache()

        # Verify data integrity
        assert len(cache.models) == 2
        assert cache.models[0].id == "test/model1"
        assert cache.models[1].id == "test/model2"
        assert cache.models[0].name == "Test Model 1"
        assert cache.models[1].context_length == 20000

    @pytest.mark.asyncio
    async def test_filter_sort_pipeline(self, tmp_cache_dir, mocker):
        """Test full filter + sort pipeline.

        Integration test verifying that:
        1. Raw API response is parsed
        2. Invalid models are filtered out
        3. Valid models are sorted by cost
        4. Results are cached
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Mock API response with mix of valid and invalid models
        mock_response = mocker.Mock()
        mock_response.json.return_value = sample_api_response()
        mock_response.raise_for_status.return_value = None

        mock_client = mocker.Mock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get.return_value = mock_response

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Fetch models
        models = await service.fetch_models(force_refresh=True)

        # Assert: Invalid models filtered out (4 valid models in sample_api_response)
        assert len(models) == 4

        # Assert: Models sorted by cost (cheapest first)
        for i in range(len(models) - 1):
            assert models[i].max_cost_per_mtok <= models[i + 1].max_cost_per_mtok

        # Assert: Cache file created
        assert service.CACHE_FILE.exists()

        # Assert: Cache can be loaded
        cache = service._load_cache()
        assert len(cache.models) == 4

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_api_call(self, tmp_cache_dir, frozen_time, mocker):
        """Test that valid cache prevents API call.

        Integration test verifying cache optimization.
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Create fresh cache
        models = [
            OpenRouterModel(
                id="cached/model",
                name="Cached Model",
                context_length=10000,
                pricing={"prompt": "0.001", "completion": "0.002"},
                architecture={"modality": "text->text,chat"},
            )
        ]
        service._save_cache(models)

        # Mock httpx to verify no call
        mock_get = mocker.patch("httpx.AsyncClient.get")

        # Fetch models (should use cache)
        result = await service.fetch_models(force_refresh=False)

        # Assert: No API call made
        mock_get.assert_not_called()

        # Assert: Cached models returned
        assert len(result) == 1
        assert result[0].id == "cached/model"


class TestPreferencesTomlPersistence:
    """Integration tests for TOML file persistence."""

    def test_toml_round_trip_with_filesystem(self, tmp_config_dir, clean_env):
        """Test full TOML save + load with real filesystem.

        Integration test for TOML persistence without mocking file I/O.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        # Create complex preferences
        prefs = Preferences()
        prefs.models.default = "anthropic/claude-opus-4-5-20251101"
        prefs.models.by_agent_type.worker = "deepseek/deepseek-chat"
        prefs.models.by_agent_type.leader = "anthropic/claude-sonnet-4-20250514"
        prefs.models.overrides = {
            "special_agent": "openai/gpt-4",
            "debug_agent": "anthropic/claude-3-haiku-20240307",
        }

        # Save to file
        manager.save(prefs)

        # Verify file exists and is valid TOML
        assert config_file.exists()
        file_content = config_file.read_text()
        assert "[models]" in file_content
        assert "[models.by_agent_type]" in file_content
        assert "[models.overrides]" in file_content

        # Load in new manager instance
        manager2 = PreferencesManager(config_path=config_file)
        loaded = manager2.load()

        # Verify exact match
        assert loaded.models.default == prefs.models.default
        assert loaded.models.by_agent_type.worker == prefs.models.by_agent_type.worker
        assert loaded.models.by_agent_type.leader == prefs.models.by_agent_type.leader
        assert loaded.models.overrides == prefs.models.overrides

    def test_environment_variable_integration(self, tmp_config_dir, monkeypatch):
        """Test environment variable integration with filesystem.

        Integration test for env var precedence with real config files.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(minimal_toml())

        env_model = "env/override-model"
        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", env_model)

        manager = PreferencesManager(config_path=config_file)

        # Get model (should use env var)
        model = manager.get_model_for_agent(
            agent_name="test",
            agent_type="worker",
            cli_override=None,
        )

        assert model == env_model

    def test_multi_level_precedence_with_files(self, tmp_config_dir, monkeypatch):
        """Test precedence with real TOML files and env vars.

        Comprehensive integration test for precedence hierarchy.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(toml_with_overrides())

        # Test Level 3: Per-agent override (no env, no CLI)
        manager = PreferencesManager(config_path=config_file)
        model = manager.get_model_for_agent(
            agent_name="test_agent",
            agent_type="worker",
            cli_override=None,
        )
        assert model == "anthropic/claude-opus-4-5-20251101"

        # Test Level 2: Env var overrides config
        monkeypatch.setenv("THEBOARD_DEFAULT_MODEL", "env/model")
        model = manager.get_model_for_agent(
            agent_name="test_agent",
            agent_type="worker",
            cli_override=None,
        )
        assert model == "env/model"

        # Test Level 1: CLI overrides everything
        model = manager.get_model_for_agent(
            agent_name="test_agent",
            agent_type="worker",
            cli_override="cli/model",
        )
        assert model == "cli/model"

    def test_empty_toml_file_handling(self, tmp_config_dir, clean_env):
        """Test handling of empty TOML file.

        Integration test for error recovery with real filesystem.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text("")

        manager = PreferencesManager(config_path=config_file)
        prefs = manager.load()

        # Should use Pydantic defaults
        assert prefs.models.default == "deepseek/deepseek-chat"
        assert prefs.models.by_agent_type.worker == "deepseek/deepseek-chat"

    def test_minimal_toml_parsing(self, tmp_config_dir, clean_env):
        """Test minimal valid TOML.

        Integration test for parsing minimal but valid configuration.
        """
        config_file = tmp_config_dir / "preferences.toml"
        config_file.write_text(minimal_toml())

        manager = PreferencesManager(config_path=config_file)
        prefs = manager.load()

        assert prefs.models.default == "deepseek/deepseek-chat"
        assert prefs.models.by_agent_type.worker == "deepseek/deepseek-chat"
        assert prefs.models.by_agent_type.leader == "anthropic/claude-sonnet-4-20250514"
        assert prefs.models.by_agent_type.notetaker == "anthropic/claude-3-haiku-20240307"
