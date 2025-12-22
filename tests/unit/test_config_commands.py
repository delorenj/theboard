"""Unit tests for config CLI commands.

Priority breakdown:
- P1 (High): Error handling, basic functionality
- P2 (Medium): Formatting, validation
"""

import pytest
from typer.testing import CliRunner
from unittest.mock import Mock

from theboard.cli_commands.config import config_app
from theboard.preferences import PreferencesManager, Preferences
from tests.fixtures.openrouter_responses import sample_api_response, empty_api_response


runner = CliRunner()


class TestConfigInit:
    """Test config init command (P1 High)."""

    def test_config_init_creates_file(self, tmp_config_dir, monkeypatch):
        """Test config file creation.

        Verifies that 'config init' creates a preferences file.
        """
        config_file = tmp_config_dir / "preferences.toml"

        # Mock get_preferences_manager to use tmp directory
        def mock_get_manager():
            return PreferencesManager(config_path=config_file)

        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            mock_get_manager,
        )

        result = runner.invoke(config_app, ["init"])

        assert result.exit_code == 0
        assert config_file.exists()
        assert "Configuration file created" in result.stdout

    def test_config_init_file_exists_without_force(self, tmp_config_dir, monkeypatch):
        """Test error when file exists without --force.

        Verifies that 'config init' fails if file already exists
        and --force is not provided.
        """
        config_file = tmp_config_dir / "preferences.toml"

        # Create existing file
        manager = PreferencesManager(config_path=config_file)
        manager.save(Preferences())

        # Mock get_preferences_manager
        def mock_get_manager():
            return manager

        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            mock_get_manager,
        )

        result = runner.invoke(config_app, ["init"])

        assert result.exit_code == 1
        assert "already exists" in result.stdout

    def test_config_init_force_overwrites(self, tmp_config_dir, monkeypatch):
        """Test --force overwrites existing file.

        Verifies that 'config init --force' overwrites existing config.
        """
        config_file = tmp_config_dir / "preferences.toml"

        # Create existing file with custom content
        manager = PreferencesManager(config_path=config_file)
        prefs = Preferences()
        prefs.models.default = "custom/model"
        manager.save(prefs)

        # Mock get_preferences_manager
        def mock_get_manager():
            return PreferencesManager(config_path=config_file)

        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            mock_get_manager,
        )

        result = runner.invoke(config_app, ["init", "--force"])

        assert result.exit_code == 0

        # Verify file was overwritten with defaults
        reloaded = manager.load()
        assert reloaded.models.default == "deepseek/deepseek-chat"


class TestConfigShow:
    """Test config show command (P2 Medium)."""

    def test_config_show_success(self, tmp_config_dir, monkeypatch):
        """Test successful config display.

        Verifies that 'config show' displays current configuration.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)
        manager.save(Preferences())

        # Mock get_preferences_manager
        def mock_get_manager():
            return manager

        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            mock_get_manager,
        )

        result = runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        assert "TheBoard Configuration" in result.stdout
        assert "deepseek/deepseek-chat" in result.stdout
        assert "anthropic/claude-sonnet-4-20250514" in result.stdout

    def test_config_show_with_overrides(self, tmp_config_dir, monkeypatch):
        """Test display with per-agent overrides.

        Verifies that 'config show' displays per-agent overrides
        when they exist.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)

        # Create preferences with overrides
        prefs = Preferences()
        prefs.models.overrides = {"test_agent": "anthropic/claude-opus-4-5-20251101"}
        manager.save(prefs)

        # Mock get_preferences_manager
        def mock_get_manager():
            return manager

        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            mock_get_manager,
        )

        result = runner.invoke(config_app, ["show"])

        assert result.exit_code == 0
        assert "Per-Agent Overrides" in result.stdout
        assert "test_agent" in result.stdout
        assert "anthropic/claude-opus-4-5-20251101" in result.stdout


class TestConfigModels:
    """Test config models command (P1 High - Error Handling)."""

    def test_config_models_empty_model_list(self, tmp_config_dir, monkeypatch, mocker):
        """Test error when API returns no models.

        Verifies that 'config models' handles empty model list gracefully.
        """
        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)
        manager.save(Preferences())

        # Mock get_preferences_manager
        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            lambda: manager,
        )

        # Mock OpenRouterService to return empty list
        mock_service = mocker.Mock()
        mock_service.fetch_models = mocker.AsyncMock(return_value=[])

        mocker.patch(
            "theboard.cli_commands.config.OpenRouterService",
            return_value=mock_service,
        )

        result = runner.invoke(config_app, ["models"])

        assert result.exit_code == 1
        assert "No models found" in result.stdout

    def test_config_models_user_cancels(self, tmp_config_dir, monkeypatch, mocker):
        """Test user cancellation flow.

        Verifies that 'config models' handles user cancellation gracefully.
        """
        from theboard.services.openrouter_service import OpenRouterModel

        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)
        manager.save(Preferences())

        # Mock get_preferences_manager
        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            lambda: manager,
        )

        # Mock OpenRouterService to return sample models
        mock_models = [
            OpenRouterModel(
                id="test/model",
                name="Test Model",
                context_length=10000,
                pricing={"prompt": "0.001", "completion": "0.002"},
                architecture={"modality": "text->text,chat"},
            )
        ]

        mock_service = mocker.Mock()
        mock_service.fetch_models = mocker.AsyncMock(return_value=mock_models)
        mock_service.group_by_cost_tier = mocker.Mock(
            return_value={"budget": mock_models, "standard": [], "premium": []}
        )

        mocker.patch(
            "theboard.cli_commands.config.OpenRouterService",
            return_value=mock_service,
        )

        # Mock typer.confirm to return False (user cancels)
        mocker.patch("typer.confirm", return_value=False)

        result = runner.invoke(config_app, ["models"])

        assert result.exit_code == 0
        assert "No changes made" in result.stdout

    def test_config_models_api_timeout(self, tmp_config_dir, monkeypatch, mocker):
        """Test API timeout handling.

        Verifies that 'config models' handles API timeout errors gracefully.
        """
        import httpx

        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)
        manager.save(Preferences())

        # Mock get_preferences_manager
        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            lambda: manager,
        )

        # Mock OpenRouterService to raise timeout
        async def mock_fetch_models(*args, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        mock_service = mocker.Mock()
        mock_service.fetch_models = mock_fetch_models

        mocker.patch(
            "theboard.cli_commands.config.OpenRouterService",
            return_value=mock_service,
        )

        result = runner.invoke(config_app, ["models"])

        assert result.exit_code == 1
        assert "Error" in result.stdout

    def test_config_models_display_logic(self, tmp_config_dir, monkeypatch, mocker):
        """Test table rendering and grouping.

        Verifies that 'config models' displays models grouped by tier
        with correct formatting.
        """
        from theboard.services.openrouter_service import OpenRouterModel

        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)
        manager.save(Preferences())

        # Mock get_preferences_manager
        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            lambda: manager,
        )

        # Create models for different tiers
        budget_model = OpenRouterModel(
            id="budget/model",
            name="Budget Model",
            context_length=10000,
            pricing={"prompt": "0.0001", "completion": "0.0005"},
            architecture={"modality": "text->text,chat"},
        )

        standard_model = OpenRouterModel(
            id="standard/model",
            name="Standard Model",
            context_length=20000,
            pricing={"prompt": "0.001", "completion": "0.005"},
            architecture={"modality": "text->text,chat"},
        )

        premium_model = OpenRouterModel(
            id="premium/model",
            name="Premium Model",
            context_length=30000,
            pricing={"prompt": "0.01", "completion": "0.05"},
            architecture={"modality": "text->text,chat"},
        )

        all_models = [budget_model, standard_model, premium_model]

        mock_service = mocker.Mock()
        mock_service.fetch_models = mocker.AsyncMock(return_value=all_models)
        mock_service.group_by_cost_tier = mocker.Mock(
            return_value={
                "budget": [budget_model],
                "standard": [standard_model],
                "premium": [premium_model],
            }
        )

        mocker.patch(
            "theboard.cli_commands.config.OpenRouterService",
            return_value=mock_service,
        )

        # Mock typer.confirm to cancel (we just want to see the display)
        mocker.patch("typer.confirm", return_value=False)

        result = runner.invoke(config_app, ["models"])

        assert result.exit_code == 0

        # Verify tier tables are displayed
        assert "Budget Tier" in result.stdout
        assert "Standard Tier" in result.stdout
        assert "Premium Tier" in result.stdout

        # Verify model names appear
        assert "Budget Model" in result.stdout
        assert "Standard Model" in result.stdout
        assert "Premium Model" in result.stdout

    def test_config_models_successful_update(self, tmp_config_dir, monkeypatch, mocker):
        """Test successful model configuration update.

        Verifies that 'config models' updates configuration when user
        provides valid input.
        """
        from theboard.services.openrouter_service import OpenRouterModel

        config_file = tmp_config_dir / "preferences.toml"
        manager = PreferencesManager(config_path=config_file)
        manager.save(Preferences())

        # Mock get_preferences_manager
        monkeypatch.setattr(
            "theboard.cli_commands.config.get_preferences_manager",
            lambda: manager,
        )

        # Mock models
        mock_models = [
            OpenRouterModel(
                id="test/model",
                name="Test Model",
                context_length=10000,
                pricing={"prompt": "0.001", "completion": "0.002"},
                architecture={"modality": "text->text,chat"},
            )
        ]

        mock_service = mocker.Mock()
        mock_service.fetch_models = mocker.AsyncMock(return_value=mock_models)
        mock_service.group_by_cost_tier = mocker.Mock(
            return_value={"budget": mock_models, "standard": [], "premium": []}
        )

        mocker.patch(
            "theboard.cli_commands.config.OpenRouterService",
            return_value=mock_service,
        )

        # Mock user interactions
        mocker.patch("typer.confirm", return_value=True)
        mocker.patch("typer.prompt", return_value="test/model")

        result = runner.invoke(config_app, ["models"])

        assert result.exit_code == 0
        assert "Configuration updated successfully" in result.stdout

        # Verify configuration was actually updated
        updated_prefs = manager.load()
        assert updated_prefs.models.default == "test/model"
        assert updated_prefs.models.by_agent_type.worker == "test/model"
