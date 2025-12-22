"""User preferences management with TOML persistence."""

import logging
import os
from pathlib import Path
from typing import Any

import tomli
import tomli_w
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelsByAgentType(BaseModel):
    """Model preferences by agent type."""

    worker: str = "deepseek/deepseek-chat"  # $0.14/MTok
    leader: str = "anthropic/claude-sonnet-4-20250514"  # $3/MTok
    notetaker: str = "anthropic/claude-3-haiku-20240307"  # Fast structured output
    compressor: str = "anthropic/claude-3-haiku-20240307"


class ModelPreferences(BaseModel):
    """Model selection preferences."""

    default: str = "deepseek/deepseek-chat"
    by_agent_type: ModelsByAgentType = Field(default_factory=ModelsByAgentType)
    overrides: dict[str, str] = Field(default_factory=dict)  # Per-agent testing


class Preferences(BaseModel):
    """User preferences for TheBoard."""

    models: ModelPreferences = Field(default_factory=ModelPreferences)


class PreferencesManager:
    """Manager for loading and saving user preferences."""

    CONFIG_DIR = Path.home() / ".config" / "theboard"
    CONFIG_FILE = CONFIG_DIR / "preferences.toml"

    ENV_VAR_DEFAULT_MODEL = "THEBOARD_DEFAULT_MODEL"

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize preferences manager.

        Args:
            config_path: Optional custom config path (for testing)
        """
        self.config_path = config_path or self.CONFIG_FILE
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def load(self) -> Preferences:
        """Load preferences from file.

        Returns:
            Preferences object (creates default if file doesn't exist)
        """
        if not self.config_path.exists():
            logger.info("Preferences file not found, creating default")
            return self._create_default()

        try:
            with open(self.config_path, "rb") as f:
                data = tomli.load(f)

            preferences = Preferences(**data)
            logger.debug("Loaded preferences from %s", self.config_path)
            return preferences

        except Exception as e:
            logger.error("Failed to load preferences: %s", e)
            logger.info("Using default preferences")
            return Preferences()

    def save(self, preferences: Preferences) -> None:
        """Save preferences to file.

        Args:
            preferences: Preferences to save
        """
        try:
            with open(self.config_path, "wb") as f:
                tomli_w.dump(preferences.model_dump(), f)

            logger.info("Saved preferences to %s", self.config_path)

        except Exception as e:
            logger.error("Failed to save preferences: %s", e)
            raise

    def _create_default(self) -> Preferences:
        """Create and save default preferences.

        Returns:
            Default preferences object
        """
        preferences = Preferences()
        self.save(preferences)
        return preferences

    def get_model_for_agent(
        self,
        agent_name: str,
        agent_type: str = "worker",
        cli_override: str | None = None,
    ) -> str:
        """Get model ID for specific agent with precedence logic.

        Precedence (highest to lowest):
        1. CLI flag override (--model)
        2. Environment variable (THEBOARD_DEFAULT_MODEL)
        3. Per-agent override in preferences.toml
        4. Agent type default in preferences.toml
        5. Global default in preferences.toml
        6. Hardcoded fallback

        Args:
            agent_name: Name of the agent
            agent_type: Type of agent (worker, leader, notetaker, compressor)
            cli_override: CLI flag value (--model)

        Returns:
            Model ID string (format: "provider/model-name")
        """
        # 1. CLI flag override (highest precedence)
        if cli_override:
            logger.debug("Using CLI override model for %s: %s", agent_name, cli_override)
            return cli_override

        # 2. Environment variable
        env_model = os.getenv(self.ENV_VAR_DEFAULT_MODEL)
        if env_model:
            logger.debug("Using env var model for %s: %s", agent_name, env_model)
            return env_model

        # Load preferences for remaining checks
        preferences = self.load()

        # 3. Per-agent override
        if agent_name in preferences.models.overrides:
            model = preferences.models.overrides[agent_name]
            logger.debug("Using per-agent override for %s: %s", agent_name, model)
            return model

        # 4. Agent type default
        agent_type_models = preferences.models.by_agent_type.model_dump()
        if agent_type in agent_type_models:
            model = agent_type_models[agent_type]
            logger.debug("Using agent type model for %s (%s): %s", agent_name, agent_type, model)
            return model

        # 5. Global default
        logger.debug(
            "Using global default model for %s: %s", agent_name, preferences.models.default
        )
        return preferences.models.default

    def set_default_model(self, model_id: str) -> None:
        """Set global default model.

        Args:
            model_id: Model ID to set as default
        """
        preferences = self.load()
        preferences.models.default = model_id
        self.save(preferences)
        logger.info("Set global default model to: %s", model_id)

    def set_agent_type_model(self, agent_type: str, model_id: str) -> None:
        """Set default model for agent type.

        Args:
            agent_type: Agent type (worker, leader, notetaker, compressor)
            model_id: Model ID to set

        Raises:
            ValueError: If agent_type is invalid
        """
        preferences = self.load()

        if not hasattr(preferences.models.by_agent_type, agent_type):
            msg = f"Invalid agent type: {agent_type}"
            raise ValueError(msg)

        setattr(preferences.models.by_agent_type, agent_type, model_id)
        self.save(preferences)
        logger.info("Set %s agent type model to: %s", agent_type, model_id)


# Global singleton
_preferences_manager: PreferencesManager | None = None


def get_preferences_manager() -> PreferencesManager:
    """Get global preferences manager instance.

    Returns:
        PreferencesManager singleton
    """
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = PreferencesManager()
    return _preferences_manager
