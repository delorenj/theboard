"""Configuration management for TheBoard application."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Iterable, Literal

from dotenv import dotenv_values
from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource, YamlConfigSettingsSource
import yaml

CONFIG_DIR = Path.home() / ".config" / "theboard"
USER_CONFIG_FILE = CONFIG_DIR / "config.yml"
REPO_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
REPO_ENV_EXAMPLE_FILE = Path(__file__).resolve().parents[2] / ".env.example"
CWD_ENV_FILE = Path.cwd() / ".env"


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return tuple(ordered)


def _dotenv_candidates() -> tuple[Path, ...]:
    """Return dotenv candidates in ascending precedence order."""
    candidates: list[Path] = []
    env_override = os.getenv("THEBOARD_ENV_FILE")
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.extend([CWD_ENV_FILE, REPO_ENV_FILE, REPO_ENV_EXAMPLE_FILE])
    return _unique_paths(candidates)


def _config_file_candidates() -> tuple[str, ...]:
    """Return config.yml candidates in ascending precedence order."""
    candidates: list[Path] = []
    config_override = os.getenv("THEBOARD_CONFIG_FILE")
    if config_override:
        candidates.append(Path(config_override).expanduser())
    candidates.append(USER_CONFIG_FILE)
    return tuple(str(path) for path in _unique_paths(candidates))


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=tuple(str(path) for path in _dotenv_candidates()),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Prefer env vars, then config.yml, then dotenv for settings."""
        yaml_source = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=_config_file_candidates(),
        )
        return (
            init_settings,
            env_settings,
            yaml_source,
            dotenv_settings,
            file_secret_settings,
        )

    # Application
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = True
    testing: bool = False

    # Event-Driven Architecture
    event_emitter: Literal["null", "rabbitmq", "inmemory"] = "null"

    # Database
    postgres_user: str = "theboard"
    postgres_password: str = "theboard_dev_pass"
    postgres_db: str = "theboard"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: PostgresDsn = Field(
        default="postgresql+psycopg://theboard:theboard_dev_pass@localhost:5432/theboard"
    )

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = "theboard_redis_pass"
    redis_db: int = 0
    redis_url: RedisDsn = Field(default="redis://:theboard_redis_pass@localhost:6379/0")

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # RabbitMQ
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "theboard"
    rabbitmq_password: str = "theboard_rabbit_pass"
    rabbitmq_url: str = "amqp://theboard:theboard_rabbit_pass@localhost:5672/"

    # LLM Provider - OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Agent Configuration
    max_rounds: int = 5
    default_strategy: Literal["sequential", "greedy"] = "sequential"
    default_agent_count: int = 5
    convergence_threshold: float = 0.2
    convergence_rounds: int = 2
    context_size_warning: int = 15000
    context_size_limit: int = 20000

    # Compression Configuration
    compression_similarity_threshold: float = 0.85
    compression_min_support: int = 2
    lazy_compression_threshold: int = 10000

    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 100

    @property
    def database_url_str(self) -> str:
        """Get database URL as string."""
        return str(self.database_url)

    @property
    def redis_url_str(self) -> str:
        """Get Redis URL as string."""
        return str(self.redis_url)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
def _filter_dotenv_for_settings(dotenv_data: dict[str, str | None]) -> dict[str, str]:
    settings_keys = set(Settings.model_fields.keys())
    filtered: dict[str, str] = {}
    for key, value in dotenv_data.items():
        if value is None:
            continue
        normalized_key = key.lower()
        if normalized_key in settings_keys:
            filtered[normalized_key] = value
    return filtered


def _write_yaml_config(config_path: Path, data: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def _bootstrap_user_config() -> None:
    if os.getenv("THEBOARD_SKIP_CONFIG_BOOTSTRAP"):
        return

    config_path = Path(_config_file_candidates()[0]).expanduser()
    if config_path.exists():
        return

    for candidate in _dotenv_candidates():
        if not candidate.exists():
            continue
        dotenv_data = dotenv_values(candidate)
        filtered = _filter_dotenv_for_settings(dotenv_data)
        if filtered:
            _write_yaml_config(config_path, filtered)
            return

    defaults = Settings.model_construct().model_dump()
    _write_yaml_config(config_path, defaults)


_bootstrap_user_config()
settings = get_settings()
