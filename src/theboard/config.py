"""Configuration management for TheBoard application."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, PostgresDsn, RedisDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = True
    testing: bool = False

    # Event-Driven Architecture
    event_emitter: Literal["null", "rabbitmq", "inmemory"] = "null"

    # Database
    postgres_user: str = Field(
        default="theboard", validation_alias=AliasChoices("POSTGRES_USER", "DEFAULT_USERNAME")
    )
    postgres_password: str = Field(
        default="theboard_dev_pass",
        validation_alias=AliasChoices("POSTGRES_PASSWORD", "DEFAULT_PASSWORD"),
    )
    postgres_db: str = Field(
        default="theboard", validation_alias=AliasChoices("POSTGRES_DB", "DEFAULT_DATABASE")
    )
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: PostgresDsn | None = None

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = Field(
        default="theboard_redis_pass",
        validation_alias=AliasChoices("REDIS_PASSWORD", "DEFAULT_PASSWORD"),
    )
    redis_db: int = 0
    redis_url: RedisDsn | None = None

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    # RabbitMQ
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = Field(
        default="theboard", validation_alias=AliasChoices("RABBITMQ_USER", "DEFAULT_USERNAME")
    )
    rabbitmq_password: str = Field(
        default="theboard_rabbit_pass",
        validation_alias=AliasChoices("RABBITMQ_PASSWORD", "DEFAULT_PASSWORD"),
    )
    rabbitmq_url: str | None = None

    # LLM Provider - OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @model_validator(mode="after")
    def assemble_urls(self) -> "Settings":
        """Assemble URLs from components if not provided."""
        if self.database_url is None:
            # Construct Postgres URL
            self.database_url = (
                f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@"
                f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

        if self.redis_url is None:
            # Construct Redis URL
            if self.redis_password:
                self.redis_url = (
                    f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
                )
            else:
                self.redis_url = f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

        if self.rabbitmq_url is None:
            # Construct RabbitMQ URL
            self.rabbitmq_url = (
                f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}@"
                f"{self.rabbitmq_host}:{self.rabbitmq_port}/"
            )

        return self

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
settings = get_settings()
