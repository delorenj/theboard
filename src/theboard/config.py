"""Configuration management for TheBoard application."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
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
settings = get_settings()
