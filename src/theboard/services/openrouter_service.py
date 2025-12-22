"""OpenRouter API service for dynamic model discovery and selection."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from theboard.config import settings

logger = logging.getLogger(__name__)


class OpenRouterModel(BaseModel):
    """OpenRouter model metadata."""

    id: str  # Format: "provider/model-name"
    name: str
    context_length: int
    pricing: dict[str, float]  # {"prompt": 0.003, "completion": 0.015}
    architecture: dict[str, Any] = Field(default_factory=dict)

    @property
    def input_cost_per_mtok(self) -> float:
        """Input cost per million tokens."""
        return self.pricing.get("prompt", 0.0) * 1_000_000

    @property
    def output_cost_per_mtok(self) -> float:
        """Output cost per million tokens."""
        return self.pricing.get("completion", 0.0) * 1_000_000

    @property
    def max_cost_per_mtok(self) -> float:
        """Maximum cost for sorting (use output cost)."""
        return self.output_cost_per_mtok

    @property
    def cost_tier(self) -> str:
        """Cost tier for grouping: budget, standard, premium."""
        max_cost = self.max_cost_per_mtok
        if max_cost < 1.0:
            return "budget"
        elif max_cost < 10.0:
            return "standard"
        else:
            return "premium"


class OpenRouterModelCache(BaseModel):
    """Cache structure for OpenRouter models."""

    models: list[OpenRouterModel]
    cached_at: datetime
    ttl_hours: int = 24

    @property
    def is_expired(self) -> bool:
        """Check if cache is expired."""
        return datetime.now() > self.cached_at + timedelta(hours=self.ttl_hours)


class OpenRouterService:
    """Service for fetching and filtering OpenRouter models."""

    CACHE_DIR = Path.home() / ".cache" / "theboard"
    CACHE_FILE = CACHE_DIR / "openrouter_models.json"

    API_ENDPOINT = "https://openrouter.ai/api/v1/models"

    # Filter criteria
    MIN_CONTEXT_LENGTH = 8000
    REQUIRED_MODALITY = "chat"  # Must support chat completion

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize OpenRouter service.

        Args:
            api_key: OpenRouter API key (defaults to settings.openrouter_api_key)
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def fetch_models(self, force_refresh: bool = False) -> list[OpenRouterModel]:
        """Fetch OpenRouter models with caching.

        Args:
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            List of filtered and sorted OpenRouter models

        Raises:
            httpx.HTTPError: If API request fails
        """
        # Check cache first
        if not force_refresh and self.CACHE_FILE.exists():
            try:
                cache = self._load_cache()
                if not cache.is_expired:
                    logger.info(
                        "Using cached OpenRouter models (cached %s, expires %s)",
                        cache.cached_at.isoformat(),
                        (cache.cached_at + timedelta(hours=cache.ttl_hours)).isoformat(),
                    )
                    return cache.models
            except Exception as e:
                logger.warning("Failed to load cache: %s", e)

        # Fetch fresh data
        logger.info("Fetching fresh OpenRouter models from API")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.API_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/delorenj/theboard",
                    "X-Title": "TheBoard",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

        # Parse and filter models
        raw_models = data.get("data", [])
        models = []

        for raw_model in raw_models:
            try:
                # Skip models without required fields
                if not all(k in raw_model for k in ["id", "name", "context_length", "pricing"]):
                    logger.debug("Skipping model missing required fields: %s", raw_model.get("id", "unknown"))
                    continue

                model = OpenRouterModel(
                    id=raw_model["id"],
                    name=raw_model["name"],
                    context_length=raw_model["context_length"],
                    pricing=raw_model["pricing"],
                    architecture=raw_model.get("architecture", {}),
                )

                # Apply filters
                if not self._passes_filters(model):
                    logger.debug(
                        "Model filtered out: %s (context: %d, pricing: %s, arch: %s)",
                        model.id,
                        model.context_length,
                        model.pricing,
                        model.architecture,
                    )
                    continue

                models.append(model)

            except Exception as e:
                logger.debug("Skipping invalid model %s: %s", raw_model.get("id", "unknown"), e)
                continue

        # Sort by cost (budget first)
        models.sort(key=lambda m: m.max_cost_per_mtok)

        # Cache results
        self._save_cache(models)

        logger.info("Fetched and cached %d OpenRouter models", len(models))
        return models

    def _passes_filters(self, model: OpenRouterModel) -> bool:
        """Check if model passes filter criteria.

        Args:
            model: Model to check

        Returns:
            True if model passes all filters
        """
        # Must have sufficient context length
        if model.context_length < self.MIN_CONTEXT_LENGTH:
            logger.debug("Model %s filtered: context too small (%d < %d)", model.id, model.context_length, self.MIN_CONTEXT_LENGTH)
            return False

        # Must have pricing information (both prompt and completion)
        if not model.pricing:
            logger.debug("Model %s filtered: no pricing info", model.id)
            return False

        if model.pricing.get("prompt", 0) == 0 and model.pricing.get("completion", 0) == 0:
            logger.debug("Model %s filtered: zero pricing", model.id)
            return False

        # Note: Removed modality check - OpenRouter API doesn't consistently provide this
        # All models in their API are assumed to support chat completion

        return True

    def _load_cache(self) -> OpenRouterModelCache:
        """Load cache from disk.

        Returns:
            Cached model data

        Raises:
            FileNotFoundError: If cache file doesn't exist
            json.JSONDecodeError: If cache file is invalid
        """
        with open(self.CACHE_FILE, "r") as f:
            data = json.load(f)

        return OpenRouterModelCache(
            models=[OpenRouterModel(**m) for m in data["models"]],
            cached_at=datetime.fromisoformat(data["cached_at"]),
            ttl_hours=data.get("ttl_hours", 24),
        )

    def _save_cache(self, models: list[OpenRouterModel]) -> None:
        """Save models to cache.

        Args:
            models: Models to cache
        """
        cache = OpenRouterModelCache(
            models=models,
            cached_at=datetime.now(),
        )

        with open(self.CACHE_FILE, "w") as f:
            json.dump(
                {
                    "models": [m.model_dump() for m in cache.models],
                    "cached_at": cache.cached_at.isoformat(),
                    "ttl_hours": cache.ttl_hours,
                },
                f,
                indent=2,
            )

    def group_by_cost_tier(
        self, models: list[OpenRouterModel]
    ) -> dict[str, list[OpenRouterModel]]:
        """Group models by cost tier for display.

        Args:
            models: Models to group

        Returns:
            Dictionary mapping tier name to models
        """
        tiers: dict[str, list[OpenRouterModel]] = {
            "budget": [],
            "standard": [],
            "premium": [],
        }

        for model in models:
            tiers[model.cost_tier].append(model)

        return tiers
