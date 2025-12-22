"""Unit tests for OpenRouterService.

Priority breakdown:
- P0 (Critical): Cost calculations, cache expiration
- P1 (High): API interaction, filtering, sorting, cache management
- P2 (Medium): Grouping, edge cases
"""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path

from theboard.services.openrouter_service import (
    OpenRouterModel,
    OpenRouterModelCache,
    OpenRouterService,
)
from tests.fixtures.openrouter_responses import (
    sample_api_response,
    empty_api_response,
    budget_tier_models,
    standard_tier_models,
    premium_tier_models,
)


class TestOpenRouterModel:
    """Test OpenRouterModel cost calculations and properties (P0 Critical)."""

    def test_input_cost_per_mtok(self):
        """Test input cost calculation per million tokens.

        Verifies that input cost is correctly calculated by multiplying
        the prompt pricing by 1,000,000.
        """
        model = OpenRouterModel(
            id="test/model",
            name="Test Model",
            context_length=10000,
            pricing={"prompt": "0.003", "completion": "0.015"},
        )

        # 0.003 * 1,000,000 = 3,000
        assert model.input_cost_per_mtok == 3000.0

    def test_output_cost_per_mtok(self):
        """Test output cost calculation per million tokens.

        Verifies that output cost is correctly calculated by multiplying
        the completion pricing by 1,000,000.
        """
        model = OpenRouterModel(
            id="test/model",
            name="Test Model",
            context_length=10000,
            pricing={"prompt": "0.003", "completion": "0.015"},
        )

        # 0.015 * 1,000,000 = 15,000
        assert model.output_cost_per_mtok == 15000.0

    def test_max_cost_per_mtok_uses_output_cost(self):
        """Test max cost uses output cost for sorting.

        Verifies that max_cost_per_mtok returns the output cost,
        which is used for model sorting.
        """
        model = OpenRouterModel(
            id="test/model",
            name="Test Model",
            context_length=10000,
            pricing={"prompt": "0.001", "completion": "0.010"},
        )

        assert model.max_cost_per_mtok == model.output_cost_per_mtok
        assert model.max_cost_per_mtok == 10000.0

    def test_cost_tier_budget(self):
        """Test budget tier classification (<$1/MTok).

        Models with output cost < $1/MTok should be classified as 'budget'.
        Note: pricing values are per-token, max_cost_per_mtok multiplies by 1M.
        """
        model = OpenRouterModel(
            id="deepseek/deepseek-chat",
            name="DeepSeek Chat",
            context_length=32000,
            pricing={"prompt": "0.00000014", "completion": "0.00000028"},  # $0.28/MTok
        )

        assert model.cost_tier == "budget"
        assert model.max_cost_per_mtok < 1.0
        assert model.max_cost_per_mtok == 0.28

    def test_cost_tier_standard(self):
        """Test standard tier classification ($1-$10/MTok).

        Models with output cost between $1 and $10/MTok should be 'standard'.
        Note: pricing values are per-token, max_cost_per_mtok multiplies by 1M.
        """
        model = OpenRouterModel(
            id="anthropic/claude-3-haiku-20240307",
            name="Claude 3 Haiku",
            context_length=200000,
            pricing={"prompt": "0.00000025", "completion": "0.00000125"},  # $1.25/MTok
        )

        assert model.cost_tier == "standard"
        assert 1.0 <= model.max_cost_per_mtok < 10.0
        assert model.max_cost_per_mtok == 1.25

    def test_cost_tier_premium(self):
        """Test premium tier classification (>$10/MTok).

        Models with output cost >= $10/MTok should be 'premium'.
        """
        model = OpenRouterModel(
            id="anthropic/claude-opus-4-5-20251101",
            name="Claude Opus 4.5",
            context_length=200000,
            pricing={"prompt": "0.015", "completion": "0.075"},  # $75/MTok
        )

        assert model.cost_tier == "premium"
        assert model.max_cost_per_mtok >= 10.0

    def test_cost_tier_boundary_conditions(self):
        """Test exact boundary values for tier classification (P0 CRITICAL).

        Tests the exact transition points:
        - $0.99/MTok -> budget
        - $1.00/MTok -> standard
        - $9.99/MTok -> standard
        - $10.00/MTok -> premium

        Note: pricing values are per-token, max_cost_per_mtok multiplies by 1M.
        """
        # Just below budget/standard boundary ($0.99/MTok)
        model_099 = OpenRouterModel(
            id="test/budget-edge",
            name="Budget Edge",
            context_length=10000,
            pricing={"prompt": "0.0000001", "completion": "0.00000099"},  # Per-token
        )
        assert model_099.cost_tier == "budget"
        assert abs(model_099.max_cost_per_mtok - 0.99) < 0.01  # Floating point tolerance

        # At standard boundary ($1.00/MTok)
        model_100 = OpenRouterModel(
            id="test/standard-low",
            name="Standard Low",
            context_length=10000,
            pricing={"prompt": "0.0000001", "completion": "0.000001"},  # Per-token
        )
        assert model_100.cost_tier == "standard"
        assert model_100.max_cost_per_mtok == 1.0

        # Just below standard/premium boundary ($9.99/MTok)
        model_999 = OpenRouterModel(
            id="test/standard-high",
            name="Standard High",
            context_length=10000,
            pricing={"prompt": "0.000001", "completion": "0.00000999"},  # Per-token
        )
        assert model_999.cost_tier == "standard"
        assert abs(model_999.max_cost_per_mtok - 9.99) < 0.01  # Floating point tolerance

        # At premium boundary ($10.00/MTok)
        model_1000 = OpenRouterModel(
            id="test/premium-low",
            name="Premium Low",
            context_length=10000,
            pricing={"prompt": "0.000001", "completion": "0.00001"},  # Per-token
        )
        assert model_1000.cost_tier == "premium"
        assert model_1000.max_cost_per_mtok == 10.0

    def test_zero_pricing_handling(self):
        """Test models with zero pricing.

        Verifies that models with zero pricing are handled correctly
        (should be filtered out by service, but model should handle gracefully).
        """
        model = OpenRouterModel(
            id="test/free-model",
            name="Free Model",
            context_length=10000,
            pricing={"prompt": "0", "completion": "0"},
        )

        assert model.input_cost_per_mtok == 0.0
        assert model.output_cost_per_mtok == 0.0
        assert model.cost_tier == "budget"  # 0 < 1.0


class TestOpenRouterModelCache:
    """Test cache expiration logic (P0 Critical)."""

    def test_is_expired_fresh_cache(self, frozen_time):
        """Test cache within TTL is not expired.

        Verifies that a cache created recently is not considered expired.
        """
        # Cache created 1 hour ago
        cached_at = frozen_time - timedelta(hours=1)

        cache = OpenRouterModelCache(
            models=[],
            cached_at=cached_at,
            ttl_hours=24,
        )

        assert not cache.is_expired

    def test_is_expired_old_cache(self, frozen_time):
        """Test cache beyond TTL is expired.

        Verifies that a cache older than TTL is considered expired.
        """
        # Cache created 25 hours ago (beyond 24h TTL)
        cached_at = frozen_time - timedelta(hours=25)

        cache = OpenRouterModelCache(
            models=[],
            cached_at=cached_at,
            ttl_hours=24,
        )

        assert cache.is_expired

    def test_is_expired_boundary_exactly_24h(self, frozen_time):
        """Test cache at exactly 24h boundary.

        Edge case: Cache is exactly 24 hours old.
        Should be considered not expired (cached_at + 24h = now, not > now).
        """
        # Cache created exactly 24 hours ago
        cached_at = frozen_time - timedelta(hours=24)

        cache = OpenRouterModelCache(
            models=[],
            cached_at=cached_at,
            ttl_hours=24,
        )

        # At exactly TTL boundary, should not be expired (not strictly greater than)
        assert not cache.is_expired

    def test_is_expired_custom_ttl(self, frozen_time):
        """Test cache with custom TTL.

        Verifies that custom TTL values work correctly.
        """
        # Cache created 50 hours ago with 48h TTL
        cached_at = frozen_time - timedelta(hours=50)

        cache = OpenRouterModelCache(
            models=[],
            cached_at=cached_at,
            ttl_hours=48,
        )

        assert cache.is_expired

        # Cache created 40 hours ago with 48h TTL
        cached_at = frozen_time - timedelta(hours=40)

        cache = OpenRouterModelCache(
            models=[],
            cached_at=cached_at,
            ttl_hours=48,
        )

        assert not cache.is_expired


class TestOpenRouterServiceFiltering:
    """Test model filtering logic (P1 High)."""

    def test_passes_filters_valid_model(self):
        """Test model passing all filters.

        Verifies that a valid model with sufficient context, chat modality,
        and non-zero pricing passes all filters.
        """
        service = OpenRouterService(api_key="test-key")

        model = OpenRouterModel(
            id="test/valid-model",
            name="Valid Model",
            context_length=10000,  # > 8000
            pricing={"prompt": "0.001", "completion": "0.002"},  # Non-zero
            architecture={"modality": "text->text,chat"},  # Has 'chat'
        )

        assert service._passes_filters(model)

    def test_passes_filters_short_context(self):
        """Test model filtered by context length.

        Models with context_length < MIN_CONTEXT_LENGTH (8000) should be filtered.
        """
        service = OpenRouterService(api_key="test-key")

        model = OpenRouterModel(
            id="test/short-context",
            name="Short Context",
            context_length=4000,  # < 8000
            pricing={"prompt": "0.001", "completion": "0.002"},
            architecture={"modality": "text->text,chat"},
        )

        assert not service._passes_filters(model)

    def test_passes_filters_no_chat_modality(self):
        """Test model filtered by modality.

        Models without 'chat' in modality string should be filtered.
        """
        service = OpenRouterService(api_key="test-key")

        model = OpenRouterModel(
            id="test/no-chat",
            name="No Chat",
            context_length=10000,
            pricing={"prompt": "0.001", "completion": "0.002"},
            architecture={"modality": "text->text"},  # Missing 'chat'
        )

        assert not service._passes_filters(model)

    def test_passes_filters_zero_pricing(self):
        """Test model filtered by pricing.

        Models with zero completion pricing should be filtered.
        """
        service = OpenRouterService(api_key="test-key")

        model = OpenRouterModel(
            id="test/zero-pricing",
            name="Zero Pricing",
            context_length=10000,
            pricing={"prompt": "0", "completion": "0"},
            architecture={"modality": "text->text,chat"},
        )

        assert not service._passes_filters(model)

    def test_passes_filters_missing_pricing(self):
        """Test model with missing pricing dict.

        Models with empty or missing pricing should be filtered.
        """
        service = OpenRouterService(api_key="test-key")

        model = OpenRouterModel(
            id="test/no-pricing",
            name="No Pricing",
            context_length=10000,
            pricing={},
            architecture={"modality": "text->text,chat"},
        )

        assert not service._passes_filters(model)

    def test_passes_filters_case_insensitive_modality(self):
        """Test modality check is case-insensitive.

        'CHAT', 'Chat', 'chat' should all be accepted.
        """
        service = OpenRouterService(api_key="test-key")

        # Test uppercase
        model_upper = OpenRouterModel(
            id="test/upper",
            name="Upper",
            context_length=10000,
            pricing={"prompt": "0.001", "completion": "0.002"},
            architecture={"modality": "TEXT->TEXT,CHAT"},
        )
        assert service._passes_filters(model_upper)

        # Test mixed case
        model_mixed = OpenRouterModel(
            id="test/mixed",
            name="Mixed",
            context_length=10000,
            pricing={"prompt": "0.001", "completion": "0.002"},
            architecture={"modality": "Text->Text,Chat"},
        )
        assert service._passes_filters(model_mixed)


class TestOpenRouterServiceCaching:
    """Test cache management (P1 High)."""

    def test_save_cache_creates_file(self, tmp_cache_dir):
        """Test cache file creation.

        Verifies that _save_cache creates a valid JSON file.
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        models = [
            OpenRouterModel(
                id="test/model",
                name="Test Model",
                context_length=10000,
                pricing={"prompt": "0.001", "completion": "0.002"},
            )
        ]

        service._save_cache(models)

        assert service.CACHE_FILE.exists()

        # Verify JSON structure
        with open(service.CACHE_FILE) as f:
            data = json.load(f)

        assert "models" in data
        assert "cached_at" in data
        assert "ttl_hours" in data
        assert len(data["models"]) == 1

    def test_load_cache_valid_file(self, tmp_cache_dir, frozen_time):
        """Test loading valid cache file.

        Verifies that _load_cache correctly deserializes cache data.
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Create cache file
        cache_data = {
            "models": [
                {
                    "id": "test/model",
                    "name": "Test Model",
                    "context_length": 10000,
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                    "architecture": {},
                }
            ],
            "cached_at": frozen_time.isoformat(),
            "ttl_hours": 24,
        }

        with open(service.CACHE_FILE, "w") as f:
            json.dump(cache_data, f)

        # Load cache
        cache = service._load_cache()

        assert len(cache.models) == 1
        assert cache.models[0].id == "test/model"
        assert cache.cached_at == frozen_time
        assert cache.ttl_hours == 24

    def test_load_cache_corrupted_file(self, tmp_cache_dir):
        """Test handling of corrupted cache file.

        Verifies that _load_cache raises appropriate error for invalid JSON.
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Create corrupted cache
        service.CACHE_FILE.write_text("invalid json {{{")

        with pytest.raises(json.JSONDecodeError):
            service._load_cache()

    @pytest.mark.asyncio
    async def test_fetch_models_uses_cache(self, tmp_cache_dir, frozen_time, mocker):
        """Test cache hit avoids API call.

        Verifies that when valid cache exists, no API call is made.
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Create fresh cache
        models = [
            OpenRouterModel(
                id="test/cached-model",
                name="Cached Model",
                context_length=10000,
                pricing={"prompt": "0.001", "completion": "0.002"},
                architecture={"modality": "text->text,chat"},
            )
        ]
        service._save_cache(models)

        # Mock httpx to verify no API call
        mock_get = mocker.patch("httpx.AsyncClient.get")

        # Fetch models (should use cache)
        result = await service.fetch_models(force_refresh=False)

        # Verify no API call
        mock_get.assert_not_called()

        # Verify cached data returned
        assert len(result) == 1
        assert result[0].id == "test/cached-model"

    @pytest.mark.asyncio
    async def test_fetch_models_force_refresh(self, tmp_cache_dir, mocker):
        """Test force_refresh bypasses cache.

        Verifies that force_refresh=True always triggers API call.
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Create cache
        service._save_cache([])

        # Mock API response
        mock_response = mocker.Mock()
        mock_response.json.return_value = sample_api_response()
        mock_response.raise_for_status.return_value = None

        # Create async context manager mock properly
        mock_client = mocker.MagicMock()
        mock_client.get = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Fetch with force refresh
        result = await service.fetch_models(force_refresh=True)

        # Verify API was called
        mock_client.get.assert_called_once()

        # Verify models were fetched (4 valid models in sample_api_response)
        assert len(result) == 4


class TestOpenRouterServiceSorting:
    """Test model sorting (P1 High)."""

    @pytest.mark.asyncio
    async def test_models_sorted_by_cost(self, tmp_cache_dir, mocker):
        """Test models sorted by cost ascending.

        Verifies that fetch_models returns models sorted by max_cost_per_mtok
        in ascending order (cheapest first).
        """
        service = OpenRouterService(api_key="test-key")
        service.CACHE_DIR = tmp_cache_dir
        service.CACHE_FILE = tmp_cache_dir / "openrouter_models.json"

        # Mock API response
        mock_response = mocker.Mock()
        mock_response.json.return_value = sample_api_response()
        mock_response.raise_for_status.return_value = None

        # Create async context manager mock properly
        mock_client = mocker.MagicMock()
        mock_client.get = mocker.AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=None)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await service.fetch_models(force_refresh=True)

        # Verify sorting (cheapest to most expensive)
        for i in range(len(result) - 1):
            assert result[i].max_cost_per_mtok <= result[i + 1].max_cost_per_mtok

        # First model should be DeepSeek (cheapest)
        assert result[0].id == "deepseek/deepseek-chat"

        # Last model should be Opus (most expensive)
        assert result[-1].id == "anthropic/claude-opus-4-5-20251101"


class TestOpenRouterServiceGrouping:
    """Test cost tier grouping (P2 Medium)."""

    def test_group_by_cost_tier(self):
        """Test grouping models by cost tier.

        Verifies that group_by_cost_tier correctly categorizes models
        into budget, standard, and premium tiers.
        Note: pricing is per-token, not per-MTok.
        """
        service = OpenRouterService(api_key="test-key")

        models = [
            OpenRouterModel(
                id="budget/model",
                name="Budget",
                context_length=10000,
                pricing={"prompt": "0.0000001", "completion": "0.0000005"},  # $0.50/MTok
            ),
            OpenRouterModel(
                id="standard/model",
                name="Standard",
                context_length=10000,
                pricing={"prompt": "0.000001", "completion": "0.000005"},  # $5/MTok
            ),
            OpenRouterModel(
                id="premium/model",
                name="Premium",
                context_length=10000,
                pricing={"prompt": "0.00001", "completion": "0.00005"},  # $50/MTok
            ),
        ]

        tiers = service.group_by_cost_tier(models)

        assert len(tiers["budget"]) == 1
        assert len(tiers["standard"]) == 1
        assert len(tiers["premium"]) == 1

        assert tiers["budget"][0].id == "budget/model"
        assert tiers["standard"][0].id == "standard/model"
        assert tiers["premium"][0].id == "premium/model"

    def test_group_by_cost_tier_empty(self):
        """Test grouping with empty model list.

        Verifies that empty model list results in empty tier groups.
        """
        service = OpenRouterService(api_key="test-key")

        tiers = service.group_by_cost_tier([])

        assert len(tiers["budget"]) == 0
        assert len(tiers["standard"]) == 0
        assert len(tiers["premium"]) == 0

    def test_group_by_cost_tier_single_tier(self):
        """Test grouping with models in single tier.

        Verifies that grouping works when all models are in same tier.
        Note: pricing is per-token, not per-MTok.
        """
        service = OpenRouterService(api_key="test-key")

        models = [
            OpenRouterModel(
                id="budget1/model",
                name="Budget 1",
                context_length=10000,
                pricing={"prompt": "0.0000001", "completion": "0.0000005"},  # $0.5/MTok
            ),
            OpenRouterModel(
                id="budget2/model",
                name="Budget 2",
                context_length=10000,
                pricing={"prompt": "0.0000002", "completion": "0.0000008"},  # $0.8/MTok
            ),
        ]

        tiers = service.group_by_cost_tier(models)

        assert len(tiers["budget"]) == 2
        assert len(tiers["standard"]) == 0
        assert len(tiers["premium"]) == 0
