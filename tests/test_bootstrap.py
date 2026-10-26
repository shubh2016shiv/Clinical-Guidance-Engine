"""Check startup decisions without contacting external services."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pymilvus import FunctionType

from src.adapters.redis.exceptions import CacheConnectionError
from src.adapters.vector_store.exceptions import VectorDBDimensionMismatchError
from src.bootstrap import bootstrap_services
from src.bootstrap.check_redis_and_milvus import check_milvus, check_redis
from src.bootstrap.check_settings import check_settings


def milvus_provider(count: int) -> MagicMock:
    provider = MagicMock()
    provider.collection_name = "pharmaceutical_drugs"
    provider.connect = AsyncMock()
    provider.disconnect = AsyncMock()
    provider.check_collection = AsyncMock()
    provider.count_entities = AsyncMock(return_value=count)
    bm25 = {"type": FunctionType.BM25, "output_field_names": ["drug_name_sparse"]}
    provider.describe_collection = AsyncMock(return_value={"functions": [bm25]})
    return provider


class StartupChecksTests(unittest.IsolatedAsyncioTestCase):
    def test_openai_key_is_required_before_creating_clients(self) -> None:
        settings = SimpleNamespace(openai_api_key=" ", openai_model_name="gpt-4o")

        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            check_settings(settings, use_drug_database=False, use_clinical_guidelines=False)

    async def test_milvus_is_ready_when_the_collection_has_data(self) -> None:
        self.assertTrue(await check_milvus(milvus_provider(count=6279)))

    async def test_an_empty_drug_collection_is_not_ready(self) -> None:
        provider = milvus_provider(count=0)

        self.assertFalse(await check_milvus(provider))
        provider.disconnect.assert_awaited_once()

    async def test_a_failed_collection_check_is_not_ready(self) -> None:
        provider = milvus_provider(count=10)
        provider.check_collection.side_effect = VectorDBDimensionMismatchError("wrong dimension")

        self.assertFalse(await check_milvus(provider))
        provider.disconnect.assert_awaited_once()

    async def test_an_unexpected_milvus_error_is_not_swallowed(self) -> None:
        provider = milvus_provider(count=10)
        provider.check_collection.side_effect = RuntimeError("bug")

        with self.assertRaises(RuntimeError):
            await check_milvus(provider)

    async def test_redis_is_ready_when_it_connects(self) -> None:
        provider = MagicMock()
        provider.connect = AsyncMock()

        self.assertTrue(await check_redis(provider))

    async def test_redis_is_not_ready_when_the_connection_fails(self) -> None:
        provider = MagicMock()
        provider.connect = AsyncMock(side_effect=CacheConnectionError("refused"))

        self.assertFalse(await check_redis(provider))


class BootstrapServicesTests(unittest.IsolatedAsyncioTestCase):
    async def run_bootstrap(self, *, redis_ok, milvus_ok, guidelines_id, **flags):
        guideline_agent = MagicMock()
        guideline_agent.close = AsyncMock()
        with (
            patch("src.bootstrap.get_settings"),
            patch("src.bootstrap.check_settings"),
            patch("src.bootstrap.RedisCacheConfig"),
            patch("src.bootstrap.RedisProvider"),
            patch("src.bootstrap.MilvusProvider"),
            patch("src.bootstrap.ClinicalGuidelineAgent", return_value=guideline_agent),
            patch("src.bootstrap.check_redis", AsyncMock(return_value=redis_ok)),
            patch("src.bootstrap.check_milvus", AsyncMock(return_value=milvus_ok)),
            patch("src.bootstrap.prepare_guidelines_store", AsyncMock(return_value=guidelines_id)),
        ):
            return await bootstrap_services(**flags), guideline_agent

    async def test_everything_ready_keeps_every_service(self) -> None:
        result, agent = await self.run_bootstrap(
            redis_ok=True, milvus_ok=True, guidelines_id="vs_1"
        )

        self.assertIsNotNone(result.redis_cache)
        self.assertIsNotNone(result.drug_provider)
        self.assertIs(result.guideline_agent, agent)
        self.assertEqual(result.warnings, [])

    async def test_failed_services_are_dropped_with_a_warning_each(self) -> None:
        result, agent = await self.run_bootstrap(
            redis_ok=False, milvus_ok=False, guidelines_id=None
        )

        self.assertIsNone(result.redis_cache)
        self.assertIsNone(result.drug_provider)
        self.assertIsNone(result.guideline_agent)
        self.assertEqual(len(result.warnings), 3)
        agent.close.assert_awaited_once()

    async def test_disabled_features_are_not_created_or_warned_about(self) -> None:
        result, _ = await self.run_bootstrap(
            redis_ok=True,
            milvus_ok=False,
            guidelines_id=None,
            use_drug_database=False,
            use_clinical_guidelines=False,
        )

        self.assertIsNone(result.drug_provider)
        self.assertIsNone(result.guideline_agent)
        self.assertEqual(result.warnings, [])
