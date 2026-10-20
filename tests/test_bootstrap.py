"""Check startup decisions without contacting external services."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.bootstrap.check_redis_and_milvus import check_milvus
from src.bootstrap.check_settings import check_settings


class StartupChecksTests(unittest.IsolatedAsyncioTestCase):
    def test_openai_key_is_required_before_creating_clients(self) -> None:
        settings = SimpleNamespace(openai_api_key=" ", openai_model_name="gpt-4o")

        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            check_settings(settings, use_drug_database=False, use_clinical_guidelines=False)

    async def test_failed_milvus_check_disables_drug_search(self) -> None:
        drug_data = MagicMock()
        drug_data.connect = AsyncMock()
        drug_data.disconnect = AsyncMock()
        drug_data.vector_db_provider.check_collection = AsyncMock(
            side_effect=RuntimeError("wrong dimension")
        )
        manager = SimpleNamespace(drug_data_manager=drug_data)
        agent = SimpleNamespace(response_manager=manager)

        self.assertFalse(await check_milvus(agent))
        drug_data.disconnect.assert_awaited_once()
        self.assertIsNone(manager.drug_data_manager)
