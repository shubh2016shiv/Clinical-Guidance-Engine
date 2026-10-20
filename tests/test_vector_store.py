"""Exercise the vector store adapters against mocked clients."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pymilvus import DataType

from src.adapters.vector_store.base import VectorDBConfig
from src.adapters.vector_store.exceptions import (
    VectorDBCollectionError,
    VectorDBConnectionError,
    VectorDBDimensionMismatchError,
    VectorDBSearchError,
)
from src.adapters.vector_store.milvus_vector_store import MilvusProvider
from src.adapters.vector_store.openai_vector_store import VectorStoreManager
from src.response_api_agent.managers.exceptions import VectorStoreError

MILVUS = "src.adapters.vector_store.milvus_vector_store"


def make_config(dimension=3):
    config = VectorDBConfig(embedding_dimension=dimension)
    config.collection.embedding_dimension = dimension
    config.collection.load_on_startup = False
    return config


def make_provider(dimension=3, embeddings=None):
    embeddings = embeddings or MagicMock(dimensions=dimension)
    return MilvusProvider(make_config(dimension), embeddings=embeddings)


def make_collection(hits):
    collection = MagicMock()
    field_names = ("id", "embedding", "drug_name")
    collection.schema.fields = [SimpleNamespace(name=name) for name in field_names]
    collection.search.return_value = [hits]
    return collection


def make_hit(hit_id, distance, **entity):
    return SimpleNamespace(id=hit_id, distance=distance, entity=SimpleNamespace(get=entity.get))


class MilvusProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_collection_check_rejects_wrong_vector_dimension(self) -> None:
        provider = make_provider(dimension=3)
        collection = MagicMock()
        collection.schema.fields = [SimpleNamespace(dtype=DataType.FLOAT_VECTOR, params={"dim": 5})]
        provider._is_connected = True
        provider._collection = collection

        with self.assertRaises(VectorDBDimensionMismatchError):
            await provider.check_collection()

    async def test_search_maps_hits_and_drops_invalid_output_fields(self) -> None:
        provider = make_provider()
        collection = make_collection([make_hit(7, 0.9, drug_name="metformin")])
        provider._is_connected = True
        provider._collection = collection

        results = await provider.search([0.1, 0.2, 0.3], output_fields=["drug_name", "id", "bogus"])

        self.assertEqual((results[0].id, results[0].score), (7, 0.9))
        self.assertEqual(results[0].fields, {"drug_name": "metformin"})
        self.assertEqual(collection.search.call_args.kwargs["output_fields"], ["drug_name"])

    async def test_search_uses_default_ef_when_config_has_none(self) -> None:
        provider = make_provider()
        collection = make_collection([])
        provider._is_connected = True
        provider._collection = collection

        await provider.search([0.1, 0.2, 0.3])

        self.assertEqual(collection.search.call_args.kwargs["param"], {"ef": 64})

    async def test_search_rejects_wrong_dimension(self) -> None:
        provider = make_provider()
        with self.assertRaises(VectorDBDimensionMismatchError):
            await provider.search([0.1, 0.2])

    async def test_search_wraps_milvus_failure(self) -> None:
        provider = make_provider()
        collection = make_collection([])
        collection.search.side_effect = RuntimeError("boom")
        provider._is_connected = True
        provider._collection = collection

        with self.assertRaises(VectorDBSearchError):
            await provider.search([0.1, 0.2, 0.3])

    async def test_search_by_text_embeds_then_searches(self) -> None:
        embeddings = MagicMock(dimensions=3)
        embeddings.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        provider = make_provider(embeddings=embeddings)
        collection = make_collection([make_hit(1, 0.5)])
        provider._is_connected = True
        provider._collection = collection

        results = await provider.search_by_text("metformin", limit=2)

        embeddings.embed.assert_awaited_once_with("metformin")
        self.assertEqual(len(results), 1)
        self.assertEqual(collection.search.call_args.kwargs["limit"], 2)

    async def test_search_by_text_wraps_embedding_failure(self) -> None:
        embeddings = MagicMock(dimensions=3)
        embeddings.embed = AsyncMock(side_effect=RuntimeError("api down"))
        provider = make_provider(embeddings=embeddings)

        with self.assertRaises(VectorDBSearchError):
            await provider.search_by_text("metformin")

    def test_constructor_rejects_embedding_dimension_mismatch(self) -> None:
        with self.assertRaises(VectorDBDimensionMismatchError):
            MilvusProvider(make_config(dimension=3), embeddings=MagicMock(dimensions=5))

    def test_constructor_builds_openai_embeddings_from_settings_by_default(self) -> None:
        with patch(f"{MILVUS}.OpenAIEmbeddings") as embeddings_cls:
            provider = MilvusProvider(make_config(dimension=3))

        model, dimension = embeddings_cls.call_args.args
        self.assertEqual((model, dimension), ("text-embedding-3-small", 3))
        self.assertIs(provider.embeddings, embeddings_cls.return_value)

    def test_connect_reports_missing_collection_as_collection_error_not_connection_error(
        self,
    ) -> None:
        provider = make_provider()
        provider.config.collection.load_on_startup = True

        with (
            patch(f"{MILVUS}.connections.connect"),
            patch(f"{MILVUS}.utility.list_collections", return_value=[]),
            patch(f"{MILVUS}.utility.has_collection", return_value=False),
        ):
            with self.assertRaises(VectorDBCollectionError):
                provider._connect()

        self.assertTrue(provider._is_connected)

    def test_connect_wraps_unreachable_server_as_connection_error(self) -> None:
        provider = make_provider()

        with patch(f"{MILVUS}.connections.connect", side_effect=RuntimeError("refused")):
            with self.assertRaises(VectorDBConnectionError):
                provider._connect()

        self.assertFalse(provider._is_connected)

    async def test_collection_lookup_uses_configured_alias(self) -> None:
        provider = make_provider()
        provider.config.connection.alias = "custom"
        provider._is_connected = True

        with patch(f"{MILVUS}.utility.has_collection", return_value=False) as has_collection:
            with self.assertRaises(VectorDBCollectionError):
                provider._ensure_collection()

        self.assertEqual(has_collection.call_args.kwargs["using"], "custom")


OPENAI_STORE = "src.adapters.vector_store.openai_vector_store"


def make_manager():
    with (
        patch(f"{OPENAI_STORE}.OpenAI") as client_cls,
        patch(f"{OPENAI_STORE}.UploadManager") as upload_cls,
    ):
        manager = VectorStoreManager()
    manager.client = client_cls.return_value
    manager.upload_manager = upload_cls.return_value
    manager.upload_manager.upload_directory_to_vector_store = AsyncMock(return_value="task-1")
    manager.upload_manager.wait_for_upload = AsyncMock()
    return manager


class VectorStoreManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_reuses_existing_guidelines_store_with_files(self) -> None:
        manager = make_manager()
        manager.list_vector_stores = AsyncMock(
            return_value=[{"id": "vs_1", "name": "guidelines", "status": "completed"}]
        )
        manager.get_vector_store_files = AsyncMock(
            return_value=[{"id": "f1", "status": "completed"}]
        )

        self.assertEqual(await manager.create_guidelines_vector_store(), "vs_1")
        manager.client.vector_stores.create.assert_not_called()

    async def test_create_makes_new_store_and_starts_upload(self) -> None:
        manager = make_manager()
        manager.list_vector_stores = AsyncMock(return_value=[])
        manager.client.vector_stores.create.return_value = SimpleNamespace(id="vs_new")
        manager.get_clinical_guidelines_directory_path = MagicMock(return_value="/guidelines")

        self.assertEqual(await manager.create_guidelines_vector_store(), "vs_new")
        upload = manager.upload_manager.upload_directory_to_vector_store
        upload.assert_awaited_once_with(
            vector_store_id="vs_new", directory_path="/guidelines", file_extensions=[".pdf", ".md"]
        )
        manager.upload_manager.wait_for_upload.assert_awaited_once_with("task-1")

    async def test_create_fails_before_creating_store_when_directory_missing(self) -> None:
        manager = make_manager()
        manager.list_vector_stores = AsyncMock(return_value=[])
        manager.get_clinical_guidelines_directory_path = MagicMock(
            side_effect=VectorStoreError("missing")
        )

        with self.assertRaises(VectorStoreError):
            await manager.create_guidelines_vector_store()
        manager.client.vector_stores.create.assert_not_called()

    async def test_get_and_delete_report_failure_without_raising(self) -> None:
        manager = make_manager()
        manager.client.vector_stores.retrieve.side_effect = RuntimeError("down")
        manager.client.vector_stores.delete.side_effect = RuntimeError("down")

        self.assertIsNone(await manager.get_vector_store("vs_1"))
        self.assertFalse(await manager.delete_vector_store("vs_1"))


if __name__ == "__main__":
    unittest.main()
