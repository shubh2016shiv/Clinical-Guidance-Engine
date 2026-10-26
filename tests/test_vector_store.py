"""Exercise the vector store adapters against mocked clients."""

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
from pymilvus import DataType

from src.adapters.vector_store.base import VectorDBConfig
from src.adapters.vector_store.exceptions import (
    VectorDBAuthenticationError,
    VectorDBCollectionError,
    VectorDBConfigError,
    VectorDBConnectionError,
    VectorDBDimensionMismatchError,
    VectorDBProviderError,
    VectorDBRateLimitError,
    VectorDBSearchError,
)
from src.adapters.vector_store.milvus_vector_store import MilvusProvider
from src.adapters.vector_store.openai_vector_store import OpenAIVectorStoreClient

MILVUS = "src.adapters.vector_store.milvus_vector_store"


def make_config(dimension=3):
    config = VectorDBConfig(embedding_dimension=dimension)
    config.collection.embedding_dimension = dimension
    config.collection.load_on_startup = False
    return config


def make_provider(dimension=3, embeddings=None):
    embeddings = embeddings or MagicMock(dimensions=dimension)
    return MilvusProvider(make_config(dimension), embeddings=embeddings)


def make_client(hits):
    client = MagicMock()
    client.describe_collection.return_value = {
        "fields": [
            {"name": "id", "type": DataType.INT64},
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 3}},
            {"name": "drug_name", "type": DataType.VARCHAR},
        ],
        "functions": [],
    }
    client.search.return_value = [hits]
    client.has_collection.return_value = True
    return client


def make_hit(hit_id, distance, **entity):
    return {"id": hit_id, "distance": distance, "entity": entity}


class MilvusProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_collection_check_rejects_wrong_vector_dimension(self) -> None:
        provider = make_provider(dimension=3)
        client = make_client([])
        client.describe_collection.return_value["fields"][1]["params"]["dim"] = 5
        provider._client = client

        with self.assertRaises(VectorDBDimensionMismatchError):
            await provider.check_collection()

    async def test_search_maps_hits_and_drops_invalid_output_fields(self) -> None:
        provider = make_provider()
        client = make_client([make_hit(7, 0.9, drug_name="metformin")])
        provider._client = client

        results = await provider.search([0.1, 0.2, 0.3], output_fields=["drug_name", "id", "bogus"])

        self.assertEqual((results[0].id, results[0].score), (7, 0.9))
        self.assertEqual(results[0].fields, {"drug_name": "metformin"})
        self.assertEqual(client.search.call_args.kwargs["output_fields"], ["drug_name"])

    async def test_search_uses_default_ef_when_config_has_none(self) -> None:
        provider = make_provider()
        client = make_client([])
        provider._client = client

        await provider.search([0.1, 0.2, 0.3])

        self.assertEqual(client.search.call_args.kwargs["search_params"], {"ef": 64})

    async def test_search_rejects_wrong_dimension(self) -> None:
        provider = make_provider()
        with self.assertRaises(VectorDBDimensionMismatchError):
            await provider.search([0.1, 0.2])

    async def test_search_wraps_milvus_failure(self) -> None:
        provider = make_provider()
        client = make_client([])
        client.search.side_effect = RuntimeError("boom")
        provider._client = client

        with self.assertRaises(VectorDBSearchError):
            await provider.search([0.1, 0.2, 0.3])

    async def test_search_by_text_embeds_then_searches(self) -> None:
        embeddings = MagicMock(dimensions=3)
        embeddings.create_text_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        provider = make_provider(embeddings=embeddings)
        client = make_client([make_hit(1, 0.5)])
        provider._client = client

        results = await provider.search_by_text("metformin", limit=2)

        embeddings.create_text_embedding.assert_awaited_once_with("metformin")
        self.assertEqual(len(results), 1)
        self.assertEqual(client.search.call_args.kwargs["limit"], 2)

    async def test_search_by_text_wraps_embedding_failure(self) -> None:
        embeddings = MagicMock(dimensions=3)
        embeddings.create_text_embedding = AsyncMock(side_effect=RuntimeError("api down"))
        provider = make_provider(embeddings=embeddings)

        with self.assertRaises(VectorDBSearchError):
            await provider.search_by_text("metformin")

    def test_constructor_rejects_embedding_dimension_mismatch(self) -> None:
        with self.assertRaises(VectorDBDimensionMismatchError):
            MilvusProvider(make_config(dimension=3), embeddings=MagicMock(dimensions=5))

    def test_constructor_builds_openai_embeddings_from_settings_by_default(self) -> None:
        with patch(f"{MILVUS}.OpenAITextEmbeddingClient") as embeddings_cls:
            provider = MilvusProvider(make_config(dimension=3))

        model, dimension = embeddings_cls.call_args.args
        self.assertEqual((model, dimension), ("text-embedding-3-small", 3))
        self.assertIs(provider.embeddings, embeddings_cls.return_value)

    async def test_disconnect_closes_only_owned_embedding_client(self) -> None:
        with patch(f"{MILVUS}.OpenAITextEmbeddingClient") as embeddings_cls:
            provider = MilvusProvider(make_config(dimension=3))
        embeddings_cls.return_value.close = AsyncMock()

        await provider.disconnect()

        embeddings_cls.return_value.close.assert_awaited_once()

        borrowed_embeddings = MagicMock(dimensions=3)
        borrowed_embeddings.close = AsyncMock()
        provider = make_provider(embeddings=borrowed_embeddings)
        await provider.disconnect()
        borrowed_embeddings.close.assert_not_awaited()

    def test_connect_reports_missing_collection_as_collection_error_not_connection_error(
        self,
    ) -> None:
        provider = make_provider()
        provider.config.collection.load_on_startup = True

        with (
            patch(f"{MILVUS}.MilvusClient") as client_class,
        ):
            client_class.return_value.has_collection.return_value = False
            with self.assertRaises(VectorDBCollectionError):
                provider._connect()

        self.assertIsNotNone(provider._client)

    def test_connect_wraps_unreachable_server_as_connection_error(self) -> None:
        provider = make_provider()

        with patch(f"{MILVUS}.MilvusClient", side_effect=RuntimeError("refused")):
            with self.assertRaises(VectorDBConnectionError):
                provider._connect()

        self.assertIsNone(provider._client)

    async def test_search_raises_ef_to_match_a_larger_limit(self) -> None:
        """Milvus rejects an HNSW search outright when ef is below the requested limit."""
        provider = make_provider()
        client = make_client([])
        provider._client = client

        await provider.search([0.1, 0.2, 0.3], limit=100)

        self.assertEqual(client.search.call_args.kwargs["search_params"], {"ef": 100})
        self.assertEqual(client.search.call_args.kwargs["limit"], 100)

    async def test_search_raises_ef_even_when_caller_passes_a_small_one(self) -> None:
        provider = make_provider()
        client = make_client([])
        provider._client = client

        await provider.search([0.1, 0.2, 0.3], limit=120, search_params={"ef": 8})

        self.assertEqual(client.search.call_args.kwargs["search_params"], {"ef": 120})

    async def test_search_keeps_an_ef_larger_than_the_limit(self) -> None:
        provider = make_provider()
        client = make_client([])
        provider._client = client

        await provider.search([0.1, 0.2, 0.3], limit=10, search_params={"ef": 512})

        self.assertEqual(client.search.call_args.kwargs["search_params"], {"ef": 512})

    def test_concurrent_connect_creates_a_single_client(self) -> None:
        """Without locking, each thread would build its own client and leak all but one."""
        provider = make_provider()

        def slow_client(**_kwargs):
            # Holding the connection open for a moment widens the window in
            # which every thread has passed the "already connected?" check,
            # which is exactly the race the lock has to close.
            time.sleep(0.05)
            return MagicMock()

        with patch(f"{MILVUS}.MilvusClient", side_effect=slow_client) as client_class:
            threads = [threading.Thread(target=provider._connect) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=15)

        self.assertEqual(client_class.call_count, 1)
        self.assertIsNotNone(provider._client)

    async def test_disconnect_makes_an_owning_provider_refuse_further_work(self) -> None:
        with patch(f"{MILVUS}.OpenAITextEmbeddingClient") as embeddings_cls:
            provider = MilvusProvider(make_config(dimension=3))
        embeddings_cls.return_value.close = AsyncMock()
        embeddings_cls.return_value.create_text_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        await provider.disconnect()

        with self.assertRaises(VectorDBConnectionError):
            await provider.search_by_text("metformin")
        with self.assertRaises(VectorDBConnectionError):
            await provider.connect()
        embeddings_cls.return_value.create_text_embedding.assert_not_awaited()

    async def test_disconnect_leaves_a_borrowing_provider_reusable(self) -> None:
        borrowed = MagicMock(dimensions=3)
        borrowed.close = AsyncMock()
        provider = make_provider(embeddings=borrowed)

        await provider.disconnect()

        self.assertFalse(provider._closed)
        with patch(f"{MILVUS}.MilvusClient"):
            await provider.connect()
        self.assertIsNotNone(provider._client)

    async def test_collection_lookup_uses_configured_name(self) -> None:
        provider = make_provider()
        client = make_client([])
        provider._client = client

        await provider.check_collection()

        client.has_collection.assert_called_with(collection_name=provider.collection_name)
        client.load_collection.assert_called_once_with(collection_name=provider.collection_name)


def make_sdk_error(error_cls):
    """Build a real OpenAI SDK exception with the fields its constructor needs."""
    request = httpx.Request("POST", "https://api.openai.com/v1/vector_stores")
    return error_cls(message="boom", response=httpx.Response(404, request=request), body=None)


class OpenAIVectorStoreClientTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self):
        return OpenAIVectorStoreClient(openai_client=MagicMock())

    async def test_not_found_becomes_collection_error(self) -> None:
        client = self.make_client()
        client.openai_client.vector_stores.retrieve = AsyncMock(
            side_effect=make_sdk_error(openai.NotFoundError)
        )

        with self.assertRaises(VectorDBCollectionError):
            await client.retrieve_vector_store("vs_missing")

    async def test_authentication_failure_becomes_authentication_error(self) -> None:
        client = self.make_client()
        client.openai_client.vector_stores.list = AsyncMock(
            side_effect=make_sdk_error(openai.AuthenticationError)
        )

        with self.assertRaises(VectorDBAuthenticationError):
            await client.list_vector_stores()

    async def test_rate_limit_becomes_rate_limit_error(self) -> None:
        client = self.make_client()
        client.openai_client.vector_stores.create = AsyncMock(
            side_effect=make_sdk_error(openai.RateLimitError)
        )

        with self.assertRaises(VectorDBRateLimitError):
            await client.create_vector_store("guidelines")

    async def test_connection_failure_becomes_connection_error(self) -> None:
        client = self.make_client()
        request = httpx.Request("GET", "https://api.openai.com/v1/vector_stores")
        client.openai_client.vector_stores.list = AsyncMock(
            side_effect=openai.APIConnectionError(request=request)
        )

        with self.assertRaises(VectorDBConnectionError):
            await client.list_vector_stores()

    async def test_unreadable_local_file_becomes_config_error(self) -> None:
        """A missing guideline file used to escape as a bare FileNotFoundError."""
        client = self.make_client()

        with self.assertRaises(VectorDBConfigError):
            await client.upload_files_and_wait("vs_1", [Path("no_such_guideline_file.pdf")])

    async def test_upload_files_and_wait_batches_and_polls_until_done(self) -> None:
        client = self.make_client()
        uploaded = [MagicMock(id="file_1"), MagicMock(id="file_2")]
        client.openai_client.files.create = AsyncMock(side_effect=uploaded)
        in_progress_batch = MagicMock(
            id="batch_1", file_counts=MagicMock(in_progress=1, completed=1, failed=0, total=2)
        )
        done_batch = MagicMock(
            id="batch_1", file_counts=MagicMock(in_progress=0, completed=2, failed=0, total=2)
        )
        client.openai_client.vector_stores.file_batches.create = AsyncMock(
            return_value=in_progress_batch
        )
        client.openai_client.vector_stores.file_batches.retrieve = AsyncMock(
            return_value=done_batch
        )

        with TemporaryDirectory() as directory:
            file_paths = [Path(directory, "a.pdf"), Path(directory, "b.pdf")]
            for file_path in file_paths:
                file_path.write_text("guideline", encoding="utf-8")

            with patch("asyncio.sleep", new=AsyncMock()):
                result = await client.upload_files_and_wait("vs_1", file_paths)

        self.assertEqual(client.openai_client.files.create.await_count, 2)
        client.openai_client.vector_stores.file_batches.create.assert_awaited_once_with(
            vector_store_id="vs_1", file_ids=["file_1", "file_2"]
        )
        client.openai_client.vector_stores.file_batches.retrieve.assert_awaited_once_with(
            "batch_1", vector_store_id="vs_1"
        )
        self.assertIs(result, done_batch)

    async def test_upload_uses_the_display_name_not_the_stored_name(self) -> None:
        """Chainlit stores uploads under generated names; citations must show the real one."""
        client = self.make_client()
        client.openai_client.files.create = AsyncMock(return_value=MagicMock(id="file_1"))
        client.openai_client.vector_stores.file_batches.create = AsyncMock(
            return_value=MagicMock(file_counts=MagicMock(in_progress=0))
        )

        with TemporaryDirectory() as directory:
            stored = Path(directory, "31a98535-8939-47d6.pdf")
            stored.write_text("guideline", encoding="utf-8")
            await client.upload_files_and_wait(
                "vs_1", [stored], display_names={stored: "hypertension.pdf"}
            )

        uploaded_name = client.openai_client.files.create.await_args.kwargs["file"][0]
        self.assertEqual(uploaded_name, "hypertension.pdf")

    async def test_upload_falls_back_to_the_file_name_without_a_display_name(self) -> None:
        client = self.make_client()
        client.openai_client.files.create = AsyncMock(return_value=MagicMock(id="file_1"))
        client.openai_client.vector_stores.file_batches.create = AsyncMock(
            return_value=MagicMock(file_counts=MagicMock(in_progress=0))
        )

        with TemporaryDirectory() as directory:
            guideline = Path(directory, "diabetes.pdf")
            guideline.write_text("guideline", encoding="utf-8")
            await client.upload_files_and_wait("vs_1", [guideline])

        self.assertEqual(
            client.openai_client.files.create.await_args.kwargs["file"][0], "diabetes.pdf"
        )

    async def test_upload_files_and_wait_skips_polling_when_already_done(self) -> None:
        client = self.make_client()
        client.openai_client.files.create = AsyncMock(return_value=MagicMock(id="file_1"))
        done_batch = MagicMock(file_counts=MagicMock(in_progress=0, completed=1, failed=0, total=1))
        client.openai_client.vector_stores.file_batches.create = AsyncMock(return_value=done_batch)
        client.openai_client.vector_stores.file_batches.retrieve = AsyncMock()

        with TemporaryDirectory() as directory:
            file_path = Path(directory, "a.pdf")
            file_path.write_text("guideline", encoding="utf-8")
            result = await client.upload_files_and_wait("vs_1", [file_path])

        client.openai_client.vector_stores.file_batches.retrieve.assert_not_awaited()
        self.assertIs(result, done_batch)

    async def test_unexpected_failure_still_becomes_an_adapter_error(self) -> None:
        client = self.make_client()
        client.openai_client.vector_stores.list = AsyncMock(side_effect=RuntimeError("weird"))

        with self.assertRaises(VectorDBProviderError):
            await client.list_vector_stores()

    async def test_mapped_error_keeps_the_sdk_error_as_cause(self) -> None:
        client = self.make_client()
        sdk_error = make_sdk_error(openai.NotFoundError)
        client.openai_client.vector_stores.retrieve = AsyncMock(side_effect=sdk_error)

        with self.assertRaises(VectorDBProviderError) as caught:
            await client.retrieve_vector_store("vs_missing")

        self.assertIs(caught.exception.__cause__, sdk_error)

    async def test_delete_vector_store_reports_confirmation(self) -> None:
        client = self.make_client()
        client.openai_client.vector_stores.delete = AsyncMock(return_value=MagicMock(deleted=True))

        self.assertTrue(await client.delete_vector_store("vs_1"))
        client.openai_client.vector_stores.delete.assert_awaited_once_with("vs_1")

    async def test_delete_vector_store_reports_an_unconfirmed_deletion(self) -> None:
        client = self.make_client()
        client.openai_client.vector_stores.delete = AsyncMock(return_value=object())

        self.assertFalse(await client.delete_vector_store("vs_1"))

    async def test_close_only_closes_an_owned_client(self) -> None:
        borrowed = MagicMock()
        borrowed.close = AsyncMock()
        await OpenAIVectorStoreClient(openai_client=borrowed).close()
        borrowed.close.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
