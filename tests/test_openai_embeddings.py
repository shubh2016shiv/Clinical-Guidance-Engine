"""Exercise OpenAITextEmbeddingClient against a mocked SDK client."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.adapters.llm_provider.exceptions import LLMProviderError, LLMProviderExecutionError
from src.adapters.llm_provider.openai import OpenAITextEmbeddingClient

CLIENT = "src.adapters.llm_provider.openai.embeddings.AsyncOpenAI"


def make_embeddings(create):
    with patch(CLIENT) as client_cls:
        client_cls.return_value.embeddings.create = create
        return OpenAITextEmbeddingClient("text-embedding-3-small", 3, api_key="test-key")


class OpenAITextEmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_requests_model_and_dimensions_and_returns_sdk_vector(self) -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 0.0, 4.0])])
        )
        embeddings = make_embeddings(create)

        vector = await embeddings.create_text_embedding("metformin")

        create.assert_awaited_once_with(
            model="text-embedding-3-small", input="metformin", dimensions=3
        )
        self.assertEqual(vector, [3.0, 0.0, 4.0])

    async def test_close_releases_sdk_client(self) -> None:
        embeddings = make_embeddings(AsyncMock())
        embeddings._openai_client.close = AsyncMock()

        await embeddings.close()

        embeddings._openai_client.close.assert_awaited_once()

    async def test_embed_wraps_api_failure(self) -> None:
        embeddings = make_embeddings(AsyncMock(side_effect=RuntimeError("rate limited")))

        with self.assertRaises(LLMProviderExecutionError):
            await embeddings.create_text_embedding("metformin")

    def test_missing_api_key_is_rejected(self) -> None:
        with (
            patch(CLIENT),
            patch(
                "src.adapters.llm_provider.openai.embeddings.get_settings",
                return_value=SimpleNamespace(openai_api_key=""),
            ),
        ):
            with self.assertRaises(LLMProviderError):
                OpenAITextEmbeddingClient("text-embedding-3-small", 3)


if __name__ == "__main__":
    unittest.main()
