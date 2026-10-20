"""OpenAI text embeddings."""

import math

from openai import AsyncOpenAI

from src.config import get_settings

from ..exceptions import LLMProviderError, LLMProviderExecutionError


class OpenAIEmbeddings:
    """Embed text with an OpenAI embedding model into unit-length vectors.

    ``dimensions`` asks the API for shortened vectors. It must equal the vector
    size of the collection being searched, and the collection must have been
    filled with the same model and dimensions.
    """

    def __init__(self, model: str, dimensions: int, api_key: str | None = None) -> None:
        api_key = api_key or get_settings().openai_api_key
        if not api_key:
            raise LLMProviderError("OpenAI API key is required.")

        self.model = model
        self.dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed(self, text: str) -> list[float]:
        """Return the embedding of ``text``. Raises LLMProviderExecutionError on failure."""
        try:
            response = await self._client.embeddings.create(
                model=self.model, input=text, dimensions=self.dimensions
            )
        except Exception as exc:
            raise LLMProviderExecutionError(f"Embedding request failed: {exc}") from exc

        vector = response.data[0].embedding
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
