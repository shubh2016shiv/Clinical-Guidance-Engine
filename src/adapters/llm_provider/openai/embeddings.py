"""Create OpenAI text embeddings for Milvus drug search."""

from openai import AsyncOpenAI

from src.config import get_settings

from ..exceptions import LLMProviderInitializationError
from .error_mapping import map_openai_error


# OpenAITextEmbeddingClient: turns a piece of text into a list of numbers
# (an "embedding") that represents its meaning. Needed because Milvus (the
# vector database storing drug records) can only compare numbers, not raw
# text, when searching for similar drugs.
class OpenAITextEmbeddingClient:
    """Create vectors with the model and dimensions used by the drug collection."""

    def __init__(
        self, embedding_model_name: str, dimensions: int, api_key: str | None = None
    ) -> None:
        # Step 1: resolve which API key to use, same pattern as provider.py.
        openai_api_key = api_key or get_settings().openai_api_key
        if not openai_api_key:
            # Step 2: fail immediately and clearly if there is no key at
            # all, instead of letting a confusing error happen on first use.
            raise LLMProviderInitializationError("OpenAI API key is required.")

        self.embedding_model_name = embedding_model_name
        # self.embedding_model_name: which OpenAI embedding model to call.
        self.dimensions = dimensions
        # self.dimensions: how many numbers each embedding vector must have.
        # This MUST match the Milvus collection's configured vector size, or
        # searches will fail or return wrong results.
        self._openai_client = AsyncOpenAI(api_key=openai_api_key)
        # self._openai_client: this client's OWN connection to OpenAI,
        # separate from the one OpenAIProvider uses, since embeddings are
        # requested independently of chat answers.

    async def create_text_embedding(self, input_text: str) -> list[float]:
        """Return OpenAI's embedding vector for ``input_text``."""
        try:
            # Step 1: ask OpenAI to turn the text into a vector, using the
            # exact model and dimension size configured above.
            response = await self._openai_client.embeddings.create(
                model=self.embedding_model_name, input=input_text, dimensions=self.dimensions
            )
            # Step 2: OpenAI's response can technically hold multiple
            # embeddings (one per input); we only sent one piece of text,
            # so we take the first (and only) result.
            return response.data[0].embedding
        except Exception as exc:
            # Translate any OpenAI SDK failure into our own exception
            # hierarchy, same pattern as chat_completions.py and
            # responses_api.py.
            raise map_openai_error(exc, "Embedding request failed") from exc

    async def close(self) -> None:
        """Close the SDK client when its owner shuts down."""
        # Always call this when done (e.g. in a "finally" block) to release
        # the network connection cleanly.
        await self._openai_client.close()
