"""
Gemini Embedding Provider Implementation

This module provides a comprehensive implementation of the EmbeddingProvider interface
using Google's Gemini embedding models.

Supported Models:
- text-embedding-004 (768 dimensions, latest model)
- embedding-001 (768 dimensions, legacy)
"""

import asyncio
import math
from typing import Any

import google.generativeai as genai

from src.config import get_settings
from src.logs import get_logger
from src.providers.exceptions import EmbeddingProviderError

from .base import EmbeddingConfig, EmbeddingProvider, EmbeddingTaskType

logger = get_logger(__name__)


# Model dimension mapping
GEMINI_MODEL_DIMENSIONS = {
    "text-embedding-004": 768,
    "embedding-001": 768,
    "models/text-embedding-004": 768,
    "models/embedding-001": 768,
}

# Task type mapping to Gemini API task types
GEMINI_TASK_TYPE_MAPPING = {
    EmbeddingTaskType.SEMANTIC_SIMILARITY: "SEMANTIC_SIMILARITY",
    EmbeddingTaskType.CLASSIFICATION: "CLASSIFICATION",
    EmbeddingTaskType.CLUSTERING: "CLUSTERING",
    EmbeddingTaskType.RETRIEVAL_QUERY: "RETRIEVAL_QUERY",
    EmbeddingTaskType.RETRIEVAL_DOCUMENT: "RETRIEVAL_DOCUMENT",
}


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Gemini embedding provider implementation

    This provider supports Google's Gemini embedding models with both
    synchronous and asynchronous operations.
    """

    def __init__(self, config: EmbeddingConfig, api_key: str | None = None):
        """
        Initialize the Gemini embedding provider.

        Args:
            config: Fully-resolved EmbeddingConfig object (built by the factory).
            api_key: Gemini API key. Defaults to the application settings.

        Raises:
            EmbeddingProviderError: If the API key is missing.
        """
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key

        if not self.api_key:
            raise EmbeddingProviderError("Gemini API key is required")

        # Configure Gemini SDK
        genai.configure(api_key=self.api_key)  # pyright: ignore[reportPrivateImportUsage]

        # Ensure model name has the required "models/" prefix for API calls
        model_name = config.model_name
        if not model_name.startswith("models/"):
            self.full_model_name = f"models/{model_name}"
        else:
            self.full_model_name = model_name

        # Initialize parent
        super().__init__(config, api_key=self.api_key)

        logger.info(
            f"Gemini embedding provider initialized - "
            f"Model: {self.model_name}, Dimension: {self.embedding_dimension}, "
            f"Task: {self.config.task_type.value}"
        )

    async def generate_embedding(self, text: str, **kwargs) -> list[float]:
        """
        Generate an embedding for a single text (async)

        Note: Gemini SDK doesn't have native async support, so we use
        asyncio.to_thread to run synchronously in a thread pool.

        Args:
            text: Text to generate embedding for
            **kwargs: Additional parameters (e.g., task_type, title)

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        try:
            # Preprocess text
            text = self.preprocess_text(text)

            # Build parameters
            params = self._build_embedding_params(text, **kwargs)

            # Execute in thread pool (Gemini SDK is synchronous)
            response = await asyncio.to_thread(genai.embed_content, **params)  # pyright: ignore[reportPrivateImportUsage]

            # Extract embedding
            if not response or "embedding" not in response:
                raise EmbeddingProviderError("No embedding in response")

            embedding = response["embedding"]

            # Validate embedding
            if not self.validate_embedding(embedding):
                raise EmbeddingProviderError("Invalid embedding dimension")

            # Normalize embedding for similarity search consistency
            # This ensures query embeddings match the normalized embeddings in the database
            embedding = self._normalize_vector(embedding)

            logger.info(
                f"[DEBUG] Generated Gemini embedding - "
                f"Text length: {len(text)}, Dimension: {len(embedding)}"
            )

            return embedding

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate Gemini embedding: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg) from e

    def generate_embedding_sync(self, text: str, **kwargs) -> list[float]:
        """
        Generate an embedding for a single text (synchronous)

        Args:
            text: Text to generate embedding for
            **kwargs: Additional parameters

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        try:
            # Preprocess text
            text = self.preprocess_text(text)

            # Build parameters
            params = self._build_embedding_params(text, **kwargs)

            # Execute API call
            response = genai.embed_content(**params)  # pyright: ignore[reportPrivateImportUsage]

            # Extract embedding
            if not response or "embedding" not in response:
                raise EmbeddingProviderError("No embedding in response")

            embedding = response["embedding"]

            # Validate embedding
            if not self.validate_embedding(embedding):
                raise EmbeddingProviderError("Invalid embedding dimension")

            # Normalize embedding for similarity search consistency
            # This ensures query embeddings match the normalized embeddings in the database
            embedding = self._normalize_vector(embedding)

            logger.debug(
                f"Generated Gemini embedding (sync) - "
                f"Text length: {len(text)}, Dimension: {len(embedding)}"
            )

            return embedding

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate Gemini embedding (sync): {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg) from e

    async def generate_embeddings_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (async)

        Note: Gemini API processes each text separately, so we batch them
        efficiently using asyncio.

        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        try:
            if not texts:
                return []

            # Preprocess texts
            processed_texts = self.preprocess_texts_batch(texts)

            if not processed_texts:
                raise EmbeddingProviderError("No valid texts after preprocessing")

            # Process in batches to avoid overwhelming the API
            embeddings = []
            batch_size = self.config.batch_size

            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i : i + batch_size]
                logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} texts")

                # Generate embeddings for batch
                batch_embeddings = await self._generate_batch_async(batch, **kwargs)
                embeddings.extend(batch_embeddings)

            logger.debug(
                f"Generated {len(embeddings)} Gemini embeddings - "
                f"Dimension: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate Gemini embeddings batch: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg) from e

    def generate_embeddings_batch_sync(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (synchronous)

        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional parameters

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        try:
            if not texts:
                return []

            # Preprocess texts
            processed_texts = self.preprocess_texts_batch(texts)

            if not processed_texts:
                raise EmbeddingProviderError("No valid texts after preprocessing")

            # Generate embeddings sequentially
            embeddings = []
            for text in processed_texts:
                embedding = self.generate_embedding_sync(text, **kwargs)
                embeddings.append(embedding)

            logger.debug(
                f"Generated {len(embeddings)} Gemini embeddings (sync) - "
                f"Dimension: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate Gemini embeddings batch (sync): {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg) from e

    async def _generate_batch_async(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        Generate embeddings for a batch using async concurrency

        Args:
            texts: List of texts to process
            **kwargs: Additional parameters

        Returns:
            List of embeddings
        """
        # Create tasks for concurrent execution
        tasks = [self.generate_embedding(text, **kwargs) for text in texts]

        # Execute concurrently
        embeddings = await asyncio.gather(*tasks)
        return embeddings

    def _build_embedding_params(self, text: str, **kwargs) -> dict[str, Any]:
        """
        Build parameters dictionary for Gemini embeddings API

        Args:
            text: Text to embed
            **kwargs: Additional parameter overrides

        Returns:
            Complete parameters dictionary for API call
        """
        # Get task type
        task_type = kwargs.get("task_type", self.config.task_type)
        gemini_task_type = GEMINI_TASK_TYPE_MAPPING.get(task_type, "SEMANTIC_SIMILARITY")

        params = {
            "model": self.full_model_name,
            "content": text,
            "task_type": gemini_task_type,
        }

        # Add optional title (useful for retrieval tasks)
        if "title" in kwargs:
            params["title"] = kwargs["title"]
        elif "title" in self.config.extra_params:
            params["title"] = self.config.extra_params["title"]

        # Add output dimensionality if supported and specified
        if "output_dimensionality" in kwargs:
            params["output_dimensionality"] = kwargs["output_dimensionality"]

        return params

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """
        Normalize a vector to unit length (L2 norm).

        This is critical for similarity search when using Inner Product (IP) metric,
        as it makes IP equivalent to cosine similarity. Both query and database
        embeddings must be normalized for accurate similarity calculations.

        Args:
            vector: Embedding vector as list of floats

        Returns:
            Normalized vector with unit length (magnitude = 1.0)
        """
        # Calculate L2 norm
        magnitude = math.sqrt(sum(x * x for x in vector))

        # Avoid division by zero (return zero vector if magnitude is 0)
        if magnitude == 0:
            logger.warning("Attempted to normalize zero vector, returning as-is")
            return vector

        # Normalize to unit length
        normalized = [x / magnitude for x in vector]

        # Verify normalization (should be ~1.0)
        normalized_magnitude = math.sqrt(sum(x * x for x in normalized))
        if abs(normalized_magnitude - 1.0) > 0.01:  # Allow small floating point errors
            logger.warning(
                f"Vector normalization check failed: magnitude={normalized_magnitude:.6f}"
            )

        return normalized

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the Gemini embedding model

        Returns:
            Dictionary with model information
        """
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "full_model_name": self.full_model_name,
            "embedding_dimension": self.embedding_dimension,
            "task_type": self.config.task_type.value,
            "batch_size": self.config.batch_size,
            "supported_models": list(GEMINI_MODEL_DIMENSIONS.keys()),
            "supported_task_types": [t.value for t in EmbeddingTaskType],
            "api_key_configured": bool(self.api_key),
        }
