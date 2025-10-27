"""
OpenAI Embedding Provider Implementation

This module provides a comprehensive implementation of the EmbeddingProvider interface
using OpenAI's embedding models.

Supported Models:
- text-embedding-3-small (1536 dimensions, cost-effective)
- text-embedding-3-large (3072 dimensions, higher quality)
- text-embedding-ada-002 (1536 dimensions, legacy)
"""

from typing import List, Optional, Dict, Any
from openai import OpenAI, AsyncOpenAI

from src.config import get_settings
from src.providers.exceptions import EmbeddingProviderError
from src.logs import get_logger
from .base import EmbeddingProvider, EmbeddingConfig

logger = get_logger(__name__)


# Model dimension mapping
OPENAI_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider implementation

    This provider supports all OpenAI embedding models with both
    synchronous and asynchronous operations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
        model_name: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI embedding provider

        Args:
            api_key: OpenAI API key (defaults to settings)
            config: Complete EmbeddingConfig object (takes precedence)
            model_name: Model name (used if config not provided)
            embedding_dimension: Embedding dimension (auto-detected if not provided)
            **kwargs: Additional configuration parameters

        Raises:
            EmbeddingProviderError: If API key is missing or configuration is invalid
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise EmbeddingProviderError("OpenAI API key is required")

        # Determine model name
        if config:
            model_name = config.model_name
        else:
            model_name = model_name or "text-embedding-3-small"

        # Auto-detect dimension if not provided
        if not embedding_dimension and not config:
            embedding_dimension = OPENAI_MODEL_DIMENSIONS.get(model_name, 1536)
            logger.debug(
                f"Auto-detected dimension for {model_name}: {embedding_dimension}"
            )

        # Build configuration if not provided
        if config is None:
            config = EmbeddingConfig(
                model_name=model_name,
                embedding_dimension=embedding_dimension or 1536,
                **kwargs,
            )

        # Initialize parent
        super().__init__(config)

        # Initialize OpenAI clients (both sync and async)
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

        logger.info(
            f"OpenAI embedding provider initialized - "
            f"Model: {self.model_name}, Dimension: {self.embedding_dimension}"
        )

    async def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate an embedding for a single text (async)

        Args:
            text: Text to generate embedding for
            **kwargs: Additional parameters (e.g., dimensions, user, encoding_format)

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
            response = await self.async_client.embeddings.create(**params)

            # Extract embedding
            if not response.data:
                raise EmbeddingProviderError("No embedding data in response")

            embedding = response.data[0].embedding

            # Validate embedding
            if not self.validate_embedding(embedding):
                raise EmbeddingProviderError("Invalid embedding dimension")

            logger.debug(
                f"Generated OpenAI embedding - "
                f"Text length: {len(text)}, Dimension: {len(embedding)}"
            )

            return embedding

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate OpenAI embedding: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg)

    def generate_embedding_sync(self, text: str, **kwargs) -> List[float]:
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

            # Execute API call (synchronous)
            response = self.client.embeddings.create(**params)

            # Extract embedding
            if not response.data:
                raise EmbeddingProviderError("No embedding data in response")

            embedding = response.data[0].embedding

            # Validate embedding
            if not self.validate_embedding(embedding):
                raise EmbeddingProviderError("Invalid embedding dimension")

            logger.debug(
                f"Generated OpenAI embedding (sync) - "
                f"Text length: {len(text)}, Dimension: {len(embedding)}"
            )

            return embedding

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate OpenAI embedding (sync): {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg)

    async def generate_embeddings_batch(
        self, texts: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (async)

        This method optimizes batch processing by sending multiple texts
        in a single API request when possible.

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

            # Check if we need to split into smaller batches
            if len(processed_texts) > self.config.batch_size:
                logger.info(
                    f"Splitting {len(processed_texts)} texts into batches of "
                    f"{self.config.batch_size}"
                )
                return await self._generate_embeddings_in_batches(
                    processed_texts, **kwargs
                )

            # Build parameters for batch
            params = self._build_embedding_params(processed_texts, **kwargs)

            # Execute API call
            response = await self.async_client.embeddings.create(**params)

            # Extract embeddings
            if not response.data:
                raise EmbeddingProviderError("No embedding data in response")

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in sorted_data]

            # Validate embeddings
            for i, embedding in enumerate(embeddings):
                if not self.validate_embedding(embedding):
                    raise EmbeddingProviderError(
                        f"Invalid embedding dimension at index {i}"
                    )

            logger.debug(
                f"Generated {len(embeddings)} OpenAI embeddings - "
                f"Dimension: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate OpenAI embeddings batch: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg)

    def generate_embeddings_batch_sync(
        self, texts: List[str], **kwargs
    ) -> List[List[float]]:
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

            # For sync version, process in single batch (or implement splitting if needed)
            params = self._build_embedding_params(processed_texts, **kwargs)

            # Execute API call (synchronous)
            response = self.client.embeddings.create(**params)

            # Extract embeddings
            if not response.data:
                raise EmbeddingProviderError("No embedding data in response")

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in sorted_data]

            # Validate embeddings
            for i, embedding in enumerate(embeddings):
                if not self.validate_embedding(embedding):
                    raise EmbeddingProviderError(
                        f"Invalid embedding dimension at index {i}"
                    )

            logger.debug(
                f"Generated {len(embeddings)} OpenAI embeddings (sync) - "
                f"Dimension: {len(embeddings[0]) if embeddings else 0}"
            )

            return embeddings

        except EmbeddingProviderError:
            raise
        except Exception as e:
            error_msg = f"Failed to generate OpenAI embeddings batch (sync): {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise EmbeddingProviderError(error_msg)

    async def _generate_embeddings_in_batches(
        self, texts: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings by splitting into smaller batches

        This method handles large text lists by processing them in
        manageable chunks.

        Args:
            texts: List of texts to process
            **kwargs: Additional parameters

        Returns:
            Complete list of embeddings
        """
        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} texts")

            batch_embeddings = await self.generate_embeddings_batch(batch, **kwargs)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _build_embedding_params(
        self,
        input_text: Any,  # Can be str or List[str]
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build parameters dictionary for OpenAI embeddings API

        Args:
            input_text: Single text or list of texts
            **kwargs: Additional parameter overrides

        Returns:
            Complete parameters dictionary for API call
        """
        params = {
            "model": kwargs.get("model", self.model_name),
            "input": input_text,
        }

        # Add optional parameters
        if "dimensions" in kwargs:
            params["dimensions"] = kwargs["dimensions"]
        elif "dimensions" in self.config.extra_params:
            params["dimensions"] = self.config.extra_params["dimensions"]

        if "user" in kwargs:
            params["user"] = kwargs["user"]

        if "encoding_format" in kwargs:
            params["encoding_format"] = kwargs["encoding_format"]

        return params

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the OpenAI embedding model

        Returns:
            Dictionary with model information
        """
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "task_type": self.config.task_type.value,
            "batch_size": self.config.batch_size,
            "supported_models": list(OPENAI_MODEL_DIMENSIONS.keys()),
            "api_key_configured": bool(self.api_key),
        }
