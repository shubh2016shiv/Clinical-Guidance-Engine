"""
Abstract Embedding Provider Interface

This module provides an abstract interface for embedding generation providers,
allowing easy extension to support different embedding models.

The interface supports:
- Single text embedding generation
- Batch embedding generation
- Both sync and async operations
- Flexible configuration management
- Dimension management
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.providers.exceptions import EmbeddingProviderError
from src.logs import get_logger

logger = get_logger(__name__)


class EmbeddingTaskType(Enum):
    """
    Types of embedding tasks for different use cases

    Different task types may optimize embeddings for specific purposes
    """

    SEMANTIC_SIMILARITY = "semantic_similarity"  # Default - general similarity
    CLASSIFICATION = "classification"  # For classification tasks
    CLUSTERING = "clustering"  # For clustering operations
    RETRIEVAL_QUERY = "retrieval_query"  # Query in retrieval systems
    RETRIEVAL_DOCUMENT = "retrieval_document"  # Document in retrieval systems


@dataclass
class EmbeddingConfig:
    """
    Configuration container for embedding provider settings

    This dataclass centralizes all configuration options for embedding providers,
    making it easy to manage and extend settings.
    """

    # Model configuration
    model_name: str
    embedding_dimension: int = 1536  # Default dimension (OpenAI text-embedding-3-small)

    # Task configuration
    task_type: EmbeddingTaskType = EmbeddingTaskType.SEMANTIC_SIMILARITY

    # Batch processing
    batch_size: int = 100  # Maximum texts per batch request

    # Performance settings
    timeout: int = 60  # Request timeout in seconds
    max_retries: int = 3  # Retry attempts for failed requests
    retry_delay: float = 1.0  # Initial retry delay in seconds

    # Text preprocessing
    strip_whitespace: bool = True  # Remove leading/trailing whitespace
    remove_empty: bool = True  # Filter out empty texts in batch
    max_text_length: Optional[int] = None  # Maximum text length (characters)

    # Additional provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/serialization"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "task_type": self.task_type.value,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class EmbeddingResponse:
    """
    Standardized response object from embedding providers

    This class ensures consistent response structure across different
    provider implementations.
    """

    embeddings: List[List[float]]  # List of embedding vectors
    model: str  # Model used for generation
    provider_type: str  # Provider identifier
    dimension: int  # Embedding dimension

    # Usage statistics (if available)
    total_tokens: Optional[int] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "embeddings_count": len(self.embeddings),
            "model": self.model,
            "provider_type": self.provider_type,
            "dimension": self.dimension,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }

    def get_single_embedding(self) -> List[float]:
        """
        Get the first embedding from the response

        Useful when generating embedding for a single text

        Returns:
            First embedding vector

        Raises:
            EmbeddingProviderError: If no embeddings exist
        """
        if not self.embeddings:
            raise EmbeddingProviderError("No embeddings in response")
        return self.embeddings[0]


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers

    This class defines the interface that all embedding provider implementations
    must follow, ensuring consistency and interchangeability.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding provider with configuration

        Args:
            config: EmbeddingConfig object containing provider settings
        """
        self.config = config
        self.model_name = config.model_name
        self.embedding_dimension = config.embedding_dimension
        self.provider_type = (
            self.__class__.__name__.lower()
            .replace("provider", "")
            .replace("embedding", "")
        )

        logger.info(
            f"Initialized {self.provider_type} embedding provider - "
            f"Model: {self.model_name}, Dimension: {self.embedding_dimension}"
        )

    @abstractmethod
    async def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate an embedding for a single text (async)

        This is the primary method for generating embeddings asynchronously.

        Args:
            text: Text to generate embedding for
            **kwargs: Additional provider-specific parameters

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        pass

    @abstractmethod
    def generate_embedding_sync(self, text: str, **kwargs) -> List[float]:
        """
        Generate an embedding for a single text (synchronous)

        This method is useful for non-async contexts or when running
        in synchronous code.

        Args:
            text: Text to generate embedding for
            **kwargs: Additional provider-specific parameters

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        pass

    @abstractmethod
    async def generate_embeddings_batch(
        self, texts: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (async)

        This method is optimized for batch processing and should be used
        when embedding multiple texts to improve efficiency.

        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional provider-specific parameters

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        pass

    @abstractmethod
    def generate_embeddings_batch_sync(
        self, texts: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (synchronous)

        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional provider-specific parameters

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingProviderError: If embedding generation fails
        """
        pass

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding generation

        Applies configuration-based preprocessing like whitespace stripping
        and length limiting.

        Args:
            text: Raw text input

        Returns:
            Preprocessed text

        Raises:
            EmbeddingProviderError: If text is invalid after preprocessing
        """
        # Strip whitespace if configured
        if self.config.strip_whitespace:
            text = text.strip()

        # Check for empty text
        if not text and self.config.remove_empty:
            raise EmbeddingProviderError("Text is empty after preprocessing")

        # Limit text length if configured
        if self.config.max_text_length and len(text) > self.config.max_text_length:
            logger.warning(
                f"Text length ({len(text)}) exceeds maximum ({self.config.max_text_length}), truncating"
            )
            text = text[: self.config.max_text_length]

        return text

    def preprocess_texts_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts for batch embedding

        Args:
            texts: List of raw text inputs

        Returns:
            List of preprocessed texts (may be shorter if empties are removed)
        """
        preprocessed = []

        for i, text in enumerate(texts):
            try:
                processed = self.preprocess_text(text)
                preprocessed.append(processed)
            except EmbeddingProviderError as e:
                if self.config.remove_empty:
                    logger.debug(f"Skipping text at index {i}: {e}")
                    continue
                else:
                    raise

        return preprocessed

    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate that an embedding has the expected dimension

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if valid, False otherwise
        """
        if not embedding:
            logger.warning("Embedding is empty")
            return False

        if len(embedding) != self.embedding_dimension:
            logger.warning(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(embedding)}"
            )
            return False

        return True

    def update_config(self, **kwargs) -> None:
        """
        Update provider configuration dynamically

        This allows runtime modification of settings without recreating
        the provider instance.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Attempted to set unknown config key: {key}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the embedding provider

        Returns:
            Dictionary with provider information and current configuration
        """
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "task_type": self.config.task_type.value,
            "config": self.config.to_dict(),
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the text

        This is a rough approximation. For production use, consider
        using provider-specific tokenizers.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: ~1 token per 4 characters
        return len(text) // 4

    def estimate_batch_tokens(self, texts: List[str]) -> int:
        """
        Estimate total tokens for a batch of texts

        Args:
            texts: List of texts

        Returns:
            Total estimated token count
        """
        return sum(self.estimate_tokens(text) for text in texts)
