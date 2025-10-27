"""
Embedding Provider Module

This module provides a centralized, professional interface for working with
embedding generation providers.

Key Features:
- Multiple provider support (OpenAI, Gemini, etc.)
- Batch and single embedding generation
- Both sync and async operations
- Flexible configuration management
- Type-safe interfaces with dataclasses
- Comprehensive error handling
- Task-specific embeddings

Quick Start:
    from src.providers.embedding_provider import (
        create_embedding_provider,
        EmbeddingTaskType
    )

    # Create a provider
    provider = create_embedding_provider(
        provider_type="openai",
        model_name="text-embedding-3-small",
        task_type=EmbeddingTaskType.SEMANTIC_SIMILARITY
    )

    # Generate embedding
    embedding = await provider.generate_embedding("Your text here")

    # Generate batch
    embeddings = await provider.generate_embeddings_batch([
        "Text 1",
        "Text 2",
        "Text 3"
    ])
"""

# Base classes and interfaces
from .base import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingResponse,
    EmbeddingTaskType,
)

# Factory functions
from .factory import (
    create_embedding_provider,
    get_embedding_provider,  # Legacy
    get_default_embedding_provider,
    get_cached_embedding_provider,
    clear_embedding_provider_cache,
    register_embedding_provider,
    get_available_embedding_providers,
)

# Provider implementations
from .openai_embedding_provider import OpenAIEmbeddingProvider
from .gemini_embedding_provider import GeminiEmbeddingProvider

# Exceptions
from .exceptions import (
    EmbeddingProviderError,
    EmbeddingProviderInitializationError,
    EmbeddingGenerationError,
    EmbeddingDimensionMismatchError,
    EmbeddingBatchSizeError,
    EmbeddingTextPreprocessingError,
    EmbeddingAuthenticationError,
    EmbeddingRateLimitError,
)


# Version
__version__ = "2.0.0"

# Public API
__all__ = [
    # Base classes
    "EmbeddingProvider",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "EmbeddingTaskType",
    # Factory functions
    "create_embedding_provider",
    "get_embedding_provider",
    "get_default_embedding_provider",
    "get_cached_embedding_provider",
    "clear_embedding_provider_cache",
    "register_embedding_provider",
    "get_available_embedding_providers",
    # Providers
    "OpenAIEmbeddingProvider",
    "GeminiEmbeddingProvider",
    # Exceptions
    "EmbeddingProviderError",
    "EmbeddingProviderInitializationError",
    "EmbeddingGenerationError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingBatchSizeError",
    "EmbeddingTextPreprocessingError",
    "EmbeddingAuthenticationError",
    "EmbeddingRateLimitError",
]
