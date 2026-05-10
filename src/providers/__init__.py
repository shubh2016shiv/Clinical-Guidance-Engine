"""
Providers Module — AI Service Abstractions

Umbrella package that exposes the complete public API of every provider
sub-package from a single import location.

Sub-packages
------------
- ``llm_provider``       — OpenAI / Gemini LLM execution (chat & responses API).
- ``embedding_provider``  — Text embedding generation (OpenAI / Gemini).
- ``vector_db_provider``  — Vector database search (Milvus).
- ``cache_provider``      — Session & response caching (Redis).

Quick Start::

    from src.providers import (
        create_llm_provider,
        create_embedding_provider,
        create_vector_db_provider,
        create_cache_provider,
        EmbeddingTaskType,
        APIType,
        StreamMode,
    )

    llm = create_llm_provider()
    emb = create_embedding_provider(task_type=EmbeddingTaskType.RETRIEVAL_QUERY)
    vdb = create_vector_db_provider()
    cache = create_cache_provider()
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Cache provider
# ---------------------------------------------------------------------------
from .cache_provider import (
    ActiveSessionContext,
    CacheAuthenticationError,
    CacheConfig,
    CacheConfigError,
    CacheConnectionError,
    CacheKeyError,
    CacheOperationError,
    CacheOperationType,
    CacheProvider,
    CacheProviderError,
    CacheSerializationError,
    CacheTTLError,
    RedisProvider,
    clear_cache_provider_cache,
    create_cache_provider,
    get_available_cache_providers,
    get_cache_provider,  # legacy alias
    get_cached_cache_provider,
    get_default_cache_provider,
    register_cache_provider,
)

# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------
from .embedding_provider import (
    EmbeddingAuthenticationError,
    EmbeddingBatchSizeError,
    EmbeddingConfig,
    EmbeddingDimensionMismatchError,
    EmbeddingGenerationError,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingProviderInitializationError,
    EmbeddingRateLimitError,
    EmbeddingResponse,
    EmbeddingTaskType,
    EmbeddingTextPreprocessingError,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
    clear_embedding_provider_cache,
    create_embedding_provider,
    get_available_embedding_providers,
    get_cached_embedding_provider,
    get_default_embedding_provider,
    get_embedding_provider,  # legacy alias
    register_embedding_provider,
)

# ---------------------------------------------------------------------------
# Top-level base exception (for unified error handling)
# ---------------------------------------------------------------------------
from .exceptions import ProviderError

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
from .llm_provider import (
    APIType,
    LLMConfig,
    LLMProvider,
    LLMProviderAuthenticationError,
    LLMProviderError,
    LLMProviderExecutionError,
    LLMProviderInitializationError,
    LLMProviderRateLimitError,
    LLMProviderStreamingError,
    LLMProviderTokenLimitError,
    LLMProviderUnsupportedOperationError,
    LLMResponse,
    OpenAIProvider,
    StreamMode,
    clear_provider_cache,
    create_llm_provider,
    get_available_providers,
    get_cached_provider,
    get_default_provider,
    get_llm_provider,  # legacy alias
    register_provider,
)

# ---------------------------------------------------------------------------
# Vector DB provider
# ---------------------------------------------------------------------------
from .vector_db_provider import (
    MilvusProvider,
    SearchMetricType,
    SearchResult,
    VectorDBAuthenticationError,
    VectorDBCollectionConfig,
    VectorDBCollectionError,
    VectorDBConfig,
    VectorDBConfigError,
    VectorDBConnectionConfig,
    VectorDBConnectionError,
    VectorDBDimensionMismatchError,
    VectorDBIndexError,
    VectorDBProvider,
    VectorDBProviderError,
    VectorDBRateLimitError,
    VectorDBSearchConfig,
    VectorDBSearchError,
    clear_vector_db_provider_cache,
    create_vector_db_provider,
    get_available_vector_db_providers,
    get_cached_vector_db_provider,
    get_default_vector_db_provider,
    register_vector_db_provider,
)

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # === Top-level base exception ===
    "ProviderError",
    # === Cache provider ===
    "CacheProvider",
    "CacheConfig",
    "CacheOperationType",
    "ActiveSessionContext",
    "create_cache_provider",
    "get_cache_provider",
    "get_default_cache_provider",
    "get_cached_cache_provider",
    "clear_cache_provider_cache",
    "register_cache_provider",
    "get_available_cache_providers",
    "RedisProvider",
    "CacheProviderError",
    "CacheConnectionError",
    "CacheAuthenticationError",
    "CacheConfigError",
    "CacheOperationError",
    "CacheSerializationError",
    "CacheKeyError",
    "CacheTTLError",
    # === Embedding provider ===
    "EmbeddingProvider",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "EmbeddingTaskType",
    "create_embedding_provider",
    "get_embedding_provider",
    "get_default_embedding_provider",
    "get_cached_embedding_provider",
    "clear_embedding_provider_cache",
    "register_embedding_provider",
    "get_available_embedding_providers",
    "GeminiEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingProviderInitializationError",
    "EmbeddingGenerationError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingBatchSizeError",
    "EmbeddingTextPreprocessingError",
    "EmbeddingAuthenticationError",
    "EmbeddingRateLimitError",
    # === LLM provider ===
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "APIType",
    "StreamMode",
    "create_llm_provider",
    "get_llm_provider",
    "get_default_provider",
    "get_cached_provider",
    "clear_provider_cache",
    "register_provider",
    "get_available_providers",
    "OpenAIProvider",
    "LLMProviderError",
    "LLMProviderInitializationError",
    "LLMProviderExecutionError",
    "LLMProviderTokenLimitError",
    "LLMProviderStreamingError",
    "LLMProviderAuthenticationError",
    "LLMProviderRateLimitError",
    "LLMProviderUnsupportedOperationError",
    # === Vector DB provider ===
    "VectorDBProvider",
    "VectorDBConfig",
    "VectorDBConnectionConfig",
    "VectorDBCollectionConfig",
    "VectorDBSearchConfig",
    "SearchResult",
    "SearchMetricType",
    "create_vector_db_provider",
    "get_default_vector_db_provider",
    "get_cached_vector_db_provider",
    "clear_vector_db_provider_cache",
    "register_vector_db_provider",
    "get_available_vector_db_providers",
    "MilvusProvider",
    "VectorDBProviderError",
    "VectorDBConnectionError",
    "VectorDBSearchError",
    "VectorDBConfigError",
    "VectorDBDimensionMismatchError",
    "VectorDBCollectionError",
    "VectorDBAuthenticationError",
    "VectorDBRateLimitError",
    "VectorDBIndexError",
]
