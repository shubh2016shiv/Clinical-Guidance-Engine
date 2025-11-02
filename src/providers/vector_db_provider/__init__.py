"""
Vector Database Provider Module

This module provides a centralized, professional interface for working with
vector database providers.

Key Features:
- Multiple provider support (Milvus, future Pinecone, etc.)
- Vector similarity search
- Filter expression support
- Both sync and async operations
- Flexible configuration management
- Type-safe interfaces with dataclasses
- Comprehensive error handling
- Integration with embedding providers

Quick Start:
    from src.providers.vector_db_provider import (
        create_vector_db_provider,
        SearchMetricType
    )

    # Create a provider
    provider = create_vector_db_provider(
        provider_type="milvus",
        host="localhost",
        port=19530,
        collection_name="pharmaceutical_drugs"
    )

    # Connect to database
    await provider.connect()

    # Search by text (automatically generates embedding)
    results = await provider.search_by_text(
        query_text="ACE inhibitors for hypertension",
        limit=10
    )

    # Search by vector
    query_vector = [0.1, 0.2, ...]  # Your embedding vector
    results = await provider.search(
        query_vector=query_vector,
        limit=10,
        filter_expression='drug_class == "ACE inhibitor"'
    )

    # Get collection info
    info = await provider.get_collection_info()

    # Disconnect
    await provider.disconnect()
"""

# Base classes and interfaces
from .base import (
    VectorDBProvider,
    VectorDBConfig,
    VectorDBConnectionConfig,
    VectorDBCollectionConfig,
    VectorDBSearchConfig,
    SearchResult,
    SearchMetricType,
)

# Factory functions
from .factory import (
    create_vector_db_provider,
    get_cached_vector_db_provider,
    clear_vector_db_provider_cache,
    get_default_vector_db_provider,
    register_vector_db_provider,
    get_available_vector_db_providers,
)

# Provider implementations
from .milvus_provider import MilvusProvider

# Exceptions
from .exceptions import (
    VectorDBProviderError,
    VectorDBConnectionError,
    VectorDBSearchError,
    VectorDBConfigError,
    VectorDBDimensionMismatchError,
    VectorDBCollectionError,
    VectorDBAuthenticationError,
    VectorDBRateLimitError,
    VectorDBIndexError,
)

# Version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Base classes
    "VectorDBProvider",
    "VectorDBConfig",
    "VectorDBConnectionConfig",
    "VectorDBCollectionConfig",
    "VectorDBSearchConfig",
    "SearchResult",
    "SearchMetricType",
    # Factory functions
    "create_vector_db_provider",
    "get_cached_vector_db_provider",
    "clear_vector_db_provider_cache",
    "get_default_vector_db_provider",
    "register_vector_db_provider",
    "get_available_vector_db_providers",
    # Providers
    "MilvusProvider",
    # Exceptions
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
