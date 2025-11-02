"""
Vector Database Provider Factory

This module provides factory functions for creating and managing vector database
provider instances. It supports:
- Multiple provider types (Milvus, future Pinecone, etc.)
- Configuration-driven instantiation
- Provider registry management
- Default provider settings from config
"""

from typing import Dict, Optional, Type
from functools import lru_cache

from src.config import get_settings
from src.providers.embedding_provider import (
    create_embedding_provider,
    EmbeddingProvider,
    EmbeddingTaskType,
)
from src.providers.vector_db_provider.exceptions import VectorDBConfigError
from src.logs import get_logger
from .base import (
    VectorDBProvider,
    VectorDBConfig,
    VectorDBConnectionConfig,
    VectorDBCollectionConfig,
    VectorDBSearchConfig,
)
from .milvus_provider import MilvusProvider

logger = get_logger(__name__)


# Provider registry - maps provider names to their implementation classes
VECTOR_DB_PROVIDER_REGISTRY: Dict[str, Type[VectorDBProvider]] = {
    "milvus": MilvusProvider,
    # Add more providers here as they're implemented
    # "pinecone": PineconeProvider,
    # "weaviate": WeaviateProvider,
}


def register_vector_db_provider(
    name: str, provider_class: Type[VectorDBProvider]
) -> None:
    """
    Register a new vector database provider implementation

    This allows dynamic registration of custom provider implementations
    without modifying the factory code.

    Args:
        name: Provider name/identifier (e.g., "milvus", "pinecone")
        provider_class: Provider class that implements VectorDBProvider interface

    Raises:
        ValueError: If provider name is invalid
    """
    name = name.lower()
    if name in VECTOR_DB_PROVIDER_REGISTRY:
        logger.warning(f"Overwriting existing vector DB provider registration: {name}")

    VECTOR_DB_PROVIDER_REGISTRY[name] = provider_class
    logger.info(f"Registered vector DB provider: {name} -> {provider_class.__name__}")


def get_available_vector_db_providers() -> list[str]:
    """
    Get list of available vector database provider names

    Returns:
        List of registered provider identifiers
    """
    return list(VECTOR_DB_PROVIDER_REGISTRY.keys())


def create_vector_db_provider(
    provider_type: Optional[str] = None,
    config: Optional[VectorDBConfig] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    **kwargs,
) -> VectorDBProvider:
    """
    Factory function to create a vector database provider instance

    This is the main entry point for creating vector database providers. It supports
    multiple ways of specifying configuration:
    1. Complete VectorDBConfig object
    2. Individual parameters (host, port, collection_name, etc.)
    3. Settings from environment/config file

    Args:
        provider_type: Provider identifier (e.g., "milvus")
                      Defaults to "milvus"
        config: Complete VectorDBConfig object (takes precedence)
        embedding_provider: Embedding provider instance for query embeddings
                          If None, creates a Gemini provider with RETRIEVAL_QUERY task type
        **kwargs: Additional configuration parameters:
            - host: Milvus host (default: "localhost")
            - port: Milvus port (default: 19530)
            - collection_name: Collection name (default: "pharmaceutical_drugs")
            - embedding_dimension: Embedding dimension (default: 768)
            - top_k: Default number of search results (default: 10)
            - search_params: Search parameters dict (default: {"ef": 64})

    Returns:
        Configured vector database provider instance

    Raises:
        VectorDBConfigError: If provider type is not supported or initialization fails

    Examples:
        # Using defaults from config
        provider = create_vector_db_provider()

        # Specify provider and connection
        provider = create_vector_db_provider(
            provider_type="milvus",
            host="localhost",
            port=19530,
            collection_name="pharmaceutical_drugs"
        )

        # Using complete config object
        from src.providers.vector_db_provider.base import (
            VectorDBConfig,
            VectorDBConnectionConfig,
            VectorDBCollectionConfig,
            VectorDBSearchConfig
        )
        config = VectorDBConfig(
            connection=VectorDBConnectionConfig(host="localhost", port=19530),
            collection=VectorDBCollectionConfig(name="pharmaceutical_drugs", embedding_dimension=768),
            search=VectorDBSearchConfig(top_k=10)
        )
        provider = create_vector_db_provider(provider_type="milvus", config=config)

        # With custom embedding provider
        from src.providers.embedding_provider import create_embedding_provider
        embedding_provider = create_embedding_provider(
            provider_type="gemini",
            embedding_dimension=768,
            task_type=EmbeddingTaskType.RETRIEVAL_QUERY
        )
        provider = create_vector_db_provider(
            provider_type="milvus",
            embedding_provider=embedding_provider
        )
    """
    settings = get_settings()

    # Determine provider type
    provider_type = (provider_type or "milvus").lower()

    # Validate provider exists
    if provider_type not in VECTOR_DB_PROVIDER_REGISTRY:
        supported = ", ".join(get_available_vector_db_providers())
        error_msg = (
            f"Unsupported vector database provider: '{provider_type}'. "
            f"Supported providers: {supported}"
        )
        logger.error(error_msg)
        raise VectorDBConfigError(error_msg)

    # Get provider class
    provider_class = VECTOR_DB_PROVIDER_REGISTRY[provider_type]

    logger.debug(
        f"Creating vector DB provider - Type: {provider_type}, "
        f"Collection: {kwargs.get('collection_name', 'default')}"
    )

    try:
        # If no config provided, create one from kwargs or defaults
        if config is None:
            # Extract connection parameters
            conn_kwargs = {
                "host": kwargs.pop("host", "localhost"),
                "port": kwargs.pop("port", 19530),
                "alias": kwargs.pop("alias", "default"),
                "timeout": kwargs.pop("timeout", 30),
                "user": kwargs.pop("user", ""),
                "password": kwargs.pop("password", ""),
            }

            # Extract collection parameters
            coll_kwargs = {
                "name": kwargs.pop("collection_name", "pharmaceutical_drugs"),
                "embedding_dimension": kwargs.pop("embedding_dimension", 768),
                "load_on_startup": kwargs.pop("load_on_startup", True),
            }

            # Extract search parameters
            search_kwargs = {
                "top_k": kwargs.pop("top_k", 10),
                "search_params": kwargs.pop("search_params", {"ef": 64}),
                "output_fields": kwargs.pop(
                    "output_fields",
                    [
                        "drug_name",
                        "drug_class",
                        "drug_sub_class",
                        "therapeutic_category",
                        "route_of_administration",
                        "formulation",
                        "dosage_strengths",
                        "search_text",
                    ],
                ),
            }

            # Build config objects
            conn_config = VectorDBConnectionConfig(**conn_kwargs)
            coll_config = VectorDBCollectionConfig(**coll_kwargs)
            search_config = VectorDBSearchConfig(**search_kwargs)

            config = VectorDBConfig(
                connection=conn_config,
                collection=coll_config,
                search=search_config,
                embedding_dimension=coll_config.embedding_dimension,
                embedding_normalized=True,
            )

        # Create embedding provider if not provided
        if embedding_provider is None:
            try:
                embedding_provider = create_embedding_provider(
                    provider_type="gemini",
                    model_name=settings.gemini_embedding_model or "text-embedding-004",
                    embedding_dimension=config.collection.embedding_dimension,
                    task_type=EmbeddingTaskType.RETRIEVAL_QUERY,
                )
                logger.debug(
                    f"Created default Gemini embedding provider - "
                    f"Dimension: {config.collection.embedding_dimension}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create default embedding provider: {e}. "
                    f"Provider will need to be provided explicitly."
                )

        # Create provider instance
        provider = provider_class(
            config=config,
            embedding_provider=embedding_provider,
            **kwargs,
        )

        logger.info(f"Successfully created {provider_type} vector DB provider")
        return provider

    except Exception as e:
        error_msg = f"Failed to initialize {provider_type} vector DB provider: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise VectorDBConfigError(error_msg)


@lru_cache(maxsize=5)
def get_cached_vector_db_provider(
    provider_type: str = "milvus",
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "pharmaceutical_drugs",
    embedding_dimension: int = 768,
) -> VectorDBProvider:
    """
    Get or create a cached vector database provider instance

    This function caches provider instances to avoid repeated initialization
    overhead. Useful when using the same provider configuration multiple times.

    Note: Only use caching when provider configuration is stable. For dynamic
    configurations, use create_vector_db_provider() directly.

    Args:
        provider_type: Provider identifier
        host: Milvus host
        port: Milvus port
        collection_name: Collection name
        embedding_dimension: Embedding dimension

    Returns:
        Cached or newly created provider instance

    Raises:
        VectorDBConfigError: If provider creation fails
    """
    return create_vector_db_provider(
        provider_type=provider_type,
        host=host,
        port=port,
        collection_name=collection_name,
        embedding_dimension=embedding_dimension,
    )


def clear_vector_db_provider_cache() -> None:
    """
    Clear the vector database provider cache

    Call this when you need to force recreation of cached providers,
    for example after configuration changes.
    """
    get_cached_vector_db_provider.cache_clear()
    logger.info("Vector DB provider cache cleared")


def get_default_vector_db_provider() -> VectorDBProvider:
    """
    Get the default vector database provider based on application settings

    This is a convenience function that creates a provider using
    all defaults from the configuration system.

    Returns:
        Default configured vector database provider instance

    Raises:
        VectorDBConfigError: If provider creation fails
    """
    return create_vector_db_provider(provider_type="milvus")
