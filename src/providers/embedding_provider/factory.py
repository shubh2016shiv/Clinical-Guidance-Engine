"""
Embedding Provider Factory

This module provides factory functions for creating and managing embedding provider instances.
It supports:
- Multiple provider types (OpenAI, Gemini, etc.)
- Configuration-driven instantiation
- Provider registry management
- Default provider settings from config
"""

from typing import Dict, Optional, Type
from functools import lru_cache

from src.config import get_settings
from src.providers.exceptions import EmbeddingProviderError
from src.logs import get_logger
from .base import EmbeddingProvider, EmbeddingConfig, EmbeddingTaskType
from .openai_embedding_provider import OpenAIEmbeddingProvider
from .gemini_embedding_provider import GeminiEmbeddingProvider

logger = get_logger(__name__)


# Provider registry - maps provider names to their implementation classes
EMBEDDING_PROVIDER_REGISTRY: Dict[str, Type[EmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
    "gemini": GeminiEmbeddingProvider,
    # Add more providers here as they're implemented
    # "cohere": CohereEmbeddingProvider,
    # "huggingface": HuggingFaceEmbeddingProvider,
}


def register_embedding_provider(
    name: str, provider_class: Type[EmbeddingProvider]
) -> None:
    """
    Register a new embedding provider implementation

    This allows dynamic registration of custom provider implementations
    without modifying the factory code.

    Args:
        name: Provider name/identifier (e.g., "openai", "custom")
        provider_class: Provider class that implements EmbeddingProvider interface

    Raises:
        ValueError: If provider name is invalid
    """
    name = name.lower()
    if name in EMBEDDING_PROVIDER_REGISTRY:
        logger.warning(f"Overwriting existing embedding provider registration: {name}")

    EMBEDDING_PROVIDER_REGISTRY[name] = provider_class
    logger.info(f"Registered embedding provider: {name} -> {provider_class.__name__}")


def get_available_embedding_providers() -> list[str]:
    """
    Get list of available embedding provider names

    Returns:
        List of registered provider identifiers
    """
    return list(EMBEDDING_PROVIDER_REGISTRY.keys())


def create_embedding_provider(
    provider_type: Optional[str] = None,
    config: Optional[EmbeddingConfig] = None,
    model_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    task_type: Optional[EmbeddingTaskType] = None,
    **kwargs,
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider instance

    This is the main entry point for creating embedding providers. It supports
    multiple ways of specifying configuration:
    1. Complete EmbeddingConfig object
    2. Individual parameters (model_name, dimension, etc.)
    3. Settings from environment/config file

    Args:
        provider_type: Provider identifier (e.g., "openai", "gemini")
                      Defaults to settings.default_embedding_provider
        config: Complete EmbeddingConfig object (takes precedence)
        model_name: Model name to use
        embedding_dimension: Dimension of embedding vectors
        task_type: Task type for embeddings (semantic similarity, retrieval, etc.)
        **kwargs: Additional configuration parameters and provider-specific args

    Returns:
        Configured embedding provider instance

    Raises:
        EmbeddingProviderError: If provider type is not supported or initialization fails

    Examples:
        # Using defaults from config
        provider = create_embedding_provider()

        # Specify provider and model
        provider = create_embedding_provider(
            provider_type="openai",
            model_name="text-embedding-3-small"
        )

        # Using complete config object
        config = EmbeddingConfig(
            model_name="text-embedding-3-large",
            embedding_dimension=3072,
            task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
            batch_size=50
        )
        provider = create_embedding_provider(provider_type="openai", config=config)

        # Gemini with specific task type
        provider = create_embedding_provider(
            provider_type="gemini",
            model_name="text-embedding-004",
            task_type=EmbeddingTaskType.CLASSIFICATION
        )
    """
    settings = get_settings()

    # Determine provider type
    provider_type = (
        provider_type
        or settings.default_embedding_provider
        or "gemini"  # Default fallback
    ).lower()

    # Validate provider exists
    if provider_type not in EMBEDDING_PROVIDER_REGISTRY:
        supported = ", ".join(get_available_embedding_providers())
        error_msg = (
            f"Unsupported embedding provider: '{provider_type}'. "
            f"Supported providers: {supported}"
        )
        logger.error(error_msg)
        raise EmbeddingProviderError(error_msg)

    # Get provider class
    provider_class = EMBEDDING_PROVIDER_REGISTRY[provider_type]

    logger.debug(
        f"Creating embedding provider - Type: {provider_type}, "
        f"Model: {model_name or 'default'}, "
        f"Task: {task_type.value if task_type else 'default'}"
    )

    try:
        # Create provider instance
        # Pass all parameters and let the provider handle them
        provider = provider_class(
            config=config,
            model_name=model_name,
            embedding_dimension=embedding_dimension,
            task_type=task_type,
            **kwargs,
        )

        logger.info(f"Successfully created {provider_type} embedding provider")
        return provider

    except Exception as e:
        error_msg = f"Failed to initialize {provider_type} embedding provider: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise EmbeddingProviderError(error_msg)


@lru_cache(maxsize=10)
def get_cached_embedding_provider(
    provider_type: str,
    model_name: str,
    embedding_dimension: int = 1536,
    task_type: str = "semantic_similarity",
) -> EmbeddingProvider:
    """
    Get or create a cached embedding provider instance

    This function caches provider instances to avoid repeated initialization
    overhead. Useful when using the same provider configuration multiple times.

    Note: Only use caching when provider configuration is stable. For dynamic
    configurations, use create_embedding_provider() directly.

    Args:
        provider_type: Provider identifier
        model_name: Model name
        embedding_dimension: Embedding dimension
        task_type: Task type as string

    Returns:
        Cached or newly created provider instance

    Raises:
        EmbeddingProviderError: If provider creation fails
    """
    # Convert string task type to enum
    task_type_enum = EmbeddingTaskType(task_type)

    return create_embedding_provider(
        provider_type=provider_type,
        model_name=model_name,
        embedding_dimension=embedding_dimension,
        task_type=task_type_enum,
    )


def clear_embedding_provider_cache() -> None:
    """
    Clear the embedding provider cache

    Call this when you need to force recreation of cached providers,
    for example after configuration changes.
    """
    get_cached_embedding_provider.cache_clear()
    logger.info("Embedding provider cache cleared")


def get_default_embedding_provider() -> EmbeddingProvider:
    """
    Get the default embedding provider based on application settings

    This is a convenience function that creates a provider using
    all defaults from the configuration system.

    Returns:
        Default configured embedding provider instance

    Raises:
        EmbeddingProviderError: If provider creation fails
    """
    settings = get_settings()

    return create_embedding_provider(
        provider_type=settings.default_embedding_provider,
        model_name=settings.default_embedding_model,
    )


# Legacy function alias for backward compatibility
def get_embedding_provider(
    provider_type: str = "gemini", **kwargs
) -> EmbeddingProvider:
    """
    Legacy factory function - use create_embedding_provider() for new code

    This function is maintained for backward compatibility with existing code.

    Args:
        provider_type: Provider identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured embedding provider instance
    """
    logger.debug(
        "Using legacy get_embedding_provider() - "
        "consider using create_embedding_provider()"
    )
    return create_embedding_provider(provider_type=provider_type, **kwargs)
