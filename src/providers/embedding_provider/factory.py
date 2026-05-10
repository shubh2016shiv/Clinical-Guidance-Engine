"""
Embedding Provider Factory

This module provides factory functions for creating and managing embedding provider instances.
It supports:
- Multiple provider types (OpenAI, Gemini, etc.)
- Configuration-driven instantiation
- Provider registry management
- Default provider settings from config
"""

from functools import lru_cache

from src.config import get_settings
from src.logs import get_logger
from src.providers.exceptions import EmbeddingProviderError

from .base import EmbeddingConfig, EmbeddingProvider, EmbeddingTaskType
from .gemini_embedding_provider import GEMINI_MODEL_DIMENSIONS, GeminiEmbeddingProvider
from .openai_embedding_provider import OPENAI_MODEL_DIMENSIONS, OpenAIEmbeddingProvider

logger = get_logger(__name__)

# Provider registry - maps provider names to their implementation classes
EMBEDDING_PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "openai": OpenAIEmbeddingProvider,
    "gemini": GeminiEmbeddingProvider,
    # Add more providers here as they're implemented
    # "cohere": CohereEmbeddingProvider,
    # "huggingface": HuggingFaceEmbeddingProvider,
}

# Provider-specific dimension defaults (auto-detected from model name)
_PROVIDER_DIMENSION_MAPS: dict[str, dict[str, int]] = {
    "openai": OPENAI_MODEL_DIMENSIONS,
    "gemini": GEMINI_MODEL_DIMENSIONS,
}

# Provider-specific default models
_PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "gemini": "text-embedding-004",
}

# Provider-specific default dimensions (fallback if model not in map)
_PROVIDER_DEFAULT_DIMENSIONS: dict[str, int] = {
    "openai": 1536,
    "gemini": 768,
}


def register_embedding_provider(name: str, provider_class: type[EmbeddingProvider]) -> None:
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
    provider_type: str | None = None,
    config: EmbeddingConfig | None = None,
    model_name: str | None = None,
    embedding_dimension: int | None = None,
    task_type: EmbeddingTaskType | None = None,
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
        provider_type or settings.default_embedding_provider or "gemini"  # Default fallback
    ).lower()

    # Validate provider exists
    if provider_type not in EMBEDDING_PROVIDER_REGISTRY:
        supported = ", ".join(get_available_embedding_providers())
        error_msg = (
            f"Unsupported embedding provider: '{provider_type}'. Supported providers: {supported}"
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

    # Extract api_key from kwargs before config resolution
    api_key = kwargs.pop("api_key", None)

    # Resolve config: build a complete EmbeddingConfig if one was not provided,
    # or overlay individual parameter overrides onto an existing config.
    if config is None:
        resolved_model = model_name or _PROVIDER_DEFAULT_MODELS.get(
            provider_type, "text-embedding-3-small"
        )
        resolved_dimension = embedding_dimension or _resolve_dimension(
            provider_type, resolved_model
        )
        resolved_task = task_type or EmbeddingTaskType.SEMANTIC_SIMILARITY

        config = EmbeddingConfig(
            model_name=resolved_model,
            embedding_dimension=resolved_dimension,
            task_type=resolved_task,
            **kwargs,
        )
    elif model_name or embedding_dimension or task_type or kwargs:
        # Merge individual overrides into the provided config object
        config = EmbeddingConfig(
            model_name=model_name or config.model_name,
            embedding_dimension=embedding_dimension or config.embedding_dimension,
            task_type=task_type or config.task_type,
            batch_size=kwargs.pop("batch_size", config.batch_size),
            timeout=kwargs.pop("timeout", config.timeout),
            max_retries=kwargs.pop("max_retries", config.max_retries),
            retry_delay=kwargs.pop("retry_delay", config.retry_delay),
            strip_whitespace=kwargs.pop("strip_whitespace", config.strip_whitespace),
            remove_empty=kwargs.pop("remove_empty", config.remove_empty),
            max_text_length=kwargs.pop("max_text_length", config.max_text_length),
            extra_params={**config.extra_params, **kwargs},
        )

    try:
        # Create provider instance with canonical config + api_key only
        provider = provider_class(config=config, api_key=api_key)
        logger.info(f"Successfully created {provider_type} embedding provider")
        return provider

    except Exception as e:
        error_msg = f"Failed to initialize {provider_type} embedding provider: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise EmbeddingProviderError(error_msg) from e


def _resolve_dimension(provider_type: str, model_name: str) -> int:
    """Resolve the embedding dimension for a given provider and model."""
    dim_map = _PROVIDER_DIMENSION_MAPS.get(provider_type, {})
    return dim_map.get(model_name, _PROVIDER_DEFAULT_DIMENSIONS.get(provider_type, 1536))


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
def get_embedding_provider(provider_type: str = "gemini", **kwargs) -> EmbeddingProvider:
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
        "Using legacy get_embedding_provider() - consider using create_embedding_provider()"
    )
    return create_embedding_provider(provider_type=provider_type, **kwargs)
