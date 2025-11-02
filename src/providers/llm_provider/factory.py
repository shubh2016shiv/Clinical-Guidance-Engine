"""
LLM Provider Factory

This module provides factory functions for creating and managing LLM provider instances.
It supports:
- Multiple provider types (OpenAI, Gemini, etc.)
- Configuration-driven instantiation
- Provider registry management
- Default provider settings from config
"""

from typing import Dict, Optional, Type
from functools import lru_cache

from src.config import get_settings
from src.providers.exceptions import LLMProviderError
from src.logs import get_logger
from .base import LLMProvider, LLMConfig, APIType, StreamMode
from .openai_provider import OpenAIProvider

logger = get_logger(__name__)


# Provider registry - maps provider names to their implementation classes
PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {
    "openai": OpenAIProvider,
    # Add more providers here as they're implemented
    # "anthropic": AnthropicProvider,
    # "gemini": GeminiProvider,
}


def register_provider(name: str, provider_class: Type[LLMProvider]) -> None:
    """
    Register a new provider implementation

    This allows dynamic registration of custom provider implementations
    without modifying the factory code.

    Args:
        name: Provider name/identifier (e.g., "openai", "custom")
        provider_class: Provider class that implements LLMProvider interface

    Raises:
        ValueError: If provider is already registered
    """
    name = name.lower()
    if name in PROVIDER_REGISTRY:
        logger.warning(f"Overwriting existing provider registration: {name}")

    PROVIDER_REGISTRY[name] = provider_class
    logger.info(f"Registered provider: {name} -> {provider_class.__name__}")


def get_available_providers() -> list[str]:
    """
    Get list of available provider names

    Returns:
        List of registered provider identifiers
    """
    return list(PROVIDER_REGISTRY.keys())


def create_llm_provider(
    provider_type: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    model_name: Optional[str] = None,
    api_type: Optional[APIType] = None,
    stream_mode: Optional[StreamMode] = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create an LLM provider instance

    This is the main entry point for creating providers. It supports multiple
    ways of specifying configuration:
    1. Complete LLMConfig object
    2. Individual parameters (model_name, api_type, etc.)
    3. Settings from environment/config file

    Args:
        provider_type: Provider identifier (e.g., "openai", "gemini")
                      Defaults to settings.default_llm_provider
        config: Complete LLMConfig object (takes precedence)
        model_name: Model name to use
        api_type: API type (chat_completion, responses, etc.)
        stream_mode: Streaming mode (enabled, disabled, auto)
        **kwargs: Additional configuration parameters and provider-specific args

    Returns:
        Configured LLM provider instance

    Raises:
        LLMProviderError: If provider type is not supported or initialization fails

    Examples:
        # Using defaults from config
        provider = create_llm_provider()

        # Specify provider and model
        provider = create_llm_provider(
            provider_type="openai",
            model_name="gpt-4-turbo"
        )

        # Using complete config object
        config = LLMConfig(
            model_name="gpt-4",
            api_type=APIType.RESPONSES,
            stream_mode=StreamMode.ENABLED,
            temperature=0.8
        )
        provider = create_llm_provider(provider_type="openai", config=config)

        # Enable streaming with Responses API
        provider = create_llm_provider(
            provider_type="openai",
            model_name="gpt-4",
            api_type=APIType.RESPONSES,
            stream_mode=StreamMode.ENABLED
        )
    """
    settings = get_settings()

    # Determine provider type
    provider_type = (provider_type or settings.default_llm_provider or "openai").lower()

    # Validate provider exists
    if provider_type not in PROVIDER_REGISTRY:
        supported = ", ".join(get_available_providers())
        error_msg = (
            f"Unsupported LLM provider: '{provider_type}'. "
            f"Supported providers: {supported}"
        )
        logger.error(error_msg)
        raise LLMProviderError(error_msg)

    # Get provider class
    provider_class = PROVIDER_REGISTRY[provider_type]

    logger.debug(
        f"Creating LLM provider - Type: {provider_type}, "
        f"Model: {model_name or 'default'}, "
        f"API: {api_type.value if api_type else 'default'}"
    )

    try:
        # Create provider instance
        # Pass all parameters and let the provider handle them
        # Note: Settings from config.py are automatically pulled by provider __init__
        # if not explicitly provided in kwargs. This ensures centralized control.
        provider = provider_class(
            config=config,
            model_name=model_name,
            api_type=api_type,
            stream_mode=stream_mode,
            **kwargs,
        )

        logger.info(f"Successfully created {provider_type} provider")
        return provider

    except Exception as e:
        error_msg = f"Failed to initialize {provider_type} provider: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise LLMProviderError(error_msg)


@lru_cache(maxsize=10)
def get_cached_provider(
    provider_type: str,
    model_name: str,
    api_type: str = "chat_completion",
    stream_mode: str = "disabled",
) -> LLMProvider:
    """
    Get or create a cached provider instance

    This function caches provider instances to avoid repeated initialization
    overhead. Useful when using the same provider configuration multiple times.

    Note: Only use caching when provider configuration is stable. For dynamic
    configurations, use create_llm_provider() directly.

    Args:
        provider_type: Provider identifier
        model_name: Model name
        api_type: API type as string
        stream_mode: Stream mode as string

    Returns:
        Cached or newly created provider instance

    Raises:
        LLMProviderError: If provider creation fails
    """
    # Convert string enums to actual enums
    api_type_enum = APIType(api_type)
    stream_mode_enum = StreamMode(stream_mode)

    return create_llm_provider(
        provider_type=provider_type,
        model_name=model_name,
        api_type=api_type_enum,
        stream_mode=stream_mode_enum,
    )


def clear_provider_cache() -> None:
    """
    Clear the provider cache

    Call this when you need to force recreation of cached providers,
    for example after configuration changes.
    """
    get_cached_provider.cache_clear()
    logger.info("Provider cache cleared")


def get_default_provider() -> LLMProvider:
    """
    Get the default LLM provider based on application settings

    This is a convenience function that creates a provider using
    all defaults from the configuration system.

    Returns:
        Default configured provider instance

    Raises:
        LLMProviderError: If provider creation fails
    """
    settings = get_settings()

    return create_llm_provider(
        provider_type=settings.default_llm_provider,
        model_name=settings.default_model_name,
    )


# Convenience function alias for backward compatibility
def get_llm_provider(provider_type: str = "openai", **kwargs) -> LLMProvider:
    """
    Legacy factory function - use create_llm_provider() for new code

    This function is maintained for backward compatibility with existing code.

    Args:
        provider_type: Provider identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured LLM provider instance
    """
    logger.debug(
        "Using legacy get_llm_provider() - consider using create_llm_provider()"
    )
    return create_llm_provider(provider_type=provider_type, **kwargs)
