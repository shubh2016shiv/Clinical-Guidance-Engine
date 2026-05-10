"""
LLM Provider Factory.

Centralised factory for creating and caching LLM provider instances.
Supports a registry of named providers so that new backends can be
plugged in without modifying the factory itself.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.config import get_settings
from src.logs import get_logger

from .base import APIType, LLMConfig, LLMProvider, StreamMode
from .exceptions import LLMProviderError
from .openai_provider import OpenAIProvider

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    # Register additional providers here, e.g.:
    # "anthropic": AnthropicProvider,
    # "gemini": GeminiProvider,
}


def register_provider(name: str, provider_class: type[LLMProvider]) -> None:
    """Register a new provider implementation at runtime.

    Args:
        name: Unique provider slug (case-insensitive).
        provider_class: Class that implements :class:`LLMProvider`.
    """
    name = name.lower()
    if name in PROVIDER_REGISTRY:
        logger.warning("Overwriting existing provider registration: %s", name)
    PROVIDER_REGISTRY[name] = provider_class
    logger.info("Registered provider: %s -> %s", name, provider_class.__name__)


def get_available_providers() -> list[str]:
    """Return the slugs of all registered providers."""
    return list(PROVIDER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm_provider(
    provider_type: str | None = None,
    config: LLMConfig | None = None,
    model_name: str | None = None,
    api_type: APIType | None = None,
    stream_mode: StreamMode | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create and return a configured LLM provider instance.

    Configuration can be supplied via a pre-built :class:`LLMConfig`
    object, as individual keyword arguments, or through application
    settings (``config.py`` / ``.env``).

    Args:
        provider_type: Provider slug (e.g. ``"openai"``).  Defaults to
            the application setting ``default_llm_provider``, or
            ``"openai"`` if none is set.
        config: Complete :class:`LLMConfig` — takes precedence over
            individual parameters.
        model_name: Override the model name.
        api_type: Override the API type.
        stream_mode: Override the streaming mode.
        **kwargs: Additional parameters forwarded to the provider
            constructor.

    Returns:
        A fully initialised :class:`LLMProvider`.

    Raises:
        LLMProviderError: If the provider slug is unknown or
            initialisation fails.
    """
    settings = get_settings()
    provider_type = (provider_type or settings.default_llm_provider or "openai").lower()

    if provider_type not in PROVIDER_REGISTRY:
        supported = ", ".join(get_available_providers())
        raise LLMProviderError(
            f"Unsupported LLM provider: '{provider_type}'. Supported: {supported}"
        )

    provider_class = PROVIDER_REGISTRY[provider_type]

    logger.debug(
        "Creating LLM provider – type=%s model=%s api=%s",
        provider_type,
        model_name or "default",
        api_type.value if api_type else "default",
    )

    try:
        # Build the constructor kwargs, omitting None values so that
        # each provider's own defaults (e.g. from Settings) can kick in.
        init_kwargs: dict[str, Any] = {}
        if config is not None:
            init_kwargs["config"] = config
        if model_name is not None:
            init_kwargs["model_name"] = model_name
        if api_type is not None:
            init_kwargs["api_type"] = api_type
        if stream_mode is not None:
            init_kwargs["stream_mode"] = stream_mode
        init_kwargs.update(kwargs)

        provider = provider_class(**init_kwargs)
        logger.info("Created %s provider successfully.", provider_type)
        return provider
    except Exception as exc:
        raise LLMProviderError(f"Failed to initialise '{provider_type}' provider: {exc}") from exc


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@lru_cache(maxsize=10)
def get_cached_provider(
    provider_type: str,
    model_name: str,
    api_type: str = "chat_completion",
    stream_mode: str = "disabled",
) -> LLMProvider:
    """Return a cached provider, or create and cache a new one.

    Only use this when provider configuration is **stable**.  For
    dynamic configurations use :func:`create_llm_provider` directly.

    Args:
        provider_type: Provider slug.
        model_name: Model name.
        api_type: API type as a string (``"chat_completion"`` or
            ``"responses"``).
        stream_mode: Stream mode as a string (``"enabled"``,
            ``"disabled"``, or ``"auto"``).

    Raises:
        ValueError: If *api_type* or *stream_mode* is not a valid enum value.
    """
    try:
        api_type_enum = APIType(api_type)
    except ValueError as exc:
        raise ValueError(
            f"Invalid api_type '{api_type}'. Valid options: {[e.value for e in APIType]}"
        ) from exc

    try:
        stream_mode_enum = StreamMode(stream_mode)
    except ValueError as exc:
        raise ValueError(
            f"Invalid stream_mode '{stream_mode}'. Valid options: {[e.value for e in StreamMode]}"
        ) from exc

    return create_llm_provider(
        provider_type=provider_type,
        model_name=model_name,
        api_type=api_type_enum,
        stream_mode=stream_mode_enum,
    )


def clear_provider_cache() -> None:
    """Invalidate all cached provider instances."""
    get_cached_provider.cache_clear()
    logger.info("Provider cache cleared.")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def get_default_provider() -> LLMProvider:
    """Return a provider configured entirely from application settings."""
    settings = get_settings()
    return create_llm_provider(
        provider_type=settings.default_llm_provider,
        model_name=settings.default_model_name,
    )


def get_llm_provider(provider_type: str = "openai", **kwargs: Any) -> LLMProvider:
    """Legacy alias for :func:`create_llm_provider`.

    .. deprecated::
        Use :func:`create_llm_provider` in new code.
    """
    logger.debug("get_llm_provider() is deprecated — use create_llm_provider().")
    return create_llm_provider(provider_type=provider_type, **kwargs)
