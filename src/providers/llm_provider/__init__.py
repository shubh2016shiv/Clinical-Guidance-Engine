"""
LLM Provider Module.

Provides a centralised interface for working with Large Language Model
providers (OpenAI, Gemini, Anthropic, etc.).

Key features:
- Multiple provider support via a pluggable registry.
- Dual API support (Chat Completions & Responses API).
- Streaming and non-streaming modes.
- Type-safe configuration and response dataclasses.
- Comprehensive, correctly-chained exception hierarchy.

Quick start::

    from src.providers.llm_provider import create_llm_provider, APIType, StreamMode

    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4o",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.ENABLED,
    )

    response = await provider.execute(
        prompt="Your prompt here",
        system_prompt="System instructions",
    )
"""

from .base import APIType, LLMConfig, LLMProvider, LLMResponse, StreamMode
from .exceptions import (
    LLMProviderAuthenticationError,
    LLMProviderError,
    LLMProviderExecutionError,
    LLMProviderInitializationError,
    LLMProviderRateLimitError,
    LLMProviderStreamingError,
    LLMProviderTokenLimitError,
    LLMProviderUnsupportedOperationError,
)
from .factory import (
    clear_provider_cache,
    create_llm_provider,
    get_available_providers,
    get_cached_provider,
    get_default_provider,
    get_llm_provider,  # legacy
    register_provider,
)
from .openai_provider import OpenAIProvider

__version__ = "2.1.0"

__all__ = [
    # Base classes
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "APIType",
    "StreamMode",
    # Factory
    "create_llm_provider",
    "get_llm_provider",
    "get_default_provider",
    "get_cached_provider",
    "clear_provider_cache",
    "register_provider",
    "get_available_providers",
    # Providers
    "OpenAIProvider",
    # Exceptions
    "LLMProviderError",
    "LLMProviderInitializationError",
    "LLMProviderExecutionError",
    "LLMProviderTokenLimitError",
    "LLMProviderStreamingError",
    "LLMProviderAuthenticationError",
    "LLMProviderRateLimitError",
    "LLMProviderUnsupportedOperationError",
]
