"""
LLM Provider Module

This module provides a centralized, professional interface for working with
Large Language Model providers.

Key Features:
- Multiple provider support (OpenAI, Gemini, etc.)
- Dual API support (Chat Completions & Responses API)
- Streaming and non-streaming modes
- Flexible configuration management
- Type-safe interfaces with dataclasses
- Comprehensive error handling
- Token validation and estimation

Quick Start:
    from src.providers import create_llm_provider, APIType, StreamMode

    # Create a provider
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.ENABLED
    )

    # Execute
    response = await provider.execute(
        prompt="Your prompt here",
        system_prompt="System instructions"
    )

See usage_examples.py for comprehensive examples.
"""

# Base classes and interfaces
from .base import (
    LLMProvider,
    LLMConfig,
    LLMResponse,
    APIType,
    StreamMode,
)

# Factory functions
from .factory import (
    create_llm_provider,
    get_llm_provider,  # Legacy
    get_default_provider,
    get_cached_provider,
    clear_provider_cache,
    register_provider,
    get_available_providers,
)

# Provider implementations
from .openai_provider import OpenAIProvider

# Exceptions
from .exceptions import (
    LLMProviderError,
    LLMProviderInitializationError,
    LLMProviderExecutionError,
    LLMProviderTokenLimitError,
    LLMProviderStreamingError,
    LLMProviderAuthenticationError,
    LLMProviderRateLimitError,
    LLMProviderUnsupportedOperationError,
)


# Version
__version__ = "2.0.0"

# Public API
__all__ = [
    # Base classes
    "LLMProvider",
    "LLMConfig",
    "LLMResponse",
    "APIType",
    "StreamMode",
    # Factory functions
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
