"""
Cache Provider Module

This module provides a centralized, professional interface for working with
cache providers (Redis, Memcached, KeyDB, etc.).

Key Features:
- Multiple provider support (Redis with plans for Memcached, KeyDB)
- Session management with TTL control
- Response chain tracking for conversation history
- Conversation history caching
- Tool execution state management
- Streaming state tracking
- Vector store validation caching
- Rate limiting with sliding window
- Request correlation for debugging
- Async operations for non-blocking I/O
- Comprehensive error handling
- Type-safe interfaces with dataclasses

Quick Start:
    from src.providers.cache_provider import (
        create_cache_provider,
        CacheOperationType
    )

    # Create a Redis provider
    provider = create_cache_provider(
        provider_type="redis",
        host="localhost",
        port=6379,
        password="redis123"
    )

    # Connect to cache server
    await provider.connect()

    # Store session metadata
    await provider.set_session_metadata(
        session_id="uuid-123",
        metadata={
            "user_id": "user-456",
            "created_at": "2024-01-01T00:00:00Z",
            "last_active_at": "2024-01-01T12:00:00Z"
        }
    )

    # Retrieve session metadata
    metadata = await provider.get_session_metadata("uuid-123")

    # Append to response chain
    await provider.append_response("uuid-123", "resp-789")

    # Cache conversation history
    await provider.cache_history("uuid-123", history_list)

    # Rate limiting
    within_limit = await provider.check_rate_limit("uuid-123", limit=100)

    # Disconnect when done
    await provider.disconnect()

See README.md for comprehensive documentation and usage examples.
"""

# Base classes and interfaces
from .base import (
    CacheProvider,
    CacheConfig,
    CacheOperationType,
    ActiveSessionContext,
)

# Factory functions
from .factory import (
    create_cache_provider,
    get_cache_provider,  # Legacy
    get_default_cache_provider,
    get_cached_cache_provider,
    clear_cache_provider_cache,
    register_cache_provider,
    get_available_cache_providers,
)

# Provider implementations
from .redis_provider import RedisProvider

# Exceptions
from .exceptions import (
    CacheProviderError,
    CacheConnectionError,
    CacheAuthenticationError,
    CacheConfigError,
    CacheOperationError,
    CacheSerializationError,
    CacheKeyError,
    CacheTTLError,
)


# Version
__version__ = "1.0.0"

# Public API
__all__ = [
    # Base classes
    "CacheProvider",
    "CacheConfig",
    "CacheOperationType",
    "ActiveSessionContext",
    # Factory functions
    "create_cache_provider",
    "get_cache_provider",
    "get_default_cache_provider",
    "get_cached_cache_provider",
    "clear_cache_provider_cache",
    "register_cache_provider",
    "get_available_cache_providers",
    # Providers
    "RedisProvider",
    # Exceptions
    "CacheProviderError",
    "CacheConnectionError",
    "CacheAuthenticationError",
    "CacheConfigError",
    "CacheOperationError",
    "CacheSerializationError",
    "CacheKeyError",
    "CacheTTLError",
]
