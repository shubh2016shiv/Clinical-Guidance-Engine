"""
Cache Provider Factory

This module provides factory functions for creating and managing cache provider
instances. It supports:
- Multiple provider types (Redis, future Memcached, KeyDB, etc.)
- Configuration-driven instantiation
- Provider registry management
- Default provider settings from config
- Cached provider instances for performance

Key Features:
- Registry-based provider system
- Dynamic provider registration
- LRU caching for provider instances
- Integration with application configuration
- Comprehensive error handling
"""

from typing import Dict, Optional, Type
from functools import lru_cache

from src.config import get_settings
from src.logs import get_logger
from .base import CacheProvider, CacheConfig
from .exceptions import CacheConfigError
from .redis_provider import RedisProvider

logger = get_logger(__name__)


# Provider registry - maps provider names to their implementation classes
CACHE_PROVIDER_REGISTRY: Dict[str, Type[CacheProvider]] = {
    "redis": RedisProvider,
    # Add more providers here as they're implemented
    # "memcached": MemcachedProvider,
    # "keydb": KeyDBProvider,
}


def register_cache_provider(name: str, provider_class: Type[CacheProvider]) -> None:
    """
    Register a new cache provider implementation

    This allows dynamic registration of custom provider implementations
    without modifying the factory code. Useful for plugins or
    custom cache backends.

    Args:
        name: Provider name/identifier (e.g., "redis", "memcached", "custom")
        provider_class: Provider class that implements CacheProvider interface

    Example:
        class CustomCacheProvider(CacheProvider):
            # Implementation
            pass

        register_cache_provider("custom", CustomCacheProvider)
        provider = create_cache_provider(provider_type="custom")
    """
    name = name.lower()

    if name in CACHE_PROVIDER_REGISTRY:
        logger.warning(
            f"Overwriting existing cache provider registration: {name}",
            component="Cache",
            subcomponent="Factory",
        )

    CACHE_PROVIDER_REGISTRY[name] = provider_class

    logger.info(
        f"Registered cache provider: {name} -> {provider_class.__name__}",
        component="Cache",
        subcomponent="Factory",
    )


def get_available_cache_providers() -> list[str]:
    """
    Get list of available cache provider names

    Returns:
        List of registered provider identifiers

    Example:
        providers = get_available_cache_providers()
        print(f"Available providers: {', '.join(providers)}")
        # Output: Available providers: redis
    """
    return list(CACHE_PROVIDER_REGISTRY.keys())


def create_cache_provider(
    provider_type: Optional[str] = None, config: Optional[CacheConfig] = None, **kwargs
) -> CacheProvider:
    """
    Factory function to create a cache provider instance

    This is the main entry point for creating cache providers. It supports
    multiple ways of specifying configuration:
    1. Complete CacheConfig object
    2. Individual parameters (host, port, password, etc.)
    3. Settings from environment/config file

    Args:
        provider_type: Provider identifier (e.g., "redis", "memcached")
                      Defaults to "redis"
        config: Complete CacheConfig object (takes precedence over kwargs)
        **kwargs: Additional configuration parameters:
            - host: Cache server host (default: from settings or "localhost")
            - port: Cache server port (default: from settings or 6379)
            - db: Database index (default: from settings or 0)
            - password: Authentication password (default: from settings)
            - max_connections: Connection pool size (default: from settings or 50)
            - socket_timeout: Socket timeout in seconds (default: from settings or 5)
            - key_prefix: Key prefix for namespacing (default: from settings or "drug_reco")
            - default_session_ttl: Default session TTL (default: from settings or 7200)
            - default_history_ttl: Default history TTL (default: from settings or 1800)
            - ... (see CacheConfig for all options)

    Returns:
        Configured cache provider instance

    Raises:
        CacheConfigError: If provider type is not supported or initialization fails

    Examples:
        # Using defaults from config
        provider = create_cache_provider()

        # Specify provider and connection
        provider = create_cache_provider(
            provider_type="redis",
            host="localhost",
            port=6379,
            password="redis123"
        )

        # Using complete config object
        config = CacheConfig(
            host="localhost",
            port=6379,
            password="redis123",
            default_session_ttl=7200
        )
        provider = create_cache_provider(provider_type="redis", config=config)

        # Auto-connect after creation
        await provider.connect()
    """
    settings = get_settings()

    # Determine provider type
    provider_type = (provider_type or "redis").lower()

    # Validate provider exists
    if provider_type not in CACHE_PROVIDER_REGISTRY:
        supported = ", ".join(get_available_cache_providers())
        error_msg = (
            f"Unsupported cache provider: '{provider_type}'. "
            f"Supported providers: {supported}"
        )
        logger.error(
            error_msg,
            component="Cache",
            subcomponent="Factory",
        )
        raise CacheConfigError(error_msg)

    # Get provider class
    provider_class = CACHE_PROVIDER_REGISTRY[provider_type]

    logger.debug(
        f"Creating cache provider - Type: {provider_type}",
        component="Cache",
        subcomponent="Factory",
    )

    try:
        # If no config provided, create one from kwargs or defaults
        if config is None:
            # Extract configuration parameters with fallback to settings
            config_params = {
                # Connection settings
                "host": kwargs.pop(
                    "host", getattr(settings, "redis_host", "localhost")
                ),
                "port": kwargs.pop("port", getattr(settings, "redis_port", 6379)),
                "db": kwargs.pop("db", getattr(settings, "redis_db", 0)),
                "password": kwargs.pop(
                    "password", getattr(settings, "redis_password", None)
                ),
                # Connection pool settings
                "max_connections": kwargs.pop(
                    "max_connections", getattr(settings, "redis_max_connections", 50)
                ),
                "socket_timeout": kwargs.pop(
                    "socket_timeout", getattr(settings, "redis_socket_timeout", 5)
                ),
                "socket_connect_timeout": kwargs.pop("socket_connect_timeout", 5),
                "decode_responses": kwargs.pop("decode_responses", True),
                "ssl": kwargs.pop("ssl", False),
                # TTL settings
                "default_session_ttl": kwargs.pop(
                    "default_session_ttl",
                    getattr(settings, "redis_default_session_ttl", 7200),
                ),
                "default_history_ttl": kwargs.pop(
                    "default_history_ttl",
                    getattr(settings, "redis_default_history_ttl", 1800),
                ),
                "default_tool_execution_ttl": kwargs.pop(
                    "default_tool_execution_ttl", 300
                ),
                "default_streaming_ttl": kwargs.pop("default_streaming_ttl", 120),
                "default_vector_store_ttl": kwargs.pop(
                    "default_vector_store_ttl", 86400
                ),
                "default_rate_limit_ttl": kwargs.pop("default_rate_limit_ttl", 3600),
                "default_request_correlation_ttl": kwargs.pop(
                    "default_request_correlation_ttl", 3600
                ),
                # Key management
                "key_prefix": kwargs.pop(
                    "key_prefix",
                    getattr(settings, "redis_key_prefix", "Asclepius_Conversations"),
                ),
                "max_chain_length": kwargs.pop("max_chain_length", 20),
                # Additional provider-specific settings
                "extra_params": kwargs,  # Pass remaining kwargs as extra_params
            }

            config = CacheConfig(**config_params)

        # Create provider instance
        provider = provider_class(config=config)

        logger.info(
            f"Successfully created {provider_type} cache provider",
            component="Cache",
            subcomponent="Factory",
            host=config.host,
            port=config.port,
        )

        return provider

    except Exception as e:
        error_msg = f"Failed to initialize {provider_type} cache provider: {str(e)}"
        logger.error(
            error_msg,
            component="Cache",
            subcomponent="Factory",
            exc_info=True,
        )
        raise CacheConfigError(error_msg)


@lru_cache(maxsize=5)
def get_cached_cache_provider(
    provider_type: str = "redis",
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
) -> CacheProvider:
    """
    Get or create a cached cache provider instance

    This function caches provider instances to avoid repeated initialization
    overhead. Useful when using the same provider configuration multiple times
    across the application.

    Note: Only use caching when provider configuration is stable. For dynamic
    configurations or testing, use create_cache_provider() directly.

    The cache key is based on provider type and connection parameters,
    so different configurations will create separate cached instances.

    Args:
        provider_type: Provider identifier (default: "redis")
        host: Cache server host (default: "localhost")
        port: Cache server port (default: 6379)
        db: Database index (default: 0)

    Returns:
        Cached or newly created provider instance

    Raises:
        CacheConfigError: If provider creation fails

    Example:
        # First call creates and caches provider
        provider1 = get_cached_cache_provider("redis", "localhost", 6379)

        # Second call with same params returns cached instance
        provider2 = get_cached_cache_provider("redis", "localhost", 6379)

        # provider1 is provider2 == True
    """
    return create_cache_provider(
        provider_type=provider_type,
        host=host,
        port=port,
        db=db,
    )


def clear_cache_provider_cache() -> None:
    """
    Clear the provider cache

    Call this when you need to force recreation of cached providers,
    for example after configuration changes or during testing.

    Example:
        # Modify configuration
        os.environ['REDIS_HOST'] = 'new-host'

        # Clear cache to pick up new config
        clear_cache_provider_cache()

        # Next call will create provider with new config
        provider = get_cached_cache_provider()
    """
    get_cached_cache_provider.cache_clear()

    logger.info(
        "Cache provider cache cleared",
        component="Cache",
        subcomponent="Factory",
    )


def get_default_cache_provider() -> CacheProvider:
    """
    Get the default cache provider based on application settings

    This is a convenience function that creates a provider using
    all defaults from the configuration system. It reads settings
    from environment variables or config file.

    Returns:
        Default configured provider instance (Redis by default)

    Raises:
        CacheConfigError: If provider creation fails

    Example:
        # Uses all settings from config.py/.env
        provider = get_default_cache_provider()
        await provider.connect()

        # Use provider
        await provider.set_session_metadata(session_id, metadata)
    """
    settings = get_settings()

    # Create provider with all settings from config
    return create_cache_provider(
        provider_type="redis",  # Default to Redis
        host=getattr(settings, "redis_host", "localhost"),
        port=getattr(settings, "redis_port", 6379),
        db=getattr(settings, "redis_db", 0),
        password=getattr(settings, "redis_password", None),
        max_connections=getattr(settings, "redis_max_connections", 50),
        socket_timeout=getattr(settings, "redis_socket_timeout", 5),
        key_prefix=getattr(settings, "redis_key_prefix", "Asclepius_Conversations"),
        default_session_ttl=getattr(settings, "redis_default_session_ttl", 7200),
        default_history_ttl=getattr(settings, "redis_default_history_ttl", 1800),
    )


# Convenience function for backward compatibility
def get_cache_provider(provider_type: str = "redis", **kwargs) -> CacheProvider:
    """
    Legacy factory function - use create_cache_provider() for new code

    This function is maintained for backward compatibility with existing code
    that may use the older naming convention.

    Args:
        provider_type: Provider identifier
        **kwargs: Additional configuration parameters

    Returns:
        Configured cache provider instance

    Example:
        # Legacy usage (still works)
        provider = get_cache_provider("redis", host="localhost")

        # Preferred usage for new code
        provider = create_cache_provider("redis", host="localhost")
    """
    logger.debug(
        "Using legacy get_cache_provider() - consider using create_cache_provider()",
        component="Cache",
        subcomponent="Factory",
    )
    return create_cache_provider(provider_type=provider_type, **kwargs)
