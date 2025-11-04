"""
Cache Provider Exceptions

This module defines custom exceptions for cache provider operations,
allowing for better error handling, debugging, and recovery strategies.

Exception Hierarchy:
    CacheProviderError (base)
    ├── CacheConnectionError
    │   └── CacheAuthenticationError
    ├── CacheConfigError
    ├── CacheOperationError
    ├── CacheSerializationError
    ├── CacheKeyError
    └── CacheTTLError

All exceptions inherit from ProviderError for consistency with other
provider implementations in the codebase.
"""

from src.providers.exceptions import ProviderError


class CacheProviderError(ProviderError):
    """
    Base exception for cache provider errors

    All cache-specific exceptions should inherit from this class to allow
    for unified error handling across different cache backends.

    This exception can be caught to handle any cache-related error generically,
    or specific subclasses can be caught for fine-grained error handling.

    Attributes:
        message: Error message describing what went wrong
        details: Optional dictionary with additional error context

    Example:
        try:
            await cache.get_session_metadata(session_id)
        except CacheProviderError as e:
            logger.error(f"Cache operation failed: {e}")
            # Fallback to database or return default
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception

        Args:
            message: Error message
            details: Optional dictionary with additional context
        """
        super().__init__(message, details)
        self.message = message
        self.details = details or {}

    def __str__(self):
        """String representation of the error"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class CacheConnectionError(CacheProviderError):
    """
    Raised when cache connection fails

    This exception indicates issues with establishing or maintaining
    a connection to the cache server.

    Common Causes:
    - Cache server is down or unreachable
    - Network connectivity issues
    - Connection timeout
    - Connection pool exhausted
    - DNS resolution failure

    Recovery Strategies:
    - Implement retry logic with exponential backoff
    - Fall back to database or in-memory cache
    - Return cached data if available (stale-while-revalidate)
    - Alert monitoring systems

    Example:
        try:
            await cache.connect()
        except CacheConnectionError as e:
            logger.error(f"Failed to connect to cache: {e}")
            # Fall back to in-memory cache or database
    """

    pass


class CacheAuthenticationError(CacheConnectionError):
    """
    Raised when cache authentication fails

    This exception indicates authentication issues when connecting
    to a password-protected cache server.

    Common Causes:
    - Invalid password
    - Missing password when required
    - Expired credentials
    - Permission denied
    - ACL restrictions (Redis 6+)

    Recovery Strategies:
    - Verify credentials in configuration
    - Check environment variables
    - Ensure password matches server configuration
    - Verify ACL permissions if using Redis 6+

    Example:
        try:
            await cache.connect()
        except CacheAuthenticationError as e:
            logger.error(f"Cache authentication failed: {e}")
            # Check credentials and configuration
    """

    pass


class CacheConfigError(CacheProviderError):
    """
    Raised when cache configuration is invalid

    This exception indicates problems with cache provider configuration,
    such as missing required settings or invalid parameter values.

    Common Causes:
    - Missing required configuration parameters
    - Invalid port number or host
    - Incompatible configuration combinations
    - Invalid TTL values (negative or zero when not allowed)
    - Invalid database index (Redis)

    Recovery Strategies:
    - Validate configuration at startup
    - Use configuration validation schemas
    - Provide clear error messages for misconfiguration
    - Document required vs optional configuration

    Example:
        try:
            config = CacheConfig(port=-1)  # Invalid port
        except CacheConfigError as e:
            logger.error(f"Invalid cache configuration: {e}")
    """

    pass


class CacheOperationError(CacheProviderError):
    """
    Raised when cache operation fails

    This exception covers failures during cache operations like
    get, set, delete, or any other cache manipulation.

    Common Causes:
    - Network error during operation
    - Cache server error
    - Operation timeout
    - Memory limit exceeded on cache server
    - Key too large
    - Value too large

    Recovery Strategies:
    - Retry operation with exponential backoff
    - Fall back to database query
    - Log operation details for debugging
    - Monitor cache server health

    Example:
        try:
            await cache.set_session_metadata(session_id, metadata)
        except CacheOperationError as e:
            logger.error(f"Failed to set cache: {e}")
            # Continue without caching, or retry
    """

    pass


class CacheSerializationError(CacheProviderError):
    """
    Raised when data serialization/deserialization fails

    This exception indicates problems converting data to/from the format
    required for cache storage (typically JSON).

    Common Causes:
    - Non-serializable objects (datetime, custom classes without __dict__)
    - Circular references in data structures
    - Invalid JSON in cached data (corruption)
    - Encoding issues
    - Data type mismatches

    Recovery Strategies:
    - Implement custom serializers for complex types
    - Validate data before caching
    - Use data validation libraries (Pydantic)
    - Invalidate corrupted cache entries
    - Convert problematic types (datetime to ISO string)

    Example:
        try:
            await cache.cache_history(session_id, history)
        except CacheSerializationError as e:
            logger.error(f"Failed to serialize history: {e}")
            # Skip caching and query database directly
    """

    pass


class CacheKeyError(CacheProviderError):
    """
    Raised when cache key is invalid or missing

    This exception indicates problems with cache key format, naming,
    or when attempting to access keys that don't exist when required.

    Common Causes:
    - Empty or None key
    - Key contains invalid characters
    - Key exceeds maximum length
    - Key namespace collision
    - Attempting to access non-existent key when existence is required

    Recovery Strategies:
    - Validate keys before use
    - Sanitize key names
    - Use consistent key naming conventions
    - Check key existence before operations that require it

    Example:
        try:
            key = cache._build_key(operation_type, "")  # Empty identifier
        except CacheKeyError as e:
            logger.error(f"Invalid cache key: {e}")
    """

    pass


class CacheTTLError(CacheProviderError):
    """
    Raised when TTL-related operation fails

    This exception indicates problems with Time-To-Live (expiration)
    settings or operations.

    Common Causes:
    - Invalid TTL value (negative when not allowed)
    - TTL value exceeds maximum allowed
    - Setting TTL on key that doesn't support it
    - TTL operation on key that doesn't exist

    Recovery Strategies:
    - Validate TTL values before operations
    - Use default TTL values when custom values are invalid
    - Document TTL constraints
    - Handle keys without TTL gracefully

    Example:
        try:
            await cache.expire(key, -100)  # Invalid negative TTL
        except CacheTTLError as e:
            logger.error(f"Invalid TTL: {e}")
            # Use default TTL instead
    """

    pass
