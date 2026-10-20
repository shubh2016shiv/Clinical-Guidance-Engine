"""Cache provider exceptions.

CacheProviderError
├── CacheConnectionError
│   └── CacheAuthenticationError
├── CacheConfigError
├── CacheOperationError
├── CacheSerializationError
├── CacheKeyError
└── CacheTTLError
"""

from src.adapters.exceptions import ProviderError


class CacheProviderError(ProviderError):
    """Base class for cache errors; ``details`` carries optional context."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message, details)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class CacheConnectionError(CacheProviderError):
    """The cache server is unreachable or the connection failed."""


class CacheAuthenticationError(CacheConnectionError):
    """The cache server rejected the credentials."""


class CacheConfigError(CacheProviderError):
    """The cache configuration is invalid."""


class CacheOperationError(CacheProviderError):
    """A cache command failed."""


class CacheSerializationError(CacheProviderError):
    """A value could not be encoded to or decoded from JSON."""


class CacheKeyError(CacheProviderError):
    """A cache key is invalid."""


class CacheTTLError(CacheProviderError):
    """A TTL value is invalid."""
