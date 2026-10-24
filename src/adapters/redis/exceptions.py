"""Cache provider exceptions.

CacheProviderError
├── CacheConnectionError
│   └── CacheAuthenticationError
├── CacheConfigError
├── CacheOperationError
├── CacheSerializationError
├── CacheKeyError
└── CacheTTLError

Big-picture flow:
1. Give every Redis-related failure its own specific exception class.
2. Let callers catch broadly (``CacheProviderError``) or narrowly (e.g. just
   ``CacheAuthenticationError``) depending on how precisely they need to
   react.
3. Keep optional extra details (like which key failed) attached to the
   exception itself, instead of only in a plain text message.

Beginner note: the tree above is a "family tree" of exception classes.
"class Foo(Bar):" means "Foo IS-A Bar, plus maybe something extra" - so
catching the parent class also catches every class indented beneath it.
"""

from src.adapters.exceptions import ProviderError


# CacheProviderError: the shared parent for every cache-related exception
# in this adapter. Needed so callers can write one "except
# CacheProviderError" to handle ANY cache failure, instead of listing every
# specific exception type.
class CacheProviderError(ProviderError):
    """Base class for cache errors; ``details`` carries optional context."""

    def __init__(self, message: str, details: dict | None = None):
        # "__init__" is the special method Python calls automatically when
        # a new instance of this class is created - it sets up the object's
        # starting state.
        super().__init__(message, details)
        # "super().__init__(...)" calls the PARENT class's own __init__, so
        # whatever setup ProviderError normally does still happens here
        # too, instead of being skipped.
        self.message = message
        # self.message: the plain-text description of what went wrong.
        self.details = details or {}
        # self.details: extra structured context (like {"key": "session:1"})
        # about the failure. "details or {}" means: use the dict that was
        # passed in, or an empty dict if None was passed instead.

    def __str__(self):
        # "__str__" is the special method Python calls when this exception
        # is converted to text, e.g. by print(exc) or str(exc). Defining it
        # lets us control exactly how the error looks in logs and tracebacks.
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            # details_str: the extra context formatted as "key=value,
            # key2=value2" so it reads naturally in a log line.
            return f"{self.message} ({details_str})"
        return self.message


# CacheConnectionError: specifically means "could not reach or stay
# connected to the cache server" - narrower than the general error above,
# so callers can react differently (e.g. retry) to connection problems.
class CacheConnectionError(CacheProviderError):
    """The cache server is unreachable or the connection failed."""


# CacheAuthenticationError: a MORE specific kind of connection error -
# notice it inherits from CacheConnectionError, not directly from
# CacheProviderError, so catching "CacheConnectionError" also catches this.
class CacheAuthenticationError(CacheConnectionError):
    """The cache server rejected the credentials."""


# CacheConfigError: the settings given to the cache provider are wrong,
# independent of whether the server itself is reachable.
class CacheConfigError(CacheProviderError):
    """The cache configuration is invalid."""


# CacheOperationError: a specific Redis command (like "set" or "get")
# failed while actually running, as opposed to a connection or config problem.
class CacheOperationError(CacheProviderError):
    """A cache command failed."""


# CacheSerializationError: a value could not be turned into JSON text to
# store, or turned back from JSON text after reading - usually means the
# data itself was not JSON-friendly (see redis_store.py's _decode_redis_value()).
class CacheSerializationError(CacheProviderError):
    """A value could not be encoded to or decoded from JSON."""


# CacheKeyError: the Redis key used to store/read something was invalid.
class CacheKeyError(CacheProviderError):
    """A cache key is invalid."""


# CacheTTLError: the expiry time (TTL) given for a cached value was invalid.
class CacheTTLError(CacheProviderError):
    """A TTL value is invalid."""
