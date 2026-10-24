"""Shared Redis configuration and key naming.

Big-picture flow:
1. Store the Redis settings (RedisCacheConfig) a provider was created with.
2. Log once, at creation time, which cache server this provider talks to.
3. Build Redis keys in one consistent format, so every method that stores
   or reads data uses the same naming rules.
"""

from src.logs import get_logger

from .schema import CacheKeyNamespace, RedisCacheConfig

logger = get_logger(__name__)
# A logger lets this file record what is happening (like "connected to
# this cache") without using plain print() statements.


# RedisCacheBase: shared Redis settings and key construction for the provider.
# This class provides common code; it does not define an abstract interface.
class RedisCacheBase:
    """Hold cache settings and build consistently prefixed keys."""

    def __init__(self, config: RedisCacheConfig) -> None:
        self.config = config
        # self.config: the connection/TTL/key settings this provider uses.
        provider_type = type(self).__name__.removesuffix("Provider").lower()
        # Turns a class name like "RedisProvider" into the short label
        # "redis", used only for this log line below.
        logger.info(
            "Initialized %s cache provider - Host: %s:%s, Key Prefix: %s",
            provider_type,
            config.host,
            config.port,
            config.key_prefix,
        )

    def _build_key(self, key_namespace: CacheKeyNamespace, *identifiers: str) -> str:
        """Build a Redis key from the prefix, namespace, and identifiers."""
        # "*identifiers" collects any number of extra string arguments the
        # caller passes (like a session_id, or a session_id plus
        # "messages") into one tuple, so this method works whether a key
        # needs one identifier or several.
        # Joins everything with ":" as a separator, e.g.
        # "Asclepius_Conversations:session:abc123" - a common Redis
        # convention for namespacing keys.
        return ":".join((self.config.key_prefix, key_namespace.value, *identifiers))
