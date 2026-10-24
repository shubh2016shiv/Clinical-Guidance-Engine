"""Redis-backed application state persistence.

Beginner note: this file is this package's "__init__.py", so whatever it
imports becomes available directly as ``from src.adapters.redis import
RedisProvider`` instead of the longer
``from src.adapters.redis.redis_store import RedisProvider``.
"""

from .base import RedisCacheBase
from .redis_store import RedisProvider
from .schema import ActiveSessionContext, RedisCacheConfig

# __all__ lists the names meant for other modules to import from here.
__all__ = ["ActiveSessionContext", "RedisCacheBase", "RedisCacheConfig", "RedisProvider"]
