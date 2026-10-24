"""Data models for the Redis cache adapter.

Big-picture flow:
1. Define the fixed set of "namespaces" cached data can belong to
   (sessions, chat history, streaming state, etc.).
2. Define ``RedisCacheConfig``: every setting needed to connect to Redis and
   decide how long things should be cached.
3. Define ``ActiveSessionContext``: the shape of a loaded, still-valid
   session's data.
"""

from dataclasses import dataclass, field

# @dataclass is a shortcut that writes the boring parts of a class (like
# __init__) for us, just from the fields listed below. "field" lets us give
# a field a safer default than writing "= {}" directly (see the note below).
from enum import Enum

# Enum creates a small, fixed set of named choices (like a multiple-choice
# question) instead of plain strings, so a typo is caught by your editor or
# type checker instead of silently creating a wrong Redis key.
from typing import Any

# "Any" is a type hint meaning "this could be any type; we are not
# restricting it."


# CacheKeyNamespace: the fixed list of "categories" of data this adapter
# caches. Needed so every cached key is automatically grouped and
# namespaced consistently (see base.py's _build_key), instead of each
# method inventing its own key format.
class CacheKeyNamespace(Enum):
    """Key namespace for each kind of cached data."""

    SESSION = "session"
    RESPONSE_CHAIN = "response_chain"
    HISTORY = "history"
    STREAMING = "streaming"
    VECTOR_STORE = "vector_store"
    CHAT_MAPPING = "chat_mapping"


# RedisCacheConfig: a plain data container holding every setting the Redis
# provider needs - connection details, timeouts, and how long different
# kinds of cached data should live (TTL = time to live) before expiring.
@dataclass
class RedisCacheConfig:
    """Connection, TTL (seconds) and key settings for a cache provider."""

    host: str = "localhost"
    # host: where to find the Redis server.
    port: int = 6381
    # port: which network port Redis is listening on.
    db: int = 0
    # db: which of Redis's 16 numbered databases to use.
    password: str | None = None
    # password: the credential used to authenticate to Redis, if required.

    max_connections: int = 50
    # max_connections: how many connections the connection pool may open at once.
    socket_timeout: int = 5
    # socket_timeout: how long to wait for Redis to respond to one command.
    socket_connect_timeout: int = 5
    # socket_connect_timeout: how long to wait while first opening the connection.
    decode_responses: bool = True
    # decode_responses: when True, Redis's raw bytes replies are
    # automatically turned into normal Python strings.
    ssl: bool = False
    # ssl: whether the connection to Redis should be encrypted.

    default_session_ttl: int = 7200
    # default_session_ttl: how long a session stays cached before expiring,
    # unless a call overrides it.
    default_history_ttl: int = 1800
    # default_history_ttl: how long cached chat history stays around.
    default_streaming_ttl: int = 120
    # default_streaming_ttl: how long in-progress streaming state stays cached.
    default_vector_store_ttl: int = 86400
    # default_vector_store_ttl: how long a cached vector-store validation
    # result stays valid.

    key_prefix: str = "Asclepius_Conversations"
    # key_prefix: text added to the front of every key this app stores in
    # Redis, so it does not collide with another app's keys.
    max_chain_length: int = 20
    # max_chain_length: the longest a stored list (like response history)
    # is allowed to grow before older entries are trimmed off.

    extra_params: dict[str, Any] = field(default_factory=dict)
    # extra_params: a catch-all dictionary for any extra settings not
    # listed above. "field(default_factory=dict)" is needed instead of
    # "= {}" because a mutable default like a dict would otherwise be
    # shared between every RedisCacheConfig instance - a classic Python trap
    # this avoids.

    @classmethod
    def from_settings(cls, settings: Any) -> "RedisCacheConfig":
        """Build the config from the application's ``redis_*`` settings."""
        # "@classmethod" means this method receives the CLASS itself
        # ("cls") instead of a specific instance ("self"), so it can be
        # called as "RedisCacheConfig.from_settings(app_settings)" without
        # needing an existing RedisCacheConfig first - it is a second way to
        # "construct" one, built from the app's global settings object.
        return cls(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            max_connections=settings.redis_max_connections,
            socket_timeout=settings.redis_socket_timeout,
            key_prefix=settings.redis_key_prefix,
            default_session_ttl=settings.redis_default_session_ttl,
            default_history_ttl=settings.redis_default_history_ttl,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the non-secret settings for logging."""
        # NOTE: "password" is deliberately left out here so it never ends
        # up in a log line.
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
            "key_prefix": self.key_prefix,
            "default_session_ttl": self.default_session_ttl,
            "default_history_ttl": self.default_history_ttl,
            "ssl": self.ssl,
        }


# ActiveSessionContext: a plain data container holding everything the app
# needs to know about one session that is currently loaded from the cache
# and still valid (not expired).
@dataclass
class ActiveSessionContext:
    """State of a session that has not expired."""

    session_id: str
    # session_id: the unique ID identifying this conversation/session.
    metadata: dict[str, Any]
    # metadata: whatever extra information was stored about this session.
    last_response_id: str | None = None
    # last_response_id: the latest OpenAI Responses API response ID in this session.
    vector_store_id: str | None = None
    # vector_store_id: the ID of the guideline vector store tied to this session, if any.
    root_response_id: str | None = None
    # root_response_id: the first Responses API response ID in this session's chain.
