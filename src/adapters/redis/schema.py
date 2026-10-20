"""Data models for the Redis cache adapter."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CacheOperationType(Enum):
    """Key namespace for each kind of cached data."""

    SESSION = "session"
    RESPONSE_CHAIN = "response_chain"
    HISTORY = "history"
    STREAMING = "streaming"
    VECTOR_STORE = "vector_store"
    CHAT_MAPPING = "chat_mapping"


@dataclass
class CacheConfig:
    """Connection, TTL (seconds) and key settings for a cache provider."""

    host: str = "localhost"
    port: int = 6381
    db: int = 0
    password: str | None = None

    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = True
    ssl: bool = False

    default_session_ttl: int = 7200
    default_history_ttl: int = 1800
    default_streaming_ttl: int = 120
    default_vector_store_ttl: int = 86400

    key_prefix: str = "Asclepius_Conversations"
    max_chain_length: int = 20

    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: Any) -> "CacheConfig":
        """Build the config from the application's ``redis_*`` settings."""
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


@dataclass
class ActiveSessionContext:
    """State of a session that has not expired."""

    session_id: str
    metadata: dict[str, Any]
    last_response_id: str | None = None
    vector_store_id: str | None = None
    root_response_id: str | None = None
