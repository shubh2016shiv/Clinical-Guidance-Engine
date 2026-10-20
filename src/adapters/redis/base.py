"""Abstract cache provider interface.

Every method raises CacheOperationError on a backend failure and
CacheSerializationError when a payload cannot be encoded or decoded.
TTL arguments are in seconds; ``None`` selects the matching default in CacheConfig.
"""

from abc import ABC, abstractmethod
from typing import Any

from src.logs import get_logger

from .schema import ActiveSessionContext, CacheConfig, CacheOperationType

logger = get_logger(__name__)


class CacheProvider(ABC):
    """Contract that every cache backend implements."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.key_prefix = config.key_prefix
        self.provider_type = self.__class__.__name__.lower().replace("provider", "")

        logger.info(
            f"Initialized {self.provider_type} cache provider - "
            f"Host: {config.host}:{config.port}, "
            f"Key Prefix: {config.key_prefix}"
        )

    # ==================== Connection Management ====================

    @abstractmethod
    async def connect(self) -> None:
        """Connect and verify the server. Raises CacheConnectionError on failure."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the connection and release resources."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the server is connected and responsive."""

    # ==================== Session Management ====================

    @abstractmethod
    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Return the session metadata, or None if the session does not exist."""

    @abstractmethod
    async def set_session_metadata(
        self, session_id: str, metadata: dict[str, Any], ttl: int | None = None
    ) -> bool:
        """Store session metadata with a TTL. Returns True when stored."""

    @abstractmethod
    async def touch_session(self, session_id: str) -> bool:
        """Extend the TTL of the session's keys. Returns True if any key was extended."""

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete all of the session's keys. Returns True if any key was deleted."""

    @abstractmethod
    async def get_active_session_context(self, session_id: str) -> ActiveSessionContext | None:
        """Return the session's context only if the session exists and has not expired."""

    # ==================== Response Chain Management ====================

    @abstractmethod
    async def append_response(self, session_id: str, response_id: str) -> bool:
        """Append to the response chain, trimmed to config.max_chain_length."""

    @abstractmethod
    async def set_last_response_id(
        self, session_id: str, response_id: str, ttl: int | None = None
    ) -> bool:
        """Store the most recent response ID. Defaults to default_session_ttl."""

    # ==================== History Caching ====================

    @abstractmethod
    async def cache_history(
        self, session_id: str, history: list[dict[str, Any]], ttl: int | None = None
    ) -> bool:
        """Cache the conversation history as JSON. Defaults to default_history_ttl."""

    @abstractmethod
    async def get_cached_history(self, session_id: str) -> list[dict[str, Any]] | None:
        """Return the cached history, or None if it is not cached."""

    @abstractmethod
    async def invalidate_history(self, session_id: str) -> bool:
        """Delete the cached history. Returns True if it existed."""

    @abstractmethod
    async def add_message(
        self,
        session_id: str,
        message: dict[str, Any],
        max_messages: int | None = None,
    ) -> bool:
        """Add a message, keeping at most max_messages (default config.max_chain_length).

        Updates ``message_count`` in the session metadata when the session exists.
        """

    # ==================== Streaming State ====================

    @abstractmethod
    async def set_streaming_state(
        self,
        session_id: str,
        request_id: str,
        state: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Store streaming state. Defaults to default_streaming_ttl."""

    @abstractmethod
    async def clear_streaming_state(self, session_id: str, request_id: str) -> bool:
        """Delete the streaming state. Returns True if it existed."""

    # ==================== Vector Store Validation ====================

    @abstractmethod
    async def cache_vector_store_validation(
        self,
        vector_store_id: str,
        validation_data: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Cache a vector store validation result. Defaults to default_vector_store_ttl."""

    @abstractmethod
    async def get_vector_store_validation(self, vector_store_id: str) -> dict[str, Any] | None:
        """Return the cached validation result, or None."""

    def _build_key(self, operation_type: CacheOperationType, *identifiers: str) -> str:
        """Return ``{key_prefix}:{operation_type}:{identifier}:...``."""
        return ":".join([self.key_prefix, operation_type.value, *identifiers])
