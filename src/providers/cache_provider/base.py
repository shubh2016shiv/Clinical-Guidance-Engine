"""
Abstract Cache Provider Interface

This module provides an abstract interface for cache providers, allowing easy
extension to support different caching backends (Redis, Memcached, KeyDB, etc.).

The interface supports:
- Session management with TTL control
- Response chain tracking for conversation history
- Conversation history caching
- Tool execution state management
- Streaming state tracking
- Vector store validation caching
- Rate limiting
- Request correlation for debugging

Key Design Principles:
- All operations are async to match existing codebase patterns
- Comprehensive error handling with specific exception types
- Structured logging for all operations
- Type-safe interfaces with dataclasses
- Flexible TTL management per operation type
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from src.logs import get_logger

logger = get_logger(__name__)


class CacheOperationType(Enum):
    """
    Types of cache operations for different use cases

    Each operation type may have different TTL requirements and
    access patterns, allowing for optimized caching strategies.
    """

    SESSION = "session"  # Session metadata
    RESPONSE_CHAIN = "response_chain"  # Response ID chains
    HISTORY = "history"  # Conversation history
    TOOL_EXECUTION = "tool_execution"  # Tool execution state
    STREAMING = "streaming"  # Streaming state
    RATE_LIMIT = "rate_limit"  # Rate limiting counters
    VECTOR_STORE = "vector_store"  # Vector store validation
    REQUEST_CORRELATION = "request_correlation"  # Request-response correlation
    CHAT_MAPPING = "chat_mapping"  # Chat ID to response ID mapping


@dataclass
class CacheConfig:
    """
    Configuration container for cache provider settings

    This dataclass centralizes all configuration options for cache providers,
    making it easy to manage and extend settings across different backends.

    Connection Settings:
        host: Cache server host address
        port: Cache server port
        db: Database index (Redis-specific, 0-15)
        password: Authentication password (optional)
        max_connections: Maximum connections in pool
        socket_timeout: Socket timeout in seconds
        socket_connect_timeout: Connection timeout in seconds
        decode_responses: Auto-decode byte responses to strings
        ssl: Enable SSL/TLS connection

    TTL Settings (in seconds):
        default_session_ttl: Default TTL for session data (2 hours)
        default_history_ttl: Default TTL for history cache (30 minutes)
        default_tool_execution_ttl: Default TTL for tool execution state (5 minutes)
        default_streaming_ttl: Default TTL for streaming state (2 minutes)
        default_vector_store_ttl: Default TTL for vector store validation (24 hours)
        default_rate_limit_ttl: Default TTL for rate limit counters (1 hour)
        default_request_correlation_ttl: Default TTL for request correlation (1 hour)

    Key Management:
        key_prefix: Prefix for all cache keys (for namespacing)
        max_chain_length: Maximum length of response chains (trim older entries)

    Additional Settings:
        extra_params: Provider-specific additional parameters
    """

    # Connection configuration
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    # Connection pool settings
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    decode_responses: bool = True
    ssl: bool = False

    # Default TTL settings (in seconds)
    default_session_ttl: int = 7200  # 2 hours
    default_history_ttl: int = 1800  # 30 minutes
    default_tool_execution_ttl: int = 300  # 5 minutes
    default_streaming_ttl: int = 120  # 2 minutes
    default_vector_store_ttl: int = 86400  # 24 hours
    default_rate_limit_ttl: int = 3600  # 1 hour
    default_request_correlation_ttl: int = 3600  # 1 hour

    # Key management
    key_prefix: str = "drug_reco"
    max_chain_length: int = 20  # Maximum response chain length

    # Additional provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary for logging/serialization

        Returns:
            Dictionary representation of configuration
        """
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


class CacheProvider(ABC):
    """
    Abstract base class for cache providers

    This class defines the interface that all cache provider implementations
    must follow, ensuring consistency and interchangeability across different
    caching backends (Redis, Memcached, KeyDB, etc.).

    All methods are async to support non-blocking I/O operations and match
    the existing codebase patterns.

    Method Categories:
    - Connection Management: connect, disconnect, health_check
    - Session Management: get/set/touch/delete session metadata
    - Response Chain: append/get response chains for conversation history
    - History Caching: cache/get/invalidate conversation history
    - Tool Execution: manage tool execution state
    - Streaming: manage streaming state
    - Vector Store: cache vector store validation results
    - Rate Limiting: track and check request counts
    - Request Correlation: link requests to responses
    - Chat ID Mapping: map chat IDs to response IDs
    - Utility: exists, delete, expire, get_ttl
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize the cache provider with configuration

        Args:
            config: CacheConfig object containing provider settings
        """
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
        """
        Establish connection to the cache server

        This method should initialize the connection pool and verify
        connectivity to the cache server.

        Raises:
            CacheConnectionError: If connection fails
            CacheAuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the cache server

        This method should gracefully close all connections in the pool
        and release resources.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if cache server is healthy and responsive

        Returns:
            True if cache server is healthy, False otherwise
        """
        pass

    # ==================== Session Management ====================

    @abstractmethod
    async def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session metadata from cache

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with session metadata, or None if not found

        Raises:
            CacheOperationError: If retrieval fails
        """
        pass

    @abstractmethod
    async def set_session_metadata(
        self, session_id: str, metadata: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """
        Store session metadata in cache

        Args:
            session_id: Unique session identifier
            metadata: Dictionary containing session metadata
            ttl: Time-to-live in seconds (uses default_session_ttl if None)

        Returns:
            True if stored successfully, False otherwise

        Raises:
            CacheOperationError: If storage fails
            CacheSerializationError: If metadata serialization fails
        """
        pass

    @abstractmethod
    async def touch_session(self, session_id: str) -> bool:
        """
        Extend session TTL (update last access time)

        This method extends the TTL of all session-related keys
        to keep the session active.

        Args:
            session_id: Unique session identifier

        Returns:
            True if TTL extended successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete all session-related data from cache

        This method removes all keys associated with the session,
        including metadata, history, response chains, etc.

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CacheOperationError: If deletion fails
        """
        pass

    # ==================== Response Chain Management ====================

    @abstractmethod
    async def append_response(self, session_id: str, response_id: str) -> bool:
        """
        Append response ID to the session's response chain

        The response chain is maintained as a list, with automatic trimming
        to max_chain_length to prevent unbounded growth.

        Args:
            session_id: Unique session identifier
            response_id: Response ID to append

        Returns:
            True if appended successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    @abstractmethod
    async def get_response_chain(self, session_id: str) -> List[str]:
        """
        Retrieve the complete response chain for a session

        Args:
            session_id: Unique session identifier

        Returns:
            List of response IDs in chronological order (oldest first)
            Returns empty list if chain doesn't exist

        Raises:
            CacheOperationError: If retrieval fails
        """
        pass

    @abstractmethod
    async def get_last_response_id(self, session_id: str) -> Optional[str]:
        """
        Get the most recent response ID for a session

        Args:
            session_id: Unique session identifier

        Returns:
            Last response ID, or None if no responses exist

        Raises:
            CacheOperationError: If retrieval fails
        """
        pass

    @abstractmethod
    async def set_last_response_id(
        self, session_id: str, response_id: str, ttl: Optional[int] = None
    ) -> bool:
        """
        Set the last response ID for quick access

        This provides O(1) access to the most recent response without
        scanning the entire chain.

        Args:
            session_id: Unique session identifier
            response_id: Response ID to set
            ttl: Time-to-live in seconds (uses default_session_ttl if None)

        Returns:
            True if set successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    # ==================== History Caching ====================

    @abstractmethod
    async def cache_history(
        self, session_id: str, history: List[Dict[str, Any]], ttl: Optional[int] = None
    ) -> bool:
        """
        Cache conversation history for quick retrieval

        History is stored as JSON-serialized data for efficient storage
        and retrieval, reducing the need for database queries.

        Args:
            session_id: Unique session identifier
            history: List of message dictionaries
            ttl: Time-to-live in seconds (uses default_history_ttl if None)

        Returns:
            True if cached successfully, False otherwise

        Raises:
            CacheOperationError: If caching fails
            CacheSerializationError: If history serialization fails
        """
        pass

    @abstractmethod
    async def get_cached_history(
        self, session_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached conversation history

        Args:
            session_id: Unique session identifier

        Returns:
            List of message dictionaries, or None if not cached

        Raises:
            CacheOperationError: If retrieval fails
            CacheSerializationError: If deserialization fails
        """
        pass

    @abstractmethod
    async def invalidate_history(self, session_id: str) -> bool:
        """
        Invalidate (delete) cached conversation history

        This should be called when history is updated to ensure
        cache consistency.

        Args:
            session_id: Unique session identifier

        Returns:
            True if invalidated successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    # ==================== Tool Execution State ====================

    @abstractmethod
    async def set_tool_execution_state(
        self,
        session_id: str,
        request_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store tool execution state for tracking in-progress operations

        Tool execution state includes status, tool calls, start time, etc.
        This enables recovery and monitoring of tool execution.

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this tool execution
            state: Dictionary containing tool execution state
            ttl: Time-to-live in seconds (uses default_tool_execution_ttl if None)

        Returns:
            True if stored successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
            CacheSerializationError: If state serialization fails
        """
        pass

    @abstractmethod
    async def get_tool_execution_state(
        self, session_id: str, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve tool execution state

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this tool execution

        Returns:
            Dictionary with tool execution state, or None if not found

        Raises:
            CacheOperationError: If retrieval fails
            CacheSerializationError: If deserialization fails
        """
        pass

    @abstractmethod
    async def clear_tool_execution_state(
        self, session_id: str, request_id: str
    ) -> bool:
        """
        Clear tool execution state after completion

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this tool execution

        Returns:
            True if cleared successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    # ==================== Streaming State ====================

    @abstractmethod
    async def set_streaming_state(
        self,
        session_id: str,
        request_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store streaming state for active streaming sessions

        Streaming state includes response_id, status, chunk count, etc.
        Short TTL since this is temporary state.

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this streaming session
            state: Dictionary containing streaming state
            ttl: Time-to-live in seconds (uses default_streaming_ttl if None)

        Returns:
            True if stored successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
            CacheSerializationError: If state serialization fails
        """
        pass

    @abstractmethod
    async def get_streaming_state(
        self, session_id: str, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve streaming state

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this streaming session

        Returns:
            Dictionary with streaming state, or None if not found

        Raises:
            CacheOperationError: If retrieval fails
            CacheSerializationError: If deserialization fails
        """
        pass

    @abstractmethod
    async def clear_streaming_state(self, session_id: str, request_id: str) -> bool:
        """
        Clear streaming state after stream completion

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this streaming session

        Returns:
            True if cleared successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    # ==================== Vector Store Validation ====================

    @abstractmethod
    async def cache_vector_store_validation(
        self,
        vector_store_id: str,
        validation_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache vector store validation results

        Vector stores are relatively stable, so validation can be cached
        for longer periods to avoid repeated API calls.

        Args:
            vector_store_id: Vector store identifier
            validation_data: Dictionary containing validation results
            ttl: Time-to-live in seconds (uses default_vector_store_ttl if None)

        Returns:
            True if cached successfully, False otherwise

        Raises:
            CacheOperationError: If caching fails
            CacheSerializationError: If validation data serialization fails
        """
        pass

    @abstractmethod
    async def get_vector_store_validation(
        self, vector_store_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached vector store validation results

        Args:
            vector_store_id: Vector store identifier

        Returns:
            Dictionary with validation results, or None if not cached

        Raises:
            CacheOperationError: If retrieval fails
            CacheSerializationError: If deserialization fails
        """
        pass

    # ==================== Rate Limiting ====================

    @abstractmethod
    async def increment_request_count(
        self, session_id: str, window_seconds: int = 3600
    ) -> int:
        """
        Increment request count for rate limiting

        Uses a sliding window approach with sorted sets to track
        request timestamps for accurate rate limiting.

        Args:
            session_id: Unique session identifier
            window_seconds: Time window in seconds for rate limiting

        Returns:
            Current request count within the window

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    @abstractmethod
    async def get_request_count(
        self, session_id: str, window_seconds: int = 3600
    ) -> int:
        """
        Get current request count within time window

        Args:
            session_id: Unique session identifier
            window_seconds: Time window in seconds

        Returns:
            Number of requests within the time window

        Raises:
            CacheOperationError: If retrieval fails
        """
        pass

    @abstractmethod
    async def check_rate_limit(
        self, session_id: str, limit: int, window_seconds: int = 3600
    ) -> bool:
        """
        Check if session has exceeded rate limit

        Args:
            session_id: Unique session identifier
            limit: Maximum requests allowed within window
            window_seconds: Time window in seconds

        Returns:
            True if within limit, False if exceeded

        Raises:
            CacheOperationError: If check fails
        """
        pass

    # ==================== Request Correlation ====================

    @abstractmethod
    async def set_request_correlation(
        self,
        session_id: str,
        request_id: str,
        correlation_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store request correlation data for debugging and tracing

        Links request IDs to response IDs and includes metadata
        for troubleshooting.

        Args:
            session_id: Unique session identifier
            request_id: Request identifier
            correlation_data: Dictionary with correlation information
            ttl: Time-to-live in seconds (uses default_request_correlation_ttl if None)

        Returns:
            True if stored successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
            CacheSerializationError: If correlation data serialization fails
        """
        pass

    @abstractmethod
    async def get_request_correlation(
        self, session_id: str, request_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve request correlation data

        Args:
            session_id: Unique session identifier
            request_id: Request identifier

        Returns:
            Dictionary with correlation data, or None if not found

        Raises:
            CacheOperationError: If retrieval fails
            CacheSerializationError: If deserialization fails
        """
        pass

    # ==================== Chat ID Mapping ====================

    @abstractmethod
    async def set_chat_mapping(
        self, chat_id: str, response_id: str, ttl: Optional[int] = None
    ) -> bool:
        """
        Map chat ID to last response ID

        This replaces the in-memory _chat_cache in ChatManager,
        providing persistence across restarts and distributed deployments.

        Args:
            chat_id: Chat identifier
            response_id: Response identifier
            ttl: Time-to-live in seconds (uses default_session_ttl if None)

        Returns:
            True if mapped successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        pass

    @abstractmethod
    async def get_chat_mapping(self, chat_id: str) -> Optional[str]:
        """
        Get response ID for a chat ID

        Args:
            chat_id: Chat identifier

        Returns:
            Response ID, or None if mapping doesn't exist

        Raises:
            CacheOperationError: If retrieval fails
        """
        pass

    # ==================== Utility Methods ====================

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise

        Raises:
            CacheOperationError: If check fails
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache

        Args:
            key: Cache key to delete

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CacheOperationError: If deletion fails
        """
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for an existing key

        Args:
            key: Cache key
            ttl: Time-to-live in seconds

        Returns:
            True if TTL set successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
            CacheTTLError: If TTL value is invalid
        """
        pass

    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for a key

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, -1 if no TTL, None if key doesn't exist

        Raises:
            CacheOperationError: If retrieval fails
        """
        pass

    def _build_key(self, operation_type: CacheOperationType, *identifiers: str) -> str:
        """
        Build a cache key with consistent naming convention

        Format: {key_prefix}:{operation_type}:{identifier1}:{identifier2}:...

        Args:
            operation_type: Type of cache operation
            *identifiers: Variable number of identifiers (session_id, request_id, etc.)

        Returns:
            Formatted cache key
        """
        parts = [self.key_prefix, operation_type.value] + list(identifiers)
        return ":".join(parts)
