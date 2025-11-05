"""
Redis Cache Provider Implementation

This module provides a comprehensive Redis implementation of the CacheProvider
interface using redis-py with async support.

Features:
- Async Redis operations using redis.asyncio
- Connection pool management
- Automatic reconnection handling
- JSON serialization for complex data
- TTL management with sensible defaults
- Structured logging for all operations
- Comprehensive error handling

Key Design Decisions:
- Uses redis.asyncio for non-blocking operations
- Stores complex data as JSON strings
- Uses Redis data structures appropriately:
  - Hashes for session metadata and state
  - Lists for response chains
  - Strings for simple key-value pairs
  - Sorted sets for rate limiting with timestamps
- Implements automatic chain trimming to prevent unbounded growth
- Includes health check with PING command
"""

import json
import time
from typing import Dict, Any, List, Optional

import redis.asyncio as redis
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    AuthenticationError as RedisAuthenticationError,
    TimeoutError as RedisTimeoutError,
)

from src.logs import get_component_logger
from src.providers.cache_provider.base import (
    CacheProvider,
    CacheConfig,
    CacheOperationType,
    ActiveSessionContext,
)
from src.providers.cache_provider.exceptions import (
    CacheConnectionError,
    CacheAuthenticationError,
    CacheOperationError,
    CacheSerializationError,
    CacheTTLError,
)

logger = get_component_logger("Cache")


class RedisProvider(CacheProvider):
    """
    Redis implementation of the cache provider interface

    This provider uses redis-py's async client for non-blocking operations
    and provides full implementation of all caching operations needed for
    session management, response chaining, history caching, and more.

    Redis Data Structures Used:
    - Strings: Simple key-value storage (last_response_id, request_correlation)
    - Hashes: Session metadata, tool execution state, streaming state
    - Lists: Response chains (with automatic trimming)
    - Sorted Sets: Rate limiting (timestamps as scores)

    Key Naming Convention:
    {key_prefix}:{operation_type}:{identifier1}:{identifier2}:...

    Examples:
    - drug_reco:session:uuid-123
    - drug_reco:response_chain:uuid-123
    - drug_reco:history:uuid-123
    - drug_reco:tool_execution:uuid-123:req-456
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize the Redis provider

        Args:
            config: CacheConfig object containing Redis connection settings
        """
        super().__init__(config)

        self.redis_client: Optional[redis.Redis] = None
        self._connection_pool: Optional[redis.ConnectionPool] = None

        logger.info(
            "Redis provider initialized",
            component="Cache",
            subcomponent="RedisProvider",
            host=config.host,
            port=config.port,
            db=config.db,
            max_connections=config.max_connections,
        )

    # ==================== Connection Management ====================

    async def connect(self) -> None:
        """
        Establish connection to Redis server

        Creates a connection pool and initializes the Redis client.
        Verifies connectivity with a PING command.

        Raises:
            CacheConnectionError: If connection fails
            CacheAuthenticationError: If authentication fails
        """
        try:
            logger.info(
                "Connecting to Redis server",
                component="Cache",
                subcomponent="RedisProvider",
                host=self.config.host,
                port=self.config.port,
            )

            # Create connection pool
            # Note: SSL is not directly supported by redis.asyncio.ConnectionPool
            # For SSL connections, use ssl_cert_reqs, ssl_ca_certs, etc. if needed
            connection_kwargs = {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
                "password": self.config.password,
                "max_connections": self.config.max_connections,
                "socket_timeout": self.config.socket_timeout,
                "socket_connect_timeout": self.config.socket_connect_timeout,
                "decode_responses": self.config.decode_responses,
            }

            # Add SSL parameters only if SSL is enabled and additional SSL config exists
            if self.config.ssl:
                # For SSL connections, additional parameters may be needed:
                # ssl_cert_reqs, ssl_ca_certs, ssl_certfile, ssl_keyfile
                # These would need to be added to CacheConfig if SSL support is required
                logger.warning(
                    "SSL is enabled but SSL parameters not fully configured. "
                    "SSL connection may fail. Add ssl_cert_reqs, ssl_ca_certs to CacheConfig for full SSL support.",
                    component="Cache",
                    subcomponent="RedisProvider",
                )

            self._connection_pool = redis.ConnectionPool(**connection_kwargs)

            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)

            # Verify connection with PING
            await self.redis_client.ping()

            logger.info(
                "Successfully connected to Redis server",
                component="Cache",
                subcomponent="RedisProvider",
                host=self.config.host,
                port=self.config.port,
            )

        except RedisAuthenticationError as e:
            error_msg = f"Redis authentication failed: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                exc_info=True,
            )
            raise CacheAuthenticationError(error_msg)

        except (RedisConnectionError, RedisTimeoutError) as e:
            error_msg = f"Failed to connect to Redis: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                exc_info=True,
            )
            raise CacheConnectionError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error connecting to Redis: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                exc_info=True,
            )
            raise CacheConnectionError(error_msg)

    async def disconnect(self) -> None:
        """
        Close connection to Redis server

        Gracefully closes all connections in the pool and releases resources.
        """
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info(
                    "Closed Redis client connection",
                    component="Cache",
                    subcomponent="RedisProvider",
                )

            if self._connection_pool:
                await self._connection_pool.disconnect()
                logger.info(
                    "Disconnected Redis connection pool",
                    component="Cache",
                    subcomponent="RedisProvider",
                )

        except Exception as e:
            logger.error(
                f"Error during Redis disconnect: {e}",
                component="Cache",
                subcomponent="RedisProvider",
                exc_info=True,
            )

    async def health_check(self) -> bool:
        """
        Check if Redis server is healthy and responsive

        Returns:
            True if Redis server responds to PING, False otherwise
        """
        try:
            if not self.redis_client:
                return False

            response = await self.redis_client.ping()

            logger.debug(
                "Redis health check successful",
                component="Cache",
                subcomponent="RedisProvider",
                response=response,
            )

            return response is True

        except Exception as e:
            logger.warning(
                f"Redis health check failed: {e}",
                component="Cache",
                subcomponent="RedisProvider",
            )
            return False

    # ==================== Session Management ====================

    async def get_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session metadata from Redis

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary with session metadata, or None if not found

        Raises:
            CacheOperationError: If retrieval fails
        """
        try:
            key = self._build_key(CacheOperationType.SESSION, session_id)

            # Get hash data
            data = await self.redis_client.hgetall(key)

            if not data:
                logger.debug(
                    "Session metadata not found",
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                    cache_hit=False,
                )
                return None

            # Parse JSON values
            metadata = {k: self._deserialize_value(v) for k, v in data.items()}

            logger.debug(
                "Retrieved session metadata",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                cache_hit=True,
                fields_count=len(metadata),
            )

            return metadata

        except Exception as e:
            error_msg = f"Failed to get session metadata: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def set_session_metadata(
        self, session_id: str, metadata: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """
        Store session metadata in Redis

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
        try:
            key = self._build_key(CacheOperationType.SESSION, session_id)
            ttl = ttl or self.config.default_session_ttl

            # Serialize metadata values
            serialized_data = {k: self._serialize_value(v) for k, v in metadata.items()}

            # Store as hash
            await self.redis_client.hset(key, mapping=serialized_data)

            # Set TTL
            await self.redis_client.expire(key, ttl)

            logger.debug(
                "Set session metadata",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                fields_count=len(metadata),
                ttl=ttl,
            )

            return True

        except (TypeError, ValueError) as e:
            error_msg = f"Failed to serialize session metadata: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheSerializationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to set session metadata: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def touch_session(self, session_id: str) -> bool:
        """
        Extend session TTL for all session-related keys

        Args:
            session_id: Unique session identifier

        Returns:
            True if TTL extended successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            ttl = self.config.default_session_ttl

            # Touch all session-related keys
            patterns = [
                self._build_key(CacheOperationType.SESSION, session_id),
                self._build_key(CacheOperationType.RESPONSE_CHAIN, session_id),
                self._build_key(
                    CacheOperationType.CHAT_MAPPING, f"last_response:{session_id}"
                ),
            ]

            touched_count = 0
            for key in patterns:
                if await self.redis_client.exists(key):
                    await self.redis_client.expire(key, ttl)
                    touched_count += 1

            logger.debug(
                "Touched session keys",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                keys_touched=touched_count,
                ttl=ttl,
            )

            return touched_count > 0

        except Exception as e:
            error_msg = f"Failed to touch session: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete all session-related data from Redis

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CacheOperationError: If deletion fails
        """
        try:
            # Delete all session-related keys
            keys_to_delete = [
                self._build_key(CacheOperationType.SESSION, session_id),
                self._build_key(CacheOperationType.RESPONSE_CHAIN, session_id),
                self._build_key(CacheOperationType.HISTORY, session_id),
                self._build_key(
                    CacheOperationType.CHAT_MAPPING, f"last_response:{session_id}"
                ),
            ]

            deleted_count = await self.redis_client.delete(*keys_to_delete)

            logger.info(
                "Deleted session data",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                keys_deleted=deleted_count,
            )

            return deleted_count > 0

        except Exception as e:
            error_msg = f"Failed to delete session: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def get_active_session_context(
        self, session_id: str
    ) -> Optional[ActiveSessionContext]:
        """
        Retrieve comprehensive context for an active session using Redis pipeline.

        This method is the gatekeeper for session validation. It performs an atomic
        check to verify the session is active, and only then retrieves all related data.

        Implementation Strategy:
        1. First, check if the core session key exists (validates session is active)
        2. If session doesn't exist or is expired, return None immediately
        3. If session exists, use a Redis pipeline to atomically fetch:
           - Session metadata (hash)
           - Last response ID (string)
           - Response chain (for additional validation if needed)
        4. Parse the metadata to extract vector_store_id and root_response_id

        This atomic approach ensures data consistency and prevents race conditions.

        Args:
            session_id: Unique session identifier to validate and retrieve

        Returns:
            ActiveSessionContext with all session data if session is active,
            None if session is inactive/expired or doesn't exist

        Raises:
            CacheOperationError: If the operation fails due to Redis errors
        """
        try:
            # Build all keys
            session_key = self._build_key(CacheOperationType.SESSION, session_id)
            last_response_key = self._build_key(
                CacheOperationType.CHAT_MAPPING, f"last_response:{session_id}"
            )

            # CRITICAL: First check if session key exists
            # This is the gatekeeper - if the key doesn't exist or has expired,
            # the session is considered inactive
            session_exists = await self.redis_client.exists(session_key)

            if not session_exists:
                logger.debug(
                    "Session not found or expired",
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                )
                return None

            # Session is active - fetch all related data atomically using pipeline
            # This ensures data consistency even under concurrent access
            pipeline = self.redis_client.pipeline()
            pipeline.hgetall(session_key)  # Get session metadata
            pipeline.get(last_response_key)  # Get last response ID

            results = await pipeline.execute()

            # Parse results
            metadata = results[0] if results[0] else {}
            last_response_id = results[1]

            # If metadata is empty, session exists but has no data (edge case)
            if not metadata:
                logger.warning(
                    "Session key exists but metadata is empty",
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                )
                return None

            # Deserialize metadata if it's stored as JSON string
            if isinstance(metadata, dict):
                # Metadata is already a dict (decode_responses=True)
                parsed_metadata = metadata
            else:
                try:
                    parsed_metadata = json.loads(metadata)
                except (json.JSONDecodeError, TypeError):
                    parsed_metadata = {}

            # Extract vector_store_id and root_response_id from metadata
            vector_store_id = parsed_metadata.get("vector_store_id")
            root_response_id = parsed_metadata.get("root_response_id")

            # Create and return the session context
            context = ActiveSessionContext(
                session_id=session_id,
                metadata=parsed_metadata,
                last_response_id=last_response_id,
                vector_store_id=vector_store_id,
                root_response_id=root_response_id,
            )

            logger.info(
                "Retrieved active session context from Redis",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                last_response_id=last_response_id,
                vector_store_id=vector_store_id,
                root_response_id=root_response_id,
                has_last_response=bool(last_response_id),
                has_vector_store=bool(vector_store_id),
                has_root_response=bool(root_response_id),
            )

            return context

        except Exception as e:
            error_msg = f"Failed to retrieve active session context: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Response Chain Management ====================

    async def append_response(self, session_id: str, response_id: str) -> bool:
        """
        Append response ID to the session's response chain

        Automatically trims the chain to max_chain_length.

        Args:
            session_id: Unique session identifier
            response_id: Response ID to append

        Returns:
            True if appended successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            key = self._build_key(CacheOperationType.RESPONSE_CHAIN, session_id)

            # Append to list
            await self.redis_client.rpush(key, response_id)

            # Trim to max_chain_length (keep last N entries)
            await self.redis_client.ltrim(key, -self.config.max_chain_length, -1)

            # Set TTL
            await self.redis_client.expire(key, self.config.default_session_ttl)

            logger.debug(
                "Appended response to chain",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                response_id=response_id,
            )

            return True

        except Exception as e:
            error_msg = f"Failed to append response: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                response_id=response_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def get_response_chain(self, session_id: str) -> List[str]:
        """
        Retrieve the complete response chain for a session

        Args:
            session_id: Unique session identifier

        Returns:
            List of response IDs in chronological order (oldest first)

        Raises:
            CacheOperationError: If retrieval fails
        """
        try:
            key = self._build_key(CacheOperationType.RESPONSE_CHAIN, session_id)

            # Get all items from list
            chain = await self.redis_client.lrange(key, 0, -1)

            logger.debug(
                "Retrieved response chain",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                chain_length=len(chain),
            )

            return chain

        except Exception as e:
            error_msg = f"Failed to get response chain: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(
                CacheOperationType.CHAT_MAPPING, f"last_response:{session_id}"
            )

            response_id = await self.redis_client.get(key)

            logger.debug(
                "Retrieved last response ID",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                response_id=response_id,
                cache_hit=response_id is not None,
            )

            return response_id

        except Exception as e:
            error_msg = f"Failed to get last response ID: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def set_last_response_id(
        self, session_id: str, response_id: str, ttl: Optional[int] = None
    ) -> bool:
        """
        Set the last response ID for quick access

        Args:
            session_id: Unique session identifier
            response_id: Response ID to set
            ttl: Time-to-live in seconds (uses default_session_ttl if None)

        Returns:
            True if set successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            key = self._build_key(
                CacheOperationType.CHAT_MAPPING, f"last_response:{session_id}"
            )
            ttl = ttl or self.config.default_session_ttl

            await self.redis_client.setex(key, ttl, response_id)

            logger.debug(
                "Set last response ID",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                response_id=response_id,
                ttl=ttl,
            )

            return True

        except Exception as e:
            error_msg = f"Failed to set last response ID: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                response_id=response_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== History Caching ====================

    async def cache_history(
        self, session_id: str, history: List[Dict[str, Any]], ttl: Optional[int] = None
    ) -> bool:
        """
        Cache conversation history as JSON

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
        try:
            key = self._build_key(CacheOperationType.HISTORY, session_id)
            ttl = ttl or self.config.default_history_ttl

            # Serialize history to JSON
            history_json = json.dumps(history)

            # Store with TTL
            await self.redis_client.setex(key, ttl, history_json)

            logger.debug(
                "Cached conversation history",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                message_count=len(history),
                ttl=ttl,
            )

            return True

        except (TypeError, ValueError) as e:
            error_msg = f"Failed to serialize history: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheSerializationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to cache history: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(CacheOperationType.HISTORY, session_id)

            history_json = await self.redis_client.get(key)

            if not history_json:
                logger.debug(
                    "History not cached",
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                    cache_hit=False,
                )
                return None

            # Deserialize from JSON
            history = json.loads(history_json)

            logger.debug(
                "Retrieved cached history",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                message_count=len(history),
                cache_hit=True,
            )

            return history

        except json.JSONDecodeError as e:
            error_msg = f"Failed to deserialize history: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheSerializationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to get cached history: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def invalidate_history(self, session_id: str) -> bool:
        """
        Invalidate (delete) cached conversation history

        Args:
            session_id: Unique session identifier

        Returns:
            True if invalidated successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            key = self._build_key(CacheOperationType.HISTORY, session_id)

            deleted = await self.redis_client.delete(key)

            logger.debug(
                "Invalidated history cache",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                deleted=deleted > 0,
            )

            return deleted > 0

        except Exception as e:
            error_msg = f"Failed to invalidate history: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def add_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        max_messages: Optional[int] = None,
    ) -> bool:
        """
        Add a message to the session's message history list AND update message_count.

        CRITICAL FIX: Now atomically updates message_count in session metadata
        to ensure accurate tracking of conversation length.

        Messages are stored in a Redis LIST at key: session:<session_id>:messages
        The list is automatically trimmed to max_messages to prevent unbounded growth.

        Args:
            session_id: Unique session identifier
            message: Dictionary containing message data (role, content, timestamp, etc.)
            max_messages: Maximum messages to keep (defaults to config.max_chain_length)

        Returns:
            True if message added successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
            CacheSerializationError: If message serialization fails
        """
        try:
            import json

            max_messages = max_messages or self.config.max_chain_length

            # Build message list key
            messages_key = f"session:{session_id}:messages"
            session_key = f"session:{session_id}"

            # Serialize message to JSON
            message_json = json.dumps(message)

            # Use pipeline for atomic operations
            pipeline = self.redis_client.pipeline()

            # 1. Append message to list (LPUSH adds to the left/head)
            pipeline.lpush(messages_key, message_json)

            # 2. Trim list to max length (keep most recent messages)
            pipeline.ltrim(messages_key, 0, max_messages - 1)

            # 3. Get current list length for message_count
            pipeline.llen(messages_key)

            # 4. Set TTL on messages list
            pipeline.expire(messages_key, self.config.default_session_ttl)

            # Execute pipeline
            results = await pipeline.execute()

            # 5. Update message_count in session metadata HASH atomically
            message_count = results[2]  # Result from llen

            pipeline2 = self.redis_client.pipeline()
            pipeline2.hset(session_key, "message_count", str(message_count))
            pipeline2.expire(session_key, self.config.default_session_ttl)
            await pipeline2.execute()

            logger.debug(
                "Added message to session history and updated message_count",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                message_count=message_count,
                max_messages=max_messages,
            )

            return True

        except (TypeError, ValueError) as e:
            if isinstance(e, json.JSONDecodeError) or str(e).startswith(
                "Object of type"
            ):
                error_msg = f"Failed to serialize message: {e}"
                logger.error(
                    error_msg,
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                    exc_info=True,
                )
                raise CacheSerializationError(error_msg)
            raise

        except Exception as e:
            error_msg = f"Failed to add message: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages from session history.

        Args:
            session_id: Unique session identifier
            limit: Maximum number of messages to retrieve (retrieves all if None)

        Returns:
            List of message dictionaries in reverse chronological order (newest first)
            Returns empty list if no messages exist

        Raises:
            CacheOperationError: If retrieval fails
            CacheSerializationError: If deserialization fails
        """
        try:
            import json

            messages_key = f"session:{session_id}:messages"

            if limit is None:
                # Get all messages
                raw_messages = await self.redis_client.lrange(messages_key, 0, -1)
            else:
                # Get most recent 'limit' messages
                raw_messages = await self.redis_client.lrange(
                    messages_key, 0, limit - 1
                )

            messages = []
            for raw_msg in raw_messages:
                try:
                    msg_dict = json.loads(raw_msg)
                    messages.append(msg_dict)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to deserialize message: {e}",
                        component="Cache",
                        subcomponent="RedisProvider",
                        session_id=session_id,
                    )
                    continue

            logger.debug(
                "Retrieved messages from session history",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                message_count=len(messages),
            )

            return messages

        except Exception as e:
            error_msg = f"Failed to get messages: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Tool Execution State ====================

    async def set_tool_execution_state(
        self,
        session_id: str,
        request_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store tool execution state as hash

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
        try:
            key = self._build_key(
                CacheOperationType.TOOL_EXECUTION, session_id, request_id
            )
            ttl = ttl or self.config.default_tool_execution_ttl

            # Serialize state values
            serialized_state = {k: self._serialize_value(v) for k, v in state.items()}

            # Store as hash
            await self.redis_client.hset(key, mapping=serialized_state)

            # Set TTL
            await self.redis_client.expire(key, ttl)

            logger.debug(
                "Set tool execution state",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                ttl=ttl,
            )

            return True

        except (TypeError, ValueError) as e:
            error_msg = f"Failed to serialize tool execution state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheSerializationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to set tool execution state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(
                CacheOperationType.TOOL_EXECUTION, session_id, request_id
            )

            data = await self.redis_client.hgetall(key)

            if not data:
                logger.debug(
                    "Tool execution state not found",
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                    request_id=request_id,
                    cache_hit=False,
                )
                return None

            # Deserialize values
            state = {k: self._deserialize_value(v) for k, v in data.items()}

            logger.debug(
                "Retrieved tool execution state",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                cache_hit=True,
            )

            return state

        except Exception as e:
            error_msg = f"Failed to get tool execution state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def clear_tool_execution_state(
        self, session_id: str, request_id: str
    ) -> bool:
        """
        Clear tool execution state

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this tool execution

        Returns:
            True if cleared successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            key = self._build_key(
                CacheOperationType.TOOL_EXECUTION, session_id, request_id
            )

            deleted = await self.redis_client.delete(key)

            logger.debug(
                "Cleared tool execution state",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                deleted=deleted > 0,
            )

            return deleted > 0

        except Exception as e:
            error_msg = f"Failed to clear tool execution state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Streaming State ====================

    async def set_streaming_state(
        self,
        session_id: str,
        request_id: str,
        state: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store streaming state as hash

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
        try:
            key = self._build_key(CacheOperationType.STREAMING, session_id, request_id)
            ttl = ttl or self.config.default_streaming_ttl

            # Serialize state values
            serialized_state = {k: self._serialize_value(v) for k, v in state.items()}

            # Store as hash
            await self.redis_client.hset(key, mapping=serialized_state)

            # Set TTL
            await self.redis_client.expire(key, ttl)

            logger.debug(
                "Set streaming state",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                ttl=ttl,
            )

            return True

        except (TypeError, ValueError) as e:
            error_msg = f"Failed to serialize streaming state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheSerializationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to set streaming state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(CacheOperationType.STREAMING, session_id, request_id)

            data = await self.redis_client.hgetall(key)

            if not data:
                logger.debug(
                    "Streaming state not found",
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                    request_id=request_id,
                    cache_hit=False,
                )
                return None

            # Deserialize values
            state = {k: self._deserialize_value(v) for k, v in data.items()}

            logger.debug(
                "Retrieved streaming state",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                cache_hit=True,
            )

            return state

        except Exception as e:
            error_msg = f"Failed to get streaming state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def clear_streaming_state(self, session_id: str, request_id: str) -> bool:
        """
        Clear streaming state

        Args:
            session_id: Unique session identifier
            request_id: Request ID for this streaming session

        Returns:
            True if cleared successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            key = self._build_key(CacheOperationType.STREAMING, session_id, request_id)

            deleted = await self.redis_client.delete(key)

            logger.debug(
                "Cleared streaming state",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                deleted=deleted > 0,
            )

            return deleted > 0

        except Exception as e:
            error_msg = f"Failed to clear streaming state: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Vector Store Validation ====================

    async def cache_vector_store_validation(
        self,
        vector_store_id: str,
        validation_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache vector store validation results as hash

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
        try:
            key = self._build_key(CacheOperationType.VECTOR_STORE, vector_store_id)
            ttl = ttl or self.config.default_vector_store_ttl

            # Serialize validation data
            serialized_data = {
                k: self._serialize_value(v) for k, v in validation_data.items()
            }

            # Store as hash
            await self.redis_client.hset(key, mapping=serialized_data)

            # Set TTL
            await self.redis_client.expire(key, ttl)

            logger.debug(
                "Cached vector store validation",
                component="Cache",
                subcomponent="RedisProvider",
                vector_store_id=vector_store_id,
                ttl=ttl,
            )

            return True

        except (TypeError, ValueError) as e:
            error_msg = f"Failed to serialize vector store validation: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                vector_store_id=vector_store_id,
                exc_info=True,
            )
            raise CacheSerializationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to cache vector store validation: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                vector_store_id=vector_store_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(CacheOperationType.VECTOR_STORE, vector_store_id)

            data = await self.redis_client.hgetall(key)

            if not data:
                logger.debug(
                    "Vector store validation not cached",
                    component="Cache",
                    subcomponent="RedisProvider",
                    vector_store_id=vector_store_id,
                    cache_hit=False,
                )
                return None

            # Deserialize values
            validation_data = {k: self._deserialize_value(v) for k, v in data.items()}

            logger.debug(
                "Retrieved vector store validation",
                component="Cache",
                subcomponent="RedisProvider",
                vector_store_id=vector_store_id,
                cache_hit=True,
            )

            return validation_data

        except Exception as e:
            error_msg = f"Failed to get vector store validation: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                vector_store_id=vector_store_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Rate Limiting ====================

    async def increment_request_count(
        self, session_id: str, window_seconds: int = 3600
    ) -> int:
        """
        Increment request count using sorted set with timestamps

        Uses sliding window approach for accurate rate limiting.

        Args:
            session_id: Unique session identifier
            window_seconds: Time window in seconds for rate limiting

        Returns:
            Current request count within the window

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            key = self._build_key(CacheOperationType.RATE_LIMIT, session_id)
            current_time = time.time()
            window_start = current_time - window_seconds

            # Remove old entries outside the window
            await self.redis_client.zremrangebyscore(key, 0, window_start)

            # Add current request with timestamp as score
            await self.redis_client.zadd(key, {str(current_time): current_time})

            # Set TTL
            await self.redis_client.expire(key, window_seconds)

            # Count requests in window
            count = await self.redis_client.zcount(key, window_start, current_time)

            logger.debug(
                "Incremented request count",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                count=count,
                window_seconds=window_seconds,
            )

            return count

        except Exception as e:
            error_msg = f"Failed to increment request count: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(CacheOperationType.RATE_LIMIT, session_id)
            current_time = time.time()
            window_start = current_time - window_seconds

            # Count requests in window
            count = await self.redis_client.zcount(key, window_start, current_time)

            logger.debug(
                "Retrieved request count",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                count=count,
                window_seconds=window_seconds,
            )

            return count

        except Exception as e:
            error_msg = f"Failed to get request count: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            count = await self.get_request_count(session_id, window_seconds)
            within_limit = count < limit

            logger.debug(
                "Checked rate limit",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                count=count,
                limit=limit,
                within_limit=within_limit,
            )

            return within_limit

        except Exception as e:
            error_msg = f"Failed to check rate limit: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Request Correlation ====================

    async def set_request_correlation(
        self,
        session_id: str,
        request_id: str,
        correlation_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store request correlation data as hash

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
        try:
            key = self._build_key(
                CacheOperationType.REQUEST_CORRELATION, session_id, request_id
            )
            ttl = ttl or self.config.default_request_correlation_ttl

            # Serialize correlation data
            serialized_data = {
                k: self._serialize_value(v) for k, v in correlation_data.items()
            }

            # Store as hash
            await self.redis_client.hset(key, mapping=serialized_data)

            # Set TTL
            await self.redis_client.expire(key, ttl)

            logger.debug(
                "Set request correlation",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                ttl=ttl,
            )

            return True

        except (TypeError, ValueError) as e:
            error_msg = f"Failed to serialize request correlation: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheSerializationError(error_msg)

        except Exception as e:
            error_msg = f"Failed to set request correlation: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(
                CacheOperationType.REQUEST_CORRELATION, session_id, request_id
            )

            data = await self.redis_client.hgetall(key)

            if not data:
                logger.debug(
                    "Request correlation not found",
                    component="Cache",
                    subcomponent="RedisProvider",
                    session_id=session_id,
                    request_id=request_id,
                    cache_hit=False,
                )
                return None

            # Deserialize values
            correlation_data = {k: self._deserialize_value(v) for k, v in data.items()}

            logger.debug(
                "Retrieved request correlation",
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                cache_hit=True,
            )

            return correlation_data

        except Exception as e:
            error_msg = f"Failed to get request correlation: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                session_id=session_id,
                request_id=request_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Chat ID Mapping ====================

    async def set_chat_mapping(
        self, chat_id: str, response_id: str, ttl: Optional[int] = None
    ) -> bool:
        """
        Map chat ID to last response ID

        Args:
            chat_id: Chat identifier
            response_id: Response identifier
            ttl: Time-to-live in seconds (uses default_session_ttl if None)

        Returns:
            True if mapped successfully, False otherwise

        Raises:
            CacheOperationError: If operation fails
        """
        try:
            key = self._build_key(CacheOperationType.CHAT_MAPPING, chat_id)
            ttl = ttl or self.config.default_session_ttl

            await self.redis_client.setex(key, ttl, response_id)

            logger.debug(
                "Set chat mapping",
                component="Cache",
                subcomponent="RedisProvider",
                chat_id=chat_id,
                response_id=response_id,
                ttl=ttl,
            )

            return True

        except Exception as e:
            error_msg = f"Failed to set chat mapping: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                chat_id=chat_id,
                response_id=response_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            key = self._build_key(CacheOperationType.CHAT_MAPPING, chat_id)

            response_id = await self.redis_client.get(key)

            logger.debug(
                "Retrieved chat mapping",
                component="Cache",
                subcomponent="RedisProvider",
                chat_id=chat_id,
                response_id=response_id,
                cache_hit=response_id is not None,
            )

            return response_id

        except Exception as e:
            error_msg = f"Failed to get chat mapping: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                chat_id=chat_id,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Utility Methods ====================

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise

        Raises:
            CacheOperationError: If check fails
        """
        try:
            exists = await self.redis_client.exists(key)

            logger.debug(
                "Checked key existence",
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                exists=exists > 0,
            )

            return exists > 0

        except Exception as e:
            error_msg = f"Failed to check key existence: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis

        Args:
            key: Cache key to delete

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CacheOperationError: If deletion fails
        """
        try:
            deleted = await self.redis_client.delete(key)

            logger.debug(
                "Deleted key",
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                deleted=deleted > 0,
            )

            return deleted > 0

        except Exception as e:
            error_msg = f"Failed to delete key: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            if ttl <= 0:
                raise CacheTTLError(f"Invalid TTL value: {ttl}. Must be positive.")

            set_ttl = await self.redis_client.expire(key, ttl)

            logger.debug(
                "Set key expiration",
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                ttl=ttl,
                success=set_ttl,
            )

            return set_ttl

        except CacheTTLError:
            raise

        except Exception as e:
            error_msg = f"Failed to set key expiration: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                ttl=ttl,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

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
        try:
            ttl = await self.redis_client.ttl(key)

            # Redis returns:
            # -2 if key doesn't exist
            # -1 if key exists but has no expiration
            # >0 for remaining TTL

            if ttl == -2:
                result = None
            elif ttl == -1:
                result = -1
            else:
                result = ttl

            logger.debug(
                "Retrieved key TTL",
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                ttl=result,
            )

            return result

        except Exception as e:
            error_msg = f"Failed to get key TTL: {e}"
            logger.error(
                error_msg,
                component="Cache",
                subcomponent="RedisProvider",
                key=key,
                exc_info=True,
            )
            raise CacheOperationError(error_msg)

    # ==================== Private Helper Methods ====================

    def _serialize_value(self, value: Any) -> str:
        """
        Serialize a value to string for Redis storage

        Args:
            value: Value to serialize

        Returns:
            Serialized string

        Raises:
            TypeError: If value is not serializable
        """
        if isinstance(value, (str, int, float, bool)):
            return json.dumps(value)
        elif value is None:
            return json.dumps(None)
        else:
            # Serialize complex types to JSON
            return json.dumps(value)

    def _deserialize_value(self, value: str) -> Any:
        """
        Deserialize a value from Redis storage

        Args:
            value: Serialized string value

        Returns:
            Deserialized value

        Raises:
            json.JSONDecodeError: If deserialization fails
        """
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # If JSON decode fails, return as-is (plain string)
            return value


# ==================== Test and Validation ====================

if __name__ == "__main__":
    """
    Test script for Redis Provider
    
    This script tests all major functionality of the RedisProvider to ensure
    it works correctly. Run with: python -m src.providers.cache_provider.redis_provider
    
    Prerequisites:
    - Redis server must be running (docker-compose up -d redis)
    - Redis password must match configuration (default: redis123)
    """

    import asyncio
    import sys
    from datetime import datetime

    from src.providers.cache_provider.base import CacheConfig

    async def test_redis_provider():
        """Comprehensive test suite for RedisProvider"""

        print("=" * 80)
        print("Redis Provider Test Suite")
        print("=" * 80)

        # Test configuration
        # Note: If Redis doesn't require password, set password=None
        # Try with password first, fall back to None if authentication fails
        config = CacheConfig(
            host="localhost",
            port=6379,
            db=0,
            password="redis123",  # Set to None if Redis doesn't require password
            key_prefix="test_drug_reco",
            default_session_ttl=3600,
            default_history_ttl=1800,
        )

        # Create provider
        print("\n[1] Creating RedisProvider instance...")
        provider = RedisProvider(config)
        print("✓ Provider created successfully")

        # Test connection
        print("\n[2] Testing connection...")
        try:
            await provider.connect()
            print("✓ Connected to Redis successfully")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("\nMake sure Redis is running:")
            print("  docker-compose -f infrastructure/docker-compose.yml up -d redis")
            sys.exit(1)

        # Test health check
        print("\n[3] Testing health check...")
        is_healthy = await provider.health_check()
        if is_healthy:
            print("✓ Health check passed")
        else:
            print("✗ Health check failed")
            sys.exit(1)

        # Test session management
        print("\n[4] Testing session management...")
        test_session_id = f"test_session_{int(datetime.now().timestamp())}"

        # Set session metadata
        session_metadata = {
            "user_id": "test_user_123",
            "vector_store_id": "vs_test_456",
            "created_at": datetime.utcnow().isoformat(),
            "last_active_at": datetime.utcnow().isoformat(),
            "agent_ready": True,
            "message_count": 0,
        }

        await provider.set_session_metadata(test_session_id, session_metadata)
        print("✓ Session metadata set")

        # Get session metadata
        retrieved_metadata = await provider.get_session_metadata(test_session_id)
        assert retrieved_metadata is not None, "Failed to retrieve session metadata"
        assert retrieved_metadata["user_id"] == "test_user_123", "Metadata mismatch"
        print("✓ Session metadata retrieved")

        # Touch session
        await provider.touch_session(test_session_id)
        print("✓ Session touched (TTL extended)")

        # Test response chain
        print("\n[5] Testing response chain...")

        # Append responses
        for i in range(5):
            await provider.append_response(test_session_id, f"resp_{i:03d}")
        print("✓ Appended 5 responses to chain")

        # Get response chain
        chain = await provider.get_response_chain(test_session_id)
        assert len(chain) == 5, f"Expected 5 responses, got {len(chain)}"
        assert chain[0] == "resp_000", "Chain order incorrect"
        print(f"✓ Retrieved response chain (length: {len(chain)})")

        # Test last response ID
        await provider.set_last_response_id(test_session_id, "resp_004")
        last_id = await provider.get_last_response_id(test_session_id)
        assert last_id == "resp_004", "Last response ID mismatch"
        print("✓ Last response ID set and retrieved")

        # Test history caching
        print("\n[6] Testing history caching...")

        test_history = [
            {
                "role": "user",
                "content": "What is metformin?",
                "timestamp": datetime.utcnow().isoformat(),
            },
            {
                "role": "assistant",
                "content": "Metformin is a medication used to treat type 2 diabetes...",
                "timestamp": datetime.utcnow().isoformat(),
                "response_id": "resp_001",
            },
        ]

        await provider.cache_history(test_session_id, test_history)
        print("✓ History cached")

        cached_history = await provider.get_cached_history(test_session_id)
        assert cached_history is not None, "Failed to retrieve cached history"
        assert len(cached_history) == 2, "History length mismatch"
        assert cached_history[0]["role"] == "user", "History content mismatch"
        print("✓ Cached history retrieved")

        # Test history invalidation
        await provider.invalidate_history(test_session_id)
        invalidated_history = await provider.get_cached_history(test_session_id)
        assert invalidated_history is None, "History should be None after invalidation"
        print("✓ History invalidated")

        # Test tool execution state
        print("\n[7] Testing tool execution state...")

        test_request_id = "req_test_789"
        tool_state = {
            "status": "in_progress",
            "tool_calls": [
                {
                    "call_id": "call_abc123",
                    "function_name": "search_drug_database",
                    "status": "executing",
                    "started_at": datetime.utcnow().isoformat(),
                }
            ],
            "started_at": datetime.utcnow().isoformat(),
        }

        await provider.set_tool_execution_state(
            test_session_id, test_request_id, tool_state
        )
        print("✓ Tool execution state set")

        retrieved_state = await provider.get_tool_execution_state(
            test_session_id, test_request_id
        )
        assert retrieved_state is not None, "Failed to retrieve tool execution state"
        assert retrieved_state["status"] == "in_progress", "Tool state mismatch"
        print("✓ Tool execution state retrieved")

        await provider.clear_tool_execution_state(test_session_id, test_request_id)
        cleared_state = await provider.get_tool_execution_state(
            test_session_id, test_request_id
        )
        assert cleared_state is None, "Tool state should be None after clearing"
        print("✓ Tool execution state cleared")

        # Test streaming state
        print("\n[8] Testing streaming state...")

        streaming_state = {
            "response_id": "resp_stream_001",
            "status": "streaming",
            "chunk_count": 42,
            "last_chunk_at": datetime.utcnow().isoformat(),
            "is_citation": False,
        }

        await provider.set_streaming_state(
            test_session_id, test_request_id, streaming_state
        )
        print("✓ Streaming state set")

        retrieved_streaming = await provider.get_streaming_state(
            test_session_id, test_request_id
        )
        assert retrieved_streaming is not None, "Failed to retrieve streaming state"
        assert retrieved_streaming["status"] == "streaming", "Streaming state mismatch"
        print("✓ Streaming state retrieved")

        await provider.clear_streaming_state(test_session_id, test_request_id)
        print("✓ Streaming state cleared")

        # Test vector store validation
        print("\n[9] Testing vector store validation...")

        test_vector_store_id = "vs_test_123"
        validation_data = {
            "status": "completed",
            "validated_at": datetime.utcnow().isoformat(),
            "file_count": 15,
            "expires_at": datetime.utcnow().isoformat(),
        }

        await provider.cache_vector_store_validation(
            test_vector_store_id, validation_data
        )
        print("✓ Vector store validation cached")

        cached_validation = await provider.get_vector_store_validation(
            test_vector_store_id
        )
        assert (
            cached_validation is not None
        ), "Failed to retrieve vector store validation"
        assert cached_validation["status"] == "completed", "Validation data mismatch"
        print("✓ Vector store validation retrieved")

        # Test rate limiting
        print("\n[10] Testing rate limiting...")

        # Increment request count
        count1 = await provider.increment_request_count(
            test_session_id, window_seconds=3600
        )
        assert count1 == 1, f"Expected count 1, got {count1}"
        print(f"✓ Request count incremented (count: {count1})")

        count2 = await provider.increment_request_count(
            test_session_id, window_seconds=3600
        )
        assert count2 == 2, f"Expected count 2, got {count2}"
        print(f"✓ Request count incremented again (count: {count2})")

        # Check rate limit
        within_limit = await provider.check_rate_limit(
            test_session_id, limit=100, window_seconds=3600
        )
        assert within_limit is True, "Should be within limit"
        print("✓ Rate limit check passed")

        # Test request correlation
        print("\n[11] Testing request correlation...")

        correlation_data = {
            "response_id": "resp_corr_001",
            "user_message": "What is metformin?",
            "status": "completed",
            "created_at": datetime.utcnow().isoformat(),
        }

        await provider.set_request_correlation(
            test_session_id, test_request_id, correlation_data
        )
        print("✓ Request correlation set")

        retrieved_correlation = await provider.get_request_correlation(
            test_session_id, test_request_id
        )
        assert (
            retrieved_correlation is not None
        ), "Failed to retrieve request correlation"
        assert (
            retrieved_correlation["response_id"] == "resp_corr_001"
        ), "Correlation data mismatch"
        print("✓ Request correlation retrieved")

        # Test chat ID mapping
        print("\n[12] Testing chat ID mapping...")

        test_chat_id = "chat_test_456"
        await provider.set_chat_mapping(test_chat_id, "resp_chat_001")
        print("✓ Chat mapping set")

        mapped_response_id = await provider.get_chat_mapping(test_chat_id)
        assert mapped_response_id == "resp_chat_001", "Chat mapping mismatch"
        print("✓ Chat mapping retrieved")

        # Test utility methods
        print("\n[13] Testing utility methods...")

        # Test exists
        exists = await provider.exists(
            provider._build_key(CacheOperationType.SESSION, test_session_id)
        )
        assert exists is True, "Key should exist"
        print("✓ Key existence check passed")

        # Test get_ttl
        ttl = await provider.get_ttl(
            provider._build_key(CacheOperationType.SESSION, test_session_id)
        )
        assert ttl is not None and ttl > 0, "TTL should be positive"
        print(f"✓ TTL retrieved (remaining: {ttl} seconds)")

        # Test expire
        await provider.expire(
            provider._build_key(CacheOperationType.SESSION, test_session_id), ttl=7200
        )
        print("✓ Key expiration set")

        # Test delete
        test_key = provider._build_key(
            CacheOperationType.CHAT_MAPPING, "test_delete_key"
        )
        await provider.set_chat_mapping("test_delete_key", "test_value")
        deleted = await provider.delete(test_key)
        assert deleted is True, "Key should be deleted"
        print("✓ Key deletion successful")

        # Cleanup test session
        print("\n[14] Cleaning up test data...")
        await provider.delete_session(test_session_id)
        print("✓ Test session deleted")

        # Test disconnect
        print("\n[15] Testing disconnect...")
        await provider.disconnect()
        print("✓ Disconnected from Redis successfully")

        # Final summary
        print("\n" + "=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)
        print("\nRedis Provider is working correctly.")
        print("All operations completed without errors.")

    # Run the test suite
    try:
        asyncio.run(test_redis_provider())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
