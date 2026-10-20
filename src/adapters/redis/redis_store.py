"""Redis implementation of the CacheProvider interface."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any, TypeVar

import redis.asyncio as aioredis
from aiobreaker import CircuitBreaker, CircuitBreakerError
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import AuthenticationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from src.config import get_settings
from src.logs import get_component_logger

from .base import CacheProvider
from .exceptions import (
    CacheAuthenticationError,
    CacheConnectionError,
    CacheOperationError,
    CacheProviderError,
    CacheSerializationError,
)
from .schema import ActiveSessionContext, CacheConfig, CacheOperationType

logger = get_component_logger("Cache")

T = TypeVar("T")


def _decode(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _decode_fields(data: dict[str, str]) -> dict[str, Any]:
    return {name: _decode(value) for name, value in data.items()}


class RedisProvider(CacheProvider):
    """Redis-backed cache with retry, circuit breaking and JSON values."""

    def __init__(self, config: CacheConfig):
        super().__init__(config)
        settings = get_settings()
        self.redis_client: aioredis.Redis | None = None
        self._retry = Retry(
            backoff=ExponentialBackoff(
                base=settings.redis_retry_base_delay,
                cap=settings.redis_retry_max_delay,
            ),
            retries=settings.redis_retry_max_attempts,
        )
        self._breaker = CircuitBreaker(
            fail_max=settings.redis_circuit_breaker_fail_max,
            timeout_duration=timedelta(seconds=settings.redis_circuit_breaker_timeout),
        )
        self._breaker.add_excluded_exceptions(CacheSerializationError, ValueError, TypeError)

    # ==================== Helpers ====================

    def _client(self) -> aioredis.Redis:
        if self.redis_client is None:
            raise CacheConnectionError("Redis client is not connected. Call connect() first.")
        return self.redis_client

    async def _call(
        self, operation: str, make_call: Callable[[], Awaitable[T]], **context: Any
    ) -> T:
        """Run one Redis operation through the circuit breaker and map its errors."""
        try:
            return await self._breaker.call_async(make_call)
        except CacheProviderError:
            raise
        except (TypeError, ValueError) as exc:
            raise CacheSerializationError(f"Failed to {operation}: {exc}") from exc
        except CircuitBreakerError as exc:
            raise CacheOperationError(f"Failed to {operation}: circuit is open") from exc
        except Exception as exc:
            logger.error(f"Failed to {operation}: {exc}", **context, exc_info=True)
            raise CacheOperationError(f"Failed to {operation}: {exc}") from exc

    async def _store_hash(
        self,
        operation: str,
        key: str,
        values: dict[str, Any],
        ttl: int,
        **context: Any,
    ) -> None:
        client = self._client()

        async def write() -> None:
            mapping = {name: json.dumps(value) for name, value in values.items()}
            await client.hset(key, mapping=mapping)
            await client.expire(key, ttl)

        await self._call(operation, write, **context)

    async def _load_hash(self, operation: str, key: str, **context: Any) -> dict[str, Any] | None:
        client = self._client()
        data = await self._call(operation, lambda: client.hgetall(key), **context)
        return _decode_fields(data) if data else None

    async def _delete_key(self, operation: str, key: str, **context: Any) -> bool:
        client = self._client()
        deleted = await self._call(operation, lambda: client.delete(key), **context)
        return deleted > 0

    def _last_response_key(self, session_id: str) -> str:
        return self._build_key(CacheOperationType.CHAT_MAPPING, f"last_response:{session_id}")

    # ==================== Connection Management ====================

    async def connect(self) -> None:
        client = aioredis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=self.config.decode_responses,
            ssl=self.config.ssl,
            retry=self._retry,
        )
        try:
            await self._breaker.call_async(client.ping)
        except AuthenticationError as exc:
            await client.aclose()
            raise CacheAuthenticationError(f"Redis authentication failed: {exc}") from exc
        except (RedisConnectionError, RedisTimeoutError, CircuitBreakerError, OSError) as exc:
            await client.aclose()
            raise CacheConnectionError(f"Could not connect to Redis: {exc}") from exc
        self.redis_client = client
        logger.info("Connected to Redis", host=self.config.host, port=self.config.port)

    async def disconnect(self) -> None:
        if self.redis_client is None:
            return
        try:
            await self.redis_client.aclose()
        except Exception as exc:
            logger.error(f"Error during Redis disconnect: {exc}", exc_info=True)
        finally:
            self.redis_client = None

    async def health_check(self) -> bool:
        if self.redis_client is None:
            return False
        try:
            return await self.redis_client.ping() is True
        except Exception as exc:
            logger.warning(f"Redis health check failed: {exc}")
            return False

    # ==================== Session Management ====================

    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        key = self._build_key(CacheOperationType.SESSION, session_id)
        return await self._load_hash("get session metadata", key, session_id=session_id)

    async def set_session_metadata(
        self, session_id: str, metadata: dict[str, Any], ttl: int | None = None
    ) -> bool:
        key = self._build_key(CacheOperationType.SESSION, session_id)
        ttl = ttl or self.config.default_session_ttl
        await self._store_hash("set session metadata", key, metadata, ttl, session_id=session_id)
        return True

    async def touch_session(self, session_id: str) -> bool:
        client = self._client()
        keys = [
            self._build_key(CacheOperationType.SESSION, session_id),
            self._build_key(CacheOperationType.RESPONSE_CHAIN, session_id),
            self._last_response_key(session_id),
        ]
        ttl = self.config.default_session_ttl

        async def touch() -> int:
            touched = 0
            for key in keys:
                if await client.expire(key, ttl):
                    touched += 1
            return touched

        touched = await self._call("touch session", touch, session_id=session_id)
        return touched > 0

    async def delete_session(self, session_id: str) -> bool:
        keys = [
            self._build_key(CacheOperationType.SESSION, session_id),
            self._build_key(CacheOperationType.RESPONSE_CHAIN, session_id),
            self._build_key(CacheOperationType.HISTORY, session_id),
            self._build_key(CacheOperationType.HISTORY, session_id, "messages"),
            self._last_response_key(session_id),
        ]
        client = self._client()
        deleted = await self._call(
            "delete session", lambda: client.delete(*keys), session_id=session_id
        )
        logger.info("Deleted session data", session_id=session_id, keys_deleted=deleted)
        return deleted > 0

    async def get_active_session_context(self, session_id: str) -> ActiveSessionContext | None:
        session_key = self._build_key(CacheOperationType.SESSION, session_id)
        last_response_key = self._last_response_key(session_id)
        client = self._client()

        async def fetch() -> list[Any]:
            async with client.pipeline(transaction=True) as pipe:
                pipe.exists(session_key)
                pipe.hgetall(session_key)
                pipe.get(last_response_key)
                return await pipe.execute()

        exists, data, last_response_id = await self._call(
            "get active session context", fetch, session_id=session_id
        )
        if not exists or not data:
            return None

        metadata = _decode_fields(data)
        return ActiveSessionContext(
            session_id=session_id,
            metadata=metadata,
            last_response_id=last_response_id,
            vector_store_id=metadata.get("vector_store_id"),
            root_response_id=metadata.get("root_response_id"),
        )

    # ==================== Response Chain Management ====================

    async def append_response(self, session_id: str, response_id: str) -> bool:
        key = self._build_key(CacheOperationType.RESPONSE_CHAIN, session_id)
        client = self._client()

        async def write() -> None:
            async with client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, response_id)
                pipe.ltrim(key, -self.config.max_chain_length, -1)
                pipe.expire(key, self.config.default_session_ttl)
                await pipe.execute()

        await self._call("append response", write, session_id=session_id)
        return True

    async def set_last_response_id(
        self, session_id: str, response_id: str, ttl: int | None = None
    ) -> bool:
        key = self._last_response_key(session_id)
        ttl = ttl or self.config.default_session_ttl
        client = self._client()
        await self._call(
            "set last response ID",
            lambda: client.setex(key, ttl, response_id),
            session_id=session_id,
        )
        return True

    # ==================== History Caching ====================

    async def cache_history(
        self, session_id: str, history: list[dict[str, Any]], ttl: int | None = None
    ) -> bool:
        key = self._build_key(CacheOperationType.HISTORY, session_id)
        ttl = ttl or self.config.default_history_ttl
        client = self._client()

        async def write() -> None:
            await client.setex(key, ttl, json.dumps(history))

        await self._call("cache history", write, session_id=session_id)
        return True

    async def get_cached_history(self, session_id: str) -> list[dict[str, Any]] | None:
        key = self._build_key(CacheOperationType.HISTORY, session_id)
        client = self._client()

        async def read() -> list[dict[str, Any]] | None:
            raw = await client.get(key)
            return json.loads(raw) if raw else None

        return await self._call("get cached history", read, session_id=session_id)

    async def invalidate_history(self, session_id: str) -> bool:
        key = self._build_key(CacheOperationType.HISTORY, session_id)
        return await self._delete_key("invalidate history", key, session_id=session_id)

    async def add_message(
        self,
        session_id: str,
        message: dict[str, Any],
        max_messages: int | None = None,
    ) -> bool:
        max_messages = max_messages or self.config.max_chain_length
        messages_key = self._build_key(CacheOperationType.HISTORY, session_id, "messages")
        session_key = self._build_key(CacheOperationType.SESSION, session_id)
        ttl = self.config.default_session_ttl
        client = self._client()

        async def write() -> None:
            payload = json.dumps(message)
            async with client.pipeline(transaction=True) as pipe:
                pipe.lpush(messages_key, payload)
                pipe.ltrim(messages_key, 0, max_messages - 1)
                pipe.llen(messages_key)
                pipe.expire(messages_key, ttl)
                _, _, message_count, _ = await pipe.execute()
            if await client.exists(session_key):
                await client.hset(session_key, "message_count", json.dumps(message_count))
                await client.expire(session_key, ttl)

        await self._call("add message", write, session_id=session_id)
        return True

    # ==================== Streaming State ====================

    async def set_streaming_state(
        self,
        session_id: str,
        request_id: str,
        state: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        key = self._build_key(CacheOperationType.STREAMING, session_id, request_id)
        ttl = ttl or self.config.default_streaming_ttl
        await self._store_hash("set streaming state", key, state, ttl, session_id=session_id)
        return True

    async def clear_streaming_state(self, session_id: str, request_id: str) -> bool:
        key = self._build_key(CacheOperationType.STREAMING, session_id, request_id)
        return await self._delete_key("clear streaming state", key, session_id=session_id)

    # ==================== Vector Store Validation ====================

    async def cache_vector_store_validation(
        self,
        vector_store_id: str,
        validation_data: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        key = self._build_key(CacheOperationType.VECTOR_STORE, vector_store_id)
        ttl = ttl or self.config.default_vector_store_ttl
        await self._store_hash("cache vector store validation", key, validation_data, ttl)
        return True

    async def get_vector_store_validation(self, vector_store_id: str) -> dict[str, Any] | None:
        key = self._build_key(CacheOperationType.VECTOR_STORE, vector_store_id)
        return await self._load_hash("get vector store validation", key)
