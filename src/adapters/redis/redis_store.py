"""Store session data in Redis with retries and clear cache errors.

Big-picture flow:
1. Open a resilient connection to Redis: one that automatically retries a
   failed command a few times, and "trips" like a circuit breaker if
   Redis keeps failing, instead of hammering a dead server forever.
2. Offer small, focused methods for each thing the app needs to cache:
   session metadata, Responses API response IDs, chat history, streaming state,
   and vector-store validation results.
3. Translate every low-level Redis/network failure into this adapter's own
   exception types (see exceptions.py), so the rest of the app never needs
   to know Redis-specific error classes.
"""

from __future__ import annotations

# See other files in this project for why this import is used - it lets
# this file use modern type-hint syntax (like "int | None") without extra
# runtime cost.
import json
from collections.abc import Awaitable, Callable

# Callable and Awaitable are type hints for "a function" and "something
# you can await", respectively. Together, "Callable[[], Awaitable[T]]"
# below means "a function that takes no arguments and, when awaited,
# eventually produces a value of type T."
from datetime import timedelta
from typing import Any, TypeVar

# TypeVar creates a placeholder type, "T", that type checkers fill in with
# whatever specific type is actually used at each call site - it lets one
# generic helper function work with many different return types while
# still being precisely typed.
import redis.asyncio as aioredis

# redis.asyncio is the Redis client library's "async" version: it lets the
# program do other work while waiting for Redis to respond, instead of
# freezing until the reply comes back.
from aiobreaker import CircuitBreaker, CircuitBreakerError

# A "circuit breaker" is a safety pattern named after electrical circuit
# breakers: after too many failures in a row, it "trips" and stops letting
# new requests through for a while, instead of letting every request wait
# and fail against an already-struggling server.
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

# "Exponential backoff" means waiting longer and longer between retries
# (e.g. 0.1s, 0.2s, 0.4s...) instead of retrying instantly, giving Redis
# time to recover.
from redis.exceptions import AuthenticationError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

# These are renamed on import ("as RedisConnectionError" / "as
# RedisTimeoutError") because Python already has its own built-in
# ConnectionError and TimeoutError - renaming avoids confusing the two.
from src.config import get_settings
from src.logs import get_component_logger

from .base import RedisCacheBase
from .exceptions import (
    CacheAuthenticationError,
    CacheConnectionError,
    CacheOperationError,
    CacheProviderError,
    CacheSerializationError,
)
from .schema import ActiveSessionContext, CacheKeyNamespace, RedisCacheConfig

logger = get_component_logger("Cache")

T = TypeVar("T")
# T: the generic placeholder type mentioned above, used by
# _run_cache_operation below so it can wrap an operation that returns
# anything (a dict, a bool, a list...) while keeping that exact return type
# visible to type checkers.


# _decode_redis_value: turns one value read from Redis back into a real
# Python value (number, dict, list, etc.) if it was stored as JSON text.
# The Redis client decodes replies to strings; JSON restores structured values.
def _decode_redis_value(raw: str) -> Any:
    try:
        # Step 1: most values are stored as JSON text, so try to parse it.
        return json.loads(raw)
    except json.JSONDecodeError:
        # Step 2: if it was not valid JSON (e.g. it was stored as plain
        # text to begin with), just return the raw string unchanged instead
        # of raising an error.
        return raw


# _decode_redis_hash: applies _decode_redis_value() to every hash field at
# once. Needed because Redis "hash" data structures come back as a flat
# dict of {field_name: raw_string}, and every value in it needs the same
# JSON-or-plain-text decoding.
def _decode_redis_hash(data: dict[str, str]) -> dict[str, Any]:
    return {name: _decode_redis_value(value) for name, value in data.items()}


# RedisProvider: the concrete cache implementation the rest of the app
# actually uses. It builds on RedisCacheBase (which only holds settings and
# builds keys) by adding the real Redis connection, retry/circuit-breaker
# resilience, and one method per thing the app needs to cache.
class RedisProvider(RedisCacheBase):
    """Redis-backed cache with retry, circuit breaking and JSON values."""

    def __init__(self, config: RedisCacheConfig):
        super().__init__(config)
        # Step 1: let the parent class store the config and log its
        # "initialized" message.
        settings = get_settings()
        self.redis_client: aioredis.Redis | None = None
        # self.redis_client: the Redis client and its connection pool, created
        # by connect(). It starts as None because no client exists yet.
        self._redis_retry_policy = Retry(
            backoff=ExponentialBackoff(
                base=settings.redis_retry_base_delay,
                cap=settings.redis_retry_max_delay,
            ),
            retries=settings.redis_retry_max_attempts,
        )
        # Step 2: build the retry policy - how many times, and with what
        # growing delay, a single failed Redis command should be retried
        # automatically before giving up.
        self._redis_circuit_breaker = CircuitBreaker(
            fail_max=settings.redis_circuit_breaker_fail_max,
            timeout_duration=timedelta(seconds=settings.redis_circuit_breaker_timeout),
        )
        # Step 3: build the circuit breaker - after this many consecutive
        # failures, it stops letting new commands through for the given
        # duration, to avoid repeatedly hammering a struggling Redis server.
        self._redis_circuit_breaker.add_excluded_exceptions(
            CacheSerializationError, ValueError, TypeError
        )
        # Step 4: tell the breaker to IGNORE these particular exception
        # types when counting failures - a bad value that fails to
        # serialize is a bug in the DATA, not a sign that Redis itself is
        # unhealthy, so it should not trip the breaker.

    # ==================== Helpers ====================

    def _connected_redis_client(self) -> aioredis.Redis:
        if self.redis_client is None:
            # Guard clause: every method below needs a live connection: if
            # connect() was never called, fail with a clear, specific error
            # instead of a confusing crash deeper inside the Redis library.
            raise CacheConnectionError("Redis client is not connected. Call connect() first.")
        return self.redis_client

    async def _run_cache_operation(
        self, operation: str, redis_command: Callable[[], Awaitable[T]], **context: Any
    ) -> T:
        """Run one Redis operation through the circuit breaker and map its errors."""
        # "redis_command" is a function with NO arguments that, when
        # called and awaited, actually talks to Redis - passing a function
        # (instead of just calling it directly) lets this helper decide
        # WHEN and HOW to run it (through the breaker, with error mapping)
        # without needing to know what the specific operation does.
        try:
            # Step 1: run the operation through the circuit breaker, which
            # either lets it through or immediately raises
            # CircuitBreakerError if the breaker is currently open.
            return await self._redis_circuit_breaker.call_async(redis_command)
        except CacheProviderError:
            # Step 2: if the operation already raised one of OUR specific
            # errors, let it pass through unchanged instead of re-wrapping it.
            raise
        except (TypeError, ValueError) as exc:
            # Step 3: these usually mean the DATA being stored/read was the
            # problem (e.g. not JSON-serializable), not Redis itself.
            raise CacheSerializationError(f"Failed to {operation}: {exc}") from exc
        except CircuitBreakerError as exc:
            # Step 4: the breaker is currently open (too many recent
            # failures), so this attempt was rejected without even trying
            # to reach Redis.
            raise CacheOperationError(f"Failed to {operation}: circuit is open") from exc
        except Exception as exc:
            # Step 5: anything else is an unexpected failure - log it with
            # full details (exc_info=True includes the traceback) before
            # translating it into our own exception type.
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
        client = self._connected_redis_client()

        async def write() -> None:
            # "write" is defined here, inside this method, so it can "see"
            # client/key/values/ttl from the surrounding method without
            # needing them passed in again - it exists purely to be handed
            # to _run_cache_operation below as the "redis_command" to run.
            mapping = {name: json.dumps(value) for name, value in values.items()}
            # mapping: every value turned into JSON text, since a Redis
            # "hash" can only store strings.
            await client.hset(key, mapping=mapping)
            await client.expire(key, ttl)

        await self._run_cache_operation(operation, write, **context)

    async def _load_hash(self, operation: str, key: str, **context: Any) -> dict[str, Any] | None:
        client = self._connected_redis_client()
        data = await self._run_cache_operation(operation, lambda: client.hgetall(key), **context)
        # "lambda: client.hgetall(key)" is a tiny, unnamed function with no
        # arguments that just calls client.hgetall(key) when run - used
        # here instead of a full "async def" because the operation is a
        # single simple call, not several steps.
        return _decode_redis_hash(data) if data else None

    async def _delete_key(self, operation: str, key: str, **context: Any) -> bool:
        client = self._connected_redis_client()
        deleted = await self._run_cache_operation(operation, lambda: client.delete(key), **context)
        return deleted > 0

    def _latest_responses_api_response_id_key(self, session_id: str) -> str:
        return self._build_key(CacheKeyNamespace.CHAT_MAPPING, f"last_response:{session_id}")

    # Connections
    async def connect(self) -> None:
        if self.redis_client is not None:
            # Step 1: if a connection already exists, close it first
            # instead of silently leaking the old one when a new one is made.
            await self.disconnect()

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
            retry=self._redis_retry_policy,
        )
        # Step 2: build the client object. Creating it does not yet prove
        # Redis is actually reachable - that is what the PING below checks.
        try:
            # Step 3: PING is the standard way to check "is the server
            # actually there and responding?" before trusting the connection.
            await self._redis_circuit_breaker.call_async(client.ping)
        except AuthenticationError as exc:
            # Step 4a: wrong password/credentials - close the half-open
            # client and report this specific, actionable failure.
            await client.aclose()
            raise CacheAuthenticationError(f"Redis authentication failed: {exc}") from exc
        except (RedisConnectionError, RedisTimeoutError, CircuitBreakerError, OSError) as exc:
            # Step 4b: any other reason the server could not be reached
            # (network down, wrong host/port, breaker already open, etc.).
            await client.aclose()
            raise CacheConnectionError(f"Could not connect to Redis: {exc}") from exc
        # Step 5: only keep the connection around once it is confirmed working.
        self.redis_client = client
        logger.info("Connected to Redis", host=self.config.host, port=self.config.port)

    async def disconnect(self) -> None:
        if self.redis_client is None:
            # Nothing to do if there was never a connection.
            return
        try:
            await self.redis_client.aclose()
        except Exception as exc:
            # Closing should not normally fail, but if it does, log it
            # rather than letting shutdown crash the whole app over it.
            logger.error(f"Error during Redis disconnect: {exc}", exc_info=True)
        finally:
            # "finally" runs whether or not the try block raised - this
            # guarantees self.redis_client is always cleared to None after
            # a disconnect attempt, even if closing it raised an error.
            self.redis_client = None

    async def health_check(self) -> bool:
        if self.redis_client is None:
            return False
        try:
            return await self.redis_client.ping() is True
        except Exception as exc:
            # A failed health check is expected, recoverable information
            # (not a bug), so it is logged as a warning and reported as
            # "unhealthy" (False) rather than raised as an exception.
            logger.warning(f"Redis health check failed: {exc}")
            return False

    # Sessions
    async def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        key = self._build_key(CacheKeyNamespace.SESSION, session_id)
        return await self._load_hash("get session metadata", key, session_id=session_id)

    async def set_session_metadata(
        self, session_id: str, metadata: dict[str, Any], ttl: int | None = None
    ) -> bool:
        key = self._build_key(CacheKeyNamespace.SESSION, session_id)
        ttl = ttl or self.config.default_session_ttl
        # "ttl or self.config.default_session_ttl" means: use the TTL the
        # caller passed in, or fall back to the configured default if the
        # caller gave None or 0 (both are "falsy").
        await self._store_hash("set session metadata", key, metadata, ttl, session_id=session_id)
        return True

    async def touch_session(self, session_id: str) -> bool:
        client = self._connected_redis_client()
        keys = [
            self._build_key(CacheKeyNamespace.SESSION, session_id),
            self._build_key(CacheKeyNamespace.RESPONSE_CHAIN, session_id),
            self._latest_responses_api_response_id_key(session_id),
        ]
        # keys: every Redis key related to this one session that should
        # have its expiry refreshed together, so they all expire in sync.
        ttl = self.config.default_session_ttl

        async def touch() -> int:
            # Step 1: reset the expiry countdown ("touch") on each related
            # key, counting how many actually existed and were refreshed.
            touched = 0
            for key in keys:
                if await client.expire(key, ttl):
                    touched += 1
            return touched

        touched = await self._run_cache_operation("touch session", touch, session_id=session_id)
        return touched > 0

    async def delete_session(self, session_id: str) -> bool:
        keys = [
            self._build_key(CacheKeyNamespace.SESSION, session_id),
            self._build_key(CacheKeyNamespace.RESPONSE_CHAIN, session_id),
            self._build_key(CacheKeyNamespace.HISTORY, session_id),
            self._build_key(CacheKeyNamespace.HISTORY, session_id, "messages"),
            self._latest_responses_api_response_id_key(session_id),
        ]
        # keys: every piece of Redis data that belongs to this session,
        # deleted together so nothing is left behind.
        client = self._connected_redis_client()
        deleted = await self._run_cache_operation(
            "delete session", lambda: client.delete(*keys), session_id=session_id
        )
        # "client.delete(*keys)" unpacks the keys list into separate
        # arguments, so Redis deletes all of them in a single command
        # instead of one call per key.
        logger.info("Deleted session data", session_id=session_id, keys_deleted=deleted)
        return deleted > 0

    async def get_active_session_context(self, session_id: str) -> ActiveSessionContext | None:
        session_key = self._build_key(CacheKeyNamespace.SESSION, session_id)
        last_response_key = self._latest_responses_api_response_id_key(session_id)
        client = self._connected_redis_client()

        async def fetch() -> list[Any]:
            # "async with client.pipeline(transaction=True) as pipe:" opens
            # a Redis PIPELINE: several commands are queued up and then
            # sent to Redis together in one network round trip. With
            # "transaction=True", Redis runs them in order without another
            # client's commands interleaving; it does not roll back errors.
            async with client.pipeline(transaction=True) as pipe:
                pipe.hgetall(session_key)
                pipe.get(last_response_key)
                return await pipe.execute()
                # pipe.execute() actually sends the queued commands and
                # returns their results as a list, in the same order they
                # were queued.

        data, last_response_id = await self._run_cache_operation(
            "get active session context", fetch, session_id=session_id
        )
        # Unpacks the two-item list from fetch() into two separate
        # variables, in the order the pipeline commands were queued above.
        if not data:
            # Step 1: no stored session metadata means the session is not
            # active (expired or never existed) - report "not found"
            # rather than building a context out of nothing.
            return None

        metadata = _decode_redis_hash(data)
        return ActiveSessionContext(
            session_id=session_id,
            metadata=metadata,
            last_response_id=last_response_id,
            vector_store_id=metadata.get("vector_store_id"),
            root_response_id=metadata.get("root_response_id"),
        )

    # OpenAI Responses API IDs
    async def append_responses_api_response_id(
        self, session_id: str, responses_api_response_id: str
    ) -> bool:
        key = self._build_key(CacheKeyNamespace.RESPONSE_CHAIN, session_id)
        client = self._connected_redis_client()

        async def write() -> None:
            async with client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, responses_api_response_id)
                # rpush: add this Responses API response ID to the stored list.
                pipe.ltrim(key, -self.config.max_chain_length, -1)
                # ltrim: keep only the LAST max_chain_length entries,
                # dropping older ones so the list cannot grow forever.
                pipe.expire(key, self.config.default_session_ttl)
                await pipe.execute()

        await self._run_cache_operation(
            "append Responses API response ID", write, session_id=session_id
        )
        return True

    async def set_latest_responses_api_response_id(
        self, session_id: str, responses_api_response_id: str, ttl: int | None = None
    ) -> bool:
        key = self._latest_responses_api_response_id_key(session_id)
        ttl = ttl or self.config.default_session_ttl
        client = self._connected_redis_client()
        await self._run_cache_operation(
            "set latest Responses API response ID",
            lambda: client.setex(key, ttl, responses_api_response_id),
            session_id=session_id,
        )
        # setex: Redis's "set with expiry" command - stores the value and
        # sets its TTL in one step.
        return True

    # Chat history
    async def set_chat_history(
        self, session_id: str, history: list[dict[str, Any]], ttl: int | None = None
    ) -> bool:
        key = self._build_key(CacheKeyNamespace.HISTORY, session_id)
        ttl = ttl or self.config.default_history_ttl
        client = self._connected_redis_client()

        async def write() -> None:
            await client.setex(key, ttl, json.dumps(history))
            # The whole history list is serialized to one JSON string,
            # since Redis's plain "set" only stores a single string value.

        await self._run_cache_operation("set chat history", write, session_id=session_id)
        return True

    async def get_chat_history(self, session_id: str) -> list[dict[str, Any]] | None:
        key = self._build_key(CacheKeyNamespace.HISTORY, session_id)
        client = self._connected_redis_client()

        async def read() -> list[dict[str, Any]] | None:
            raw = await client.get(key)
            # If nothing was cached (raw is empty/None), return None
            # instead of trying to parse an empty value as JSON.
            return json.loads(raw) if raw else None

        return await self._run_cache_operation("get chat history", read, session_id=session_id)

    async def delete_chat_history(self, session_id: str) -> bool:
        key = self._build_key(CacheKeyNamespace.HISTORY, session_id)
        return await self._delete_key("delete chat history", key, session_id=session_id)

    async def add_message(
        self,
        session_id: str,
        message: dict[str, Any],
        max_messages: int | None = None,
    ) -> bool:
        max_messages = max_messages or self.config.max_chain_length
        messages_key = self._build_key(CacheKeyNamespace.HISTORY, session_id, "messages")
        session_key = self._build_key(CacheKeyNamespace.SESSION, session_id)
        ttl = self.config.default_session_ttl
        client = self._connected_redis_client()

        async def write() -> None:
            payload = json.dumps(message)
            async with client.pipeline(transaction=True) as pipe:
                pipe.lpush(messages_key, payload)
                # lpush: add this message to the FRONT of the stored list
                # (so the newest message is always first).
                pipe.ltrim(messages_key, 0, max_messages - 1)
                # ltrim: keep only the first max_messages entries, dropping
                # the oldest ones once the list grows too long.
                pipe.llen(messages_key)
                # llen: count how many messages are stored now, after trimming.
                pipe.expire(messages_key, ttl)
                _, _, message_count, _ = await pipe.execute()
                # pipe.execute() returns one result per queued command, in
                # order: [lpush result, ltrim result, llen result, expire
                # result]. "_" is the conventional name for "a value we are
                # not going to use" - only the llen result (message_count)
                # is actually needed here.
            if await client.exists(session_key):
                # Step: only update the session's message count if the
                # session itself still exists - there is nothing useful to
                # update on a session that has already expired.
                await client.hset(session_key, "message_count", json.dumps(message_count))
                await client.expire(session_key, ttl)

        await self._run_cache_operation("add message", write, session_id=session_id)
        return True

    # Streaming state
    async def set_streaming_state(
        self,
        session_id: str,
        request_id: str,
        state: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        key = self._build_key(CacheKeyNamespace.STREAMING, session_id, request_id)
        ttl = ttl or self.config.default_streaming_ttl
        await self._store_hash("set streaming state", key, state, ttl, session_id=session_id)
        return True

    async def clear_streaming_state(self, session_id: str, request_id: str) -> bool:
        key = self._build_key(CacheKeyNamespace.STREAMING, session_id, request_id)
        return await self._delete_key("clear streaming state", key, session_id=session_id)

    # Vector store validation
    async def cache_vector_store_validation(
        self,
        vector_store_id: str,
        validation_data: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        key = self._build_key(CacheKeyNamespace.VECTOR_STORE, vector_store_id)
        ttl = ttl or self.config.default_vector_store_ttl
        await self._store_hash("cache vector store validation", key, validation_data, ttl)
        return True

    async def get_vector_store_validation(self, vector_store_id: str) -> dict[str, Any] | None:
        key = self._build_key(CacheKeyNamespace.VECTOR_STORE, vector_store_id)
        return await self._load_hash("get vector store validation", key)
