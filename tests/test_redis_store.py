"""Exercise RedisProvider against a mocked Redis client."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from redis.exceptions import ConnectionError as RedisConnectionError

from src.adapters.redis.exceptions import CacheOperationError
from src.adapters.redis.redis_store import RedisProvider
from src.adapters.redis.schema import CacheConfig, CacheOperationType


def make_pipeline(results):
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    pipe.execute = AsyncMock(return_value=results)
    return pipe


def make_provider(pipeline_results=None):
    provider = RedisProvider(CacheConfig(key_prefix="test"))
    client = AsyncMock()
    pipe = make_pipeline(pipeline_results or [])
    client.pipeline = MagicMock(return_value=pipe)
    provider.redis_client = client
    return provider, client, pipe


class RedisProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_passes_ssl_and_retry_to_client(self) -> None:
        provider = RedisProvider(CacheConfig(ssl=True))
        client = AsyncMock()
        redis_path = "src.adapters.redis.redis_store.aioredis.Redis"
        with patch(redis_path, return_value=client) as redis_cls:
            await provider.connect()
        kwargs = redis_cls.call_args.kwargs
        self.assertTrue(kwargs["ssl"])
        self.assertIs(kwargs["retry"], provider._retry)
        self.assertIs(provider.redis_client, client)

    async def test_add_message_uses_prefixed_key_and_skips_missing_session(self) -> None:
        provider, client, pipe = make_provider([1, True, 3, True])
        client.exists.return_value = 0

        await provider.add_message("s1", {"role": "user", "content": "hi"})

        messages_key = provider._build_key(CacheOperationType.HISTORY, "s1", "messages")
        self.assertEqual(pipe.lpush.call_args.args[0], messages_key)
        client.hset.assert_not_awaited()

    async def test_add_message_updates_existing_session_count(self) -> None:
        provider, client, _ = make_provider([1, True, 3, True])
        client.exists.return_value = 1

        await provider.add_message("s1", {"role": "user", "content": "hi"})

        session_key = provider._build_key(CacheOperationType.SESSION, "s1")
        client.hset.assert_awaited_once_with(session_key, "message_count", "3")

    async def test_delete_session_removes_message_list(self) -> None:
        provider, client, _ = make_provider()
        client.delete.return_value = 1

        await provider.delete_session("s1")

        deleted_keys = client.delete.call_args.args
        messages_key = provider._build_key(CacheOperationType.HISTORY, "s1", "messages")
        self.assertIn(messages_key, deleted_keys)

    async def test_active_session_context_returns_decoded_vector_store_id(self) -> None:
        metadata = {"vector_store_id": json.dumps("vs_1"), "root_response_id": json.dumps("r0")}
        provider, _, _ = make_provider([True, metadata, "r5"])

        context = await provider.get_active_session_context("s1")

        self.assertEqual(context.vector_store_id, "vs_1")
        self.assertEqual(context.root_response_id, "r0")
        self.assertEqual(context.last_response_id, "r5")

    async def test_active_session_context_is_none_for_missing_session(self) -> None:
        provider, _, _ = make_provider([False, {}, None])
        self.assertIsNone(await provider.get_active_session_context("s1"))

    async def test_redis_failure_maps_to_cache_operation_error(self) -> None:
        provider, client, _ = make_provider()
        client.get.side_effect = RedisConnectionError("down")
        with self.assertRaises(CacheOperationError):
            await provider.get_cached_history("s1")

    def test_config_is_built_from_settings(self) -> None:
        settings = SimpleNamespace(
            redis_host="redis.internal",
            redis_port=6380,
            redis_db=2,
            redis_password="secret",
            redis_max_connections=10,
            redis_socket_timeout=3,
            redis_key_prefix="app",
            redis_default_session_ttl=100,
            redis_default_history_ttl=50,
        )

        config = CacheConfig.from_settings(settings)

        self.assertEqual((config.host, config.port, config.db), ("redis.internal", 6380, 2))
        self.assertEqual((config.key_prefix, config.default_session_ttl), ("app", 100))


if __name__ == "__main__":
    unittest.main()
