"""Exercise RedisProvider against a mocked Redis client."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from redis.exceptions import ConnectionError as RedisConnectionError

from src.adapters.redis.exceptions import CacheOperationError, CacheTTLError
from src.adapters.redis.redis_store import RedisProvider
from src.adapters.redis.schema import CacheKeyNamespace, RedisCacheConfig


def make_pipeline(results):
    pipe = MagicMock()
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    pipe.execute = AsyncMock(return_value=results)
    return pipe


def make_provider(pipeline_results=None):
    provider = RedisProvider(RedisCacheConfig(key_prefix="test"))
    client = AsyncMock()
    pipe = make_pipeline(pipeline_results or [])
    client.pipeline = MagicMock(return_value=pipe)
    provider.redis_client = client
    return provider, client, pipe


class RedisProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_passes_ssl_and_retry_to_client(self) -> None:
        provider = RedisProvider(RedisCacheConfig(ssl=True))
        client = AsyncMock()
        redis_path = "src.adapters.redis.redis_store.aioredis.Redis"
        with patch(redis_path, return_value=client) as redis_cls:
            await provider.connect()
        kwargs = redis_cls.call_args.kwargs
        self.assertTrue(kwargs["ssl"])
        self.assertIs(kwargs["retry"], provider._redis_retry_policy)
        self.assertIs(provider.redis_client, client)

    async def test_reconnect_closes_previous_client(self) -> None:
        provider, previous_client, _ = make_provider()
        new_client = AsyncMock()
        with patch("src.adapters.redis.redis_store.aioredis.Redis", return_value=new_client):
            await provider.connect()

        previous_client.aclose.assert_awaited_once()
        new_client.ping.assert_awaited_once()
        self.assertIs(provider.redis_client, new_client)

    async def test_failed_connect_closes_new_client(self) -> None:
        provider = RedisProvider(RedisCacheConfig())
        client = AsyncMock()
        client.ping.side_effect = RuntimeError("unexpected setup failure")
        with patch("src.adapters.redis.redis_store.aioredis.Redis", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "unexpected setup failure"):
                await provider.connect()

        client.aclose.assert_awaited_once()
        self.assertIsNone(provider.redis_client)

    async def test_add_message_uses_prefixed_key_and_skips_missing_session(self) -> None:
        provider, client, pipe = make_provider([1, True, 3, True])
        client.exists.return_value = 0

        await provider.add_message("s1", {"role": "user", "content": "hi"})

        messages_key = provider._build_key(CacheKeyNamespace.HISTORY, "s1", "messages")
        self.assertEqual(pipe.lpush.call_args.args[0], messages_key)
        client.hset.assert_not_awaited()

    async def test_add_message_updates_existing_session_count(self) -> None:
        provider, client, _ = make_provider([1, True, 3, True])
        client.exists.return_value = 1

        await provider.add_message("s1", {"role": "user", "content": "hi"})

        session_key = provider._build_key(CacheKeyNamespace.SESSION, "s1")
        client.hset.assert_awaited_once_with(session_key, "message_count", "3")

    async def test_delete_session_removes_message_list(self) -> None:
        provider, client, _ = make_provider()
        client.delete.return_value = 1

        async def streaming_keys(**_):
            yield "test:streaming:s1:req1"
            yield "test:streaming:s10:req2"

        client.scan_iter = MagicMock(side_effect=streaming_keys)

        await provider.delete_session("s1")

        deleted_keys = client.delete.call_args.args
        messages_key = provider._build_key(CacheKeyNamespace.HISTORY, "s1", "messages")
        self.assertIn(messages_key, deleted_keys)
        self.assertIn("test:streaming:s1:req1", deleted_keys)
        self.assertNotIn("test:streaming:s10:req2", deleted_keys)

    async def test_invalid_ttl_never_writes_or_deletes_session(self) -> None:
        provider, client, pipe = make_provider()
        for ttl in (0, -1):
            with self.assertRaises(CacheTTLError):
                await provider.set_session_metadata("s1", {"user": "a"}, ttl=ttl)
        pipe.execute.assert_not_awaited()
        client.expire.assert_not_awaited()

    async def test_session_hash_and_expiry_use_one_transaction(self) -> None:
        provider, _, pipe = make_provider([1, True])

        await provider.set_session_metadata("s1", {"user": "a"}, ttl=60)

        pipe.hset.assert_called_once_with("test:session:s1", mapping={"user": '"a"'})
        pipe.expire.assert_called_once_with("test:session:s1", 60)
        pipe.execute.assert_awaited_once()

    async def test_responses_api_id_methods_keep_existing_redis_keys(self) -> None:
        provider, client, pipe = make_provider()

        await provider.append_responses_api_response_id("s1", responses_api_response_id="resp_1")
        await provider.set_latest_responses_api_response_id(
            "s1", responses_api_response_id="resp_1"
        )

        pipe.rpush.assert_called_once_with("test:response_chain:s1", "resp_1")
        client.setex.assert_awaited_once_with("test:chat_mapping:last_response:s1", 7200, "resp_1")

    async def test_active_session_context_returns_decoded_vector_store_id(self) -> None:
        metadata = {"vector_store_id": json.dumps("vs_1"), "root_response_id": json.dumps("r0")}
        provider, _, pipe = make_provider([metadata, "r5"])

        context = await provider.get_active_session_context("s1")

        self.assertEqual(context.vector_store_id, "vs_1")
        self.assertEqual(context.root_response_id, "r0")
        self.assertEqual(context.last_response_id, "r5")
        pipe.exists.assert_not_called()

    async def test_active_session_context_is_none_for_missing_session(self) -> None:
        provider, _, _ = make_provider([{}, None])
        self.assertIsNone(await provider.get_active_session_context("s1"))

    async def test_redis_failure_maps_to_cache_operation_error(self) -> None:
        provider, client, _ = make_provider()
        client.get.side_effect = RedisConnectionError("down")
        with self.assertRaises(CacheOperationError):
            await provider.get_chat_history("s1")

    def test_config_is_built_from_settings(self) -> None:
        settings = SimpleNamespace(
            redis_host="redis.internal",
            redis_port=6380,
            redis_db=2,
            redis_password="secret",
            redis_max_connections=10,
            redis_socket_timeout=3,
            redis_socket_connect_timeout=4,
            redis_ssl=True,
            redis_key_prefix="app",
            redis_default_session_ttl=100,
            redis_default_history_ttl=50,
        )

        config = RedisCacheConfig.from_settings(settings)

        self.assertEqual((config.host, config.port, config.db), ("redis.internal", 6380, 2))
        self.assertEqual((config.key_prefix, config.default_session_ttl), ("app", 100))
        self.assertEqual((config.socket_connect_timeout, config.ssl), (4, True))


if __name__ == "__main__":
    unittest.main()
