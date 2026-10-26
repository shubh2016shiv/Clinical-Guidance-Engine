"""Exercise the OpenAI provider clients without network requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai

from src.adapters.llm_provider.base import APIType, LLMConfig
from src.adapters.llm_provider.exceptions import (
    LLMProviderAuthenticationError,
    LLMProviderExecutionError,
    LLMProviderInitializationError,
    LLMProviderRateLimitError,
    LLMProviderStreamingError,
    LLMProviderTokenLimitError,
)
from src.adapters.llm_provider.openai import OpenAIProvider, OpenAITextEmbeddingClient


def _make_sdk_error(error_cls: type[Exception]) -> Exception:
    """Build a real OpenAI SDK exception with the fields its constructor requires."""
    request = httpx.Request("POST", "https://api.openai.com/v1/x")
    response = httpx.Response(status_code=401, request=request)
    return error_cls(message="boom", response=response, body=None)


class OpenAIAdapterRefactorTests(unittest.IsolatedAsyncioTestCase):
    """Check public response shapes using fake SDK clients."""

    async def test_token_budget_reserves_space_for_the_reply(self) -> None:
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", token_limit=5, max_tokens=2),
        )
        with patch.object(provider, "estimate_tokens", return_value=3):
            self.assertTrue(provider.validate_token_limit("input"))
            self.assertFalse(provider.validate_token_limit("input", max_tokens=3))

    async def test_close_releases_async_sdk_client(self) -> None:
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))
        provider._async_openai_client.close = AsyncMock()

        await provider.close()

        provider._async_openai_client.close.assert_awaited_once()

    def _chat_provider_capturing_requests(self, **config_fields):
        """A provider whose SDK boundary records what would be sent to OpenAI."""
        provider = OpenAIProvider(
            api_key="test-key", config=LLMConfig(model_name="test-model", **config_fields)
        )
        sdk_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
            usage=None,
            model="test-model",
            id="chat-1",
            created=1,
        )
        create = AsyncMock(return_value=sdk_response)
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return provider, create

    async def test_configured_reply_limit_is_sent_as_max_completion_tokens(self) -> None:
        """The SDK deprecates max_tokens; newer models reject it outright."""
        provider, create = self._chat_provider_capturing_requests(max_tokens=100)

        await provider.execute_chat_completion([{"role": "user", "content": "hi"}])

        sent = create.await_args.kwargs
        self.assertEqual(sent["max_completion_tokens"], 100)
        self.assertNotIn("max_tokens", sent)

    async def test_per_call_max_tokens_overrides_the_configured_limit(self) -> None:
        provider, create = self._chat_provider_capturing_requests(max_tokens=100)

        await provider.execute_chat_completion([{"role": "user", "content": "hi"}], max_tokens=50)

        sent = create.await_args.kwargs
        self.assertEqual(sent["max_completion_tokens"], 50)
        self.assertNotIn("max_tokens", sent)

    async def test_an_explicit_max_completion_tokens_wins_over_the_configured_limit(self) -> None:
        provider, create = self._chat_provider_capturing_requests(max_tokens=100)

        await provider.execute_chat_completion(
            [{"role": "user", "content": "hi"}], max_completion_tokens=70
        )

        sent = create.await_args.kwargs
        self.assertEqual(sent["max_completion_tokens"], 70)
        self.assertNotIn("max_tokens", sent)

    async def test_no_reply_limit_is_sent_when_none_is_configured(self) -> None:
        provider, create = self._chat_provider_capturing_requests(max_tokens=None)

        await provider.execute_chat_completion([{"role": "user", "content": "hi"}])

        sent = create.await_args.kwargs
        self.assertNotIn("max_completion_tokens", sent)
        self.assertNotIn("max_tokens", sent)

    async def test_budget_check_honours_an_explicit_max_completion_tokens(self) -> None:
        """A reply limit passed under the new name must still count against the budget."""
        provider, create = self._chat_provider_capturing_requests(token_limit=10, max_tokens=None)

        with patch.object(provider, "estimate_tokens", return_value=3):
            with self.assertRaises(LLMProviderTokenLimitError):
                await provider.execute_chat_completion(
                    [{"role": "user", "content": "hi"}], max_completion_tokens=8
                )
        create.assert_not_awaited()

    async def test_chat_completion_maps_response_and_parameters(self) -> None:
        """A valid chat reply becomes ``LLMResponse`` with usage data."""
        # Construct a provider with a local key and existing config contract.
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))
        # Model the SDK's first choice and token usage.
        sdk_response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="hello"), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            model="test-model",
            id="chat-1",
            created=1,
        )
        # Replace the asynchronous SDK boundary with a deterministic reply.
        create = AsyncMock(return_value=sdk_response)
        # Supply the minimum SDK attribute chain read by the chat module.
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        # Execute through the unchanged public method.
        result = await provider.execute_chat_completion([{"role": "user", "content": "hi"}])
        # Check the normalized response and SDK request.
        self.assertEqual(
            (result.content, result.total_tokens, result.response_id), ("hello", 5, "chat-1")
        )
        # Verify the configured model reached the SDK.
        self.assertEqual(create.await_args.kwargs["model"], "test-model")

    async def test_chat_completion_forwards_a_structured_output_format(self) -> None:
        """Without this the SDK would never see the schema and the reply would be free text."""
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))
        sdk_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), finish_reason="stop")],
            usage=None,
            model="test-model",
            id="chat-2",
            created=1,
        )
        create = AsyncMock(return_value=sdk_response)
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        response_format = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}

        await provider.execute_chat_completion(
            [{"role": "user", "content": "hi"}], response_format=response_format, seed=123
        )

        self.assertEqual(create.await_args.kwargs["response_format"], response_format)
        self.assertEqual(create.await_args.kwargs["seed"], 123)

    async def test_responses_stream_returns_sdk_events(self) -> None:
        """A Responses API stream keeps the SDK's typed events."""
        # Construct a provider configured for the Responses API.
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", api_type=APIType.RESPONSES),
        )

        # Provide typed Responses events, including a non-text completion event.
        async def events():
            """Yield fake SDK events; no preconditions, async event output."""
            yield SimpleNamespace(type="response.output_text.delta", delta="hello")
            yield SimpleNamespace(type="response.completed")

        # Return the fake event stream from the SDK boundary.
        create = AsyncMock(return_value=events())
        # Supply the minimum Responses API attribute chain.
        provider._async_openai_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        # Request a stream through the provider's single Responses API method.
        response_stream = await provider.execute_responses_api("hi", stream=True)
        # Read both events, including the completion event with no text.
        received_events = [event async for event in response_stream]
        # Verify event order and streaming SDK arguments.
        self.assertEqual(
            [event.type for event in received_events],
            ["response.output_text.delta", "response.completed"],
        )
        self.assertEqual(received_events[0].delta, "hello")
        # Confirm that streaming was requested from the SDK.
        self.assertTrue(create.await_args.kwargs["stream"])

    async def test_responses_call_returns_sdk_object_without_stream_flag(self) -> None:
        """A non-streaming call returns the SDK response and sends stream=False."""
        # Construct a provider configured for the Responses API.
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", api_type=APIType.RESPONSES),
        )
        # Model the SDK response object the agent managers read directly.
        sdk_response = SimpleNamespace(id="resp-1", output=[])
        # Replace the asynchronous SDK boundary with a deterministic reply.
        create = AsyncMock(return_value=sdk_response)
        provider._async_openai_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        # Send structured input items, as the tool-output continuation does.
        input_items = [{"type": "function_call_output", "call_id": "c1", "output": "{}"}]
        result = await provider.execute_responses_api(
            input_message=input_items, previous_response_id="resp-0", stream=False, model="m"
        )
        # The SDK object is returned unchanged.
        self.assertIs(result, sdk_response)
        # The request carries the items, chain id, model override and stream=False.
        self.assertEqual(
            create.await_args.kwargs,
            {"model": "m", "input": input_items, "stream": False, "previous_response_id": "resp-0"},
        )

    async def test_responses_call_forwards_storage_choice(self) -> None:
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", api_type=APIType.RESPONSES),
        )
        create = AsyncMock(return_value=SimpleNamespace(id="resp-1"))
        provider._async_openai_client = SimpleNamespace(responses=SimpleNamespace(create=create))

        await provider.execute_responses_api("question", stream=False, store=True)

        self.assertTrue(create.await_args.kwargs["store"])

    async def test_missing_api_key_raises_initialization_error(self) -> None:
        with patch("src.adapters.llm_provider.openai.provider.get_settings") as get_settings:
            get_settings.return_value = SimpleNamespace(openai_api_key=None)
            with self.assertRaises(LLMProviderInitializationError):
                OpenAIProvider(config=LLMConfig(model_name="test-model"))

    async def test_embeddings_missing_api_key_raises_initialization_error(self) -> None:
        with patch("src.adapters.llm_provider.openai.embeddings.get_settings") as get_settings:
            get_settings.return_value = SimpleNamespace(openai_api_key=None)
            with self.assertRaises(LLMProviderInitializationError):
                OpenAITextEmbeddingClient(embedding_model_name="test-model", dimensions=8)

    async def test_chat_completion_exceeding_token_budget_raises(self) -> None:
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", token_limit=1),
        )
        with self.assertRaises(LLMProviderTokenLimitError):
            await provider.execute_chat_completion([{"role": "user", "content": "x" * 100}])

    async def test_chat_completion_maps_authentication_error(self) -> None:
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))
        create = AsyncMock(side_effect=_make_sdk_error(openai.AuthenticationError))
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with self.assertRaises(LLMProviderAuthenticationError):
            await provider.execute_chat_completion([{"role": "user", "content": "hi"}])

    async def test_chat_completion_maps_rate_limit_error(self) -> None:
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))
        create = AsyncMock(side_effect=_make_sdk_error(openai.RateLimitError))
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with self.assertRaises(LLMProviderRateLimitError):
            await provider.execute_chat_completion([{"role": "user", "content": "hi"}])

    async def test_chat_completion_maps_unexpected_error_to_execution_error(self) -> None:
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))
        create = AsyncMock(side_effect=RuntimeError("network down"))
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with self.assertRaises(LLMProviderExecutionError):
            await provider.execute_chat_completion([{"role": "user", "content": "hi"}])

    async def test_chat_completion_stream_failing_to_start_maps_sdk_error(self) -> None:
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))
        create = AsyncMock(side_effect=_make_sdk_error(openai.RateLimitError))
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        text_stream = await provider.execute_chat_completion(
            [{"role": "user", "content": "hi"}], stream=True
        )
        with self.assertRaises(LLMProviderRateLimitError):
            async for _ in text_stream:
                pass

    async def test_chat_completion_stream_failing_mid_stream_raises_streaming_error(self) -> None:
        provider = OpenAIProvider(api_key="test-key", config=LLMConfig(model_name="test-model"))

        async def broken_stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))])
            raise RuntimeError("connection dropped")

        create = AsyncMock(return_value=broken_stream())
        provider._async_openai_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        text_stream = await provider.execute_chat_completion(
            [{"role": "user", "content": "hi"}], stream=True
        )
        with self.assertRaises(LLMProviderStreamingError):
            async for _ in text_stream:
                pass

    async def test_responses_api_maps_sdk_error(self) -> None:
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", api_type=APIType.RESPONSES),
        )
        create = AsyncMock(side_effect=_make_sdk_error(openai.AuthenticationError))
        provider._async_openai_client = SimpleNamespace(responses=SimpleNamespace(create=create))

        with self.assertRaises(LLMProviderAuthenticationError):
            await provider.execute_responses_api("question", stream=False)

    async def test_embedding_request_maps_sdk_error(self) -> None:
        client = OpenAITextEmbeddingClient(
            embedding_model_name="test-model", dimensions=8, api_key="test-key"
        )
        client._openai_client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=AsyncMock(side_effect=_make_sdk_error(openai.RateLimitError))
            )
        )
        with self.assertRaises(LLMProviderRateLimitError):
            await client.create_text_embedding("metformin")
