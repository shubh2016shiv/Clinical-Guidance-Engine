"""Exercise the OpenAI provider clients without network requests."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.adapters.llm_provider.base import APIType, LLMConfig
from src.adapters.llm_provider.openai import OpenAIProvider


class OpenAIAdapterRefactorTests(unittest.IsolatedAsyncioTestCase):
    """Check public response shapes using fake SDK clients."""

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
        provider._async_client = SimpleNamespace(
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

    async def test_responses_stream_yields_typed_text_deltas(self) -> None:
        """A Responses API stream yields only text delta events."""
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
        provider._async_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        # Request a stream through the unchanged public method.
        response_stream = await provider.execute_responses_api("hi", stream=True)
        # Consume the stream to check translated text values.
        chunks = [chunk async for chunk in response_stream]
        # Verify event order and streaming SDK arguments.
        self.assertEqual(chunks, ["hello"])
        # Confirm that streaming was requested from the SDK.
        self.assertTrue(create.await_args.kwargs["stream"])

    async def test_raw_responses_call_returns_sdk_object_without_stream_flag(self) -> None:
        """A non-streaming raw call returns the SDK response and sends stream=False."""
        # Construct a provider configured for the Responses API.
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", api_type=APIType.RESPONSES),
        )
        # Model the SDK response object the agent managers read directly.
        sdk_response = SimpleNamespace(id="resp-1", output=[])
        # Replace the asynchronous SDK boundary with a deterministic reply.
        create = AsyncMock(return_value=sdk_response)
        provider._async_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        # Send structured input items, as the tool-output continuation does.
        input_items = [{"type": "function_call_output", "call_id": "c1", "output": "{}"}]
        result = await provider.execute_responses_api_raw(
            input_message=input_items, previous_response_id="resp-0", stream=False, model="m"
        )
        # The SDK object is returned unchanged.
        self.assertIs(result, sdk_response)
        # The request carries the items, chain id, model override and stream=False.
        self.assertEqual(
            create.await_args.kwargs,
            {"model": "m", "input": input_items, "stream": False, "previous_response_id": "resp-0"},
        )

    async def test_raw_responses_call_forwards_storage_choice(self) -> None:
        provider = OpenAIProvider(
            api_key="test-key",
            config=LLMConfig(model_name="test-model", api_type=APIType.RESPONSES),
        )
        create = AsyncMock(return_value=SimpleNamespace(id="resp-1"))
        provider._async_client = SimpleNamespace(responses=SimpleNamespace(create=create))

        await provider.execute_responses_api_raw("question", stream=False, store=True)

        self.assertTrue(create.await_args.kwargs["store"])
