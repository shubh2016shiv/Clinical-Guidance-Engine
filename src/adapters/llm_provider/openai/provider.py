"""Own the async OpenAI client used by the app's two API helpers.

Big picture: this file defines ``OpenAIProvider``, the main object the rest
of the healthcare assistant uses to talk to OpenAI. It does three things:
1. Creates one shared connection ("client") to OpenAI's API.
2. Hands that connection to two helper objects - one for Chat Completions,
   one for the Responses API - so they can send requests through it.
3. Exposes simple methods (``execute``, ``execute_chat_completion``,
   ``execute_responses_api``) that the rest of the app calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

# AsyncOpenAI is OpenAI's official library for talking to their API using
# "async" code (see the note on async/await below). An async client lets
# the program do other work while waiting for OpenAI's servers to respond,
# instead of freezing until the answer comes back.
from src.config import get_settings

from ..base import APIType, LLMConfig, LLMProvider, LLMResponse, StreamMode
from ..exceptions import LLMProviderInitializationError
from .chat_completions import ChatCompletionsClient
from .responses_api import ResponsesApiClient


# OpenAIProvider: the concrete implementation of the LLMProvider "contract"
# from base.py, specifically for OpenAI. Needed because base.py only
# describes WHAT a provider must do - this class describes HOW OpenAI does it.
class OpenAIProvider(LLMProvider):
    """Share one async SDK client between Chat Completions and Responses."""

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMConfig | None = None,
        model_name: str | None = None,
        api_type: APIType | None = None,
        stream_mode: StreamMode | None = None,
        **llm_config_options: Any,
    ) -> None:
        """Read defaults and create the SDK client and API helpers."""
        # "**llm_config_options" collects any extra keyword arguments the
        # caller passes (like temperature=0.2) that are not already named
        # above, so callers can tweak settings without this method needing
        # to list every single possible option by name.

        # Step 1: figure out which API key to use - either the one passed
        # in directly, or the one configured for the whole app.
        settings = get_settings()
        openai_api_key = api_key or settings.openai_api_key
        if not openai_api_key:
            # Step 2: without a key we cannot do anything useful, so fail
            # loudly and immediately instead of letting a confusing error
            # happen later when a request is actually sent.
            raise LLMProviderInitializationError("OpenAI API key is required.")

        if config is None:
            # Step 3: if the caller did not build a full LLMConfig
            # themselves, build one from individual arguments and app
            # settings. ".pop(name, default)" removes "name" from the
            # llm_config_options dict and returns its value, or "default" if
            # it was not present - this avoids passing the same setting
            # twice into LLMConfig below. Whatever is left over in
            # llm_config_options after the pops below is spread into
            # LLMConfig too, in case it has other matching fields (like
            # frequency_penalty).
            config = LLMConfig(
                model_name=model_name or settings.openai_model_name,
                api_type=api_type or APIType.CHAT_COMPLETION,
                stream_mode=stream_mode or StreamMode.DISABLED,
                token_limit=llm_config_options.pop("token_limit", 8192),
                temperature=llm_config_options.pop("temperature", settings.openai_temperature),
                top_p=llm_config_options.pop("top_p", settings.openai_top_p),
                max_tokens=llm_config_options.pop("max_tokens", settings.openai_max_output_tokens),
                **llm_config_options,
            )

        # Step 4: let the parent class (LLMProvider in base.py) store the
        # config and log that this provider was created.
        super().__init__(config)
        # Step 5: create the actual connection to OpenAI, and the two helper
        # clients that know how to format Chat Completions and Responses API
        # requests. Both helpers are given "self" so they can reach back
        # into this provider (for example, to use the shared SDK client or
        # call validate_token_limit).
        self._async_openai_client = AsyncOpenAI(api_key=openai_api_key)
        self._chat_completions_client = ChatCompletionsClient(config, self)
        self._responses_api_client = ResponsesApiClient(config, self)

    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **request_options: Any,
    ) -> Any:
        """Send a plain prompt to the API selected by the config."""
        # "async def" means this function can be paused while it waits for
        # OpenAI's response, letting other code run in the meantime. Callers
        # must use "await provider.execute(...)" to run it.
        # Step 1: if the caller did not explicitly ask to stream, fall back
        # to this provider's configured default (see base.py should_stream).
        if not stream:
            stream = self.should_stream()

        # Step 2: the Responses API and Chat Completions expect different
        # request shapes, so branch based on which one this config uses.
        if self.config.api_type == APIType.RESPONSES:
            return await self.execute_responses_api(
                prompt, system_prompt, stream=stream, **request_options
            )

        # Step 3 (Chat Completions path): build the "messages" list Chat
        # Completions expects - an optional system instruction, then the
        # user's actual prompt.
        chat_messages: list[dict[str, str]] = []
        # chat_messages: the conversation turns being sent to the model.
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        chat_messages.append({"role": "user", "content": prompt})
        return await self.execute_chat_completion(chat_messages, stream=stream, **request_options)

    async def execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **chat_completion_options: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Return an app reply or a stream of chat text."""
        # This simply hands the work off to the Chat Completions helper -
        # this method exists so callers always go through "the provider",
        # without needing to know a separate helper object exists.
        return await self._chat_completions_client.create_chat_completion(
            messages, stream=stream, **chat_completion_options
        )

    async def execute_responses_api(
        self,
        input_message: str | list[dict[str, Any]],
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **responses_api_options: Any,
    ) -> Any:
        """Return the OpenAI Responses SDK result or event stream."""
        # Same idea as execute_chat_completion, but for the Responses API
        # helper. "previous_response_id" lets OpenAI continue an earlier
        # conversation server-side without resending the whole history.
        return await self._responses_api_client.create_response(
            input_message,
            instructions,
            previous_response_id,
            tools,
            stream=stream,
            **responses_api_options,
        )

    async def close(self) -> None:
        """Close the SDK client owned by this provider."""
        # Always call this when you are done using the provider (for
        # example in a "finally" block) so the network connection is
        # released cleanly instead of leaking resources.
        await self._async_openai_client.close()
