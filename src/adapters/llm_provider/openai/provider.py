"""OpenAI provider: configuration and API routing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, OpenAI

from src.config import get_settings

from ..base import APIType, LLMConfig, LLMProvider, LLMResponse, StreamMode
from ..exceptions import LLMProviderError
from .chat_completions import ChatCompletionsClient
from .responses_api import ResponsesApiClient


class OpenAIProvider(LLMProvider):
    """Route OpenAI requests using an API key and an ``LLMConfig``.

    Constructs Chat Completions and Responses API clients.
    Requires a configured API key. Public calls return ``LLMResponse`` or
    ``AsyncIterator[str]`` according to the requested stream mode.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMConfig | None = None,
        model_name: str | None = None,
        api_type: APIType | None = None,
        stream_mode: StreamMode | None = None,
        **kwargs: Any,
    ) -> None:
        """Build SDK clients and routing configuration.

        Requires API key. Raises LLMProviderError if key is unavailable.
        """
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        if not self._api_key:
            raise LLMProviderError("OpenAI API key is required.")

        if config is None:
            config = LLMConfig(
                model_name=model_name or settings.openai_model_name or "gpt-4",
                api_type=api_type or APIType.CHAT_COMPLETION,
                stream_mode=stream_mode or StreamMode.DISABLED,
                token_limit=kwargs.pop("token_limit", 8192),
                temperature=kwargs.pop("temperature", settings.openai_temperature),
                top_p=kwargs.pop("top_p", settings.openai_top_p),
                max_tokens=kwargs.pop("max_tokens", settings.openai_max_output_tokens),
                **kwargs,
            )

        super().__init__(config)
        self._client = OpenAI(api_key=self._api_key)
        self._async_client = AsyncOpenAI(api_key=self._api_key)

        self._chat_client = ChatCompletionsClient(config, self)
        self._responses_client = ResponsesApiClient(config, self)

    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Route a prompt to the configured API."""
        if not stream:
            stream = self.should_stream()

        if self.config.api_type == APIType.RESPONSES:
            return await self.execute_responses_api(prompt, system_prompt, stream=stream, **kwargs)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.execute_chat_completion(messages, stream=stream, **kwargs)

    async def execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Call Chat Completions for role/content messages."""
        return await self._chat_client.execute(messages, stream=stream, **kwargs)

    async def execute_responses_api(
        self,
        input_message: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Call the Responses API with optional instructions and tools."""
        return await self._responses_client.execute(
            input_message, instructions, previous_response_id, tools, stream=stream, **kwargs
        )

    async def execute_responses_api_raw(
        self,
        input_message: str | list[dict[str, Any]],
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Return raw SDK Responses API object without normalization."""
        return await self._responses_client.execute_raw(
            input_message, instructions, previous_response_id, tools, stream=stream, **kwargs
        )

    async def close(self) -> None:
        """Close SDK clients owned by this provider."""
        await self._async_client.close()
        self._client.close()
