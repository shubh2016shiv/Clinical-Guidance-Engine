"""
OpenAI Provider Implementation.

Supports both Chat Completions and Responses APIs with full streaming
capabilities.  All public methods honour the :class:`LLMProvider`
contract defined in :mod:`.base`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, overload

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from src.config import get_settings
from src.logs import get_logger

from .base import APIType, LLMConfig, LLMProvider, LLMResponse, StreamMode
from .exceptions import (
    LLMProviderError,
    LLMProviderExecutionError,
    LLMProviderStreamingError,
    LLMProviderTokenLimitError,
)

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI provider supporting Chat Completions and Responses APIs.

    Both APIs can operate in streaming and non-streaming mode.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        api_key: str | None = None,
        config: LLMConfig | None = None,
        model_name: str | None = None,
        api_type: APIType | None = None,
        stream_mode: StreamMode | None = None,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key

        if not self._api_key:
            raise LLMProviderError("OpenAI API key is required.")

        # Build config when one isn't supplied ---------------------------------
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

        # Clients — async is the primary workhorse; sync kept for edge cases.
        self._client = OpenAI(api_key=self._api_key)
        self._async_client = AsyncOpenAI(api_key=self._api_key)

        logger.info(
            "OpenAI provider ready – model=%s api=%s",
            self.model_name,
            self.config.api_type.value,
        )

    # ------------------------------------------------------------------
    # Public API — execute
    # ------------------------------------------------------------------

    @overload
    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @overload
    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...

    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        # When the caller hasn't explicitly requested streaming, honour the
        # provider-level StreamMode configuration.
        if not stream:
            stream = self.should_stream()

        if self.config.api_type == APIType.RESPONSES:
            return await self.execute_responses_api(
                input_message=prompt,
                instructions=system_prompt,
                stream=stream,
                **kwargs,
            )

        # Chat Completions path ------------------------------------------------
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return await self.execute_chat_completion(messages=messages, stream=stream, **kwargs)

    # ------------------------------------------------------------------
    # Chat Completions API
    # ------------------------------------------------------------------

    @overload
    async def execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @overload
    async def execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...

    async def execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        try:
            total_prompt = " ".join(msg.get("content", "") for msg in messages)
            if not self.validate_token_limit(total_prompt, kwargs.get("max_tokens")):
                raise LLMProviderTokenLimitError("Token budget exceeded for chat completion.")

            params = self._build_chat_params(messages, stream, **kwargs)

            if stream:
                return self._stream_chat_completion(params)
            return await self._execute_chat_completion(params)

        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderExecutionError(f"Chat completion failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Responses API
    # ------------------------------------------------------------------

    @overload
    async def execute_responses_api(
        self,
        input_message: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> LLMResponse: ...

    @overload
    async def execute_responses_api(
        self,
        input_message: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...

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
        try:
            full_prompt = f"{instructions or ''}\n{input_message}"
            if not self.validate_token_limit(full_prompt, kwargs.get("max_tokens")):
                raise LLMProviderTokenLimitError("Token budget exceeded for Responses API call.")

            params: dict[str, Any] = {
                "model": kwargs.get("model", self.model_name),
                "input": input_message,
            }
            if instructions:
                params["instructions"] = instructions
            if previous_response_id:
                params["previous_response_id"] = previous_response_id
            if tools:
                params["tools"] = tools
                logger.debug(
                    "Responses API call includes %d tool(s): %s",
                    len(tools),
                    [t.get("name", "unknown") for t in tools if isinstance(t, dict)],
                )

            params["stream"] = stream
            params.update(self._get_generation_params(**kwargs))

            if stream:
                return self._stream_responses_api(params)
            return await self._execute_responses_api(params)

        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderExecutionError(f"Responses API call failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Raw access (for callers that need SDK-level objects)
    # ------------------------------------------------------------------

    async def execute_responses_api_raw(
        self,
        input_message: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute a Responses API call and return the **raw** OpenAI SDK
        stream object.

        .. warning::

            This method deliberately **breaks** the ``LLMProvider``
            abstraction.  It exists solely for higher-level components
            (e.g. ``StreamManager``) that need access to SDK-specific
            event types.  Prefer :meth:`execute_responses_api` for all
            other use-cases.
        """
        params: dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "input": input_message,
            "stream": True,
        }
        if instructions:
            params["instructions"] = instructions
        if previous_response_id:
            params["previous_response_id"] = previous_response_id
        if tools:
            params["tools"] = tools

        params.update(self._get_generation_params(**kwargs))

        logger.debug("Raw Responses API stream requested.")
        return await self._async_client.responses.create(**params)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    # -- parameter builders ---------------------------------------------------

    def _build_chat_params(
        self,
        messages: list[dict[str, str]],
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "stream": stream,
        }
        params.update(self._get_generation_params(**kwargs))
        return params

    def _get_generation_params(self, **kwargs: Any) -> dict[str, Any]:
        """Return a dict of generation parameters, merging config defaults
        with explicit *kwargs*.  Parameters that are unsupported by the
        active API type are automatically excluded."""
        params: dict[str, Any] = {}
        is_responses = self.config.api_type == APIType.RESPONSES

        # Temperature ---------------------------------------------------------
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature is not None and not is_responses:
            params["temperature"] = temperature

        # Max tokens ----------------------------------------------------------
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
        if max_tokens is not None and not is_responses:
            params["max_tokens"] = max_tokens

        # Top-P ---------------------------------------------------------------
        top_p = kwargs.get("top_p", self.config.top_p)
        if top_p is not None and not is_responses:
            params["top_p"] = top_p

        # Chat Completions – only -------------------------------------------------
        if not is_responses:
            for attr in (
                "frequency_penalty",
                "presence_penalty",
                "stop",
            ):
                value = kwargs.get(attr, getattr(self.config, attr, None))
                if value is not None:
                    params[attr] = value

        return params

    # -- Chat Completions execution -------------------------------------------

    async def _execute_chat_completion(self, params: dict[str, Any]) -> LLMResponse:
        response: ChatCompletion = await self._async_client.chat.completions.create(**params)

        if not response.choices or not response.choices[0].message:
            raise LLMProviderExecutionError("Chat completion returned no choices.")

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider_type=self.provider_type,
            api_type=APIType.CHAT_COMPLETION,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            total_tokens=response.usage.total_tokens if response.usage else None,
            finish_reason=choice.finish_reason,
            response_id=response.id,
            metadata={
                "created": response.created,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
        )

    async def _stream_chat_completion(self, params: dict[str, Any]) -> AsyncIterator[str]:
        try:
            stream = await self._async_client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            raise LLMProviderStreamingError(f"Chat completion stream failed: {exc}") from exc

    # -- Responses API execution ----------------------------------------------

    async def _execute_responses_api(self, params: dict[str, Any]) -> LLMResponse:
        params_copy = {k: v for k, v in params.items() if k != "stream"}
        response = await self._async_client.responses.create(**params_copy)

        content = getattr(response, "content", "") or str(response)

        return LLMResponse(
            content=content,
            model=params["model"],
            provider_type=self.provider_type,
            api_type=APIType.RESPONSES,
            response_id=getattr(response, "id", None),
            metadata={"response_object": response},
        )

    async def _stream_responses_api(self, params: dict[str, Any]) -> AsyncIterator[str]:
        try:
            stream = await self._async_client.responses.create(**params)
            async for event in stream:
                # OpenAI streaming responses emit events with different
                # shapes depending on the event type.
                if hasattr(event, "delta") and event.delta:
                    yield event.delta
                elif hasattr(event, "content"):
                    yield event.content
        except Exception as exc:
            raise LLMProviderStreamingError(f"Responses API stream failed: {exc}") from exc
