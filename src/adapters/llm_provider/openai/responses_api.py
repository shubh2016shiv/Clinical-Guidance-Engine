"""Responses API client for OpenAI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from src.logs import get_logger

from ..base import APIType, LLMConfig, LLMResponse
from ..exceptions import (
    LLMProviderError,
    LLMProviderExecutionError,
    LLMProviderStreamingError,
    LLMProviderTokenLimitError,
)

if TYPE_CHECKING:
    from .provider import OpenAIProvider

logger = get_logger(__name__)


class ResponsesApiClient:
    """Call OpenAI Responses API."""

    def __init__(self, config: LLMConfig, provider: OpenAIProvider) -> None:
        self._config = config
        self._provider = provider

    async def execute(
        self,
        input_message: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Execute Responses API request with token validation.

        Validates token budget, builds request, calls SDK.
        Returns response object or text stream.
        """
        try:
            full_prompt = f"{instructions or ''}\n{input_message}"
            if not self._provider.validate_token_limit(full_prompt, kwargs.get("max_tokens")):
                raise LLMProviderTokenLimitError("Token budget exceeded for Responses API call.")

            params = self._build_params(
                input_message, instructions, previous_response_id, tools, stream, **kwargs
            )

            if tools:
                logger.debug(
                    "Responses API call includes %d tool(s): %s",
                    len(tools),
                    [tool.get("name", "unknown") for tool in tools if isinstance(tool, dict)],
                )

            if stream:
                return self._stream(params)

            request_params = {k: v for k, v in params.items() if k != "stream"}
            response = await self._provider._async_client.responses.create(**request_params)
            content = getattr(response, "output_text", "")

            return LLMResponse(
                content=content,
                model=params["model"],
                provider_type=self._provider.provider_type,
                api_type=APIType.RESPONSES,
                response_id=getattr(response, "id", None),
                metadata={"response_object": response},
            )

        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderExecutionError(f"Responses API call failed: {exc}") from exc

    async def execute_raw(
        self,
        input_message: str | list[dict[str, Any]],
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Return raw SDK object without token validation or normalization."""
        params = self._build_params(
            input_message, instructions, previous_response_id, tools, stream, **kwargs
        )
        logger.debug("Raw Responses API call requested (stream=%s).", stream)
        return await self._provider._async_client.responses.create(**params)

    def _build_params(
        self,
        input_message: str | list[dict[str, Any]],
        instructions: str | None,
        previous_response_id: str | None,
        tools: list[dict[str, Any]] | None,
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": kwargs.get("model", self._config.model_name),
            "input": input_message,
            "stream": stream,
        }
        if instructions:
            params["instructions"] = instructions
        if previous_response_id:
            params["previous_response_id"] = previous_response_id
        if tools:
            params["tools"] = tools
        for name in ("store", "include", "max_output_tokens", "tool_choice"):
            if name in kwargs:
                params[name] = kwargs[name]
        return params

    async def _stream(self, params: dict[str, Any]) -> AsyncIterator[str]:
        try:
            stream = await self._provider._async_client.responses.create(**params)
            async for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type in {"response.failed", "response.incomplete", "error"}:
                    raise LLMProviderStreamingError(f"Responses API stream ended: {event.type}")
        except Exception as exc:
            raise LLMProviderStreamingError(f"Responses API stream failed: {exc}") from exc
