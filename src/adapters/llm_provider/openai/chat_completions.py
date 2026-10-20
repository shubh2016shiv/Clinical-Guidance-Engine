"""Chat Completions client for OpenAI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ..base import APIType, LLMConfig, LLMResponse
from ..exceptions import (
    LLMProviderError,
    LLMProviderExecutionError,
    LLMProviderStreamingError,
    LLMProviderTokenLimitError,
)

if TYPE_CHECKING:
    from .provider import OpenAIProvider


class ChatCompletionsClient:
    """Call OpenAI Chat Completions API."""

    def __init__(self, config: LLMConfig, provider: OpenAIProvider) -> None:
        self._config = config
        self._provider = provider

    async def execute(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Execute Chat Completions request.

        Validates token budget, builds request, and calls SDK.
        Returns response object or text stream.
        """
        try:
            total_prompt = " ".join(msg.get("content", "") for msg in messages)
            if not self._provider.validate_token_limit(total_prompt, kwargs.get("max_tokens")):
                raise LLMProviderTokenLimitError("Token budget exceeded for chat completion.")

            model = kwargs.get("model", self._config.model_name)
            params: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}

            option_names = (
                "temperature",
                "max_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
            )
            for name in option_names:
                value = kwargs.get(name, getattr(self._config, name, None))
                if value is not None:
                    params[name] = value

            if stream:
                return self._stream(params)

            response = await self._provider._async_client.chat.completions.create(**params)
            if not response.choices or not response.choices[0].message:
                raise LLMProviderExecutionError("Chat completion returned no choices.")

            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                provider_type=self._provider.provider_type,
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

        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderExecutionError(f"Chat completion failed: {exc}") from exc

    async def _stream(self, params: dict[str, Any]) -> AsyncIterator[str]:
        try:
            stream = await self._provider._async_client.chat.completions.create(**params)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            raise LLMProviderStreamingError(f"Chat completion stream failed: {exc}") from exc
