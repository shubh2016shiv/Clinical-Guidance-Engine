"""Send Responses API requests and return OpenAI's result unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..base import LLMConfig
from .error_mapping import map_openai_error

if TYPE_CHECKING:
    # See chat_completions.py for why this import only happens for type
    # checkers, not while the program is actually running.
    from .provider import OpenAIProvider


# ResponsesApiClient: knows how to build a Responses API request from this
# app's arguments, and send it. Unlike ChatCompletionsClient, it does NOT
# reshape OpenAI's reply - it hands back OpenAI's own response object as-is,
# because the guideline agent needs OpenAI-specific details like citations.
class ResponsesApiClient:
    """Keep Responses API request fields in one place."""

    def __init__(self, config: LLMConfig, provider: OpenAIProvider) -> None:
        self._config = config
        # self._config: default model name and settings for this API.
        self._provider = provider
        # self._provider: reference back to the provider that owns the
        # actual network connection to OpenAI.

    async def create_response(
        self,
        input_message: str | list[dict[str, Any]],
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **request_options: Any,
    ) -> Any:
        """Return the SDK response, or its event stream when streaming."""
        # Step 1: start the request dictionary with the fields every
        # Responses API call needs. "request_options.pop("model", default)"
        # lets a caller override the model for just this one call, while
        # removing "model" from request_options so it is not duplicated
        # when the rest of request_options is merged in below. Any other
        # options the caller passed (like "store=True") get merged straight
        # into the request dictionary via "**request_options".
        request = {
            "model": request_options.pop("model", self._config.model_name),
            "input": input_message,
            "stream": stream,
            **request_options,
        }
        # Step 2: only add these fields if they were actually provided -
        # OpenAI's API treats a missing field differently from an explicit
        # empty one, so we avoid sending fields the caller did not ask for.
        if instructions is not None:
            request["instructions"] = instructions
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id
        if tools is not None:
            request["tools"] = tools

        try:
            # Step 3: send the request. When "stream=True", the SDK returns
            # an object you loop over for events instead of one finished
            # reply - either way, we return exactly what the SDK gives us.
            return await self._provider._async_openai_client.responses.create(**request)
        except Exception as exc:
            # Translate any OpenAI SDK failure into our own exception
            # hierarchy, the same way chat_completions.py does, so callers
            # can catch "LLMProviderError" regardless of which API failed.
            raise map_openai_error(exc, "Responses API request failed") from exc
