"""Send Chat Completions requests and return the reply shape used by the app."""

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
from .error_mapping import map_openai_error

if TYPE_CHECKING:
    # TYPE_CHECKING is always False while the program actually runs, but
    # type checkers (like mypy or your editor) pretend it is True. This lets
    # us import OpenAIProvider ONLY for type hints below, without creating a
    # real circular import at runtime (provider.py also imports this file).
    from .provider import OpenAIProvider


# ChatCompletionsClient: knows how to turn this app's chat messages into an
# actual OpenAI Chat Completions request, and how to turn OpenAI's reply
# back into this app's own LLMResponse shape. Needed so OpenAIProvider does
# not have to know these formatting details itself.
class ChatCompletionsClient:
    """Call the SDK and convert completed replies to ``LLMResponse``."""

    def __init__(self, llm_config: LLMConfig, openai_provider: OpenAIProvider) -> None:
        self._llm_config = llm_config
        # self._llm_config: the default settings (temperature, max_tokens...)
        # to use unless a specific call overrides them.
        self._openai_provider = openai_provider
        # self._openai_provider: a reference back to the provider that owns
        # the actual network connection, so this client can use it.

    async def create_chat_completion(
        self,
        chat_messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **chat_completion_options: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Return a completed app reply or an async stream of text."""
        try:
            # Step 1: estimate whether this request fits the token budget
            # BEFORE spending time sending it to OpenAI. We roughly combine
            # all message contents into one string for the estimate.
            prompt_text = " ".join(message.get("content", "") for message in chat_messages)
            # prompt_text: a rough, combined version of everything being
            # sent to the model, used only for the token-size estimate.
            # The reply-length limit can be passed under either name (see the rename
            # below), so the budget check looks for both.
            per_call_token_limit = chat_completion_options.get(
                "max_completion_tokens", chat_completion_options.get("max_tokens")
            )
            if not self._openai_provider.validate_token_limit(prompt_text, per_call_token_limit):
                # Step 2: if the estimate is over budget, stop here with a
                # specific error instead of letting OpenAI reject it later.
                raise LLMProviderTokenLimitError("Token budget exceeded for chat completion.")

            # Step 3: work out the final request options. Start from this
            # client's configured defaults (skipping anything that is None),
            # then let whatever the CALLER passed for this one request win.
            # "getattr(obj, name)" reads the attribute named by the string
            # "name" dynamically - like writing "obj.temperature" but looped
            # over a list of names instead of repeating code for each one.
            # "value := ..." is the "walrus operator": it assigns to
            # "value" AND lets the surrounding "if" test that same value,
            # without computing it twice.
            # Use configured defaults, then let this request override them.
            default_options = {
                name: value
                for name in (
                    "temperature",
                    "max_tokens",
                    "top_p",
                    "frequency_penalty",
                    "presence_penalty",
                    "stop",
                )
                if (value := getattr(self._llm_config, name)) is not None
            }
            # default_options: this client's non-empty configured defaults.
            # "{**a, **b}" merges two dicts into a new one; matching keys
            # from "b" (the per-call options) win over keys from "a" (the
            # defaults).
            request_options = {
                name: value
                for name, value in {**default_options, **chat_completion_options}.items()
                if value is not None
            }
            # request_options: the final merged settings for this request.
            # The OpenAI SDK deprecates "max_tokens" in favour of
            # "max_completion_tokens", and newer models reject the old name
            # outright. LLMConfig and callers keep calling it max_tokens; it is
            # renamed here, at the point it goes to OpenAI. A limit the caller
            # passed under the new name wins over the configured default, because
            # "setdefault" only fills the key when it is not already there.
            if "max_tokens" in request_options:
                request_options.setdefault(
                    "max_completion_tokens", request_options.pop("max_tokens")
                )
            request_parameters: dict[str, Any] = {
                **request_options,
                "model": chat_completion_options.get("model", self._llm_config.model_name),
                "messages": chat_messages,
                "stream": stream,
            }
            # request_parameters: everything that will actually be sent to
            # the OpenAI SDK's chat.completions.create(...) call.

            if stream:
                # Step 4a: if streaming was requested, hand off to the
                # streaming helper below instead of waiting for one big
                # reply. This returns an async generator OBJECT - the
                # actual network call has not started yet at this point.
                return self._stream_chat_completion_text(request_parameters)

            # Step 4b: non-streaming path - "await" pauses this function
            # until OpenAI's full reply comes back.
            chat_completion = (
                await self._openai_provider._async_openai_client.chat.completions.create(
                    **request_parameters
                )
            )
            if not chat_completion.choices or not chat_completion.choices[0].message:
                # Step 5: defend against an unexpected empty reply - better
                # to fail clearly here than let a confusing error happen
                # later when code tries to read missing data.
                raise LLMProviderExecutionError("Chat completion returned no choices.")

            # Step 6: pull out the first (and normally only) answer choice
            # and the token-usage numbers, if OpenAI included them.
            first_choice = chat_completion.choices[0]
            # first_choice: the model's single generated answer and why it
            # stopped generating.
            usage = chat_completion.usage
            # usage: how many tokens were used, or None if not reported.
            # Step 7: translate OpenAI's reply shape into this app's own
            # LLMResponse shape, so callers never need to know OpenAI's
            # specific field names. ("getattr(obj, name, default)" below
            # reads an attribute only if it exists, otherwise uses the
            # default - useful for optional/newer SDK fields.)
            return LLMResponse(
                content=first_choice.message.content or "",
                model=chat_completion.model,
                provider_type=self._openai_provider.provider_type,
                api_type=APIType.CHAT_COMPLETION,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
                finish_reason=first_choice.finish_reason,
                response_id=chat_completion.id,
                metadata={
                    "created": chat_completion.created,
                    "system_fingerprint": getattr(chat_completion, "system_fingerprint", None),
                },
            )

        except LLMProviderError:
            # If the code above already raised one of OUR specific errors
            # (like the token-limit check), let it pass through unchanged -
            # we do not want to accidentally re-wrap it below.
            raise
        except Exception as exc:
            # Any OTHER kind of failure (network error, OpenAI SDK error,
            # etc.) gets translated into one of our own exception types by
            # map_openai_error, so callers only ever need to catch our
            # exceptions, never OpenAI's. "from exc" keeps the original
            # error attached as the "cause", which shows up in tracebacks
            # for debugging without hiding what really happened.
            raise map_openai_error(exc, "Chat completion failed") from exc

    async def _stream_chat_completion_text(
        self, request_parameters: dict[str, Any]
    ) -> AsyncIterator[str]:
        """Yield the text in each SDK streaming chunk."""
        # This function contains a "yield" below, which makes it an async
        # GENERATOR rather than a normal function: calling it does not run
        # the code immediately. Instead it hands back an object that moves
        # one small step forward each time something does "async for ... in
        # it", pausing in between. This is how text can be delivered to the
        # caller piece by piece as it arrives, instead of all at once.
        try:
            # Step 1: start the request. This is the moment the network call
            # actually begins - OpenAI starts sending back a stream of small
            # "chunks" instead of one big reply.
            completion_stream = (
                await self._openai_provider._async_openai_client.chat.completions.create(
                    **request_parameters
                )
            )
        except Exception as exc:
            # If the request fails to even START (for example, bad API key,
            # or rate limited), map it the same way as the non-streaming
            # path so callers get a consistent, specific exception type.
            raise map_openai_error(exc, "Chat completion stream failed to start") from exc

        try:
            # Step 2: read each small chunk as it arrives and yield out just
            # the new text piece, skipping chunks that have no text (like
            # the final "end of stream" signal).
            async for completion_chunk in completion_stream:
                if completion_chunk.choices and completion_chunk.choices[0].delta.content:
                    yield completion_chunk.choices[0].delta.content
        except Exception as exc:
            # If the connection breaks AFTER streaming already started (so
            # the caller may have already seen some text), raise a
            # streaming-specific error rather than the general mapping
            # above, so callers can tell "failed before anything arrived"
            # apart from "failed partway through".
            raise LLMProviderStreamingError(f"Chat completion stream failed: {exc}") from exc
