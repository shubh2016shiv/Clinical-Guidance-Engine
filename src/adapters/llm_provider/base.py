"""Shared settings and reply types for language-model calls.

Think of this file as the "contract" for talking to any language model (LLM).
It defines:
- What settings a request needs (LLMConfig).
- What a finished answer looks like (LLMResponse).
- The rules every model provider (like OpenAI) must follow (LLMProvider).

Big-picture flow:
1. The app builds an ``LLMConfig`` with the model name and request settings.
2. A concrete provider (for example, ``OpenAIProvider`` in the ``openai``
   subfolder) is created with that config.
3. The app calls the provider's ``execute`` or ``execute_chat_completion``
   methods to get an answer.
4. Before sending the request, the provider checks the token budget using
   ``validate_token_limit`` so it does not try to send something too large.
5. The provider returns an ``LLMResponse`` (or a stream of text pieces).
"""

from __future__ import annotations

# "from __future__ import annotations" tells Python to treat type hints
# (like "int | None") as plain text instead of evaluating them immediately.
# This lets us use modern type-hint syntax even on older Python versions,
# and avoids a small amount of work building type objects that only editors
# and type checkers actually look at, not the running program.
from abc import ABC, abstractmethod

# ABC (Abstract Base Class) lets us define a "template" class that cannot be
# created directly - only its subclasses can be. abstractmethod marks a
# method that every subclass MUST implement, or Python raises an error.
from collections.abc import AsyncIterator

# AsyncIterator is a type hint meaning "something you can loop over with
# 'async for', one piece at a time, without loading everything into memory
# first." It is used here for streaming text responses.
from dataclasses import dataclass, field

# @dataclass is a shortcut that auto-generates the boring parts of a class
# (like __init__) just from the list of fields you declare below it. "field"
# lets us give a field a more complex default, such as an empty dictionary.
from enum import Enum

# Enum creates a small, fixed set of named choices (like a multiple-choice
# question) instead of using plain strings, so a typo is caught by your
# editor or type checker instead of causing a silent bug later.
from typing import Any

# "Any" is a type hint meaning "this could be any type; we are not
# restricting it." It is used where the exact type is not known or not
# important to enforce here.
from src.logs import get_logger

logger = get_logger(__name__)
# A logger lets the app print structured messages about what is happening
# (info, warnings, errors) instead of using plain print() statements. Every
# log message below is tagged with this file's module name.


# APIType: a fixed set of choices for which OpenAI API style is being used.
# Needed because Chat Completions and the Responses API expect different
# request shapes, so the rest of the code must know which one is active.
class APIType(Enum):
    """The two OpenAI APIs used by this app."""

    CHAT_COMPLETION = "chat_completion"
    RESPONSES = "responses"


# StreamMode: a fixed set of choices for whether "plain prompt" calls should
# stream their answer piece-by-piece by default. Needed so callers do not
# have to repeat "stream=True/False" every time if they want one behavior.
class StreamMode(Enum):
    """Whether plain-prompt calls stream by default."""

    DISABLED = "disabled"
    ENABLED = "enabled"


# LLMConfig: a plain data container holding all the settings for a model
# request (which model, how creative it should be, token limits, etc.).
# Needed so these settings can be built once and passed around as a single
# object instead of as many separate function arguments.
@dataclass
class LLMConfig:
    """Model settings used by the OpenAI provider and Chat Completions client."""

    model_name: str
    # model_name: which OpenAI model to call, e.g. "gpt-4o".
    api_type: APIType = APIType.CHAT_COMPLETION
    # api_type: which OpenAI API style this config is meant for.
    token_limit: int = 8192
    # token_limit: the hard ceiling for (input + reserved output) tokens.
    temperature: float = 0.7
    # temperature: how random/creative the model's answer should be.
    # Lower is more predictable, higher is more varied.
    max_tokens: int | None = None
    # max_tokens: how many tokens to reserve for the model's reply.
    top_p: float | None = None
    # top_p: an alternative way (besides temperature) to control randomness.
    frequency_penalty: float | None = None
    # frequency_penalty: discourages the model from repeating the same words.
    presence_penalty: float | None = None
    # presence_penalty: discourages the model from repeating the same topics.
    stop: str | list[str] | None = None
    # stop: text that, if generated, tells the model to stop early.
    stream_mode: StreamMode = StreamMode.DISABLED
    # stream_mode: whether plain-prompt calls stream text by default.


# LLMResponse: a plain data container holding a finished answer from the
# model, in one shape the rest of the app can rely on no matter which
# provider produced it. Needed so callers never need to know OpenAI's
# specific reply format.
@dataclass
class LLMResponse:
    """Completed Chat Completions reply used by the drug agent."""

    content: str
    # content: the actual answer text the model generated.
    model: str
    # model: which model actually produced this answer (useful for logging).
    provider_type: str
    # provider_type: which provider class produced this (e.g. "openai").
    api_type: APIType
    # api_type: which API style was used for this specific request.
    prompt_tokens: int | None = None
    # prompt_tokens: how many tokens the input used, if OpenAI reported it.
    completion_tokens: int | None = None
    # completion_tokens: how many tokens the answer used, if reported.
    total_tokens: int | None = None
    # total_tokens: prompt_tokens + completion_tokens, if reported.
    finish_reason: str | None = None
    # finish_reason: why the model stopped generating (e.g. "stop", "length").
    response_id: str | None = None
    # response_id: an ID OpenAI assigns to this specific response.
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata: a catch-all dictionary for any extra details worth keeping.
    # "field(default_factory=dict)" is needed instead of "= {}" because a
    # mutable default like a dict would otherwise be shared between every
    # instance of this class - a classic Python trap this avoids.


# LLMProvider: the "contract" every model provider must follow, plus some
# shared, ready-to-use logic (token estimation and validation) so each
# provider does not need to reimplement it. Needed so the rest of the app
# can call "a provider" without caring whether it is OpenAI, Azure, etc.
class LLMProvider(ABC):
    """Interface and token-budget check shared by model providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # self.config: the settings this provider instance will use for
        # every request it makes, unless a call overrides something.
        self.provider_type = type(self).__name__.removesuffix("Provider").lower()
        # Turns a class name like "OpenAIProvider" into the short label
        # "openai", used for logging and for the LLMResponse.provider_type
        # field above.
        logger.info(
            "Initialised %s provider - model=%s api=%s stream=%s",
            self.provider_type,
            config.model_name,
            config.api_type.value,
            config.stream_mode.value,
        )
        # Logs one line when a provider is created, so you can see in the
        # logs exactly which model/settings are in use without guessing.

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a plain prompt to the configured API."""
        # This method has no body because it is "abstract": it only
        # describes what every subclass must implement, not how to do it.
        # "async def" means this function must be awaited and can pause
        # while waiting on network I/O without freezing the whole program.
        # The "*" marks everything after it as keyword-only (callers must
        # write "stream=True" by name, not pass it positionally).
        # "**kwargs" collects any extra named arguments the caller passes,
        # so this method stays flexible without listing every possible option.

    @abstractmethod
    async def execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Send chat messages and return an app reply or text stream."""
        # Another abstract method. Its return type hint says it gives back
        # EITHER a finished LLMResponse OR a stream you read piece-by-piece,
        # depending on whether "stream" was True.

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate the input size; use a rough fallback if tokenization fails."""
        # "@staticmethod" means this method does not need "self" - it does
        # not use anything specific to a particular provider instance, so it
        # can be called directly on the class if needed.
        try:
            import tiktoken
            # tiktoken is OpenAI's library for counting tokens the same way
            # the model does. It is imported here (inside the function)
            # instead of at the top of the file so this whole module still
            # works even if tiktoken is not installed.

            return len(tiktoken.get_encoding("cl100k_base").encode(text))
            # Step 1: encode the text into tokens using the same rules the
            # model uses, then count how many there are.
        except Exception:
            # Step 2 (fallback): if tiktoken is missing or fails for any
            # reason, fall back to a rough guess of about 4 characters per
            # token. This is not exact, but keeps the app working.
            return max(1, len(text) // 4)

    def validate_token_limit(
        self,
        prompt: str,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> bool:
        """Check estimated input and reserved output against the configured limit."""
        # Step 1: estimate how many tokens the main prompt will use.
        input_tokens = self.estimate_tokens(prompt)
        # input_tokens: running total of tokens the request's text will use.
        if system_prompt:
            # Step 2: if there is also a system prompt (instructions for the
            # model's behavior), add its tokens to the same running total.
            input_tokens += self.estimate_tokens(system_prompt)

        # Step 3: figure out how many tokens must be reserved for the
        # model's reply - either the amount passed in for this one call, or
        # the config's default, or zero if neither is set.
        output_tokens = max_tokens if max_tokens is not None else self.config.max_tokens or 0
        # output_tokens: tokens being reserved for the model's answer.
        total_tokens = input_tokens + output_tokens
        # total_tokens: the full token budget this request would use.
        if total_tokens > self.config.token_limit:
            # Step 4: if the budget is too big, log a warning with the exact
            # numbers (useful for debugging) and tell the caller "no".
            logger.warning(
                "Token limit exceeded: %d > %d (input=%d, response=%d)",
                total_tokens,
                self.config.token_limit,
                input_tokens,
                output_tokens,
            )
            return False
        # Step 5: otherwise, the request fits within budget - say "yes".
        return True

    def should_stream(self) -> bool:
        """Return whether plain-prompt calls stream by default."""
        return self.config.stream_mode == StreamMode.ENABLED
