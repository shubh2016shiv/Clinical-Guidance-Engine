"""Map OpenAI SDK exceptions to this adapter's error hierarchy.

Chat Completions, Responses, and Embeddings all call the same SDK, so they
share this mapping to guarantee ``except LLMProviderError`` catches every
failure the same way regardless of which API raised it.
"""

from __future__ import annotations

import openai

# The "openai" package itself (not just AsyncOpenAI) defines the specific
# exception classes it can raise, like openai.AuthenticationError and
# openai.RateLimitError. We import the package directly so we can check
# "isinstance(exc, openai.AuthenticationError)" below.
from ..exceptions import (
    LLMProviderAuthenticationError,
    LLMProviderError,
    LLMProviderExecutionError,
    LLMProviderRateLimitError,
)


# map_openai_error: a small translator function. Needed so every place that
# calls the OpenAI SDK can turn WHATEVER error the SDK raised into one of
# THIS app's own exception types with one line of code, instead of
# repeating the same if/elif checks in three different files.
def map_openai_error(exc: Exception, message: str) -> LLMProviderError:
    """Return the adapter exception matching ``exc``, prefixed with ``message``."""
    # "isinstance(exc, SomeClass)" checks "is this error specifically a
    # SomeClass (or a subclass of it)?" - more precise than checking the
    # error's text message, which could change between SDK versions.
    if isinstance(exc, openai.AuthenticationError):
        # Step 1: the API key was missing, invalid, or rejected.
        return LLMProviderAuthenticationError(f"{message}: {exc}")
    if isinstance(exc, openai.RateLimitError):
        # Step 2: too many requests were sent too quickly.
        return LLMProviderRateLimitError(f"{message}: {exc}")
    # Step 3: anything else (timeouts, 5xx errors, bad parameters, etc.)
    # becomes the general execution-failure exception.
    return LLMProviderExecutionError(f"{message}: {exc}")
    # Note: this function BUILDS the exception and returns it; it does not
    # raise it itself. Callers are expected to write
    # "raise map_openai_error(exc, ...) from exc" so the original error
    # stays attached as the cause, visible in tracebacks for debugging.
