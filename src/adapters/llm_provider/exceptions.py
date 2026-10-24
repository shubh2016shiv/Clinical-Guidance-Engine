"""Give model failures names that callers can handle consistently.

These classes identify what went wrong without changing how errors are raised.
1. Start with the shared adapter error type.
2. Group all language-model errors under one class.
3. Use narrower classes for requests, limits, streams, and credentials.

Beginner note: "class Foo(Bar):" means "Foo IS-A Bar, plus maybe some extra
behavior." Here every exception class is built on top of a more general one,
like a family tree. This means code that catches the general class (for
example ``except LLMProviderError``) also catches every more specific class
below it, without needing to list them all one by one.
"""

from __future__ import annotations

# See base.py for why this import is used - it lets this file use modern
# type-hint syntax without extra runtime cost.
# ProviderError is the common parent for errors from all adapter types.
from src.adapters.exceptions import ProviderError


# Catch this parent class when the caller handles all model failures alike.
class LLMProviderError(ProviderError):
    """Base exception for LLM provider errors.

    Every LLM-provider-specific exception inherits from this class so
    callers can handle all LLM failures with a single ``except`` clause.
    """

    # This class adds no code of its own - it exists purely to give every
    # other exception below it a shared "family name" to catch.


# This name lets startup code distinguish setup failures from request failures.
class LLMProviderInitializationError(LLMProviderError):
    """Raised when provider initialisation fails.

    Common causes:
    - Missing or invalid API keys.
    - Unreachable API endpoint.
    - Invalid configuration values.
    """

    # Raised BEFORE any request is sent - something was wrong with how the
    # provider itself was set up (see openai/provider.py and embeddings.py).


# This error tells callers that a request failed while it was being made.
class LLMProviderExecutionError(LLMProviderError):
    """Raised when an LLM API call fails at runtime.

    Common causes:
    - API returned an error response (4xx / 5xx).
    - Request timed out.
    - Malformed request parameters.
    """

    # The "catch-all" for request failures that are not specifically an
    # authentication problem or a rate limit (see openai/error_mapping.py).


# Callers can shorten a request when its token budget is too large.
class LLMProviderTokenLimitError(LLMProviderError):
    """Raised when the prompt + expected response exceeds the token budget.

    Callers can catch this specifically to implement truncation or
    fall-back-to-shorter-model strategies.
    """

    # Raised BEFORE the request is even sent to OpenAI - see
    # LLMProvider.validate_token_limit in base.py.


# A stream may fail after some text has already reached the caller.
class LLMProviderStreamingError(LLMProviderError):
    """Raised when a streaming response fails part-way through.

    Common causes:
    - Connection dropped mid-stream.
    - Malformed server-sent event.
    """

    # "Streaming" means the answer arrives in small pieces over time instead
    # of all at once. This error means some pieces may already have reached
    # the user before the failure happened.


# This category describes rejected credentials so callers can report them clearly.
class LLMProviderAuthenticationError(LLMProviderError):
    """Raised when the provider rejects the API key or credentials."""

    # Usually means OPENAI_API_KEY is missing, wrong, or expired.


# This category helps callers recognize when requests are being sent too often.
class LLMProviderRateLimitError(LLMProviderError):
    """Raised when the provider's rate limit is exceeded.

    Callers can implement exponential-backoff retry logic specifically
    for this exception type.
    """

    # "Exponential backoff" means waiting longer and longer between retries
    # (e.g. 1s, 2s, 4s...) instead of retrying instantly, to give the API
    # time to recover.


# This tells callers an API type cannot perform the requested operation.
class LLMProviderUnsupportedOperationError(LLMProviderError):
    """Raised when the configured provider / API type does not support
    the requested operation (e.g. streaming on a provider that only
    offers batch completions).
    """

    # NOTE: this exception is defined but not currently raised anywhere in
    # this codebase. It is reserved for a future provider that cannot
    # support something another provider offers (see README.md).
