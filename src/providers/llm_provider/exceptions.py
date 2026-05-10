"""
LLM Provider Exception Hierarchy.

All exceptions inherit from the shared ``ProviderError`` base class
(``src.providers.exceptions``) so that a single ``except ProviderError``
can catch failures from *any* provider subsystem.
"""

from __future__ import annotations

from src.providers.exceptions import ProviderError


class LLMProviderError(ProviderError):
    """Base exception for LLM provider errors.

    Every LLM-provider-specific exception inherits from this class so
    callers can handle all LLM failures with a single ``except`` clause.
    """


class LLMProviderInitializationError(LLMProviderError):
    """Raised when provider initialisation fails.

    Common causes:
    - Missing or invalid API keys.
    - Unreachable API endpoint.
    - Invalid configuration values.
    """


class LLMProviderExecutionError(LLMProviderError):
    """Raised when an LLM API call fails at runtime.

    Common causes:
    - API returned an error response (4xx / 5xx).
    - Request timed out.
    - Malformed request parameters.
    """


class LLMProviderTokenLimitError(LLMProviderError):
    """Raised when the prompt + expected response exceeds the token budget.

    Callers can catch this specifically to implement truncation or
    fall-back-to-shorter-model strategies.
    """


class LLMProviderStreamingError(LLMProviderError):
    """Raised when a streaming response fails part-way through.

    Common causes:
    - Connection dropped mid-stream.
    - Malformed server-sent event.
    """


class LLMProviderAuthenticationError(LLMProviderError):
    """Raised when the provider rejects the API key or credentials."""


class LLMProviderRateLimitError(LLMProviderError):
    """Raised when the provider's rate limit is exceeded.

    Callers can implement exponential-backoff retry logic specifically
    for this exception type.
    """


class LLMProviderUnsupportedOperationError(LLMProviderError):
    """Raised when the configured provider / API type does not support
    the requested operation (e.g. streaming on a provider that only
    offers batch completions).
    """
