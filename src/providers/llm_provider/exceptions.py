class LLMProviderError(Exception):
    """
    Base exception for LLM provider errors

    All provider-specific exceptions should inherit from this class
    to allow for unified error handling.
    """

    pass


class LLMProviderInitializationError(LLMProviderError):
    """
    Raised when provider initialization fails

    Examples: Missing API keys, invalid configuration, network issues
    """

    pass


class LLMProviderExecutionError(LLMProviderError):
    """
    Raised when LLM execution fails

    Examples: API errors, timeout, rate limiting, invalid parameters
    """

    pass


class LLMProviderTokenLimitError(LLMProviderError):
    """
    Raised when token limits are exceeded

    This helps distinguish token limit issues from other execution errors.
    """

    pass


class LLMProviderStreamingError(LLMProviderError):
    """
    Raised when streaming operations fail

    Examples: Connection drops, incomplete streams, parsing errors
    """

    pass


class LLMProviderAuthenticationError(LLMProviderError):
    """
    Raised when authentication fails

    Examples: Invalid API key, expired credentials
    """

    pass


class LLMProviderRateLimitError(LLMProviderError):
    """
    Raised when rate limits are hit

    This allows for specific retry logic for rate limiting scenarios.
    """

    pass


class LLMProviderUnsupportedOperationError(LLMProviderError):
    """
    Raised when an unsupported operation is attempted

    Examples: Streaming not supported by provider, invalid API type
    """

    pass
