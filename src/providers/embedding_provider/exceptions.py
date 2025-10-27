"""
Embedding Provider Exceptions

This module defines custom exceptions for embedding provider operations,
allowing for better error handling and debugging.
"""


class EmbeddingProviderError(Exception):
    """
    Base exception for embedding provider errors

    All embedding-specific exceptions should inherit from this class
    to allow for unified error handling.

    Attributes:
        message: Error message
        details: Optional dictionary with additional error context
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception

        Args:
            message: Error message
            details: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        """String representation of the error"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class EmbeddingProviderInitializationError(EmbeddingProviderError):
    """
    Raised when embedding provider initialization fails

    Examples: Missing API keys, invalid configuration, unsupported models
    """

    pass


class EmbeddingGenerationError(EmbeddingProviderError):
    """
    Raised when embedding generation fails

    Examples: API errors, timeout, invalid input text
    """

    pass


class EmbeddingDimensionMismatchError(EmbeddingProviderError):
    """
    Raised when embedding dimensions don't match expectations

    This helps catch configuration issues early.
    """

    pass


class EmbeddingBatchSizeError(EmbeddingProviderError):
    """
    Raised when batch size limits are exceeded

    Different providers have different batch size limits.
    """

    pass


class EmbeddingTextPreprocessingError(EmbeddingProviderError):
    """
    Raised when text preprocessing fails

    Examples: Empty text, invalid characters, length exceeded
    """

    pass


class EmbeddingAuthenticationError(EmbeddingProviderError):
    """
    Raised when authentication fails

    Examples: Invalid API key, expired credentials
    """

    pass


class EmbeddingRateLimitError(EmbeddingProviderError):
    """
    Raised when rate limits are hit

    This allows for specific retry logic for rate limiting scenarios.
    """

    pass
