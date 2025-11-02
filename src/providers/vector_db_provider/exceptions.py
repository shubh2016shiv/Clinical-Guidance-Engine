"""
Vector Database Provider Exceptions

This module defines custom exceptions for vector database provider operations,
allowing for better error handling and debugging.
"""

from src.providers.exceptions import ProviderError


class VectorDBProviderError(ProviderError):
    """
    Base exception for vector database provider errors

    All vector database-specific exceptions should inherit from this class
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
        super().__init__(message, details)
        self.message = message
        self.details = details or {}

    def __str__(self):
        """String representation of the error"""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class VectorDBConnectionError(VectorDBProviderError):
    """
    Raised when vector database connection fails

    Examples: Connection timeout, authentication failure, network errors
    """

    pass


class VectorDBSearchError(VectorDBProviderError):
    """
    Raised when vector database search operation fails

    Examples: Search timeout, invalid query vector, index errors
    """

    pass


class VectorDBConfigError(VectorDBProviderError):
    """
    Raised when vector database configuration is invalid

    Examples: Missing required settings, invalid parameter values, dimension mismatches
    """

    pass


class VectorDBDimensionMismatchError(VectorDBConfigError):
    """
    Raised when embedding dimensions don't match expectations

    This helps catch configuration issues early.
    The query vector dimension must match the collection's embedding dimension.
    """

    pass


class VectorDBCollectionError(VectorDBProviderError):
    """
    Raised when collection operations fail

    Examples: Collection doesn't exist, collection not loaded, schema mismatches
    """

    pass


class VectorDBAuthenticationError(VectorDBConnectionError):
    """
    Raised when authentication fails

    Examples: Invalid credentials, expired tokens, permission denied
    """

    pass


class VectorDBRateLimitError(VectorDBProviderError):
    """
    Raised when rate limits are hit

    This allows for specific retry logic for rate limiting scenarios.
    """

    pass


class VectorDBIndexError(VectorDBProviderError):
    """
    Raised when index operations fail

    Examples: Index doesn't exist, index creation fails, index corruption
    """

    pass
