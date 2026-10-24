"""
Vector Database Provider Exceptions

This module defines custom exceptions for vector database provider operations,
allowing for better error handling and debugging.

Big-picture flow:
1. Give every vector-database failure its own specific exception class.
2. Let callers catch broadly (``VectorDBProviderError``) or narrowly (e.g.
   just ``VectorDBDimensionMismatchError``) depending on how precisely they
   need to react.
3. Keep optional extra details (like the expected vs. actual dimension)
   attached to the exception itself, instead of only in a plain text message.

Beginner note: lines indented under a class are a "family tree" of
exception classes. "class Foo(Bar):" means "Foo IS-A Bar, plus maybe
something extra" - so catching the parent class also catches every more
specific class built on top of it below.
"""

from src.adapters.exceptions import ProviderError


# VectorDBProviderError: the shared parent for every vector-database
# exception in this adapter. Needed so callers can write one "except
# VectorDBProviderError" to handle ANY vector-database failure, instead of
# listing every specific exception type.
class VectorDBProviderError(ProviderError):
    """
    Base exception for vector database provider errors

    All vector database-specific exceptions should inherit from this class
    to allow for unified error handling.

    Attributes:
        message: Error message
        details: Optional dictionary with additional error context
    """

    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize the exception

        Args:
            message: Error message
            details: Optional dictionary with additional context
        """
        # "__init__" is the special method Python calls automatically when
        # a new instance of this class is created.
        super().__init__(message, details)
        # "super().__init__(...)" calls the PARENT class's own __init__, so
        # whatever setup ProviderError normally does still happens here too.
        self.message = message
        # self.message: the plain-text description of what went wrong.
        self.details = details or {}
        # self.details: extra structured context about the failure.
        # "details or {}" means: use the dict passed in, or an empty dict
        # if None was passed instead.

    def __str__(self):
        """String representation of the error"""
        # "__str__" is the special method Python calls when this exception
        # is converted to text, e.g. by print(exc) or str(exc).
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            # details_str: the extra context formatted as "key=value,
            # key2=value2" so it reads naturally in a log line.
            return f"{self.message} ({details_str})"
        return self.message


# VectorDBConnectionError: specifically means "could not reach or stay
# connected to the vector database" - narrower than the general error
# above, so callers can react differently (e.g. retry) to connection problems.
class VectorDBConnectionError(VectorDBProviderError):
    """
    Raised when vector database connection fails

    Examples: Connection timeout, authentication failure, network errors
    """

    pass
    # "pass" means "this class has no extra code of its own" - it exists
    # purely to give this kind of failure its own specific, catchable name.


# VectorDBSearchError: a search request itself failed (as opposed to a
# connection or configuration problem).
class VectorDBSearchError(VectorDBProviderError):
    """
    Raised when vector database search operation fails

    Examples: Search timeout, invalid query vector, index errors
    """

    pass


# VectorDBConfigError: the settings given to the vector database provider
# are invalid, independent of whether the server itself is reachable.
class VectorDBConfigError(VectorDBProviderError):
    """
    Raised when vector database configuration is invalid

    Examples: Missing required settings, invalid parameter values, dimension mismatches
    """

    pass


# VectorDBDimensionMismatchError: a MORE specific kind of configuration
# error - notice it inherits from VectorDBConfigError, not directly from
# VectorDBProviderError, so catching "VectorDBConfigError" also catches this.
class VectorDBDimensionMismatchError(VectorDBConfigError):
    """
    Raised when embedding dimensions don't match expectations

    This helps catch configuration issues early.
    The query vector dimension must match the collection's embedding dimension.
    """

    pass


# VectorDBCollectionError: something about the collection itself (Milvus's
# equivalent of a database table) is wrong.
class VectorDBCollectionError(VectorDBProviderError):
    """
    Raised when collection operations fail

    Examples: Collection doesn't exist, collection not loaded, schema mismatches
    """

    pass


# VectorDBAuthenticationError: a MORE specific kind of connection error -
# it inherits from VectorDBConnectionError, so catching that parent class
# also catches this.
class VectorDBAuthenticationError(VectorDBConnectionError):
    """
    Raised when authentication fails

    Examples: Invalid credentials, expired tokens, permission denied
    """

    pass


# VectorDBRateLimitError: too many requests were sent too quickly.
class VectorDBRateLimitError(VectorDBProviderError):
    """
    Raised when rate limits are hit

    This allows for specific retry logic for rate limiting scenarios.
    """

    pass


# VectorDBIndexError: something about the search index (the data structure
# that makes similarity search fast) is wrong.
class VectorDBIndexError(VectorDBProviderError):
    """
    Raised when index operations fail

    Examples: Index doesn't exist, index creation fails, index corruption
    """

    pass
