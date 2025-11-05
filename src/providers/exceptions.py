class ProviderError(Exception):
    """Base exception for provider errors"""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize provider error

        Args:
            message: Error message
            details: Optional dictionary with error details
        """
        super().__init__(message)
        self.details = details or {}


class LLMProviderError(ProviderError):
    """Exception raised by LLM providers"""

    pass


class EmbeddingProviderError(ProviderError):
    """Exception raised by embedding providers"""

    pass


class CacheProviderError(ProviderError):
    """Exception raised by cache providers"""

    pass
