"""
Custom exceptions for the OpenAI Responses API integration.

This module defines a hierarchy of custom exceptions for better error handling
when working with the OpenAI Responses API and related components.
"""


class OpenAIResponsesError(Exception):
    """Base exception for OpenAI Responses API errors."""

    def __init__(self, message="An error occurred with the OpenAI Responses API"):
        self.message = message
        super().__init__(self.message)


class ResponsesAPIError(OpenAIResponsesError):
    """Exception raised for errors returned by the OpenAI Responses API."""

    def __init__(
        self,
        status_code=None,
        error_type=None,
        message="OpenAI Responses API returned an error",
    ):
        self.status_code = status_code
        self.error_type = error_type
        error_info = (
            f" (Status: {status_code}, Type: {error_type})" if status_code else ""
        )
        super().__init__(f"{message}{error_info}")


class StreamConnectionError(OpenAIResponsesError):
    """Exception raised for errors in streaming connections."""

    def __init__(self, message="Failed to establish or maintain streaming connection"):
        super().__init__(message)


class ContentParsingError(OpenAIResponsesError):
    """Exception raised when parsing response content fails."""

    def __init__(self, message="Failed to parse response content"):
        super().__init__(message)


class VectorStoreError(OpenAIResponsesError):
    """Exception raised for errors related to vector stores."""

    def __init__(self, message="An error occurred with the vector store"):
        super().__init__(message)


class ToolConfigurationError(OpenAIResponsesError):
    """Exception raised for errors in tool configuration."""

    def __init__(self, message="Failed to configure tool"):
        super().__init__(message)
