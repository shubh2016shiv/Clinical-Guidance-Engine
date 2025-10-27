"""
Abstract LLM Provider Interface

This module provides an abstract interface for LLM providers,
allowing easy extension to support different AI models and APIs.

The interface supports:
- Multiple API types (Chat Completions, Responses, etc.)
- Streaming and non-streaming responses
- Token validation and estimation
- Flexible configuration management
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from enum import Enum
from dataclasses import dataclass, field

from src.logs import get_logger

logger = get_logger(__name__)


class APIType(Enum):
    """Supported API types for LLM providers"""

    CHAT_COMPLETION = "chat_completion"
    RESPONSES = "responses"
    COMPLETION = "completion"


class StreamMode(Enum):
    """Streaming modes for LLM responses"""

    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"  # Provider decides based on context


@dataclass
class LLMConfig:
    """
    Configuration container for LLM provider settings

    This dataclass centralizes all configuration options for LLM providers,
    making it easy to manage and extend settings.
    """

    # Model configuration
    model_name: str
    api_type: APIType = APIType.CHAT_COMPLETION
    token_limit: int = 8192

    # Generation parameters
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None

    # Streaming configuration
    stream_mode: StreamMode = StreamMode.DISABLED

    # API-specific settings
    timeout: int = 60  # Request timeout in seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # Initial retry delay in seconds

    # Additional provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/serialization"""
        return {
            "model_name": self.model_name,
            "api_type": self.api_type.value,
            "token_limit": self.token_limit,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_mode": self.stream_mode.value,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class LLMResponse:
    """
    Standardized response object from LLM providers

    This class ensures consistent response structure across different
    provider implementations and API types.
    """

    content: str
    model: str
    provider_type: str
    api_type: APIType

    # Usage statistics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Response metadata
    finish_reason: Optional[str] = None
    response_id: Optional[str] = None

    # Additional provider-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "content": self.content,
            "model": self.model,
            "provider_type": self.provider_type,
            "api_type": self.api_type.value,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "metadata": self.metadata,
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers

    This class defines the interface that all LLM provider implementations
    must follow, ensuring consistency and interchangeability.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM provider with configuration

        Args:
            config: LLMConfig object containing provider settings
        """
        self.config = config
        self.model_name = config.model_name
        self.token_limit = config.token_limit
        self.provider_type = self.__class__.__name__.lower().replace("provider", "")

        logger.info(
            f"Initialized {self.provider_type} provider - "
            f"Model: {self.model_name}, API: {config.api_type.value}, "
            f"Stream: {config.stream_mode.value}"
        )

    @abstractmethod
    async def execute(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """
        Execute an LLM call with the given prompt

        This is the main entry point for LLM execution. Based on configuration,
        it returns either a complete response or a streaming iterator.

        Args:
            prompt: The user prompt/input
            system_prompt: Optional system instructions
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse object for non-streaming, or AsyncIterator for streaming

        Raises:
            LLMProviderError: If execution fails
        """
        pass

    @abstractmethod
    async def execute_chat_completion(
        self, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """
        Execute a chat completion API call

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional API-specific parameters

        Returns:
            LLMResponse or streaming iterator

        Raises:
            LLMProviderError: If execution fails
        """
        pass

    @abstractmethod
    async def execute_responses_api(
        self,
        input_message: str,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """
        Execute a Responses API call (OpenAI-specific)

        The Responses API is a newer interface that simplifies interactions
        with conversational context and tool use.

        Args:
            input_message: User input message
            instructions: System-level instructions
            previous_response_id: ID of previous response for context
            tools: List of available tools/functions
            stream: Whether to stream the response
            **kwargs: Additional API-specific parameters

        Returns:
            LLMResponse or streaming iterator

        Raises:
            LLMProviderError: If execution fails
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text

        This is a simple approximation. For production use, consider
        using provider-specific tokenizers (e.g., tiktoken for OpenAI).

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is conservative; actual tokenization may vary
        return len(text) // 4

    def validate_token_limit(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> bool:
        """
        Validate that the prompt and expected response fit within token limits

        Args:
            prompt: The user prompt text
            max_tokens: Maximum tokens for response (uses config default if None)
            system_prompt: Optional system prompt to include in calculation

        Returns:
            True if within limits, False otherwise
        """
        # Calculate total input tokens
        prompt_tokens = self.estimate_tokens(prompt)
        if system_prompt:
            prompt_tokens += self.estimate_tokens(system_prompt)

        # Determine max response tokens
        response_tokens = (
            max_tokens or self.config.max_tokens or (self.token_limit - prompt_tokens)
        )

        total_tokens = prompt_tokens + response_tokens

        if total_tokens > self.token_limit:
            logger.warning(
                f"Token limit exceeded: {total_tokens} > {self.token_limit} "
                f"(input: {prompt_tokens}, response: {response_tokens})"
            )
            return False

        logger.debug(
            f"Token validation passed: {total_tokens}/{self.token_limit} "
            f"(input: {prompt_tokens}, response: {response_tokens})"
        )
        return True

    def update_config(self, **kwargs) -> None:
        """
        Update provider configuration dynamically

        This allows runtime modification of settings without recreating
        the provider instance.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Attempted to set unknown config key: {key}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the LLM provider

        Returns:
            Dictionary with provider information and current configuration
        """
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "api_type": self.config.api_type.value,
            "token_limit": self.token_limit,
            "stream_mode": self.config.stream_mode.value,
            "config": self.config.to_dict(),
        }

    def should_stream(self) -> bool:
        """
        Determine if responses should be streamed based on configuration

        Returns:
            True if streaming should be enabled
        """
        if self.config.stream_mode == StreamMode.ENABLED:
            return True
        elif self.config.stream_mode == StreamMode.DISABLED:
            return False
        else:  # AUTO mode
            # In AUTO mode, providers can implement custom logic
            # Default: disable streaming
            return False
