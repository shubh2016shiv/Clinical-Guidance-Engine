"""
OpenAI Provider Implementation

This module provides a comprehensive implementation of the LLM provider interface
using OpenAI's API, supporting both Chat Completions and Responses API with
full streaming capabilities.

Features:
- Chat Completions API (standard interface)
- Responses API (newer conversational interface)
- Streaming and non-streaming modes
- Robust error handling and retries
- Token usage tracking
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

from src.config import get_settings
from src.providers.exceptions import LLMProviderError
from src.logs import get_logger
from .base import LLMProvider, LLMConfig, LLMResponse, APIType, StreamMode

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation supporting multiple API types

    This provider supports:
    1. Chat Completions API - Standard chat interface
    2. Responses API - Simplified conversational interface with built-in context

    Both APIs support streaming and non-streaming modes.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        model_name: Optional[str] = None,
        api_type: Optional[APIType] = None,
        stream_mode: Optional[StreamMode] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI provider

        Args:
            api_key: OpenAI API key (defaults to settings)
            config: Complete LLMConfig object (takes precedence)
            model_name: Model name (used if config not provided)
            api_type: API type to use (used if config not provided)
            stream_mode: Streaming mode (used if config not provided)
            **kwargs: Additional configuration parameters

        Raises:
            LLMProviderError: If API key is missing
        """
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key

        if not self.api_key:
            raise LLMProviderError("OpenAI API key is required")

        # Build configuration if not provided
        if config is None:
            config = LLMConfig(
                model_name=model_name or settings.openai_model_name or "gpt-4",
                api_type=api_type or APIType.CHAT_COMPLETION,
                stream_mode=stream_mode or StreamMode.DISABLED,
                token_limit=kwargs.pop("token_limit", 8192),
                **kwargs,
            )

        # Initialize parent
        super().__init__(config)

        # Initialize OpenAI clients (both sync and async)
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)

        logger.info(
            f"OpenAI provider initialized - "
            f"Model: {self.model_name}, API: {self.config.api_type.value}"
        )

    async def execute(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """
        Execute an LLM call based on configured API type

        This method routes to the appropriate API implementation based on
        the provider's configuration.

        Args:
            prompt: User input/prompt
            system_prompt: Optional system instructions
            **kwargs: Additional API-specific parameters

        Returns:
            LLMResponse or streaming iterator based on stream_mode

        Raises:
            LLMProviderError: If execution fails
        """
        # Determine if we should stream
        stream = kwargs.pop("stream", self.should_stream())

        # Route to appropriate API
        if self.config.api_type == APIType.RESPONSES:
            return await self.execute_responses_api(
                input_message=prompt,
                instructions=system_prompt,
                stream=stream,
                **kwargs,
            )
        else:  # Default to CHAT_COMPLETION
            # Build messages list
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            return await self.execute_chat_completion(
                messages=messages, stream=stream, **kwargs
            )

    async def execute_chat_completion(
        self, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    ) -> Union[LLMResponse, AsyncIterator[str]]:
        """
        Execute a Chat Completion API call

        The Chat Completions API is OpenAI's standard interface for
        conversational interactions.

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional parameters (overrides config defaults)

        Returns:
            LLMResponse for non-streaming, AsyncIterator[str] for streaming

        Raises:
            LLMProviderError: If API call fails
        """
        try:
            # Validate token limits (estimate from messages)
            total_prompt = " ".join(msg.get("content", "") for msg in messages)
            if not self.validate_token_limit(total_prompt, kwargs.get("max_tokens")):
                raise LLMProviderError("Token limit exceeded")

            # Build API parameters
            params = self._build_chat_params(messages, stream, **kwargs)

            # Execute based on streaming mode
            if stream:
                return self._stream_chat_completion(params)
            else:
                return await self._execute_chat_completion(params)

        except Exception as e:
            error_msg = f"Chat completion execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise LLMProviderError(error_msg)

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
        Execute a Responses API call

        The Responses API is a newer OpenAI interface that simplifies
        conversational interactions with built-in context management.

        Args:
            input_message: User's input message
            instructions: System-level instructions for the assistant
            previous_response_id: ID of previous response for context continuity
            tools: List of available tools/functions
            stream: Whether to stream the response
            **kwargs: Additional API parameters

        Returns:
            LLMResponse for non-streaming, AsyncIterator[str] for streaming

        Raises:
            LLMProviderError: If API call fails
        """
        try:
            # Validate token limits
            full_prompt = f"{instructions or ''}\n{input_message}"
            if not self.validate_token_limit(full_prompt, kwargs.get("max_tokens")):
                raise LLMProviderError("Token limit exceeded")

            # Build API parameters
            params = {
                "model": kwargs.get("model", self.model_name),
                "input": input_message,
            }

            # Add optional parameters
            if instructions:
                params["instructions"] = instructions
            if previous_response_id:
                params["previous_response_id"] = previous_response_id
            if tools:
                params["tools"] = tools

            # Add streaming flag
            params["stream"] = stream

            # Merge with config defaults and kwargs
            params.update(self._get_generation_params(**kwargs))

            # Execute based on streaming mode
            if stream:
                # For streaming Responses API, return raw stream object directly
                # This allows StreamManager to process OpenAI SDK events properly
                logger.info("Returning raw Responses API stream for StreamManager")
                return await self.async_client.responses.create(**params)
            else:
                return await self._execute_responses_api(params)

        except Exception as e:
            error_msg = f"Responses API execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise LLMProviderError(error_msg)

    def _build_chat_params(
        self, messages: List[Dict[str, str]], stream: bool, **kwargs
    ) -> Dict[str, Any]:
        """
        Build parameters dictionary for Chat Completions API

        Args:
            messages: List of message dictionaries
            stream: Whether to enable streaming
            **kwargs: Additional parameter overrides

        Returns:
            Complete parameters dictionary for API call
        """
        params = {
            "model": kwargs.get("model", self.model_name),
            "messages": messages,
            "stream": stream,
        }

        # Merge generation parameters from config and kwargs
        params.update(self._get_generation_params(**kwargs))

        return params

    def _get_generation_params(self, **kwargs) -> Dict[str, Any]:
        """
        Build generation parameters from config and kwargs

        Priority: kwargs > config > defaults

        Args:
            **kwargs: Parameter overrides

        Returns:
            Dictionary of generation parameters
        """
        params = {}

        # Temperature
        if "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]
        elif self.config.temperature is not None:
            params["temperature"] = self.config.temperature

        # Max tokens
        if "max_tokens" in kwargs:
            params["max_tokens"] = kwargs["max_tokens"]
        elif self.config.max_tokens is not None:
            params["max_tokens"] = self.config.max_tokens

        # Top P
        if "top_p" in kwargs:
            params["top_p"] = kwargs["top_p"]
        elif self.config.top_p is not None:
            params["top_p"] = self.config.top_p

        # Frequency penalty
        if "frequency_penalty" in kwargs:
            params["frequency_penalty"] = kwargs["frequency_penalty"]
        elif self.config.frequency_penalty is not None:
            params["frequency_penalty"] = self.config.frequency_penalty

        # Presence penalty
        if "presence_penalty" in kwargs:
            params["presence_penalty"] = kwargs["presence_penalty"]
        elif self.config.presence_penalty is not None:
            params["presence_penalty"] = self.config.presence_penalty

        # Stop sequences
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]
        elif self.config.stop is not None:
            params["stop"] = self.config.stop

        return params

    async def _execute_chat_completion(self, params: Dict[str, Any]) -> LLMResponse:
        """
        Execute non-streaming Chat Completion API call

        Args:
            params: API parameters

        Returns:
            LLMResponse with complete response

        Raises:
            LLMProviderError: If API call fails
        """
        response: ChatCompletion = await self.async_client.chat.completions.create(
            **params
        )

        # Extract content
        if not response.choices or not response.choices[0].message:
            raise LLMProviderError("No content in API response")

        content = response.choices[0].message.content or ""

        # Build standardized response
        return LLMResponse(
            content=content,
            model=response.model,
            provider_type=self.provider_type,
            api_type=APIType.CHAT_COMPLETION,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens
            if response.usage
            else None,
            total_tokens=response.usage.total_tokens if response.usage else None,
            finish_reason=response.choices[0].finish_reason,
            response_id=response.id,
            metadata={
                "created": response.created,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
        )

    async def _stream_chat_completion(
        self, params: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        Execute streaming Chat Completion API call

        Args:
            params: API parameters

        Yields:
            Content chunks as they arrive

        Raises:
            LLMProviderError: If streaming fails
        """
        try:
            stream = await self.async_client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_msg = f"Streaming chat completion failed: {str(e)}"
            logger.error(error_msg)
            raise LLMProviderError(error_msg)

    async def _execute_responses_api(self, params: Dict[str, Any]) -> LLMResponse:
        """
        Execute non-streaming Responses API call

        Note: This uses synchronous client with asyncio.to_thread because
        the Responses API might not have full async support yet.

        Args:
            params: API parameters

        Returns:
            LLMResponse with complete response

        Raises:
            LLMProviderError: If API call fails
        """
        # Remove stream parameter for non-streaming call
        params_copy = params.copy()
        params_copy.pop("stream", None)

        # Execute in thread pool to avoid blocking
        response = await asyncio.to_thread(self.client.responses.create, **params_copy)

        # Extract content (adapt based on actual response structure)
        content = getattr(response, "content", "") or str(response)

        # Build standardized response
        return LLMResponse(
            content=content,
            model=params["model"],
            provider_type=self.provider_type,
            api_type=APIType.RESPONSES,
            response_id=getattr(response, "id", None),
            metadata={
                "response_object": response,
            },
        )

    async def _stream_responses_api(self, params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Execute streaming Responses API call

        Args:
            params: API parameters

        Yields:
            Content chunks as they arrive

        Raises:
            LLMProviderError: If streaming fails
        """
        try:
            stream = await self.async_client.responses.create(**params)

            async for chunk in stream:
                # Adapt based on actual streaming response structure
                if hasattr(chunk, "content"):
                    yield chunk.content
                elif hasattr(chunk, "delta"):
                    yield chunk.delta

        except Exception as e:
            error_msg = f"Streaming Responses API failed: {str(e)}"
            logger.error(error_msg)
            raise LLMProviderError(error_msg)
