"""
Adapter for LLM provider integration with fallback to direct Response API calls.

This ensures business continuity while transitioning to the new provider architecture.
"""

from typing import Any, Optional, AsyncIterator
import asyncio
import time

from src.providers.llm_provider import (
    create_llm_provider,
    APIType,
    StreamMode,
    LLMResponse,
)
from src.providers.exceptions import LLMProviderError
from src.logs import get_logger
from src.response_api_agent.managers.adapter_monitoring import monitor_adapter_call

logger = get_logger(__name__)


class ResponseAdapter:
    """Adapter class that mimics OpenAI Response API object structure"""

    def __init__(
        self, llm_response: LLMResponse, previous_response_id: Optional[str] = None
    ):
        """Initialize from LLMResponse"""
        self.id = llm_response.response_id or f"resp_{id(llm_response)}"
        self.model = llm_response.model
        self.object = "response"
        self.created = int(time.time())

        # Main content output
        self.output = [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": llm_response.content}],
            }
        ]

        # Usage statistics
        self.usage = {
            "prompt_tokens": llm_response.prompt_tokens or 0,
            "completion_tokens": llm_response.completion_tokens or 0,
            "total_tokens": llm_response.total_tokens or 0,
        }

        # Previous response ID for chaining
        self.previous_response_id = previous_response_id

        # Tool calls (empty by default, can be extended)
        self.tool_calls = None


class ResponseAPIAdapter:
    """Adapter that tries LLM provider first, falls back to direct Response API"""

    def __init__(self, client, async_client):
        self.client = client
        self.async_client = async_client
        self.llm_provider = None

    async def _init_llm_provider(self):
        """Lazy initialization of LLM provider"""
        if self.llm_provider is None:
            try:
                self.llm_provider = create_llm_provider(
                    provider_type="openai",
                    api_type=APIType.RESPONSES,
                    stream_mode=StreamMode.AUTO,
                )
                logger.info("Successfully initialized LLM provider for Response API")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider: {e}")
                self.llm_provider = None

    @monitor_adapter_call
    async def create_response(self, **kwargs) -> Any:
        """Create response with LLM provider fallback"""
        try:
            await self._init_llm_provider()
            if self.llm_provider:
                # Try LLM provider first
                logger.info("Using LLM provider for Response API execution")
                response = await self.llm_provider.execute_responses_api(
                    input_message=kwargs.get("input"),
                    instructions=kwargs.get("instructions"),
                    tools=kwargs.get("tools", []),
                    previous_response_id=kwargs.get("previous_response_id"),
                    stream=False,
                )
                return self._map_llm_response_to_api(
                    response, kwargs.get("previous_response_id")
                )
        except LLMProviderError as e:
            logger.warning(f"LLM provider failed, falling back to direct API: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error with LLM provider, falling back: {e}")

        # Fallback to direct Response API
        logger.info("Falling back to direct Response API execution")
        return await asyncio.to_thread(self.client.responses.create, **kwargs)

    @monitor_adapter_call
    async def create_streaming_response(self, **kwargs) -> AsyncIterator:
        """Create streaming response with fallback"""
        try:
            await self._init_llm_provider()
            if self.llm_provider:
                # Try LLM provider first
                logger.info("Using LLM provider for streaming Response API execution")
                stream = await self.llm_provider.execute_responses_api(
                    input_message=kwargs.get("input"),
                    instructions=kwargs.get("instructions"),
                    tools=kwargs.get("tools", []),
                    previous_response_id=kwargs.get("previous_response_id"),
                    stream=True,
                )
                return stream
        except LLMProviderError as e:
            logger.warning(f"LLM provider streaming failed, falling back: {e}")
        except Exception as e:
            logger.warning(
                f"Unexpected error with LLM provider streaming, falling back: {e}"
            )

        # Fallback to direct streaming API
        logger.info("Falling back to direct streaming Response API execution")
        return await self.async_client.responses.create(**kwargs)

    def _map_llm_response_to_api(
        self, llm_response: LLMResponse, previous_response_id: Optional[str] = None
    ):
        """Map LLMResponse to Response API format using ResponseAdapter"""
        return ResponseAdapter(llm_response, previous_response_id)
