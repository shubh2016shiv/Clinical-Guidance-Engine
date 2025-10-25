"""
Stream Manager for handling streaming responses from the OpenAI Responses API.

This module provides functionality to stream responses from the OpenAI Responses API
using Server-Sent Events (SSE).
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from openai import OpenAI, AsyncOpenAI
from src.core.config import get_settings
from src.core.managers.exceptions import StreamConnectionError, ContentParsingError

logger = logging.getLogger(__name__)

class StreamManager:
    """
    Manages streaming responses from the OpenAI Responses API.
    
    Handles Server-Sent Events (SSE) for real-time streaming of model responses.
    """

    def __init__(self):
        """Initialize the Stream Manager."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)

    async def stream_response(
        self, 
        message: str, 
        model: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response from the OpenAI Responses API.
        
        Args:
            message: User message.
            model: Model to use (default: from settings).
            previous_response_id: Optional ID of previous response for context.
            tools: Optional tools to include.
            callback: Optional callback function to process each chunk.
            
        Yields:
            Text chunks from the streaming response.
        """
        try:
            model = model or self.settings.openai_model_name
            logger.info(f"Starting response stream using model {model}")
            
            # Create streaming response
            stream = await self.async_client.responses.create(
                model=model,
                input=message,
                previous_response_id=previous_response_id,
                tools=tools or [],
                stream=True  # Enable streaming
            )
            
            # Process streaming response
            async for chunk in stream:
                try:
                    # Extract text from chunk
                    if hasattr(chunk, 'content') and chunk.content:
                        for content_block in chunk.content:
                            if hasattr(content_block, 'text') and hasattr(content_block.text, 'value'):
                                text = content_block.text.value
                                if text:
                                    # Call callback if provided
                                    if callback:
                                        callback(text)
                                    yield text
                except Exception as e:
                    logger.warning(f"Error processing response chunk: {str(e)}")
                    raise ContentParsingError(f"Failed to parse response chunk: {str(e)}")
                    
            logger.info("Response stream completed")
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            raise StreamConnectionError(f"Failed to stream response: {str(e)}")

    async def stream_chat_continuation(
        self,
        chat_id: str,
        message: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a continuation of an existing chat.
        
        Args:
            chat_id: Existing chat ID (previous response ID).
            message: New user message.
            model: Model to use (default: from settings).
            tools: Optional tools to include.
            callback: Optional callback function to process each chunk.
            
        Yields:
            Text chunks from the streaming response.
        """
        try:
            model = model or self.settings.openai_model_name
            logger.info(f"Streaming chat continuation for {chat_id}")
            
            # Create streaming response with previous_response_id
            stream = await self.async_client.responses.create(
                model=model,
                input=message,
                previous_response_id=chat_id,  # Use chat_id as previous_response_id
                tools=tools or [],
                stream=True  # Enable streaming
            )
            
            # Process streaming response
            async for chunk in stream:
                try:
                    # Extract text from chunk
                    if hasattr(chunk, 'content') and chunk.content:
                        for content_block in chunk.content:
                            if hasattr(content_block, 'text') and hasattr(content_block.text, 'value'):
                                text = content_block.text.value
                                if text:
                                    # Call callback if provided
                                    if callback:
                                        callback(text)
                                    yield text
                except Exception as e:
                    logger.warning(f"Error processing response chunk: {str(e)}")
                    raise ContentParsingError(f"Failed to parse response chunk: {str(e)}")
                    
            logger.info("Chat continuation stream completed")
            
        except Exception as e:
            logger.error(f"Chat streaming error: {str(e)}")
            raise StreamConnectionError(f"Failed to stream chat continuation: {str(e)}")

    async def stream_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        callback: Optional[Callable[[str], None]] = None,
        tool_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a response with tools, handling tool calls.
        
        Args:
            message: User message.
            tools: List of tools to include.
            model: Model to use (default: from settings).
            previous_response_id: Optional ID of previous response for context.
            callback: Optional callback function to process text chunks.
            tool_callback: Optional callback function to process tool calls.
            
        Yields:
            Dictionaries containing text chunks and/or tool call information.
        """
        try:
            model = model or self.settings.openai_model_name
            logger.info(f"Starting response stream with tools using model {model}")
            
            # Create streaming response with tools
            stream = await self.async_client.responses.create(
                model=model,
                input=message,
                previous_response_id=previous_response_id,
                tools=tools,
                stream=True  # Enable streaming
            )
            
            # Process streaming response
            async for chunk in stream:
                try:
                    result = {}
                    
                    # Handle text content
                    if hasattr(chunk, 'content') and chunk.content:
                        for content_block in chunk.content:
                            if hasattr(content_block, 'text') and hasattr(content_block.text, 'value'):
                                text = content_block.text.value
                                if text:
                                    result['text'] = text
                                    if callback:
                                        callback(text)
                    
                    # Handle tool calls
                    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                        result['tool_calls'] = chunk.tool_calls
                        if tool_callback:
                            tool_callback(chunk.tool_calls)
                    
                    if result:
                        yield result
                        
                except Exception as e:
                    logger.warning(f"Error processing response chunk: {str(e)}")
                    raise ContentParsingError(f"Failed to parse response chunk: {str(e)}")
                    
            logger.info("Response stream with tools completed")
            
        except Exception as e:
            logger.error(f"Streaming error with tools: {str(e)}")
            raise StreamConnectionError(f"Failed to stream response with tools: {str(e)}")

    async def create_sse_generator(
        self,
        message: str,
        chat_id: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Create a Server-Sent Events (SSE) generator for streaming responses.
        
        Args:
            message: User message.
            chat_id: Optional chat ID for continuing a conversation.
            model: Model to use (default: from settings).
            tools: Optional tools to include.
            
        Yields:
            SSE-formatted strings for streaming to clients.
        """
        try:
            if chat_id:
                # Stream chat continuation
                async for text in self.stream_chat_continuation(chat_id, message, model, tools):
                    yield f"data: {text}\n\n"
            else:
                # Stream new chat
                async for text in self.stream_response(message, model, None, tools):
                    yield f"data: {text}\n\n"
                    
            # Signal completion
            yield "event: close\ndata: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"SSE generator error: {str(e)}")
            yield f"event: error\ndata: {str(e)}\n\n"
