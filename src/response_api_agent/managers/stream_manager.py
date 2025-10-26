"""
Stream Manager for handling streaming responses from the OpenAI Responses API.

This module provides functionality to stream responses from the OpenAI Responses API
using Server-Sent Events (SSE).
"""

from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from openai import OpenAI, AsyncOpenAI
from src.config import get_settings
from src.response_api_agent.managers.exceptions import StreamConnectionError, ContentParsingError
from src.logs import get_component_logger, time_execution
from src.prompts.asclepius_system_prompt import get_system_prompt

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
        self.logger = get_component_logger("Stream")

    @time_execution("Stream", "StreamResponse")
    async def stream_response(
        self, 
        message: str, 
        model: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a response from the OpenAI Responses API.
        
        Args:
            message: User message.
            model: Model to use (default: from settings).
            previous_response_id: Optional ID of previous response for context.
            tools: Optional tools to include.
            callback: Optional callback function to process each chunk.
            
        Yields:
            Dictionary containing 'text' and 'response_id' from the streaming response.
        """
        try:
            model = model or self.settings.openai_model_name
            
            self.logger.info(
                "Starting response stream",
                component="Stream",
                subcomponent="StreamResponse",
                model=model,
                has_previous_response=bool(previous_response_id),
                has_tools=bool(tools),
                message_length=len(message)
            )
            
            # Create streaming response
            stream = await self.async_client.responses.create(
                model=model,
                input=message,
                previous_response_id=previous_response_id,
                instructions=get_system_prompt(),
                tools=tools or [],
                stream=True  # Enable streaming
            )
            
            chunk_count = 0
            response_id = None
            self.logger.info(
                "Beginning to process stream chunks",
                component="Stream",
                subcomponent="StreamResponse"
            )
            
            # Process streaming response
            async for chunk in stream:
                try:
                    # CORRECTED: Handle Response API streaming format
                    text = None
                    chunk_type = type(chunk).__name__
                    
                    # Extract response ID from ResponseCreatedEvent
                    if chunk_type == "ResponseCreatedEvent" and hasattr(chunk, 'response'):
                        response_id = chunk.response.id
                        self.logger.info(
                            "Extracted response ID from ResponseCreatedEvent",
                            component="Stream",
                            subcomponent="StreamResponse",
                            response_id=response_id
                        )
                    
                    # Handle ResponseTextDeltaEvent - this is where the actual text content is
                    elif chunk_type == "ResponseTextDeltaEvent" and hasattr(chunk, 'delta'):
                        text = chunk.delta
                        if text and text.strip():
                            chunk_count += 1
                            # self.logger.info(
                            #     "Extracted text from ResponseTextDeltaEvent",
                            #     component="Stream",
                            #     subcomponent="StreamResponse",
                            #     text_length=len(text),
                            #     chunk_count=chunk_count
                            # )
                            
                            if callback:
                                callback(text)
                            print(text)
                            yield {
                                "text": text,
                                "response_id": response_id
                            }
                        else:
                            self.logger.debug(
                                "Empty text in ResponseTextDeltaEvent",
                                component="Stream",
                                subcomponent="StreamResponse"
                            )
                    else:
                        # Log other chunk types for debugging
                        self.logger.debug(
                            "Non-text chunk received",
                            component="Stream",
                            subcomponent="StreamResponse",
                            chunk_type=chunk_type,
                            has_delta=hasattr(chunk, 'delta'),
                            delta_value=getattr(chunk, 'delta', None) if hasattr(chunk, 'delta') else None
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        "Error processing response chunk",
                        component="Stream",
                        subcomponent="StreamResponse",
                        error=str(e),
                        chunk_type=type(chunk).__name__
                    )
                    # Don't raise - continue processing other chunks
                    continue
            
            self.logger.info(
                "Response stream completed",
                component="Stream",
                subcomponent="StreamResponse",
                chunk_count=chunk_count
            )
            
        except Exception as e:
            self.logger.error(
                "Streaming error",
                component="Stream",
                subcomponent="StreamResponse",
                error=str(e)
            )
            raise StreamConnectionError(f"Failed to stream response: {str(e)}")

    @time_execution("Stream", "StreamChatContinuation")
    async def stream_chat_continuation(
        self,
        chat_id: str,
        message: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a continuation of an existing chat.
        
        Args:
            chat_id: Existing chat ID (previous response ID).
            message: New user message.
            model: Model to use (default: from settings).
            tools: Optional tools to include.
            callback: Optional callback function to process each chunk.
            
        Yields:
            Dictionary containing 'text' and 'response_id' from the streaming response.
        """
        try:
            model = model or self.settings.openai_model_name
            
            self.logger.info(
                "Starting chat continuation stream",
                component="Stream",
                subcomponent="StreamChatContinuation",
                chat_id=chat_id,
                model=model,
                has_tools=bool(tools),
                message_length=len(message)
            )
            
            # Create streaming response with previous_response_id
            stream = await self.async_client.responses.create(
                model=model,
                input=message,
                previous_response_id=chat_id,  # Use chat_id as previous_response_id
                instructions=get_system_prompt(),
                tools=tools or [],
                stream=True  # Enable streaming
            )
            
            chunk_count = 0
            response_id = None
            # Process streaming response
            async for chunk in stream:
                try:
                    # CORRECTED: Handle Response API streaming format for chat continuation
                    text = None
                    chunk_type = type(chunk).__name__
                    
                    # Extract response ID from ResponseCreatedEvent
                    if chunk_type == "ResponseCreatedEvent" and hasattr(chunk, 'response'):
                        response_id = chunk.response.id
                        self.logger.info(
                            "Extracted response ID from chat continuation ResponseCreatedEvent",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                            response_id=response_id
                        )
                    
                    # Handle ResponseTextDeltaEvent - this is where the actual text content is
                    elif chunk_type == "ResponseTextDeltaEvent" and hasattr(chunk, 'delta'):
                        text = chunk.delta
                        if text and text.strip():
                            chunk_count += 1
                            self.logger.info(
                                "Extracted text from chat continuation ResponseTextDeltaEvent",
                                component="Stream",
                                subcomponent="StreamChatContinuation",
                                text_length=len(text),
                                chunk_count=chunk_count
                            )
                            
                            if callback:
                                callback(text)
                            yield {
                                "text": text,
                                "response_id": response_id
                            }
                        else:
                            self.logger.debug(
                                "Empty text in chat continuation ResponseTextDeltaEvent",
                                component="Stream",
                                subcomponent="StreamChatContinuation"
                            )
                    else:
                        # Log other chunk types for debugging
                        self.logger.debug(
                            "Non-text chunk in chat continuation",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                            chunk_type=chunk_type
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        "Error processing chat continuation chunk",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        chat_id=chat_id,
                        error=str(e)
                    )
                    # Don't raise - continue processing other chunks
                    continue
            
            self.logger.info(
                "Chat continuation stream completed",
                component="Stream",
                subcomponent="StreamChatContinuation",
                chat_id=chat_id,
                chunk_count=chunk_count
            )
            
        except Exception as e:
            self.logger.error(
                "Chat streaming error",
                component="Stream",
                subcomponent="StreamChatContinuation",
                chat_id=chat_id,
                error=str(e)
            )
            raise StreamConnectionError(f"Failed to stream chat continuation: {str(e)}")

    @time_execution("Stream", "StreamWithTools")
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
            
            self.logger.info(
                "Starting response stream with tools",
                component="Stream",
                subcomponent="StreamWithTools",
                model=model,
                has_previous_response=bool(previous_response_id),
                tool_count=len(tools),
                message_length=len(message)
            )
            
            # Create streaming response with tools
            stream = await self.async_client.responses.create(
                model=model,
                input=message,
                previous_response_id=previous_response_id,
                instructions=get_system_prompt(),
                tools=tools,
                stream=True  # Enable streaming
            )
            
            chunk_count = 0
            tool_call_count = 0
            
            # Process streaming response
            async for chunk in stream:
                try:
                    result = {}
                    chunk_type = type(chunk).__name__
                    
                    # Handle ResponseTextDeltaEvent for text content
                    if chunk_type == "ResponseTextDeltaEvent" and hasattr(chunk, 'delta'):
                        text = chunk.delta
                        if text and text.strip():
                            chunk_count += 1
                            result['text'] = text
                            if callback:
                                callback(text)
                    
                    # Handle tool calls (this would be in different event types)
                    if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                        tool_call_count += 1
                        result['tool_calls'] = chunk.tool_calls
                        if tool_callback:
                            tool_callback(chunk.tool_calls)
                        
                        self.logger.info(
                            "Received tool call in stream",
                            component="Stream",
                            subcomponent="StreamWithTools",
                            tool_call_count=tool_call_count
                        )
                    
                    if result:
                        yield result
                    else:
                        # Log other chunk types for debugging
                        self.logger.debug(
                            "Non-text chunk in stream with tools",
                            component="Stream",
                            subcomponent="StreamWithTools",
                            chunk_type=chunk_type
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        "Error processing response chunk with tools",
                        component="Stream",
                        subcomponent="StreamWithTools",
                        error=str(e)
                    )
                    # Don't raise - continue processing other chunks
                    continue
            
            self.logger.info(
                "Response stream with tools completed",
                component="Stream",
                subcomponent="StreamWithTools",
                chunk_count=chunk_count,
                tool_call_count=tool_call_count
            )
            
        except Exception as e:
            self.logger.error(
                "Streaming error with tools",
                component="Stream",
                subcomponent="StreamWithTools",
                error=str(e)
            )
            raise StreamConnectionError(f"Failed to stream response with tools: {str(e)}")

    @time_execution("Stream", "CreateSSEGenerator")
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
            self.logger.info(
                "Creating SSE generator",
                component="Stream",
                subcomponent="CreateSSEGenerator",
                has_chat_id=bool(chat_id),
                model=model or self.settings.openai_model_name,
                has_tools=bool(tools),
                message_length=len(message)
            )
            
            chunk_count = 0
            
            if chat_id:
                # Stream chat continuation
                self.logger.info(
                    "Streaming chat continuation for SSE",
                    component="Stream",
                    subcomponent="CreateSSEGenerator",
                    chat_id=chat_id
                )
                
                async for text in self.stream_chat_continuation(chat_id, message, model, tools):
                    chunk_count += 1
                    yield f"data: {text}\n\n"
            else:
                # Stream new chat
                self.logger.info(
                    "Streaming new chat for SSE",
                    component="Stream",
                    subcomponent="CreateSSEGenerator"
                )
                
                async for text in self.stream_response(message, model, None, tools):
                    chunk_count += 1
                    yield f"data: {text}\n\n"
            
            # Signal completion
            self.logger.info(
                "SSE stream completed",
                component="Stream",
                subcomponent="CreateSSEGenerator",
                chunk_count=chunk_count
            )
            yield "event: close\ndata: [DONE]\n\n"
            
        except Exception as e:
            self.logger.error(
                "SSE generator error",
                component="Stream",
                subcomponent="CreateSSEGenerator",
                error=str(e)
            )
            yield f"event: error\ndata: {str(e)}\n\n"
