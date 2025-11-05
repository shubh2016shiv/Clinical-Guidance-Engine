"""
Stream Manager for handling streaming responses from the OpenAI Responses API.

This module provides functionality to stream responses from the OpenAI Responses API
using Server-Sent Events (SSE) WITH PROPER CITATION HANDLING.

User Query
    ↓
Start Streaming (stream=True)
    ↓
[ResponseCreatedEvent] → Capture response_id
    ↓
[ResponseTextDeltaEvent] → Stream text chunks to user
[ResponseTextDeltaEvent] → (more text)
[ResponseTextDeltaEvent] → (more text)
    ↓
[ResponseOutputTextAnnotationAddedEvent] → (Optional: track that citations exist)
    ↓
[ResponseFileSearchCallCompleted] → (File search finished)
    ↓
[ResponseCompletedEvent] → Stream ends
    ↓
Call responses.retrieve(response_id) → Get complete response
    ↓
Extract annotations from response.output[1].content[0].annotations
    ↓
Format citations
    ↓
Emit citation chunk
    ↓
Done!




"""

import json
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, TYPE_CHECKING
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
from src.config import get_settings
from src.response_api_agent.managers.exceptions import StreamConnectionError
from src.response_api_agent.managers.citation_manager import CitationManager
from src.response_api_agent.managers.llm_provider_adapter import ResponseAPIAdapter
from src.logs import get_component_logger, time_execution
from src.prompts.asclepius_system_prompt import get_system_prompt
from src.providers.cache_provider import CacheProvider

if TYPE_CHECKING:
    from src.response_api_agent.managers.chat_manager import ChatManager


class StreamManager:
    """
    Manages streaming responses from the OpenAI Responses API.

    Handles Server-Sent Events (SSE) for real-time streaming of model responses.
    """

    def __init__(
        self,
        chat_manager: Optional["ChatManager"] = None,
        cache_provider: Optional[CacheProvider] = None,
    ):
        """
        Initialize the Stream Manager.

        Args:
            chat_manager: Optional ChatManager instance for tool call extraction and execution.
            cache_provider: Optional CacheProvider for Redis caching of streaming state.
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.response_adapter = ResponseAPIAdapter(self.client, self.async_client)
        self.citation_manager = CitationManager(client=self.async_client)
        self.chat_manager = chat_manager  # Store the chat manager reference
        self.cache_provider = cache_provider
        self.logger = get_component_logger("Stream")

    @time_execution("Stream", "StreamResponse")
    async def stream_response(
        self,
        message: str,
        model: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None,
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
        # Generate request ID for tracking
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        session_id = previous_response_id or "new_session"

        try:
            model = model or self.settings.openai_model_name

            self.logger.info(
                "Starting response stream",
                component="Stream",
                subcomponent="StreamResponse",
                model=model,
                has_previous_response=bool(previous_response_id),
                has_tools=bool(tools),
                message_length=len(message),
                request_id=request_id,
            )

            # Create streaming response
            stream = await self.response_adapter.create_streaming_response(
                model=model,
                input=message,
                previous_response_id=previous_response_id,
                instructions=get_system_prompt(),
                tools=tools or [],
                stream=True,  # Enable streaming
            )

            chunk_count = 0
            response_id = None
            collected_annotations = []  # CRITICAL: Collect annotations during stream

            # Set initial streaming state
            if self.cache_provider:
                try:
                    await self.cache_provider.set_streaming_state(
                        session_id=session_id,
                        request_id=request_id,
                        state={
                            "status": "streaming",
                            "chunk_count": 0,
                            "last_chunk_at": datetime.utcnow().isoformat(),
                            "response_id": None,
                            "is_citation": False,
                        },
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to set streaming state: {e}",
                        component="Stream",
                        subcomponent="StreamResponse",
                    )

            self.logger.info(
                "Beginning to process stream chunks",
                component="Stream",
                subcomponent="StreamResponse",
            )

            # Process streaming response
            async for chunk in stream:
                try:
                    text = None
                    chunk_type = type(chunk).__name__

                    # Extract response ID from ResponseCreatedEvent
                    if chunk_type == "ResponseCreatedEvent" and hasattr(
                        chunk, "response"
                    ):
                        response_id = chunk.response.id
                        self.logger.info(
                            "Extracted response ID from ResponseCreatedEvent",
                            component="Stream",
                            subcomponent="StreamResponse",
                            response_id=response_id,
                        )

                        # Update streaming state with response_id
                        if self.cache_provider:
                            try:
                                await self.cache_provider.set_streaming_state(
                                    session_id=session_id,
                                    request_id=request_id,
                                    state={
                                        "status": "streaming",
                                        "response_id": response_id,
                                        "chunk_count": chunk_count,
                                        "last_chunk_at": datetime.utcnow().isoformat(),
                                    },
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to update streaming state: {e}",
                                    component="Stream",
                                    subcomponent="StreamResponse",
                                )

                    # Handle ResponseTextDeltaEvent - this is where the actual text content is
                    elif chunk_type == "ResponseTextDeltaEvent" and hasattr(
                        chunk, "delta"
                    ):
                        text = chunk.delta
                        if text and text.strip():
                            chunk_count += 1

                            if callback:
                                callback(text)
                            print(text, end="", flush=True)
                            yield {"text": text, "response_id": response_id}

                            # Update streaming state periodically (every 10 chunks)
                            if self.cache_provider and chunk_count % 10 == 0:
                                try:
                                    await self.cache_provider.set_streaming_state(
                                        session_id=session_id,
                                        request_id=request_id,
                                        state={
                                            "status": "streaming",
                                            "response_id": response_id,
                                            "chunk_count": chunk_count,
                                            "last_chunk_at": datetime.utcnow().isoformat(),
                                        },
                                    )
                                except Exception:
                                    pass  # Don't log every update failure

                    # CRITICAL: Capture annotation events during streaming
                    # These events contain file citation information
                    elif chunk_type == "ResponseOutputTextAnnotationAddedEvent":
                        if hasattr(chunk, "annotation"):
                            collected_annotations.append(chunk.annotation)
                            self.logger.info(
                                "Annotation captured during stream",
                                component="Stream",
                                subcomponent="StreamResponse",
                                annotation_count=len(collected_annotations),
                                has_filename=hasattr(chunk.annotation, "filename"),
                            )

                    # Also check for annotations in other potential event types
                    elif hasattr(chunk, "annotations") and chunk.annotations:
                        collected_annotations.extend(chunk.annotations)
                        self.logger.info(
                            "Multiple annotations found in chunk",
                            component="Stream",
                            subcomponent="StreamResponse",
                            annotation_count=len(chunk.annotations),
                        )

                    # Log file search completion for debugging
                    elif chunk_type == "ResponseFileSearchCallCompleted":
                        self.logger.info(
                            "File search call completed",
                            component="Stream",
                            subcomponent="StreamResponse",
                        )

                    else:
                        # Log other chunk types for debugging
                        self.logger.debug(
                            "Other chunk received",
                            component="Stream",
                            subcomponent="StreamResponse",
                            chunk_type=chunk_type,
                            has_annotations=hasattr(chunk, "annotations"),
                        )

                except Exception as e:
                    self.logger.warning(
                        "Error processing response chunk",
                        component="Stream",
                        subcomponent="StreamResponse",
                        error=str(e),
                        chunk_type=type(chunk).__name__,
                    )
                    # Don't raise - continue processing other chunks
                    continue

            # CRITICAL: After streaming completes, get final response for complete citation data
            if response_id:
                try:
                    self.logger.info(
                        "Stream completed, retrieving final response for citations",
                        component="Stream",
                        subcomponent="StreamResponse",
                        response_id=response_id,
                        annotations_during_stream=len(collected_annotations),
                    )

                    # Get the final response to extract complete citation information
                    final_response = await self.async_client.responses.retrieve(
                        response_id=response_id
                    )

                    # Extract citations using the citation manager
                    citations = (
                        await self.citation_manager.extract_citations_from_response(
                            final_response
                        )
                    )

                    self.logger.info(
                        "Citations extracted from final response",
                        component="Stream",
                        subcomponent="StreamResponse",
                        citation_count=len(citations),
                    )

                    # Emit citations as a separate chunk
                    if citations:
                        citation_text = (
                            "\n\n"
                            + self.citation_manager.format_citations_section(citations)
                        )
                        print(citation_text)  # Print to console
                        yield {
                            "text": citation_text,
                            "response_id": response_id,
                            "is_citation": True,  # Mark as citation chunk
                            "citations": citations,  # Include citation data
                        }
                    else:
                        self.logger.warning(
                            "No citations found in final response",
                            component="Stream",
                            subcomponent="StreamResponse",
                        )

                except Exception as e:
                    self.logger.error(
                        "Error retrieving final response for citations",
                        component="Stream",
                        subcomponent="StreamResponse",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
            else:
                self.logger.warning(
                    "No response_id available for citation extraction",
                    component="Stream",
                    subcomponent="StreamResponse",
                )

            # CRITICAL FIX: Always yield response_id at the end, even if no text chunks were yielded
            # This ensures tool execution streaming can capture the response_id
            if response_id:
                yield {
                    "text": "",  # Empty text, but include response_id
                    "response_id": response_id,
                    "is_citation": False,
                }

            # Clear streaming state after completion
            if self.cache_provider:
                try:
                    await self.cache_provider.set_streaming_state(
                        session_id=session_id,
                        request_id=request_id,
                        state={
                            "status": "completed",
                            "response_id": response_id,
                            "chunk_count": chunk_count,
                            "last_chunk_at": datetime.utcnow().isoformat(),
                        },
                    )
                    # Clear after a brief moment (will auto-expire with TTL anyway)
                    await self.cache_provider.clear_streaming_state(
                        session_id, request_id
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to clear streaming state: {e}",
                        component="Stream",
                        subcomponent="StreamResponse",
                    )

            self.logger.info(
                "Response stream completed",
                component="Stream",
                subcomponent="StreamResponse",
                chunk_count=chunk_count,
                final_citation_count=len(citations) if response_id else 0,
            )

        except Exception as e:
            self.logger.error(
                "Streaming error",
                component="Stream",
                subcomponent="StreamResponse",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise StreamConnectionError(f"Failed to stream response: {str(e)}")

    @time_execution("Stream", "StreamResponseWithToolExecution")
    async def stream_response_with_tool_execution(
        self,
        message: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        vector_store_id: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5,
        callback: Optional[Callable[[str], None]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a response with automatic tool execution loop.

        This method streams the initial response, then if tool calls are detected,
        it automatically executes them and streams continuation responses until
        either no tool calls remain or max iterations is reached.

        Args:
            message: User message.
            model: Model to use (default: from settings).
            tools: List of tools to include (should include both file_search and functions).
            vector_store_id: Optional vector store ID for file_search.
            functions: Optional function definitions for function calling.
            max_iterations: Maximum tool execution iterations (default: 5).
            callback: Optional callback function to process text chunks.

        Yields:
            Dictionaries containing text chunks, tool execution info, and citations.
        """
        try:
            model = model or self.settings.openai_model_name
            iteration = 0
            current_response_id = None
            tool_execution_history = []

            self.logger.info(
                "Starting response stream with tool execution",
                component="Stream",
                subcomponent="StreamResponseWithToolExecution",
                model=model,
                tools_count=len(tools or []),
                max_iterations=max_iterations,
            )

            # Phase 1: Stream initial response
            self.logger.info(
                "Phase 1: Streaming initial response",
                component="Stream",
                subcomponent="StreamResponseWithToolExecution",
            )

            async for chunk_data in self.stream_response(
                message=message, model=model, tools=tools, callback=callback
            ):
                # Extract response_id from first chunk
                if current_response_id is None and chunk_data.get("response_id"):
                    current_response_id = chunk_data["response_id"]

                yield chunk_data

            # Phase 2: Tool execution loop
            if not current_response_id:
                self.logger.warning(
                    "No response_id captured during stream, skipping tool execution",
                    component="Stream",
                    subcomponent="StreamResponseWithToolExecution",
                )
            elif not self.chat_manager:
                self.logger.warning(
                    "ChatManager not available, skipping tool execution",
                    component="Stream",
                    subcomponent="StreamResponseWithToolExecution",
                )

            if current_response_id and self.chat_manager:
                try:
                    # Retrieve final response to check for tool calls
                    final_response = await self.async_client.responses.retrieve(
                        current_response_id
                    )

                    # Extract tool calls from response
                    tool_calls = self.chat_manager._extract_tool_calls_from_response(
                        final_response
                    )

                    self.logger.info(
                        "Phase 2: Checking for tool calls after initial stream",
                        component="Stream",
                        subcomponent="StreamResponseWithToolExecution",
                        response_id=current_response_id,
                        tool_call_count=len(tool_calls),
                    )

                    # Conditional print for streaming with function calls
                    if any(tc.get("type") == "function_call" for tc in tool_calls):
                        print("STREAMING STARTED >>>", flush=True)

                    # Tool execution loop
                    while tool_calls and iteration < max_iterations:
                        iteration += 1
                        self.logger.info(
                            f"Tool execution iteration {iteration}/{max_iterations}",
                            component="Stream",
                            subcomponent="StreamResponseWithToolExecution",
                            tool_call_count=len(tool_calls),
                        )

                        # Execute all tool calls
                        tool_outputs = []
                        for tool_call in tool_calls:
                            try:
                                call_id = tool_call["call_id"]
                                function_name = tool_call["function_name"]
                                arguments = tool_call["arguments"]

                                self.logger.info(
                                    f"→ Executing Function: {function_name}",
                                    component="Stream",
                                    subcomponent="StreamResponseWithToolExecution",
                                    function_name=function_name,
                                    call_id=call_id,
                                    arguments_count=len(arguments),
                                )

                                # Execute function
                                result_str = (
                                    await self.chat_manager._execute_function_call(
                                        function_name, arguments
                                    )
                                )

                                # Collect tool output
                                tool_outputs.append(
                                    {"call_id": call_id, "output": result_str}
                                )

                                tool_execution_history.append(
                                    {
                                        "iteration": iteration,
                                        "call_id": call_id,
                                        "function": function_name,
                                        "arguments": arguments,
                                        "result": result_str,
                                    }
                                )

                            except Exception as e:
                                error_msg = f"Error executing tool: {str(e)}"
                                self.logger.error(
                                    error_msg,
                                    component="Stream",
                                    subcomponent="StreamResponseWithToolExecution",
                                    exc_info=True,
                                )
                                tool_outputs.append(
                                    {
                                        "call_id": tool_call.get(
                                            "call_id", f"error_{iteration}"
                                        ),
                                        "output": json.dumps({"error": error_msg}),
                                    }
                                )

                        if not tool_outputs:
                            break

                        # # Continue chat with tool outputs
                        # self.logger.info(
                        #     "Submitting tool outputs and continuing stream",
                        #     component="Stream",
                        #     subcomponent="StreamResponseWithToolExecution",
                        #     tool_output_count=len(tool_outputs),
                        # )

                        # current_response_id = (
                        #     await self.chat_manager.continue_chat_with_tool_outputs(
                        #         previous_response_id=current_response_id,
                        #         tool_outputs=tool_outputs,
                        #         model=model,
                        #         tools=tools,
                        #     )
                        # )

                        # # Retrieve the continuation response (continue_chat_with_tool_outputs creates non-streaming response)
                        # self.logger.info(
                        #     f"Retrieving continuation response (iteration {iteration})",
                        #     component="Stream",
                        #     subcomponent="StreamResponseWithToolExecution",
                        #     response_id=current_response_id,
                        # )

                        # # Retrieve the response that was just created
                        # continuation_response = (
                        #     await self.async_client.responses.retrieve(
                        #         current_response_id
                        #     )
                        # )

                        # # Extract text content from the continuation response
                        # text_content = self.chat_manager._extract_text_content(
                        #     continuation_response
                        # )

                        # # Yield the text content in chunks for streaming-like behavior
                        # if text_content:
                        #     # Yield text in reasonable chunks
                        #     chunk_size = 100
                        #     for i in range(0, len(text_content), chunk_size):
                        #         chunk = text_content[i : i + chunk_size]
                        #         if callback:
                        #             callback(chunk)
                        #         print(chunk, end="", flush=True)
                        #         yield {
                        #             "text": chunk,
                        #             "response_id": current_response_id,
                        #             "is_citation": False,
                        #         }

                        # # Extract citations from continuation response
                        # continuation_citations = (
                        #     await self.citation_manager.extract_citations_from_response(
                        #         continuation_response
                        #     )
                        # )

                        # # Yield citations if any
                        # if continuation_citations:
                        #     citation_text = (
                        #         "\n\n"
                        #         + self.citation_manager.format_citations_section(
                        #             continuation_citations
                        #         )
                        #     )
                        #     print(citation_text, end="", flush=True)
                        #     yield {
                        #         "text": citation_text,
                        #         "response_id": current_response_id,
                        #         "is_citation": True,
                        #         "citations": continuation_citations,
                        #     }

                        # # Use this response to check for more tool calls
                        # final_response = continuation_response

                        # Continue chat with tool outputs using STREAMING
                        self.logger.info(
                            "Submitting tool outputs and continuing stream",
                            component="Stream",
                            subcomponent="StreamResponseWithToolExecution",
                            tool_output_count=len(tool_outputs),
                        )

                        # Format tool outputs as function_call_output items
                        input_items = []
                        for tool_output in tool_outputs:
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_output["call_id"],
                                    "output": tool_output["output"],
                                }
                            )

                        # CRITICAL FIX: Create STREAMING response with tool outputs
                        # This enables true streaming instead of manual chunking
                        stream = await self.response_adapter.create_streaming_response(
                            model=model,
                            previous_response_id=current_response_id,
                            input=input_items,  # Tool outputs as structured items
                            instructions=get_system_prompt(),
                            tools=tools or [],
                            stream=True,  # Enable streaming
                        )

                        # Stream the continuation response in real-time
                        self.logger.info(
                            f"Streaming continuation response (iteration {iteration})",
                            component="Stream",
                            subcomponent="StreamResponseWithToolExecution",
                            response_id=current_response_id,
                        )

                        # Process streaming response chunks
                        continuation_response_id = None
                        collected_annotations = []  # Track citations during stream
                        async for chunk in stream:
                            try:
                                chunk_type = type(chunk).__name__

                                # Extract response ID from ResponseCreatedEvent
                                if chunk_type == "ResponseCreatedEvent" and hasattr(
                                    chunk, "response"
                                ):
                                    continuation_response_id = chunk.response.id
                                    if continuation_response_id:
                                        current_response_id = continuation_response_id

                                # Handle ResponseTextDeltaEvent for text content
                                elif chunk_type == "ResponseTextDeltaEvent" and hasattr(
                                    chunk, "delta"
                                ):
                                    text = chunk.delta
                                    if text and text.strip():
                                        if callback:
                                            callback(text)
                                        print(text, end="", flush=True)
                                        yield {
                                            "text": text,
                                            "response_id": current_response_id,
                                            "is_citation": False,
                                        }

                                # CRITICAL: Capture annotation events during streaming (for inline citations)
                                elif (
                                    chunk_type
                                    == "ResponseOutputTextAnnotationAddedEvent"
                                ):
                                    if hasattr(chunk, "annotation"):
                                        collected_annotations.append(chunk.annotation)
                                        self.logger.debug(
                                            "Annotation captured during continuation stream",
                                            component="Stream",
                                            subcomponent="StreamResponseWithToolExecution",
                                            annotation_count=len(collected_annotations),
                                        )

                                # Log file search completion (informational)
                                elif chunk_type == "ResponseFileSearchCallCompleted":
                                    self.logger.info(
                                        "File search call completed in continuation stream",
                                        component="Stream",
                                        subcomponent="StreamResponseWithToolExecution",
                                    )

                            except Exception as e:
                                self.logger.warning(
                                    "Error processing continuation stream chunk",
                                    component="Stream",
                                    subcomponent="StreamResponseWithToolExecution",
                                    error=str(e),
                                )
                                continue

                        # Log collected annotations if any were captured during streaming
                        if collected_annotations:
                            self.logger.info(
                                "Captured annotations during continuation stream",
                                component="Stream",
                                subcomponent="StreamResponseWithToolExecution",
                                annotation_count=len(collected_annotations),
                            )

                        # After streaming completes, retrieve final response for tool call checking
                        if current_response_id:
                            continuation_response = (
                                await self.async_client.responses.retrieve(
                                    current_response_id
                                )
                            )

                            # Extract citations from continuation response
                            continuation_citations = await self.citation_manager.extract_citations_from_response(
                                continuation_response
                            )

                            # Yield citations if any
                            if continuation_citations:
                                citation_text = (
                                    "\n\n"
                                    + self.citation_manager.format_citations_section(
                                        continuation_citations
                                    )
                                )
                                print(citation_text, end="", flush=True)
                                yield {
                                    "text": citation_text,
                                    "response_id": current_response_id,
                                    "is_citation": True,
                                    "citations": continuation_citations,
                                }

                            # Use this response to check for more tool calls
                            final_response = continuation_response
                        else:
                            self.logger.warning(
                                "No response_id from continuation stream",
                                component="Stream",
                                subcomponent="StreamResponseWithToolExecution",
                            )
                            final_response = None

                        # Extract tool calls for next iteration
                        if final_response:
                            tool_calls = (
                                self.chat_manager._extract_tool_calls_from_response(
                                    final_response
                                )
                            )
                        else:
                            tool_calls = []  # No more tool calls if response is None
                            self.logger.warning(
                                "Cannot extract tool calls - final_response is None",
                                component="Stream",
                                subcomponent="StreamResponseWithToolExecution",
                            )

                    # Check if we hit max iterations with pending tool calls
                    if iteration >= max_iterations and tool_calls:
                        self.logger.warning(
                            f"Reached max iterations ({max_iterations}) with pending tool calls",
                            component="Stream",
                            subcomponent="StreamResponseWithToolExecution",
                        )

                    self.logger.info(
                        f"Tool execution phase completed in {iteration} iterations",
                        component="Stream",
                        subcomponent="StreamResponseWithToolExecution",
                        iterations_completed=iteration,
                        tool_execution_count=len(tool_execution_history),
                    )

                except Exception as e:
                    self.logger.error(
                        "Error during tool execution phase",
                        component="Stream",
                        subcomponent="StreamResponseWithToolExecution",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    yield {
                        "error": f"Tool execution error: {str(e)}",
                        "response_id": current_response_id,
                    }

        except Exception as e:
            self.logger.error(
                "Streaming with tool execution error",
                component="Stream",
                subcomponent="StreamResponseWithToolExecution",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise StreamConnectionError(
                f"Failed to stream response with tool execution: {str(e)}"
            )

    @time_execution("Stream", "StreamChatContinuation")
    async def stream_chat_continuation(
        self,
        chat_id: str,
        message: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        enable_tool_execution: bool = False,
        max_iterations: int = 5,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a continuation of an existing chat with optional tool execution.

        Args:
            chat_id: Existing chat ID (previous response ID).
            message: New user message.
            model: Model to use (default: from settings).
            tools: Optional tools to include.
            callback: Optional callback function to process each chunk.
            functions: Optional function definitions for tool execution.
            enable_tool_execution: Whether to execute tool calls (default: False).
            max_iterations: Maximum tool execution iterations (default: 5).

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
                message_length=len(message),
            )

            # Create streaming response with previous_response_id
            stream = await self.response_adapter.create_streaming_response(
                model=model,
                input=message,
                previous_response_id=chat_id,  # Use chat_id as previous_response_id
                instructions=get_system_prompt(),
                tools=tools or [],
                stream=True,  # Enable streaming
            )

            chunk_count = 0
            response_id = None
            collected_annotations = []  # CRITICAL: Collect annotations during stream

            # Process streaming response
            async for chunk in stream:
                try:
                    text = None
                    chunk_type = type(chunk).__name__

                    # Extract response ID from ResponseCreatedEvent
                    if chunk_type == "ResponseCreatedEvent" and hasattr(
                        chunk, "response"
                    ):
                        response_id = chunk.response.id
                        self.logger.info(
                            "Extracted response ID from chat continuation",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                            response_id=response_id,
                        )

                    # Handle ResponseTextDeltaEvent
                    elif chunk_type == "ResponseTextDeltaEvent" and hasattr(
                        chunk, "delta"
                    ):
                        text = chunk.delta
                        if text and text.strip():
                            chunk_count += 1

                            if callback:
                                callback(text)
                            yield {"text": text, "response_id": response_id}

                    # CRITICAL: Capture annotation events
                    elif chunk_type == "ResponseOutputTextAnnotationAddedEvent":
                        if hasattr(chunk, "annotation"):
                            collected_annotations.append(chunk.annotation)
                            self.logger.info(
                                "Annotation captured during chat continuation",
                                component="Stream",
                                subcomponent="StreamChatContinuation",
                                annotation_count=len(collected_annotations),
                            )

                    elif hasattr(chunk, "annotations") and chunk.annotations:
                        collected_annotations.extend(chunk.annotations)

                    else:
                        self.logger.debug(
                            "Other chunk in chat continuation",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                            chunk_type=chunk_type,
                        )

                except Exception as e:
                    self.logger.warning(
                        "Error processing chat continuation chunk",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        chat_id=chat_id,
                        error=str(e),
                    )
                    continue

            # Get final response for citations
            if response_id:
                try:
                    self.logger.info(
                        "Retrieving final response for citations (chat continuation)",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        response_id=response_id,
                    )

                    final_response = await self.async_client.responses.retrieve(
                        response_id=response_id
                    )

                    # Extract citations
                    citations = (
                        await self.citation_manager.extract_citations_from_response(
                            final_response
                        )
                    )

                    self.logger.info(
                        "Citations extracted (chat continuation)",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        citation_count=len(citations),
                    )

                    # Emit citations
                    if citations:
                        citation_text = (
                            "\n\n"
                            + self.citation_manager.format_citations_section(citations)
                        )
                        yield {
                            "text": citation_text,
                            "response_id": response_id,
                            "is_citation": True,
                            "citations": citations,
                        }

                except Exception as e:
                    self.logger.error(
                        "Error retrieving final response for citations (chat continuation)",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        error=str(e),
                    )

            # CRITICAL FIX: Tool execution for continuation responses
            # Check for tool calls and execute them if enabled
            if (
                enable_tool_execution
                and functions
                and response_id
                and self.chat_manager
            ):
                self.logger.info(
                    "Checking for tool calls in continuation response",
                    component="Stream",
                    subcomponent="StreamChatContinuation",
                    response_id=response_id,
                    enable_tool_execution=enable_tool_execution,
                )

                try:
                    # Retrieve final response to check for tool calls
                    final_response = await self.async_client.responses.retrieve(
                        response_id=response_id
                    )

                    # Extract tool calls from response
                    tool_calls = self.chat_manager._extract_tool_calls_from_response(
                        final_response
                    )

                    self.logger.info(
                        "Checking for tool calls after initial continuation stream",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        response_id=response_id,
                        tool_call_count=len(tool_calls),
                    )

                    current_response_id = response_id
                    iteration = 0

                    # Tool execution loop
                    while tool_calls and iteration < max_iterations:
                        iteration += 1
                        self.logger.info(
                            f"Tool execution iteration {iteration}/{max_iterations}",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                            tool_call_count=len(tool_calls),
                        )

                        # Execute all tool calls
                        tool_outputs = []
                        for tool_call in tool_calls:
                            try:
                                call_id = tool_call["call_id"]
                                function_name = tool_call["function_name"]
                                arguments = tool_call["arguments"]

                                self.logger.info(
                                    f"→ Executing Function: {function_name}",
                                    component="Stream",
                                    subcomponent="StreamChatContinuation",
                                    function_name=function_name,
                                    call_id=call_id,
                                    arguments_count=len(arguments),
                                )

                                # Execute function
                                result_str = (
                                    await self.chat_manager._execute_function_call(
                                        function_name, arguments
                                    )
                                )

                                # Collect tool output
                                tool_outputs.append(
                                    {"call_id": call_id, "output": result_str}
                                )

                            except Exception as e:
                                error_msg = f"Error executing tool: {str(e)}"
                                self.logger.error(
                                    error_msg,
                                    component="Stream",
                                    subcomponent="StreamChatContinuation",
                                    exc_info=True,
                                )
                                tool_outputs.append(
                                    {
                                        "call_id": tool_call.get(
                                            "call_id", f"error_{iteration}"
                                        ),
                                        "output": json.dumps({"error": error_msg}),
                                    }
                                )

                        if not tool_outputs:
                            break

                        # Continue chat with tool outputs using STREAMING
                        self.logger.info(
                            "Submitting tool outputs and continuing stream",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                            tool_output_count=len(tool_outputs),
                        )

                        # Format tool outputs as function_call_output items
                        input_items = []
                        for tool_output in tool_outputs:
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_output["call_id"],
                                    "output": tool_output["output"],
                                }
                            )

                        # Create STREAMING response with tool outputs
                        stream = await self.response_adapter.create_streaming_response(
                            model=model,
                            previous_response_id=current_response_id,
                            input=input_items,  # Tool outputs as structured items
                            instructions=get_system_prompt(),
                            tools=tools or [],
                            stream=True,  # Enable streaming
                        )

                        # Stream the continuation response in real-time
                        self.logger.info(
                            f"Streaming continuation response (iteration {iteration})",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                            response_id=current_response_id,
                        )

                        # Process streaming response chunks
                        continuation_response_id = None
                        collected_annotations = []  # Track citations during stream
                        async for chunk in stream:
                            try:
                                chunk_type = type(chunk).__name__

                                # Extract response ID from ResponseCreatedEvent
                                if chunk_type == "ResponseCreatedEvent" and hasattr(
                                    chunk, "response"
                                ):
                                    continuation_response_id = chunk.response.id
                                    if continuation_response_id:
                                        current_response_id = continuation_response_id

                                # Handle ResponseTextDeltaEvent for text content
                                elif chunk_type == "ResponseTextDeltaEvent" and hasattr(
                                    chunk, "delta"
                                ):
                                    text = chunk.delta
                                    if text and text.strip():
                                        chunk_count += 1
                                        if callback:
                                            callback(text)
                                        yield {
                                            "text": text,
                                            "response_id": current_response_id,
                                            "is_citation": False,
                                        }

                                # CRITICAL: Capture annotation events during streaming
                                elif (
                                    chunk_type
                                    == "ResponseOutputTextAnnotationAddedEvent"
                                ):
                                    if hasattr(chunk, "annotation"):
                                        collected_annotations.append(chunk.annotation)
                                        self.logger.debug(
                                            "Annotation captured during continuation stream",
                                            component="Stream",
                                            subcomponent="StreamChatContinuation",
                                            annotation_count=len(collected_annotations),
                                        )

                            except Exception as e:
                                self.logger.warning(
                                    "Error processing continuation stream chunk",
                                    component="Stream",
                                    subcomponent="StreamChatContinuation",
                                    error=str(e),
                                )
                                continue

                        # After streaming completes, retrieve final response for tool call checking
                        if current_response_id:
                            continuation_response = (
                                await self.async_client.responses.retrieve(
                                    current_response_id
                                )
                            )

                            # Extract citations from continuation response
                            continuation_citations = await self.citation_manager.extract_citations_from_response(
                                continuation_response
                            )

                            # Yield citations if any
                            if continuation_citations:
                                citation_text = (
                                    "\n\n"
                                    + self.citation_manager.format_citations_section(
                                        continuation_citations
                                    )
                                )
                                yield {
                                    "text": citation_text,
                                    "response_id": current_response_id,
                                    "is_citation": True,
                                    "citations": continuation_citations,
                                }

                            # Use this response to check for more tool calls
                            final_response = continuation_response
                        else:
                            self.logger.warning(
                                "No response_id from continuation stream",
                                component="Stream",
                                subcomponent="StreamChatContinuation",
                            )
                            final_response = None

                        # Extract tool calls for next iteration
                        if final_response:
                            tool_calls = (
                                self.chat_manager._extract_tool_calls_from_response(
                                    final_response
                                )
                            )
                        else:
                            tool_calls = []  # No more tool calls if response is None
                            self.logger.warning(
                                "Cannot extract tool calls - final_response is None",
                                component="Stream",
                                subcomponent="StreamChatContinuation",
                            )

                    # Check if we hit max iterations with pending tool calls
                    if iteration >= max_iterations and tool_calls:
                        self.logger.warning(
                            f"Reached max iterations ({max_iterations}) with pending tool calls",
                            component="Stream",
                            subcomponent="StreamChatContinuation",
                        )

                    self.logger.info(
                        f"Tool execution phase completed in {iteration} iterations",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        iterations_completed=iteration,
                    )

                except Exception as e:
                    self.logger.error(
                        "Error during tool execution phase in continuation",
                        component="Stream",
                        subcomponent="StreamChatContinuation",
                        error=str(e),
                        error_type=type(e).__name__,
                    )

            # Always yield response_id at the end if we have it
            if response_id:
                yield {
                    "text": "",  # Empty text, but include response_id
                    "response_id": response_id,
                    "is_citation": False,
                }

            self.logger.info(
                "Chat continuation stream completed",
                component="Stream",
                subcomponent="StreamChatContinuation",
                chat_id=chat_id,
                chunk_count=chunk_count,
                response_id=response_id,
            )

        except Exception as e:
            self.logger.error(
                "Chat streaming error",
                component="Stream",
                subcomponent="StreamChatContinuation",
                chat_id=chat_id,
                error=str(e),
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
        tool_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
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
                message_length=len(message),
            )

            # Create streaming response with tools
            stream = await self.response_adapter.create_streaming_response(
                model=model,
                input=message,
                previous_response_id=previous_response_id,
                instructions=get_system_prompt(),
                tools=tools,
                stream=True,  # Enable streaming
            )

            chunk_count = 0
            tool_call_count = 0

            # Process streaming response
            async for chunk in stream:
                try:
                    result = {}
                    chunk_type = type(chunk).__name__

                    # Handle ResponseTextDeltaEvent for text content
                    if chunk_type == "ResponseTextDeltaEvent" and hasattr(
                        chunk, "delta"
                    ):
                        text = chunk.delta
                        if text and text.strip():
                            chunk_count += 1
                            result["text"] = text
                            if callback:
                                callback(text)

                    # Handle tool calls (this would be in different event types)
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        tool_call_count += 1
                        result["tool_calls"] = chunk.tool_calls
                        if tool_callback:
                            tool_callback(chunk.tool_calls)

                        self.logger.info(
                            "Received tool call in stream",
                            component="Stream",
                            subcomponent="StreamWithTools",
                            tool_call_count=tool_call_count,
                        )

                    if result:
                        yield result
                    else:
                        # Log other chunk types for debugging
                        self.logger.debug(
                            "Non-text chunk in stream with tools",
                            component="Stream",
                            subcomponent="StreamWithTools",
                            chunk_type=chunk_type,
                        )

                except Exception as e:
                    self.logger.warning(
                        "Error processing response chunk with tools",
                        component="Stream",
                        subcomponent="StreamWithTools",
                        error=str(e),
                    )
                    # Don't raise - continue processing other chunks
                    continue

            self.logger.info(
                "Response stream with tools completed",
                component="Stream",
                subcomponent="StreamWithTools",
                chunk_count=chunk_count,
                tool_call_count=tool_call_count,
            )

        except Exception as e:
            self.logger.error(
                "Streaming error with tools",
                component="Stream",
                subcomponent="StreamWithTools",
                error=str(e),
            )
            raise StreamConnectionError(
                f"Failed to stream response with tools: {str(e)}"
            )

    @time_execution("Stream", "CreateSSEGenerator")
    async def create_sse_generator(
        self,
        message: str,
        chat_id: Optional[str] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
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
                message_length=len(message),
            )

            chunk_count = 0

            if chat_id:
                # Stream chat continuation
                self.logger.info(
                    "Streaming chat continuation for SSE",
                    component="Stream",
                    subcomponent="CreateSSEGenerator",
                    chat_id=chat_id,
                )

                async for chunk in self.stream_chat_continuation(
                    chat_id, message, model, tools
                ):
                    chunk_count += 1
                    # Format as SSE
                    import json

                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Stream new chat
                self.logger.info(
                    "Streaming new chat for SSE",
                    component="Stream",
                    subcomponent="CreateSSEGenerator",
                )

                async for chunk in self.stream_response(message, model, None, tools):
                    chunk_count += 1
                    # Format as SSE
                    import json

                    yield f"data: {json.dumps(chunk)}\n\n"

            # Signal completion
            self.logger.info(
                "SSE stream completed",
                component="Stream",
                subcomponent="CreateSSEGenerator",
                chunk_count=chunk_count,
            )
            yield "event: close\ndata: [DONE]\n\n"

        except Exception as e:
            self.logger.error(
                "SSE generator error",
                component="Stream",
                subcomponent="CreateSSEGenerator",
                error=str(e),
            )
            yield f"event: error\ndata: {str(e)}\n\n"


# ------------------------------------------------------------
"""
# Streaming with Function Call Tool Execution in OpenAI Responses API

## Overview

The Stream Manager handles real-time streaming responses from the OpenAI Responses API with automatic function call tool execution. This enables true streaming behavior even when the model needs to call functions (like `search_drug_database`) during the conversation flow.

## Function Call Tool Execution During Streaming

### Overview

When streaming is enabled with tool execution (`stream_response_with_tool_execution`), the system automatically:
1. Streams the initial response
2. Detects function calls in the response
3. Executes function calls automatically
4. Streams continuation responses with tool results
5. Repeats until no more tool calls or max iterations reached

### Flow with Function Call Tool Execution

```
User Query (e.g., "What is metformin?")
    ↓
Phase 1: Stream Initial Response
    ↓
[ResponseCreatedEvent] → Capture response_id
    ↓
[ResponseTextDeltaEvent] → Stream text chunks (may be empty if only tool calls)
    ↓
[ResponseCompletedEvent] → Initial stream ends
    ↓
Retrieve final response → Check for tool calls
    ↓
[Function Call Detected] → search_drug_database(drug_class="antidiabetic", drug_name="metformin")
    ↓
Execute Function → Call search_drug_database() → Get results
    ↓
Phase 2: Stream Continuation Response (with tool outputs)
    ↓
[ResponseCreatedEvent] → Capture continuation response_id
    ↓
[ResponseTextDeltaEvent] → Stream text chunks with tool results
    ↓
[ResponseOutputTextAnnotationAddedEvent] → Track citations (if any)
    ↓
[ResponseCompletedEvent] → Continuation stream ends
    ↓
Check for more tool calls → If none, done; if yes, repeat loop
    ↓
Extract citations → Format and emit
    ↓
Done!
```

### Key Phases

#### Phase 1: Initial Response Streaming
- Streams the initial user query
- May contain tool calls (function calls) or text
- Captures `response_id` from `ResponseCreatedEvent`
- Always yields `response_id` even if no text chunks (for tool call detection)

#### Phase 2: Tool Execution Loop
- Detects tool calls from initial response
- Executes each function call (e.g., `search_drug_database`)
- Formats tool outputs as `function_call_output` items
- Creates **streaming** response with tool outputs (not non-streaming)
- Streams continuation response chunk-by-chunk
- Checks for additional tool calls
- Repeats until no tool calls or max iterations (default: 5)

### Function Call Tool Execution Details

#### Tool Call Detection
```python
# After initial stream completes
final_response = await self.async_client.responses.retrieve(response_id)
tool_calls = self.chat_manager._extract_tool_calls_from_response(final_response)

# Tool calls structure:
# [
#   {
#     "call_id": "call_abc123",
#     "type": "function_call",
#     "function_name": "search_drug_database",
#     "arguments": {"drug_class": "antidiabetic", "drug_name": "metformin", "limit": 5}
#   }
# ]
```

#### Tool Execution
```python
# Execute each function call
for tool_call in tool_calls:
    function_name = tool_call["function_name"]
    arguments = tool_call["arguments"]
    result = await self.chat_manager._execute_function_call(function_name, arguments)
    
    # Format as function_call_output
    tool_outputs.append({
        "call_id": tool_call["call_id"],
        "output": result
    })
```

#### Streaming Continuation Response
```python
# Format tool outputs as structured items
input_items = []
for tool_output in tool_outputs:
    input_items.append({
        "type": "function_call_output",
        "call_id": tool_output["call_id"],
        "output": tool_output["output"]
    })

# Create STREAMING response with tool outputs
stream = await self.response_adapter.create_streaming_response(
    model=model,
    previous_response_id=current_response_id,
    input=input_items,  # Tool outputs as structured items
    instructions=get_system_prompt(),
    tools=tools or [],
    stream=True,  # CRITICAL: Enable streaming
)

# Process streaming chunks in real-time
async for chunk in stream:
    # Handle ResponseTextDeltaEvent for text chunks
    # Handle ResponseCreatedEvent for response_id
    # Handle ResponseOutputTextAnnotationAddedEvent for citations
    yield chunk
```

### Tool Types Supported

#### 1. Function Calls (Custom Functions)
- **Example**: `search_drug_database`
- **Execution**: Custom function executor registered with `ChatManager`
- **Output**: Structured data (e.g., drug information from Milvus database)
- **Streaming**: Continuation response streams after tool execution

#### 2. File Search (Built-in Tool)
- **Example**: Vector store search for clinical guidelines
- **Execution**: Handled automatically by OpenAI Responses API
- **Output**: File citations and content
- **Streaming**: File search results are embedded in stream events

### Event Handling During Tool Execution

#### Initial Stream Events
- `ResponseCreatedEvent`: Capture initial `response_id`
- `ResponseTextDeltaEvent`: Stream text (may be empty if only tool calls)
- `ResponseCompletedEvent`: Signal end of initial stream

#### Continuation Stream Events (After Tool Execution)
- `ResponseCreatedEvent`: Capture continuation `response_id`
- `ResponseTextDeltaEvent`: Stream text chunks with tool results
- `ResponseOutputTextAnnotationAddedEvent`: Track inline citations
- `ResponseFileSearchCallCompleted`: File search completion (if used)
- `ResponseCompletedEvent`: Signal end of continuation stream

### Critical Implementation Details

#### 1. Response ID Management
- Always capture `response_id` from `ResponseCreatedEvent`
- Update `current_response_id` when continuation stream starts
- Use `current_response_id` for retrieving final response and checking for more tool calls

#### 2. Tool Output Formatting
- Tool outputs must be formatted as `function_call_output` items
- Each item requires:
  - `type`: `"function_call_output"`
  - `call_id`: Matching the original tool call ID
  - `output`: Result string from function execution

#### 3. Streaming vs Non-Streaming
- **Initial stream**: Always uses `create_streaming_response`
- **Continuation stream**: Must use `create_streaming_response` (not `create_response`)
- This ensures true chunk-by-chunk streaming, not manual text slicing

#### 4. Tool Call Detection
- After each stream completes, retrieve final response
- Extract tool calls using `_extract_tool_calls_from_response()`
- Check for `type == "function_call"` to identify function calls
- Loop continues while `tool_calls` exist and `iteration < max_iterations`

### Error Handling

#### Null Response Handling
```python
# Always check for None before extracting tool calls
if final_response:
    tool_calls = self.chat_manager._extract_tool_calls_from_response(final_response)
else:
    tool_calls = []  # No more tool calls
    self.logger.warning("Cannot extract tool calls - final_response is None")
```

#### Function Execution Errors
- Errors during function execution are caught and logged
- Error messages are formatted as tool outputs
- Tool execution loop continues with error outputs
- Prevents single function failure from breaking entire stream

#### Stream Processing Errors
- Individual chunk processing errors are caught and logged
- Stream continues processing other chunks
- Errors don't break the entire streaming flow

### Best Practices

#### 1. Tool Execution Configuration
- Set `enable_tool_execution=True` when initializing streaming
- Provide `function_definitions` for function tools
- Set appropriate `max_iterations` (default: 5) to prevent infinite loops

#### 2. Streaming Performance
- Use true streaming (not manual chunking) for continuation responses
- Process chunks as they arrive for real-time user experience
- Track response_id for proper conversation chaining

#### 3. Tool Call Detection
- Always retrieve final response after stream completes
- Check for tool calls even if no text was streamed
- Handle both function calls and file_search tool calls

#### 4. Response ID Tracking
- Capture response_id from `ResponseCreatedEvent` in both phases
- Update `current_response_id` when continuation stream starts
- Use `current_response_id` for retrieving final response

### Example Flow: Drug Database Query

```
User: "What is the formulation and available dosages for metformin?"
    ↓
Phase 1: Initial Stream
    - Stream starts (tools: file_search, search_drug_database)
    - Model makes tool call: search_drug_database(drug_class="antidiabetic", drug_name="metformin")
    - Stream completes with 0 text chunks (only tool call)
    ↓
Phase 2: Tool Execution
    - Detect tool call: search_drug_database
    - Execute function: Query Milvus database → Get 5 results
    - Format tool output: function_call_output with call_id and results
    ↓
Phase 3: Continuation Stream
    - Create streaming response with tool outputs
    - Stream text chunks: "Metformin is available in various formulations..."
    - Stream completes with full response
    ↓
Check for more tool calls: None → Done
Extract citations: None → Done
Final output: Complete response with metformin information
```

### Debugging Tips

#### Common Issues
1. **No tool execution**: Check if `enable_tool_execution=True` and `function_definitions` provided
2. **No streaming chunks**: Verify `stream=True` in `create_streaming_response`
3. **Tool execution fails**: Check function executor registration and error logs
4. **Infinite loop**: Check `max_iterations` limit and tool call detection logic

#### Debug Logging
```python
self.logger.info(
    "Tool execution iteration",
    component="Stream",
    subcomponent="StreamResponseWithToolExecution",
    iteration=iteration,
    tool_call_count=len(tool_calls),
    function_names=[tc.get("function_name") for tc in tool_calls]
)
```

---

# Streaming Citations in OpenAI Responses API

## Overview

The Stream Manager handles real-time streaming responses from the OpenAI Responses API with proper citation handling. This documentation explains how streaming works and best practices for implementation.

## Core Concepts

### Basic Flow
```
User Query
    ↓
Start Streaming (stream=True)
    ↓
[ResponseCreatedEvent] → Capture response_id
    ↓
[ResponseTextDeltaEvent] → Stream text chunks to user
    ↓
[ResponseOutputTextAnnotationAddedEvent] → Track citations
    ↓
[ResponseFileSearchCallCompleted] → File search done
    ↓
[ResponseCompletedEvent] → Stream ends
    ↓
Call responses.retrieve(response_id) → Get complete response
    ↓
Extract annotations from response.output[1].content[0].annotations
    ↓
Format citations
    ↓
Emit citation chunk
    ↓
Done!
```

## Stream Event Types

During streaming, the Responses API emits these events in a specific sequence. Each event serves a distinct purpose in the streaming response lifecycle:

| Event Type | Contains | When Emitted | Purpose & Data Structure |
|------------|----------|--------------|---------------------------|
| **ResponseCreatedEvent** | `response.id` | Once at start | **Response Initialization**: Signals the start of a new response. Contains the unique response ID needed for retrieving the complete response after streaming ends. This is the first event emitted and is crucial for citation handling. |
| **ResponseTextDeltaEvent** | Text chunks (`delta`) | Multiple times during generation | **Content Streaming**: The core streaming mechanism. Each event contains a `delta` field with incremental text chunks as the model generates the response. These are emitted token-by-token for real-time display to users. |
| **ResponseOutputTextAnnotationAddedEvent** | Annotation objects | When citations appear in text | **Citation Markers**: Emitted when the model references external sources (like files from file_search). Contains annotation objects with metadata about the referenced documents. These are emitted inline as citation markers like `[1]` appear in the generated text. |
| **ResponseFileSearchCallCompleted** | Search completion status | After file search tool execution | **Tool Completion Signal**: Indicates that a file_search tool call has finished executing. This happens when the model uses the file_search tool to retrieve information from uploaded documents. |
| **ResponseCompletedEvent** | Final status | Once at end | **Stream Termination**: The final event indicating the response generation is complete. After this event, no more content will be streamed, signaling the end of the real-time response phase. |

### Event Sequence and Data Flow

The events follow this precise chronological order:

```
1. ResponseCreatedEvent
   → Capture response_id (critical for citations!)

2. ResponseTextDeltaEvent (repeated)
   → Stream text chunks to user in real-time
   → May interleave with annotation events

3. ResponseOutputTextAnnotationAddedEvent (when citations occur)
   → Track that citations exist in the response
   → Store annotation data for later processing

4. ResponseFileSearchCallCompleted (if file_search was used)
   → Confirm external document search completed

5. ResponseCompletedEvent
   → End of streaming phase
   → Time to retrieve complete response for citations
```

### Critical Event Details

#### ResponseCreatedEvent
```python
# Example structure (based on OpenAI API)
{
    "type": "response.created",
    "response": {
        "id": "resp_abc1234567890",
        "created_at": 1234567890.0,
        "model": "gpt-4o-mini",
        "status": "in_progress"
    }
}
```
**Key Field**: `response.id` - This ID is essential for retrieving the complete response after streaming ends.

#### ResponseTextDeltaEvent
```python
# Example structure
{
    "type": "response.output_text.delta",
    "delta": "According to the clinical guidelines",
    "content_index": 0
}
```
**Key Field**: `delta` - Contains the actual text content to display to the user.

#### ResponseOutputTextAnnotationAddedEvent
```python
# Example structure
{
    "type": "response.output_text.annotation.added",
    "annotation": {
        "type": "file_citation",
        "file_citation": {
            "file_id": "file_abc123",
            "filename": "clinical_guidelines.pdf",
            "quote": "relevant text from the document"
        },
        "index": 1
    },
    "content_index": 0
}
```
**Key Field**: `annotation` - Contains citation metadata including filename and quote from the referenced document.

#### ResponseFileSearchCallCompleted
```python
# Example structure
{
    "type": "response.file_search_call.completed",
    "status": "completed",
    "file_search_call": {
        "id": "call_abc123",
        "status": "completed"
    }
}
```
**Key Field**: `status` - Confirms the file search operation completed successfully.

#### ResponseCompletedEvent
```python
# Example structure
{
    "type": "response.completed",
    "response": {
        "id": "resp_abc1234567890",
        "status": "completed",
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 200,
            "total_tokens": 350
        }
    }
}
```
**Key Field**: `response.status` - Should be "completed" indicating successful generation.

### Why These Events Matter for Citation Handling

The streaming events are specifically designed to handle the complexity of real-time citation management:

1. **ResponseCreatedEvent** → **Critical Foundation**
   - Without capturing `response.id` here, you cannot retrieve the complete response later
   - This is the single point where the response ID becomes available
   - Missing this event breaks the entire citation retrieval process

2. **ResponseTextDeltaEvent** → **Real-time User Experience**
   - Provides immediate text feedback to users
   - Maintains responsive UI during long responses
   - Can be displayed incrementally as content is generated

3. **ResponseOutputTextAnnotationAddedEvent** → **Citation Detection**
   - Real-time indication that citations exist in the response
   - Provides annotation metadata as citations are generated
   - Links text content with source documents
   - Enables inline citation markers (like `[1]`) to appear with the text

4. **ResponseFileSearchCallCompleted** → **Tool Execution Confirmation**
   - Confirms that file_search tool executed successfully
   - Indicates that external document content was retrieved
   - Helps predict whether citations will be present in the final response

5. **ResponseCompletedEvent** → **Citation Retrieval Trigger**
   - Signals that streaming phase is complete
   - Indicates it's safe to retrieve the complete response
   - Marks the transition from streaming to citation extraction

### Event Timing and Dependencies

The events have strict timing dependencies:

- **ResponseCreatedEvent** must be captured before any other events
- **ResponseTextDeltaEvent** and **ResponseOutputTextAnnotationAddedEvent** can interleave
- **ResponseFileSearchCallCompleted** only occurs if file_search tool is used
- **ResponseCompletedEvent** is always the final event in the sequence

### Error Handling Considerations

Each event type requires specific error handling:

- **ResponseCreatedEvent**: If missing, streaming cannot proceed (fatal error)
- **ResponseTextDeltaEvent**: Can be skipped if text processing fails (non-fatal)
- **ResponseOutputTextAnnotationAddedEvent**: Can be ignored if citation processing fails (non-fatal)
- **ResponseFileSearchCallCompleted**: Indicates tool success/failure (informational)
- **ResponseCompletedEvent**: If missing, may indicate streaming error (potential issue)

## Implementation Guide

### 1. Capture Response ID During Streaming
```python
if chunk_type == "ResponseCreatedEvent" and hasattr(chunk, 'response'):
    response_id = chunk.response.id  # Save this!
```

### 2. Stream Text in Real-time
```python
elif chunk_type == "ResponseTextDeltaEvent" and hasattr(chunk, 'delta'):
    text = chunk.delta
    if text and text.strip():
        yield {
            "text": text,
            "response_id": response_id
        }
```

### 3. After Stream Ends, Get Complete Response
```python
final_response = await self.async_client.responses.retrieve(response_id)
citations = await self.citation_manager.extract_citations_from_response(final_response)
```

### 4. Emit Citations as Final Chunk
```python
if citations:
    citation_text = "\\n\\n" + self.citation_manager.format_citations_section(citations)
    yield {
        "text": citation_text,
        "response_id": response_id,
        "is_citation": True
    }
```

## Best Practices

### Response ID Management
- Always capture response_id from ResponseCreatedEvent
- Store it for the duration of the stream
- Use it to retrieve the complete response after streaming

### Citation Handling
- Don't try to extract complete citations during streaming
- Wait for stream completion
- Use responses.retrieve() to get full response
- Extract citations from complete response object

### Error Handling
- Wrap streaming in try-except blocks
- Log errors with appropriate context
- Continue processing other chunks on error
- Provide meaningful error messages

### Performance Tips
- Only retrieve final response if citations are expected
- Consider conditional retrieval based on tool usage
- Log key events for debugging
- Use appropriate chunk processing


## Debugging Tips

### Common Issues
1. Response ID not captured
2. Final response not retrieved
3. Citations not extracted
4. Citations not emitted

### Debug Logging
```python
self.logger.info(
    "Stream status",
    component="Stream",
    subcomponent="StreamResponse",
    response_id=response_id,
    chunk_count=chunk_count,
    citation_count=len(citations)
)
```

## Summary

The Stream Manager provides robust handling of streaming responses with proper citation support. Key points:

1. Capture response_id early
2. Stream text chunks in real-time
3. Retrieve complete response after stream
4. Extract and emit citations as final chunk
5. Handle errors gracefully
6. Log key events for debugging

For detailed implementation, refer to the stream_response() and stream_chat_continuation() methods in this class.
"""
# ------------------------------------------------------------
