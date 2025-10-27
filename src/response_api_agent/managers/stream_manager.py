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

from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from openai import OpenAI, AsyncOpenAI
from src.config import get_settings
from src.response_api_agent.managers.exceptions import StreamConnectionError
from src.response_api_agent.managers.citation_manager import CitationManager
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
        self.citation_manager = CitationManager(client=self.async_client)
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
            )

            # Create streaming response
            stream = await self.async_client.responses.create(
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

    @time_execution("Stream", "StreamChatContinuation")
    async def stream_chat_continuation(
        self,
        chat_id: str,
        message: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str], None]] = None,
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
                message_length=len(message),
            )

            # Create streaming response with previous_response_id
            stream = await self.async_client.responses.create(
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

            self.logger.info(
                "Chat continuation stream completed",
                component="Stream",
                subcomponent="StreamChatContinuation",
                chat_id=chat_id,
                chunk_count=chunk_count,
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
            stream = await self.async_client.responses.create(
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
