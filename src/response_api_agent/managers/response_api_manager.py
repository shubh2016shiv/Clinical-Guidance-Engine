"""
OpenAI Response API Manager

Enterprise-grade manager for orchestrating OpenAI Response API workflows including:
- Vector store management for semantic search
- Tool configuration and validation
- Conversation management with history
- Streaming and non-streaming responses
- Resource lifecycle management

This manager provides a unified interface for all Response API operations with
robust error handling, logging, and resource cleanup capabilities.
"""

import asyncio
from typing import Dict, Any, List, Optional, AsyncGenerator
from src.config import get_settings
from src.response_api_agent.managers.vector_store_manager import VectorStoreManager
from src.response_api_agent.managers.tool_manager import ToolManager
from src.response_api_agent.managers.chat_manager import ChatManager
from src.response_api_agent.managers.stream_manager import StreamManager
from src.response_api_agent.managers.exceptions import (
    ResponsesAPIError,
    VectorStoreError,
    ToolConfigurationError,
    StreamConnectionError
)
from src.logs import get_component_logger, time_execution


class OpenAIResponseManager:
    """
    Enterprise-grade manager for OpenAI Response API operations.
    
    Coordinates vector stores, tools, conversations, and streaming responses
    with comprehensive error handling, validation, and resource management.
    
    Attributes:
        vector_store_manager: Manages vector stores for semantic search
        tool_manager: Handles tool configuration and validation
        chat_manager: Manages conversation state and history
        stream_manager: Handles streaming response generation
        settings: Application configuration settings
        logger: Component-specific logger
    """

    # Configuration Constants
    DEFAULT_CHAT_HISTORY_LIMIT = 10
    DEFAULT_BATCH_SIZE = 5
    DEFAULT_RATE_LIMIT_DELAY = 1.0
    VECTOR_STORE_POLL_INTERVAL_SECONDS = 10
    VECTOR_STORE_MAX_WAIT_SECONDS = 300
    VALID_MODEL_PREFIXES = ["gpt-3.5", "gpt-4", "o1", "o3"]

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        chat_history_limit: int = DEFAULT_CHAT_HISTORY_LIMIT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY
    ):
        """
        Initialize the OpenAI Response Manager.
        
        Args:
            vector_store_manager: Pre-configured VectorStoreManager instance.
                If None, a new instance will be created.
            chat_history_limit: Maximum number of messages to retain in chat history.
            batch_size: Batch size for vector store operations.
            rate_limit_delay: Delay in seconds between rate-limited operations.
        """
        self.settings = get_settings()
        self.logger = get_component_logger("OpenAIResponseManager")
        
        # Initialize or use provided vector store manager
        if vector_store_manager:
            self.vector_store_manager = vector_store_manager
        else:
            self.vector_store_manager = VectorStoreManager(
                batch_size=batch_size,
                rate_limit_delay=rate_limit_delay
            )
        
        # Initialize dependent managers
        self.tool_manager = ToolManager(self.vector_store_manager)
        self.chat_manager = ChatManager(
            tool_manager=self.tool_manager,
            chat_history_limit=chat_history_limit
        )
        self.stream_manager = StreamManager()
        
        self.logger.info(
            "OpenAI Response Manager initialized",
            component="OpenAIResponseManager",
            subcomponent="Init",
            chat_history_limit=chat_history_limit,
            batch_size=batch_size
        )

    def _validate_model_identifier(self, model_name: str) -> bool:
        """
        Validate that model identifier follows OpenAI naming conventions.
        
        Args:
            model_name: Model identifier to validate.
            
        Returns:
            True if model name is valid, False otherwise.
        """
        return any(
            model_name.startswith(prefix)
            for prefix in self.VALID_MODEL_PREFIXES
        )

    def _resolve_model_name(self, model_override: Optional[str] = None) -> str:
        """
        Resolve the model name to use, with validation.
        
        Args:
            model_override: Optional model name override. If None, uses default from settings.
            
        Returns:
            Resolved model name.
        """
        model_name = model_override or self.settings.openai_model_name
        
        if not self._validate_model_identifier(model_name):
            self.logger.warning(
                "Model identifier may not follow OpenAI conventions",
                component="OpenAIResponseManager",
                subcomponent="_resolve_model_name",
                model=model_name
            )
        
        return model_name

    async def _validate_and_prepare_vector_store(
        self, 
        vector_store_id: Optional[str]
    ) -> Optional[str]:
        """
        Validate vector store readiness and return ID if ready.
        
        Args:
            vector_store_id: Vector store ID to validate.
            
        Returns:
            Vector store ID if ready, None if not ready or validation fails.
        """
        if not vector_store_id:
            return None
        
        try:
            store_info = await self.vector_store_manager.get_vector_store(vector_store_id)
            
            if not store_info:
                self.logger.warning(
                    "Vector store not found",
                    component="OpenAIResponseManager",
                    subcomponent="_validate_and_prepare_vector_store",
                    vector_store_id=vector_store_id
                )
                return None
            
            if store_info["status"] != "completed":
                self.logger.warning(
                    "Vector store not in completed state",
                    component="OpenAIResponseManager",
                    subcomponent="_validate_and_prepare_vector_store",
                    vector_store_id=vector_store_id,
                    status=store_info["status"]
                )
                return None
            
            self.logger.info(
                "Vector store validated and ready",
                component="OpenAIResponseManager",
                subcomponent="_validate_and_prepare_vector_store",
                vector_store_id=vector_store_id
            )
            return vector_store_id
            
        except Exception as e:
            self.logger.error(
                "Error validating vector store",
                component="OpenAIResponseManager",
                subcomponent="_validate_and_prepare_vector_store",
                vector_store_id=vector_store_id,
                error=str(e)
            )
            return None

    async def _prepare_tools_configuration(
        self,
        vector_store_id: Optional[str],
        function_definitions: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare and validate tools configuration for the request.
        
        Args:
            vector_store_id: Optional vector store ID for file_search tool.
            function_definitions: Optional custom function definitions.
            
        Returns:
            List of validated tool configurations, or empty list on failure.
        """
        if not vector_store_id and not function_definitions:
            return []
        
        # Validate vector store before creating tools
        validated_vector_store_id = await self._validate_and_prepare_vector_store(
            vector_store_id
        )
        
        if not validated_vector_store_id and not function_definitions:
            return []
        
        try:
            tools = await self.tool_manager.get_tools_for_response(
                vector_store_id=validated_vector_store_id,
                functions=function_definitions
            )
            
            is_valid = await self.tool_manager.validate_tools(tools)
            if not is_valid:
                self.logger.warning(
                    "Tool validation failed, proceeding without tools",
                    component="OpenAIResponseManager",
                    subcomponent="_prepare_tools_configuration"
                )
                return []
            
            self.logger.info(
                "Tools prepared and validated successfully",
                component="OpenAIResponseManager",
                subcomponent="_prepare_tools_configuration",
                tool_count=len(tools)
            )
            return tools
            
        except Exception as e:
            self.logger.warning(
                "Error preparing tools, proceeding without tools",
                component="OpenAIResponseManager",
                subcomponent="_prepare_tools_configuration",
                error=str(e)
            )
            return []

    async def _extract_response_content(
        self, 
        response: Any
    ) -> tuple[str, List[Any]]:
        """
        Extract text content and tool calls from response object.
        
        Args:
            response: OpenAI API response object.
            
        Returns:
            Tuple of (content_text, tool_calls_list).
        """
        content = self.chat_manager._extract_text_content(response)
        
        # Extract tool calls with fallback for different response formats
        tool_calls = []
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and 'call' in item.type:
                    tool_calls.append(item)
        else:
            tool_calls = getattr(response, 'tool_calls', [])
        
        return content, tool_calls

    @time_execution("OpenAIResponseManager", "process_query")
    async def process_query(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        function_definitions: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        enable_streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user query with optional tools and streaming.
        
        This is the primary method for handling user queries. It supports:
        - New conversations and continuation of existing ones
        - Vector store integration for semantic search
        - Custom function tools
        - Streaming and non-streaming responses
        
        Args:
            user_message: The user's message/query text.
            conversation_id: Optional ID to continue an existing conversation.
            vector_store_id: Optional vector store ID for file_search capability.
            function_definitions: Optional list of custom function definitions.
            model_name: Optional model override. Uses default if not specified.
            enable_streaming: If True, returns a streaming generator function.
        
        Returns:
            Dictionary containing:
                - conversation_id: ID for the conversation
                - content: Response text (if not streaming)
                - stream_generator: Async generator function (if streaming)
                - tool_calls: List of tool calls made
                - tools: Tools configuration used
        
        Raises:
            ResponsesAPIError: On general query processing failures.
            ToolConfigurationError: On tool configuration issues.
            VectorStoreError: On vector store access issues.
        """
        try:
            self.logger.info(
                "Processing user query",
                component="OpenAIResponseManager",
                subcomponent="process_query",
                message_length=len(user_message),
                has_conversation_id=bool(conversation_id),
                has_vector_store=bool(vector_store_id),
                has_functions=bool(function_definitions),
                streaming_enabled=enable_streaming
            )
            
            # Resolve model configuration
            resolved_model = self._resolve_model_name(model_name)
            self.logger.info(
                "Using model configuration",
                component="OpenAIResponseManager",
                subcomponent="process_query",
                model=resolved_model
            )
            
            # Prepare tools if needed
            tools = await self._prepare_tools_configuration(
                vector_store_id, 
                function_definitions
            )
            
            if enable_streaming:
                return await self._handle_streaming_query(
                    user_message, conversation_id, resolved_model, tools
                )
            else:
                return await self._handle_standard_query(
                    user_message, conversation_id, resolved_model, tools
                )
                
        except (ToolConfigurationError, VectorStoreError) as e:
            self.logger.error(
                "Tool or vector store error in query processing",
                component="OpenAIResponseManager",
                subcomponent="process_query",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to process query",
                component="OpenAIResponseManager",
                subcomponent="process_query",
                error=str(e)
            )
            raise ResponsesAPIError(message=f"Query processing failed: {str(e)}")

    async def _handle_streaming_query(
        self,
        user_message: str,
        conversation_id: Optional[str],
        model_name: str,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle a streaming query request.
        
        Args:
            user_message: User's message text.
            conversation_id: Optional conversation ID.
            model_name: Resolved model name.
            tools: Prepared tools configuration.
            
        Returns:
            Dictionary with conversation_id, stream_generator function, and tools.
        """
        self.logger.info(
            "Setting up streaming response",
            component="OpenAIResponseManager",
            subcomponent="_handle_streaming_query",
            has_conversation_id=bool(conversation_id)
        )
        
        async def create_stream_generator() -> AsyncGenerator[str, None]:
            async for chunk in self.stream_manager.stream_response(
                message=user_message, 
                model=model_name, 
                tools=tools
            ):
                yield chunk
        
        # Create or use existing conversation
        if not conversation_id:
            conversation_id = await self.chat_manager.create_chat(
                message=user_message, 
                model=model_name, 
                tools=tools
            )
        
        self.logger.info(
            "Streaming response configured",
            component="OpenAIResponseManager",
            subcomponent="_handle_streaming_query",
            conversation_id=conversation_id
        )
        
        return {
            "conversation_id": conversation_id,
            "stream_generator": create_stream_generator,
            "tools": tools
        }

    async def _handle_standard_query(
        self,
        user_message: str,
        conversation_id: Optional[str],
        model_name: str,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle a standard (non-streaming) query request.
        
        Args:
            user_message: User's message text.
            conversation_id: Optional conversation ID.
            model_name: Resolved model name.
            tools: Prepared tools configuration.
            
        Returns:
            Dictionary with conversation_id, content, tool_calls, and tools.
        """
        if conversation_id:
            # Continue existing conversation
            self.logger.info(
                "Continuing existing conversation",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query",
                conversation_id=conversation_id
            )
            
            result = await self.chat_manager.continue_chat_with_tools(
                chat_id=conversation_id,
                message=user_message,
                vector_store_id=None,  # Already included in tools
                functions=None,  # Already included in tools
                model=model_name
            )
            
            self.logger.info(
                "Conversation continued successfully",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query",
                conversation_id=conversation_id,
                has_tool_calls=bool(result.get("tool_calls", []))
            )
            
            return {
                "conversation_id": conversation_id,
                "content": result["content"],
                "tool_calls": result["tool_calls"],
                "tools": tools
            }
        else:
            # Start new conversation
            self.logger.info(
                "Starting new conversation",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query"
            )
            
            conversation_id = await self.chat_manager.create_chat(
                message=user_message, 
                model=model_name, 
                tools=tools
            )
            
            # Retrieve and parse response
            response = await asyncio.to_thread(
                self.chat_manager.client.responses.retrieve,
                response_id=conversation_id
            )
            content, tool_calls = await self._extract_response_content(response)
            
            self.logger.info(
                "New conversation created successfully",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query",
                conversation_id=conversation_id,
                has_tool_calls=bool(tool_calls)
            )
            
            return {
                "conversation_id": conversation_id,
                "content": content,
                "tool_calls": tool_calls,
                "tools": tools
            }

    @time_execution("OpenAIResponseManager", "process_streaming_query")
    async def process_streaming_query(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        function_definitions: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query with streaming response and conversation history.
        
        This method yields chunks of the response as they arrive, making it
        suitable for real-time user interfaces.
        
        Args:
            user_message: The user's message/query text.
            conversation_id: Optional ID to continue an existing conversation.
            vector_store_id: Optional vector store ID for file_search capability.
            function_definitions: Optional list of custom function definitions.
            model_name: Optional model override. Uses default if not specified.
        
        Yields:
            Dictionary containing:
                - chunk: Text chunk from the stream
                - conversation_id: ID for the conversation
                - tools: Tools configuration used
                - error: Error message (only if error occurs)
        """
        try:
            self.logger.info(
                "Processing streaming query",
                component="OpenAIResponseManager",
                subcomponent="process_streaming_query",
                message_length=len(user_message),
                has_conversation_id=bool(conversation_id),
                has_vector_store=bool(vector_store_id),
                has_functions=bool(function_definitions)
            )
            
            # Resolve model configuration
            resolved_model = self._resolve_model_name(model_name)
            self.logger.info(
                "Using model configuration",
                component="OpenAIResponseManager",
                subcomponent="process_streaming_query",
                model=resolved_model
            )
            
            # Prepare tools
            tools = await self._prepare_tools_configuration(
                vector_store_id,
                function_definitions
            )
            
            chunk_count = 0
            
            if conversation_id:
                # Stream continuation of existing conversation
                self.logger.info(
                    "Streaming conversation continuation",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                    conversation_id=conversation_id
                )
                
                async for chunk_data in self.stream_manager.stream_chat_continuation(
                    chat_id=conversation_id,
                    message=user_message,
                    model=resolved_model,
                    tools=tools
                ):
                    chunk_count += 1
                    yield {
                        "chunk": chunk_data.get("text", ""),
                        "conversation_id": conversation_id,
                        "tools": tools
                    }
                
                self.logger.info(
                    "Conversation continuation streaming completed",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                    conversation_id=conversation_id,
                    chunk_count=chunk_count
                )
            else:
                # CRITICAL FIX: Stream directly without creating chat first
                self.logger.info(
                    "Starting new streaming conversation",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query"
                )
                
                # Stream the response directly - no create_chat call
                response_id = None
                async for chunk_data in self.stream_manager.stream_response(
                    message=user_message,
                    model=resolved_model,
                    tools=tools
                ):
                    chunk_count += 1
                    
                    # Extract response_id from first chunk if available
                    if response_id is None and chunk_data.get("response_id"):
                        response_id = chunk_data["response_id"]
                    
                    yield {
                        "chunk": chunk_data.get("text", ""),
                        "conversation_id": response_id,  # Use extracted response_id instead of None
                        "tools": tools
                    }
                
                self.logger.info(
                    "New conversation streaming completed",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                    chunk_count=chunk_count
                )
                    
        except (ToolConfigurationError, VectorStoreError, StreamConnectionError) as e:
            self.logger.error(
                "Error in streaming query processing",
                component="OpenAIResponseManager",
                subcomponent="process_streaming_query",
                error=str(e),
                error_type=type(e).__name__
            )
            yield {
                "error": str(e),
                "conversation_id": conversation_id,
                "tools": []
            }
        except Exception as e:
            self.logger.error(
                "Failed to process streaming query",
                component="OpenAIResponseManager",
                subcomponent="process_streaming_query",
                error=str(e)
            )
            yield {
                "error": f"Streaming query processing failed: {str(e)}",
                "conversation_id": conversation_id,
                "tools": []
            }

    @time_execution("OpenAIResponseManager", "retrieve_conversation_history")
    async def retrieve_conversation_history(
        self, 
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the complete history for a conversation.
        
        Args:
            conversation_id: The conversation ID to retrieve history for.
        
        Returns:
            List of message dictionaries containing content and tool calls.
        
        Raises:
            ResponsesAPIError: If history retrieval fails.
        """
        try:
            self.logger.info(
                "Retrieving conversation history",
                component="OpenAIResponseManager",
                subcomponent="retrieve_conversation_history",
                conversation_id=conversation_id
            )
            
            history = await self.chat_manager.get_chat_history(conversation_id)
            
            self.logger.info(
                "Conversation history retrieved successfully",
                component="OpenAIResponseManager",
                subcomponent="retrieve_conversation_history",
                conversation_id=conversation_id,
                message_count=len(history)
            )
            
            return history
        except Exception as e:
            self.logger.error(
                "Failed to retrieve conversation history",
                component="OpenAIResponseManager",
                subcomponent="retrieve_conversation_history",
                conversation_id=conversation_id,
                error=str(e)
            )
            raise ResponsesAPIError(
                message=f"History retrieval failed: {str(e)}"
            )

    @time_execution("OpenAIResponseManager", "cleanup_resources")
    async def cleanup_resources(
        self,
        vector_store_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        clear_all_caches: bool = True
    ) -> Dict[str, bool]:
        """
        Clean up resources and optionally clear caches.
        
        This method should be called when resources are no longer needed to
        prevent resource leaks and manage costs.
        
        Args:
            vector_store_id: Optional vector store ID to delete.
            conversation_id: Optional conversation ID to delete.
            clear_all_caches: If True, clears all manager caches.
        
        Returns:
            Dictionary with cleanup status for each resource:
                - vector_store: True if deleted successfully
                - conversation: True if deleted successfully
                - caches_cleared: True if caches were cleared
                - error: Error message if cleanup failed
        """
        try:
            self.logger.info(
                "Starting resource cleanup",
                component="OpenAIResponseManager",
                subcomponent="cleanup_resources",
                has_vector_store=bool(vector_store_id),
                has_conversation=bool(conversation_id),
                clear_caches=clear_all_caches
            )
            
            cleanup_status = {}
            
            # Clean up vector store
            if vector_store_id:
                try:
                    cleanup_status["vector_store"] = (
                        await self.vector_store_manager.delete_vector_store(
                            vector_store_id
                        )
                    )
                    self.logger.info(
                        "Vector store cleanup completed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        vector_store_id=vector_store_id,
                        success=cleanup_status["vector_store"]
                    )
                except Exception as e:
                    self.logger.error(
                        "Vector store cleanup failed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        vector_store_id=vector_store_id,
                        error=str(e)
                    )
                    cleanup_status["vector_store"] = False
            
            # Clean up conversation
            if conversation_id:
                try:
                    cleanup_status["conversation"] = (
                        await self.chat_manager.delete_chat(conversation_id)
                    )
                    self.logger.info(
                        "Conversation cleanup completed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        conversation_id=conversation_id,
                        success=cleanup_status["conversation"]
                    )
                except Exception as e:
                    self.logger.error(
                        "Conversation cleanup failed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        conversation_id=conversation_id,
                        error=str(e)
                    )
                    cleanup_status["conversation"] = False
            
            # Clear caches if requested
            if clear_all_caches:
                try:
                    self.tool_manager.clear_tool_cache()
                    self.chat_manager.clear_cache()
                    self.vector_store_manager.clear_cache()
                    cleanup_status["caches_cleared"] = True
                    self.logger.info(
                        "All caches cleared successfully",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources"
                    )
                except Exception as e:
                    self.logger.error(
                        "Cache clearing failed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        error=str(e)
                    )
                    cleanup_status["caches_cleared"] = False
            
            return cleanup_status
            
        except Exception as e:
            self.logger.error(
                "Resource cleanup error",
                component="OpenAIResponseManager",
                subcomponent="cleanup_resources",
                error=str(e)
            )
            return {"error": str(e)}

    @time_execution("OpenAIResponseManager", "create_vector_store_from_guidelines")
    async def create_vector_store_from_guidelines(
        self,
        poll_for_completion: bool = True,
        max_wait_seconds: int = VECTOR_STORE_MAX_WAIT_SECONDS
    ) -> str:
        """
        Create a vector store from guidelines with optional polling for completion.
        
        This method creates a vector store and can optionally wait for it to be
        ready before returning. Useful for ensuring vector stores are available
        before using them in queries.
        
        Args:
            poll_for_completion: If True, waits until vector store is ready.
            max_wait_seconds: Maximum time to wait for completion (if polling).
        
        Returns:
            Vector store ID.
        
        Raises:
            VectorStoreError: If creation fails or times out.
            ResponsesAPIError: On general setup failures.
        """
        try:
            self.logger.info(
                "Creating vector store from guidelines",
                component="OpenAIResponseManager",
                subcomponent="create_vector_store_from_guidelines",
                poll_for_completion=poll_for_completion,
                max_wait_seconds=max_wait_seconds
            )
            
            vector_store_id = (
                await self.vector_store_manager.create_guidelines_vector_store()
            )
            
            self.logger.info(
                "Vector store created",
                component="OpenAIResponseManager",
                subcomponent="create_vector_store_from_guidelines",
                vector_store_id=vector_store_id
            )
            
            if not poll_for_completion:
                return vector_store_id
            
            # Poll for completion
            total_wait_seconds = 0
            
            while total_wait_seconds < max_wait_seconds:
                store_info = await self.vector_store_manager.get_vector_store(
                    vector_store_id
                )
                status = store_info['status'] if store_info else 'not_found'
                
                self.logger.info(
                    "Polling vector store status",
                    component="OpenAIResponseManager",
                    subcomponent="create_vector_store_from_guidelines",
                    vector_store_id=vector_store_id,
                    status=status,
                    elapsed_seconds=total_wait_seconds
                )
                
                if store_info and store_info["status"] == "completed":
                    self.logger.info(
                        "Vector store ready",
                        component="OpenAIResponseManager",
                        subcomponent="create_vector_store_from_guidelines",
                        vector_store_id=vector_store_id,
                        total_wait_seconds=total_wait_seconds
                    )
                    return vector_store_id
                    
                if store_info and store_info["status"] == "failed":
                    self.logger.error(
                        "Vector store creation failed",
                        component="OpenAIResponseManager",
                        subcomponent="create_vector_store_from_guidelines",
                        vector_store_id=vector_store_id
                    )
                    raise VectorStoreError(
                        f"Vector store {vector_store_id} failed during creation"
                    )
                
                await asyncio.sleep(self.VECTOR_STORE_POLL_INTERVAL_SECONDS)
                total_wait_seconds += self.VECTOR_STORE_POLL_INTERVAL_SECONDS
            
            # Timeout reached
            self.logger.error(
                "Vector store creation timed out",
                component="OpenAIResponseManager",
                subcomponent="create_vector_store_from_guidelines",
                vector_store_id=vector_store_id,
                max_wait_seconds=max_wait_seconds
            )
            raise VectorStoreError(
                f"Vector store {vector_store_id} not ready after "
                f"{max_wait_seconds} seconds"
            )
            
        except VectorStoreError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to create vector store from guidelines",
                component="OpenAIResponseManager",
                subcomponent="create_vector_store_from_guidelines",
                error=str(e)
            )
            raise ResponsesAPIError(
                message=f"Vector store creation failed: {str(e)}"
            )

    async def get_vector_store_status(
        self, 
        vector_store_id: str
    ) -> Dict[str, Any]:
        """
        Get the current status and details of a vector store.
        
        Args:
            vector_store_id: The vector store ID to check.
        
        Returns:
            Dictionary containing vector store status and metadata.
        
        Raises:
            VectorStoreError: If status retrieval fails.
        """
        try:
            self.logger.info(
                "Retrieving vector store status",
                component="OpenAIResponseManager",
                subcomponent="get_vector_store_status",
                vector_store_id=vector_store_id
            )
            
            store_info = await self.vector_store_manager.get_vector_store(
                vector_store_id
            )
            
            if not store_info:
                self.logger.warning(
                    "Vector store not found",
                    component="OpenAIResponseManager",
                    subcomponent="get_vector_store_status",
                    vector_store_id=vector_store_id
                )
                raise VectorStoreError(f"Vector store {vector_store_id} not found")
            
            self.logger.info(
                "Vector store status retrieved",
                component="OpenAIResponseManager",
                subcomponent="get_vector_store_status",
                vector_store_id=vector_store_id,
                status=store_info.get("status")
            )
            
            return store_info
            
        except VectorStoreError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to retrieve vector store status",
                component="OpenAIResponseManager",
                subcomponent="get_vector_store_status",
                vector_store_id=vector_store_id,
                error=str(e)
            )
            raise VectorStoreError(
                f"Status retrieval failed for {vector_store_id}: {str(e)}"
            )

    def clear_all_caches(self) -> None:
        """
        Clear all internal caches across all managers.
        
        This is useful for freeing memory or ensuring fresh data retrieval.
        """
        try:
            self.logger.info(
                "Clearing all manager caches",
                component="OpenAIResponseManager",
                subcomponent="clear_all_caches"
            )
            
            self.tool_manager.clear_tool_cache()
            self.chat_manager.clear_cache()
            self.vector_store_manager.clear_cache()
            
            self.logger.info(
                "All caches cleared successfully",
                component="OpenAIResponseManager",
                subcomponent="clear_all_caches"
            )
        except Exception as e:
            self.logger.error(
                "Error clearing caches",
                component="OpenAIResponseManager",
                subcomponent="clear_all_caches",
                error=str(e)
            )

    async def validate_configuration(self) -> Dict[str, bool]:
        """
        Validate the manager configuration and all dependent components.
        
        Returns:
            Dictionary with validation results for each component.
        """
        self.logger.info(
            "Validating manager configuration",
            component="OpenAIResponseManager",
            subcomponent="validate_configuration"
        )
        
        validation_results = {
            "settings_loaded": bool(self.settings),
            "vector_store_manager": bool(self.vector_store_manager),
            "tool_manager": bool(self.tool_manager),
            "chat_manager": bool(self.chat_manager),
            "stream_manager": bool(self.stream_manager),
            "model_name_valid": self._validate_model_identifier(
                self.settings.openai_model_name
            )
        }
        
        all_valid = all(validation_results.values())
        
        self.logger.info(
            "Configuration validation completed",
            component="OpenAIResponseManager",
            subcomponent="validate_configuration",
            all_valid=all_valid,
            results=validation_results
        )
        
        return validation_results

    def get_manager_info(self) -> Dict[str, Any]:
        """
        Get information about the manager and its configuration.
        
        Returns:
            Dictionary with manager configuration and status information.
        """
        return {
            "manager_type": "OpenAIResponseManager",
            "model_name": self.settings.openai_model_name,
            "chat_history_limit": self.chat_manager.chat_history_limit 
                if hasattr(self.chat_manager, 'chat_history_limit') else None,
            "vector_store_batch_size": self.vector_store_manager.batch_size 
                if hasattr(self.vector_store_manager, 'batch_size') else None,
            "components": {
                "vector_store_manager": type(self.vector_store_manager).__name__,
                "tool_manager": type(self.tool_manager).__name__,
                "chat_manager": type(self.chat_manager).__name__,
                "stream_manager": type(self.stream_manager).__name__
            }
        }