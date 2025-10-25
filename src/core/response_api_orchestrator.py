"""
Response API Orchestrator for integrating VectorStoreManager, ToolManager, ChatManager, and StreamManager.

This module orchestrates the components for end-to-end workflows with the OpenAI Responses API,
including vector store setup, tool configuration, conversation management, and streaming responses.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from src.core.config import get_settings
from src.core.managers.vector_store_manager import VectorStoreManager
from src.core.managers.tool_manager import ToolManager
from src.core.managers.chat_manager import ChatManager
from src.core.managers.stream_manager import StreamManager
from src.core.managers.exceptions import (
    ResponsesAPIError, VectorStoreError, ToolConfigurationError, 
    StreamConnectionError, ContentParsingError
)
from src.core.logs import get_component_logger, log_execution_time, time_execution


class ResponseAPIOrchestrator:
    """
    Orchestrates the OpenAI Responses API components for complete workflows.
    
    Initializes and coordinates VectorStoreManager, ToolManager, ChatManager, and StreamManager.
    Supports setup of vector stores, tool-enabled conversations, and streaming responses.
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        chat_history_limit: int = 10,
        batch_size: int = 5,
        rate_limit_delay: float = 1.0
    ):
        """Initialize the Orchestrator.
        
        Args:
            vector_store_manager: Optional pre-initialized VectorStoreManager.
            chat_history_limit: Max chain length for ChatManager.
            batch_size: Batch size for VectorStoreManager.
            rate_limit_delay: Rate limit delay for VectorStoreManager.
        """
        self.settings = get_settings()
        self.logger = get_component_logger("Orchestrator")
        
        # Initialize managers
        if vector_store_manager:
            self.vector_store_manager = vector_store_manager
        else:
            self.vector_store_manager = VectorStoreManager(
                batch_size=batch_size, rate_limit_delay=rate_limit_delay
            )
        
        self.tool_manager = ToolManager(self.vector_store_manager)
        self.chat_manager = ChatManager(
            tool_manager=self.tool_manager, chat_history_limit=chat_history_limit
        )
        self.stream_manager = StreamManager()

    def _validate_model_name(self, model: str) -> bool:
        """Validate that model name follows expected OpenAI format."""
        valid_prefixes = ["gpt-3.5", "gpt-4", "o1", "o3"]
        return any(model.startswith(prefix) for prefix in valid_prefixes)

    @time_execution("Orchestrator", "HandleQuery")
    async def handle_query(
        self,
        message: str,
        chat_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Handle a user query with optional tools and streaming.

        Args:
            message: User message.
            chat_id: Optional existing chat ID for continuation.
            vector_store_id: Optional vector store ID for file_search tool.
            functions: Optional list of function definitions.
            model: Optional model override.
            stream: If True, return streaming generator function; else, full response.

        Returns:
            Dict with chat_id, content (or generator function if streaming), and tool_calls.
            When streaming, use: async for chunk in result["stream_generator"](): ...
        """
        try:
            self.logger.info(
                "Handling query",
                component="Orchestrator",
                subcomponent="HandleQuery",
                message_length=len(message),
                has_chat_id=bool(chat_id),
                has_vector_store=bool(vector_store_id),
                has_functions=bool(functions),
                streaming=stream
            )
            
            # Resolve model to use default from settings if not provided
            resolved_model = model or self.settings.openai_model_name
            if not self._validate_model_name(resolved_model):
                self.logger.warning(
                    "Model name may be invalid",
                    component="Orchestrator",
                    subcomponent="HandleQuery",
                    model=resolved_model
                )
                
            self.logger.info(
                "Using model",
                component="Orchestrator",
                subcomponent="HandleQuery",
                model=resolved_model
            )
            
            # Prepare tools if provided
            tools = []
            if vector_store_id or functions:
                # Validate vector store is ready before creating tools
                if vector_store_id:
                    store_info = await self.vector_store_manager.get_vector_store(vector_store_id)
                    if not store_info or store_info["status"] != "completed":
                        self.logger.warning(
                            "Vector store not ready, skipping file_search tool",
                            component="Orchestrator",
                            subcomponent="HandleQuery",
                            vector_store_id=vector_store_id,
                            status=store_info["status"] if store_info else "Not found"
                        )
                        vector_store_id = None  # Don't use the vector store if not ready
                    else:
                        self.logger.info(
                            "Vector store is ready for tool creation",
                            component="Orchestrator",
                            subcomponent="HandleQuery",
                            vector_store_id=vector_store_id
                        )

                if vector_store_id or functions:
                    try:
                        tools = await self.tool_manager.get_tools_for_response(
                            vector_store_id=vector_store_id, functions=functions
                        )
                        # Strict validation - fail on error
                        is_valid = await self.tool_manager.validate_tools(tools)
                        if not is_valid:
                            self.logger.warning(
                                "Tool validation failed, proceeding without tools",
                                component="Orchestrator",
                                subcomponent="HandleQuery"
                            )
                            tools = []
                    except Exception as e:
                        self.logger.warning(
                            "Error preparing tools, proceeding without tools",
                            component="Orchestrator",
                            subcomponent="HandleQuery",
                            error=str(e)
                        )
                        tools = []
            
            if stream:
                # Streaming response
                self.logger.info(
                    "Setting up streaming response",
                    component="Orchestrator",
                    subcomponent="HandleQuery",
                    streaming=True
                )
                
                async def stream_gen() -> AsyncGenerator[str, None]:
                    async for chunk in self.stream_manager.stream_response(
                        message=message, model=resolved_model, tools=tools
                    ):
                        yield chunk
                
                # For streaming, we need to create a chat first to get chat_id for history
                if not chat_id:
                    chat_id = await self.chat_manager.create_chat(
                        message=message, model=resolved_model, tools=tools
                    )
                
                self.logger.info(
                    "Streaming response setup complete",
                    component="Orchestrator",
                    subcomponent="HandleQuery",
                    chat_id=chat_id
                )
                
                return {
                    "chat_id": chat_id,
                    "stream_generator": stream_gen,  # Call this function to get the async generator: stream_gen()
                    "tools": tools
                }
            else:
                # Non-streaming response
                if chat_id:
                    # Continue existing conversation
                    self.logger.info(
                        "Continuing existing conversation",
                        component="Orchestrator",
                        subcomponent="HandleQuery",
                        chat_id=chat_id
                    )
                    
                    result = await self.chat_manager.continue_chat_with_tools(
                        chat_id=chat_id,
                        message=message,
                        vector_store_id=vector_store_id,
                        functions=functions,
                        model=resolved_model
                    )
                    
                    self.logger.info(
                        "Continued conversation successfully",
                        component="Orchestrator",
                        subcomponent="HandleQuery",
                        chat_id=chat_id,
                        has_tool_calls=bool(result.get("tool_calls", []))
                    )
                    
                    return {
                        "chat_id": chat_id,
                        "content": result["content"],
                        "tool_calls": result["tool_calls"],
                        "tools": tools
                    }
                else:
                    # Start new conversation
                    self.logger.info(
                        "Starting new conversation",
                        component="Orchestrator",
                        subcomponent="HandleQuery"
                    )
                    
                    chat_id = await self.chat_manager.create_chat(
                        message=message, model=resolved_model, tools=tools
                    )
                    
                    # Get the response content
                    response = await asyncio.to_thread(
                        self.chat_manager.client.responses.retrieve, response_id=chat_id
                    )
                    content = self.chat_manager._extract_text_content(response)
                    
                    # Extract tool calls from the new response structure
                    tool_calls = []
                    if hasattr(response, 'output') and response.output:
                        for item in response.output:
                            # Look for tool call items
                            if hasattr(item, 'type') and 'call' in item.type:
                                tool_calls.append(item)
                    # Fallback to legacy format
                    else:
                        tool_calls = getattr(response, 'tool_calls', [])
                    
                    self.logger.info(
                        "New conversation started successfully",
                        component="Orchestrator",
                        subcomponent="HandleQuery",
                        chat_id=chat_id,
                        has_tool_calls=bool(tool_calls)
                    )
                    
                    return {
                        "chat_id": chat_id,
                        "content": content,
                        "tool_calls": tool_calls,
                        "tools": tools
                    }
                
        except (ToolConfigurationError, VectorStoreError) as e:
            self.logger.error(
                "Tool/Vector store error in handle_query",
                component="Orchestrator",
                subcomponent="HandleQuery",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to handle query",
                component="Orchestrator",
                subcomponent="HandleQuery",
                error=str(e)
            )
            raise ResponsesAPIError(message=f"Query handling failed: {str(e)}")

    @time_execution("Orchestrator", "HandleStreamingQuery")
    async def handle_streaming_query(
        self,
        message: str,
        chat_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a streaming query with conversation history maintained.
        
        Args:
            message: User message.
            chat_id: Optional existing chat ID for continuation.
            vector_store_id: Optional vector store ID for file_search tool.
            functions: Optional function definitions.
            model: Optional model override.
            
        Yields:
            Dict with chunk data and metadata.
        """
        try:
            self.logger.info(
                "Handling streaming query",
                component="Orchestrator",
                subcomponent="HandleStreamingQuery",
                message_length=len(message),
                has_chat_id=bool(chat_id),
                has_vector_store=bool(vector_store_id),
                has_functions=bool(functions)
            )
            
            # Resolve model to use default from settings if not provided
            resolved_model = model or self.settings.openai_model_name
            if not self._validate_model_name(resolved_model):
                self.logger.warning(
                    "Model name may be invalid",
                    component="Orchestrator",
                    subcomponent="HandleStreamingQuery",
                    model=resolved_model
                )
                
            self.logger.info(
                "Using model",
                component="Orchestrator",
                subcomponent="HandleStreamingQuery",
                model=resolved_model
            )
            
            # Prepare tools if provided
            tools = []
            if vector_store_id or functions:
                # Validate vector store is ready before creating tools
                if vector_store_id:
                    store_info = await self.vector_store_manager.get_vector_store(vector_store_id)
                    if not store_info or store_info["status"] != "completed":
                        self.logger.warning(
                            "Vector store not ready, skipping file_search tool",
                            component="Orchestrator",
                            subcomponent="HandleStreamingQuery",
                            vector_store_id=vector_store_id,
                            status=store_info["status"] if store_info else "Not found"
                        )
                        vector_store_id = None  # Don't use the vector store if not ready
                    else:
                        self.logger.info(
                            "Vector store is ready for tool creation",
                            component="Orchestrator",
                            subcomponent="HandleStreamingQuery",
                            vector_store_id=vector_store_id
                        )

                if vector_store_id or functions:
                    try:
                        tools = await self.tool_manager.get_tools_for_response(
                            vector_store_id=vector_store_id, functions=functions
                        )
                        # Strict validation - fail on error
                        is_valid = await self.tool_manager.validate_tools(tools)
                        if not is_valid:
                            self.logger.warning(
                                "Tool validation failed, proceeding without tools",
                                component="Orchestrator",
                                subcomponent="HandleStreamingQuery"
                            )
                            tools = []
                    except Exception as e:
                        self.logger.warning(
                            "Error preparing tools, proceeding without tools",
                            component="Orchestrator",
                            subcomponent="HandleStreamingQuery",
                            error=str(e)
                        )
                        tools = []
            
            # Use appropriate streaming method
            chunk_count = 0
            
            if chat_id:
                # Continue existing conversation
                self.logger.info(
                    "Streaming chat continuation",
                    component="Orchestrator",
                    subcomponent="HandleStreamingQuery",
                    chat_id=chat_id
                )
                
                async for chunk in self.stream_manager.stream_chat_continuation(
                    chat_id=chat_id, message=message, model=resolved_model, tools=tools
                ):
                    chunk_count += 1
                    yield {
                        "chunk": chunk,
                        "chat_id": chat_id,
                        "tools": tools
                    }
                    
                self.logger.info(
                    "Chat continuation streaming completed",
                    component="Orchestrator",
                    subcomponent="HandleStreamingQuery",
                    chat_id=chat_id,
                    chunk_count=chunk_count
                )
            else:
                # Start new conversation
                self.logger.info(
                    "Starting new streaming conversation",
                    component="Orchestrator",
                    subcomponent="HandleStreamingQuery"
                )
                
                new_chat_id = await self.chat_manager.create_chat(
                    message=message, model=resolved_model, tools=tools
                )
                
                # Stream the response
                async for chunk in self.stream_manager.stream_response(
                    message=message, model=resolved_model, tools=tools
                ):
                    chunk_count += 1
                    yield {
                        "chunk": chunk,
                        "chat_id": new_chat_id,
                        "tools": tools
                    }
                
                self.logger.info(
                    "New conversation streaming completed",
                    component="Orchestrator",
                    subcomponent="HandleStreamingQuery",
                    chat_id=new_chat_id,
                    chunk_count=chunk_count
                )
                    
        except (ToolConfigurationError, VectorStoreError, StreamConnectionError) as e:
            self.logger.error(
                "Error in streaming query",
                component="Orchestrator",
                subcomponent="HandleStreamingQuery",
                error=str(e),
                error_type=type(e).__name__
            )
            yield {
                "error": str(e),
                "chat_id": chat_id,
                "tools": tools
            }
        except Exception as e:
            self.logger.error(
                "Failed to handle streaming query",
                component="Orchestrator",
                subcomponent="HandleStreamingQuery",
                error=str(e)
            )
            yield {
                "error": f"Streaming query failed: {str(e)}",
                "chat_id": chat_id,
                "tools": tools
            }

    @time_execution("Orchestrator", "GetHistory")
    async def get_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a chat.
        
        Args:
            chat_id: Chat ID.
            
        Returns:
            List of response dicts with content and tool calls.
        """
        try:
            self.logger.info(
                "Retrieving chat history",
                component="Orchestrator",
                subcomponent="GetHistory",
                chat_id=chat_id
            )
            
            history = await self.chat_manager.get_chat_history(chat_id)
            
            self.logger.info(
                "Retrieved chat history successfully",
                component="Orchestrator",
                subcomponent="GetHistory",
                chat_id=chat_id,
                history_length=len(history)
            )
            
            return history
        except Exception as e:
            self.logger.error(
                "Failed to get conversation history",
                component="Orchestrator",
                subcomponent="GetHistory",
                chat_id=chat_id,
                error=str(e)
            )
            raise ResponsesAPIError(message=f"History retrieval failed: {str(e)}")

    @time_execution("Orchestrator", "CleanupResources")
    async def cleanup_resources(
        self, 
        vector_store_id: Optional[str] = None, 
        chat_id: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Clean up resources like vector stores and chats.
        
        Args:
            vector_store_id: Optional vector store ID to delete.
            chat_id: Optional chat ID to delete.
            
        Returns:
            Dict with cleanup status for each resource.
        """
        try:
            self.logger.info(
                "Starting resource cleanup",
                component="Orchestrator",
                subcomponent="CleanupResources",
                has_vector_store=bool(vector_store_id),
                has_chat=bool(chat_id)
            )
            
            cleanup_status = {}
            
            if vector_store_id:
                cleanup_status["vector_store"] = await self.vector_store_manager.delete_vector_store(vector_store_id)
                self.logger.info(
                    "Vector store cleanup completed",
                    component="Orchestrator",
                    subcomponent="CleanupResources",
                    vector_store_id=vector_store_id,
                    success=cleanup_status["vector_store"]
                )
            
            if chat_id:
                cleanup_status["chat"] = await self.chat_manager.delete_chat(chat_id)
                self.logger.info(
                    "Chat cleanup completed",
                    component="Orchestrator",
                    subcomponent="CleanupResources",
                    chat_id=chat_id,
                    success=cleanup_status["chat"]
                )
            
            # Clear caches
            self.tool_manager.clear_tool_cache()
            self.chat_manager.clear_cache()
            self.vector_store_manager.clear_cache()
            
            self.logger.info(
                "All caches cleared, orchestrator cleanup completed",
                component="Orchestrator",
                subcomponent="CleanupResources"
            )
            return cleanup_status
            
        except Exception as e:
            self.logger.error(
                "Cleanup error",
                component="Orchestrator",
                subcomponent="CleanupResources",
                error=str(e)
            )
            return {"error": str(e)}

    @time_execution("Orchestrator", "SetupGuidelinesVectorStore")
    async def setup_guidelines_vector_store(self) -> str:
        """
        Set up the guidelines vector store with background file uploads.

        Returns:
            Vector store ID.
        """
        try:
            self.logger.info(
                "Setting up guidelines vector store",
                component="Orchestrator",
                subcomponent="SetupGuidelinesVectorStore"
            )
            
            vector_store_id = await self.vector_store_manager.create_guidelines_vector_store()
            self.logger.info(
                "Vector store created, polling for completion",
                component="Orchestrator",
                subcomponent="SetupGuidelinesVectorStore",
                vector_store_id=vector_store_id
            )

            # Poll vector store status until completed (with timeout)
            max_wait_time = 300  # 5 minutes max wait
            poll_interval = 10   # Check every 10 seconds
            total_waited = 0

            while total_waited < max_wait_time:
                store_info = await self.vector_store_manager.get_vector_store(vector_store_id)
                status = store_info['status'] if store_info else 'Not found'
                
                self.logger.info(
                    "Polling vector store status",
                    component="Orchestrator",
                    subcomponent="SetupGuidelinesVectorStore",
                    vector_store_id=vector_store_id,
                    status=status,
                    waited_seconds=total_waited
                )

                if store_info and store_info["status"] == "completed":
                    self.logger.info(
                        "Guidelines vector store ready",
                        component="Orchestrator",
                        subcomponent="SetupGuidelinesVectorStore",
                        vector_store_id=vector_store_id,
                        total_wait_time=total_waited
                    )
                    return vector_store_id
                elif store_info and store_info["status"] == "failed":
                    self.logger.error(
                        "Vector store setup failed",
                        component="Orchestrator",
                        subcomponent="SetupGuidelinesVectorStore",
                        vector_store_id=vector_store_id,
                        status="failed"
                    )
                    raise VectorStoreError(f"Vector store {vector_store_id} failed during setup")

                await asyncio.sleep(poll_interval)
                total_waited += poll_interval

            # If we get here, we timed out
            self.logger.error(
                "Vector store setup timed out",
                component="Orchestrator",
                subcomponent="SetupGuidelinesVectorStore",
                vector_store_id=vector_store_id,
                max_wait_time=max_wait_time
            )
            raise VectorStoreError(f"Vector store {vector_store_id} not ready after {max_wait_time} seconds")

        except VectorStoreError as e:
            self.logger.error(
                "Vector store setup error",
                component="Orchestrator",
                subcomponent="SetupGuidelinesVectorStore",
                error=str(e),
                error_type="VectorStoreError"
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to setup guidelines vector store",
                component="Orchestrator",
                subcomponent="SetupGuidelinesVectorStore",
                error=str(e)
            )
            raise ResponsesAPIError(message=f"Setup failed: {str(e)}")


# Example usage (for testing; run with asyncio)
async def example_workflow():
    """Example end-to-end workflow using the Orchestrator.

    This example demonstrates:
    1. Setting up a vector store for medical guidelines
    2. Using the vector store for semantic search in queries when available
    3. Graceful fallback to non-tool operation if vector store setup fails
    4. Robust error handling at each step with appropriate fallbacks
    5. Proper resource cleanup with fallback strategies
    """
    # Create a logger for the example workflow
    logger = get_component_logger("ExampleWorkflow")
    
    try:
        logger.info(
            "Starting example workflow",
            component="ExampleWorkflow",
            subcomponent="Main"
        )
        
        # Initialize orchestrator
        orchestrator = ResponseAPIOrchestrator(chat_history_limit=5)
        logger.info(
            "Initialized orchestrator",
            component="ExampleWorkflow",
            subcomponent="Main"
        )

        # Setup vector store (with error handling)
        vector_store_id = None
        try:
            vector_store_id = await orchestrator.setup_guidelines_vector_store()
            logger.info(
                "Vector store setup successful",
                component="ExampleWorkflow",
                subcomponent="VectorStoreSetup",
                vector_store_id=vector_store_id
            )
        except Exception as e:
            logger.warning(
                "Vector store setup failed, continuing without tools",
                component="ExampleWorkflow",
                subcomponent="VectorStoreSetup",
                error=str(e)
            )
            vector_store_id = None

        # Use vector store if available to enable semantic search over guidelines
        try:
            logger.info(
                "Sending initial query",
                component="ExampleWorkflow",
                subcomponent="InitialQuery",
                has_vector_store=bool(vector_store_id)
            )
            
            result = await orchestrator.handle_query(
                message="What are the latest guidelines for diabetes treatment?",
                chat_id=None,  # Start new conversation
                vector_store_id=vector_store_id,  # Use vector store if setup was successful
                functions=None,  # No custom functions
                model=orchestrator.settings.openai_model_name,  # Use default model from settings
                stream=False
            )
            
            # If we got here with vector_store_id, tools were successfully used
            if vector_store_id:
                logger.info(
                    "Successfully used vector store for semantic search",
                    component="ExampleWorkflow",
                    subcomponent="InitialQuery",
                    vector_store_id=vector_store_id
                )
        except Exception as e:
            # Fall back to no tools if there's an issue with the vector store
            logger.warning(
                "Error using vector store in query, retrying without tools",
                component="ExampleWorkflow",
                subcomponent="InitialQuery",
                error=str(e)
            )
            
            vector_store_id = None
            result = await orchestrator.handle_query(
                message="What are the latest guidelines for diabetes treatment?",
                chat_id=None,
                vector_store_id=None,
                functions=None,
                model=orchestrator.settings.openai_model_name,  # Use default model from settings
                stream=False
            )
            
        print(f"Response: {result['content']}...")

        # Continue conversation with same vector store to maintain context
        try:
            logger.info(
                "Sending follow-up query",
                component="ExampleWorkflow",
                subcomponent="FollowUpQuery",
                chat_id=result["chat_id"],
                has_vector_store=bool(vector_store_id)
            )
            
            continue_result = await orchestrator.handle_query(
                message="Can you provide more details about medication recommendations?",
                chat_id=result["chat_id"],  # Continue existing conversation
                vector_store_id=vector_store_id,  # Use same vector store as first query
                functions=None,  # No custom functions
                model=orchestrator.settings.openai_model_name,  # Use default model from settings
                stream=False
            )
        except Exception as e:
            # Fall back to no tools if there's an issue with the vector store
            logger.warning(
                "Error using vector store in follow-up query, continuing without tools",
                component="ExampleWorkflow",
                subcomponent="FollowUpQuery",
                chat_id=result["chat_id"],
                error=str(e)
            )
            
            continue_result = await orchestrator.handle_query(
                message="Can you provide more details about medication recommendations?",
                chat_id=result["chat_id"],
                vector_store_id=None,
                functions=None,
                model=orchestrator.settings.openai_model_name,  # Use default model from settings
                stream=False
            )
            
        print(f"Continued: {continue_result['content'][:100]}...")

        # Get history
        logger.info(
            "Retrieving conversation history",
            component="ExampleWorkflow",
            subcomponent="GetHistory",
            chat_id=result["chat_id"]
        )
        
        history = await orchestrator.get_history(result["chat_id"])
        
        logger.info(
            "Retrieved history",
            component="ExampleWorkflow",
            subcomponent="GetHistory",
            chat_id=result["chat_id"],
            history_length=len(history)
        )
        
        print(f"History length: {len(history)}")

        # Cleanup both chat and vector store resources
        try:
            logger.info(
                "Cleaning up resources",
                component="ExampleWorkflow",
                subcomponent="Cleanup",
                chat_id=result["chat_id"],
                has_vector_store=bool(vector_store_id)
            )
            
            cleanup_result = await orchestrator.cleanup_resources(
                chat_id=result["chat_id"],
                vector_store_id=vector_store_id if vector_store_id else None
            )
            
            logger.info(
                "Cleanup completed successfully",
                component="ExampleWorkflow",
                subcomponent="Cleanup",
                result=cleanup_result
            )
        except Exception as e:
            logger.error(
                "Error during cleanup",
                component="ExampleWorkflow",
                subcomponent="Cleanup",
                error=str(e)
            )
            
            # Try to at least clean up the chat if vector store cleanup fails
            if result and "chat_id" in result:
                try:
                    await orchestrator.cleanup_resources(chat_id=result["chat_id"])
                    logger.info(
                        "Chat cleanup successful after vector store cleanup failure",
                        component="ExampleWorkflow",
                        subcomponent="Cleanup",
                        chat_id=result["chat_id"]
                    )
                except Exception as e2:
                    logger.error(
                        "Failed to clean up chat after vector store cleanup failure",
                        component="ExampleWorkflow",
                        subcomponent="Cleanup",
                        chat_id=result["chat_id"],
                        error=str(e2)
                    )

        logger.info(
            "Example workflow completed successfully",
            component="ExampleWorkflow",
            subcomponent="Main"
        )
    except Exception as e:
        logger.error(
            "Example workflow error",
            component="ExampleWorkflow",
            subcomponent="Main",
            error=str(e)
        )


if __name__ == "__main__":
    asyncio.run(example_workflow())
