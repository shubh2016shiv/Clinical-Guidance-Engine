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
from src.response_api_agent.managers.citation_manager import CitationManager
from src.response_api_agent.managers.drug_data_manager import DrugDataManager
from src.response_api_agent.managers.adapter_monitoring import start_metrics_reporting
from src.response_api_agent.managers.exceptions import (
    ResponsesAPIError,
    VectorStoreError,
    ToolConfigurationError,
    StreamConnectionError,
)
from src.logs import get_component_logger, time_execution
from src.providers.cache_provider import (
    CacheProvider,
    create_cache_provider,
    CacheConnectionError,
)


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
        cache_provider: Optional[CacheProvider] = None,
        chat_history_limit: int = DEFAULT_CHAT_HISTORY_LIMIT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    ):
        """
        Initialize the OpenAI Response Manager.

        Args:
            vector_store_manager: Pre-configured VectorStoreManager instance.
                If None, a new instance will be created.
            cache_provider: Pre-configured CacheProvider instance for Redis caching.
                If None, a default Redis provider will be created and connected.
            chat_history_limit: Maximum number of messages to retain in chat history.
            batch_size: Batch size for vector store operations.
            rate_limit_delay: Delay in seconds between rate-limited operations.
        """
        self.settings = get_settings()
        self.logger = get_component_logger("OpenAIResponseManager")

        # Start adapter metrics reporting
        start_metrics_reporting()

        # Initialize or use provided cache provider
        self.cache_provider = cache_provider
        if self.cache_provider is None:
            try:
                self.logger.info(
                    "Initializing Redis cache provider",
                    component="OpenAIResponseManager",
                    subcomponent="Init",
                )
                self.cache_provider = create_cache_provider(provider_type="redis")
            except Exception as e:
                self.logger.warning(
                    f"Failed to create cache provider: {e}. Continuing without caching.",
                    component="OpenAIResponseManager",
                    subcomponent="Init",
                )
                self.cache_provider = None

        # Connect to cache provider if it exists (whether created here or passed in)
        if self.cache_provider:
            # Connect to Redis asynchronously
            asyncio.create_task(self._connect_cache())

        # Initialize or use provided vector store manager
        if vector_store_manager:
            self.vector_store_manager = vector_store_manager
        else:
            self.vector_store_manager = VectorStoreManager(
                batch_size=batch_size,
                rate_limit_delay=rate_limit_delay,
                cache_provider=self.cache_provider,
            )

        # Initialize dependent managers
        self.tool_manager = ToolManager(self.vector_store_manager)
        self.chat_manager = ChatManager(
            tool_manager=self.tool_manager,
            chat_history_limit=chat_history_limit,
            cache_provider=self.cache_provider,
        )
        self.stream_manager = StreamManager(
            chat_manager=self.chat_manager, cache_provider=self.cache_provider
        )
        self.citation_manager = CitationManager(client=self.chat_manager.client)

        # Initialize drug data manager for Milvus integration
        try:
            self.drug_data_manager = DrugDataManager()
            # Register the drug search function with chat manager
            self.chat_manager.register_function_executor(
                "search_drug_database", self.drug_data_manager.search_drug_database
            )
            self.logger.info(
                "Drug data manager initialized and registered",
                component="OpenAIResponseManager",
                subcomponent="Init",
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to initialize drug data manager: {e}. Drug search will not be available.",
                component="OpenAIResponseManager",
                subcomponent="Init",
            )
            self.drug_data_manager = None

        self.logger.info(
            "OpenAI Response Manager initialized",
            component="OpenAIResponseManager",
            subcomponent="Init",
            chat_history_limit=chat_history_limit,
            batch_size=batch_size,
            cache_enabled=self.cache_provider is not None,
        )

    async def _connect_cache(self) -> None:
        """
        Connect to the cache provider asynchronously.

        Handles connection errors gracefully, disabling caching if connection fails.
        """
        if self.cache_provider is None:
            return

        try:
            await self.cache_provider.connect()

            # Verify health
            is_healthy = await self.cache_provider.health_check()
            if is_healthy:
                self.logger.info(
                    "Cache provider connected and healthy",
                    component="OpenAIResponseManager",
                    subcomponent="ConnectCache",
                )
            else:
                self.logger.warning(
                    "Cache provider connected but health check failed",
                    component="OpenAIResponseManager",
                    subcomponent="ConnectCache",
                )
                self.cache_provider = None

        except CacheConnectionError as e:
            self.logger.warning(
                f"Failed to connect to cache provider: {e}. Continuing without caching.",
                component="OpenAIResponseManager",
                subcomponent="ConnectCache",
            )
            self.cache_provider = None
        except Exception as e:
            self.logger.warning(
                f"Unexpected error connecting to cache: {e}. Continuing without caching.",
                component="OpenAIResponseManager",
                subcomponent="ConnectCache",
            )
            self.cache_provider = None

    def _validate_model_identifier(self, model_name: str) -> bool:
        """
        Validate that model identifier follows OpenAI naming conventions.

        Args:
            model_name: Model identifier to validate.

        Returns:
            True if model name is valid, False otherwise.
        """
        return any(
            model_name.startswith(prefix) for prefix in self.VALID_MODEL_PREFIXES
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
                model=model_name,
            )

        return model_name

    async def _validate_and_prepare_vector_store(
        self, vector_store_id: Optional[str]
    ) -> Optional[str]:
        """
        Validate vector store readiness and return ID if ready.
        Uses Redis cache to avoid repeated validation calls.

        Args:
            vector_store_id: Vector store ID to validate.

        Returns:
            Vector store ID if ready, None if not ready or validation fails.
        """
        if not vector_store_id:
            return None

        # Check cache first
        if self.cache_provider:
            try:
                cached_validation = (
                    await self.cache_provider.get_vector_store_validation(
                        vector_store_id
                    )
                )
                if cached_validation and cached_validation.get("status") == "completed":
                    self.logger.info(
                        "Vector store validation retrieved from cache",
                        component="OpenAIResponseManager",
                        subcomponent="_validate_and_prepare_vector_store",
                        vector_store_id=vector_store_id,
                    )
                    return vector_store_id
            except Exception as e:
                self.logger.warning(
                    f"Failed to get vector store validation from cache: {e}",
                    component="OpenAIResponseManager",
                    subcomponent="_validate_and_prepare_vector_store",
                )

        try:
            store_info = await self.vector_store_manager.get_vector_store(
                vector_store_id
            )

            if not store_info:
                self.logger.warning(
                    "Vector store not found",
                    component="OpenAIResponseManager",
                    subcomponent="_validate_and_prepare_vector_store",
                    vector_store_id=vector_store_id,
                )
                return None

            if store_info["status"] != "completed":
                self.logger.warning(
                    "Vector store not in completed state",
                    component="OpenAIResponseManager",
                    subcomponent="_validate_and_prepare_vector_store",
                    vector_store_id=vector_store_id,
                    status=store_info["status"],
                )
                return None

            # Cache the validation result
            if self.cache_provider:
                try:
                    from datetime import datetime

                    # Handle file_counts - it might be a dict or an object
                    file_counts = store_info.get("file_counts", {})
                    if hasattr(file_counts, "completed"):
                        file_count = file_counts.completed
                    elif isinstance(file_counts, dict):
                        file_count = file_counts.get("completed", 0)
                    else:
                        file_count = 0

                    validation_data = {
                        "status": "completed",
                        "validated_at": datetime.utcnow().isoformat(),
                        "file_count": file_count,
                    }
                    await self.cache_provider.cache_vector_store_validation(
                        vector_store_id, validation_data
                    )
                    self.logger.debug(
                        "Cached vector store validation",
                        component="OpenAIResponseManager",
                        subcomponent="_validate_and_prepare_vector_store",
                        vector_store_id=vector_store_id,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to cache vector store validation: {e}",
                        component="OpenAIResponseManager",
                        subcomponent="_validate_and_prepare_vector_store",
                    )

            self.logger.info(
                "Vector store validated and ready",
                component="OpenAIResponseManager",
                subcomponent="_validate_and_prepare_vector_store",
                vector_store_id=vector_store_id,
            )
            return vector_store_id

        except Exception as e:
            self.logger.error(
                "Error validating vector store",
                component="OpenAIResponseManager",
                subcomponent="_validate_and_prepare_vector_store",
                vector_store_id=vector_store_id,
                error=str(e),
            )
            return None

    async def _prepare_tools_configuration(
        self,
        vector_store_id: Optional[str],
        function_definitions: Optional[List[Dict[str, Any]]],
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
                functions=function_definitions,
            )

            is_valid = await self.tool_manager.validate_tools(tools)
            if not is_valid:
                self.logger.warning(
                    "Tool validation failed, proceeding without tools",
                    component="OpenAIResponseManager",
                    subcomponent="_prepare_tools_configuration",
                )
                return []

            self.logger.info(
                "Tools prepared and validated successfully",
                component="OpenAIResponseManager",
                subcomponent="_prepare_tools_configuration",
                tool_count=len(tools),
            )
            return tools

        except Exception as e:
            self.logger.warning(
                "Error preparing tools, proceeding without tools",
                component="OpenAIResponseManager",
                subcomponent="_prepare_tools_configuration",
                error=str(e),
            )
            return []

    async def _extract_response_content(
        self, response: Any
    ) -> tuple[str, List[Any], List[Dict[str, str]]]:
        """
        Extract text content, tool calls, and citations from response object.

        Args:
            response: OpenAI API response object.

        Returns:
            Tuple of (content_text, tool_calls_list, citations_list).
        """
        content = self.chat_manager._extract_text_content(response)

        # Extract tool calls with fallback for different response formats
        tool_calls = []
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if hasattr(item, "type") and "call" in item.type:
                    tool_calls.append(item)
        else:
            tool_calls = getattr(response, "tool_calls", [])

        # Extract citations from response
        citations = await self.citation_manager.extract_citations_from_response(
            response
        )

        # Append citations if present
        if citations:
            content = self.citation_manager.append_citations_to_content(
                content, citations
            )

        return content, tool_calls, citations

    @time_execution("OpenAIResponseManager", "process_query")
    async def process_query(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        function_definitions: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        enable_streaming: bool = False,
        use_drug_database: bool = True,
        enable_tool_execution: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a user query with optional tools and streaming.

        This is the primary method for handling user queries. It supports:
        - New conversations and continuation of existing ones
        - Vector store integration for semantic search
        - Custom function tools
        - Drug database queries via Milvus
        - Automatic tool execution
        - Streaming and non-streaming responses

        Args:
            user_message: The user's message/query text.
            conversation_id: Optional ID to continue an existing conversation.
            vector_store_id: Optional vector store ID for file_search capability.
            function_definitions: Optional list of custom function definitions.
            model_name: Optional model override. Uses default if not specified.
            enable_streaming: If True, returns a streaming generator function.
            use_drug_database: If True, enables drug database search tool (default: True).
            enable_tool_execution: If True, automatically executes tool calls (default: True).

        Returns:
            Dictionary containing:
                - conversation_id: ID for the conversation
                - content: Response text (if not streaming)
                - stream_generator: Async generator function (if streaming)
                - tool_calls: List of tool calls made (if not auto-executed)
                - tool_execution_history: History of tool executions (if auto-executed)
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
                streaming_enabled=enable_streaming,
                use_drug_database=use_drug_database,
                enable_tool_execution=enable_tool_execution,
            )

            # Resolve model configuration
            resolved_model = self._resolve_model_name(model_name)
            self.logger.info(
                "Using model configuration",
                component="OpenAIResponseManager",
                subcomponent="process_query",
                model=resolved_model,
            )

            # CRITICAL: Validate active session if conversation_id is provided
            # This ensures we only use data from active, non-expired sessions
            session_vector_store_id = None
            session_last_response_id = None

            if conversation_id and self.cache_provider:
                try:
                    # Get active session context - this validates the session is still active
                    session_context = (
                        await self.cache_provider.get_active_session_context(
                            conversation_id
                        )
                    )

                    if session_context:
                        # Session is active - use its context
                        session_vector_store_id = session_context.vector_store_id
                        session_last_response_id = session_context.last_response_id

                        self.logger.info(
                            "Active session context retrieved from Redis cache",
                            component="OpenAIResponseManager",
                            subcomponent="process_query",
                            conversation_id=conversation_id,
                            session_vector_store_id=session_vector_store_id,
                            session_last_response_id=session_last_response_id,
                            using_vector_store_from_session=bool(
                                session_vector_store_id
                            ),
                            will_continue_conversation=bool(session_last_response_id),
                        )

                        # Override vector_store_id with session's vector store if not explicitly provided
                        # This ensures consistency within a session
                        if not vector_store_id and session_vector_store_id:
                            vector_store_id = session_vector_store_id
                            self.logger.info(
                                "Using vector store from active session cache",
                                component="OpenAIResponseManager",
                                subcomponent="process_query",
                                retrieved_vector_store_id=session_vector_store_id,
                                conversation_id=conversation_id,
                            )
                    else:
                        # Session is inactive/expired - treat as new conversation
                        self.logger.warning(
                            "Session inactive or expired - treating as new conversation",
                            component="OpenAIResponseManager",
                            subcomponent="process_query",
                            provided_conversation_id=conversation_id,
                        )
                        # Clear conversation_id to trigger new conversation flow
                        conversation_id = None

                except Exception as e:
                    self.logger.warning(
                        f"Failed to retrieve session context: {e}. Treating as new conversation.",
                        component="OpenAIResponseManager",
                        subcomponent="process_query",
                    )
                    # On error, treat as new conversation for safety
                    conversation_id = None

            # Add drug database function if enabled and available
            if use_drug_database and self.drug_data_manager:
                drug_function = self.drug_data_manager.get_function_schema()
                if function_definitions is None:
                    function_definitions = [drug_function]
                else:
                    # Add drug function to existing functions if not already present
                    if not any(
                        f.get("name") == "search_drug_database"
                        for f in function_definitions
                    ):
                        function_definitions = function_definitions + [drug_function]

                self.logger.info(
                    "Drug database tool added to function definitions",
                    component="OpenAIResponseManager",
                    subcomponent="process_query",
                )

            # Prepare tools if needed
            tools = await self._prepare_tools_configuration(
                vector_store_id, function_definitions
            )

            if enable_streaming:
                return await self._handle_streaming_query(
                    user_message,
                    conversation_id,
                    resolved_model,
                    tools,
                    vector_store_id,
                )
            else:
                return await self._handle_standard_query(
                    user_message,
                    conversation_id,
                    resolved_model,
                    tools,
                    vector_store_id,
                    function_definitions,
                    enable_tool_execution,
                )

        except (ToolConfigurationError, VectorStoreError) as e:
            self.logger.error(
                "Tool or vector store error in query processing",
                component="OpenAIResponseManager",
                subcomponent="process_query",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to process query",
                component="OpenAIResponseManager",
                subcomponent="process_query",
                error=str(e),
            )
            raise ResponsesAPIError(message=f"Query processing failed: {str(e)}")

    async def _handle_streaming_query(
        self,
        user_message: str,
        conversation_id: Optional[str],
        model_name: str,
        tools: List[Dict[str, Any]],
        vector_store_id: Optional[str] = None,
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
            has_conversation_id=bool(conversation_id),
        )

        async def create_stream_generator() -> AsyncGenerator[str, None]:
            async for chunk in self.stream_manager.stream_response(
                message=user_message, model=model_name, tools=tools
            ):
                yield chunk

        # Create or use existing conversation
        if not conversation_id:
            # CRITICAL FIX: Generate session ID centrally before creating chat
            session_id = self.chat_manager.generate_session_id()

            conversation_id = await self.chat_manager.create_chat(
                message=user_message,
                model=model_name,
                tools=tools,
                vector_store_id=vector_store_id,
                session_id=session_id,  # Pass session ID explicitly
            )

        self.logger.info(
            "Streaming response configured",
            component="OpenAIResponseManager",
            subcomponent="_handle_streaming_query",
            conversation_id=conversation_id,
        )

        return {
            "conversation_id": conversation_id,
            "stream_generator": create_stream_generator,
            "tools": tools,
        }

    async def _handle_standard_query(
        self,
        user_message: str,
        conversation_id: Optional[str],
        model_name: str,
        tools: List[Dict[str, Any]],
        vector_store_id: Optional[str],
        function_definitions: Optional[List[Dict[str, Any]]],
        enable_tool_execution: bool = True,
    ) -> Dict[str, Any]:
        """
        Handle a standard (non-streaming) query request.

        CRITICAL FIX: Properly handles tool execution by passing the initial response object.
        """
        if conversation_id:
            # Continue existing conversation
            self.logger.info(
                "Continuing existing conversation",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query",
                conversation_id=conversation_id,
                enable_tool_execution=enable_tool_execution,
            )

            if enable_tool_execution and function_definitions:
                # For existing conversations, use the regular continue_chat_with_tools
                # which will detect tool calls and execute them
                result = await self.chat_manager.continue_chat_with_tools(
                    chat_id=conversation_id,
                    message=user_message,
                    vector_store_id=vector_store_id,
                    functions=function_definitions,
                    model=model_name,
                )

                # Check if there are tool calls to execute
                if result.get("tool_calls"):
                    # Retrieve the response object
                    response = await asyncio.to_thread(
                        self.chat_manager.client.responses.retrieve,
                        response_id=result["response_id"],
                    )

                    # Execute tools using the response object
                    execution_result = (
                        await self.chat_manager.continue_chat_with_tool_execution(
                            chat_id=conversation_id,
                            initial_response=response,
                            vector_store_id=vector_store_id,
                            functions=function_definitions,
                            model=model_name,
                        )
                    )

                    return {
                        "conversation_id": execution_result["response_id"],
                        "content": execution_result["content"],
                        "tool_execution_history": execution_result[
                            "tool_execution_history"
                        ],
                        "tools": tools,
                        "citations": execution_result.get("citations", []),
                    }

                # No tool calls, return as-is
                return {
                    "conversation_id": result["response_id"],
                    "content": result["content"],
                    "tool_calls": result.get("tool_calls", []),
                    "tools": tools,
                    "citations": result.get("citations", []),
                }
            else:
                # Original behavior without tool execution
                result = await self.chat_manager.continue_chat_with_tools(
                    chat_id=conversation_id,
                    message=user_message,
                    vector_store_id=vector_store_id,
                    functions=function_definitions,
                    model=model_name,
                )

                return {
                    "conversation_id": result["response_id"],
                    "content": result["content"],
                    "tool_calls": result.get("tool_calls", []),
                    "tools": tools,
                    "citations": result.get("citations", []),
                }
        else:
            # Start new conversation
            self.logger.info(
                "Starting new conversation",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query",
                enable_tool_execution=enable_tool_execution,
            )

            # CRITICAL FIX: Generate session ID centrally before creating chat
            session_id = self.chat_manager.generate_session_id()

            # Create initial chat with explicit session ID
            conversation_id = await self.chat_manager.create_chat(
                message=user_message,
                model=model_name,
                tools=tools,
                vector_store_id=vector_store_id,
                session_id=session_id,  # Pass session ID explicitly
            )

            # Get the response ID from the chat mapping (conversation_id is now session_id)
            response_id = await self.chat_manager._get_chat_mapping(conversation_id)
            if not response_id:
                raise Exception(
                    f"Failed to retrieve response ID for session {conversation_id}"
                )

            # Retrieve the response object using the response_id
            response = await asyncio.to_thread(
                self.chat_manager.client.responses.retrieve, response_id=response_id
            )

            # Extract content and tool calls
            content, tool_calls, citations = await self._extract_response_content(
                response
            )

            self.logger.info(
                f"Initial response created with {len(tool_calls)} tool calls",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query",
                conversation_id=conversation_id,
                has_tool_calls=bool(tool_calls),
            )

            # CRITICAL FIX: If tool execution is enabled and there are tool calls,
            # pass the RESPONSE OBJECT (not a message)
            if enable_tool_execution and tool_calls and function_definitions:
                self.logger.info(
                    "Tool calls detected in new conversation, executing tools",
                    component="OpenAIResponseManager",
                    subcomponent="_handle_standard_query",
                    tool_call_count=len(tool_calls),
                )

                # Pass the response object that already contains tool calls
                result = await self.chat_manager.continue_chat_with_tool_execution(
                    chat_id=conversation_id,
                    initial_response=response,  # Pass response object, not empty message
                    vector_store_id=vector_store_id,
                    functions=function_definitions,
                    model=model_name,
                )

                return {
                    "conversation_id": result["response_id"],
                    "content": result["content"],
                    "tool_execution_history": result["tool_execution_history"],
                    "tools": tools,
                    "citations": result.get("citations", []),
                }

            self.logger.info(
                "New conversation created successfully",
                component="OpenAIResponseManager",
                subcomponent="_handle_standard_query",
                conversation_id=conversation_id,
                has_tool_calls=bool(tool_calls),
            )

            return {
                "conversation_id": conversation_id,
                "content": content,
                "tool_calls": tool_calls,
                "tools": tools,
                "citations": citations,
            }

    @time_execution("OpenAIResponseManager", "process_streaming_query")
    async def process_streaming_query(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        vector_store_id: Optional[str] = None,
        function_definitions: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None,
        use_drug_database: bool = True,
        enable_tool_execution: bool = True,
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
            use_drug_database: If True, enables drug database search tool (default: True).
            enable_tool_execution: If True, enables automatic tool execution (default: True).

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
                has_functions=bool(function_definitions),
                use_drug_database=use_drug_database,
                enable_tool_execution=enable_tool_execution,
            )

            # Resolve model configuration
            resolved_model = self._resolve_model_name(model_name)
            self.logger.info(
                "Using model configuration",
                component="OpenAIResponseManager",
                subcomponent="process_streaming_query",
                model=resolved_model,
            )

            # CRITICAL: Validate active session if conversation_id is provided
            # This ensures we only use data from active, non-expired sessions
            session_vector_store_id = None
            session_last_response_id = None

            if conversation_id and self.cache_provider:
                try:
                    # Get active session context - this validates the session is still active
                    session_context = (
                        await self.cache_provider.get_active_session_context(
                            conversation_id
                        )
                    )

                    if session_context:
                        # Session is active - use its context
                        session_vector_store_id = session_context.vector_store_id
                        session_last_response_id = session_context.last_response_id

                        self.logger.info(
                            "Active session context retrieved from Redis cache",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                            conversation_id=conversation_id,
                            session_vector_store_id=session_vector_store_id,
                            session_last_response_id=session_last_response_id,
                            using_vector_store_from_session=bool(
                                session_vector_store_id
                            ),
                            will_continue_conversation=bool(session_last_response_id),
                        )

                        # Override vector_store_id with session's vector store if not explicitly provided
                        # This ensures consistency within a session
                        if not vector_store_id and session_vector_store_id:
                            vector_store_id = session_vector_store_id
                            self.logger.info(
                                "Using vector store from active session cache",
                                component="OpenAIResponseManager",
                                subcomponent="process_streaming_query",
                                retrieved_vector_store_id=session_vector_store_id,
                                conversation_id=conversation_id,
                            )
                    else:
                        # Session is inactive/expired - treat as new conversation
                        self.logger.warning(
                            "Session inactive or expired - treating as new conversation",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                            provided_conversation_id=conversation_id,
                        )
                        # Clear conversation_id to trigger new conversation flow
                        conversation_id = None

                except Exception as e:
                    self.logger.warning(
                        f"Failed to retrieve session context: {e}. Treating as new conversation.",
                        component="OpenAIResponseManager",
                        subcomponent="process_streaming_query",
                    )
                    # On error, treat as new conversation for safety
                    conversation_id = None

            # Add drug database function if enabled and available
            if use_drug_database and self.drug_data_manager:
                drug_function = self.drug_data_manager.get_function_schema()
                if function_definitions is None:
                    function_definitions = [drug_function]
                else:
                    # Add drug function to existing functions if not already present
                    if not any(
                        f.get("name") == "search_drug_database"
                        for f in function_definitions
                    ):
                        function_definitions = function_definitions + [drug_function]

                self.logger.info(
                    "Drug database tool added to streaming function definitions",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                )

            # Prepare tools
            tools = await self._prepare_tools_configuration(
                vector_store_id, function_definitions
            )

            chunk_count = 0

            # CRITICAL FIX: Check if this is a true continuation (has previous messages)
            # A session can exist without messages yet (e.g., created by UI on_chat_start)
            if conversation_id and session_last_response_id:
                # Stream continuation of existing conversation with previous messages
                # Use last_response_id from session context for proper conversation chaining
                previous_response_id_for_continuation = session_last_response_id

                self.logger.info(
                    "Streaming conversation continuation",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                    conversation_id=conversation_id,
                    previous_response_id=previous_response_id_for_continuation,
                    session_vector_store_id=session_vector_store_id,
                )

                # =========================================================================
                # CRITICAL: Track user message in continuation
                # =========================================================================
                if conversation_id and conversation_id.startswith("session_"):
                    try:
                        await self.chat_manager._add_message_to_session(
                            session_id=conversation_id,
                            role="user",
                            content=user_message,
                            metadata={
                                "model": resolved_model,
                                "has_tools": bool(tools),
                                "previous_response_id": previous_response_id_for_continuation,
                            },
                        )
                        self.logger.debug(
                            "Tracked user message in continuation",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                            session_id=conversation_id,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to add user message to continuation: {e}",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                        )

                continuation_response_id = None
                async for chunk_data in self.stream_manager.stream_chat_continuation(
                    chat_id=previous_response_id_for_continuation,
                    message=user_message,
                    model=resolved_model,
                    tools=tools,
                ):
                    chunk_count += 1

                    # Extract response_id from chunks
                    if continuation_response_id is None and chunk_data.get(
                        "response_id"
                    ):
                        continuation_response_id = chunk_data["response_id"]

                    yield {
                        "chunk": chunk_data.get("text", ""),
                        "conversation_id": conversation_id,
                        "tools": tools,
                        "is_citation": chunk_data.get("is_citation", False),
                        "citations": chunk_data.get("citations", []),
                    }

                # Cache the continuation response
                if continuation_response_id:
                    try:
                        # Update chat mapping with new response_id
                        await self.chat_manager._set_chat_mapping(
                            conversation_id, continuation_response_id
                        )

                        # Track response chain
                        if self.cache_provider:
                            try:
                                await self.cache_provider.append_response(
                                    conversation_id, continuation_response_id
                                )
                                await self.cache_provider.set_last_response_id(
                                    conversation_id, continuation_response_id
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to track response chain: {e}",
                                    component="OpenAIResponseManager",
                                    subcomponent="process_streaming_query",
                                )

                        # =========================================================================
                        # CRITICAL: Track assistant response in continuation
                        # =========================================================================
                        if conversation_id and conversation_id.startswith("session_"):
                            try:
                                # Retrieve the complete response to get full text
                                import asyncio

                                final_response = await asyncio.to_thread(
                                    self.chat_manager.client.responses.retrieve,
                                    response_id=continuation_response_id,
                                )

                                if final_response:
                                    # Extract text content
                                    assistant_content = (
                                        self.chat_manager._extract_text_content(
                                            final_response
                                        )
                                    )

                                    # Add to session history
                                    await self.chat_manager._add_message_to_session(
                                        session_id=conversation_id,
                                        role="assistant",
                                        content=assistant_content,
                                        metadata={
                                            "response_id": continuation_response_id,
                                            "model": resolved_model,
                                            "is_continuation": True,
                                        },
                                    )

                                    self.logger.debug(
                                        "Tracked assistant response in continuation",
                                        component="OpenAIResponseManager",
                                        subcomponent="process_streaming_query",
                                        session_id=conversation_id,
                                        response_id=continuation_response_id,
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to add assistant response to continuation: {e}",
                                    component="OpenAIResponseManager",
                                    subcomponent="process_streaming_query",
                                )

                        self.logger.debug(
                            "Cached conversation continuation",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                            conversation_id=conversation_id,
                            new_response_id=continuation_response_id,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to cache conversation continuation: {e}",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                        )

                self.logger.info(
                    "Conversation continuation streaming completed",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                    conversation_id=conversation_id,
                    chunk_count=chunk_count,
                )
            else:
                # CRITICAL: Handle new conversation (first message in session)
                # This includes two cases:
                # 1. conversation_id provided but session has no messages yet (first message)
                # 2. No conversation_id provided at all (completely new session)
                # In both cases, we stream the first response without previous_response_id
                self.logger.info(
                    "Starting new streaming conversation",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                    has_session_id=bool(conversation_id),
                    enable_tool_execution=enable_tool_execution,
                    has_function_definitions=bool(function_definitions),
                )

                # =========================================================================
                # CRITICAL: Track user message BEFORE streaming begins
                # =========================================================================
                if conversation_id and conversation_id.startswith("session_"):
                    try:
                        await self.chat_manager._add_message_to_session(
                            session_id=conversation_id,
                            role="user",
                            content=user_message,
                            metadata={
                                "model": resolved_model,
                                "has_tools": bool(tools),
                                "tool_count": len(tools) if tools else 0,
                            },
                        )
                        self.logger.debug(
                            "Tracked user message in new conversation",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                            session_id=conversation_id,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to add user message to session: {e}",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                        )

                # Stream the response directly - no create_chat call
                response_id = None

                # Choose streaming method based on tool execution needs
                if enable_tool_execution and function_definitions:
                    # Use tool execution streaming
                    self.logger.info(
                        "Using streaming with tool execution",
                        component="OpenAIResponseManager",
                        subcomponent="process_streaming_query",
                        function_count=len(function_definitions),
                    )
                    stream_source = (
                        self.stream_manager.stream_response_with_tool_execution(
                            message=user_message,
                            model=resolved_model,
                            tools=tools,
                            vector_store_id=vector_store_id,
                            functions=function_definitions,
                        )
                    )
                else:
                    # Use regular streaming (no tool execution)
                    stream_source = self.stream_manager.stream_response(
                        message=user_message, model=resolved_model, tools=tools
                    )

                async for chunk_data in stream_source:
                    chunk_count += 1

                    # Extract response_id from chunks (keep updating to get the final one)
                    if chunk_data.get("response_id"):
                        response_id = chunk_data["response_id"]

                    yield {
                        "chunk": chunk_data.get("text", ""),
                        "conversation_id": response_id,  # Use extracted response_id instead of None
                        "tools": tools,
                        "is_citation": chunk_data.get("is_citation", False),
                        "citations": chunk_data.get("citations", []),
                    }

                # Cache the conversation after streaming completes
                if response_id:
                    try:
                        # CRITICAL FIX: Use existing conversation_id if provided (from on_chat_start),
                        # otherwise generate a new session_id
                        # This prevents duplicate session creation when UI already created a session
                        if conversation_id and conversation_id.startswith("session_"):
                            # Reuse the session that was already created (e.g., by Chainlit on_chat_start)
                            session_id = conversation_id
                            self.logger.info(
                                "Reusing existing session for new conversation",
                                component="OpenAIResponseManager",
                                subcomponent="process_streaming_query",
                                session_id=session_id,
                                response_id=response_id,
                            )
                        else:
                            # Generate new session_id only if no valid session was provided
                            session_id = self.chat_manager.generate_session_id()
                            self.logger.info(
                                "Generated new session for conversation",
                                component="OpenAIResponseManager",
                                subcomponent="process_streaming_query",
                                session_id=session_id,
                                response_id=response_id,
                            )

                        # Cache chat mapping: session_id -> response_id
                        await self.chat_manager._set_chat_mapping(
                            session_id, response_id
                        )

                        # Create session metadata with proper session_id
                        await self.chat_manager._set_session_metadata(
                            session_id=session_id,
                            vector_store_id=vector_store_id,
                            root_response_id=response_id,
                            last_response_id=response_id,  # Also set as last_response_id
                        )

                        # Track response chain
                        if self.cache_provider:
                            try:
                                await self.cache_provider.append_response(
                                    session_id, response_id
                                )
                                await self.cache_provider.set_last_response_id(
                                    session_id, response_id
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to track response chain: {e}",
                                    component="OpenAIResponseManager",
                                    subcomponent="process_streaming_query",
                                )

                        # =========================================================================
                        # CRITICAL: Track assistant response AFTER streaming completes
                        # =========================================================================
                        if session_id and session_id.startswith("session_"):
                            try:
                                # Retrieve the complete response to get full text
                                import asyncio

                                final_response = await asyncio.to_thread(
                                    self.chat_manager.client.responses.retrieve,
                                    response_id=response_id,
                                )

                                if final_response:
                                    # Extract text content
                                    assistant_content = (
                                        self.chat_manager._extract_text_content(
                                            final_response
                                        )
                                    )

                                    # Add to session history
                                    await self.chat_manager._add_message_to_session(
                                        session_id=session_id,
                                        role="assistant",
                                        content=assistant_content,
                                        metadata={
                                            "response_id": response_id,
                                            "model": resolved_model,
                                            "has_citations": bool(
                                                chunk_data.get("citations")
                                            )
                                            if "chunk_data" in locals()
                                            else False,
                                        },
                                    )

                                    self.logger.debug(
                                        "Tracked assistant response in new conversation",
                                        component="OpenAIResponseManager",
                                        subcomponent="process_streaming_query",
                                        session_id=session_id,
                                        response_id=response_id,
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to add assistant response to session: {e}",
                                    component="OpenAIResponseManager",
                                    subcomponent="process_streaming_query",
                                )

                        self.logger.info(
                            "Cached new conversation session",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                            session_id=session_id,
                            response_id=response_id,
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to cache conversation session: {e}",
                            component="OpenAIResponseManager",
                            subcomponent="process_streaming_query",
                        )

                self.logger.info(
                    "New conversation streaming completed",
                    component="OpenAIResponseManager",
                    subcomponent="process_streaming_query",
                    chunk_count=chunk_count,
                    conversation_id=response_id,
                )
        except (ToolConfigurationError, VectorStoreError, StreamConnectionError) as e:
            self.logger.error(
                "Error in streaming query processing",
                component="OpenAIResponseManager",
                subcomponent="process_streaming_query",
                error=str(e),
                error_type=type(e).__name__,
            )
            yield {"error": str(e), "conversation_id": conversation_id, "tools": []}
        except Exception as e:
            self.logger.error(
                "Failed to process streaming query",
                component="OpenAIResponseManager",
                subcomponent="process_streaming_query",
                error=str(e),
            )
            yield {
                "error": f"Streaming query processing failed: {str(e)}",
                "conversation_id": conversation_id,
                "tools": [],
            }

    @time_execution("OpenAIResponseManager", "retrieve_conversation_history")
    async def retrieve_conversation_history(
        self, conversation_id: str
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
                conversation_id=conversation_id,
            )

            history = await self.chat_manager.get_chat_history(conversation_id)

            self.logger.info(
                "Conversation history retrieved successfully",
                component="OpenAIResponseManager",
                subcomponent="retrieve_conversation_history",
                conversation_id=conversation_id,
                message_count=len(history),
            )

            return history
        except Exception as e:
            self.logger.error(
                "Failed to retrieve conversation history",
                component="OpenAIResponseManager",
                subcomponent="retrieve_conversation_history",
                conversation_id=conversation_id,
                error=str(e),
            )
            raise ResponsesAPIError(message=f"History retrieval failed: {str(e)}")

    @time_execution("OpenAIResponseManager", "cleanup_resources")
    async def cleanup_resources(
        self,
        vector_store_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        clear_all_caches: bool = True,
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
                clear_caches=clear_all_caches,
            )

            cleanup_status = {}

            # Clean up vector store
            if vector_store_id:
                try:
                    cleanup_status[
                        "vector_store"
                    ] = await self.vector_store_manager.delete_vector_store(
                        vector_store_id
                    )
                    self.logger.info(
                        "Vector store cleanup completed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        vector_store_id=vector_store_id,
                        success=cleanup_status["vector_store"],
                    )
                except Exception as e:
                    self.logger.error(
                        "Vector store cleanup failed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        vector_store_id=vector_store_id,
                        error=str(e),
                    )
                    cleanup_status["vector_store"] = False

            # Clean up conversation
            if conversation_id:
                try:
                    cleanup_status[
                        "conversation"
                    ] = await self.chat_manager.delete_chat(conversation_id)
                    self.logger.info(
                        "Conversation cleanup completed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        conversation_id=conversation_id,
                        success=cleanup_status["conversation"],
                    )
                except Exception as e:
                    self.logger.error(
                        "Conversation cleanup failed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        conversation_id=conversation_id,
                        error=str(e),
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
                        subcomponent="cleanup_resources",
                    )
                except Exception as e:
                    self.logger.error(
                        "Cache clearing failed",
                        component="OpenAIResponseManager",
                        subcomponent="cleanup_resources",
                        error=str(e),
                    )
                    cleanup_status["caches_cleared"] = False

            return cleanup_status

        except Exception as e:
            self.logger.error(
                "Resource cleanup error",
                component="OpenAIResponseManager",
                subcomponent="cleanup_resources",
                error=str(e),
            )
            return {"error": str(e)}

    @time_execution("OpenAIResponseManager", "create_vector_store_from_guidelines")
    async def create_vector_store_from_guidelines(
        self,
        poll_for_completion: bool = True,
        max_wait_seconds: int = VECTOR_STORE_MAX_WAIT_SECONDS,
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
                max_wait_seconds=max_wait_seconds,
            )

            vector_store_id = (
                await self.vector_store_manager.create_guidelines_vector_store()
            )

            self.logger.info(
                "Vector store created",
                component="OpenAIResponseManager",
                subcomponent="create_vector_store_from_guidelines",
                vector_store_id=vector_store_id,
            )

            if not poll_for_completion:
                return vector_store_id

            # Poll for completion
            total_wait_seconds = 0

            while total_wait_seconds < max_wait_seconds:
                store_info = await self.vector_store_manager.get_vector_store(
                    vector_store_id
                )
                status = store_info["status"] if store_info else "not_found"

                self.logger.info(
                    "Polling vector store status",
                    component="OpenAIResponseManager",
                    subcomponent="create_vector_store_from_guidelines",
                    vector_store_id=vector_store_id,
                    status=status,
                    elapsed_seconds=total_wait_seconds,
                )

                if store_info and store_info["status"] == "completed":
                    self.logger.info(
                        "Vector store ready",
                        component="OpenAIResponseManager",
                        subcomponent="create_vector_store_from_guidelines",
                        vector_store_id=vector_store_id,
                        total_wait_seconds=total_wait_seconds,
                    )
                    return vector_store_id

                if store_info and store_info["status"] == "failed":
                    self.logger.error(
                        "Vector store creation failed",
                        component="OpenAIResponseManager",
                        subcomponent="create_vector_store_from_guidelines",
                        vector_store_id=vector_store_id,
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
                max_wait_seconds=max_wait_seconds,
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
                error=str(e),
            )
            raise ResponsesAPIError(message=f"Vector store creation failed: {str(e)}")

    async def get_vector_store_status(self, vector_store_id: str) -> Dict[str, Any]:
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
                vector_store_id=vector_store_id,
            )

            store_info = await self.vector_store_manager.get_vector_store(
                vector_store_id
            )

            if not store_info:
                self.logger.warning(
                    "Vector store not found",
                    component="OpenAIResponseManager",
                    subcomponent="get_vector_store_status",
                    vector_store_id=vector_store_id,
                )
                raise VectorStoreError(f"Vector store {vector_store_id} not found")

            self.logger.info(
                "Vector store status retrieved",
                component="OpenAIResponseManager",
                subcomponent="get_vector_store_status",
                vector_store_id=vector_store_id,
                status=store_info.get("status"),
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
                error=str(e),
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
                subcomponent="clear_all_caches",
            )

            self.tool_manager.clear_tool_cache()
            self.chat_manager.clear_cache()
            self.vector_store_manager.clear_cache()

            self.logger.info(
                "All caches cleared successfully",
                component="OpenAIResponseManager",
                subcomponent="clear_all_caches",
            )
        except Exception as e:
            self.logger.error(
                "Error clearing caches",
                component="OpenAIResponseManager",
                subcomponent="clear_all_caches",
                error=str(e),
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
            subcomponent="validate_configuration",
        )

        validation_results = {
            "settings_loaded": bool(self.settings),
            "vector_store_manager": bool(self.vector_store_manager),
            "tool_manager": bool(self.tool_manager),
            "chat_manager": bool(self.chat_manager),
            "stream_manager": bool(self.stream_manager),
            "model_name_valid": self._validate_model_identifier(
                self.settings.openai_model_name
            ),
        }

        all_valid = all(validation_results.values())

        self.logger.info(
            "Configuration validation completed",
            component="OpenAIResponseManager",
            subcomponent="validate_configuration",
            all_valid=all_valid,
            results=validation_results,
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
            if hasattr(self.chat_manager, "chat_history_limit")
            else None,
            "vector_store_batch_size": self.vector_store_manager.batch_size
            if hasattr(self.vector_store_manager, "batch_size")
            else None,
            "components": {
                "vector_store_manager": type(self.vector_store_manager).__name__,
                "tool_manager": type(self.tool_manager).__name__,
                "chat_manager": type(self.chat_manager).__name__,
                "stream_manager": type(self.stream_manager).__name__,
            },
        }
