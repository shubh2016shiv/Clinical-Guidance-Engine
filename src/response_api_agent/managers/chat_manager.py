import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from openai import OpenAI, AsyncOpenAI
from src.config import get_settings
from src.response_api_agent.managers.tool_manager import (
    ToolManager,
)  # Optional integration for tools
from src.response_api_agent.managers.citation_manager import CitationManager
from src.response_api_agent.managers.llm_provider_adapter import ResponseAPIAdapter
from src.response_api_agent.managers.exceptions import (
    ResponsesAPIError,
    ContentParsingError,
    ToolConfigurationError,
)
from src.logs import get_component_logger, time_execution
from src.prompts.asclepius_system_prompt import get_system_prompt


class ChatManager:
    """
    Manages conversations for the OpenAI Responses API using response chaining.

    Handles stateful interactions via `previous_response_id` for server-side history persistence.
    No client-side message storage—relies on API chaining and retrieval.
    Includes client-side limit on chain length to prevent excessively long conversations.
    """

    def __init__(
        self, tool_manager: Optional[ToolManager] = None, chat_history_limit: int = 10
    ):
        """Initialize the Chat Manager.

        Args:
            tool_manager: Optional ToolManager for including tools in responses.
            chat_history_limit: Maximum number of previous responses in the chain (default: 10).
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.response_adapter = ResponseAPIAdapter(self.client, self.async_client)
        self.tool_manager = tool_manager
        self.chat_history_limit = chat_history_limit
        self._chat_cache: Dict[str, str] = {}  # Cache: chat_id -> last_response_id
        self.citation_manager = CitationManager(client=self.client)
        self.logger = get_component_logger("Chat")

        # Registry for function tool executors
        self._function_executors: Dict[str, Callable] = {}

    def register_function_executor(
        self, function_name: str, executor: Callable
    ) -> None:
        """
        Register a function executor for tool calls.

        Args:
            function_name: Name of the function tool
            executor: Callable that executes the function (can be async or sync)
        """
        self._function_executors[function_name] = executor
        self.logger.info(
            f"Registered function executor: {function_name}",
            component="Chat",
            subcomponent="RegisterFunctionExecutor",
        )

    async def _execute_function_call(
        self, function_name: str, arguments: Dict[str, Any]
    ) -> str:
        """
        Execute a function tool call.

        Args:
            function_name: Name of the function to execute
            arguments: Dictionary of function arguments

        Returns:
            JSON string result from function execution
        """
        if function_name not in self._function_executors:
            error_msg = f"No executor registered for function: {function_name}"
            self.logger.error(
                error_msg,
                component="Chat",
                subcomponent="ExecuteFunctionCall",
            )
            return json.dumps({"error": error_msg})

        try:
            executor = self._function_executors[function_name]

            # Execute (handle both sync and async executors)
            if asyncio.iscoroutinefunction(executor):
                result = await executor(**arguments)
            else:
                result = executor(**arguments)

            self.logger.info(
                f"Successfully executed function: {function_name}",
                component="Chat",
                subcomponent="ExecuteFunctionCall",
                arguments=arguments,
            )

            return result if isinstance(result, str) else json.dumps(result)

        except Exception as e:
            error_msg = f"Error executing function {function_name}: {str(e)}"
            self.logger.error(
                error_msg,
                component="Chat",
                subcomponent="ExecuteFunctionCall",
                exc_info=True,
            )
            return json.dumps({"error": error_msg})

    @time_execution("Chat", "ExtractTextContent")
    def _extract_text_content(self, response):
        """
        Extract text content from a Responses API response object.

        Args:
            response: Response object from the Responses API.

        Returns:
            Extracted text content as string.
        """
        try:
            response_id = getattr(response, "id", "unknown")

            self.logger.info(
                "[DEBUG] Extracting text content from response",
                component="Chat",
                subcomponent="ExtractTextContent",
                response_id=response_id,
                response_type=type(response).__name__,
            )

            # Check for the new response structure with output field
            if hasattr(response, "output") and response.output:
                # CRITICAL FIX: Detect if response contains tool calls (valid scenario with no text)
                has_tool_calls = any(
                    getattr(item, "type", None) == "function_call"
                    for item in response.output
                )

                text_parts = []
                for item in response.output:
                    # Look for ResponseOutputMessage type items
                    if hasattr(item, "type") and item.type == "message":
                        if hasattr(item, "content") and item.content:
                            for content_block in item.content:
                                # Extract text from ResponseOutputText
                                if (
                                    hasattr(content_block, "type")
                                    and content_block.type == "output_text"
                                ):
                                    if hasattr(content_block, "text"):
                                        text_parts.append(content_block.text)

                if text_parts:
                    self.logger.info(
                        "[DEBUG] Extracted text from output structure",
                        component="Chat",
                        subcomponent="ExtractTextContent",
                        response_id=response_id,
                        text_parts_count=len(text_parts),
                    )
                    return "\n".join(text_parts)

                # CRITICAL FIX: Handle tool-call-only responses gracefully
                # When response contains only tool calls and no text, this is valid behavior
                if has_tool_calls:
                    self.logger.info(
                        "Response contains tool calls but no text content (valid scenario - model chose to call tools)",
                        component="Chat",
                        subcomponent="ExtractTextContent",
                        response_id=response_id,
                        has_tool_calls=True,
                    )
                    return ""

            # Legacy format check
            elif hasattr(response, "content") and response.content:
                # Process content array
                text_parts = []
                for content_block in response.content:
                    if hasattr(content_block, "text") and hasattr(
                        content_block.text, "value"
                    ):
                        text_parts.append(content_block.text.value)

                if text_parts:
                    self.logger.info(
                        "[DEBUG] Extracted text from legacy content structure",
                        component="Chat",
                        subcomponent="ExtractTextContent",
                        response_id=response_id,
                        text_parts_count=len(text_parts),
                    )
                    return "\n".join(text_parts)

            # Fallback for unexpected response structure
            self.logger.warning(
                "Unable to extract text from response structure",
                component="Chat",
                subcomponent="ExtractTextContent",
                response_id=response_id,
                response_type=type(response).__name__,
            )
            return str(response)
        except Exception as e:
            self.logger.warning(
                "Error extracting text content",
                component="Chat",
                subcomponent="ExtractTextContent",
                response_id=getattr(response, "id", "unknown"),
                error=str(e),
            )
            raise ContentParsingError(f"Failed to parse response content: {str(e)}")

    def _extract_tool_calls_from_response(self, response) -> List[Dict[str, Any]]:
        """
        Extract tool calls from a response object in normalized format.

        Handles both new (response.output) and legacy (response.tool_calls) structures.

        Args:
            response: Response object from the Responses API.

        Returns:
            List of dicts with keys: 'call_id', 'type', 'function_name', 'arguments'
        """
        tool_calls = []
        file_search_detected = False
        function_calls_detected = []

        try:
            # DEBUG: Log the entire response object structure
            self.logger.info(
                "DEBUG: Full response object for tool call extraction",
                component="Chat",
                subcomponent="ExtractToolCalls",
                response_vars=vars(response)
                if hasattr(response, "__dict__")
                else str(response),
            )
            # Check for new response structure with output field
            if hasattr(response, "output") and response.output:
                # First pass: detect file_search tool
                for item in response.output:
                    if hasattr(item, "type") and item.type == "file_search":
                        file_search_detected = True
                        self.logger.info(
                            "FILE_SEARCH TOOL DETECTED IN RESPONSE OUTPUT",
                            component="Chat",
                            subcomponent="ExtractToolCalls",
                            tool_type="file_search",
                        )

                # Second pass: extract function calls
                for item in response.output:
                    if hasattr(item, "type") and item.type == "function_call":
                        # Extract function call details from new format
                        call_id = getattr(item, "call_id", None)
                        function_name = getattr(item, "name", None)
                        # Handle None, empty string, or missing attribute cases
                        arguments_raw = getattr(item, "arguments", None)
                        arguments_str = (
                            arguments_raw if arguments_raw not in (None, "") else "{}"
                        )

                        # Debug logging to inspect raw arguments value
                        if arguments_raw is None or arguments_raw == "":
                            self.logger.info(
                                "[DEBUG] Function call has None or empty arguments, using default",
                                component="Chat",
                                subcomponent="ExtractToolCalls",
                                call_id=call_id,
                                function_name=function_name,
                                raw_arguments=arguments_raw,
                                item_type=type(item).__name__,
                                item_attrs=[
                                    attr
                                    for attr in dir(item)
                                    if not attr.startswith("_")
                                ],
                            )

                        if call_id and function_name:
                            try:
                                # Parse arguments JSON string if needed
                                arguments = (
                                    json.loads(arguments_str)
                                    if isinstance(arguments_str, str)
                                    else arguments_str
                                )
                            except json.JSONDecodeError:
                                self.logger.warning(
                                    "Failed to parse function call arguments as JSON",
                                    component="Chat",
                                    subcomponent="ExtractToolCalls",
                                    call_id=call_id,
                                    arguments_str=arguments_str,
                                )
                                arguments = {}

                            tool_calls.append(
                                {
                                    "call_id": call_id,
                                    "type": "function_call",
                                    "function_name": function_name,
                                    "arguments": arguments,
                                }
                            )
                            function_calls_detected.append(function_name)

            # Legacy format check - response.tool_calls
            elif hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    if hasattr(tool_call, "function"):
                        call_id = getattr(tool_call, "id", None)
                        function_name = tool_call.function.name
                        # Handle None, empty string, or missing attribute cases
                        arguments_raw = getattr(tool_call.function, "arguments", None)
                        arguments_str = (
                            arguments_raw if arguments_raw not in (None, "") else "{}"
                        )

                        # Debug logging to inspect raw arguments value
                        if arguments_raw is None or arguments_raw == "":
                            self.logger.info(
                                "[DEBUG] Function call has None or empty arguments, using default",
                                component="Chat",
                                subcomponent="ExtractToolCalls",
                                call_id=call_id,
                                function_name=function_name,
                                raw_arguments=arguments_raw,
                                tool_call_type=type(tool_call).__name__,
                            )

                        if call_id and function_name:
                            try:
                                # Parse arguments JSON string if needed
                                arguments = (
                                    json.loads(arguments_str)
                                    if isinstance(arguments_str, str)
                                    else arguments_str
                                )
                            except json.JSONDecodeError:
                                self.logger.warning(
                                    "Failed to parse function call arguments as JSON",
                                    component="Chat",
                                    subcomponent="ExtractToolCalls",
                                    call_id=call_id,
                                    arguments_str=arguments_str,
                                )
                                arguments = {}

                            tool_calls.append(
                                {
                                    "call_id": call_id,
                                    "type": "function_call",
                                    "function_name": function_name,
                                    "arguments": arguments,
                                }
                            )
                            function_calls_detected.append(function_name)

            if tool_calls:
                self.logger.info(
                    f"Extracted {len(tool_calls)} tool calls from response",
                    component="Chat",
                    subcomponent="ExtractToolCalls",
                )
                # Log extracted arguments for debugging
                for tool_call in tool_calls:
                    self.logger.info(
                        "[DEBUG] Extracted tool call details",
                        component="Chat",
                        subcomponent="ExtractToolCalls",
                        call_id=tool_call.get("call_id"),
                        function_name=tool_call.get("function_name"),
                        arguments=tool_call.get("arguments"),
                    )

            # Emphasized logging: determine which tools were triggered
            tool_trigger_type = None
            if file_search_detected and function_calls_detected:
                tool_trigger_type = "BOTH"
            elif file_search_detected:
                tool_trigger_type = "FILE_SEARCH"
            elif function_calls_detected:
                tool_trigger_type = "FUNCTION_CALL"

            if tool_trigger_type:
                self.logger.info(
                    f"--- TOOL TRIGGER DETECTED ---\n"
                    f"TYPE: {tool_trigger_type}\n"
                    f"FILE_SEARCH: {file_search_detected}\n"
                    f"FUNCTION_CALLS: {len(function_calls_detected)}\n"
                    f"FUNCTION_NAMES: {function_calls_detected}\n"
                    f"-----------------------------",
                    component="Chat",
                    subcomponent="ExtractToolCalls",
                    file_search_detected=file_search_detected,
                    function_calls_count=len(function_calls_detected),
                    function_names=function_calls_detected,
                    total_tool_calls=len(tool_calls),
                )

            # CRITICAL INFO LOG: Function calls detected with arguments - visible in logs
            if tool_calls:
                for tool_call in tool_calls:
                    function_name = tool_call.get("function_name", "unknown")
                    arguments = tool_call.get("arguments", {})
                    call_id = tool_call.get("call_id", "unknown")

                    self.logger.info(
                        f"FUNCTION CALL DETECTED: {function_name}",
                        component="Chat",
                        subcomponent="ExtractToolCalls",
                        function_name=function_name,
                        call_id=call_id,
                        arguments_extracted=arguments,
                        arguments_count=len(arguments),
                        is_empty=not bool(arguments),
                    )

        except Exception as e:
            self.logger.error(
                f"Error extracting tool calls: {e}",
                component="Chat",
                subcomponent="ExtractToolCalls",
                exc_info=True,
            )

        return tool_calls

    @time_execution("Chat", "CreateChat")
    async def create_chat(
        self,
        message: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        streaming: bool = False,
    ) -> str:
        """
        Start a new conversation by creating the first response.

        Args:
            message: Initial user message.
            model: Model to use (default: from settings).
            tools: Optional tools (e.g., from ToolManager) to include.
            streaming: If True, optimizes for streaming (returns placeholder ID).

        Returns:
            Chat ID (the initial response ID, used as chain root).
        """
        try:
            model = model or self.settings.openai_model_name
            self.logger.info(
                "Creating new chat with initial message",
                component="Chat",
                subcomponent="CreateChat",
                model=model,
                has_tools=bool(tools),
                streaming=streaming,
            )

            if streaming:
                # For streaming, we don't need to wait for the full response
                # Just return a placeholder ID that will be updated after streaming
                import time

                chat_id = f"streaming_{int(time.time())}"
                self._chat_cache[chat_id] = None  # Will be updated after streaming
                self.logger.info(
                    "Created streaming chat placeholder",
                    component="Chat",
                    subcomponent="CreateChat",
                    chat_id=chat_id,
                )
                return chat_id

            # Non-streaming: create full response
            response = await self.response_adapter.create_response(
                model=model,
                input=message,  # User input; API handles as first message
                instructions=get_system_prompt(),
                tools=tools or [],
            )

            chat_id = response.id  # Use response ID as chat identifier
            self._chat_cache[chat_id] = response.id  # Cache last response ID

            self.logger.info(
                "Created chat successfully",
                component="Chat",
                subcomponent="CreateChat",
                chat_id=chat_id,
                response_id=response.id,
            )
            return chat_id

        except Exception as e:
            self.logger.error(
                "Failed to create chat",
                component="Chat",
                subcomponent="CreateChat",
                error=str(e),
            )
            raise ResponsesAPIError(message=f"Failed to create chat: {str(e)}")

    @time_execution("Chat", "ContinueChat")
    async def continue_chat(
        self,
        chat_id: str,
        message: str,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Continue a conversation by chaining a new response, with history limit enforcement.

        Args:
            chat_id: Existing chat ID (root or last response ID).
            message: New user message.
            model: Optional model override.
            tools: Optional tools to include.

        Returns:
            New response ID (update chat_id for next call).
        """
        try:
            if chat_id not in self._chat_cache:
                # Fallback: If not cached, assume chat_id is the last response ID
                last_response_id = chat_id
            else:
                last_response_id = self._chat_cache[chat_id]

            model = model or self.settings.openai_model_name

            self.logger.info(
                "Continuing chat",
                component="Chat",
                subcomponent="ContinueChat",
                chat_id=chat_id,
                last_response_id=last_response_id,
                model=model,
                has_tools=bool(tools),
            )

            # Check current chain length
            history = await self.get_chat_history(chat_id)
            current_length = len(history)

            if current_length >= self.chat_history_limit:
                self.logger.info(
                    "Chat reached history limit; summarizing and resetting chain",
                    component="Chat",
                    subcomponent="ContinueChat",
                    chat_id=chat_id,
                    history_limit=self.chat_history_limit,
                    current_length=current_length,
                )
                # Summarize old history
                summary = await self._summarize_history(history, model)
                # Create new root response from summary + new message
                reset_message = f"{summary}\n\nNew query: {message}"
                new_response = await self.response_adapter.create_response(
                    model=model,
                    input=reset_message,
                    instructions=get_system_prompt(),
                    tools=tools or [],
                )
                # Update chat_id to new root and cache
                chat_id = new_response.id
                self._chat_cache[chat_id] = new_response.id

                self.logger.info(
                    "Reset chat to new root after summarization",
                    component="Chat",
                    subcomponent="ContinueChat",
                    new_chat_id=chat_id,
                    response_id=new_response.id,
                )
                return chat_id
            else:
                # Chain new response normally
                response = await self.response_adapter.create_response(
                    model=model,
                    previous_response_id=last_response_id,
                    input=message,
                    instructions=get_system_prompt(),
                    tools=tools or [],
                )

                # Update cache: chat_id -> new response ID
                self._chat_cache[chat_id] = response.id

                self.logger.info(
                    "Continued chat with new response",
                    component="Chat",
                    subcomponent="ContinueChat",
                    chat_id=chat_id,
                    response_id=response.id,
                    chain_length=current_length + 1,
                )
                return (
                    response.id
                )  # Return new ID for immediate use, but chat_id remains root

        except ContentParsingError as e:
            self.logger.error(
                "Content parsing error in continue_chat",
                component="Chat",
                subcomponent="ContinueChat",
                chat_id=chat_id,
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to continue chat",
                component="Chat",
                subcomponent="ContinueChat",
                chat_id=chat_id,
                error=str(e),
            )
            raise ResponsesAPIError(message=f"Failed to continue chat: {str(e)}")

    @time_execution("Chat", "ContinueChatWithTools")
    async def continue_chat_with_tools(
        self,
        chat_id: str,
        message: str,
        vector_store_id: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Continue a conversation with tools from ToolManager.

        Args:
            chat_id: Existing chat ID.
            message: New user message.
            vector_store_id: Optional vector store ID for file_search.
            functions: Optional function definitions for function calling.
            model: Optional model override.

        Returns:
            Dict with response ID and processed response content.
        """
        try:
            if not self.tool_manager:
                self.logger.error(
                    "Tool manager is required for this operation",
                    component="Chat",
                    subcomponent="ContinueChatWithTools",
                    chat_id=chat_id,
                )
                raise ToolConfigurationError(
                    "Tool manager is required for this operation"
                )

            self.logger.info(
                "Continuing chat with tools",
                component="Chat",
                subcomponent="ContinueChatWithTools",
                chat_id=chat_id,
                has_vector_store=bool(vector_store_id),
                has_functions=bool(functions),
                model=model or self.settings.openai_model_name,
            )

            # Validate vector store is ready before creating tools
            if vector_store_id:
                store_info = (
                    await self.tool_manager.vector_store_manager.get_vector_store(
                        vector_store_id
                    )
                )
                if not store_info or store_info["status"] != "completed":
                    self.logger.warning(
                        "Vector store not ready, skipping file_search tool",
                        component="Chat",
                        subcomponent="ContinueChatWithTools",
                        vector_store_id=vector_store_id,
                        status=store_info["status"] if store_info else "Not found",
                    )
                    vector_store_id = None  # Don't use the vector store if not ready
                else:
                    self.logger.info(
                        "Vector store is ready for tool creation",
                        component="Chat",
                        subcomponent="ContinueChatWithTools",
                        vector_store_id=vector_store_id,
                    )

            # Get tools from tool manager
            try:
                # NEW: Log functions before tool preparation
                if functions:
                    self.logger.info(
                        "[DEBUG] Functions received for tool preparation",
                        component="Chat",
                        subcomponent="ContinueChatWithTools",
                        function_count=len(functions),
                        function_names=[f.get("name") for f in functions],
                    )
                    # Log detailed parameters for each function
                    for func in functions:
                        func_name = func.get("name", "unknown")
                        params = func.get("parameters", {})
                        props = params.get("properties", {})
                        self.logger.info(
                            f"[DEBUG] Function '{func_name}' parameters detail",
                            component="Chat",
                            subcomponent="ContinueChatWithTools",
                            function_name=func_name,
                            has_parameters=bool(params),
                            properties_count=len(props),
                            property_names=list(props.keys()) if props else [],
                            required_fields=params.get("required", []),
                        )

                tools = await self.tool_manager.get_tools_for_response(
                    vector_store_id=vector_store_id, functions=functions
                )

                # NEW: Log tools after preparation
                if tools:
                    self.logger.info(
                        "[DEBUG] Tools prepared successfully",
                        component="Chat",
                        subcomponent="ContinueChatWithTools",
                        tool_count=len(tools),
                        tool_types=[t.get("type") for t in tools],
                    )
                    # Log detailed parameters for each function tool
                    for tool in tools:
                        if tool.get("type") == "function":
                            func_def = tool.get("function", {})
                            func_name = func_def.get("name", "unknown")
                            params = func_def.get("parameters", {})
                            props = params.get("properties", {})
                            self.logger.info(
                                f"[DEBUG] Tool function '{func_name}' configuration",
                                component="Chat",
                                subcomponent="ContinueChatWithTools",
                                function_name=func_name,
                                has_parameters=bool(params),
                                properties_count=len(props),
                                property_names=list(props.keys()) if props else [],
                                required_fields=params.get("required", []),
                            )

                # Validate tools
                is_valid = await self.tool_manager.validate_tools(tools)
                if not is_valid:
                    self.logger.warning(
                        "Tool validation failed, proceeding without tools",
                        component="Chat",
                        subcomponent="ContinueChatWithTools",
                    )
                    tools = []
            except Exception as e:
                self.logger.warning(
                    "Error preparing tools, proceeding without tools",
                    component="Chat",
                    subcomponent="ContinueChatWithTools",
                    error=str(e),
                )
                tools = []

            # Emphasized logging: Initial LLM input context before API call
            tool_types_list = [t.get("type", "unknown") for t in (tools or [])]

            self.logger.info(
                f"--- LLM INPUT CONTEXT (INITIAL QUERY) ---\n"
                f"MESSAGE_LENGTH: {len(message)} chars\n"
                f"TOOLS_AVAILABLE: {tool_types_list}\n"
                f"-----------------------------------------",
                component="Chat",
                subcomponent="ContinueChatWithTools",
                message_length=len(message),
                tools_available=tool_types_list,
                tools_count=len(tools or []),
            )

            # Continue chat with tools
            response_id = await self.continue_chat(
                chat_id=chat_id, message=message, model=model, tools=tools
            )

            # Get response for processing
            response = await asyncio.to_thread(
                self.client.responses.retrieve, response_id=response_id
            )

            # Process response content
            content = self._extract_text_content(response)

            # Extract tool calls using normalized extraction method
            tool_calls = self._extract_tool_calls_from_response(response)

            # Extract and append citations
            citations = await self.citation_manager.extract_citations_from_response(
                response
            )
            if citations:
                content = self.citation_manager.append_citations_to_content(
                    content, citations
                )

            self.logger.info(
                "Chat with tools continued successfully",
                component="Chat",
                subcomponent="ContinueChatWithTools",
                chat_id=chat_id,
                response_id=response_id,
                has_tool_calls=bool(tool_calls),
            )

            return {
                "response_id": response_id,
                "content": content,
                "tool_calls": tool_calls,
                "citations": citations,
            }

        except (ContentParsingError, ToolConfigurationError, ResponsesAPIError) as e:
            self.logger.error(
                "Error in continue_chat_with_tools",
                component="Chat",
                subcomponent="ContinueChatWithTools",
                chat_id=chat_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to continue chat with tools",
                component="Chat",
                subcomponent="ContinueChatWithTools",
                chat_id=chat_id,
                error=str(e),
            )
            raise ResponsesAPIError(
                message=f"Failed to continue chat with tools: {str(e)}"
            )

    @time_execution("Chat", "ContinueChatWithToolOutputs")
    async def continue_chat_with_tool_outputs(
        self,
        previous_response_id: str,
        tool_outputs: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Continue a conversation by submitting function call outputs.

        This is the CORRECT way to handle function calling with Responses API.
        The API requires function outputs to be submitted as structured
        function_call_output items with matching call_id values.

        Args:
            previous_response_id: ID of the response that made the tool calls
            tool_outputs: List of dicts with 'call_id' and 'output' keys
            model: Optional model override
            tools: Optional tools to include in the next response

        Returns:
            New response ID

        Raises:
            ResponsesAPIError: If submission fails
        """
        try:
            model = model or self.settings.openai_model_name

            self.logger.info(
                "Continuing chat with tool outputs",
                component="Chat",
                subcomponent="ContinueChatWithToolOutputs",
                previous_response_id=previous_response_id,
                tool_output_count=len(tool_outputs),
            )

            # Format input as function_call_output items
            # NOTE: Response API requires structured function_call_output items,
            # not plain text messages. Each item must have:
            # - type: "function_call_output"
            # - call_id: matching the original tool call ID
            # - output: the result string from function execution
            input_items = []
            for tool_output in tool_outputs:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_output["call_id"],
                        "output": tool_output["output"],
                    }
                )

            # Emphasized logging: LLM input context before API call
            tools_available = [t.get("type", "unknown") for t in (tools or [])]

            # Calculate total output size
            total_output_size = sum(
                len(str(item.get("output", ""))) for item in input_items
            )

            self.logger.info(
                f"--- LLM INPUT CONTEXT (TOOL OUTPUTS) ---\n"
                f"PREVIOUS_RESPONSE_ID: {previous_response_id}\n"
                f"TOOL_OUTPUTS_COUNT: {len(input_items)}\n"
                f"TOOL_OUTPUTS_TOTAL_SIZE: {total_output_size} chars\n"
                f"TOOLS_AVAILABLE: {tools_available}\n"
                f"----------------------------------------",
                component="Chat",
                subcomponent="ContinueChatWithToolOutputs",
                previous_response_id=previous_response_id,
                tool_outputs_count=len(input_items),
                total_output_size=total_output_size,
                tools_available=tools_available,
                tools_count=len(tools or []),
            )

            # Create response with tool outputs
            # The input parameter accepts a list of structured items for function outputs
            response = await self.response_adapter.create_response(
                model=model,
                previous_response_id=previous_response_id,
                input=input_items,  # Send structured tool outputs, not text
                instructions=get_system_prompt(),
                tools=tools or [],
            )

            self.logger.info(
                "Continued chat with tool outputs successfully",
                component="Chat",
                subcomponent="ContinueChatWithToolOutputs",
                response_id=response.id,
            )

            return response.id

        except Exception as e:
            self.logger.error(
                "Failed to continue chat with tool outputs",
                component="Chat",
                subcomponent="ContinueChatWithToolOutputs",
                error=str(e),
                exc_info=True,
            )
            raise ResponsesAPIError(
                message=f"Failed to continue chat with tool outputs: {str(e)}"
            )

    @time_execution("Chat", "ContinueChatWithToolExecution")
    async def continue_chat_with_tool_execution(
        self,
        chat_id: str,
        message: Optional[str] = None,
        initial_response: Optional[Any] = None,
        vector_store_id: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Continue chat with automatic tool execution loop.

        When the model makes tool calls, this method automatically executes them
        and continues the conversation with the tool results until the model
        provides a final text response (or max iterations is reached).

        Args:
            chat_id: Existing chat ID (or response ID for initial_response case)
            message: New user message (required if initial_response is not provided)
            initial_response: Response object with tool calls already present
                (required if message is not provided). When provided, skips
                the initial API call and extracts tool calls directly.
            vector_store_id: Optional vector store ID for file_search
            functions: Optional function definitions for function calling
            model: Optional model override
            max_iterations: Maximum number of tool execution iterations (default: 5)

        Returns:
            Dict with final response content, tool_calls history, and citations

        Raises:
            ResponsesAPIError: If neither message nor initial_response is provided
        """
        try:
            # Parameter validation
            if not message and not initial_response:
                raise ResponsesAPIError(
                    "Either 'message' or 'initial_response' must be provided"
                )
            if message and initial_response:
                raise ResponsesAPIError(
                    "Cannot provide both 'message' and 'initial_response'. "
                    "Use 'message' for continuing conversations, "
                    "'initial_response' for new conversations with existing tool calls."
                )

            self.logger.info(
                "Starting chat with tool execution loop",
                component="Chat",
                subcomponent="ContinueChatWithToolExecution",
                chat_id=chat_id,
                max_iterations=max_iterations,
                has_initial_response=initial_response is not None,
                has_message=message is not None,
            )

            tool_execution_history = []
            current_response_id = None
            iteration = 0
            content = ""
            tool_calls = []
            citations = []

            # Handle initial_response case: extract tool calls directly from response
            if initial_response:
                self.logger.info(
                    "Using provided initial_response, extracting tool calls directly",
                    component="Chat",
                    subcomponent="ContinueChatWithToolExecution",
                    response_id=getattr(initial_response, "id", chat_id),
                )

                # Extract tool calls, content, and citations from the response object
                current_response_id = getattr(initial_response, "id", chat_id)
                content = self._extract_text_content(initial_response)
                tool_calls = self._extract_tool_calls_from_response(initial_response)

                # INFO LOG: Tool calls detected after extraction
                if tool_calls:
                    self.logger.info(
                        "TOOL CALLS DETECTED IN RESPONSE",
                        component="Chat",
                        subcomponent="ContinueChatWithToolExecution",
                        response_id=current_response_id,
                        tool_call_count=len(tool_calls),
                        tool_calls=[
                            {
                                "function": tc.get("function_name"),
                                "arguments": tc.get("arguments"),
                                "call_id": tc.get("call_id"),
                            }
                            for tc in tool_calls
                        ],
                    )

                    # Log tool trigger summary
                    file_search_count = sum(
                        1 for tc in tool_calls if tc.get("type") == "file_search"
                    )
                    function_call_count = sum(
                        1 for tc in tool_calls if tc.get("type") == "function_call"
                    )
                    self.logger.info(
                        f"TOOLS TRIGGERED: FILE_SEARCH={file_search_count}, FUNCTION_CALL={function_call_count}, TOTAL={len(tool_calls)}",
                        component="Chat",
                        subcomponent="ContinueChatWithToolExecution",
                        file_search_count=file_search_count,
                        function_call_count=function_call_count,
                        total_tools=len(tool_calls),
                    )

                citations = await self.citation_manager.extract_citations_from_response(
                    initial_response
                )

                if citations:
                    content = self.citation_manager.append_citations_to_content(
                        content, citations
                    )
            else:
                # Handle message case: get initial response with tools via API call
                self.logger.info(
                    "Getting initial response with tools via API call",
                    component="Chat",
                    subcomponent="ContinueChatWithToolExecution",
                )

                result = await self.continue_chat_with_tools(
                    chat_id=chat_id,
                    message=message,
                    vector_store_id=vector_store_id,
                    functions=functions,
                    model=model,
                )

                current_response_id = result["response_id"]
                content = result["content"]
                tool_calls = result.get("tool_calls", [])
                citations = result.get("citations", [])

            # Get tools for subsequent iterations
            if self.tool_manager:
                tools = await self.tool_manager.get_tools_for_response(
                    vector_store_id=vector_store_id, functions=functions
                )
            else:
                tools = []

            # Tool execution loop
            while tool_calls and iteration < max_iterations:
                iteration += 1
                self.logger.info(
                    f"Tool execution iteration {iteration}/{max_iterations}",
                    component="Chat",
                    subcomponent="ContinueChatWithToolExecution",
                    tool_call_count=len(tool_calls),
                )

                # Execute all tool calls and collect outputs
                # NOTE: Tool calls are now normalized dicts with 'call_id', 'function_name', 'arguments'
                tool_outputs = []
                for tool_call in tool_calls:
                    try:
                        call_id = tool_call["call_id"]
                        function_name = tool_call["function_name"]
                        arguments = tool_call["arguments"]

                        # CRITICAL INFO LOG: Function call execution with arguments
                        self.logger.info(
                            f"→ Executing Function: {function_name}",
                            component="Chat",
                            subcomponent="ContinueChatWithToolExecution",
                            function_name=function_name,
                            call_id=call_id,
                            arguments_extracted=arguments,
                            arguments_count=len(arguments),
                            is_empty=not bool(arguments),
                        )

                        # Execute function
                        result_str = await self._execute_function_call(
                            function_name, arguments
                        )

                        # Collect tool output in format required by continue_chat_with_tool_outputs
                        tool_outputs.append(
                            {
                                "call_id": call_id,
                                "output": result_str,
                            }
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
                        error_msg = f"Error processing tool call: {str(e)}"
                        self.logger.error(
                            error_msg,
                            component="Chat",
                            subcomponent="ContinueChatWithToolExecution",
                            exc_info=True,
                        )
                        # Still submit error output with call_id if available
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

                # CRITICAL FIX: Use the proper method to submit tool outputs
                # Response API requires structured function_call_output items with call_id,
                # not plain text messages
                current_response_id = await self.continue_chat_with_tool_outputs(
                    previous_response_id=current_response_id,
                    tool_outputs=tool_outputs,
                    model=model,
                    tools=tools,
                )

                # Retrieve the new response
                response = await asyncio.to_thread(
                    self.client.responses.retrieve, response_id=current_response_id
                )

                # Extract content and tool calls from new response
                content = self._extract_text_content(response)
                tool_calls = self._extract_tool_calls_from_response(response)
                new_citations = (
                    await self.citation_manager.extract_citations_from_response(
                        response
                    )
                )
                citations.extend(new_citations)

            # Check if we hit max iterations
            if iteration >= max_iterations and tool_calls:
                self.logger.warning(
                    f"Reached max iterations ({max_iterations}) with pending tool calls",
                    component="Chat",
                    subcomponent="ContinueChatWithToolExecution",
                )

            self.logger.info(
                f"Tool execution loop completed in {iteration} iterations",
                component="Chat",
                subcomponent="ContinueChatWithToolExecution",
                final_response_id=current_response_id,
            )

            return {
                "response_id": current_response_id,
                "content": content,
                "tool_execution_history": tool_execution_history,
                "citations": citations,
                "iterations": iteration,
            }

        except Exception as e:
            self.logger.error(
                "Failed in tool execution loop",
                component="Chat",
                subcomponent="ContinueChatWithToolExecution",
                error=str(e),
                exc_info=True,
            )
            raise ResponsesAPIError(message=f"Failed in tool execution loop: {str(e)}")

    def _format_tool_results_message(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Format tool execution results as a message for the next conversation turn.

        Args:
            tool_results: List of tool execution results

        Returns:
            Formatted message string
        """
        formatted_parts = []
        for result in tool_results:
            formatted_parts.append(
                f"Tool '{result['function_name']}' returned:\n{result['result']}"
            )

        return "Tool execution results:\n\n" + "\n\n".join(formatted_parts)

    @time_execution("Chat", "SummarizeHistory")
    async def _summarize_history(
        self, history: List[Dict[str, Any]], model: str
    ) -> str:
        """
        Summarize conversation history to reset chain.

        Args:
            history: List of response dicts.
            model: Model for summarization.

        Returns:
            Concise summary string.
        """
        try:
            self.logger.info(
                "Summarizing conversation history",
                component="Chat",
                subcomponent="SummarizeHistory",
                history_length=len(history),
                model=model,
            )

            # Build prompt from history (limit to key content)
            history_text = "\n".join(
                [
                    f"Response {i+1}: {h['content'][:500]}..."
                    for i, h in enumerate(history)
                ]
            )  # Truncate for prompt
            summary_prompt = f"Summarize this conversation history concisely (under 1000 tokens):\n{history_text}"

            summary_response = await self.response_adapter.create_response(
                model=model,
                input=summary_prompt,
                instructions="Summarize the conversation history concisely.",
            )

            # Extract text content using the helper method
            summary = self._extract_text_content(summary_response)

            self.logger.info(
                "Generated history summary for chain reset",
                component="Chat",
                subcomponent="SummarizeHistory",
                summary_length=len(summary),
            )
            return summary

        except ContentParsingError as e:
            self.logger.error(
                "Content parsing error in summarize_history",
                component="Chat",
                subcomponent="SummarizeHistory",
                error=str(e),
            )
            return "Previous conversation history (summarized)."
        except Exception as e:
            self.logger.error(
                "Failed to summarize history",
                component="Chat",
                subcomponent="SummarizeHistory",
                error=str(e),
            )
            # Fallback: Basic summary
            return "Previous conversation history (summarized)."

    @time_execution("Chat", "GetChatHistory")
    async def get_chat_history(
        self, chat_id: str, full_chain: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history by traversing the response chain.

        Args:
            chat_id: Chat ID (root response ID).
            full_chain: If True, fetch entire chain; else, just last response.

        Returns:
            List of response objects in reverse chronological order (newest first).
        """
        try:
            self.logger.info(
                "Retrieving chat history",
                component="Chat",
                subcomponent="GetChatHistory",
                chat_id=chat_id,
                full_chain=full_chain,
            )

            history = []
            current_id = self._chat_cache.get(chat_id, chat_id)  # Start from last known

            while current_id:
                response = await asyncio.to_thread(
                    self.client.responses.retrieve, response_id=current_id
                )

                try:
                    # Extract content using helper method
                    content = self._extract_text_content(response)

                    # Build comprehensive response data
                    response_data = {
                        "id": response.id,
                        "content": content,
                        "previous_response_id": getattr(
                            response, "previous_response_id", None
                        ),
                        "created_at": getattr(response, "created_at", None),
                    }

                    # Add tool calls if present
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        response_data["tool_calls"] = response.tool_calls

                    history.append(response_data)
                except ContentParsingError as e:
                    self.logger.warning(
                        "Error parsing response in history",
                        component="Chat",
                        subcomponent="GetChatHistory",
                        response_id=current_id,
                        error=str(e),
                    )
                    # Add minimal response data on parsing error
                    history.append(
                        {
                            "id": response.id,
                            "content": str(response),
                            "previous_response_id": getattr(
                                response, "previous_response_id", None
                            ),
                            "created_at": getattr(response, "created_at", None),
                            "parsing_error": str(e),
                        }
                    )

                current_id = (
                    response.previous_response_id
                    if hasattr(response, "previous_response_id")
                    else None
                )

                if not full_chain or not current_id:
                    break

            history.reverse()  # Chronological order
            self.logger.info(
                "Retrieved chat history",
                component="Chat",
                subcomponent="GetChatHistory",
                chat_id=chat_id,
                response_count=len(history),
            )
            return history

        except Exception as e:
            self.logger.error(
                "Failed to get chat history",
                component="Chat",
                subcomponent="GetChatHistory",
                chat_id=chat_id,
                error=str(e),
            )
            raise ResponsesAPIError(message=f"Failed to get chat history: {str(e)}")

    @time_execution("Chat", "DeleteChat")
    async def delete_chat(self, chat_id: str) -> bool:
        """
        Delete the last response in the chain (partial cleanup; full chain deletion requires traversing).

        Args:
            chat_id: Chat ID.

        Returns:
            True if successful.
        """
        try:
            last_response_id = self._chat_cache.get(chat_id, chat_id)

            self.logger.info(
                "Deleting chat",
                component="Chat",
                subcomponent="DeleteChat",
                chat_id=chat_id,
                last_response_id=last_response_id,
            )

            await asyncio.to_thread(
                self.client.responses.delete, response_id=last_response_id
            )

            # Remove from cache
            if chat_id in self._chat_cache:
                del self._chat_cache[chat_id]

            self.logger.info(
                "Deleted chat successfully",
                component="Chat",
                subcomponent="DeleteChat",
                chat_id=chat_id,
                response_id=last_response_id,
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to delete chat",
                component="Chat",
                subcomponent="DeleteChat",
                chat_id=chat_id,
                error=str(e),
            )
            return False

    def list_chats(self) -> List[str]:
        """
        List active chat IDs from cache.

        Returns:
            List of chat IDs.
        """
        try:
            chats = list(self._chat_cache.keys())
            self.logger.info(
                "Listed chats",
                component="Chat",
                subcomponent="ListChats",
                chat_count=len(chats),
            )
            return chats
        except Exception as e:
            self.logger.error(
                "Failed to list chats",
                component="Chat",
                subcomponent="ListChats",
                error=str(e),
            )
            return []

    def clear_cache(self) -> None:
        """Clear chat cache."""
        self._chat_cache.clear()
        self.logger.info(
            "Chat cache cleared", component="Chat", subcomponent="ClearCache"
        )
