import asyncio
from typing import Dict, Any, List, Optional
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

            self.logger.debug(
                "Extracting text content from response",
                component="Chat",
                subcomponent="ExtractTextContent",
                response_id=response_id,
                response_type=type(response).__name__,
            )

            # Check for the new response structure with output field
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    # Look for ResponseOutputMessage type items
                    if hasattr(item, "type") and item.type == "message":
                        if hasattr(item, "content") and item.content:
                            text_parts = []
                            for content_block in item.content:
                                # Extract text from ResponseOutputText
                                if (
                                    hasattr(content_block, "type")
                                    and content_block.type == "output_text"
                                ):
                                    if hasattr(content_block, "text"):
                                        text_parts.append(content_block.text)

                            if text_parts:
                                self.logger.debug(
                                    "Extracted text from output structure",
                                    component="Chat",
                                    subcomponent="ExtractTextContent",
                                    response_id=response_id,
                                    text_parts_count=len(text_parts),
                                )
                                return "\n".join(text_parts)

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
                    self.logger.debug(
                        "Extracted text from legacy content structure",
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
                tools = await self.tool_manager.get_tools_for_response(
                    vector_store_id=vector_store_id, functions=functions
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

            # Process tool calls if present
            tool_calls = []
            # Check for new response structure with output field
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    # Look for tool call items
                    if hasattr(item, "type") and "call" in item.type:
                        tool_calls.append(item)
            # Fallback to legacy format
            elif hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

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
