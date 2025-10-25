import asyncio
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI
from src.core.config import get_settings
from src.core.managers.tool_manager import ToolManager  # Optional integration for tools
from src.core.managers.exceptions import ResponsesAPIError, ContentParsingError, ToolConfigurationError

logger = logging.getLogger(__name__)

class ChatManager:
    """
    Manages conversations for the OpenAI Responses API using response chaining.
    
    Handles stateful interactions via `previous_response_id` for server-side history persistence.
    No client-side message storage—relies on API chaining and retrieval.
    Includes client-side limit on chain length to prevent excessively long conversations.
    """

    def __init__(self, tool_manager: Optional[ToolManager] = None, chat_history_limit: int = 10):
        """Initialize the Chat Manager.
        
        Args:
            tool_manager: Optional ToolManager for including tools in responses.
            chat_history_limit: Maximum number of previous responses in the chain (default: 10).
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.tool_manager = tool_manager
        self.chat_history_limit = chat_history_limit
        self._chat_cache: Dict[str, str] = {}  # Cache: chat_id -> last_response_id

    def _extract_text_content(self, response):
        """
        Extract text content from a Responses API response object.
        
        Args:
            response: Response object from the Responses API.
            
        Returns:
            Extracted text content as string.
        """
        try:
            # Check for the new response structure with output field
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    # Look for ResponseOutputMessage type items
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            text_parts = []
                            for content_block in item.content:
                                # Extract text from ResponseOutputText
                                if hasattr(content_block, 'type') and content_block.type == 'output_text':
                                    if hasattr(content_block, 'text'):
                                        text_parts.append(content_block.text)
                            
                            if text_parts:
                                return "\n".join(text_parts)
            
            # Legacy format check
            elif hasattr(response, 'content') and response.content:
                # Process content array
                text_parts = []
                for content_block in response.content:
                    if hasattr(content_block, 'text') and hasattr(content_block.text, 'value'):
                        text_parts.append(content_block.text.value)
                
                if text_parts:
                    return "\n".join(text_parts)
            
            # Fallback for unexpected response structure
            logger.warning(f"Unable to extract text from response structure: {type(response)}")
            return str(response)
        except Exception as e:
            logger.warning(f"Error extracting text content: {str(e)}")
            raise ContentParsingError(f"Failed to parse response content: {str(e)}")

    async def create_chat(
        self, 
        message: str, 
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Start a new conversation by creating the first response.
        
        Args:
            message: Initial user message.
            model: Model to use (default: from settings).
            tools: Optional tools (e.g., from ToolManager) to include.
            
        Returns:
            Chat ID (the initial response ID, used as chain root).
        """
        try:
            model = model or self.settings.openai_model_name
            logger.info(f"Creating new chat with initial message using model {model}")

            # Create first response (no previous_response_id)
            response = await asyncio.to_thread(
                self.client.responses.create,
                model=model,
                input=message,  # User input; API handles as first message
                tools=tools or []
            )

            chat_id = response.id  # Use response ID as chat identifier
            self._chat_cache[chat_id] = response.id  # Cache last response ID

            logger.info(f"Created chat {chat_id} with initial response {response.id}")
            return chat_id

        except Exception as e:
            logger.error(f"Failed to create chat: {str(e)}")
            raise ResponsesAPIError(message=f"Failed to create chat: {str(e)}")

    async def continue_chat(
        self, 
        chat_id: str, 
        message: str, 
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
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

            # Check current chain length
            history = await self.get_chat_history(chat_id)
            current_length = len(history)

            if current_length >= self.chat_history_limit:
                logger.info(f"Chat {chat_id} reached history limit ({self.chat_history_limit}); summarizing and resetting chain")
                # Summarize old history
                summary = await self._summarize_history(history, model)
                # Create new root response from summary + new message
                reset_message = f"{summary}\n\nNew query: {message}"
                new_response = await asyncio.to_thread(
                    self.client.responses.create,
                    model=model,
                    input=reset_message,
                    tools=tools or []
                )
                # Update chat_id to new root and cache
                chat_id = new_response.id
                self._chat_cache[chat_id] = new_response.id
                logger.info(f"Reset chat to new root {chat_id} after summarization")
                return chat_id
            else:
                # Chain new response normally
                response = await asyncio.to_thread(
                    self.client.responses.create,
                    model=model,
                    previous_response_id=last_response_id,
                    input=message,
                    tools=tools or []
                )

                # Update cache: chat_id -> new response ID
                self._chat_cache[chat_id] = response.id

                logger.info(f"Continued chat {chat_id} with new response {response.id} (length: {current_length + 1})")
                return response.id  # Return new ID for immediate use, but chat_id remains root

        except ContentParsingError as e:
            logger.error(f"Content parsing error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to continue chat {chat_id}: {str(e)}")
            raise ResponsesAPIError(message=f"Failed to continue chat: {str(e)}")

    async def continue_chat_with_tools(
        self, 
        chat_id: str, 
        message: str, 
        vector_store_id: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None
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
                raise ToolConfigurationError("Tool manager is required for this operation")

            # Validate vector store is ready before creating tools
            if vector_store_id:
                store_info = await self.tool_manager.vector_store_manager.get_vector_store(vector_store_id)
                if not store_info or store_info["status"] != "completed":
                    logger.warning(f"Vector store {vector_store_id} not ready (status: {store_info['status'] if store_info else 'Not found'}). Skipping file_search tool.")
                    vector_store_id = None  # Don't use the vector store if not ready
                else:
                    logger.info(f"Vector store {vector_store_id} is ready for tool creation")

            # Get tools from tool manager
            try:
                tools = await self.tool_manager.get_tools_for_response(
                    vector_store_id=vector_store_id,
                    functions=functions
                )
                # Validate tools
                is_valid = await self.tool_manager.validate_tools(tools)
                if not is_valid:
                    logger.warning("Tool validation failed, proceeding without tools")
                    tools = []
            except Exception as e:
                logger.warning(f"Error preparing tools: {str(e)}, proceeding without tools")
                tools = []
            
            # Continue chat with tools
            response_id = await self.continue_chat(
                chat_id=chat_id,
                message=message,
                model=model,
                tools=tools
            )
            
            # Get response for processing
            response = await asyncio.to_thread(
                self.client.responses.retrieve,
                response_id=response_id
            )
            
            # Process response content
            content = self._extract_text_content(response)
            
            # Process tool calls if present
            tool_calls = []
            # Check for new response structure with output field
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    # Look for tool call items
                    if hasattr(item, 'type') and 'call' in item.type:
                        tool_calls.append(item)
            # Fallback to legacy format
            elif hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = response.tool_calls
            
            return {
                "response_id": response_id,
                "content": content,
                "tool_calls": tool_calls
            }
        
        except (ContentParsingError, ToolConfigurationError, ResponsesAPIError) as e:
            logger.error(f"Error in continue_chat_with_tools: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to continue chat with tools: {str(e)}")
            raise ResponsesAPIError(message=f"Failed to continue chat with tools: {str(e)}")

    async def _summarize_history(self, history: List[Dict[str, Any]], model: str) -> str:
        """
        Summarize conversation history to reset chain.
        
        Args:
            history: List of response dicts.
            model: Model for summarization.
            
        Returns:
            Concise summary string.
        """
        try:
            # Build prompt from history (limit to key content)
            history_text = "\n".join([f"Response {i+1}: {h['content'][:500]}..." for i, h in enumerate(history)])  # Truncate for prompt
            summary_prompt = f"Summarize this conversation history concisely (under 1000 tokens):\n{history_text}"
            
            summary_response = await asyncio.to_thread(
                self.client.responses.create,
                model=model,
                input=summary_prompt
            )
            
            # Extract text content using the helper method
            summary = self._extract_text_content(summary_response)
            
            logger.info("Generated history summary for chain reset")
            return summary

        except ContentParsingError as e:
            logger.error(f"Content parsing error in summarize_history: {str(e)}")
            return "Previous conversation history (summarized)."
        except Exception as e:
            logger.error(f"Failed to summarize history: {str(e)}")
            # Fallback: Basic summary
            return "Previous conversation history (summarized)."

    async def get_chat_history(self, chat_id: str, full_chain: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history by traversing the response chain.
        
        Args:
            chat_id: Chat ID (root response ID).
            full_chain: If True, fetch entire chain; else, just last response.
            
        Returns:
            List of response objects in reverse chronological order (newest first).
        """
        try:
            history = []
            current_id = self._chat_cache.get(chat_id, chat_id)  # Start from last known
            
            while current_id:
                response = await asyncio.to_thread(
                    self.client.responses.retrieve,
                    response_id=current_id
                )
                
                try:
                    # Extract content using helper method
                    content = self._extract_text_content(response)
                    
                    # Build comprehensive response data
                    response_data = {
                        "id": response.id,
                        "content": content,
                        "previous_response_id": getattr(response, 'previous_response_id', None),
                        "created_at": getattr(response, 'created_at', None)
                    }
                    
                    # Add tool calls if present
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        response_data["tool_calls"] = response.tool_calls
                    
                    history.append(response_data)
                except ContentParsingError as e:
                    logger.warning(f"Error parsing response {current_id}: {str(e)}")
                    # Add minimal response data on parsing error
                    history.append({
                        "id": response.id,
                        "content": str(response),
                        "previous_response_id": getattr(response, 'previous_response_id', None),
                        "created_at": getattr(response, 'created_at', None),
                        "parsing_error": str(e)
                    })
                
                current_id = response.previous_response_id if hasattr(response, 'previous_response_id') else None
                
                if not full_chain or not current_id:
                    break
            
            history.reverse()  # Chronological order
            logger.info(f"Retrieved history for chat {chat_id} ({len(history)} responses)")
            return history
        
        except Exception as e:
            logger.error(f"Failed to get chat history {chat_id}: {str(e)}")
            raise ResponsesAPIError(message=f"Failed to get chat history: {str(e)}")

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

            await asyncio.to_thread(
                self.client.responses.delete,
                response_id=last_response_id
            )

            # Remove from cache
            if chat_id in self._chat_cache:
                del self._chat_cache[chat_id]

            logger.info(f"Deleted last response {last_response_id} for chat {chat_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete chat {chat_id}: {str(e)}")
            return False

    def list_chats(self) -> List[str]:
        """
        List active chat IDs from cache.
        
        Returns:
            List of chat IDs.
        """
        try:
            chats = list(self._chat_cache.keys())
            logger.info(f"Listed {len(chats)} chats")
            return chats
        except Exception as e:
            logger.error(f"Failed to list chats: {str(e)}")
            return []

    def clear_cache(self) -> None:
        """Clear chat cache."""
        self._chat_cache.clear()
        logger.info("Chat cache cleared")