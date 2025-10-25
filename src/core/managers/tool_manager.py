import asyncio
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI
from src.core.config import get_settings
from src.core.managers.vector_store_manager import VectorStoreManager
from src.core.managers.exceptions import ToolConfigurationError, ResponsesAPIError, VectorStoreError

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manages tools for the OpenAI Responses API, including file_search and function calling.
    
    Configures and validates tools for use in responses.create calls.
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initialize the Tool Manager.
        
        Args:
            vector_store_manager: Instance of VectorStoreManager to access vector stores.
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.vector_store_manager = vector_store_manager
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}  # Cache tools by key (e.g., vector_store_id or function name)

    async def configure_file_search_tool(self, vector_store_id: str, max_results: int = 20) -> Dict[str, Any]:
        """
        Configure a file_search tool with a vector store for Responses API.

        Args:
            vector_store_id: Vector store ID to attach to file_search.
            max_results: Maximum number of search results to return (default: 20).

        Returns:
            Tool configuration dictionary for responses.create.
        """
        try:
            # Verify vector store exists and is ready
            vector_store_info = await self.vector_store_manager.get_vector_store(vector_store_id)
            logger.info(f"Vector store info for {vector_store_id}: {vector_store_info}")
            if not vector_store_info or vector_store_info["status"] != "completed":
                logger.warning(f"Vector store {vector_store_id} not ready or not found")
                raise VectorStoreError(f"Vector store {vector_store_id} is not ready for file_search")

            # Configure file_search tool (Responses API schema)
            # Note: vector_store_ids must be at the root level of the tool object
            tool_config = {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": max_results
            }

            logger.info(f"Created file_search tool config: {tool_config}")

            # Cache tool configuration by vector_store_id
            if vector_store_id not in self._tool_cache:
                self._tool_cache[vector_store_id] = []
            self._tool_cache[vector_store_id].append(tool_config)
            logger.info(f"Configured file_search tool for vector store {vector_store_id}")

            return tool_config

        except VectorStoreError as e:
            logger.error(f"Vector store error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to configure file_search tool: {str(e)}")
            raise ToolConfigurationError(f"Failed to configure file_search tool: {str(e)}")

    def configure_function_tool(self, function: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure a function tool for Responses API.
        
        Args:
            function: Single function definition with 'name', 'description', 'parameters', and optional 'strict'.
            
        Returns:
            Tool configuration dictionary for responses.create.
        """
        try:
            # Validate function (basic check for required fields)
            if not all(key in function for key in ["name", "description", "parameters"]):
                raise ToolConfigurationError(f"Function {function.get('name', 'unknown')} missing required fields")

            # Configure function tool (Responses API schema)
            tool_config = {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function["description"],
                    "parameters": function["parameters"],
                    "strict": function.get("strict", True)  # Use provided strict value or default to True
                }
            }
            
            # Cache tool configuration
            cache_key = f"function_{function['name']}"
            if cache_key not in self._tool_cache:
                self._tool_cache[cache_key] = []
            self._tool_cache[cache_key].append(tool_config)
            logger.info(f"Configured function tool for {function['name']}")

            return tool_config

        except ToolConfigurationError as e:
            logger.error(f"Tool configuration error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to configure function tool: {str(e)}")
            raise ToolConfigurationError(f"Failed to configure function tool: {str(e)}")

    async def get_tools_for_response(
        self,
        vector_store_id: Optional[str] = None,
        functions: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tools for a responses.create call, combining file_search and/or function calling.

        Args:
            vector_store_id: Optional vector store ID to include file_search tool.
            functions: Optional list of function definitions to include function tool.

        Returns:
            List of tool configurations.
        """
        try:
            tools = []
            logger.info(f"get_tools_for_response called with vector_store_id={vector_store_id}, functions={bool(functions)}")

            # Add file_search tool if vector store ID is provided
            if vector_store_id:
                logger.info(f"Attempting to create file_search tool for vector store {vector_store_id}")
                if vector_store_id in self._tool_cache:
                    tools.extend(self._tool_cache[vector_store_id])
                    logger.info(f"Using cached tools for vector store {vector_store_id}")
                else:
                    # Configure on-the-fly if not cached
                    file_search_tool = await self.configure_file_search_tool(vector_store_id)
                    tools.append(file_search_tool)
                    logger.info(f"Created new file_search tool: {file_search_tool}")

            # Add function tools if functions are provided
            if functions and len(functions) > 0:
                for function in functions:
                    function_tool = self.configure_function_tool(function)
                    tools.append(function_tool)
                    logger.info(f"Created function tool for {function['name']}")

            logger.info(f"Prepared {len(tools)} tools for response: {tools}")
            return tools

        except (VectorStoreError, ToolConfigurationError) as e:
            logger.error(f"Error preparing tools for response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error preparing tools for response: {str(e)}")
            raise ToolConfigurationError(f"Failed to prepare tools for response: {str(e)}")

    async def validate_tools(self, tools: List[Dict[str, Any]]) -> bool:
        """
        Validate tool configurations for Responses API.
        
        Args:
            tools: List of tool configurations.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            for tool in tools:
                if tool["type"] == "file_search":
                    # vector_store_ids is at the root level of the tool object
                    vector_store_ids = tool.get("vector_store_ids", [])
                    logger.info(f"Validating file_search tool with vector_store_ids: {vector_store_ids}")
                    if not vector_store_ids:
                        logger.error("file_search tool missing vector_store_ids")
                        return False
                    for vs_id in vector_store_ids:
                        vs_info = await self.vector_store_manager.get_vector_store(vs_id)
                        if not vs_info or vs_info["status"] != "completed":
                            logger.error(f"Vector store {vs_id} not ready for file_search")
                            return False
                elif tool["type"] == "function":
                    function = tool.get("function", {})
                    if not function:
                        logger.error("function tool missing function definition")
                        return False
                    if not all(key in function for key in ["name", "description", "parameters"]):
                        logger.error(f"Function {function.get('name', 'unknown')} missing required fields")
                        return False
            return True

        except Exception as e:
            logger.error(f"Error validating tools: {str(e)}")
            return False

    def clear_tool_cache(self) -> None:
        """Clear the tool configuration cache."""
        self._tool_cache.clear()
        logger.info("Tool cache cleared")