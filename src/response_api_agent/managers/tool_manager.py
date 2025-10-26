from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI
from src.config import get_settings
from src.response_api_agent.managers.vector_store_manager import VectorStoreManager
from src.response_api_agent.managers.exceptions import ToolConfigurationError, VectorStoreError
from src.logs import get_component_logger, time_execution

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
        self.logger = get_component_logger("Tool")

    @time_execution("Tool", "ConfigureFileSearchTool")
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
            self.logger.info(
                "Configuring file_search tool",
                component="Tool",
                subcomponent="ConfigureFileSearchTool",
                vector_store_id=vector_store_id,
                max_results=max_results
            )
            
            # Verify vector store exists and is ready
            vector_store_info = await self.vector_store_manager.get_vector_store(vector_store_id)
            
            if not vector_store_info or vector_store_info["status"] != "completed":
                self.logger.warning(
                    "Vector store not ready or not found",
                    component="Tool",
                    subcomponent="ConfigureFileSearchTool",
                    vector_store_id=vector_store_id,
                    status=vector_store_info["status"] if vector_store_info else "Not found"
                )
                raise VectorStoreError(f"Vector store {vector_store_id} is not ready for file_search")

            # Configure file_search tool (Responses API schema)
            # Note: vector_store_ids must be at the root level of the tool object
            tool_config = {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": max_results
            }

            self.logger.info(
                "Created file_search tool config",
                component="Tool",
                subcomponent="ConfigureFileSearchTool",
                vector_store_id=vector_store_id,
                max_results=max_results
            )

            # Cache tool configuration by vector_store_id
            if vector_store_id not in self._tool_cache:
                self._tool_cache[vector_store_id] = []
            self._tool_cache[vector_store_id].append(tool_config)
            
            self.logger.info(
                "Configured file_search tool successfully",
                component="Tool",
                subcomponent="ConfigureFileSearchTool",
                vector_store_id=vector_store_id
            )

            return tool_config

        except VectorStoreError as e:
            self.logger.error(
                "Vector store error",
                component="Tool",
                subcomponent="ConfigureFileSearchTool",
                vector_store_id=vector_store_id,
                error=str(e)
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to configure file_search tool",
                component="Tool",
                subcomponent="ConfigureFileSearchTool",
                vector_store_id=vector_store_id,
                error=str(e)
            )
            raise ToolConfigurationError(f"Failed to configure file_search tool: {str(e)}")

    @time_execution("Tool", "ConfigureFunctionTool")
    def configure_function_tool(self, function: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure a function tool for Responses API.
        
        Args:
            function: Single function definition with 'name', 'description', 'parameters', and optional 'strict'.
            
        Returns:
            Tool configuration dictionary for responses.create.
        """
        try:
            function_name = function.get('name', 'unknown')
            
            self.logger.info(
                "Configuring function tool",
                component="Tool",
                subcomponent="ConfigureFunctionTool",
                function_name=function_name
            )
            
            # Validate function (basic check for required fields)
            if not all(key in function for key in ["name", "description", "parameters"]):
                self.logger.error(
                    "Function missing required fields",
                    component="Tool",
                    subcomponent="ConfigureFunctionTool",
                    function_name=function_name,
                    missing_fields=[key for key in ["name", "description", "parameters"] if key not in function]
                )
                raise ToolConfigurationError(f"Function {function_name} missing required fields")

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
            
            self.logger.info(
                "Configured function tool successfully",
                component="Tool",
                subcomponent="ConfigureFunctionTool",
                function_name=function["name"]
            )

            return tool_config

        except ToolConfigurationError as e:
            self.logger.error(
                "Tool configuration error",
                component="Tool",
                subcomponent="ConfigureFunctionTool",
                function_name=function.get('name', 'unknown'),
                error=str(e)
            )
            raise
        except Exception as e:
            self.logger.error(
                "Failed to configure function tool",
                component="Tool",
                subcomponent="ConfigureFunctionTool",
                function_name=function.get('name', 'unknown'),
                error=str(e)
            )
            raise ToolConfigurationError(f"Failed to configure function tool: {str(e)}")

    @time_execution("Tool", "GetToolsForResponse")
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
            self.logger.info(
                "Getting tools for response",
                component="Tool",
                subcomponent="GetToolsForResponse",
                has_vector_store=bool(vector_store_id),
                has_functions=bool(functions),
                function_count=len(functions) if functions else 0
            )

            # Add file_search tool if vector store ID is provided
            if vector_store_id:
                self.logger.info(
                    "Adding file_search tool",
                    component="Tool",
                    subcomponent="GetToolsForResponse",
                    vector_store_id=vector_store_id
                )
                
                if vector_store_id in self._tool_cache:
                    tools.extend(self._tool_cache[vector_store_id])
                    self.logger.info(
                        "Using cached file_search tool",
                        component="Tool",
                        subcomponent="GetToolsForResponse",
                        vector_store_id=vector_store_id
                    )
                else:
                    # Configure on-the-fly if not cached
                    file_search_tool = await self.configure_file_search_tool(vector_store_id)
                    tools.append(file_search_tool)
                    self.logger.info(
                        "Created new file_search tool",
                        component="Tool",
                        subcomponent="GetToolsForResponse",
                        vector_store_id=vector_store_id
                    )

            # Add function tools if functions are provided
            if functions and len(functions) > 0:
                self.logger.info(
                    "Adding function tools",
                    component="Tool",
                    subcomponent="GetToolsForResponse",
                    function_count=len(functions)
                )
                
                for function in functions:
                    function_tool = self.configure_function_tool(function)
                    tools.append(function_tool)
                    self.logger.info(
                        "Added function tool",
                        component="Tool",
                        subcomponent="GetToolsForResponse",
                        function_name=function['name']
                    )

            self.logger.info(
                "Prepared tools for response",
                component="Tool",
                subcomponent="GetToolsForResponse",
                tool_count=len(tools)
            )
            return tools

        except (VectorStoreError, ToolConfigurationError) as e:
            self.logger.error(
                "Error preparing tools for response",
                component="Tool",
                subcomponent="GetToolsForResponse",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        except Exception as e:
            self.logger.error(
                "Error preparing tools for response",
                component="Tool",
                subcomponent="GetToolsForResponse",
                error=str(e)
            )
            raise ToolConfigurationError(f"Failed to prepare tools for response: {str(e)}")

    @time_execution("Tool", "ValidateTools")
    async def validate_tools(self, tools: List[Dict[str, Any]]) -> bool:
        """
        Validate tool configurations for Responses API.
        
        Args:
            tools: List of tool configurations.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            self.logger.info(
                "Validating tools",
                component="Tool",
                subcomponent="ValidateTools",
                tool_count=len(tools)
            )
            
            for tool in tools:
                if tool["type"] == "file_search":
                    # vector_store_ids is at the root level of the tool object
                    vector_store_ids = tool.get("vector_store_ids", [])
                    self.logger.info(
                        "Validating file_search tool",
                        component="Tool",
                        subcomponent="ValidateTools",
                        vector_store_ids=vector_store_ids
                    )
                    
                    if not vector_store_ids:
                        self.logger.error(
                            "file_search tool missing vector_store_ids",
                            component="Tool",
                            subcomponent="ValidateTools"
                        )
                        return False
                        
                    for vs_id in vector_store_ids:
                        vs_info = await self.vector_store_manager.get_vector_store(vs_id)
                        if not vs_info or vs_info["status"] != "completed":
                            self.logger.error(
                                "Vector store not ready for file_search",
                                component="Tool",
                                subcomponent="ValidateTools",
                                vector_store_id=vs_id,
                                status=vs_info["status"] if vs_info else "Not found"
                            )
                            return False
                            
                elif tool["type"] == "function":
                    function = tool.get("function", {})
                    function_name = function.get('name', 'unknown')
                    
                    self.logger.info(
                        "Validating function tool",
                        component="Tool",
                        subcomponent="ValidateTools",
                        function_name=function_name
                    )
                    
                    if not function:
                        self.logger.error(
                            "function tool missing function definition",
                            component="Tool",
                            subcomponent="ValidateTools"
                        )
                        return False
                        
                    if not all(key in function for key in ["name", "description", "parameters"]):
                        self.logger.error(
                            "Function missing required fields",
                            component="Tool",
                            subcomponent="ValidateTools",
                            function_name=function_name,
                            missing_fields=[key for key in ["name", "description", "parameters"] if key not in function]
                        )
                        return False
                        
            self.logger.info(
                "Tools validation successful",
                component="Tool",
                subcomponent="ValidateTools",
                tool_count=len(tools)
            )
            return True

        except Exception as e:
            self.logger.error(
                "Error validating tools",
                component="Tool",
                subcomponent="ValidateTools",
                error=str(e)
            )
            return False

    def clear_tool_cache(self) -> None:
        """Clear the tool configuration cache."""
        self._tool_cache.clear()
        self.logger.info(
            "Tool cache cleared",
            component="Tool",
            subcomponent="ClearCache"
        )