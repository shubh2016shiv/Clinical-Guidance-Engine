"""Configure OpenAI's hosted file search tool."""

from ..schema import FileSearchTool


def file_search_tool(vector_store_id: str, *, max_results: int | None = None) -> FileSearchTool:
    """Build a file_search tool for one prepared guidelines store."""
    if not vector_store_id:
        raise ValueError("A vector store ID is required for file search")
    tool: FileSearchTool = {
        "type": "file_search",
        "vector_store_ids": [vector_store_id],
    }
    if max_results is not None:
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        tool["max_num_results"] = max_results
    return tool
