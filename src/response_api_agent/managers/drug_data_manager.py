"""
Drug Data Manager

This module provides an interface between the OpenAI Responses API tool calls
and the Milvus vector database provider for drug data retrieval.

Key Features:
- Integrates MilvusProvider for drug database queries
- Provides function schema for Responses API tool calling
- Handles drug search by name, class, or both
- Formats results as JSON strings for LLM consumption
"""

import json
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from tabulate import tabulate

from src.providers.vector_db_provider import (
    create_vector_db_provider,
    VectorDBProvider,
    SearchResult,
)
from src.providers.vector_db_provider.exceptions import (
    VectorDBSearchError,
)
from src.logs import get_component_logger, time_execution


class DrugDataManager:
    """
    Manages drug data retrieval from Milvus vector database for Responses API.

    This manager acts as the bridge between OpenAI Responses API function calls
    and the Milvus vector database, providing intelligent drug information retrieval.
    """

    def __init__(self, vector_db_provider: Optional[VectorDBProvider] = None):
        """
        Initialize the Drug Data Manager.

        Args:
            vector_db_provider: Optional pre-configured VectorDBProvider instance.
                If None, creates a default Milvus provider.
        """
        self.logger = get_component_logger("DrugData")

        # Initialize or use provided vector database provider
        if vector_db_provider is None:
            try:
                self.vector_db_provider = create_vector_db_provider(
                    provider_type="milvus"
                )
                self.logger.info("Created default Milvus provider for drug data")
            except Exception as e:
                self.logger.error(f"Failed to create Milvus provider: {e}")
                raise
        else:
            self.vector_db_provider = vector_db_provider
            self.logger.info("Using provided vector database provider")

    @time_execution("DrugData", "SearchDrugDatabase")
    async def search_drug_database(
        self,
        drug_name: Optional[str] = None,
        drug_class: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Search the drug database for information about drugs.

        This method is called by the OpenAI Responses API when the model decides
        it needs drug information to answer the user's query.

        Args:
            drug_name: Specific drug name to search for (e.g., "Lisinopril", "Metformin")
            drug_class: Drug class to search for (e.g., "ACE inhibitor", "beta-blocker")
            limit: Maximum number of results to return (default: 5)

        Returns:
            Formatted string containing search results:
            - If successful: Tabulated format with drug information for LLM consumption
            - If error: JSON error object with error message and empty results

        Examples:
            >>> # Search by drug name
            >>> await search_drug_database(drug_name="Lisinopril")

            >>> # Search by drug class
            >>> await search_drug_database(drug_class="ACE inhibitor")

            >>> # Search by both (uses formatted query)
            >>> await search_drug_database(drug_name="hypertension", drug_class="ACE inhibitor")
        """
        try:
            self.logger.info(
                "Drug database search requested",
                component="DrugData",
                subcomponent="SearchDrugDatabase",
                drug_name=drug_name,
                drug_class=drug_class,
                limit=limit,
            )

            # Build query and filter
            query, filter_expr = self._build_query_and_filter(drug_name, drug_class)

            if not query:
                error_msg = "Either drug_name or drug_class must be provided"
                self.logger.error(error_msg)
                return json.dumps({"error": error_msg, "results": []}, indent=2)

            # Perform search using the vector database provider
            results = await self.vector_db_provider.search_by_text(
                query_text=query,
                limit=limit,
                filter_expression=filter_expr,
            )

            # Format results as table for LLM consumption
            formatted_table = self._format_results_as_table(results)

            # Emphasized logging: Function call result with chunk count
            self.logger.info(
                f"=== FUNCTION CALL RESULT ===\n"
                f"FUNCTION: search_drug_database\n"
                f"CHUNKS_RETRIEVED: {len(results)}\n"
                f"DRUG_NAME: {drug_name}\n"
                f"DRUG_CLASS: {drug_class}\n"
                f"LIMIT_REQUESTED: {limit}\n"
                f"===========================",
                component="DrugData",
                subcomponent="SearchDrugDatabase",
                chunk_count=len(results),
                drug_name=drug_name,
                drug_class=drug_class,
                limit_requested=limit,
            )

            self.logger.info(
                f"Drug database search completed - Found {len(results)} results",
                component="DrugData",
                subcomponent="SearchDrugDatabase",
            )

            return formatted_table

        except VectorDBSearchError as e:
            error_msg = f"Drug database search failed: {str(e)}"
            self.logger.error(
                error_msg,
                component="DrugData",
                subcomponent="SearchDrugDatabase",
                exc_info=True,
            )
            return json.dumps({"error": error_msg, "results": []}, indent=2)
        except Exception as e:
            error_msg = f"Unexpected error during drug search: {str(e)}"
            self.logger.error(
                error_msg,
                component="DrugData",
                subcomponent="SearchDrugDatabase",
                exc_info=True,
            )
            return json.dumps({"error": error_msg, "results": []}, indent=2)

    def search_drug_database_sync(
        self,
        drug_name: Optional[str] = None,
        drug_class: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Synchronous version of search_drug_database.

        Args:
            drug_name: Specific drug name to search for
            drug_class: Drug class to search for
            limit: Maximum number of results to return

        Returns:
            JSON string containing search results
        """
        try:
            # Use asyncio.run for synchronous execution
            return asyncio.run(self.search_drug_database(drug_name, drug_class, limit))
        except Exception as e:
            error_msg = f"Sync drug search failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg, "results": []}, indent=2)

    def _format_search_query_text(
        self, drug_name: Optional[str], drug_class: Optional[str]
    ) -> str:
        """
        Format drug name and/or class into structured query text for embedding generation.

        This method creates a consistent query format that matches the ingestion pipeline's
        search_text format to ensure better semantic similarity matching. The formatted text
        is used for embedding generation before performing vector similarity search against
        the Milvus collection.

        Args:
            drug_name: Drug name to search for (e.g., "metformin", "lisinopril").
                      Case-insensitive; will be used as-is for semantic search.
            drug_class: Drug class to search for (e.g., "antidiabetic", "ACE inhibitor").
                       Case-insensitive; will be used as-is for semantic search.

        Returns:
            Formatted query string in the format: "Drug: <name> | Drug Class: <class>"
            If only one parameter is provided, returns simplified format with just that field.
            Examples:
                - Both provided: "Drug: metformin | Drug Class: antidiabetic"
                - Drug name only: "Drug: metformin"
                - Drug class only: "Drug Class: antidiabetic"

        Raises:
            ValueError: If both drug_name and drug_class are None or empty strings.
        """
        # Build query text based on available parameters
        query_parts = []

        if drug_name and drug_name.strip():
            query_parts.append(f"Drug: {drug_name.strip()}")

        if drug_class and drug_class.strip():
            query_parts.append(f"Drug Class: {drug_class.strip()}")

        # Join parts with separator
        formatted_query = " | ".join(query_parts)

        if not formatted_query:
            raise ValueError(
                "At least one of drug_name or drug_class must be provided and non-empty"
            )

        return formatted_query

    def _build_query_and_filter(
        self, drug_name: Optional[str], drug_class: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        """
        Build search query for semantic search.

        This method constructs a formatted query text for embedding generation.
        Note: Filter expressions are NOT used to avoid exact match issues with
        LLM-generated values. Semantic search via embeddings is sufficient.

        Args:
            drug_name: Drug name for semantic search
            drug_class: Drug class for semantic search

        Returns:
            Tuple of (query_text, filter_expression) where:
            - query_text: Formatted text like "Drug: metformin | Drug Class: antidiabetic"
            - filter_expression: Always None (filter expressions disabled for semantic search)
        """
        try:
            # Use the helper to format query text
            query = self._format_search_query_text(drug_name, drug_class)

            # DO NOT use filter expressions - rely on semantic search via embeddings.
            # The embedding search will naturally find relevant drugs based on similarity.
            # LLM-generated values may not exactly match database values, so exact-match
            # filters are unreliable. Semantic similarity is sufficient.
            filter_expr = None

            return query, filter_expr

        except ValueError as e:
            # If formatting fails, return empty query (will be caught later)
            self.logger.debug(f"Query formatting error: {str(e)}")
            return "", None

    def _format_results_for_llm(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Format search results for LLM consumption.

        Args:
            results: List of SearchResult objects from vector database

        Returns:
            Dictionary formatted for JSON serialization
        """
        formatted = {
            "count": len(results),
            "results": [],
        }

        for result in results:
            drug_info = {
                "relevance_score": round(result.score, 4),
                "drug_name": result.get_field("drug_name", "Unknown"),
                "drug_class": result.get_field("drug_class", "Not specified"),
                "drug_sub_class": result.get_field("drug_sub_class", "Not specified"),
                "therapeutic_category": result.get_field(
                    "therapeutic_category", "Not specified"
                ),
                "route_of_administration": result.get_field(
                    "route_of_administration", "Not specified"
                ),
                "formulation": result.get_field("formulation", "Not specified"),
                "dosage_strengths": result.get_field(
                    "dosage_strengths", "Not specified"
                ),
            }
            formatted["results"].append(drug_info)

        return formatted

    def _format_results_as_table(self, results: List[SearchResult]) -> str:
        """
        Format search results as a tabulated string for LLM consumption.

        This method converts SearchResult objects into a clean tabular format using the
        tabulate library. Tabular format helps the LLM understand structured data better
        and reduces hallucinations in responses by presenting information in a clear,
        organized table structure.

        The table provides all essential drug information in columns that are easy for
        the LLM to parse and reference when generating responses.

        Args:
            results: List of SearchResult objects from vector database search

        Returns:
            String containing formatted table with headers and drug information.
            If no results are found, returns a message indicating no data available.

        Logs:
            Info level: Logs the number of rows formatted in the table with full context

        Table Columns (in order):
            - Relevance: Similarity score (0-1 scale, higher = better match)
            - Drug Name: Name of the medication
            - Drug Class: Pharmacological classification
            - Sub-Class: More specific classification within the class
            - Therapeutic: Medical use/therapeutic indication
            - Route: Administration route (e.g., oral, IV, intramuscular)
            - Formulation: Physical form (e.g., tablet, solution, capsule)
            - Dosages: Available dosage strengths for the drug

        Example output:
            Relevance    Drug Name    Drug Class    Sub-Class    Therapeutic    Route    Formulation    Dosages
            ----------   -----------  -----------   -----------  -----------    ------   -----------    --------
            0.9523       Metformin    Antidiabetic  Biguanide    Type 2 DM      Oral     Tablet         500mg...
            0.9412       Glimepiride  Antidiabetic  Sulfonylurea Type 2 DM      Oral     Tablet         1mg...
        """
        try:
            if not results:
                self.logger.info(
                    "No search results available to format as table",
                    component="DrugData",
                    subcomponent="FormatResultsAsTable",
                    row_count=0,
                )
                return "No drug information found matching the search criteria."

            # Prepare table data
            table_data = []
            for result in results:
                row = [
                    round(result.score, 4),  # Relevance score
                    result.get_field("drug_name", "N/A"),
                    result.get_field("drug_class", "N/A"),
                    result.get_field("drug_sub_class", "N/A"),
                    result.get_field("therapeutic_category", "N/A"),
                    result.get_field("route_of_administration", "N/A"),
                    result.get_field("formulation", "N/A"),
                    result.get_field("dosage_strengths", "N/A"),
                ]
                table_data.append(row)

            # Define table headers
            headers = [
                "Relevance",
                "Drug Name",
                "Drug Class",
                "Sub-Class",
                "Therapeutic",
                "Route",
                "Formulation",
                "Dosages",
            ]

            # Format as table using tabulate
            formatted_table = tabulate(
                table_data,
                headers=headers,
                tablefmt="grid",  # Grid format for clear visualization
                stralign="left",
                numalign="right",
                showindex=False,
            )

            # Log the formatting operation with row count
            self.logger.info(
                f"Formatted {len(results)} search results as table",
                component="DrugData",
                subcomponent="FormatResultsAsTable",
                row_count=len(results),
            )

            return formatted_table

        except Exception as e:
            error_msg = f"Failed to format results as table: {str(e)}"
            self.logger.error(
                error_msg,
                component="DrugData",
                subcomponent="FormatResultsAsTable",
                exc_info=True,
            )
            # Fallback to unformatted text if tabulation fails
            return f"Error formatting results: {error_msg}"

    def get_function_schema(self) -> Dict[str, Any]:
        """
        Get the OpenAI function schema for the drug search tool.

        This schema is used by the Responses API to understand when and how
        to call the drug search function.

        Returns:
            Function schema dictionary compatible with Responses API
        """
        # return {
        #     "name": "search_drug_database",
        #     "description": (
        #         "Search the pharmaceutical drug database for detailed medication information. "
        #         "This tool MUST be used when the user asks about:\n"
        #         "- Specific drug names (e.g., 'metformin', 'lisinopril', 'aspirin')\n"
        #         "- Drug formulations and dosage strengths\n"
        #         "- Drug classifications or therapeutic categories\n"
        #         "- Routes of administration\n"
        #         "- Any medication-specific details\n\n"
        #         "IMPORTANT: Always extract the drug name or class from the user's query and pass it as a parameter. "
        #         "If the user mentions a specific drug name, extract it and use the drug_name parameter. "
        #         "If the user asks about a drug class, extract it and use the drug_class parameter."
        #     ),
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "drug_name": {
        #                 "type": "string",
        #                 "description": (
        #                     "The name of the drug to search for. Extract this from the user's question. "
        #                     "Examples: 'metformin', 'lisinopril', 'aspirin', 'ibuprofen'. "
        #                     "REQUIRED when the user mentions a specific medication name."
        #                 ),
        #             },
        #             "drug_class": {
        #                 "type": "string",
        #                 "description": (
        #                     "The drug class or category to search for. "
        #                     "Examples: 'ACE inhibitor', 'beta blocker', 'NSAID', 'antidiabetic'. "
        #                     "Use this parameter when the user asks about a class of medications."
        #                 ),
        #             },
        #             "limit": {
        #                 "type": "integer",
        #                 "description": (
        #                     "Maximum number of results to return (1-20). "
        #                     "Default is 5. Use higher values for broader searches."
        #                 ),
        #                 "default": 5,
        #                 "minimum": 1,
        #                 "maximum": 20,
        #             },
        #         },
        #         "anyOf": [
        #             {"required": ["drug_name"]},
        #             {"required": ["drug_class"]},
        #         ],
        #         # Make drug_name required by default - this handles the most common use case
        #         # When user asks about a drug class, the model should still extract it as drug_class
        #         # "required": ["drug_name"],
        #         "additionalProperties": False,
        #     },
        #     "strict": True,
        # }
        return {
            "name": "search_drug_database",
            "description": "Search the pharmaceutical drug database for medication information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_class": {
                        "type": "string",
                        "description": "The drug class or category (REQUIRED).",
                    },
                    "drug_name": {
                        "type": "string",
                        "description": "Specific drug name (optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return.",
                    },
                },
                "required": ["drug_class"],
            },
            "strict": False,
        }

    async def connect(self) -> None:
        """Connect to the vector database."""
        try:
            await self.vector_db_provider.connect()
            self.logger.info("Connected to drug database")
        except Exception as e:
            self.logger.error(f"Failed to connect to drug database: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        try:
            await self.vector_db_provider.disconnect()
            self.logger.info("Disconnected from drug database")
        except Exception as e:
            self.logger.error(f"Failed to disconnect from drug database: {e}")

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the vector database provider.

        Returns:
            Dictionary with provider configuration and status
        """
        try:
            return {
                "provider_type": self.vector_db_provider.provider_type,
                "collection_name": self.vector_db_provider.collection_name,
                "embedding_dimension": self.vector_db_provider.embedding_dimension,
                "is_connected": self.vector_db_provider._is_connected,
            }
        except Exception as e:
            self.logger.error(f"Failed to get provider info: {e}")
            return {"error": str(e)}
