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
from typing import Optional, Dict, Any, List

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
                self.provider = create_vector_db_provider(provider_type="milvus")
                self.logger.info("Created default Milvus provider for drug data")
            except Exception as e:
                self.logger.error(f"Failed to create Milvus provider: {e}")
                raise
        else:
            self.provider = vector_db_provider
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
            JSON string containing search results with drug information

        Examples:
            >>> # Search by drug name
            >>> await search_drug_database(drug_name="Lisinopril")

            >>> # Search by drug class
            >>> await search_drug_database(drug_class="ACE inhibitor")

            >>> # Search by both (uses filter)
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
            results = await self.provider.search_by_text(
                query_text=query,
                limit=limit,
                filter_expression=filter_expr,
            )

            # Format results for LLM consumption
            formatted_results = self._format_results_for_llm(results)

            self.logger.info(
                f"Drug database search completed - Found {len(results)} results",
                component="DrugData",
                subcomponent="SearchDrugDatabase",
            )

            return json.dumps(formatted_results, indent=2)

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

    def _build_query_and_filter(
        self, drug_name: Optional[str], drug_class: Optional[str]
    ) -> tuple[str, Optional[str]]:
        """
        Build search query and optional filter expression.

        Args:
            drug_name: Drug name for semantic search
            drug_class: Drug class for filtering

        Returns:
            Tuple of (query_text, filter_expression)
        """
        # Build query text for semantic search
        if drug_name and drug_class:
            # Both provided: search by name, filter by class
            query = f"{drug_name} {drug_class}"
            filter_expr = f'drug_class == "{drug_class}"'
        elif drug_name:
            # Only drug name: semantic search
            query = drug_name
            filter_expr = None
        elif drug_class:
            # Only drug class: search broadly, filter by class
            query = drug_class
            filter_expr = f'drug_class == "{drug_class}"'
        else:
            # Neither provided
            query = ""
            filter_expr = None

        return query, filter_expr

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

    def get_function_schema(self) -> Dict[str, Any]:
        """
        Get the OpenAI function schema for the drug search tool.

        This schema is used by the Responses API to understand when and how
        to call the drug search function.

        Returns:
            Function schema dictionary compatible with Responses API
        """
        return {
            "name": "search_drug_database",
            "description": (
                "Search the pharmaceutical drug database for detailed medication information. "
                "This tool MUST be used when the user asks about:\n"
                "- Specific drug names (e.g., 'metformin', 'lisinopril', 'aspirin')\n"
                "- Drug formulations and dosage strengths\n"
                "- Drug classifications or therapeutic categories\n"
                "- Routes of administration\n"
                "- Any medication-specific details\n\n"
                "IMPORTANT: Always extract the drug name or class from the user's query and pass it as a parameter. "
                "If the user mentions a specific drug name, extract it and use the drug_name parameter. "
                "If the user asks about a drug class, extract it and use the drug_class parameter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": (
                            "The name of the drug to search for. Extract this from the user's question. "
                            "Examples: 'metformin', 'lisinopril', 'aspirin', 'ibuprofen'. "
                            "REQUIRED when the user mentions a specific medication name."
                        ),
                    },
                    "drug_class": {
                        "type": "string",
                        "description": (
                            "The drug class or category to search for. "
                            "Examples: 'ACE inhibitor', 'beta blocker', 'NSAID', 'antidiabetic'. "
                            "Use this parameter when the user asks about a class of medications."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            "Maximum number of results to return (1-20). "
                            "Default is 5. Use higher values for broader searches."
                        ),
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "anyOf": [
                    {"required": ["drug_name"]},
                    {"required": ["drug_class"]},
                ],
                # Make drug_name required by default - this handles the most common use case
                # When user asks about a drug class, the model should still extract it as drug_class
                # "required": ["drug_name"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    async def connect(self) -> None:
        """Connect to the vector database."""
        try:
            await self.provider.connect()
            self.logger.info("Connected to drug database")
        except Exception as e:
            self.logger.error(f"Failed to connect to drug database: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        try:
            await self.provider.disconnect()
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
                "provider_type": self.provider.provider_type,
                "collection_name": self.provider.collection_name,
                "embedding_dimension": self.provider.embedding_dimension,
                "is_connected": self.provider._is_connected,
            }
        except Exception as e:
            self.logger.error(f"Failed to get provider info: {e}")
            return {"error": str(e)}
