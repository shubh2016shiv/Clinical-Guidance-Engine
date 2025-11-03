"""
Asclepius Healthcare Agent

healthcare AI agent that provides evidence-based medical recommendations
by consulting clinical guidelines and drug databases through tool integration.

The agent maintains conversation context, leverages semantic search over clinical knowledge,
and coordinates with multiple data sources to deliver comprehensive healthcare insights.
"""

from typing import Dict, Any, List, Optional
from src.response_api_agent.managers.response_api_manager import OpenAIResponseManager
from src.logs import get_component_logger
from src.response_api_agent.managers.exceptions import ResponsesAPIError


class AsclepiusHealthcareAgent:
    """
    Asclepius - A healthcare AI agent for evidence-based medical guidance.

    This agent provides:
    - Evidence-based answers to healthcare and drug-related questions
    - Semantic search over clinical guidelines and best practices
    - Multi-turn conversation with context awareness
    - Tool-based access to drug databases and clinical data (via Milvus integration)
    - Conversation history management and resource cleanup

    The agent delegates response_api_agent AI operations to OpenAIResponseManager while focusing
    on healthcare-specific orchestration and knowledge base management.
    """

    def __init__(self, chat_history_limit: int = 10):
        """
        Initialize the Asclepius Healthcare Agent.

        Args:
            chat_history_limit: Maximum number of messages to retain in conversation history.
        """
        self.logger = get_component_logger("AsclepiusHealthcareAgent")
        self.response_manager = OpenAIResponseManager(
            chat_history_limit=chat_history_limit
        )
        self._vector_store_id: Optional[str] = None

        self.logger.info(
            "Asclepius Healthcare Agent initialized",
            component="AsclepiusHealthcareAgent",
            subcomponent="Init",
        )

    async def initialize_knowledge_base(self) -> Optional[str]:
        """
        Initialize the knowledge base by setting up the clinical guidelines vector store.

        This loads clinical guidelines into the vector store for semantic search capability,
        enabling evidence-based recommendations.

        Returns:
            Vector store ID for use in consultations, or None if initialization fails.
        """
        try:
            self.logger.info(
                "Initializing clinical knowledge base",
                component="AsclepiusHealthcareAgent",
                subcomponent="InitializeKnowledgeBase",
            )

            self._vector_store_id = (
                await self.response_manager.create_vector_store_from_guidelines()
            )

            self.logger.info(
                "Clinical knowledge base ready",
                component="AsclepiusHealthcareAgent",
                subcomponent="InitializeKnowledgeBase",
                vector_store_id=self._vector_store_id,
            )

            return self._vector_store_id

        except Exception as e:
            self.logger.error(
                "Failed to initialize clinical knowledge base",
                component="AsclepiusHealthcareAgent",
                subcomponent="InitializeKnowledgeBase",
                error=str(e),
            )
            # Continue without vector store - agent will still respond but without clinical context
            return None

    async def consult(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        use_clinical_guidelines: bool = True,
        use_drug_database: bool = True,
        streaming: bool = False,
        enable_tool_execution: bool = True,
    ) -> Dict[str, Any]:
        """
        Consult the agent on a healthcare or medication-related question.

        Args:
            query: The user's healthcare question or concern.
            conversation_id: Optional ID to continue an existing consultation session.
            use_clinical_guidelines: Whether to enhance responses with clinical guidelines.
            use_drug_database: Whether to enable drug database search tool (default: True).
            streaming: If True, return streaming response for real-time delivery.
            enable_tool_execution: Whether to automatically execute tool calls (default: True).

        Returns:
            Dictionary containing:
                - content: The agent's response (if not streaming)
                - stream_generator: Async generator for streaming response chunks
                - conversation_id: ID for tracking this consultation session
                - tool_calls: Any tool calls made to fetch data (if tool execution disabled)
                - tool_execution_history: History of executed tools (if tool execution enabled)
                - guidelines_used: Whether clinical guidelines enhanced this response
                - drug_database_used: Whether drug database was available

        Raises:
            ResponsesAPIError: If consultation fails.
        """
        try:
            vector_store_id = self._vector_store_id if use_clinical_guidelines else None

            self.logger.info(
                "Processing healthcare consultation",
                component="AsclepiusHealthcareAgent",
                subcomponent="Consult",
                query_length=len(query),
                use_clinical_guidelines=use_clinical_guidelines,
                use_drug_database=use_drug_database,
                streaming=streaming,
                enable_tool_execution=enable_tool_execution,
            )

            if streaming:
                # Use process_streaming_query which returns AsyncGenerator directly
                stream_gen = self.response_manager.process_streaming_query(
                    user_message=query,
                    conversation_id=conversation_id,
                    vector_store_id=vector_store_id,
                    use_drug_database=use_drug_database,
                    enable_tool_execution=enable_tool_execution,
                )

                return {
                    "stream_generator": stream_gen,
                    "conversation_id": conversation_id,
                    "guidelines_used": vector_store_id is not None,
                    "drug_database_used": use_drug_database
                    and self.response_manager.drug_data_manager is not None,
                }
            else:
                # Standard response with drug database and tool execution support
                result = await self.response_manager.process_query(
                    user_message=query,
                    conversation_id=conversation_id,
                    vector_store_id=vector_store_id,
                    enable_streaming=False,
                    use_drug_database=use_drug_database,
                    enable_tool_execution=enable_tool_execution,
                )

                return {
                    "content": result["content"],
                    "conversation_id": result["conversation_id"],
                    "tool_calls": result.get("tool_calls", []),
                    "tool_execution_history": result.get("tool_execution_history", []),
                    "citations": result.get("citations", []),
                    "guidelines_used": vector_store_id is not None,
                    "drug_database_used": use_drug_database
                    and self.response_manager.drug_data_manager is not None,
                }

        except Exception as e:
            self.logger.error(
                "Failed to process consultation",
                component="AsclepiusHealthcareAgent",
                subcomponent="Consult",
                error=str(e),
            )
            raise ResponsesAPIError(f"Consultation failed: {str(e)}")

    async def continue_consultation(
        self,
        conversation_id: str,
        follow_up_query: str,
        use_clinical_guidelines: bool = True,
        use_drug_database: bool = True,
    ) -> Dict[str, Any]:
        """
        Continue an ongoing healthcare consultation with a follow-up question.

        Args:
            conversation_id: The existing consultation session ID to continue.
            follow_up_query: The follow-up healthcare question or request.
            use_clinical_guidelines: Whether to enhance response with clinical guidelines.
            use_drug_database: Whether to enable drug database search tool (default: True).

        Returns:
            Dictionary with response content and metadata (same as consult method).
        """
        return await self.consult(
            query=follow_up_query,
            conversation_id=conversation_id,
            use_clinical_guidelines=use_clinical_guidelines,
            use_drug_database=use_drug_database,
            streaming=False,
        )

    async def retrieve_consultation_history(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the complete history of a healthcare consultation.

        Args:
            conversation_id: The consultation session ID to retrieve history for.

        Returns:
            List of consultation messages in chronological order.

        Raises:
            ResponsesAPIError: If history retrieval fails.
        """
        try:
            self.logger.info(
                "Retrieving consultation history",
                component="AsclepiusHealthcareAgent",
                subcomponent="RetrieveConsultationHistory",
                conversation_id=conversation_id,
            )

            history = await self.response_manager.retrieve_conversation_history(
                conversation_id
            )

            return history

        except Exception as e:
            self.logger.error(
                "Failed to retrieve consultation history",
                component="AsclepiusHealthcareAgent",
                subcomponent="RetrieveConsultationHistory",
                conversation_id=conversation_id,
                error=str(e),
            )
            raise ResponsesAPIError(f"History retrieval failed: {str(e)}")

    async def close_session(self) -> Dict[str, bool]:
        """
        Close the agent session and clean up all resources.

        This should be called when the healthcare consultation is complete to
        ensure proper cleanup of vector stores and conversation caches.

        Returns:
            Dictionary with cleanup status for each resource.
        """
        try:
            self.logger.info(
                "Closing Asclepius healthcare session",
                component="AsclepiusHealthcareAgent",
                subcomponent="CloseSession",
            )

            cleanup_status = await self.response_manager.cleanup_resources(
                vector_store_id=self._vector_store_id, clear_all_caches=True
            )

            self._vector_store_id = None

            self.logger.info(
                "Healthcare session closed successfully",
                component="AsclepiusHealthcareAgent",
                subcomponent="CloseSession",
                status=cleanup_status,
            )

            return cleanup_status

        except Exception as e:
            self.logger.error(
                "Error closing healthcare session",
                component="AsclepiusHealthcareAgent",
                subcomponent="CloseSession",
                error=str(e),
            )
            return {"error": str(e)}

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get the current status and configuration of Asclepius.

        Returns:
            Dictionary with agent status information including knowledge base readiness,
            drug database availability, and manager configuration.
        """
        manager_info = self.response_manager.get_manager_info()

        drug_db_available = self.response_manager.drug_data_manager is not None
        drug_db_info = None
        if drug_db_available:
            try:
                drug_db_info = (
                    self.response_manager.drug_data_manager.get_provider_info()
                )
            except Exception as e:
                self.logger.warning(f"Failed to get drug database info: {e}")

        return {
            "agent_type": "AsclepiusHealthcareAgent",
            "knowledge_base_ready": self._vector_store_id is not None,
            "vector_store_id": self._vector_store_id,
            "drug_database_available": drug_db_available,
            "drug_database_info": drug_db_info,
            "response_manager": manager_info,
        }

    # Convenience methods for backward compatibility and domain-specific usage
    async def setup_clinical_guidelines(self) -> Optional[str]:
        """
        Convenience method for setting up clinical guidelines vector store.

        Returns:
            Vector store ID for use in queries, or None if setup fails.
        """
        return await self.initialize_knowledge_base()

    async def ask_question(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        use_clinical_context: bool = True,
        use_drug_database: bool = True,
        streaming: bool = False,
    ) -> Dict[str, Any]:
        """
        Convenience method for asking healthcare questions (backward compatibility).

        Args:
            question: The user's question about drugs/treatments.
            conversation_id: Optional existing conversation to continue.
            use_clinical_context: Whether to use clinical guidelines for enhanced responses.
            use_drug_database: Whether to enable drug database search tool (default: True).
            streaming: If True, return streaming response.

        Returns:
            Dictionary with response content, conversation_id, and metadata.
        """
        return await self.consult(
            query=question,
            conversation_id=conversation_id,
            use_clinical_guidelines=use_clinical_context,
            use_drug_database=use_drug_database,
            streaming=streaming,
        )
