"""
Chat Handler for Chainlit UI.

Bridges the Chainlit UI with the AsclepiusHealthcareAgent, handling message processing,
streaming responses, and citation collection.
"""

import sys
from pathlib import Path

# Add project root to Python path to enable src imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uuid  # noqa: E402
from typing import Dict, Any, List, Optional, AsyncGenerator  # noqa: E402
from src.response_api_agent.asclepius_agent import AsclepiusHealthcareAgent  # noqa: E402
from src.response_api_agent.managers.exceptions import ResponsesAPIError  # noqa: E402
from chatbot_ui.persistence_manager import ConversationPersistenceManager  # noqa: E402
from chatbot_ui.citation_formatter import ChainlitCitationFormatter  # noqa: E402


class ChainlitChatHandler:
    """
    Handles chat message processing and agent interaction for Chainlit UI.

    Coordinates between user messages, agent responses, streaming, citations,
    and conversation persistence.
    """

    def __init__(
        self,
        persistence_manager: Optional[ConversationPersistenceManager] = None,
        citation_formatter: Optional[ChainlitCitationFormatter] = None,
    ):
        """
        Initialize the chat handler.

        Args:
            persistence_manager: Manager for conversation persistence.
            citation_formatter: Formatter for citation display.
        """
        self.persistence_manager = (
            persistence_manager or ConversationPersistenceManager()
        )
        self.citation_formatter = citation_formatter or ChainlitCitationFormatter()

    async def handle_message(
        self,
        message: str,
        agent: AsclepiusHealthcareAgent,
        session_id: str,
        use_clinical_guidelines: bool = True,
        use_drug_database: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a user message and stream the agent's response.

        This is an async generator that yields chunks of the response as they
        are generated, allowing for real-time streaming in the UI.

        Args:
            message: The user's message text.
            agent: The initialized AsclepiusHealthcareAgent instance.
            session_id: Unique session identifier for conversation tracking.
            use_clinical_guidelines: Whether to use clinical guidelines.
            use_drug_database: Whether to enable drug database search.

        Yields:
            Dictionary with chunk information:
                - chunk: Text chunk to display
                - response_id: Response ID from Response API
                - is_citation: Whether this chunk is a citation
                - is_complete: Whether streaming is complete
                - citations: List of citation objects (only in final chunk)
                - error: Error message if processing failed
        """
        request_id = str(uuid.uuid4())
        response_id = None
        full_response = []
        citations = []

        try:
            # Call agent with streaming enabled
            # CRITICAL FIX: Pass session_id as conversation_id to continue existing session
            result = await agent.consult(
                query=message,
                conversation_id=session_id,  # Pass session_id to continue existing conversation
                use_clinical_guidelines=use_clinical_guidelines,
                use_drug_database=use_drug_database,
                streaming=True,
                enable_tool_execution=True,
            )

            # Extract stream generator
            stream_generator = result.get("stream_generator")

            if not stream_generator:
                # Fallback to non-streaming if stream generator not available
                yield {
                    "chunk": result.get("content", "No response available"),
                    "response_id": result.get("conversation_id"),
                    "is_citation": False,
                    "is_complete": True,
                    "citations": result.get("citations", []),
                }
                return

            # Process streaming response
            async for chunk_data in stream_generator:
                chunk_text = chunk_data.get("chunk", "")
                current_response_id = chunk_data.get("conversation_id")
                is_citation = chunk_data.get("is_citation", False)

                # Update response_id if available
                if current_response_id and response_id is None:
                    response_id = current_response_id

                # Collect citations from citation chunks
                if is_citation:
                    chunk_citations = chunk_data.get("citations", [])
                    if chunk_citations:
                        citations.extend(chunk_citations)

                # Collect response text (excluding citations)
                if chunk_text and not is_citation:
                    full_response.append(chunk_text)

                # Yield chunk for display
                yield {
                    "chunk": chunk_text,
                    "response_id": response_id,
                    "is_citation": is_citation,
                    "is_complete": False,
                    "citations": [],
                }

            # After streaming completes, check if we have citations from chunks
            # If not, try to get from result metadata as fallback
            if not citations:
                citations = result.get("citations", [])

            # Combine full response text
            full_response_text = "".join(full_response)

            # Save conversation to persistence
            if response_id:
                metadata = {
                    "citations": citations,
                    "guidelines_used": result.get("guidelines_used", False),
                    "drug_database_used": result.get("drug_database_used", False),
                }

                self.persistence_manager.save_conversation(
                    session_id=session_id,
                    request_id=request_id,
                    response_id=response_id,
                    user_message=message,
                    assistant_response=full_response_text,
                    metadata=metadata,
                )

            # Yield final completion signal with citations
            yield {
                "chunk": "",
                "response_id": response_id,
                "is_citation": False,
                "is_complete": True,
                "citations": citations,
            }

        except ResponsesAPIError as e:
            error_message = f"I encountered an error processing your request: {str(e)}"
            yield {
                "chunk": error_message,
                "response_id": None,
                "is_citation": False,
                "is_complete": True,
                "citations": [],
                "error": str(e),
            }

        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            yield {
                "chunk": error_message,
                "response_id": None,
                "is_citation": False,
                "is_complete": True,
                "citations": [],
                "error": str(e),
            }

    async def handle_message_non_streaming(
        self,
        message: str,
        agent: AsclepiusHealthcareAgent,
        session_id: str,
        use_clinical_guidelines: bool = True,
        use_drug_database: bool = True,
    ) -> Dict[str, Any]:
        """
        Handle a user message without streaming (for fallback scenarios).

        Args:
            message: The user's message text.
            agent: The initialized AsclepiusHealthcareAgent instance.
            session_id: Unique session identifier for conversation tracking.
            use_clinical_guidelines: Whether to use clinical guidelines.
            use_drug_database: Whether to enable drug database search.

        Returns:
            Dictionary with response information:
                - content: The full response text
                - response_id: Response ID from Response API
                - citations: List of citation objects
                - error: Error message if processing failed
        """
        request_id = str(uuid.uuid4())

        try:
            # Call agent without streaming
            # CRITICAL FIX: Pass session_id as conversation_id to continue existing session
            result = await agent.consult(
                query=message,
                conversation_id=session_id,  # Pass session_id to continue existing conversation
                use_clinical_guidelines=use_clinical_guidelines,
                use_drug_database=use_drug_database,
                streaming=False,
                enable_tool_execution=True,
            )

            response_content = result.get("content", "")
            response_id = result.get("conversation_id")
            citations = result.get("citations", [])

            # Save conversation to persistence
            if response_id:
                metadata = {
                    "citations": citations,
                    "guidelines_used": result.get("guidelines_used", False),
                    "drug_database_used": result.get("drug_database_used", False),
                }

                self.persistence_manager.save_conversation(
                    session_id=session_id,
                    request_id=request_id,
                    response_id=response_id,
                    user_message=message,
                    assistant_response=response_content,
                    metadata=metadata,
                )

            return {
                "content": response_content,
                "response_id": response_id,
                "citations": citations,
                "error": None,
            }

        except ResponsesAPIError as e:
            return {
                "content": f"I encountered an error processing your request: {str(e)}",
                "response_id": None,
                "citations": [],
                "error": str(e),
            }

        except Exception as e:
            return {
                "content": f"An unexpected error occurred: {str(e)}",
                "response_id": None,
                "citations": [],
                "error": str(e),
            }

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of message dictionaries in chronological order.
        """
        return self.persistence_manager.load_conversation_history(session_id)

    def format_response_with_citations(
        self, response_text: str, citations: List[Dict[str, Any]]
    ) -> tuple[str, List[Any]]:
        """
        Format response text with citations for display.

        Args:
            response_text: The response text from the agent.
            citations: List of citation dictionaries.

        Returns:
            Tuple of (formatted_text, citation_elements) for Chainlit display.
        """
        return self.citation_formatter.format_combined_citations(
            response_text, citations
        )
