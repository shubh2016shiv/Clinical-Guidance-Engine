"""Adapt clinical guideline answers for the Chainlit interface."""

from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.clinical_guideline_agent.schema import FileCitation, GuidelineAnswer

from .citation_formatter import ChainlitCitationFormatter
from .persistence_manager import ConversationPersistenceManager


class ChainlitChatHandler:
    """Stream answers and save completed guideline conversations."""

    def __init__(
        self,
        persistence_manager: ConversationPersistenceManager | None = None,
        citation_formatter: ChainlitCitationFormatter | None = None,
    ) -> None:
        self.persistence_manager = persistence_manager or ConversationPersistenceManager()
        self.citation_formatter = citation_formatter or ChainlitCitationFormatter()

    async def handle_message(
        self,
        *,
        message: str,
        agent: ClinicalGuidelineAgent,
        session_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield text chunks and a final chunk containing the answer citations."""
        request_id = str(uuid4())
        async for update in agent.stream_guideline_answer_updates(message):
            if update.type == "text":
                yield {"chunk": update.text, "is_complete": False}
            elif update.answer is not None:
                answer = update.answer
                citations = self._citation_dicts(answer.citations)
                self._save_answer(session_id, request_id, message, answer, citations)
                yield {
                    "chunk": "",
                    "response_id": answer.response_id,
                    "citations": citations,
                    "is_complete": True,
                }

    async def handle_message_non_streaming(
        self,
        *,
        message: str,
        agent: ClinicalGuidelineAgent,
        session_id: str,
    ) -> dict[str, Any]:
        """Return one completed guideline answer with its citations."""
        answer = await agent.answer_guideline_question(message)
        citations = self._citation_dicts(answer.citations)
        self._save_answer(session_id, str(uuid4()), message, answer, citations)
        return {
            "content": answer.text,
            "response_id": answer.response_id,
            "citations": citations,
        }

    def get_conversation_history(self, session_id: str) -> list[dict[str, Any]]:
        return self.persistence_manager.load_conversation_history(session_id)

    def format_response_with_citations(
        self, response_text: str, citations: list[dict[str, Any]]
    ) -> tuple[str, list[Any]]:
        return self.citation_formatter.format_combined_citations(response_text, citations)

    @staticmethod
    def _citation_dicts(citations: list[FileCitation]) -> list[dict[str, Any]]:
        return [asdict(citation) for citation in citations]

    def _save_answer(
        self,
        session_id: str,
        request_id: str,
        message: str,
        answer: GuidelineAnswer,
        citations: list[dict[str, Any]],
    ) -> None:
        self.persistence_manager.save_conversation(
            session_id=session_id,
            request_id=request_id,
            response_id=answer.response_id,
            user_message=message,
            assistant_response=answer.text,
            metadata={"citations": citations, "guidelines_used": bool(citations)},
        )
