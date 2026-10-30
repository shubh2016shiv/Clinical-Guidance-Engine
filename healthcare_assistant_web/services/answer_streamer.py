"""Asks the orchestrator a question and records the finished answer."""

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from src.orchestrator import OrchestratorUpdate

from .conversation_log import ConversationLog


class AnswerStreamer:
    """Streams an answer to the page, and saves it once it is complete."""

    def __init__(
        self, orchestrator: Any, session_id: str, conversation_log: ConversationLog
    ) -> None:
        self.orchestrator = orchestrator
        self.session_id = session_id
        self.conversation_log = conversation_log

    async def stream_answer(self, question: str) -> AsyncIterator[OrchestratorUpdate]:
        """Yield the orchestrator's updates unchanged: text pieces, then the final answer."""
        request_id = str(uuid4())
        # The session id doubles as the conversation thread id, so every question asked
        # in this browser tab sees the earlier questions and answers of that tab.
        async for update in self.orchestrator.stream_clinical_drug_answer_updates(
            question, conversation_thread_id=self.session_id
        ):
            if update.type == "completed" and update.answer is not None:
                self.conversation_log.save_turn(
                    self.session_id, request_id, question, update.answer
                )
            yield update
