"""Check the Chainlit chat handler drives the orchestrator, keeps citations and saves turns."""

import unittest
from unittest.mock import MagicMock

from healthcare_assistant_ui.chat_handler import ChainlitChatHandler
from src.clinical_guideline_agent.schema import FileCitation, GuidelineAnswer
from src.drugs_agent import DrugAnswer
from src.orchestrator import OrchestratorAnswer, OrchestratorUpdate

GUIDELINE_ANSWER = GuidelineAnswer(
    response_id="resp_1",
    text="Start an ACE inhibitor.",
    citations=[FileCitation(file_id="file_1", filename="hypertension.pdf", index=3)],
)
ORCHESTRATOR_ANSWER = OrchestratorAnswer(
    route="clinical_and_drug",
    text="Use an ACE inhibitor.${}^{[1]}$\n\n| Drug |",
    sources=GUIDELINE_ANSWER.citations,
    guideline_answer=GUIDELINE_ANSWER,
    drug_classes=["angiotensin-converting enzyme inhibitors"],
    drug_answer=DrugAnswer(found=False, matches=[]),
)


DRUG_ONLY_ANSWER = OrchestratorAnswer(
    route="drug",
    text="**Available in the catalog**\n\n*ACE inhibitors*\n- **Lisinopril**",
    sources=[],
    # The drug route never asks the guidelines, so there is no response to chain from.
    guideline_answer=None,
    drug_classes=[],
    drug_answer=DrugAnswer(found=False, matches=[]),
)


class FakeOrchestrator:
    def __init__(self, answer: OrchestratorAnswer = ORCHESTRATOR_ANSWER) -> None:
        self.questions: list[str] = []
        self.answer = answer

    async def stream_clinical_drug_answer_updates(self, question):
        self.questions.append(question)
        yield OrchestratorUpdate(type="text", text="Use an ")
        yield OrchestratorUpdate(type="text", text="ACE inhibitor.")
        yield OrchestratorUpdate(type="completed", answer=self.answer)


def chat_handler():
    persistence_manager = MagicMock()
    return ChainlitChatHandler(persistence_manager=persistence_manager), persistence_manager


class ChatHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_text_then_a_final_update_with_citations(self) -> None:
        handler, _ = chat_handler()
        orchestrator = FakeOrchestrator()

        updates = [
            update
            async for update in handler.handle_message(
                message="How is hypertension treated?",
                orchestrator=orchestrator,
                session_id="s1",
            )
        ]

        self.assertEqual(orchestrator.questions, ["How is hypertension treated?"])
        self.assertEqual([u["chunk"] for u in updates[:-1]], ["Use an ", "ACE inhibitor."])
        self.assertFalse(any(u["is_complete"] for u in updates[:-1]))
        final = updates[-1]
        self.assertTrue(final["is_complete"])
        self.assertEqual(final["route"], "clinical_and_drug")
        self.assertEqual(final["response_id"], "resp_1")
        self.assertEqual(final["content"], ORCHESTRATOR_ANSWER.text)
        self.assertEqual(
            final["citations"], [{"file_id": "file_1", "filename": "hypertension.pdf", "index": 3}]
        )

    async def test_the_saved_turn_holds_the_full_answer_and_the_route(self) -> None:
        handler, persistence_manager = chat_handler()

        async for _ in handler.handle_message(
            message="q", orchestrator=FakeOrchestrator(), session_id="s1"
        ):
            pass

        persistence_manager.save_conversation.assert_called_once()
        saved = persistence_manager.save_conversation.call_args.kwargs
        self.assertEqual(saved["session_id"], "s1")
        self.assertEqual(saved["response_id"], "resp_1")
        self.assertEqual(saved["assistant_response"], ORCHESTRATOR_ANSWER.text)
        self.assertEqual(saved["metadata"]["route"], "clinical_and_drug")
        self.assertEqual(
            saved["metadata"]["drug_classes"], ["angiotensin-converting enzyme inhibitors"]
        )
        self.assertTrue(saved["metadata"]["guidelines_used"])

    async def test_the_non_streaming_path_returns_the_whole_answer_once(self) -> None:
        handler, persistence_manager = chat_handler()

        answer = await handler.handle_message_non_streaming(
            message="q", orchestrator=FakeOrchestrator(), session_id="s1"
        )

        self.assertEqual(answer["content"], ORCHESTRATOR_ANSWER.text)
        self.assertEqual(len(answer["citations"]), 1)
        persistence_manager.save_conversation.assert_called_once()

    async def test_a_drug_only_answer_has_no_guideline_response_to_report(self) -> None:
        """Reading the response ID off a missing guideline answer used to crash here."""
        handler, persistence_manager = chat_handler()

        answer = await handler.handle_message_non_streaming(
            message="What strengths does lisinopril come in?",
            orchestrator=FakeOrchestrator(DRUG_ONLY_ANSWER),
            session_id="s1",
        )

        self.assertEqual(answer["route"], "drug")
        self.assertEqual(answer["citations"], [])
        self.assertEqual(answer["content"], DRUG_ONLY_ANSWER.text)
        # The request ID stands in for the absent guideline response ID.
        self.assertTrue(answer["response_id"])

        saved = persistence_manager.save_conversation.call_args.kwargs
        self.assertEqual(saved["response_id"], answer["response_id"])
        self.assertEqual(saved["metadata"]["route"], "drug")
        self.assertFalse(saved["metadata"]["guidelines_used"])


if __name__ == "__main__":
    unittest.main()
